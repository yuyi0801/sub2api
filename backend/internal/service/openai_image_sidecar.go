package service

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/util/responseheaders"
	"github.com/Wei-Shaw/sub2api/internal/util/urlvalidator"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"go.uber.org/zap"
)

const (
	defaultOpenAIImageSidecarModel = "gpt-image-2"
	openAIImageSidecarCaller       = "sub2api"
	openAIImageSidecarSource       = "openai-image-sidecar"
	openAIImageSidecarAccountID    = int64(0)
	openAIImageSidecarAccountName  = "openai-image-sidecar"
)

type openAIImageSidecarConfig struct {
	baseURL string
	apiKey  string
	model   string
	timeout time.Duration
}

func (s *OpenAIGatewayService) openAIImageSidecarConfig() (openAIImageSidecarConfig, bool, error) {
	if s == nil || s.cfg == nil || !s.cfg.Gateway.OpenAIImageSidecar.Enabled {
		return openAIImageSidecarConfig{}, false, nil
	}
	rawBaseURL := strings.TrimRight(strings.TrimSpace(s.cfg.Gateway.OpenAIImageSidecar.BaseURL), "/")
	if rawBaseURL == "" {
		return openAIImageSidecarConfig{}, true, fmt.Errorf("openai image sidecar base_url is required")
	}
	baseURL, err := urlvalidator.ValidateURLFormat(rawBaseURL, true)
	if err != nil {
		return openAIImageSidecarConfig{}, true, fmt.Errorf("invalid openai image sidecar base_url: %w", err)
	}
	apiKey := strings.TrimSpace(s.cfg.Gateway.OpenAIImageSidecar.APIKey)
	if apiKey == "" {
		return openAIImageSidecarConfig{}, true, fmt.Errorf("openai image sidecar api_key is required")
	}
	model := strings.TrimSpace(s.cfg.Gateway.OpenAIImageSidecar.Model)
	if model == "" {
		model = defaultOpenAIImageSidecarModel
	}
	timeout := time.Duration(s.cfg.Gateway.OpenAIImageSidecar.TimeoutSeconds) * time.Second
	if timeout <= 0 {
		timeout = 600 * time.Second
	}
	return openAIImageSidecarConfig{
		baseURL: baseURL,
		apiKey:  apiKey,
		model:   model,
		timeout: timeout,
	}, true, nil
}

// IsOpenAIImageSidecarConfigured reports whether the sidecar feature flag is on.
// It intentionally ignores config validity so callers can bypass account
// scheduling and surface sidecar config errors instead of "no available accounts".
func (s *OpenAIGatewayService) IsOpenAIImageSidecarConfigured() bool {
	return s != nil && s.cfg != nil && s.cfg.Gateway.OpenAIImageSidecar.Enabled
}

// OpenAIImageSidecarUsageAccount returns the virtual account used only for
// usage_logs foreign-key compatibility. It must never be scheduled upstream.
func OpenAIImageSidecarUsageAccount() *Account {
	return &Account{
		ID:          openAIImageSidecarAccountID,
		Name:        openAIImageSidecarAccountName,
		Platform:    PlatformOpenAI,
		Type:        AccountTypeUpstream,
		Credentials: map[string]any{},
		Extra:       map[string]any{"virtual": true, "purpose": "openai_image_sidecar_usage"},
		Concurrency: 0,
		Priority:    100,
		Status:      StatusDisabled,
		Schedulable: false,
	}
}

// IsOpenAIImageSidecarUsageAccount identifies the virtual sidecar usage account.
func IsOpenAIImageSidecarUsageAccount(account *Account) bool {
	return account != nil &&
		account.ID == openAIImageSidecarAccountID &&
		account.Platform == PlatformOpenAI &&
		account.Type == AccountTypeUpstream &&
		strings.TrimSpace(account.Name) == openAIImageSidecarAccountName
}

func OpenAIImageSidecarUpstreamEndpoint(endpoint string) string {
	normalized := normalizeImageGenerationEndpoint(endpoint)
	if normalized == "" {
		normalized = openAIImagesGenerationsEndpoint
	}
	return openAIImageSidecarSource + ":" + normalized
}

func OpenAIResponsesImageSidecarEndpointFromBody(body []byte) string {
	reqBody := cloneRequestMapForImageIntent(body)
	parsed, err := buildOpenAIImagesRequestFromResponsesBody(reqBody, defaultOpenAIImageSidecarModel)
	if err != nil || parsed == nil || strings.TrimSpace(parsed.Endpoint) == "" {
		return openAIImagesGenerationsEndpoint
	}
	return parsed.Endpoint
}

func IsCodexSparkModel(model string) bool {
	return isCodexSparkModel(model)
}

func (cfg openAIImageSidecarConfig) endpointURL(endpoint string) string {
	return buildOpenAIImagesURL(cfg.baseURL, endpoint)
}

func (s *OpenAIGatewayService) ForwardImagesSidecar(
	ctx context.Context,
	c *gin.Context,
	parsed *OpenAIImagesRequest,
	channelMappedModel string,
) (*OpenAIForwardResult, error) {
	sidecarCfg, enabled, cfgErr := s.openAIImageSidecarConfig()
	if !enabled {
		err := fmt.Errorf("openai image sidecar is not enabled")
		s.writeOpenAIImageSidecarError(c, http.StatusBadGateway, "config", err.Error(), "")
		if c != nil && c.Writer != nil && !c.Writer.Written() {
			c.JSON(http.StatusBadGateway, gin.H{"error": gin.H{"type": "upstream_error", "message": err.Error()}})
		}
		return nil, err
	}
	if cfgErr != nil {
		s.writeOpenAIImageSidecarError(c, http.StatusBadGateway, "config", cfgErr.Error(), "")
		if c != nil && c.Writer != nil && !c.Writer.Written() {
			c.JSON(http.StatusBadGateway, gin.H{"error": gin.H{"type": "upstream_error", "message": cfgErr.Error()}})
		}
		return nil, cfgErr
	}
	return s.forwardOpenAIImagesOAuthSidecar(ctx, c, parsed, channelMappedModel, sidecarCfg)
}

func (s *OpenAIGatewayService) ForwardResponsesImageSidecar(
	ctx context.Context,
	c *gin.Context,
	body []byte,
	originalModel string,
	upstreamModel string,
	stream bool,
	startTime time.Time,
) (*OpenAIForwardResult, error) {
	sidecarCfg, enabled, cfgErr := s.openAIImageSidecarConfig()
	if !enabled {
		err := fmt.Errorf("openai image sidecar is not enabled")
		s.writeOpenAIImageSidecarError(c, http.StatusBadGateway, "config", err.Error(), "")
		if c != nil && c.Writer != nil && !c.Writer.Written() {
			c.JSON(http.StatusBadGateway, gin.H{"error": gin.H{"type": "upstream_error", "message": err.Error()}})
		}
		return nil, err
	}
	if cfgErr != nil {
		s.writeOpenAIImageSidecarError(c, http.StatusBadGateway, "config", cfgErr.Error(), "")
		if c != nil && c.Writer != nil && !c.Writer.Written() {
			c.JSON(http.StatusBadGateway, gin.H{"error": gin.H{"type": "upstream_error", "message": cfgErr.Error()}})
		}
		return nil, cfgErr
	}
	reqBody, err := getOpenAIRequestBodyMap(c, body)
	if err != nil {
		return nil, err
	}
	if normalized := strings.TrimSpace(originalModel); normalized != "" {
		reqBody["model"] = normalized
	}
	if v, ok := reqBody["model"].(string); ok && strings.TrimSpace(v) != "" {
		originalModel = strings.TrimSpace(v)
	}
	if v, ok := reqBody["stream"].(bool); ok {
		stream = v
	}
	if strings.TrimSpace(upstreamModel) == "" {
		upstreamModel = originalModel
	}
	_, imageSizeTier, imageCfgErr := resolveOpenAIResponsesImageBillingConfig(reqBody, sidecarCfg.model)
	if imageCfgErr != nil {
		setOpsUpstreamError(c, http.StatusBadRequest, imageCfgErr.Error(), "")
		if c != nil && c.Writer != nil && !c.Writer.Written() {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": gin.H{
					"type":    "invalid_request_error",
					"message": imageCfgErr.Error(),
					"param":   "size",
				},
			})
		}
		return nil, imageCfgErr
	}
	return s.forwardOpenAIResponsesImageSidecar(ctx, c, reqBody, originalModel, upstreamModel, imageSizeTier, stream, sidecarCfg, startTime)
}

func (s *OpenAIGatewayService) forwardOpenAIImagesOAuthSidecar(
	ctx context.Context,
	c *gin.Context,
	parsed *OpenAIImagesRequest,
	channelMappedModel string,
	cfg openAIImageSidecarConfig,
) (*OpenAIForwardResult, error) {
	startTime := time.Now()
	requestModel := strings.TrimSpace(parsed.Model)
	if mapped := strings.TrimSpace(channelMappedModel); mapped != "" {
		requestModel = mapped
	}
	if requestModel == "" {
		requestModel = cfg.model
	}
	if err := validateOpenAIImagesModel(requestModel); err != nil {
		return nil, err
	}

	forwardBody, forwardContentType, err := buildOpenAIImagesSidecarBody(parsed, cfg.model)
	if err != nil {
		return nil, err
	}
	if !parsed.Multipart {
		setOpsUpstreamRequestBody(c, forwardBody)
	}

	sidecarCtx := ctx
	var cancel context.CancelFunc
	if cfg.timeout > 0 {
		sidecarCtx, cancel = context.WithTimeout(ctx, cfg.timeout)
		defer cancel()
	}
	upstreamReq, err := http.NewRequestWithContext(sidecarCtx, http.MethodPost, cfg.endpointURL(parsed.Endpoint), bytes.NewReader(forwardBody))
	if err != nil {
		return nil, err
	}
	upstreamReq.Header.Set("Authorization", "Bearer "+cfg.apiKey)
	applyOpenAIImageSidecarTraceHeaders(upstreamReq)
	if strings.TrimSpace(forwardContentType) != "" {
		upstreamReq.Header.Set("Content-Type", forwardContentType)
	}
	if parsed.Stream {
		upstreamReq.Header.Set("Accept", "text/event-stream")
	}

	upstreamStart := time.Now()
	resp, err := s.httpUpstream.Do(upstreamReq, "", 0, 0)
	SetOpsLatencyMs(c, OpsUpstreamLatencyMsKey, time.Since(upstreamStart).Milliseconds())
	if err != nil {
		s.writeOpenAIImageSidecarError(c, 0, "generation", err.Error(), cfg.endpointURL(parsed.Endpoint))
		if c != nil && c.Writer != nil && !c.Writer.Written() {
			c.JSON(http.StatusBadGateway, gin.H{"error": gin.H{"type": "upstream_error", "message": "OpenAI image sidecar request failed"}})
		}
		return nil, fmt.Errorf("openai image sidecar request failed: %s", sanitizeUpstreamErrorMessage(err.Error()))
	}
	if resp.StatusCode >= 400 {
		return nil, s.handleOpenAIImageSidecarErrorResponse(resp, c, "generation", cfg.endpointURL(parsed.Endpoint))
	}
	defer func() { _ = resp.Body.Close() }()

	var usage OpenAIUsage
	imageCount := parsed.N
	var firstTokenMs *int
	if parsed.Stream && isEventStreamResponse(resp.Header) {
		streamUsage, streamCount, ttft, err := s.handleOpenAIImagesStreamingResponse(resp, c, startTime)
		if err != nil {
			if streamCount > 0 {
				return &OpenAIForwardResult{
					RequestID:       resp.Header.Get("x-request-id"),
					Usage:           streamUsage,
					Model:           requestModel,
					UpstreamModel:   cfg.model,
					BillingModel:    cfg.model,
					Stream:          parsed.Stream,
					ResponseHeaders: resp.Header.Clone(),
					Duration:        time.Since(startTime),
					FirstTokenMs:    ttft,
					ImageCount:      streamCount,
					ImageSize:       parsed.SizeTier,
				}, err
			}
			return nil, err
		}
		usage = streamUsage
		imageCount = streamCount
		firstTokenMs = ttft
	} else {
		nonStreamUsage, nonStreamCount, err := s.handleOpenAIImagesNonStreamingResponse(resp, c)
		if err != nil {
			return nil, err
		}
		usage = nonStreamUsage
		if nonStreamCount > 0 {
			imageCount = nonStreamCount
		}
	}
	if imageCount <= 0 {
		imageCount = parsed.N
	}
	result := &OpenAIForwardResult{
		RequestID:       resp.Header.Get("x-request-id"),
		Usage:           usage,
		Model:           requestModel,
		UpstreamModel:   cfg.model,
		BillingModel:    cfg.model,
		Stream:          parsed.Stream,
		ResponseHeaders: resp.Header.Clone(),
		Duration:        time.Since(startTime),
		FirstTokenMs:    firstTokenMs,
		ImageCount:      imageCount,
		ImageSize:       parsed.SizeTier,
	}
	logOpenAIImageSidecarForwardSuccess(result, parsed.Endpoint, false)
	return result, nil
}

func buildOpenAIImagesSidecarBody(parsed *OpenAIImagesRequest, sidecarModel string) ([]byte, string, error) {
	if parsed == nil {
		return nil, "", fmt.Errorf("parsed images request is required")
	}
	if parsed.Multipart {
		return buildOpenAIImagesSidecarMultipartBody(parsed, sidecarModel)
	}
	body := append([]byte(nil), parsed.Body...)
	var err error
	body, err = sjson.SetBytes(body, "model", strings.TrimSpace(sidecarModel))
	if err != nil {
		return nil, "", fmt.Errorf("rewrite image sidecar model: %w", err)
	}
	if !gjson.GetBytes(body, "response_format").Exists() {
		body, _ = sjson.SetBytes(body, "response_format", "b64_json")
	}
	return body, parsed.ContentType, nil
}

func buildOpenAIImagesSidecarMultipartBody(parsed *OpenAIImagesRequest, sidecarModel string) ([]byte, string, error) {
	_, params, err := mime.ParseMediaType(parsed.ContentType)
	if err != nil {
		return nil, "", fmt.Errorf("parse multipart content-type: %w", err)
	}
	boundary := strings.TrimSpace(params["boundary"])
	if boundary == "" {
		return nil, "", fmt.Errorf("multipart boundary is required")
	}

	reader := multipart.NewReader(bytes.NewReader(parsed.Body), boundary)
	var buffer bytes.Buffer
	writer := multipart.NewWriter(&buffer)
	modelWritten := false
	responseFormatWritten := false

	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, "", fmt.Errorf("read multipart body: %w", err)
		}
		formName := strings.TrimSpace(part.FormName())
		header := cloneMultipartHeader(part.Header)
		target, err := writer.CreatePart(header)
		if err != nil {
			_ = part.Close()
			return nil, "", fmt.Errorf("create multipart part: %w", err)
		}
		if part.FileName() == "" && formName == "model" {
			if _, err := target.Write([]byte(strings.TrimSpace(sidecarModel))); err != nil {
				_ = part.Close()
				return nil, "", fmt.Errorf("rewrite multipart model: %w", err)
			}
			modelWritten = true
			_ = part.Close()
			continue
		}
		if part.FileName() == "" && formName == "response_format" {
			responseFormatWritten = true
		}
		if _, err := io.Copy(target, part); err != nil {
			_ = part.Close()
			return nil, "", fmt.Errorf("copy multipart part: %w", err)
		}
		_ = part.Close()
	}
	if !modelWritten {
		if err := writer.WriteField("model", strings.TrimSpace(sidecarModel)); err != nil {
			return nil, "", fmt.Errorf("append multipart model field: %w", err)
		}
	}
	if !responseFormatWritten {
		if err := writer.WriteField("response_format", "b64_json"); err != nil {
			return nil, "", fmt.Errorf("append multipart response_format field: %w", err)
		}
	}
	if err := writer.Close(); err != nil {
		return nil, "", fmt.Errorf("finalize multipart body: %w", err)
	}
	return buffer.Bytes(), writer.FormDataContentType(), nil
}

func (s *OpenAIGatewayService) forwardOpenAIResponsesImageSidecar(
	ctx context.Context,
	c *gin.Context,
	reqBody map[string]any,
	originalModel string,
	upstreamModel string,
	imageSizeTier string,
	stream bool,
	cfg openAIImageSidecarConfig,
	startTime time.Time,
) (*OpenAIForwardResult, error) {
	parsed, err := buildOpenAIImagesRequestFromResponsesBody(reqBody, cfg.model)
	if err != nil {
		s.writeOpenAIImageSidecarError(c, http.StatusBadRequest, "config", err.Error(), "")
		c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"type": "invalid_request_error", "message": err.Error()}})
		return nil, err
	}
	forwardBody, forwardContentType, err := buildOpenAIImagesSidecarBody(parsed, cfg.model)
	if err != nil {
		s.writeOpenAIImageSidecarError(c, http.StatusBadRequest, "config", err.Error(), "")
		c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"type": "invalid_request_error", "message": err.Error()}})
		return nil, err
	}
	setOpsUpstreamRequestBody(c, forwardBody)

	sidecarCtx := ctx
	var cancel context.CancelFunc
	if cfg.timeout > 0 {
		sidecarCtx, cancel = context.WithTimeout(ctx, cfg.timeout)
		defer cancel()
	}
	upstreamReq, err := http.NewRequestWithContext(sidecarCtx, http.MethodPost, cfg.endpointURL(parsed.Endpoint), bytes.NewReader(forwardBody))
	if err != nil {
		return nil, err
	}
	upstreamReq.Header.Set("Authorization", "Bearer "+cfg.apiKey)
	applyOpenAIImageSidecarTraceHeaders(upstreamReq)
	if strings.TrimSpace(forwardContentType) != "" {
		upstreamReq.Header.Set("Content-Type", forwardContentType)
	}

	upstreamStart := time.Now()
	resp, err := s.httpUpstream.Do(upstreamReq, "", 0, 0)
	SetOpsLatencyMs(c, OpsUpstreamLatencyMsKey, time.Since(upstreamStart).Milliseconds())
	if err != nil {
		s.writeOpenAIImageSidecarError(c, 0, "generation", err.Error(), cfg.endpointURL(parsed.Endpoint))
		c.JSON(http.StatusBadGateway, gin.H{"error": gin.H{"type": "upstream_error", "message": "OpenAI image sidecar request failed"}})
		return nil, fmt.Errorf("openai image sidecar request failed: %s", sanitizeUpstreamErrorMessage(err.Error()))
	}
	if resp.StatusCode >= 400 {
		return nil, s.handleOpenAIImageSidecarErrorResponse(resp, c, "generation", cfg.endpointURL(parsed.Endpoint))
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := ReadUpstreamResponseBody(resp.Body, s.cfg, c, openAITooLargeError)
	if err != nil {
		return nil, err
	}
	results, usage := collectOpenAIImageSidecarResults(body)
	if len(results) == 0 {
		err := fmt.Errorf("openai image sidecar did not return image output")
		s.writeOpenAIImageSidecarError(c, http.StatusBadGateway, "response_wrap", err.Error(), cfg.endpointURL(parsed.Endpoint))
		c.JSON(http.StatusBadGateway, gin.H{"error": gin.H{"type": "upstream_error", "message": err.Error()}})
		return nil, err
	}

	responseBody, responseID, err := buildOpenAIResponsesSidecarImageResponse(originalModel, results, usage)
	if err != nil {
		return nil, err
	}
	responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	if stream {
		c.Header("Content-Type", "text/event-stream")
		c.Status(http.StatusOK)
		_, _ = fmt.Fprintf(c.Writer, "data: %s\n\n", buildOpenAIResponsesSidecarOutputItemEvent(responseBody))
		_, _ = fmt.Fprintf(c.Writer, "data: {\"type\":\"response.completed\",\"response\":%s}\n\n", responseBody)
		_, _ = io.WriteString(c.Writer, "data: [DONE]\n\n")
		if flusher, ok := c.Writer.(http.Flusher); ok {
			flusher.Flush()
		}
	} else {
		c.Data(http.StatusOK, "application/json; charset=utf-8", responseBody)
	}
	result := &OpenAIForwardResult{
		RequestID:       resp.Header.Get("x-request-id"),
		ResponseID:      responseID,
		Usage:           OpenAIUsage{},
		Model:           originalModel,
		UpstreamModel:   upstreamModel,
		BillingModel:    cfg.model,
		Stream:          stream,
		ResponseHeaders: resp.Header.Clone(),
		Duration:        time.Since(startTime),
		ImageCount:      len(results),
		ImageSize:       imageSizeTier,
	}
	logOpenAIImageSidecarForwardSuccess(result, parsed.Endpoint, true)
	return result, nil
}

func applyOpenAIImageSidecarTraceHeaders(req *http.Request) {
	if req == nil {
		return
	}
	req.Header.Set("X-Caller", openAIImageSidecarCaller)
	req.Header.Set("X-Request-Source", openAIImageSidecarSource)
}

func logOpenAIImageSidecarForwardSuccess(result *OpenAIForwardResult, endpoint string, responsesWrapped bool) {
	if result == nil || result.ImageCount <= 0 {
		return
	}
	logger.L().With(
		zap.String("component", "service.openai_gateway"),
		zap.String("request_id", result.RequestID),
		zap.String("response_id", result.ResponseID),
		zap.String("billing_model", result.BillingModel),
		zap.String("model", result.Model),
		zap.Int("image_count", result.ImageCount),
		zap.String("image_size", result.ImageSize),
		zap.String("endpoint", normalizeImageGenerationEndpoint(endpoint)),
		zap.Bool("responses_wrapped", responsesWrapped),
	).Info("openai.image_sidecar_forward_success")
}

func buildOpenAIImagesRequestFromResponsesBody(reqBody map[string]any, sidecarModel string) (*OpenAIImagesRequest, error) {
	prompt := strings.TrimSpace(extractOpenAIResponsesImagePrompt(reqBody))
	if prompt == "" {
		return nil, fmt.Errorf("prompt is required for image generation")
	}
	images := extractOpenAIResponsesInputImages(reqBody)
	tool := firstOpenAIResponsesImageGenerationTool(reqBody)
	size := strings.TrimSpace(firstNonEmptyString(tool["size"]))
	if size == "" {
		size = strings.TrimSpace(firstNonEmptyString(reqBody["size"]))
	}
	outputFormat := strings.TrimSpace(firstNonEmptyString(tool["output_format"]))
	if outputFormat == "" {
		outputFormat = "png"
	}
	n := 1
	if rawN, ok := tool["n"].(float64); ok && rawN > 0 {
		n = int(rawN)
	}
	endpoint := openAIImagesGenerationsEndpoint
	if len(images) > 0 {
		endpoint = openAIImagesEditsEndpoint
	}
	req := &OpenAIImagesRequest{
		Endpoint:       endpoint,
		ContentType:    "application/json",
		Model:          sidecarModel,
		ExplicitModel:  true,
		Prompt:         prompt,
		N:              n,
		Size:           size,
		ResponseFormat: "b64_json",
		Quality:        strings.TrimSpace(firstNonEmptyString(tool["quality"])),
		Background:     strings.TrimSpace(firstNonEmptyString(tool["background"])),
		OutputFormat:   outputFormat,
		Moderation:     strings.TrimSpace(firstNonEmptyString(tool["moderation"])),
		InputFidelity:  strings.TrimSpace(firstNonEmptyString(tool["input_fidelity"])),
		Style:          strings.TrimSpace(firstNonEmptyString(tool["style"])),
		InputImageURLs: images,
	}
	if raw, ok := tool["output_compression"].(float64); ok {
		v := int(raw)
		req.OutputCompression = &v
	}
	if raw, ok := tool["partial_images"].(float64); ok {
		v := int(raw)
		req.PartialImages = &v
	}
	if mask, ok := tool["input_image_mask"].(map[string]any); ok {
		req.MaskImageURL = strings.TrimSpace(firstNonEmptyString(mask["image_url"]))
		req.HasMask = req.MaskImageURL != ""
	}
	req.SizeTier = normalizeOpenAIImageSizeTier(req.Size)
	req.Body = buildOpenAIImagesSidecarJSONBody(req)
	return req, nil
}

func buildOpenAIImagesSidecarJSONBody(req *OpenAIImagesRequest) []byte {
	body := []byte(`{"model":"","prompt":"","n":1,"response_format":"b64_json"}`)
	body, _ = sjson.SetBytes(body, "model", req.Model)
	body, _ = sjson.SetBytes(body, "prompt", req.Prompt)
	body, _ = sjson.SetBytes(body, "n", req.N)
	if req.Size != "" {
		body, _ = sjson.SetBytes(body, "size", req.Size)
	}
	for _, field := range []struct {
		path  string
		value string
	}{
		{path: "quality", value: req.Quality},
		{path: "background", value: req.Background},
		{path: "output_format", value: req.OutputFormat},
		{path: "moderation", value: req.Moderation},
		{path: "input_fidelity", value: req.InputFidelity},
		{path: "style", value: req.Style},
	} {
		if strings.TrimSpace(field.value) != "" {
			body, _ = sjson.SetBytes(body, field.path, field.value)
		}
	}
	if req.OutputCompression != nil {
		body, _ = sjson.SetBytes(body, "output_compression", *req.OutputCompression)
	}
	if req.PartialImages != nil {
		body, _ = sjson.SetBytes(body, "partial_images", *req.PartialImages)
	}
	if len(req.InputImageURLs) > 0 {
		body, _ = sjson.SetRawBytes(body, "images", []byte(`[]`))
		for _, imageURL := range req.InputImageURLs {
			item := []byte(`{"image_url":""}`)
			item, _ = sjson.SetBytes(item, "image_url", imageURL)
			body, _ = sjson.SetRawBytes(body, "images.-1", item)
		}
	}
	if req.MaskImageURL != "" {
		body, _ = sjson.SetBytes(body, "mask.image_url", req.MaskImageURL)
	}
	return body
}

func extractOpenAIResponsesImagePrompt(reqBody map[string]any) string {
	if reqBody == nil {
		return ""
	}
	if input, ok := reqBody["input"].(string); ok {
		return strings.TrimSpace(input)
	}
	var parts []string
	collectText := func(v any) {
		switch item := v.(type) {
		case string:
			if trimmed := strings.TrimSpace(item); trimmed != "" {
				parts = append(parts, trimmed)
			}
		case map[string]any:
			itemType := strings.TrimSpace(firstNonEmptyString(item["type"]))
			if itemType == "" || itemType == "input_text" || itemType == "text" {
				if text := strings.TrimSpace(firstNonEmptyString(item["text"])); text != "" {
					parts = append(parts, text)
				}
			}
		}
	}
	if inputItems, ok := reqBody["input"].([]any); ok {
		for _, rawItem := range inputItems {
			item, ok := rawItem.(map[string]any)
			if !ok {
				collectText(rawItem)
				continue
			}
			if content, ok := item["content"].([]any); ok {
				for _, part := range content {
					collectText(part)
				}
				continue
			}
			if text := strings.TrimSpace(extractTextFromContent(item["content"])); text != "" {
				parts = append(parts, text)
				continue
			}
			collectText(item)
		}
	}
	return strings.TrimSpace(strings.Join(parts, "\n"))
}

func extractOpenAIResponsesInputImages(reqBody map[string]any) []string {
	if reqBody == nil {
		return nil
	}
	var images []string
	var walk func(any)
	walk = func(v any) {
		switch item := v.(type) {
		case []any:
			for _, child := range item {
				walk(child)
			}
		case map[string]any:
			itemType := strings.TrimSpace(firstNonEmptyString(item["type"]))
			if itemType == "input_image" {
				for _, key := range []string{"image_url", "url"} {
					if imageURL := strings.TrimSpace(firstNonEmptyString(item[key])); imageURL != "" {
						images = append(images, imageURL)
						return
					}
				}
			}
			for _, child := range item {
				walk(child)
			}
		}
	}
	walk(reqBody["input"])
	return images
}

func firstOpenAIResponsesImageGenerationTool(reqBody map[string]any) map[string]any {
	if reqBody == nil {
		return nil
	}
	tools, _ := reqBody["tools"].([]any)
	for _, rawTool := range tools {
		tool, ok := rawTool.(map[string]any)
		if !ok || strings.TrimSpace(firstNonEmptyString(tool["type"])) != "image_generation" {
			continue
		}
		return tool
	}
	return nil
}

func collectOpenAIImageSidecarResults(body []byte) ([]openAIResponsesImageResult, OpenAIUsage) {
	var usage OpenAIUsage
	if parsedUsage, ok := extractOpenAIUsageFromJSONBytes(body); ok {
		usage = parsedUsage
	}
	if results, _, _, _, ok, _ := collectOpenAIImagesFromResponsesBody(body); ok && len(results) > 0 {
		return results, usage
	}
	var results []openAIResponsesImageResult
	data := gjson.GetBytes(body, "data")
	if data.IsArray() {
		for _, item := range data.Array() {
			result := strings.TrimSpace(item.Get("b64_json").String())
			if result == "" {
				result = dataURLBase64Payload(item.Get("url").String())
			}
			if result == "" {
				continue
			}
			results = append(results, openAIResponsesImageResult{
				Result:        result,
				RevisedPrompt: strings.TrimSpace(item.Get("revised_prompt").String()),
				OutputFormat:  strings.TrimSpace(gjson.GetBytes(body, "output_format").String()),
				Size:          strings.TrimSpace(gjson.GetBytes(body, "size").String()),
				Background:    strings.TrimSpace(gjson.GetBytes(body, "background").String()),
				Quality:       strings.TrimSpace(gjson.GetBytes(body, "quality").String()),
				Model:         strings.TrimSpace(gjson.GetBytes(body, "model").String()),
			})
		}
	}
	return results, usage
}

func dataURLBase64Payload(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if !strings.HasPrefix(trimmed, "data:") {
		return ""
	}
	idx := strings.Index(trimmed, ",")
	if idx < 0 {
		return ""
	}
	return strings.TrimSpace(trimmed[idx+1:])
}

func buildOpenAIResponsesSidecarImageResponse(model string, results []openAIResponsesImageResult, usage OpenAIUsage) ([]byte, string, error) {
	responseID := fmt.Sprintf("resp_%d", time.Now().UnixNano())
	body := []byte(`{"id":"","object":"response","created_at":0,"status":"completed","model":"","output":[],"usage":{"input_tokens":0,"output_tokens":0,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"image_tokens":0}}}`)
	body, _ = sjson.SetBytes(body, "id", responseID)
	body, _ = sjson.SetBytes(body, "created_at", time.Now().Unix())
	body, _ = sjson.SetBytes(body, "model", strings.TrimSpace(model))
	body, _ = sjson.SetBytes(body, "usage.input_tokens", usage.InputTokens)
	body, _ = sjson.SetBytes(body, "usage.output_tokens", usage.OutputTokens)
	body, _ = sjson.SetBytes(body, "usage.input_tokens_details.cached_tokens", usage.CacheReadInputTokens)
	body, _ = sjson.SetBytes(body, "usage.output_tokens_details.image_tokens", usage.ImageOutputTokens)
	for idx, img := range results {
		item := []byte(`{"id":"","type":"image_generation_call","status":"completed","result":""}`)
		item, _ = sjson.SetBytes(item, "id", fmt.Sprintf("ig_%d", idx+1))
		item, _ = sjson.SetBytes(item, "result", img.Result)
		for _, field := range []struct {
			path  string
			value string
		}{
			{path: "revised_prompt", value: img.RevisedPrompt},
			{path: "output_format", value: img.OutputFormat},
			{path: "size", value: img.Size},
			{path: "background", value: img.Background},
			{path: "quality", value: img.Quality},
		} {
			if strings.TrimSpace(field.value) != "" {
				item, _ = sjson.SetBytes(item, field.path, field.value)
			}
		}
		body, _ = sjson.SetRawBytes(body, "output.-1", item)
	}
	return body, responseID, nil
}

func buildOpenAIResponsesSidecarOutputItemEvent(responseBody []byte) []byte {
	item := gjson.GetBytes(responseBody, "output.0")
	event := []byte(`{"type":"response.output_item.done","item":{}}`)
	if item.Exists() {
		event, _ = sjson.SetRawBytes(event, "item", []byte(item.Raw))
	}
	return event
}

func (s *OpenAIGatewayService) handleOpenAIImageSidecarErrorResponse(resp *http.Response, c *gin.Context, stage string, upstreamURL string) error {
	respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	_ = resp.Body.Close()
	resp.Body = io.NopCloser(bytes.NewReader(respBody))
	message := sanitizeUpstreamErrorMessage(strings.TrimSpace(extractUpstreamErrorMessage(respBody)))
	if message == "" {
		message = "OpenAI image sidecar request failed"
	}
	s.writeOpenAIImageSidecarError(c, resp.StatusCode, stage, message, upstreamURL)
	if c != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": gin.H{"type": "upstream_error", "message": message}})
	}
	return fmt.Errorf("openai image sidecar error: %d message=%s", resp.StatusCode, message)
}

func (s *OpenAIGatewayService) writeOpenAIImageSidecarError(c *gin.Context, status int, stage string, message string, upstreamURL string) {
	safeMessage := sanitizeUpstreamErrorMessage(message)
	setOpsUpstreamError(c, status, safeMessage, "")
	appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
		Platform:           PlatformOpenAI,
		UpstreamStatusCode: status,
		UpstreamURL:        safeUpstreamURL(upstreamURL),
		Kind:               "openai_image_sidecar_" + strings.TrimSpace(stage),
		Message:            safeMessage,
	})
}
