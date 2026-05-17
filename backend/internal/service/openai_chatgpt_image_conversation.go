package service

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	openAIChatGPTRequirementsURL = "https://chatgpt.com/backend-api/sentinel/chat-requirements"
	openAIChatGPTPrepareURL      = "https://chatgpt.com/backend-api/f/conversation/prepare"
	openAIChatGPTConversationURL = "https://chatgpt.com/backend-api/f/conversation"
	openAIChatGPTConversationAPI = "https://chatgpt.com/backend-api/conversation"

	openAIChatGPTImageBillingModel = "gpt-image-2"
	openAIChatGPTImageOutputFormat = "png"
	openAIChatGPTImagePollAttempts = 8
	openAIChatGPTImagePollDelay    = 4 * time.Second
)

type openAIChatGPTImageConversationRequest struct {
	Prompt       string
	MainModel    string
	ImageModel   string
	Size         string
	Quality      string
	Background   string
	OutputFormat string
	Uploads      []OpenAIImagesUpload
	InputURLs    []string
	Stream       bool
	ResponseKind string
}

type openAIChatGPTImageConversationResult struct {
	RequestID    string
	Conversation string
	CreatedAt    int64
	Usage        OpenAIUsage
	Images       []openAIResponsesImageResult
}

type openAIChatGPTImageUploadRef struct {
	FileID      string
	FileName    string
	FileSize    int
	MimeType    string
	Width       int
	Height      int
	AssetURL    string
}

type openAIChatGPTImageStageError struct {
	Stage      string
	StatusCode int
	Message    string
	Body       []byte
	URL        string
	Err        error
}

func (e *openAIChatGPTImageStageError) Error() string {
	if e == nil {
		return ""
	}
	msg := strings.TrimSpace(e.Message)
	if msg == "" && e.Err != nil {
		msg = e.Err.Error()
	}
	if msg == "" {
		msg = "ChatGPT image conversation failed"
	}
	if e.Stage != "" {
		return e.Stage + ": " + msg
	}
	return msg
}

func (e *openAIChatGPTImageStageError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

func (s *OpenAIGatewayService) forwardOpenAIChatGPTImageConversation(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	token string,
	req openAIChatGPTImageConversationRequest,
) (*openAIChatGPTImageConversationResult, error) {
	req.Prompt = strings.TrimSpace(req.Prompt)
	if req.Prompt == "" {
		return nil, &openAIChatGPTImageStageError{Stage: "request", Message: "image prompt is required"}
	}
	if req.ImageModel == "" {
		req.ImageModel = openAIChatGPTImageBillingModel
	}
	if req.OutputFormat == "" {
		req.OutputFormat = openAIChatGPTImageOutputFormat
	}
	headers := openAIChatGPTImageHeaders(account, token)
	refs := make([]openAIChatGPTImageUploadRef, 0, len(req.Uploads))
	for _, imageURL := range req.InputURLs {
		upload, err := s.downloadOpenAIChatGPTImageInputURL(ctx, c, account, imageURL)
		if err != nil {
			return nil, err
		}
		req.Uploads = append(req.Uploads, upload)
	}
	for _, upload := range req.Uploads {
		ref, err := s.uploadOpenAIChatGPTImageReference(ctx, c, account, headers, upload)
		if err != nil {
			return nil, err
		}
		refs = append(refs, ref)
	}
	requirements, err := s.openAIChatGPTImageRequirements(ctx, c, account, headers)
	if err != nil {
		return nil, err
	}
	conduitToken, err := s.prepareOpenAIChatGPTImageConversation(ctx, c, account, headers, requirements, req)
	if err != nil {
		return nil, err
	}
	body, requestID, err := s.startOpenAIChatGPTImageConversation(ctx, c, account, headers, requirements, conduitToken, req, refs)
	if err != nil {
		return nil, err
	}
	result, err := s.collectOpenAIChatGPTImageConversationResult(ctx, c, account, headers, body, requestID, req)
	if err != nil {
		return nil, err
	}
	if len(result.Images) == 0 {
		return nil, &openAIChatGPTImageStageError{Stage: "poll", Message: "ChatGPT image conversation completed without image output"}
	}
	return result, nil
}

func openAIChatGPTImageHeaders(account *Account, token string) http.Header {
	headers := http.Header{}
	headers.Set("Authorization", "Bearer "+token)
	headers.Set("Accept", "application/json")
	headers.Set("Content-Type", "application/json")
	headers.Set("Origin", "https://chatgpt.com")
	headers.Set("Referer", "https://chatgpt.com/")
	headers.Set("User-Agent", openAIImageBackendUserAgent)
	if account != nil {
		if ua := strings.TrimSpace(account.GetOpenAIUserAgent()); ua != "" {
			headers.Set("User-Agent", ua)
		}
		if chatgptAccountID := strings.TrimSpace(account.GetChatGPTAccountID()); chatgptAccountID != "" {
			headers.Set("chatgpt-account-id", chatgptAccountID)
		}
	}
	return headers
}

func (s *OpenAIGatewayService) openAIChatGPTImageDo(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	headers http.Header,
	method string,
	rawURL string,
	body []byte,
	stage string,
) (*http.Response, []byte, error) {
	var reader io.Reader
	if len(body) > 0 {
		reader = bytes.NewReader(body)
		setOpsUpstreamRequestBody(c, body)
	}
	req, err := http.NewRequestWithContext(ctx, method, rawURL, reader)
	if err != nil {
		return nil, nil, &openAIChatGPTImageStageError{Stage: stage, URL: rawURL, Err: err}
	}
	if strings.HasPrefix(rawURL, openAIChatGPTStartURL) {
		req.Host = "chatgpt.com"
	}
	if !strings.HasPrefix(rawURL, openAIChatGPTStartURL) {
		req.Header.Del("Authorization")
		req.Header.Del("chatgpt-account-id")
		req.Header.Del("OpenAI-Sentinel-Chat-Requirements-Token")
		req.Header.Del("OpenAI-Sentinel-Proof-Token")
		req.Header.Del("X-Conduit-Token")
	}
	for key, values := range headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}
	if method == http.MethodGet {
		req.Header.Del("Content-Type")
	}
	proxyURL := ""
	if account != nil && account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}
	start := time.Now()
	resp, err := s.httpUpstream.Do(req, proxyURL, account.ID, account.Concurrency)
	SetOpsLatencyMs(c, OpsUpstreamLatencyMsKey, time.Since(start).Milliseconds())
	if err != nil {
		return nil, nil, s.openAIChatGPTImageStageRequestError(c, account, stage, rawURL, err)
	}
	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
		_ = resp.Body.Close()
		return resp, respBody, s.openAIChatGPTImageStageHTTPError(c, account, stage, rawURL, resp, respBody)
	}
	return resp, nil, nil
}

func (s *OpenAIGatewayService) openAIChatGPTImageStageRequestError(c *gin.Context, account *Account, stage, rawURL string, err error) error {
	safeErr := sanitizeUpstreamErrorMessage(err.Error())
	setOpsUpstreamError(c, 0, stage+": "+safeErr, "")
	appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
		Platform:    account.Platform,
		AccountID:   account.ID,
		AccountName: account.Name,
		UpstreamURL: safeUpstreamURL(rawURL),
		Kind:        "request_error",
		Message:     stage + ": " + safeErr,
	})
	return &openAIChatGPTImageStageError{Stage: stage, URL: rawURL, Message: safeErr, Err: err}
}

func (s *OpenAIGatewayService) openAIChatGPTImageStageHTTPError(c *gin.Context, account *Account, stage, rawURL string, resp *http.Response, body []byte) error {
	upstreamMsg := strings.TrimSpace(extractUpstreamErrorMessage(body))
	if upstreamMsg == "" {
		upstreamMsg = http.StatusText(resp.StatusCode)
	}
	upstreamMsg = sanitizeUpstreamErrorMessage(upstreamMsg)
	setOpsUpstreamError(c, resp.StatusCode, stage+": "+upstreamMsg, truncateString(string(body), 2048))
	appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
		Platform:             account.Platform,
		AccountID:            account.ID,
		AccountName:          account.Name,
		UpstreamStatusCode:   resp.StatusCode,
		UpstreamRequestID:    resp.Header.Get("x-request-id"),
		UpstreamURL:          safeUpstreamURL(rawURL),
		UpstreamResponseBody: truncateString(string(body), 2048),
		Kind:                 "http_error",
		Message:              stage + ": " + upstreamMsg,
	})
	return &openAIChatGPTImageStageError{
		Stage:      stage,
		StatusCode: resp.StatusCode,
		Message:    upstreamMsg,
		Body:       body,
		URL:        rawURL,
	}
}

func (s *OpenAIGatewayService) openAIChatGPTImageRequirements(ctx context.Context, c *gin.Context, account *Account, headers http.Header) (map[string]string, error) {
	body := []byte(`{"p":""}`)
	resp, respBody, err := s.openAIChatGPTImageDo(ctx, c, account, headers, http.MethodPost, openAIChatGPTRequirementsURL, body, "requirements")
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if respBody == nil {
		respBody, _ = io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	}
	return map[string]string{
		"token":       strings.TrimSpace(gjson.GetBytes(respBody, "token").String()),
		"proof_token": strings.TrimSpace(gjson.GetBytes(respBody, "proof_token").String()),
	}, nil
}

func (s *OpenAIGatewayService) prepareOpenAIChatGPTImageConversation(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	headers http.Header,
	requirements map[string]string,
	req openAIChatGPTImageConversationRequest,
) (string, error) {
	payload := map[string]any{
		"action":               "next",
		"fork_from_shared_post": false,
		"parent_message_id":    uuid.NewString(),
		"model":                openAIChatGPTImageModelSlug(req.ImageModel),
		"client_prepare_state": "success",
		"timezone_offset_min":  -480,
		"timezone":             "Asia/Shanghai",
		"conversation_mode":    map[string]any{"kind": "primary_assistant"},
		"system_hints":         []string{"picture_v2"},
		"partial_query": map[string]any{
			"id":      uuid.NewString(),
			"author":  map[string]any{"role": "user"},
			"content": map[string]any{"content_type": "text", "parts": []string{req.Prompt}},
		},
		"supports_buffering":      true,
		"supported_encodings":     []string{"v1"},
		"client_contextual_info":  map[string]any{"app_name": "chatgpt.com"},
	}
	body, _ := json.Marshal(payload)
	stageHeaders := cloneHTTPHeader(headers)
	applyOpenAIChatGPTImageRequirementHeaders(stageHeaders, requirements, "")
	resp, respBody, err := s.openAIChatGPTImageDo(ctx, c, account, stageHeaders, http.MethodPost, openAIChatGPTPrepareURL, body, "prepare")
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()
	if respBody == nil {
		respBody, _ = io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	}
	conduit := strings.TrimSpace(gjson.GetBytes(respBody, "conduit_token").String())
	if conduit == "" {
		return "", &openAIChatGPTImageStageError{Stage: "prepare", Message: "missing conduit_token", Body: respBody, URL: openAIChatGPTPrepareURL}
	}
	return conduit, nil
}

func (s *OpenAIGatewayService) startOpenAIChatGPTImageConversation(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	headers http.Header,
	requirements map[string]string,
	conduitToken string,
	req openAIChatGPTImageConversationRequest,
	refs []openAIChatGPTImageUploadRef,
) ([]byte, string, error) {
	payload := buildOpenAIChatGPTImageConversationPayload(req, refs)
	body, _ := json.Marshal(payload)
	stageHeaders := cloneHTTPHeader(headers)
	stageHeaders.Set("Accept", "text/event-stream")
	applyOpenAIChatGPTImageRequirementHeaders(stageHeaders, requirements, conduitToken)
	resp, _, err := s.openAIChatGPTImageDo(ctx, c, account, stageHeaders, http.MethodPost, openAIChatGPTConversationURL, body, "conversation")
	if err != nil {
		return nil, "", err
	}
	defer func() { _ = resp.Body.Close() }()
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return nil, resp.Header.Get("x-request-id"), &openAIChatGPTImageStageError{Stage: "conversation", Message: err.Error(), Err: err, URL: openAIChatGPTConversationURL}
	}
	return respBody, resp.Header.Get("x-request-id"), nil
}

func buildOpenAIChatGPTImageConversationPayload(req openAIChatGPTImageConversationRequest, refs []openAIChatGPTImageUploadRef) map[string]any {
	parts := make([]any, 0, len(refs)+1)
	for _, ref := range refs {
		parts = append(parts, map[string]any{
			"content_type":  "image_asset_pointer",
			"asset_pointer": "file-service://" + ref.FileID,
			"width":         ref.Width,
			"height":        ref.Height,
			"size_bytes":    ref.FileSize,
		})
	}
	parts = append(parts, req.Prompt)
	content := map[string]any{"content_type": "text", "parts": []string{req.Prompt}}
	if len(refs) > 0 {
		content = map[string]any{"content_type": "multimodal_text", "parts": parts}
	}
	metadata := map[string]any{
		"developer_mode_connector_ids": []string{},
		"selected_github_repos":        []string{},
		"selected_all_github_repos":    false,
		"system_hints":                 []string{"picture_v2"},
		"serialization_metadata":       map[string]any{"custom_symbol_offsets": []any{}},
	}
	if len(refs) > 0 {
		attachments := make([]map[string]any, 0, len(refs))
		for _, ref := range refs {
			attachments = append(attachments, map[string]any{
				"id":       ref.FileID,
				"mimeType": ref.MimeType,
				"name":     ref.FileName,
				"size":     ref.FileSize,
				"width":    ref.Width,
				"height":   ref.Height,
			})
		}
		metadata["attachments"] = attachments
	}
	return map[string]any{
		"action": "next",
		"messages": []map[string]any{{
			"id":          uuid.NewString(),
			"author":      map[string]any{"role": "user"},
			"create_time": float64(time.Now().UnixMilli()) / 1000,
			"content":     content,
			"metadata":    metadata,
		}},
		"parent_message_id":       uuid.NewString(),
		"model":                   openAIChatGPTImageModelSlug(req.ImageModel),
		"client_prepare_state":    "sent",
		"timezone_offset_min":     -480,
		"timezone":                "Asia/Shanghai",
		"conversation_mode":       map[string]any{"kind": "primary_assistant"},
		"enable_message_followups": true,
		"system_hints":            []string{"picture_v2"},
		"supports_buffering":      true,
		"supported_encodings":     []string{"v1"},
		"client_contextual_info": map[string]any{
			"is_dark_mode":      false,
			"time_since_loaded": 1200,
			"page_height":       1072,
			"page_width":        1724,
			"pixel_ratio":       1.2,
			"screen_height":     1440,
			"screen_width":      2560,
			"app_name":          "chatgpt.com",
		},
		"paragen_cot_summary_display_override": "allow",
		"force_parallel_switch":                "auto",
	}
}

func applyOpenAIChatGPTImageRequirementHeaders(headers http.Header, requirements map[string]string, conduitToken string) {
	if token := strings.TrimSpace(requirements["token"]); token != "" {
		headers.Set("OpenAI-Sentinel-Chat-Requirements-Token", token)
	}
	if proof := strings.TrimSpace(requirements["proof_token"]); proof != "" {
		headers.Set("OpenAI-Sentinel-Proof-Token", proof)
	}
	if conduitToken != "" {
		headers.Set("X-Conduit-Token", conduitToken)
	}
}

func openAIChatGPTImageModelSlug(model string) string {
	switch strings.TrimSpace(model) {
	case "gpt-image-2", "":
		return "gpt-5-3"
	default:
		return strings.TrimSpace(model)
	}
}

func (s *OpenAIGatewayService) collectOpenAIChatGPTImageConversationResult(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	headers http.Header,
	body []byte,
	requestID string,
	req openAIChatGPTImageConversationRequest,
) (*openAIChatGPTImageConversationResult, error) {
	conversationID, createdAt, usage, pointers := parseOpenAIChatGPTImageConversationSSE(body, req)
	if createdAt <= 0 {
		createdAt = time.Now().Unix()
	}
	if len(pointers) == 0 && conversationID != "" {
		polled, err := s.pollOpenAIChatGPTImageConversation(ctx, c, account, headers, conversationID, req.Prompt)
		if err != nil {
			return nil, err
		}
		pointers = mergeOpenAIImagePointerInfos(pointers, polled)
	}
	images := make([]openAIResponsesImageResult, 0, len(pointers))
	for _, pointer := range pointers {
		imageBytes, err := s.resolveOpenAIChatGPTImageBytes(ctx, c, account, headers, conversationID, pointer)
		if err != nil {
			return nil, err
		}
		images = append(images, openAIResponsesImageResult{
			Result:        base64.StdEncoding.EncodeToString(imageBytes),
			RevisedPrompt: firstNonEmptyString(pointer.Prompt, req.Prompt),
			OutputFormat:  req.OutputFormat,
			Size:          req.Size,
			Background:    req.Background,
			Quality:       req.Quality,
			Model:         openAIChatGPTImageBillingModel,
		})
	}
	return &openAIChatGPTImageConversationResult{
		RequestID:    requestID,
		Conversation: conversationID,
		CreatedAt:    createdAt,
		Usage:        usage,
		Images:       images,
	}, nil
}

func parseOpenAIChatGPTImageConversationSSE(body []byte, req openAIChatGPTImageConversationRequest) (string, int64, OpenAIUsage, []openAIImagePointerInfo) {
	var conversationID string
	var createdAt int64
	var usage OpenAIUsage
	var pointers []openAIImagePointerInfo
	scanner := bufio.NewScanner(bytes.NewReader(body))
	scanner.Buffer(make([]byte, 0, 64*1024), defaultMaxLineSize)
	for scanner.Scan() {
		line := scanner.Text()
		data, ok := extractOpenAISSEDataLine(line)
		if !ok || strings.TrimSpace(data) == "[DONE]" {
			continue
		}
		dataBytes := []byte(data)
		if cid := strings.TrimSpace(gjson.GetBytes(dataBytes, "conversation_id").String()); cid != "" {
			conversationID = cid
		}
		if cid := strings.TrimSpace(gjson.GetBytes(dataBytes, "v.conversation_id").String()); cid != "" {
			conversationID = cid
		}
		if id := strings.TrimSpace(gjson.GetBytes(dataBytes, "response.conversation_id").String()); id != "" {
			conversationID = id
		}
		if createdAt == 0 {
			createdAt = firstPositiveInt64(
				gjson.GetBytes(dataBytes, "create_time").Int(),
				gjson.GetBytes(dataBytes, "message.create_time").Int(),
				gjson.GetBytes(dataBytes, "response.created_at").Int(),
			)
		}
		if parsed, ok := extractOpenAIUsageFromJSONBytes([]byte(`{"usage":` + gjson.GetBytes(dataBytes, "response.usage").Raw + `}`)); ok && gjson.GetBytes(dataBytes, "response.usage").Exists() {
			usage = parsed
		}
		if openAIChatGPTImageSSEDataContainsGeneratedImage(dataBytes) {
			for _, pointer := range collectOpenAIImagePointers(dataBytes) {
				if pointer.Prompt == "" {
					pointer.Prompt = req.Prompt
				}
				pointers = mergeOpenAIImagePointerInfos(pointers, []openAIImagePointerInfo{pointer})
			}
		}
	}
	return conversationID, createdAt, usage, pointers
}

func openAIChatGPTImageSSEDataContainsGeneratedImage(data []byte) bool {
	if len(data) == 0 {
		return false
	}
	if strings.TrimSpace(gjson.GetBytes(data, "message.metadata.async_task_type").String()) == "image_gen" {
		return true
	}
	if strings.TrimSpace(gjson.GetBytes(data, "metadata.async_task_type").String()) == "image_gen" {
		return true
	}
	raw := string(data)
	return strings.Contains(raw, `"async_task_type"`) && strings.Contains(raw, `"image_gen"`)
}

func firstPositiveInt64(values ...int64) int64 {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func (s *OpenAIGatewayService) pollOpenAIChatGPTImageConversation(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	headers http.Header,
	conversationID string,
	prompt string,
) ([]openAIImagePointerInfo, error) {
	url := openAIChatGPTConversationAPI + "/" + conversationID
	for attempt := 0; attempt < openAIChatGPTImagePollAttempts; attempt++ {
		resp, body, err := s.openAIChatGPTImageDo(ctx, c, account, headers, http.MethodGet, url, nil, "poll")
		if err != nil {
			return nil, err
		}
		if body == nil {
			body, _ = io.ReadAll(io.LimitReader(resp.Body, 8<<20))
			_ = resp.Body.Close()
		}
		pointers := collectOpenAIChatGPTImageConversationPointers(body, prompt)
		if len(pointers) > 0 {
			return pointers, nil
		}
		if attempt == openAIChatGPTImagePollAttempts-1 {
			break
		}
		timer := time.NewTimer(openAIChatGPTImagePollDelay)
		select {
		case <-ctx.Done():
			if !timer.Stop() {
				<-timer.C
			}
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
	return nil, nil
}

func collectOpenAIChatGPTImageConversationPointers(body []byte, prompt string) []openAIImagePointerInfo {
	if len(body) == 0 || !gjson.ValidBytes(body) {
		return nil
	}
	var decoded any
	if err := json.Unmarshal(body, &decoded); err != nil {
		return nil
	}
	var out []openAIImagePointerInfo
	walkOpenAIChatGPTImageConversationPointers(decoded, strings.TrimSpace(prompt), &out)
	return out
}

func walkOpenAIChatGPTImageConversationPointers(node any, prompt string, out *[]openAIImagePointerInfo) {
	switch value := node.(type) {
	case map[string]any:
		if author, _ := value["author"].(map[string]any); strings.TrimSpace(firstNonEmptyString(author["role"])) == "tool" {
			metadata, _ := value["metadata"].(map[string]any)
			if strings.TrimSpace(firstNonEmptyString(metadata["async_task_type"])) == "image_gen" {
				if raw, err := json.Marshal(value); err == nil {
					for _, pointer := range collectOpenAIImagePointers(raw) {
						if pointer.Prompt == "" {
							pointer.Prompt = prompt
						}
						*out = mergeOpenAIImagePointerInfos(*out, []openAIImagePointerInfo{pointer})
					}
				}
			}
		}
		for _, child := range value {
			walkOpenAIChatGPTImageConversationPointers(child, prompt, out)
		}
	case []any:
		for _, child := range value {
			walkOpenAIChatGPTImageConversationPointers(child, prompt, out)
		}
	}
}

func (s *OpenAIGatewayService) resolveOpenAIChatGPTImageBytes(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	headers http.Header,
	conversationID string,
	pointer openAIImagePointerInfo,
) ([]byte, error) {
	if normalized := normalizeOpenAIImageBase64(pointer.B64JSON); normalized != "" {
		return base64.StdEncoding.DecodeString(normalized)
	}
	downloadURL := strings.TrimSpace(pointer.DownloadURL)
	if downloadURL == "" && strings.TrimSpace(pointer.Pointer) != "" {
		resolvedURL, err := s.fetchOpenAIChatGPTImageDownloadURL(ctx, c, account, headers, conversationID, pointer.Pointer)
		if err != nil {
			return nil, err
		}
		downloadURL = resolvedURL
	}
	if downloadURL == "" {
		return nil, &openAIChatGPTImageStageError{Stage: "download", Message: "image asset is missing pointer, url, and base64 data"}
	}
	return s.downloadOpenAIChatGPTImageBytes(ctx, c, account, headers, downloadURL)
}

func (s *OpenAIGatewayService) fetchOpenAIChatGPTImageDownloadURL(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	headers http.Header,
	conversationID string,
	pointer string,
) (string, error) {
	var url string
	switch {
	case strings.HasPrefix(pointer, "file-service://"):
		fileID := strings.TrimPrefix(pointer, "file-service://")
		url = openAIChatGPTFilesURL + "/" + fileID + "/download"
	case strings.HasPrefix(pointer, "sediment://"):
		if strings.TrimSpace(conversationID) == "" {
			return "", &openAIChatGPTImageStageError{Stage: "download", Message: "conversation id is required for sediment attachment"}
		}
		attachmentID := strings.TrimPrefix(pointer, "sediment://")
		url = openAIChatGPTConversationAPI + "/" + conversationID + "/attachment/" + attachmentID + "/download"
	default:
		return "", &openAIChatGPTImageStageError{Stage: "download", Message: "unsupported image pointer: " + pointer}
	}
	resp, body, err := s.openAIChatGPTImageDo(ctx, c, account, headers, http.MethodGet, url, nil, "download")
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()
	if body == nil {
		body, _ = io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	}
	downloadURL := strings.TrimSpace(firstNonEmptyString(
		gjson.GetBytes(body, "download_url").String(),
		gjson.GetBytes(body, "url").String(),
	))
	if downloadURL == "" {
		return "", &openAIChatGPTImageStageError{Stage: "download", Message: "missing image download url", Body: body, URL: url}
	}
	return downloadURL, nil
}

func (s *OpenAIGatewayService) downloadOpenAIChatGPTImageBytes(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	headers http.Header,
	downloadURL string,
) ([]byte, error) {
	downloadHeaders := cloneHTTPHeader(headers)
	downloadHeaders.Set("Accept", "image/*,*/*;q=0.8")
	downloadHeaders.Del("Content-Type")
	if !strings.HasPrefix(downloadURL, openAIChatGPTStartURL) {
		userAgent := strings.TrimSpace(headers.Get("User-Agent"))
		if userAgent == "" {
			userAgent = openAIImageBackendUserAgent
		}
		downloadHeaders = http.Header{}
		downloadHeaders.Set("User-Agent", userAgent)
		downloadHeaders.Set("Accept", "image/*,*/*;q=0.8")
	}
	resp, _, err := s.openAIChatGPTImageDo(ctx, c, account, downloadHeaders, http.MethodGet, downloadURL, nil, "download")
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	return io.ReadAll(io.LimitReader(resp.Body, openAIImageMaxDownloadBytes))
}

func (s *OpenAIGatewayService) uploadOpenAIChatGPTImageReference(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	headers http.Header,
	upload OpenAIImagesUpload,
) (openAIChatGPTImageUploadRef, error) {
	if len(upload.Data) == 0 {
		return openAIChatGPTImageUploadRef{}, &openAIChatGPTImageStageError{Stage: "upload", Message: "image upload is empty"}
	}
	contentType := strings.TrimSpace(upload.ContentType)
	if contentType == "" {
		contentType = http.DetectContentType(upload.Data)
	}
	fileName := strings.TrimSpace(upload.FileName)
	if fileName == "" {
		fileName = "image.png"
	}
	metaReq := map[string]any{
		"file_name": fileName,
		"file_size": len(upload.Data),
		"use_case":  "multimodal",
	}
	if upload.Width > 0 {
		metaReq["width"] = upload.Width
	}
	if upload.Height > 0 {
		metaReq["height"] = upload.Height
	}
	body, _ := json.Marshal(metaReq)
	resp, respBody, err := s.openAIChatGPTImageDo(ctx, c, account, headers, http.MethodPost, openAIChatGPTFilesURL, body, "upload")
	if err != nil {
		return openAIChatGPTImageUploadRef{}, err
	}
	if respBody == nil {
		respBody, _ = io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	}
	_ = resp.Body.Close()
	fileID := strings.TrimSpace(gjson.GetBytes(respBody, "file_id").String())
	uploadURL := strings.TrimSpace(gjson.GetBytes(respBody, "upload_url").String())
	if fileID == "" || uploadURL == "" {
		return openAIChatGPTImageUploadRef{}, &openAIChatGPTImageStageError{Stage: "upload", Message: "missing file_id or upload_url", Body: respBody, URL: openAIChatGPTFilesURL}
	}
	putHeaders := http.Header{}
	putHeaders.Set("Content-Type", contentType)
	putHeaders.Set("x-ms-blob-type", "BlockBlob")
	putHeaders.Set("x-ms-version", "2020-04-08")
	putHeaders.Set("Origin", "https://chatgpt.com")
	putHeaders.Set("Referer", "https://chatgpt.com/")
	putHeaders.Set("User-Agent", headers.Get("User-Agent"))
	putHeaders.Set("Accept", "application/json, text/plain, */*")
	resp, _, err = s.openAIChatGPTImageDo(ctx, c, account, putHeaders, http.MethodPut, uploadURL, upload.Data, "upload")
	if err != nil {
		return openAIChatGPTImageUploadRef{}, err
	}
	_ = resp.Body.Close()
	uploadedURL := openAIChatGPTFilesURL + "/" + fileID + "/uploaded"
	resp, _, err = s.openAIChatGPTImageDo(ctx, c, account, headers, http.MethodPost, uploadedURL, []byte(`{}`), "upload")
	if err != nil {
		return openAIChatGPTImageUploadRef{}, err
	}
	_ = resp.Body.Close()
	width, height := upload.Width, upload.Height
	if width <= 0 {
		width = 1024
	}
	if height <= 0 {
		height = 1024
	}
	return openAIChatGPTImageUploadRef{
		FileID:   fileID,
		FileName: fileName,
		FileSize: len(upload.Data),
		MimeType: contentType,
		Width:    width,
		Height:   height,
	}, nil
}

func (s *OpenAIGatewayService) downloadOpenAIChatGPTImageInputURL(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	imageURL string,
) (OpenAIImagesUpload, error) {
	imageURL = strings.TrimSpace(imageURL)
	if imageURL == "" {
		return OpenAIImagesUpload{}, &openAIChatGPTImageStageError{Stage: "upload", Message: "image URL is empty"}
	}
	if strings.HasPrefix(strings.ToLower(imageURL), "data:") {
		if upload, ok := openAIImageUploadFromDataURL(imageURL); ok {
			return upload, nil
		}
		return OpenAIImagesUpload{}, &openAIChatGPTImageStageError{Stage: "upload", Message: "invalid image data URL"}
	}
	headers := http.Header{}
	headers.Set("Accept", "image/*,*/*;q=0.8")
	headers.Set("User-Agent", openAIImageBackendUserAgent)
	if account != nil {
		if ua := strings.TrimSpace(account.GetOpenAIUserAgent()); ua != "" {
			headers.Set("User-Agent", ua)
		}
	}
	resp, _, err := s.openAIChatGPTImageDo(ctx, c, account, headers, http.MethodGet, imageURL, nil, "upload")
	if err != nil {
		return OpenAIImagesUpload{}, err
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(io.LimitReader(resp.Body, openAIImageMaxDownloadBytes))
	if err != nil {
		return OpenAIImagesUpload{}, &openAIChatGPTImageStageError{Stage: "upload", Message: err.Error(), Err: err, URL: imageURL}
	}
	contentType := strings.TrimSpace(resp.Header.Get("Content-Type"))
	if contentType == "" {
		contentType = http.DetectContentType(data)
	}
	return OpenAIImagesUpload{
		FileName:    openAIImageFileNameFromURL(imageURL, contentType),
		ContentType: contentType,
		Data:        data,
	}, nil
}

func openAIImageFileNameFromURL(imageURL string, contentType string) string {
	trimmed := strings.TrimSpace(imageURL)
	if idx := strings.IndexByte(trimmed, '?'); idx >= 0 {
		trimmed = trimmed[:idx]
	}
	if idx := strings.LastIndex(trimmed, "/"); idx >= 0 && idx+1 < len(trimmed) {
		name := strings.TrimSpace(trimmed[idx+1:])
		if name != "" && strings.Contains(name, ".") {
			return name
		}
	}
	switch strings.ToLower(strings.TrimSpace(contentType)) {
	case "image/jpeg", "image/jpg":
		return "image.jpg"
	case "image/webp":
		return "image.webp"
	default:
		return "image.png"
	}
}

func extractOpenAIResponsesImagePrompt(reqBody map[string]any) string {
	if reqBody == nil {
		return ""
	}
	if prompt := strings.TrimSpace(firstNonEmptyString(reqBody["prompt"])); prompt != "" {
		return prompt
	}
	return strings.TrimSpace(extractOpenAIResponsesImagePromptValue(reqBody["input"]))
}

func extractOpenAIResponsesImagePromptValue(value any) string {
	switch v := value.(type) {
	case string:
		return strings.TrimSpace(v)
	case []any:
		for i := len(v) - 1; i >= 0; i-- {
			if prompt := extractOpenAIResponsesImagePromptValue(v[i]); prompt != "" {
				return prompt
			}
		}
	case map[string]any:
		for _, key := range []string{"text", "content", "input_text"} {
			if prompt := strings.TrimSpace(firstNonEmptyString(v[key])); prompt != "" {
				return prompt
			}
		}
		if content, ok := v["content"]; ok {
			if prompt := extractOpenAIResponsesImagePromptValue(content); prompt != "" {
				return prompt
			}
		}
	}
	return ""
}

func extractOpenAIResponsesImageUploads(reqBody map[string]any) []OpenAIImagesUpload {
	if reqBody == nil {
		return nil
	}
	var uploads []OpenAIImagesUpload
	walkOpenAIResponsesImageUploads(reqBody["input"], &uploads)
	return uploads
}

func extractOpenAIResponsesImageInputURLs(reqBody map[string]any) []string {
	if reqBody == nil {
		return nil
	}
	var urls []string
	walkOpenAIResponsesImageInputURLs(reqBody["input"], &urls)
	return urls
}

func walkOpenAIResponsesImageInputURLs(value any, urls *[]string) {
	switch v := value.(type) {
	case []any:
		for _, item := range v {
			walkOpenAIResponsesImageInputURLs(item, urls)
		}
	case map[string]any:
		if strings.TrimSpace(firstNonEmptyString(v["type"])) == "input_image" {
			if imageURL := strings.TrimSpace(firstNonEmptyString(v["image_url"])); imageURL != "" && !strings.HasPrefix(strings.ToLower(imageURL), "data:") {
				*urls = append(*urls, imageURL)
			}
		}
		for _, child := range v {
			walkOpenAIResponsesImageInputURLs(child, urls)
		}
	}
}

func walkOpenAIResponsesImageUploads(value any, uploads *[]OpenAIImagesUpload) {
	switch v := value.(type) {
	case []any:
		for _, item := range v {
			walkOpenAIResponsesImageUploads(item, uploads)
		}
	case map[string]any:
		if strings.TrimSpace(firstNonEmptyString(v["type"])) == "input_image" {
			if imageURL := strings.TrimSpace(firstNonEmptyString(v["image_url"])); strings.HasPrefix(strings.ToLower(imageURL), "data:") {
				if upload, ok := openAIImageUploadFromDataURL(imageURL); ok {
					*uploads = append(*uploads, upload)
				}
			}
		}
		for _, child := range v {
			walkOpenAIResponsesImageUploads(child, uploads)
		}
	}
}

func openAIImageUploadFromDataURL(dataURL string) (OpenAIImagesUpload, bool) {
	if !strings.HasPrefix(strings.ToLower(strings.TrimSpace(dataURL)), "data:") {
		return OpenAIImagesUpload{}, false
	}
	parts := strings.SplitN(dataURL, ",", 2)
	if len(parts) != 2 {
		return OpenAIImagesUpload{}, false
	}
	meta := strings.TrimPrefix(parts[0], "data:")
	meta = strings.TrimSuffix(meta, ";base64")
	contentType := strings.TrimSpace(meta)
	if contentType == "" {
		contentType = "image/png"
	}
	decoded, err := base64.StdEncoding.DecodeString(parts[1])
	if err != nil || len(decoded) == 0 {
		return OpenAIImagesUpload{}, false
	}
	ext := "png"
	if slash := strings.LastIndex(contentType, "/"); slash >= 0 && slash+1 < len(contentType) {
		ext = strings.TrimSpace(contentType[slash+1:])
	}
	return OpenAIImagesUpload{FileName: "input." + ext, ContentType: contentType, Data: decoded}, true
}

func buildOpenAIResponsesImageConversationRequest(reqBody map[string]any, mainModel string, stream bool) openAIChatGPTImageConversationRequest {
	imageModel, _, _ := resolveOpenAIResponsesImageBillingConfig(reqBody, openAIChatGPTImageBillingModel)
	if strings.TrimSpace(imageModel) == "" || !isOpenAIImageBillingModelAlias(imageModel) {
		imageModel = openAIChatGPTImageBillingModel
	}
	tool := firstOpenAIResponsesImageGenerationTool(reqBody)
	outputFormat := strings.TrimSpace(firstNonEmptyString(tool["output_format"], reqBody["output_format"]))
	if outputFormat == "" {
		outputFormat = openAIChatGPTImageOutputFormat
	}
	return openAIChatGPTImageConversationRequest{
		Prompt:       extractOpenAIResponsesImagePrompt(reqBody),
		MainModel:    mainModel,
		ImageModel:   imageModel,
		Size:         strings.TrimSpace(firstNonEmptyString(tool["size"], reqBody["size"])),
		Quality:      strings.TrimSpace(firstNonEmptyString(tool["quality"], reqBody["quality"])),
		Background:   strings.TrimSpace(firstNonEmptyString(tool["background"], reqBody["background"])),
		OutputFormat: outputFormat,
		Uploads:      extractOpenAIResponsesImageUploads(reqBody),
		InputURLs:    extractOpenAIResponsesImageInputURLs(reqBody),
		Stream:       stream,
		ResponseKind: openAIResponsesEndpoint,
	}
}

func firstOpenAIResponsesImageGenerationTool(reqBody map[string]any) map[string]any {
	if reqBody == nil {
		return nil
	}
	tools, _ := reqBody["tools"].([]any)
	for _, rawTool := range tools {
		tool, ok := rawTool.(map[string]any)
		if ok && strings.TrimSpace(firstNonEmptyString(tool["type"])) == "image_generation" {
			return tool
		}
	}
	return nil
}

func (s *OpenAIGatewayService) writeOpenAIResponsesImageConversationResult(c *gin.Context, req openAIChatGPTImageConversationRequest, result *openAIChatGPTImageConversationResult) error {
	if c == nil || result == nil {
		return nil
	}
	responseID := "resp_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	createdAt := result.CreatedAt
	if createdAt <= 0 {
		createdAt = time.Now().Unix()
	}
	if req.Stream {
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")
		c.Header("X-Accel-Buffering", "no")
		if result.RequestID != "" {
			c.Header("x-request-id", result.RequestID)
		}
		flusher, ok := c.Writer.(http.Flusher)
		if !ok {
			return errors.New("streaming not supported")
		}
		created := buildOpenAIResponsesImageCreatedEvent(responseID, req.MainModel, createdAt, req)
		if _, err := c.Writer.Write([]byte("data: " + string(created) + "\n\n")); err != nil {
			return err
		}
		for i, img := range result.Images {
			item := buildOpenAIResponsesImageOutputItem(responseID, i, req.MainModel, img)
			if _, err := c.Writer.Write([]byte("data: " + string(item) + "\n\n")); err != nil {
				return err
			}
		}
		completed := buildOpenAIResponsesImageCompletedEvent(responseID, req.MainModel, createdAt, req, result)
		if _, err := c.Writer.Write([]byte("data: " + string(completed) + "\n\n")); err != nil {
			return err
		}
		if _, err := c.Writer.Write([]byte("data: [DONE]\n\n")); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}
	body := buildOpenAIResponsesImageResponse(responseID, req.MainModel, createdAt, req, result)
	c.Data(http.StatusOK, "application/json; charset=utf-8", body)
	return nil
}

func buildOpenAIResponsesImageResponse(responseID string, model string, createdAt int64, req openAIChatGPTImageConversationRequest, result *openAIChatGPTImageConversationResult) []byte {
	body := []byte(`{"id":"","object":"response","created_at":0,"model":"","status":"completed","output":[],"usage":{}}`)
	body, _ = sjson.SetBytes(body, "id", responseID)
	body, _ = sjson.SetBytes(body, "created_at", createdAt)
	body, _ = sjson.SetBytes(body, "model", model)
	for i, img := range result.Images {
		item := buildOpenAIResponsesImageOutputObject(i, img)
		body, _ = sjson.SetRawBytes(body, "output.-1", item)
	}
	body, _ = sjson.SetRawBytes(body, "usage", marshalOpenAIUsageJSON(result.Usage))
	body, _ = sjson.SetRawBytes(body, "tool_usage.image_gen", []byte(`{"images":`+strconv.Itoa(len(result.Images))+`}`))
	tool := buildOpenAIResponsesImageToolMeta(req)
	body, _ = sjson.SetRawBytes(body, "tools.0", tool)
	return body
}

func buildOpenAIResponsesImageCreatedEvent(responseID string, model string, createdAt int64, req openAIChatGPTImageConversationRequest) []byte {
	body := []byte(`{"type":"response.created","response":{"id":"","object":"response","created_at":0,"model":"","status":"in_progress","tools":[]}}`)
	body, _ = sjson.SetBytes(body, "response.id", responseID)
	body, _ = sjson.SetBytes(body, "response.created_at", createdAt)
	body, _ = sjson.SetBytes(body, "response.model", model)
	body, _ = sjson.SetRawBytes(body, "response.tools.0", buildOpenAIResponsesImageToolMeta(req))
	return body
}

func buildOpenAIResponsesImageOutputItem(responseID string, index int, model string, img openAIResponsesImageResult) []byte {
	body := []byte(`{"type":"response.output_item.done","output_index":0,"item":{}}`)
	body, _ = sjson.SetBytes(body, "output_index", index)
	body, _ = sjson.SetRawBytes(body, "item", buildOpenAIResponsesImageOutputObject(index, img))
	return body
}

func buildOpenAIResponsesImageCompletedEvent(responseID string, model string, createdAt int64, req openAIChatGPTImageConversationRequest, result *openAIChatGPTImageConversationResult) []byte {
	body := []byte(`{"type":"response.completed","response":{}}`)
	body, _ = sjson.SetRawBytes(body, "response", buildOpenAIResponsesImageResponse(responseID, model, createdAt, req, result))
	return body
}

func buildOpenAIResponsesImageOutputObject(index int, img openAIResponsesImageResult) []byte {
	item := []byte(`{"id":"","type":"image_generation_call","status":"completed","result":""}`)
	item, _ = sjson.SetBytes(item, "id", fmt.Sprintf("ig_%d", index+1))
	item, _ = sjson.SetBytes(item, "result", img.Result)
	if img.RevisedPrompt != "" {
		item, _ = sjson.SetBytes(item, "revised_prompt", img.RevisedPrompt)
	}
	if img.OutputFormat != "" {
		item, _ = sjson.SetBytes(item, "output_format", img.OutputFormat)
	}
	if img.Size != "" {
		item, _ = sjson.SetBytes(item, "size", img.Size)
	}
	if img.Quality != "" {
		item, _ = sjson.SetBytes(item, "quality", img.Quality)
	}
	return item
}

func buildOpenAIResponsesImageToolMeta(req openAIChatGPTImageConversationRequest) []byte {
	tool := []byte(`{"type":"image_generation","model":"gpt-image-2","output_format":"png"}`)
	if req.Size != "" {
		tool, _ = sjson.SetBytes(tool, "size", req.Size)
	}
	if req.Quality != "" {
		tool, _ = sjson.SetBytes(tool, "quality", req.Quality)
	}
	if req.Background != "" {
		tool, _ = sjson.SetBytes(tool, "background", req.Background)
	}
	if req.OutputFormat != "" {
		tool, _ = sjson.SetBytes(tool, "output_format", req.OutputFormat)
	}
	return tool
}

func marshalOpenAIUsageJSON(usage OpenAIUsage) []byte {
	body := []byte(`{"input_tokens":0,"output_tokens":0,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"image_tokens":0}}`)
	body, _ = sjson.SetBytes(body, "input_tokens", usage.InputTokens)
	body, _ = sjson.SetBytes(body, "output_tokens", usage.OutputTokens)
	body, _ = sjson.SetBytes(body, "input_tokens_details.cached_tokens", usage.CacheReadInputTokens)
	body, _ = sjson.SetBytes(body, "output_tokens_details.image_tokens", usage.ImageOutputTokens)
	return body
}

func (s *OpenAIGatewayService) writeOpenAIChatGPTImageError(c *gin.Context, err error) {
	if c == nil || c.Writer.Written() {
		return
	}
	var stageErr *openAIChatGPTImageStageError
	message := "ChatGPT image generation failed"
	if errors.As(err, &stageErr) {
		message = stageErr.Error()
	}
	setOpsUpstreamError(c, http.StatusBadGateway, message, "")
	c.JSON(http.StatusBadGateway, gin.H{
		"error": gin.H{
			"type":    "upstream_error",
			"message": message,
		},
	})
}

func buildOpenAIImagesConversationRequest(parsed *OpenAIImagesRequest, imageModel string) openAIChatGPTImageConversationRequest {
	outputFormat := strings.TrimSpace(parsed.OutputFormat)
	if outputFormat == "" {
		outputFormat = openAIChatGPTImageOutputFormat
	}
	uploads := make([]OpenAIImagesUpload, 0, len(parsed.Uploads)+len(parsed.InputImageURLs)+1)
	inputURLs := make([]string, 0, len(parsed.InputImageURLs)+1)
	uploads = append(uploads, parsed.Uploads...)
	for _, imageURL := range parsed.InputImageURLs {
		if upload, ok := openAIImageUploadFromDataURL(imageURL); ok {
			uploads = append(uploads, upload)
			continue
		}
		inputURLs = append(inputURLs, imageURL)
	}
	if parsed.MaskUpload != nil {
		uploads = append(uploads, *parsed.MaskUpload)
	} else if upload, ok := openAIImageUploadFromDataURL(parsed.MaskImageURL); ok {
		uploads = append(uploads, upload)
	} else if strings.TrimSpace(parsed.MaskImageURL) != "" {
		inputURLs = append(inputURLs, parsed.MaskImageURL)
	}
	return openAIChatGPTImageConversationRequest{
		Prompt:       parsed.Prompt,
		MainModel:    imageModel,
		ImageModel:   openAIChatGPTImageBillingModel,
		Size:         parsed.Size,
		Quality:      parsed.Quality,
		Background:   parsed.Background,
		OutputFormat: outputFormat,
		Uploads:      uploads,
		InputURLs:    inputURLs,
		Stream:       parsed.Stream,
		ResponseKind: parsed.Endpoint,
	}
}

func (s *OpenAIGatewayService) writeOpenAIImagesConversationResult(c *gin.Context, parsed *OpenAIImagesRequest, req openAIChatGPTImageConversationRequest, result *openAIChatGPTImageConversationResult, model string) error {
	if c == nil || result == nil {
		return nil
	}
	if parsed.Stream {
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")
		c.Header("X-Accel-Buffering", "no")
		if result.RequestID != "" {
			c.Header("x-request-id", result.RequestID)
		}
		flusher, ok := c.Writer.(http.Flusher)
		if !ok {
			return errors.New("streaming not supported")
		}
		prefix := openAIImagesStreamPrefix(parsed)
		for i, img := range result.Images {
			payload := buildOpenAIImagesStreamCompletedPayload(prefix+".completed", img, parsed.ResponseFormat, result.CreatedAt, marshalOpenAIUsageJSON(result.Usage))
			if _, err := c.Writer.Write([]byte("event: " + prefix + ".completed\n")); err != nil {
				return err
			}
			if _, err := c.Writer.Write([]byte("data: " + string(payload) + "\n\n")); err != nil {
				return err
			}
			if i == 0 {
				flusher.Flush()
			}
		}
		if _, err := c.Writer.Write([]byte("data: [DONE]\n\n")); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}
	body, err := buildOpenAIImagesAPIResponse(result.Images, result.CreatedAt, marshalOpenAIUsageJSON(result.Usage), openAIResponsesImageResult{
		OutputFormat: req.OutputFormat,
		Size:         req.Size,
		Background:   req.Background,
		Quality:      req.Quality,
		Model:        model,
	}, parsed.ResponseFormat)
	if err != nil {
		return err
	}
	c.Data(http.StatusOK, "application/json; charset=utf-8", body)
	return nil
}
