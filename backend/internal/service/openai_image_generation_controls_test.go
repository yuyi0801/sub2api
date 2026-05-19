package service

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestOpenAIGatewayServiceForward_RejectsDisabledImageGenerationIntents(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name string
		body []byte
	}{
		{
			name: "image model",
			body: []byte(`{"model":"gpt-image-2","input":"draw"}`),
		},
		{
			name: "image tool",
			body: []byte(`{"model":"gpt-5.4","input":"draw","tools":[{"type":"image_generation"}]}`),
		},
		{
			name: "image tool choice",
			body: []byte(`{"model":"gpt-5.4","input":"draw","tool_choice":{"type":"image_generation"}}`),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream := &httpUpstreamRecorder{}
			svc := newOpenAIImageGenerationControlTestService(upstream)
			c, recorder := newOpenAIImageGenerationControlTestContext(false, "unit-test-agent/1.0")
			account := newOpenAIImageGenerationControlTestAccount()

			result, err := svc.Forward(context.Background(), c, account, tt.body)

			require.Error(t, err)
			require.Nil(t, result)
			require.Equal(t, http.StatusForbidden, recorder.Code)
			require.Equal(t, "permission_error", gjson.GetBytes(recorder.Body.Bytes(), "error.type").String())
			require.Nil(t, upstream.lastReq, "disabled image request must not reach upstream")
		})
	}
}

func TestOpenAIGatewayServiceForward_DisabledGroupAllowsTextOnlyResponses(t *testing.T) {
	gin.SetMode(gin.TestMode)

	upstream := &httpUpstreamRecorder{
		resp: &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"id":"resp_text","model":"gpt-5.4","usage":{"input_tokens":3,"output_tokens":2}}`)),
		},
	}
	svc := newOpenAIImageGenerationControlTestService(upstream)
	c, recorder := newOpenAIImageGenerationControlTestContext(false, "unit-test-agent/1.0")
	account := newOpenAIImageGenerationControlTestAccount()

	result, err := svc.Forward(context.Background(), c, account, []byte(`{"model":"gpt-5.4","input":"write code","stream":false}`))

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, http.StatusOK, recorder.Code)
	require.Equal(t, 3, result.Usage.InputTokens)
	require.Equal(t, 2, result.Usage.OutputTokens)
	require.Equal(t, 0, result.ImageCount)
	require.NotNil(t, upstream.lastReq)
}

func TestOpenAIGatewayServiceForward_CodexImageInjectionRespectsGroupCapability(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name          string
		allowImages   bool
		bridgeEnabled bool
		wantInjected  bool
	}{
		{name: "disabled group skips injection", allowImages: false, bridgeEnabled: true, wantInjected: false},
		{name: "enabled group skips injection by default", allowImages: true, bridgeEnabled: false, wantInjected: false},
		{name: "enabled group injects image tool when bridge enabled", allowImages: true, bridgeEnabled: true, wantInjected: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream := &httpUpstreamRecorder{
				resp: &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"application/json"}},
					Body:       io.NopCloser(strings.NewReader(`{"id":"resp_codex","model":"gpt-5.4","usage":{"input_tokens":1,"output_tokens":1}}`)),
				},
			}
			svc := newOpenAIImageGenerationControlTestService(upstream)
			svc.cfg.Gateway.CodexImageGenerationBridgeEnabled = tt.bridgeEnabled
			c, _ := newOpenAIImageGenerationControlTestContext(tt.allowImages, "codex_cli_rs/0.98.0")
			account := newOpenAIImageGenerationControlTestAccount()

			result, err := svc.Forward(context.Background(), c, account, []byte(`{"model":"gpt-5.4","input":"write code","stream":false}`))

			require.NoError(t, err)
			require.NotNil(t, result)
			require.NotNil(t, upstream.lastReq)
			hasImageTool := gjson.GetBytes(upstream.lastBody, `tools.#(type=="image_generation")`).Exists()
			require.Equal(t, tt.wantInjected, hasImageTool)
			if tt.wantInjected {
				require.Equal(t, openAIResponsesImageGenerationToolModel, gjson.GetBytes(upstream.lastBody, `tools.#(type=="image_generation").model`).String())
				require.Equal(t, "png", gjson.GetBytes(upstream.lastBody, `tools.#(type=="image_generation").output_format`).String())
			}
			instructions := gjson.GetBytes(upstream.lastBody, "instructions").String()
			require.Equal(t, tt.wantInjected, strings.Contains(instructions, "image_generation"))
		})
	}
}

func TestOpenAIGatewayServiceForward_ExplicitImageToolWorksWithBridgeDisabled(t *testing.T) {
	gin.SetMode(gin.TestMode)

	upstream := &httpUpstreamRecorder{
		resp: &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"id":"resp_explicit_image","model":"gpt-5.4","usage":{"input_tokens":2,"output_tokens":1}}`)),
		},
	}
	svc := newOpenAIImageGenerationControlTestService(upstream)
	c, _ := newOpenAIImageGenerationControlTestContext(true, "codex_cli_rs/0.98.0")
	account := newOpenAIImageGenerationControlTestAccount()
	body := []byte(`{"model":"gpt-5.4","input":"draw","stream":false,"tools":[{"type":"image_generation","model":"gpt-image-1.5","format":"jpeg"}],"tool_choice":{"type":"image_generation"}}`)

	result, err := svc.Forward(context.Background(), c, account, body)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, upstream.lastReq)
	require.True(t, gjson.GetBytes(upstream.lastBody, `tools.#(type=="image_generation")`).Exists())
	require.Equal(t, openAIResponsesImageGenerationToolModel, gjson.GetBytes(upstream.lastBody, `tools.#(type=="image_generation").model`).String())
	require.Equal(t, "jpeg", gjson.GetBytes(upstream.lastBody, `tools.#(type=="image_generation").output_format`).String())
	require.False(t, gjson.GetBytes(upstream.lastBody, `tools.#(type=="image_generation").format`).Exists())
	require.Equal(t, "image_generation", gjson.GetBytes(upstream.lastBody, "tool_choice.type").String())
	instructions := gjson.GetBytes(upstream.lastBody, "instructions").String()
	require.NotContains(t, instructions, "image_generation")
}

func TestOpenAIGatewayServiceForward_CodexImageToolUsesImage2AndPreservesMainModel(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []string{"gpt-5.4", "gpt-5.4-mini", "gpt-5.5"}
	for _, model := range tests {
		t.Run(model, func(t *testing.T) {
			upstream := &httpUpstreamRecorder{resp: &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"req_sidecar"}},
				Body:       io.NopCloser(strings.NewReader(`{"created":1710000000,"data":[{"b64_json":"aGVsbG8=","revised_prompt":"draw"}],"usage":{"input_tokens":3,"output_tokens":4,"output_tokens_details":{"image_tokens":2}}}`)),
			}}
			svc := newOpenAIImageGenerationControlTestService(upstream)
			svc.cfg.Gateway.OpenAIImageSidecar.Enabled = true
			svc.cfg.Gateway.OpenAIImageSidecar.BaseURL = "http://chatgpt2api-image:80"
			svc.cfg.Gateway.OpenAIImageSidecar.APIKey = "sidecar-key"
			svc.cfg.Gateway.OpenAIImageSidecar.Model = "gpt-image-2"
			c, recorder := newOpenAIImageGenerationControlTestContext(true, "codex_cli_rs/0.98.0")
			account := newOpenAIImageGenerationControlTestOAuthAccount()
			body := []byte(`{"model":"` + model + `","input":"draw","stream":false,"tool_choice":{"type":"image_generation"}}`)

			result, err := svc.Forward(context.Background(), c, account, body)

			require.NoError(t, err)
			require.NotNil(t, result)
			require.NotNil(t, upstream.lastReq)
			require.Equal(t, "http://chatgpt2api-image:80/v1/images/generations", upstream.lastReq.URL.String())
			require.Equal(t, "Bearer sidecar-key", upstream.lastReq.Header.Get("Authorization"))
			require.Equal(t, "sub2api", upstream.lastReq.Header.Get("X-Caller"))
			require.Equal(t, "openai-image-sidecar", upstream.lastReq.Header.Get("X-Request-Source"))
			require.Equal(t, "gpt-image-2", gjson.GetBytes(upstream.lastBody, "model").String())
			require.Equal(t, "draw", gjson.GetBytes(upstream.lastBody, "prompt").String())
			require.Equal(t, model, gjson.Get(recorder.Body.String(), "model").String())
			require.Equal(t, "image_generation_call", gjson.Get(recorder.Body.String(), "output.0.type").String())
			require.Equal(t, "aGVsbG8=", gjson.Get(recorder.Body.String(), "output.0.result").String())
			require.Equal(t, "gpt-image-2", result.BillingModel)
			require.Equal(t, 1, result.ImageCount)
			require.Zero(t, result.Usage.InputTokens)
			require.Zero(t, result.Usage.OutputTokens)
			require.Equal(t, int64(3), gjson.Get(recorder.Body.String(), "usage.input_tokens").Int())
			require.Equal(t, int64(4), gjson.Get(recorder.Body.String(), "usage.output_tokens").Int())
		})
	}
}

func TestOpenAIGatewayServiceForwardResponsesImageSidecar_BypassesUpstreamAccount(t *testing.T) {
	gin.SetMode(gin.TestMode)

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"req_direct_sidecar"}},
		Body:       io.NopCloser(strings.NewReader(`{"created":1710000000,"data":[{"b64_json":"ZGlyZWN0","revised_prompt":"draw"}]}`)),
	}}
	svc := newOpenAIImageGenerationControlTestService(upstream)
	svc.cfg.Gateway.OpenAIImageSidecar.Enabled = true
	svc.cfg.Gateway.OpenAIImageSidecar.BaseURL = "http://chatgpt2api-image:80"
	svc.cfg.Gateway.OpenAIImageSidecar.APIKey = "sidecar-key"
	svc.cfg.Gateway.OpenAIImageSidecar.Model = "gpt-image-2"
	c, recorder := newOpenAIImageGenerationControlTestContext(true, "codex_cli_rs/0.98.0")
	body := []byte(`{"model":"gpt-5.5","input":"draw","stream":false,"tool_choice":{"type":"image_generation"}}`)

	result, err := svc.ForwardResponsesImageSidecar(context.Background(), c, body, "gpt-5.5", "gpt-5.5", false, time.Now())

	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, upstream.lastReq)
	require.Equal(t, "http://chatgpt2api-image:80/v1/images/generations", upstream.lastReq.URL.String())
	require.Equal(t, "Bearer sidecar-key", upstream.lastReq.Header.Get("Authorization"))
	require.Equal(t, "sub2api", upstream.lastReq.Header.Get("X-Caller"))
	require.Equal(t, "openai-image-sidecar", upstream.lastReq.Header.Get("X-Request-Source"))
	require.Equal(t, "gpt-image-2", gjson.GetBytes(upstream.lastBody, "model").String())
	require.Equal(t, "draw", gjson.GetBytes(upstream.lastBody, "prompt").String())
	require.Equal(t, "gpt-5.5", gjson.Get(recorder.Body.String(), "model").String())
	require.Equal(t, "image_generation_call", gjson.Get(recorder.Body.String(), "output.0.type").String())
	require.Equal(t, "ZGlyZWN0", gjson.Get(recorder.Body.String(), "output.0.result").String())
	require.Equal(t, "gpt-image-2", result.BillingModel)
	require.Equal(t, 1, result.ImageCount)
	require.Equal(t, "gpt-5.5", result.Model)
}

func TestOpenAIGatewayServiceForward_CodexImageSidecarStreamsResponsesEvents(t *testing.T) {
	gin.SetMode(gin.TestMode)

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"req_sidecar_stream"}},
		Body:       io.NopCloser(strings.NewReader(`{"created":1710000000,"data":[{"b64_json":"c3RyZWFtLWltYWdl","revised_prompt":"draw"}]}`)),
	}}
	svc := newOpenAIImageGenerationControlTestService(upstream)
	svc.cfg.Gateway.OpenAIImageSidecar.Enabled = true
	svc.cfg.Gateway.OpenAIImageSidecar.BaseURL = "http://chatgpt2api-image:80"
	svc.cfg.Gateway.OpenAIImageSidecar.APIKey = "sidecar-key"
	svc.cfg.Gateway.OpenAIImageSidecar.Model = "gpt-image-2"
	c, recorder := newOpenAIImageGenerationControlTestContext(true, "codex_cli_rs/0.98.0")
	account := newOpenAIImageGenerationControlTestOAuthAccount()
	body := []byte(`{"model":"gpt-5.5","input":"draw","stream":true,"tools":[{"type":"image_generation"}],"tool_choice":{"type":"image_generation"}}`)

	result, err := svc.Forward(context.Background(), c, account, body)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.True(t, result.Stream)
	require.Equal(t, "gpt-image-2", result.BillingModel)
	require.Equal(t, 1, result.ImageCount)
	require.Equal(t, "sub2api", upstream.lastReq.Header.Get("X-Caller"))
	require.Equal(t, "openai-image-sidecar", upstream.lastReq.Header.Get("X-Request-Source"))
	require.Equal(t, "text/event-stream", recorder.Header().Get("Content-Type"))
	require.Contains(t, recorder.Body.String(), `"type":"response.output_item.done"`)
	require.Contains(t, recorder.Body.String(), `"type":"response.completed"`)
	require.Contains(t, recorder.Body.String(), `"type":"image_generation_call"`)
	require.Contains(t, recorder.Body.String(), `"result":"c3RyZWFtLWltYWdl"`)
	require.Contains(t, recorder.Body.String(), "data: [DONE]")
}

func TestOpenAIGatewayServiceForward_CodexTextRequestDoesNotForceImageTool(t *testing.T) {
	gin.SetMode(gin.TestMode)

	upstream := &httpUpstreamRecorder{resp: openAICompatSSECompletedResponse("resp_text_only", "gpt-5.4")}
	svc := newOpenAIImageGenerationControlTestService(upstream)
	c, _ := newOpenAIImageGenerationControlTestContext(true, "codex_cli_rs/0.98.0")
	account := newOpenAIImageGenerationControlTestOAuthAccount()
	body := []byte(`{"model":"gpt-5.4","input":"write code","stream":false}`)

	result, err := svc.Forward(context.Background(), c, account, body)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, upstream.lastReq)
	require.False(t, gjson.GetBytes(upstream.lastBody, `tools.#(type=="image_generation")`).Exists())
}

func TestOpenAIGatewayServiceForward_CodexBridgeTextDoesNotRouteToSidecar(t *testing.T) {
	gin.SetMode(gin.TestMode)

	upstream := &httpUpstreamRecorder{resp: openAICompatSSECompletedResponse("resp_bridge_text_only", "gpt-5.4")}
	svc := newOpenAIImageGenerationControlTestService(upstream)
	svc.cfg.Gateway.CodexImageGenerationBridgeEnabled = true
	svc.cfg.Gateway.OpenAIImageSidecar.Enabled = true
	svc.cfg.Gateway.OpenAIImageSidecar.BaseURL = "http://chatgpt2api-image:80"
	svc.cfg.Gateway.OpenAIImageSidecar.APIKey = "sidecar-key"
	svc.cfg.Gateway.OpenAIImageSidecar.Model = "gpt-image-2"
	c, _ := newOpenAIImageGenerationControlTestContext(true, "codex_cli_rs/0.98.0")
	account := newOpenAIImageGenerationControlTestOAuthAccount()
	body := []byte(`{"model":"gpt-5.4","input":"write code","stream":false}`)

	result, err := svc.Forward(context.Background(), c, account, body)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, upstream.lastReq)
	require.Equal(t, chatgptCodexURL, upstream.lastReq.URL.String())
	require.True(t, gjson.GetBytes(upstream.lastBody, `tools.#(type=="image_generation")`).Exists())
	require.Equal(t, 0, result.ImageCount)
}

func TestOpenAIGatewayServiceForward_CodexSparkDoesNotInjectImage2(t *testing.T) {
	gin.SetMode(gin.TestMode)

	upstream := &httpUpstreamRecorder{resp: openAICompatSSECompletedResponse("resp_spark", "gpt-5.3-codex-spark")}
	svc := newOpenAIImageGenerationControlTestService(upstream)
	svc.cfg.Gateway.CodexImageGenerationBridgeEnabled = true
	c, _ := newOpenAIImageGenerationControlTestContext(true, "codex_cli_rs/0.98.0")
	account := newOpenAIImageGenerationControlTestOAuthAccount()
	body := []byte(`{"model":"gpt-5.3-codex-spark","input":"write code","stream":false}`)

	result, err := svc.Forward(context.Background(), c, account, body)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, upstream.lastReq)
	require.False(t, gjson.GetBytes(upstream.lastBody, `tools.#(type=="image_generation")`).Exists())
	require.Contains(t, gjson.GetBytes(upstream.lastBody, "instructions").String(), codexSparkImageUnsupportedMarker)
}

func TestOpenAIGatewayServiceForward_ChannelBridgeOverrideEnablesCodexInjection(t *testing.T) {
	gin.SetMode(gin.TestMode)

	upstream := &httpUpstreamRecorder{
		resp: &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"id":"resp_channel_bridge","model":"gpt-5.4","usage":{"input_tokens":1,"output_tokens":1}}`)),
		},
	}
	svc := newOpenAIImageGenerationControlTestService(upstream)
	groupID := int64(4242)
	svc.channelService = newOpenAIImageGenerationControlChannelService(groupID, &Channel{
		ID:     9001,
		Status: StatusActive,
		FeaturesConfig: map[string]any{
			featureKeyCodexImageGenerationBridge: map[string]any{PlatformOpenAI: true},
		},
	})
	c, _ := newOpenAIImageGenerationControlTestContext(true, "codex_cli_rs/0.98.0")
	account := newOpenAIImageGenerationControlTestAccount()

	result, err := svc.Forward(context.Background(), c, account, []byte(`{"model":"gpt-5.4","input":"write code","stream":false}`))

	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, upstream.lastReq)
	require.True(t, gjson.GetBytes(upstream.lastBody, `tools.#(type=="image_generation")`).Exists())
	instructions := gjson.GetBytes(upstream.lastBody, "instructions").String()
	require.Contains(t, instructions, "image_generation")
}

func TestOpenAIGatewayService_CodexImageGenerationBridgeOverridePrecedence(t *testing.T) {
	groupID := int64(4242)

	tests := []struct {
		name    string
		global  bool
		channel *Channel
		account *Account
		want    bool
	}{
		{
			name:   "global default enables bridge",
			global: true,
			account: &Account{
				Platform: PlatformOpenAI,
			},
			want: true,
		},
		{
			name:   "channel true overrides disabled global",
			global: false,
			channel: &Channel{ID: 1, Status: StatusActive, FeaturesConfig: map[string]any{
				featureKeyCodexImageGenerationBridge: map[string]any{PlatformOpenAI: true},
			}},
			account: &Account{Platform: PlatformOpenAI},
			want:    true,
		},
		{
			name:   "channel false overrides enabled global",
			global: true,
			channel: &Channel{ID: 1, Status: StatusActive, FeaturesConfig: map[string]any{
				featureKeyCodexImageGenerationBridge: map[string]any{PlatformOpenAI: false},
			}},
			account: &Account{Platform: PlatformOpenAI},
			want:    false,
		},
		{
			name:   "account false overrides channel and global true",
			global: true,
			channel: &Channel{ID: 1, Status: StatusActive, FeaturesConfig: map[string]any{
				featureKeyCodexImageGenerationBridge: map[string]any{PlatformOpenAI: true},
			}},
			account: &Account{
				Platform: PlatformOpenAI,
				Extra:    map[string]any{featureKeyCodexImageGenerationBridge: false},
			},
			want: false,
		},
		{
			name:   "nested account true overrides channel false",
			global: false,
			channel: &Channel{ID: 1, Status: StatusActive, FeaturesConfig: map[string]any{
				featureKeyCodexImageGenerationBridge: map[string]any{PlatformOpenAI: false},
			}},
			account: &Account{
				Platform: PlatformOpenAI,
				Extra: map[string]any{
					PlatformOpenAI: map[string]any{"codex_image_generation_bridge_enabled": true},
				},
			},
			want: true,
		},
		{
			name:   "non openai account extra is ignored",
			global: false,
			account: &Account{
				Platform: PlatformAnthropic,
				Extra:    map[string]any{featureKeyCodexImageGenerationBridge: true},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := newOpenAIImageGenerationControlTestService(&httpUpstreamRecorder{})
			svc.cfg.Gateway.CodexImageGenerationBridgeEnabled = tt.global
			if tt.channel != nil {
				svc.channelService = newOpenAIImageGenerationControlChannelService(groupID, tt.channel)
			}
			apiKey := &APIKey{GroupID: &groupID}

			got := svc.isCodexImageGenerationBridgeEnabled(context.Background(), tt.account, apiKey)

			require.Equal(t, tt.want, got)
		})
	}
}

func TestOpenAIGatewayServiceHandleResponsesImageOutputs_NonStreaming(t *testing.T) {
	gin.SetMode(gin.TestMode)

	svc := newOpenAIImageGenerationControlTestService(&httpUpstreamRecorder{})
	c, _ := newOpenAIImageGenerationControlTestContext(true, "unit-test-agent/1.0")
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(`{
			"id":"resp_image_json",
			"model":"gpt-5.4",
			"output":[{"id":"ig_json_1","type":"image_generation_call","result":"final-image"}],
			"usage":{"input_tokens":7,"output_tokens":3,"output_tokens_details":{"image_tokens":2}}
		}`)),
	}

	result, err := svc.handleNonStreamingResponse(context.Background(), resp, c, &Account{ID: 1, Type: AccountTypeAPIKey}, "gpt-5.4", "gpt-5.4")

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, 1, result.imageCount)
	require.NotNil(t, result.usage)
	require.Equal(t, 7, result.usage.InputTokens)
	require.Equal(t, 3, result.usage.OutputTokens)
	require.Equal(t, 2, result.usage.ImageOutputTokens)
}

func TestOpenAIGatewayServiceHandleResponsesImageOutputs_Streaming(t *testing.T) {
	gin.SetMode(gin.TestMode)

	svc := newOpenAIImageGenerationControlTestService(&httpUpstreamRecorder{})
	c, _ := newOpenAIImageGenerationControlTestContext(true, "unit-test-agent/1.0")
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body: io.NopCloser(strings.NewReader(
			"data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"ig_stream_1\",\"type\":\"image_generation_call\",\"result\":\"final-image\"}}\n\n" +
				"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_image_stream\",\"model\":\"gpt-5.5\",\"output\":[{\"id\":\"ig_stream_1\",\"type\":\"image_generation_call\",\"result\":\"final-image\"}],\"usage\":{\"input_tokens\":11,\"output_tokens\":5,\"output_tokens_details\":{\"image_tokens\":4}}}}\n\n",
		)),
	}

	result, err := svc.handleStreamingResponse(context.Background(), resp, c, &Account{ID: 1}, time.Now(), "gpt-5.5", "gpt-5.5")

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, 1, result.imageCount)
	require.NotNil(t, result.usage)
	require.Equal(t, 11, result.usage.InputTokens)
	require.Equal(t, 5, result.usage.OutputTokens)
	require.Equal(t, 4, result.usage.ImageOutputTokens)
}

func newOpenAIImageGenerationControlTestService(upstream *httpUpstreamRecorder) *OpenAIGatewayService {
	cfg := &config.Config{}
	return &OpenAIGatewayService{
		cfg:              cfg,
		httpUpstream:     upstream,
		cache:            &stubGatewayCache{},
		openaiWSResolver: NewOpenAIWSProtocolResolver(cfg),
		toolCorrector:    NewCodexToolCorrector(),
	}
}

func newOpenAIImageGenerationControlChannelService(groupID int64, ch *Channel) *ChannelService {
	svc := &ChannelService{}
	cache := newEmptyChannelCache()
	if ch != nil {
		cache.channelByGroupID[groupID] = ch
		cache.byID[ch.ID] = ch
	}
	cache.loadedAt = time.Now()
	svc.cache.Store(cache)
	return svc
}

func newOpenAIImageGenerationControlTestContext(allowImages bool, userAgent string) (*gin.Context, *httptest.ResponseRecorder) {
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/openai/v1/responses", nil)
	c.Request.Header.Set("User-Agent", userAgent)
	groupID := int64(4242)
	c.Set("api_key", &APIKey{
		ID:      2424,
		GroupID: &groupID,
		Group: &Group{
			ID:                   groupID,
			AllowImageGeneration: allowImages,
			RateMultiplier:       1,
			ImageRateMultiplier:  1,
		},
	})
	return c, recorder
}

func newOpenAIImageGenerationControlTestAccount() *Account {
	return &Account{
		ID:          5151,
		Name:        "openai-image-controls",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Status:      StatusActive,
		Schedulable: true,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key": "sk-test",
		},
	}
}

func newOpenAIImageGenerationControlTestOAuthAccount() *Account {
	return &Account{
		ID:          6161,
		Name:        "openai-image-controls-oauth",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeOAuth,
		Status:      StatusActive,
		Schedulable: true,
		Concurrency: 1,
		Credentials: map[string]any{
			"access_token":       "oauth-token",
			"chatgpt_account_id": "chatgpt-acc",
		},
	}
}
