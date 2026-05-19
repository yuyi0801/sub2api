package handler

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/pagination"
	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	middleware2 "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestOpenAIGatewayHandlerImages_DisabledGroupRejectsBeforeScheduling(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"gpt-image-2","prompt":"draw","size":"1024x1024"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/images/generations", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = req
	groupID := int64(111)
	c.Set(string(middleware2.ContextKeyAPIKey), &service.APIKey{
		ID:      222,
		GroupID: &groupID,
		Group: &service.Group{
			ID:                   groupID,
			AllowImageGeneration: false,
		},
		User: &service.User{ID: 333},
	})
	c.Set(string(middleware2.ContextKeyUser), middleware2.AuthSubject{UserID: 333, Concurrency: 1})

	h := &OpenAIGatewayHandler{
		gatewayService:      &service.OpenAIGatewayService{},
		billingCacheService: &service.BillingCacheService{},
		apiKeyService:       &service.APIKeyService{},
		concurrencyHelper:   &ConcurrencyHelper{concurrencyService: &service.ConcurrencyService{}},
	}

	h.Images(c)

	require.Equal(t, http.StatusForbidden, rec.Code)
	require.Equal(t, "permission_error", gjson.GetBytes(rec.Body.Bytes(), "error.type").String())
	require.Contains(t, rec.Body.String(), service.ImageGenerationPermissionMessage())
}

func TestOpenAIGatewayHandlerImages_SidecarBypassesOpenAIAccountSelection(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"gpt-image-2","prompt":"draw","size":"1024x1024"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/images/generations", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = req
	groupID := int64(111)
	user := &service.User{ID: 333, Balance: 10, Concurrency: 1}
	c.Set(string(middleware2.ContextKeyAPIKey), &service.APIKey{
		ID:      222,
		Quota:   10,
		GroupID: &groupID,
		Group: &service.Group{
			ID:                   groupID,
			AllowImageGeneration: true,
			RateMultiplier:       1,
			ImageRateMultiplier:  1,
		},
		User: user,
	})
	c.Set(string(middleware2.ContextKeyUser), middleware2.AuthSubject{UserID: 333, Concurrency: 1})

	upstream := &handlerImagesHTTPUpstream{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"req_handler_sidecar"}},
		Body:       io.NopCloser(bytes.NewReader([]byte(`{"created":1710000000,"data":[{"b64_json":"aW1n"}]}`))),
	}}
	userRepo := &handlerImagesUserRepo{user: user}
	usageRepo := &handlerImagesUsageLogRepo{}
	billingRepo := &handlerImagesBillingRepo{}
	apiKeyRepo := &handlerImagesAPIKeyRepo{}
	cfg := &config.Config{RunMode: config.RunModeStandard}
	cfg.Gateway.OpenAIImageSidecar.Enabled = true
	cfg.Gateway.OpenAIImageSidecar.BaseURL = "http://chatgpt2api-image:80"
	cfg.Gateway.OpenAIImageSidecar.APIKey = "sidecar-key"
	cfg.Gateway.OpenAIImageSidecar.Model = "gpt-image-2"
	cfg.Security.URLAllowlist.AllowInsecureHTTP = true
	gatewayBillingCache := service.NewBillingCacheService(nil, userRepo, nil, nil, nil, nil, cfg)
	defer gatewayBillingCache.Stop()
	handlerBillingCache := service.NewBillingCacheService(nil, userRepo, nil, nil, nil, nil, cfg)
	defer handlerBillingCache.Stop()
	gatewaySvc := service.NewOpenAIGatewayService(
		&handlerImagesAccountRepo{},
		usageRepo,
		billingRepo,
		userRepo,
		nil,
		nil,
		nil,
		cfg,
		nil,
		nil,
		service.NewBillingService(cfg, nil),
		nil,
		gatewayBillingCache,
		upstream,
		&service.DeferredService{},
		nil,
		nil,
		nil,
		nil,
		nil,
	)

	h := &OpenAIGatewayHandler{
		gatewayService:        gatewaySvc,
		billingCacheService:   handlerBillingCache,
		apiKeyService:         service.NewAPIKeyService(apiKeyRepo, userRepo, nil, nil, nil, nil, cfg),
		concurrencyHelper:     &ConcurrencyHelper{concurrencyService: service.NewConcurrencyService(&helperConcurrencyCacheStub{userSeq: []bool{true}})},
		usageRecordWorkerPool: service.NewUsageRecordWorkerPoolWithOptions(service.UsageRecordWorkerPoolOptions{WorkerCount: 1, QueueSize: 1}),
		cfg:                   cfg,
		imageLimiter:          &imageConcurrencyLimiter{},
	}
	h.usageRecordWorkerPool.Stop()
	defer h.usageRecordWorkerPool.Stop()

	h.Images(c)

	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, 1, upstream.calls)
	require.Equal(t, "http://chatgpt2api-image:80/v1/images/generations", upstream.lastReq.URL.String())
	require.Equal(t, "sub2api", upstream.lastReq.Header.Get("X-Caller"))
	require.Equal(t, "openai-image-sidecar", upstream.lastReq.Header.Get("X-Request-Source"))
	require.Equal(t, "gpt-image-2", gjson.GetBytes(upstream.lastBody, "model").String())
	require.Equal(t, "aW1n", gjson.GetBytes(rec.Body.Bytes(), "data.0.b64_json").String())
	require.Eventually(t, func() bool { return usageRepo.lastLog != nil }, time.Second, 10*time.Millisecond)
	require.NotNil(t, usageRepo.lastLog)
	require.Equal(t, int64(0), usageRepo.lastLog.AccountID)
	require.Equal(t, "gpt-image-2", usageRepo.lastLog.Model)
	require.Equal(t, 1, usageRepo.lastLog.ImageCount)
	require.NotNil(t, usageRepo.lastLog.BillingMode)
	require.Equal(t, string(service.BillingModeImage), *usageRepo.lastLog.BillingMode)
	require.NotNil(t, usageRepo.lastLog.UpstreamEndpoint)
	require.Equal(t, "openai-image-sidecar:/v1/images/generations", *usageRepo.lastLog.UpstreamEndpoint)
	require.Eventually(t, func() bool { return billingRepo.lastCmd != nil }, time.Second, 10*time.Millisecond)
	require.NotNil(t, billingRepo.lastCmd)
	require.Equal(t, int64(0), billingRepo.lastCmd.AccountID)
	require.Equal(t, 1, billingRepo.lastCmd.ImageCount)
}

type handlerImagesHTTPUpstream struct {
	resp     *http.Response
	lastReq  *http.Request
	lastBody []byte
	calls    int
}

func (u *handlerImagesHTTPUpstream) Do(req *http.Request, proxyURL string, accountID int64, accountConcurrency int) (*http.Response, error) {
	u.calls++
	u.lastReq = req
	if req != nil && req.Body != nil {
		body, _ := io.ReadAll(req.Body)
		u.lastBody = body
		_ = req.Body.Close()
		req.Body = io.NopCloser(bytes.NewReader(body))
	}
	return u.resp, nil
}

func (u *handlerImagesHTTPUpstream) DoWithTLS(req *http.Request, proxyURL string, accountID int64, accountConcurrency int, profile *tlsfingerprint.Profile) (*http.Response, error) {
	return u.Do(req, proxyURL, accountID, accountConcurrency)
}

type handlerImagesUserRepo struct {
	service.UserRepository
	user *service.User
}

func (r *handlerImagesUserRepo) GetByID(ctx context.Context, id int64) (*service.User, error) {
	return r.user, nil
}

func (r *handlerImagesUserRepo) DeductBalance(ctx context.Context, id int64, amount float64) error {
	if r.user != nil {
		r.user.Balance -= amount
	}
	return nil
}

type handlerImagesUsageLogRepo struct {
	service.UsageLogRepository
	lastLog *service.UsageLog
}

func (r *handlerImagesUsageLogRepo) Create(ctx context.Context, log *service.UsageLog) (bool, error) {
	r.lastLog = log
	return true, nil
}

type handlerImagesBillingRepo struct {
	service.UsageBillingRepository
	lastCmd *service.UsageBillingCommand
}

func (r *handlerImagesBillingRepo) Apply(ctx context.Context, cmd *service.UsageBillingCommand) (*service.UsageBillingApplyResult, error) {
	r.lastCmd = cmd
	return &service.UsageBillingApplyResult{Applied: true}, nil
}

type handlerImagesAPIKeyRepo struct {
	service.APIKeyRepository
}

func (r *handlerImagesAPIKeyRepo) IncrementQuotaUsed(ctx context.Context, id int64, amount float64) (float64, error) {
	return amount, nil
}

func (r *handlerImagesAPIKeyRepo) GetByID(ctx context.Context, id int64) (*service.APIKey, error) {
	return &service.APIKey{ID: id, Key: "sk-test", Quota: 10}, nil
}

func (r *handlerImagesAPIKeyRepo) IncrementRateLimitUsage(ctx context.Context, id int64, cost float64) error {
	return nil
}

type handlerImagesAccountRepo struct {
	service.AccountRepository
}

func (r *handlerImagesAccountRepo) ListSchedulableByGroupIDAndPlatform(ctx context.Context, groupID int64, platform string) ([]service.Account, error) {
	return nil, service.ErrNoAvailableAccounts
}

func (r *handlerImagesAccountRepo) ListSchedulableUngroupedByPlatform(ctx context.Context, platform string) ([]service.Account, error) {
	return nil, service.ErrNoAvailableAccounts
}

func (r *handlerImagesAccountRepo) ListSchedulableByPlatform(ctx context.Context, platform string) ([]service.Account, error) {
	return nil, service.ErrNoAvailableAccounts
}

func (r *handlerImagesAccountRepo) GetByID(ctx context.Context, id int64) (*service.Account, error) {
	return nil, service.ErrNoAvailableAccounts
}

func (r *handlerImagesAccountRepo) List(ctx context.Context, params pagination.PaginationParams) ([]service.Account, *pagination.PaginationResult, error) {
	return nil, nil, service.ErrNoAvailableAccounts
}

func (r *handlerImagesAccountRepo) ListWithFilters(ctx context.Context, params pagination.PaginationParams, platform, accountType, status, search string, groupID int64, privacyMode string) ([]service.Account, *pagination.PaginationResult, error) {
	return nil, nil, service.ErrNoAvailableAccounts
}

func (r *handlerImagesAccountRepo) ListByGroup(ctx context.Context, groupID int64) ([]service.Account, error) {
	return nil, service.ErrNoAvailableAccounts
}

func (r *handlerImagesAccountRepo) ListActive(ctx context.Context) ([]service.Account, error) {
	return nil, service.ErrNoAvailableAccounts
}

func (r *handlerImagesAccountRepo) ListByPlatform(ctx context.Context, platform string) ([]service.Account, error) {
	return nil, service.ErrNoAvailableAccounts
}

func (r *handlerImagesAccountRepo) UpdateLastUsed(ctx context.Context, id int64) error {
	return nil
}

func (r *handlerImagesAccountRepo) BatchUpdateLastUsed(ctx context.Context, updates map[int64]time.Time) error {
	return nil
}
