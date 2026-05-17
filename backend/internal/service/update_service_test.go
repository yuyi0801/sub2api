package service

import (
	"context"
	"encoding/json"
	"testing"
	"time"
)

type updateServiceReleaseClientStub struct {
	repo string
}

func (s *updateServiceReleaseClientStub) FetchLatestRelease(_ context.Context, repo string) (*GitHubRelease, error) {
	s.repo = repo
	return &GitHubRelease{TagName: "v0.1.128"}, nil
}

func (s *updateServiceReleaseClientStub) DownloadFile(context.Context, string, string, int64) error {
	return nil
}

func (s *updateServiceReleaseClientStub) FetchChecksumFile(context.Context, string) ([]byte, error) {
	return nil, nil
}

func TestUpdateServiceUsesConfiguredGitHubRepo(t *testing.T) {
	client := &updateServiceReleaseClientStub{}
	svc := NewUpdateService(nil, client, "0.1.127-image2-split.3", "release", "yuyi0801/sub2api")

	info, err := svc.fetchLatestRelease(context.Background())
	if err != nil {
		t.Fatalf("fetchLatestRelease() error: %v", err)
	}
	if client.repo != "yuyi0801/sub2api" {
		t.Fatalf("FetchLatestRelease repo = %q, want %q", client.repo, "yuyi0801/sub2api")
	}
	if !info.HasUpdate {
		t.Fatalf("HasUpdate = false, want true")
	}
}

func TestCompareVersionsIgnoresPreReleaseSuffix(t *testing.T) {
	if got := compareVersions("0.1.127-image2-split.3", "0.1.127"); got != 0 {
		t.Fatalf("compareVersions equal suffix version = %d, want 0", got)
	}
	if got := compareVersions("0.1.127-image2-split.3", "0.1.128"); got >= 0 {
		t.Fatalf("compareVersions newer latest = %d, want < 0", got)
	}
}

type updateServiceCacheStub struct {
	data string
}

func (s updateServiceCacheStub) GetUpdateInfo(context.Context) (string, error) {
	return s.data, nil
}

func (s updateServiceCacheStub) SetUpdateInfo(context.Context, string, time.Duration) error {
	return nil
}

func TestUpdateServiceIgnoresCacheFromDifferentRepo(t *testing.T) {
	data, err := json.Marshal(struct {
		Latest    string `json:"latest"`
		Repo      string `json:"repo"`
		Timestamp int64  `json:"timestamp"`
	}{
		Latest:    "0.1.126",
		Repo:      "Wei-Shaw/sub2api",
		Timestamp: time.Now().Unix(),
	})
	if err != nil {
		t.Fatalf("marshal cache: %v", err)
	}

	svc := NewUpdateService(updateServiceCacheStub{data: string(data)}, nil, "0.1.127", "release", "yuyi0801/sub2api")
	if cached, err := svc.getFromCache(context.Background()); err == nil || cached != nil {
		t.Fatalf("getFromCache() = (%v, %v), want repository mismatch", cached, err)
	}
}
