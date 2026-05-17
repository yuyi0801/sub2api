package service

import (
	"context"
	"testing"
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
