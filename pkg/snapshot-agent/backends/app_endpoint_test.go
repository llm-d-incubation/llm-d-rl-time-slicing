package backends_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
)

func appConfig(app pb.App, endpoint string, mode pb.SuspendMode, tags ...string) backends.Request {
	return backends.Request{
		JobID: "test-job",
		Config: &pb.BackendConfig{
			Backend: &pb.BackendConfig_AppEndpoint{
				AppEndpoint: &pb.AppEndpointConfig{
					App:       app,
					Endpoints: []string{endpoint},
					Mode:      mode,
					Tags:      tags,
				},
			},
		},
	}
}

// readJSONBody reads and unmarshals a JSON request body, reporting failures
// on the test. It returns nil for an empty body.
func readJSONBody(t *testing.T, r *http.Request) map[string]any {
	t.Helper()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Errorf("failed to read request body: %v", err)
		return nil
	}
	if len(body) == 0 {
		return nil
	}
	var parsed map[string]any
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Errorf("failed to unmarshal request body %q: %v", string(body), err)
		return nil
	}
	return parsed
}

func TestVLLMSnapshotDiscard(t *testing.T) {
	var gotMethod, gotPath, gotQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	backend := backends.NewAppEndpointBackend()
	err := backend.Snapshot(context.Background(),
		appConfig(pb.App_APP_VLLM, srv.URL, pb.SuspendMode_SUSPEND_MODE_DISCARD))
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	if gotMethod != http.MethodPost {
		t.Errorf("expected POST, got %s", gotMethod)
	}
	if gotPath != "/sleep" {
		t.Errorf("expected /sleep, got %s", gotPath)
	}
	if gotQuery != "level=2" {
		t.Errorf("expected level=2 for DISCARD, got %s", gotQuery)
	}
}

func TestTrailingSlashEndpointNormalized(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	backend := backends.NewAppEndpointBackend()
	err := backend.Snapshot(context.Background(),
		appConfig(pb.App_APP_VLLM, srv.URL+"/", pb.SuspendMode_SUSPEND_MODE_DISCARD))
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	if gotPath != "/sleep" {
		t.Errorf("expected /sleep for trailing-slash endpoint, got %s", gotPath)
	}
}

func TestVLLMSnapshotDefaultModeIsOffload(t *testing.T) {
	var gotQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.RawQuery
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	backend := backends.NewAppEndpointBackend()
	err := backend.Snapshot(context.Background(),
		appConfig(pb.App_APP_VLLM, srv.URL, pb.SuspendMode_SUSPEND_MODE_UNSPECIFIED))
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	if gotQuery != "level=1" {
		t.Errorf("expected level=1 for unspecified mode, got %s", gotQuery)
	}
}

func TestVLLMSnapshotOffload(t *testing.T) {
	var gotQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.RawQuery
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	backend := backends.NewAppEndpointBackend()
	err := backend.Snapshot(context.Background(),
		appConfig(pb.App_APP_VLLM, srv.URL, pb.SuspendMode_SUSPEND_MODE_OFFLOAD))
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	if gotQuery != "level=1" {
		t.Errorf("expected level=1 for OFFLOAD, got %s", gotQuery)
	}
}

func TestVLLMRestoreWithTags(t *testing.T) {
	var gotPath, gotQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	backend := backends.NewAppEndpointBackend()
	err := backend.Restore(context.Background(),
		appConfig(pb.App_APP_VLLM, srv.URL, pb.SuspendMode_SUSPEND_MODE_UNSPECIFIED, "weights", "kv_cache"))
	if err != nil {
		t.Fatalf("Restore failed: %v", err)
	}
	if gotPath != "/wake_up" {
		t.Errorf("expected /wake_up, got %s", gotPath)
	}
	if gotQuery != "tags=weights&tags=kv_cache" {
		t.Errorf("expected tags query params, got %s", gotQuery)
	}
}

func TestVLLMRestoreNoTags(t *testing.T) {
	var gotQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.RawQuery
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	backend := backends.NewAppEndpointBackend()
	err := backend.Restore(context.Background(),
		appConfig(pb.App_APP_VLLM, srv.URL, pb.SuspendMode_SUSPEND_MODE_UNSPECIFIED))
	if err != nil {
		t.Fatalf("Restore failed: %v", err)
	}
	if gotQuery != "" {
		t.Errorf("expected no query params, got %s", gotQuery)
	}
}

func TestSGLangSnapshotWithTags(t *testing.T) {
	var gotMethod, gotPath, gotContentType string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotContentType = r.Header.Get("Content-Type")
		gotBody = readJSONBody(t, r)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	backend := backends.NewAppEndpointBackend()
	err := backend.Snapshot(context.Background(),
		appConfig(pb.App_APP_SGLANG, srv.URL, pb.SuspendMode_SUSPEND_MODE_UNSPECIFIED, "weights", "kv_cache"))
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	if gotMethod != http.MethodPost {
		t.Errorf("expected POST, got %s", gotMethod)
	}
	if gotPath != "/release_memory_occupation" {
		t.Errorf("expected /release_memory_occupation, got %s", gotPath)
	}
	if gotContentType != "application/json" {
		t.Errorf("expected application/json, got %s", gotContentType)
	}
	tags, ok := gotBody["tags"].([]any)
	if !ok || len(tags) != 2 {
		t.Errorf("expected 2 tags, got %v", gotBody["tags"])
	}
}

func TestSGLangSnapshotNoTags(t *testing.T) {
	var gotBody string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("failed to read request body: %v", err)
		}
		gotBody = string(body)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	backend := backends.NewAppEndpointBackend()
	err := backend.Snapshot(context.Background(),
		appConfig(pb.App_APP_SGLANG, srv.URL, pb.SuspendMode_SUSPEND_MODE_UNSPECIFIED))
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	if gotBody != "{}" {
		t.Errorf("expected empty JSON body {}, got %s", gotBody)
	}
}

func TestSGLangRestore(t *testing.T) {
	var gotPath string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotBody = readJSONBody(t, r)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	backend := backends.NewAppEndpointBackend()
	err := backend.Restore(context.Background(),
		appConfig(pb.App_APP_SGLANG, srv.URL, pb.SuspendMode_SUSPEND_MODE_UNSPECIFIED, "weights"))
	if err != nil {
		t.Fatalf("Restore failed: %v", err)
	}
	if gotPath != "/resume_memory_occupation" {
		t.Errorf("expected /resume_memory_occupation, got %s", gotPath)
	}
	tags, ok := gotBody["tags"].([]any)
	if !ok || len(tags) != 1 || tags[0] != "weights" {
		t.Errorf("expected [weights], got %v", gotBody["tags"])
	}
}

func TestMultipleEndpoints(t *testing.T) {
	callCount := 0
	srv1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		callCount++
		w.WriteHeader(http.StatusOK)
	}))
	defer srv1.Close()
	srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		callCount++
		w.WriteHeader(http.StatusOK)
	}))
	defer srv2.Close()

	backend := backends.NewAppEndpointBackend()
	cfg := &pb.BackendConfig{
		Backend: &pb.BackendConfig_AppEndpoint{
			AppEndpoint: &pb.AppEndpointConfig{
				App:       pb.App_APP_VLLM,
				Endpoints: []string{srv1.URL, srv2.URL},
			},
		},
	}
	err := backend.Snapshot(context.Background(), backends.Request{JobID: "test-job", Config: cfg})
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	if callCount != 2 {
		t.Errorf("expected 2 calls, got %d", callCount)
	}
}

func TestNoEndpoints(t *testing.T) {
	backend := backends.NewAppEndpointBackend()
	err := backend.Snapshot(context.Background(), backends.Request{Config: &pb.BackendConfig{
		Backend: &pb.BackendConfig_AppEndpoint{
			AppEndpoint: &pb.AppEndpointConfig{App: pb.App_APP_VLLM},
		},
	}})
	if err == nil {
		t.Fatal("expected error for no endpoints")
	}
}

func TestUnspecifiedApp(t *testing.T) {
	backend := backends.NewAppEndpointBackend()
	err := backend.Snapshot(context.Background(), backends.Request{Config: &pb.BackendConfig{
		Backend: &pb.BackendConfig_AppEndpoint{
			AppEndpoint: &pb.AppEndpointConfig{Endpoints: []string{"http://localhost:8000"}},
		},
	}})
	if err == nil {
		t.Fatal("expected error for unspecified app")
	}
}

func TestHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		if _, err := w.Write([]byte("engine error")); err != nil {
			t.Errorf("failed to write response: %v", err)
		}
	}))
	defer srv.Close()

	backend := backends.NewAppEndpointBackend()
	err := backend.Snapshot(context.Background(),
		appConfig(pb.App_APP_VLLM, srv.URL, pb.SuspendMode_SUSPEND_MODE_DISCARD))
	if err == nil {
		t.Fatal("expected error for HTTP 500")
	}
}

func TestAppEndpointNilConfig(t *testing.T) {
	backend := backends.NewAppEndpointBackend()
	err := backend.Snapshot(context.Background(), backends.Request{})
	if err == nil {
		t.Fatal("expected error for nil config")
	}
}

func TestInterfaceCompliance(t *testing.T) {
	var _ backends.Backend = (*backends.AppEndpointBackend)(nil)
}
