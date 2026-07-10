package backends

import "context"

// BackendType represents the type of accelerator backend.
type BackendType string

const (
	// BackendCuda is the CUDA-based checkpointing backend.
	BackendCuda BackendType = "cuda"
	// BackendNoop is a dummy backend for testing.
	BackendNoop BackendType = "noop"
	// BackendVLLM is the vLLM application-aware sleep/wake backend.
	BackendVLLM BackendType = "vllm"
	// BackendSGLang is the SGLang application-aware memory release/resume backend.
	BackendSGLang BackendType = "sglang"
)

// Backend defines the interface for checkpoint and restore operations.
type Backend interface {
	// Snapshot triggers a snapshot of the accelerator context for a job.
	// Returns storageBytes, deviceBytes, and error.
	Snapshot(ctx context.Context, pids []string) error

	// Restore triggers a restoration of the accelerator context for a job.
	Restore(ctx context.Context, pids []string) error

	// HealthCheck checks if the backend is healthy by initializing the backend
	// and the discovery provider.
	HealthCheck(ctx context.Context) error
}

// AppConfig holds per-request configuration for application-aware backends.
type AppConfig struct {
	Endpoints  []string
	SleepLevel int32
	Tags       []string
}

// AppAwareBackend extends Backend for application-level sleep/wake operations.
// These backends operate via HTTP endpoints rather than PIDs.
type AppAwareBackend interface {
	Backend
	SnapshotApp(ctx context.Context, config AppConfig) error
	RestoreApp(ctx context.Context, config AppConfig) error
}
