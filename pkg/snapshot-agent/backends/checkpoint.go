package backends

import (
	"context"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
)

// BackendType represents the type of accelerator backend.
type BackendType string

const (
	// BackendCuda is the CUDA-based checkpointing backend.
	BackendCuda BackendType = "cuda"
	// BackendNoop is a dummy backend for testing.
	BackendNoop BackendType = "noop"
)

// Request carries one backend invocation: the job it targets and the
// caller-provided (or server-resolved) backend configuration. Server-side
// per-job context needed by backends belongs here, keeping the wire API and
// the Backend interface stable as it grows.
type Request struct {
	JobID  string
	Config *pb.BackendConfig
}

// Backend defines the interface for checkpoint and restore operations.
// Each backend extracts what it needs from the Request and validates its
// own required fields, returning a clear error if inputs are missing.
type Backend interface {
	// Snapshot triggers a snapshot of the accelerator context for a job.
	Snapshot(ctx context.Context, req Request) error

	// Restore triggers a restoration of the accelerator context for a job.
	Restore(ctx context.Context, req Request) error

	// HealthCheck checks if the backend is healthy and able to serve
	// snapshot/restore requests.
	HealthCheck(ctx context.Context) error
}
