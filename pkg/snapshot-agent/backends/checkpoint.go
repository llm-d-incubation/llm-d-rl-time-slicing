package backends

import "context"

// Backend defines the interface for checkpoint and restore operations.
type Backend interface {
	// Snapshot triggers a snapshot of the accelerator context for a job.
	// Returns storageBytes, deviceBytes, and error.
	Snapshot(ctx context.Context, pids []string) error

	// Restore triggers a restoration of the accelerator context for a job.
	Restore(ctx context.Context, pids []string) error
}
