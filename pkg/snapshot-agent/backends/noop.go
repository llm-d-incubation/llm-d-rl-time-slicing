package backends

import (
	"context"
	"log/slog"
	"time"
)

// NoopBackend is a dummy implementation of the Backend interface.
type NoopBackend struct{}

// NewNoopBackend creates a new NoopBackend instance.
func NewNoopBackend() *NoopBackend {
	return &NoopBackend{}
}

// Snapshot simulates a snapshot operation.
func (b *NoopBackend) Snapshot(ctx context.Context, pids []string) error {
	slog.InfoContext(ctx, "NoopBackend: Snapshot called", "pids", pids)
	// Simulate some work
	time.Sleep(500 * time.Millisecond)
	return nil
}

// Restore simulates a restore operation.
func (b *NoopBackend) Restore(ctx context.Context, pids []string) error {
	slog.InfoContext(ctx, "NoopBackend: Restore called", "pids", pids)
	// Simulate some work
	time.Sleep(500 * time.Millisecond)
	return nil
}

// HealthCheck simulates a health check operation.
func (b *NoopBackend) HealthCheck(ctx context.Context) error {
	slog.InfoContext(ctx, "NoopBackend: HealthCheck called")
	return nil
}
