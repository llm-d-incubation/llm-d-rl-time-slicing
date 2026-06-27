package backends

import (
	"context"
	"fmt"
	"log/slog"
	"os/exec"
	"sync"
	"time"
)

// GpuCr implements the Backend interface using cr_client.
type GpuCr struct {
	mu          sync.Mutex
	execCommand func(ctx context.Context, name string, args ...string) ([]byte, error)
	lookPath    func(string) (string, error)
}

// NewGpuCr creates a new GpuCr backend.
func NewGpuCr() *GpuCr {
	return &GpuCr{
		execCommand: func(ctx context.Context, name string, args ...string) ([]byte, error) {
			return exec.CommandContext(ctx, name, args...).CombinedOutput()
		},
		lookPath: exec.LookPath,
	}
}

// Snapshot triggers a snapshot of the accelerator context for a job.
func (g *GpuCr) Snapshot(ctx context.Context, pids []string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	slog.InfoContext(ctx, "Snapshotting PIDs using GPU-CR", "pids", pids)

	t0 := time.Now()
	for _, pid := range pids {
		if err := g.checkpointPID(ctx, pid); err != nil {
			return fmt.Errorf("cr_client checkpoint failed for PID %s: %w", pid, err)
		}
	}
	slog.InfoContext(ctx, "GPU-CR checkpoint took", "duration", time.Since(t0))
	return nil
}

// Restore triggers a restoration of the accelerator context for a job.
func (g *GpuCr) Restore(ctx context.Context, pids []string) error {
	if len(pids) == 0 {
		return fmt.Errorf("at least one PID is required")
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	slog.InfoContext(ctx, "Restoring PIDs using GPU-CR", "pids", pids)
	t0 := time.Now()
	for _, pid := range pids {
		if err := g.restorePID(ctx, pid); err != nil {
			return fmt.Errorf("cr_client restore failed for PID %s: %w", pid, err)
		}
	}
	slog.InfoContext(ctx, "GPU-CR restore took", "duration", time.Since(t0), "pids", pids)
	return nil
}

func (g *GpuCr) getCrClientPath() string {
	if path, err := g.lookPath("cr_client"); err == nil {
		return path
	}
	return "/usr/local/bin/cr_client"
}

func (g *GpuCr) runCommand(ctx context.Context, name string, args ...string) error {
	if out, err := g.execCommand(ctx, name, args...); err != nil {
		return fmt.Errorf("command failed: %w, output: %s", err, string(out))
	}
	return nil
}

func (g *GpuCr) checkpointPID(ctx context.Context, pid string) error {
	binaryPath := g.getCrClientPath()
	if err := g.runCommand(ctx, binaryPath, "-c", "-p", pid); err != nil {
		return err
	}
	return nil
}

func (g *GpuCr) restorePID(ctx context.Context, pid string) error {
	binaryPath := g.getCrClientPath()
	if err := g.runCommand(ctx, binaryPath, "-r", "-p", pid); err != nil {
		return err
	}
	return nil
}

// HealthCheck checks if the cr_client backend is healthy.
func (g *GpuCr) HealthCheck(ctx context.Context) error {
	binaryPath := g.getCrClientPath()
	if _, err := g.lookPath(binaryPath); err != nil {
		return fmt.Errorf("cr_client executable not found: %w", err)
	}
	return nil
}
