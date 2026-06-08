package backends

import (
	"context"
	"fmt"
	"os/exec"
	"sync"
	"time"

	"k8s.io/klog/v2"
)

// CudaCheckpoint implements the Backend interface using cuda-checkpoint and optionally CRIU.
type CudaCheckpoint struct {
	mu sync.Mutex
}

// NewCudaCheckpoint creates a new CudaCheckpoint backend.
func NewCudaCheckpoint() *CudaCheckpoint {
	return &CudaCheckpoint{}
}

// Snapshot triggers a snapshot of the accelerator context for a job.
func (c *CudaCheckpoint) Snapshot(ctx context.Context, pids []string) error {
	logger := klog.FromContext(ctx)
	c.mu.Lock()
	defer c.mu.Unlock()

	logger.Info("Snapshotting PIDs", "pids", pids)

	// 1. Lock and Checkpoint CUDA
	t0 := time.Now()
	binaryPath := c.getCudaCheckpointPath()

	pidArgs := make([]string, 0, 2*len(pids))
	for _, pid := range pids {
		pidArgs = append(pidArgs, "--pid", pid)
	}

	if err := c.runSudoCommand(ctx, binaryPath, append([]string{"--action", "lock"}, pidArgs...)...); err != nil {
		return fmt.Errorf("cuda-checkpoint lock failed: %w", err)
	}
	if err := c.runSudoCommand(ctx, binaryPath, append([]string{"--action", "checkpoint"}, pidArgs...)...); err != nil {
		return fmt.Errorf("cuda-checkpoint checkpoint failed: %w", err)
	}
	logger.Info("cuda-checkpoint action took", "duration", time.Since(t0))

	return nil
}

// Restore triggers a restoration of the accelerator context for a job.
func (c *CudaCheckpoint) Restore(ctx context.Context, pids []string) error {
	logger := klog.FromContext(ctx)
	if len(pids) == 0 {
		return fmt.Errorf("at least one PID is required")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	logger.Info("Restoring PIDs", "pids", pids)
	t0 := time.Now()
	binaryPath := c.getCudaCheckpointPath()
	pidArgs := make([]string, 0, 2*len(pids))
	for _, pid := range pids {
		pidArgs = append(pidArgs, "--pid", pid)
	}

	if err := c.runSudoCommand(ctx, binaryPath, append([]string{"--toggle"}, pidArgs...)...); err != nil {
		return fmt.Errorf("cuda-checkpoint toggle failed: %w", err)
	}
	logger.Info("cuda-checkpoint toggle took", "duration", time.Since(t0), "pids", pids)

	return nil
}

func (c *CudaCheckpoint) getCudaCheckpointPath() string {
	// First check if it's in the PATH
	if path, err := exec.LookPath("cuda-checkpoint"); err == nil {
		return path
	}
	// Fallback to the relative path used in development
	return "/usr/local/bin/cuda-checkpoint"
}

func (c *CudaCheckpoint) runSudoCommand(ctx context.Context, name string, args ...string) error {
	cmd := exec.CommandContext(ctx, name, args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("command failed: %w, output: %s", err, string(out))
	}
	return nil
}
