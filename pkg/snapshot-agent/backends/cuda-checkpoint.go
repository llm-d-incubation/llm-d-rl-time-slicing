package backends

import (
	"context"
	"fmt"
	"log/slog"
	"os/exec"
	"sync"
	"time"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
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
	if err := c.checkpointPIDs(ctx, pids); err != nil {
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
	if err := c.restorePIDs(ctx, pids); err != nil {
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

func (c *CudaCheckpoint) checkpointPIDs(ctx context.Context, pids []string) error {
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
	return nil
}

func (c *CudaCheckpoint) restorePIDs(ctx context.Context, pids []string) error {
	binaryPath := c.getCudaCheckpointPath()
	pidArgs := make([]string, 0, 2*len(pids))
	for _, pid := range pids {
		pidArgs = append(pidArgs, "--pid", pid)
	}
	if err := c.runSudoCommand(ctx, binaryPath, append([]string{"--toggle"}, pidArgs...)...); err != nil {
		return fmt.Errorf("cuda-checkpoint toggle failed: %w", err)
	}
	return nil
}

// HealthCheck checks if the cuda-checkpoint backend is healthy by initializing the backend
// and the discovery provider.
func (c *CudaCheckpoint) HealthCheck(ctx context.Context) error {
	// 1. Check if cuda-checkpoint executable is available
	binaryPath := c.getCudaCheckpointPath()
	if _, err := exec.LookPath(binaryPath); err != nil {
		return fmt.Errorf("cuda-checkpoint executable not found: %w", err)
	}

	// 2. Initialize NVML
	if ret := nvml.Init(); ret != nvml.SUCCESS {
		return fmt.Errorf("failed to initialize NVML: %v", nvml.ErrorString(ret))
	}
	defer func() {
		if ret := nvml.Shutdown(); ret != nvml.SUCCESS {
			slog.Error("Failed to shutdown NVML", "error", nvml.ErrorString(ret))
		}
	}()

	// 3. Check if there are any GPUs attached to the system
	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS {
		return fmt.Errorf("failed to get device count: %v", nvml.ErrorString(ret))
	}

	if count == 0 {
		return fmt.Errorf("no GPUs found on the system")
	}

	return nil
}
