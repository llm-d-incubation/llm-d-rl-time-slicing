package backends

import (
	"context"
	"fmt"
	"log"
	"os/exec"
	"sync"
	"time"
)

// CudaCheckpoint implements the Backend interface using cuda-checkpoint and optionally CRIU.
type CudaCheckpoint struct {
	mu          sync.Mutex
}

// NewCudaCheckpoint creates a new CudaCheckpoint backend.
func NewCudaCheckpoint() *CudaCheckpoint {
	return &CudaCheckpoint{}
}

// Snapshot triggers a snapshot of the accelerator context for a job.
func (c *CudaCheckpoint) Snapshot(ctx context.Context, pids []string) error {

	c.mu.Lock()
	defer c.mu.Unlock()

	log.Printf("Snapshotting PIDs %v", pids)

	// 1. Lock and Checkpoint CUDA
	t0 := time.Now()
	binaryPath := c.getCudaCheckpointPath()

	var pidArgs []string
	for _, pid := range pids {
		pidArgs = append(pidArgs, "--pid", pid)
	}

	if err := c.runSudoCommand(binaryPath, append([]string{"--action", "lock"}, pidArgs...)...); err != nil {
		return fmt.Errorf("cuda-checkpoint lock failed: %w", err)
	}
	if err := c.runSudoCommand(binaryPath, append([]string{"--action", "checkpoint"}, pidArgs...)...); err != nil {
		return fmt.Errorf("cuda-checkpoint checkpoint failed: %w", err)
	}
	log.Printf("[Metric] cuda-checkpoint action took %v", time.Since(t0))

	return  nil
}

// Restore triggers a restoration of the accelerator context for a job.
func (c *CudaCheckpoint) Restore(ctx context.Context, pids []string) (error) {
	if len(pids) == 0 {
		return fmt.Errorf("at least one PID is required")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	log.Printf("Restoring PIDs %v", pids)
	t0 := time.Now()
	binaryPath := c.getCudaCheckpointPath()
	var pidArgs []string
	for _, pid := range pids {
		pidArgs = append(pidArgs, "--pid", pid)
	}

	if err := c.runSudoCommand(binaryPath, append([]string{"--toggle"}, pidArgs...)...); err != nil {
		return fmt.Errorf("cuda-checkpoint toggle failed: %w", err)
	}
	log.Printf("[Metric] cuda-checkpoint toggle took %v for PIDs %v", time.Since(t0), pids)

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

func (c *CudaCheckpoint) runSudoCommand(name string, args ...string) error {
	// Check if 'sudo' exists in PATH
	_, err := exec.LookPath("sudo")
	var cmd *exec.Cmd
	if err != nil {
		log.Printf("'sudo' not found in PATH, attempting to run command directly: %s %v", name, args)
		cmd = exec.Command(name, args...)
	} else {
		cmd = exec.Command("sudo", append([]string{name}, args...)...)
	}

	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("command failed: %v, output: %s", err, string(out))
	}
	return nil
}