package backends

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
)

// DirectMemory implements the Backend interface using cr_client.
type DirectMemory struct {
	mu          sync.Mutex
	execCommand func(ctx context.Context, name string, args ...string) ([]byte, error)
	lookPath    func(string) (string, error)
	statFunc    func(string) (os.FileInfo, error)
}

// NewDirectMemory creates a new DirectMemory backend.
func NewDirectMemory() *DirectMemory {
	return &DirectMemory{
		execCommand: func(ctx context.Context, name string, args ...string) ([]byte, error) {
			return exec.CommandContext(ctx, name, args...).CombinedOutput()
		},
		lookPath: exec.LookPath,
		statFunc: os.Stat,
	}
}

// Snapshot triggers a snapshot of the target processes for a job using cr_client.
func (d *DirectMemory) Snapshot(ctx context.Context, req Request) error {
	pids := ExtractDirectMemoryPIDStrings(req.Config)
	if len(pids) == 0 {
		return fmt.Errorf("at least one PID is required for Direct Memory snapshot")
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	slog.InfoContext(ctx, "Snapshotting PIDs using Direct Memory", "pids", pids)

	t0 := time.Now()
	for _, pid := range pids {
		if err := d.checkpointPID(ctx, pid); err != nil {
			return fmt.Errorf("cr_client checkpoint failed for PID %s: %w", pid, err)
		}
	}
	slog.InfoContext(ctx, "cr_client checkpoint took", "duration", time.Since(t0))
	return nil
}

// Restore triggers a restoration of the target processes for a job using cr_client.
func (d *DirectMemory) Restore(ctx context.Context, req Request) error {
	pids := ExtractDirectMemoryPIDStrings(req.Config)
	if len(pids) == 0 {
		return fmt.Errorf("at least one PID is required for Direct Memory restore")
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	slog.InfoContext(ctx, "Restoring PIDs using Direct Memory", "pids", pids)
	t0 := time.Now()
	for _, pid := range pids {
		if err := d.restorePID(ctx, pid); err != nil {
			return fmt.Errorf("cr_client restore failed for PID %s: %w", pid, err)
		}
	}
	slog.InfoContext(ctx, "cr_client restore took", "duration", time.Since(t0), "pids", pids)
	return nil
}

func (d *DirectMemory) getCrClientPath() string {
	for _, p := range []string{
		"cr_client",
		"/usr/bin/cr_client",
		"/bin/cr_client",
		"/opt/bin/cr_client",
		"/usr/local/bin/cr_client",
	} {
		if path, err := d.lookPath(p); err == nil {
			return path
		}
		if _, err := d.statFunc(p); err == nil {
			return p
		}
	}
	return "/opt/bin/cr_client"
}

func (d *DirectMemory) runCommand(ctx context.Context, name string, args ...string) error {
	if out, err := d.execCommand(ctx, name, args...); err != nil {
		return fmt.Errorf("command failed: %w, output: %s", err, string(out))
	}
	return nil
}

func (d *DirectMemory) checkpointPID(ctx context.Context, pid string) error {
	binaryPath := d.getCrClientPath()
	if err := d.runCommand(ctx, binaryPath, "-c", "-p", pid); err != nil {
		return err
	}
	return nil
}

func (d *DirectMemory) restorePID(ctx context.Context, pid string) error {
	binaryPath := d.getCrClientPath()
	if err := d.runCommand(ctx, binaryPath, "-r", "-p", pid); err != nil {
		return err
	}
	return nil
}

// HealthCheck checks if the Direct Memory backend is healthy.
func (d *DirectMemory) HealthCheck(ctx context.Context) error {
	binaryPath := d.getCrClientPath()
	if _, err := d.lookPath(binaryPath); err != nil {
		if _, errStat := d.statFunc(binaryPath); errStat != nil {
			return fmt.Errorf("cr_client executable not found: %w", err)
		}
	}
	return nil
}

// ExtractDirectMemoryPIDStrings extracts PID strings from a DirectMemory BackendConfig.
func ExtractDirectMemoryPIDStrings(config *pb.BackendConfig) []string {
	if config == nil {
		return nil
	}
	dm := config.GetDirectMemory()
	if dm == nil {
		return nil
	}
	target := dm.GetExplicitTarget()
	if target == nil {
		return nil
	}
	pids := make([]string, 0, len(target.GetPids()))
	for _, pid := range target.GetPids() {
		pids = append(pids, strconv.Itoa(int(pid)))
	}
	return pids
}

// BuildDirectMemoryConfig wraps PID strings into a DirectMemory BackendConfig.
func BuildDirectMemoryConfig(pidStrings []string) (*pb.BackendConfig, error) {
	pids := make([]int32, 0, len(pidStrings))
	for _, s := range pidStrings {
		pid, err := strconv.ParseInt(s, 10, 32)
		if err != nil {
			return nil, fmt.Errorf("invalid PID string %q: %w", s, err)
		}
		pids = append(pids, int32(pid))
	}
	return &pb.BackendConfig{
		Backend: &pb.BackendConfig_DirectMemory{
			DirectMemory: &pb.DirectMemoryBackendConfig{
				ExplicitTarget: &pb.ProcessTarget{Pids: pids},
			},
		},
	}, nil
}
