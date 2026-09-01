package backends

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/tpu"
)

const (
	// tpuCheckpointTimeoutSecs is the per-process --timeout passed to the
	// CLI for one checkpoint attempt.
	tpuCheckpointTimeoutSecs = 120

	// tpuRestoreCLITimeoutSecs is the per-process --timeout passed to the
	// CLI for the restore rendezvous: every mesh member's RESTORE parks in
	// libtpu's barrier until its peers arrive, so this bounds the whole
	// slice-wide rendezvous, not one process's work.
	tpuRestoreCLITimeoutSecs = 600

	// tpuVfioGateTimeout bounds the wait for the previous occupant to
	// release its vfio iommu groups before restore is issued.
	tpuVfioGateTimeout = 600 * time.Second

	// tpuSnapshotTimeout is a context backstop for one Snapshot call: two
	// checkpoint attempts (the CLI runs concurrently across processes, so
	// one attempt costs one per-process timeout) plus slack. The CLI
	// enforces its own, tighter timeout and exits on its own.
	tpuSnapshotTimeout = 300 * time.Second

	// tpuRestoreTimeout is a context backstop for one Restore call: up to
	// 600s waiting for the previous occupant's vfio groups to be released,
	// up to 600s for the slice-wide restore rendezvous, plus slack.
	tpuRestoreTimeout = 1300 * time.Second

	// tpuCheckpointRetryBackoff is the pause before the single checkpoint
	// retry. Restore is never retried at any layer.
	tpuCheckpointRetryBackoff = 2 * time.Second

	// tpuVfioDir must exist for this node to host TPU jobs; libtpu attaches
	// chips through their vfio iommu groups.
	tpuVfioDir = "/dev/vfio"
)

// TpuCheckpoint implements the Backend interface for TPU processes using
// gVisor's tpucheckpoint CLI (libtpu control-pipe protocol), mirroring how
// the CUDA backend shells out to NVIDIA's cuda-checkpoint utility. The CLI
// is batch-native (gVisor PR #14455): it takes --pid <pid>[,<pid>...], runs
// the operation concurrently across PIDs, and reports per-PID failures on
// stderr as "tpucheckpoint: pid <N>: <err>". The backend issues ONE
// invocation per job and owns the job-level contracts inherited from
// libtpu:
//
//   - Restore is a slice-wide rendezvous: every mesh member's RESTORE must
//     be pending simultaneously (a lone request parks until its peers
//     arrive). The CLI's internal fan-out satisfies this as long as ALL of
//     the job's PIDs go into a single invocation.
//   - A failed restore must NEVER be retried (the job faults instead): a
//     timed-out-but-pending RESTORE is still waiting in the barrier, and
//     re-sending injects a duplicate request that wedges libtpu's state
//     machine.
//   - A checkpoint retry must target ONLY the PIDs the CLI reported as
//     failed: the rest are already parked, and a second CHECKPOINT to a
//     parked process is not part of the libtpu contract.
//   - Restore gates on the previous occupant releasing its vfio iommu
//     groups (tpu.WaitVfioFree), ONCE per job before the RESTORE is
//     issued — early members hold their own groups while parked, so gating
//     per-process would deadlock against the job's own mesh.
type TpuCheckpoint struct {
	mu           sync.Mutex
	execCommand  func(ctx context.Context, name string, args ...string) ([]byte, error)
	lookPath     func(string) (string, error)
	statPath     func(string) (os.FileInfo, error)
	waitVfioFree func(ctx context.Context) error
	clearLocks   func(ctx context.Context, pids []string)
	retryBackoff time.Duration
}

// NewTpuCheckpoint creates a new TpuCheckpoint backend.
func NewTpuCheckpoint() *TpuCheckpoint {
	return &TpuCheckpoint{
		execCommand: func(ctx context.Context, name string, args ...string) ([]byte, error) {
			return exec.CommandContext(ctx, name, args...).CombinedOutput()
		},
		lookPath: exec.LookPath,
		statPath: os.Stat,
		waitVfioFree: func(ctx context.Context) error {
			return tpu.WaitVfioFree(ctx, tpuVfioGateTimeout)
		},
		clearLocks:   tpu.ClearLockfiles,
		retryBackoff: tpuCheckpointRetryBackoff,
	}
}

// Snapshot parks the TPU state of every process of the job with one batched
// CLI invocation. Failed PIDs get one retry (a failed checkpoint leaves the
// process attached and is safe to re-issue, unlike restore); PIDs that
// succeeded are parked and are excluded from the retry.
func (t *TpuCheckpoint) Snapshot(ctx context.Context, req Request) error {
	pids := ExtractTpuPIDStrings(req.Config)
	if len(pids) == 0 {
		return fmt.Errorf("at least one PID is required for TPU snapshot")
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	slog.InfoContext(ctx, "Snapshotting TPU PIDs", "pids", pids)
	t0 := time.Now()
	cmdCtx, cancel := context.WithTimeout(ctx, tpuSnapshotTimeout)
	defer cancel()
	if err := t.checkpointWithRetry(cmdCtx, pids); err != nil {
		return fmt.Errorf("tpucheckpoint checkpoint failed: %w", err)
	}
	// A stale container-side lockfile blocks the next libtpu init; parked
	// processes release their vfio groups within seconds, so log who still
	// holds what for stall postmortems.
	t.clearLocks(ctx, pids)
	slog.InfoContext(ctx, "tpucheckpoint checkpoint took", "duration", time.Since(t0), "pids", pids,
		"vfioHoldersAfter", tpu.VfioGroupHolders())
	return nil
}

// Restore restores the TPU state of every process of the job: one vfio gate,
// then one batched CLI invocation carrying ALL PIDs (the CLI fans out
// concurrently, satisfying the rendezvous). Exactly one attempt: on failure
// the error propagates and the state machine marks the job FAULTED — do not
// add retries at any layer.
func (t *TpuCheckpoint) Restore(ctx context.Context, req Request) error {
	pids := ExtractTpuPIDStrings(req.Config)
	if len(pids) == 0 {
		return fmt.Errorf("at least one PID is required for TPU restore")
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	slog.InfoContext(ctx, "Restoring TPU PIDs", "pids", pids)
	t0 := time.Now()
	cmdCtx, cancel := context.WithTimeout(ctx, tpuRestoreTimeout)
	defer cancel()

	// Restoring into a busy iommu group fails AND poisons the driver, so
	// contention must be detected BEFORE any RESTORE is issued.
	if err := t.waitVfioFree(cmdCtx); err != nil {
		return fmt.Errorf("vfio gate before restore failed (no RESTORE was issued; safe to retry after the groups free up): %w", err)
	}

	if _, err := t.runBatch(cmdCtx, "restore", pids, tpuRestoreCLITimeoutSecs); err != nil {
		return fmt.Errorf("tpucheckpoint restore failed (job must be treated as faulted, never re-issue a restore): %w", err)
	}
	slog.InfoContext(ctx, "tpucheckpoint restore took", "duration", time.Since(t0), "pids", pids)
	return nil
}

// HealthCheck verifies the CLI is installed and the node exposes vfio groups.
func (t *TpuCheckpoint) HealthCheck(ctx context.Context) error {
	binaryPath := t.getTpuCheckpointPath()
	if _, err := t.lookPath(binaryPath); err != nil {
		return fmt.Errorf("tpucheckpoint executable not found: %w", err)
	}
	if _, err := t.statPath(tpuVfioDir); err != nil {
		return fmt.Errorf("no %s on this node (not a TPU host?): %w", tpuVfioDir, err)
	}
	return nil
}

// checkpointWithRetry issues one batched checkpoint and, on failure, retries
// exactly the PIDs the CLI reported as failed. PIDs absent from the failure
// report are parked and must not receive a second CHECKPOINT.
func (t *TpuCheckpoint) checkpointWithRetry(ctx context.Context, pids []string) error {
	out, err := t.runBatch(ctx, "checkpoint", pids, tpuCheckpointTimeoutSecs)
	if err == nil {
		return nil
	}
	// If the CLI died without reporting per-PID results (exec failure,
	// usage error — both happen before any control request is issued),
	// re-issuing the whole batch is safe.
	retryPids := failedPids(out, pids)
	if len(retryPids) == 0 {
		retryPids = pids
	}
	select {
	case <-ctx.Done():
		return fmt.Errorf("checkpoint pids %v: %w (last attempt: %w)", pids, ctx.Err(), err)
	case <-time.After(t.retryBackoff):
	}
	if _, retryErr := t.runBatch(ctx, "checkpoint", retryPids, tpuCheckpointTimeoutSecs); retryErr != nil {
		return fmt.Errorf("checkpoint pids %v failed after retry: %w (first attempt: %w)", retryPids, retryErr, err)
	}
	return nil
}

// runBatch runs one CLI invocation covering all pids; the CLI processes them
// concurrently, which restore's rendezvous semantics depend on.
func (t *TpuCheckpoint) runBatch(ctx context.Context, action string, pids []string, timeoutSecs int) ([]byte, error) {
	args := []string{"--action", action, "--pid", strings.Join(pids, ","), "--timeout", strconv.Itoa(timeoutSecs)}
	t0 := time.Now()
	out, err := t.execCommand(ctx, t.getTpuCheckpointPath(), args...)
	if err != nil {
		wrapped := fmt.Errorf("%s pids %v: command failed: %w, output: %s", action, pids, err, string(out))
		slog.WarnContext(ctx, "tpucheckpoint failed",
			"action", action, "pids", pids, "duration", time.Since(t0), "error", wrapped)
		return out, wrapped
	}
	slog.InfoContext(ctx, "tpucheckpoint succeeded",
		"action", action, "pids", pids, "duration", time.Since(t0), "output", string(out))
	return out, nil
}

// tpuFailureLine matches the CLI's per-PID failure report on stderr:
// "tpucheckpoint: pid <N>: <err>".
var tpuFailureLine = regexp.MustCompile(`(?m)^tpucheckpoint: pid (\d+):`)

// failedPids extracts the PIDs the CLI reported as failed, restricted to the
// PIDs that were actually requested (in request order).
func failedPids(out []byte, requested []string) []string {
	reported := make(map[string]bool)
	for _, m := range tpuFailureLine.FindAllSubmatch(out, -1) {
		reported[string(m[1])] = true
	}
	var failed []string
	for _, pid := range requested {
		if reported[pid] {
			failed = append(failed, pid)
		}
	}
	return failed
}

func (t *TpuCheckpoint) getTpuCheckpointPath() string {
	// First check the PATH, under both the gVisor binary name and the
	// hyphenated alias.
	for _, name := range []string{"tpucheckpoint", "tpu-checkpoint"} {
		if path, err := t.lookPath(name); err == nil {
			return path
		}
	}
	// Fallback to the location the agent image installs it to.
	return "/usr/local/bin/tpucheckpoint"
}

// ExtractTpuPIDStrings extracts PID strings from a TPU BackendConfig.
func ExtractTpuPIDStrings(config *pb.BackendConfig) []string {
	if config == nil {
		return nil
	}
	tpuCfg := config.GetTpu()
	if tpuCfg == nil {
		return nil
	}
	target := tpuCfg.GetExplicitTarget()
	if target == nil {
		return nil
	}
	pids := make([]string, 0, len(target.GetPids()))
	for _, pid := range target.GetPids() {
		pids = append(pids, strconv.Itoa(int(pid)))
	}
	return pids
}

// BuildTpuConfig wraps PID strings into a TPU BackendConfig.
func BuildTpuConfig(pidStrings []string) *pb.BackendConfig {
	pids := make([]int32, 0, len(pidStrings))
	for _, s := range pidStrings {
		if pid, err := strconv.ParseInt(s, 10, 32); err == nil {
			pids = append(pids, int32(pid))
		}
	}
	return &pb.BackendConfig{
		Backend: &pb.BackendConfig_Tpu{
			Tpu: &pb.TpuBackendConfig{
				ExplicitTarget: &pb.ProcessTarget{Pids: pids},
			},
		},
	}
}
