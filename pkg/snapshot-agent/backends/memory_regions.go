package backends

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/utils"
)

// MemoryRegions implements the Backend interface for selective checkpoint
// and restore of explicit device-memory regions of a running process, using
// the GPU-CR cr_client with destination-path dumps.
// Regions are provided by the caller through MemoryRegionsBackendConfig; the
// backend performs no discovery.
//
// The agent moves NO dump bytes itself: each snapshot is
// written by the workload's preloader directly into a per-slot destination
// file (cr_client -o), and each restore reads straight from it. The agent
// only names files, pre-creates them (via cr_client), and garbage-collects —
// none of which faults a hugetlb page, which is what lets the DaemonSet run
// with no hugepages-2Mi request.
//
// Environment configuration:
//
//	EXPORT_FILE_PATH        GPU-CR data dir: dump/staging buffers and the
//	                        destination group store (default /mnt/huge-ckpt)
//	GPU_CR_CTL_PATH         control-plane tmpfs (control-<pid>,
//	                        pid_map_<pid>, ctl-ready-<pid>); unset = legacy
//	                        layout sharing the data dir
//	GPU_CR_GROUP_STORE      destination store override (default <data>/groups)
//	GPU_CR_OP_TIMEOUT_SEC   per-cr_client-invocation timeout (default 120)
type MemoryRegions struct {
	mu          sync.Mutex
	execCommand func(ctx context.Context, name string, args ...string) ([]byte, error)
	statFunc    func(string) (os.FileInfo, error)
	// starttime reads a pid's starttime (utils.ProcStarttime in
	// production; injectable for tests, which use pids with no procfs
	// entry).
	starttime func(pid string) (int64, error)
	// procRoot is "/proc" in production; injectable for tests.
	procRoot string
}

// NewMemoryRegions creates a new MemoryRegions backend.
func NewMemoryRegions() *MemoryRegions {
	return &MemoryRegions{
		execCommand: func(ctx context.Context, name string, args ...string) ([]byte, error) {
			return exec.CommandContext(ctx, name, args...).CombinedOutput()
		},
		statFunc:  os.Stat,
		starttime: utils.ProcStarttime,
		procRoot:  "/proc",
	}
}

// ctlFilesDir is where the control plane lives: control-<pid>, pid_map_<pid>,
// ctl-ready-<pid>. With GPU_CR_CTL_PATH set that's a tmpfs; unset
// means the legacy layout where control files share the data dir.
func ctlFilesDir() string {
	if d := os.Getenv("GPU_CR_CTL_PATH"); d != "" {
		return d
	}
	return utils.DataDir()
}

// groupDir validates a snapshot slot name and returns its directory under
// the destination store. Slots must be a single path segment: nesting is
// rejected along with traversal because GC reaps store entries at the top
// level only (a nested slot's owner directories would be invisible to it).
func groupDir(slot string) (string, error) {
	store := utils.GroupStoreDir()
	dir := filepath.Join(store, filepath.Clean(slot))
	rel, err := filepath.Rel(store, dir)
	if err != nil || rel == "." || strings.HasPrefix(rel, "..") || strings.ContainsRune(rel, os.PathSeparator) {
		return "", fmt.Errorf("invalid snapshot slot (path traversal or nested path): %q", slot)
	}
	return dir, nil
}

// ownerDir returns the per-owner directory inside a slot:
// <slot>/<pid>-<starttime>. The dirname IS the slot's ownership metadata —
// GC parses it for the owner-liveness delete decision — encoded in the path
// because a name, unlike file contents, cannot fail to be written on a
// hugetlbfs store (which rejects write(2)), and it exists atomically with
// the dump destination it contains. Starttime is re-read from procfs on
// every operation: for the process that took the snapshot it is a constant,
// so restore recomputes the identical path, while a recycled pid computes a
// different one and correctly finds nothing.
func (g *MemoryRegions) ownerDir(slotDir, pid string) (string, error) {
	st, err := g.starttime(pid)
	if err != nil {
		return "", fmt.Errorf("starttime of owner pid %s: %w", pid, err)
	}
	return filepath.Join(slotDir, fmt.Sprintf("%s-%d", pid, st)), nil
}

// touchGroup bumps the slot dir mtime explicitly on every op.
func touchGroup(dir string) {
	now := time.Now()
	if err := os.Chtimes(dir, now, now); err != nil {
		slog.Warn("touch group dir", "dir", dir, "err", err)
	}
}

// regionSpecs validates the config's regions and groups them per PID,
// preserving order, each formatted as the cr_client "0xADDR:SIZE" spec.
func regionSpecs(cfg *pb.MemoryRegionsBackendConfig) (map[int32][]string, error) {
	regions := cfg.GetRegions()
	if len(regions) == 0 {
		return nil, fmt.Errorf("at least one memory region is required")
	}
	specs := make(map[int32][]string)
	for _, r := range regions {
		if r.GetPid() <= 0 {
			return nil, fmt.Errorf("memory region pid must be positive, got %d", r.GetPid())
		}
		if r.GetSizeBytes() == 0 {
			return nil, fmt.Errorf("memory region size_bytes must be positive (pid %d, address 0x%x)", r.GetPid(), r.GetAddress())
		}
		specs[r.GetPid()] = append(specs[r.GetPid()], fmt.Sprintf("0x%x:%d", r.GetAddress(), r.GetSizeBytes()))
	}
	return specs, nil
}

// regionPIDs returns the config's PIDs in first-appearance order, so
// cr_client invocations are deterministic.
func regionPIDs(cfg *pb.MemoryRegionsBackendConfig) []int32 {
	seen := make(map[int32]bool)
	var pids []int32
	for _, r := range cfg.GetRegions() {
		if !seen[r.GetPid()] {
			seen[r.GetPid()] = true
			pids = append(pids, r.GetPid())
		}
	}
	return pids
}

// snapshotSlot returns the destination-store slot for the request:
// MemoryRegionsBackendConfig.snapshot_name, falling back to the job ID when
// empty. The request's `group` is deliberately NOT used: group identifies a
// set of related jobs for the orchestrator and does not name agent-side
// storage. The returned slot is guaranteed to resolve inside the store.
func snapshotSlot(req Request) (string, error) {
	slot := req.Config.GetMemoryRegions().GetSnapshotName()
	if slot == "" {
		slot = req.JobID
	}
	if slot == "" {
		return "", fmt.Errorf("snapshot slot is empty: set snapshot_name or job_id")
	}
	if _, err := groupDir(slot); err != nil {
		return "", err
	}
	return slot, nil
}

// Snapshot triggers a selective snapshot of the configured memory regions,
// dumped by the preloader directly into <group store>/<slot>/<id>.
func (g *MemoryRegions) Snapshot(ctx context.Context, req Request) error {
	cfg := req.Config.GetMemoryRegions()
	if cfg == nil {
		return fmt.Errorf("memory-regions backend requires BackendConfig.memory_regions")
	}
	specs, err := regionSpecs(cfg)
	if err != nil {
		return err
	}
	slot, err := snapshotSlot(req)
	if err != nil {
		return err
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	utils.StoreMu.Lock()
	defer utils.StoreMu.Unlock()

	slog.InfoContext(ctx, "Snapshotting memory regions using GPU-CR",
		"jobID", req.JobID, "slot", slot, "pids", regionPIDs(cfg), "regions", len(cfg.GetRegions()))

	targetDir, err := groupDir(slot)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		return fmt.Errorf("failed to create slot dir %s: %w", targetDir, err)
	}

	t0 := time.Now()
	for _, pid := range regionPIDs(cfg) {
		pidStr := strconv.Itoa(int(pid))
		id, err := g.ensureIDForPid(ctx, pidStr)
		if err != nil {
			return fmt.Errorf("failed to resolve PID %d to ID: %w", pid, err)
		}
		// The owner dir doubles as the slot's ownership record, so it is
		// created BEFORE the checkpoint runs: a dump file can never exist
		// in a slot without its owner recorded, and a checkpoint failure
		// partway through a multi-PID request leaves only slots the
		// sweeper reclaims normally once the recorded owners die.
		ownDir, err := g.ownerDir(targetDir, pidStr)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(ownDir, 0o755); err != nil {
			return fmt.Errorf("failed to create owner dir %s: %w", ownDir, err)
		}
		dest := filepath.Join(ownDir, id)
		specStr := strings.Join(specs[pid], ",")
		if err := g.checkpointRegions(ctx, pid, specStr, dest); err != nil {
			return fmt.Errorf("cr_client checkpoint failed for PID %d with specs %s: %w", pid, specStr, err)
		}
	}
	slog.InfoContext(ctx, "GPU-CR selective checkpoint (direct-to-destination) took", "duration", time.Since(t0))

	// Explicit utimes: GC timing must never ride on mmap-driven mtimes,
	// which hugetlbfs does not reliably update.
	touchGroup(targetDir)

	return nil
}

// Restore triggers a selective restoration of the configured memory regions,
// read by the preloader directly from <group store>/<slot>/<id>.
func (g *MemoryRegions) Restore(ctx context.Context, req Request) error {
	cfg := req.Config.GetMemoryRegions()
	if cfg == nil {
		return fmt.Errorf("memory-regions backend requires BackendConfig.memory_regions")
	}
	specs, err := regionSpecs(cfg)
	if err != nil {
		return err
	}
	slot, err := snapshotSlot(req)
	if err != nil {
		return err
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	utils.StoreMu.Lock()
	defer utils.StoreMu.Unlock()

	slog.InfoContext(ctx, "Restoring memory regions using GPU-CR",
		"jobID", req.JobID, "slot", slot, "pids", regionPIDs(cfg), "regions", len(cfg.GetRegions()))

	targetDir, err := groupDir(slot)
	if err != nil {
		return err
	}
	if _, err := os.Stat(targetDir); err != nil {
		return fmt.Errorf("snapshot slot %q not found in group store %s: %w", slot, utils.GroupStoreDir(), err)
	}

	t0 := time.Now()
	for _, pid := range regionPIDs(cfg) {
		pidStr := strconv.Itoa(int(pid))
		id, err := g.ensureIDForPid(ctx, pidStr)
		if err != nil {
			return fmt.Errorf("failed to resolve PID %d to ID: %w", pid, err)
		}
		// Same-process recomputation: the pid's current starttime yields
		// the exact path Snapshot used, while a different (or recycled)
		// process yields a path that does not exist.
		ownDir, err := g.ownerDir(targetDir, pidStr)
		if err != nil {
			return err
		}
		dest := filepath.Join(ownDir, id)
		if _, err := os.Stat(dest); err != nil {
			return fmt.Errorf("slot %q holds no parked state for PID %d (owner dir %s): %w",
				slot, pid, filepath.Base(ownDir), err)
		}
		specStr := strings.Join(specs[pid], ",")
		if err := g.restoreRegions(ctx, pid, specStr, dest); err != nil {
			return fmt.Errorf("cr_client restore failed for PID %d with specs %s: %w", pid, specStr, err)
		}
	}
	slog.InfoContext(ctx, "GPU-CR selective restore (direct-from-destination) took", "duration", time.Since(t0))
	touchGroup(targetDir)
	return nil
}

// HealthCheck reports whether the cr_client binary is present at its fixed
// install location, so grpc.health.v1.Health/Check with service
// "memory-regions" reflects backend readiness (an agent image built without
// cr_client reports NOT_SERVING).
func (g *MemoryRegions) HealthCheck(ctx context.Context) error {
	if _, err := g.statFunc(crClientPath); err != nil {
		return fmt.Errorf("cr_client not found at %s: %w", crClientPath, err)
	}
	return nil
}

func (g *MemoryRegions) runCommand(ctx context.Context, name string, args ...string) error {
	if out, err := g.execCommand(ctx, name, args...); err != nil {
		return fmt.Errorf("command failed: %w, output: %s", err, string(out))
	}
	return nil
}

func (g *MemoryRegions) checkpointRegions(ctx context.Context, pid int32, spec, dest string) error {
	ctx, cancel := context.WithTimeout(ctx, opTimeout())
	defer cancel()
	if err := g.runCommand(ctx, crClientPath, "-c", "-p", strconv.Itoa(int(pid)), "-s", spec, "-o", dest); err != nil {
		return fmt.Errorf("cr_client checkpoint (timeout %s): %w", opTimeout(), err)
	}
	return nil
}

func (g *MemoryRegions) restoreRegions(ctx context.Context, pid int32, spec, dest string) error {
	ctx, cancel := context.WithTimeout(ctx, opTimeout())
	defer cancel()
	if err := g.runCommand(ctx, crClientPath, "-r", "-p", strconv.Itoa(int(pid)), "-s", spec, "-o", dest); err != nil {
		return fmt.Errorf("cr_client restore (timeout %s): %w", opTimeout(), err)
	}
	return nil
}

// ensureIDForPid returns the GPU-CR dump-buffer id for pid, driving the
// preloader's lazy init when needed. Destination-path ops must know the id
// BEFORE the first cr_client signal (the -o file is named by it), but the
// preloader only writes pid_map/creates the buffer inside init_CR — which
// historically ran lazily off the first checkpoint signal. cr_client -i
// triggers exactly that init and is idempotent for already-initialized
// processes.
func (g *MemoryRegions) ensureIDForPid(ctx context.Context, pid string) (string, error) {
	id, err := g.resolvePidToID(pid)
	if err == nil {
		return id, nil
	}
	slog.InfoContext(ctx, "PID not resolvable yet; driving preloader init via cr_client -i", "pid", pid)
	ictx, cancel := context.WithTimeout(ctx, opTimeout())
	defer cancel()
	if ierr := g.runCommand(ictx, crClientPath, "-i", "-p", pid); ierr != nil {
		return "", fmt.Errorf("preloader init failed: %w (original resolve error: %w)", ierr, err)
	}
	return g.resolvePidToID(pid)
}

// resolvePidToID maps a workload PID to its GPU-CR dump-buffer id.
func (g *MemoryRegions) resolvePidToID(pid string) (string, error) {
	// pid_map lives in the ctl dir: the supported preloaders write it there
	// with write(2), so it is non-empty on the ctl tmpfs. When
	// GPU_CR_CTL_PATH is unset, ctlFilesDir() already resolves to the data
	// dir, so the shared-dir configuration needs no second lookup.
	var lastErr error
	mapPath := filepath.Join(ctlFilesDir(), fmt.Sprintf("pid_map_%s", pid))
	data, err := os.ReadFile(mapPath)
	if err != nil {
		lastErr = err
	} else {
		// Strip NULs as well as whitespace: an mmap-written map file is
		// hugepage-sized with a zero-padded tail.
		id := strings.TrimSpace(strings.TrimRight(string(data), "\x00"))
		if isAllDigits(id) {
			return id, nil
		}
	}

	// Fallback for lost or damaged bookkeeping (e.g. a ctl tmpfs recreated
	// under live workloads): the dump buffer mapping is visible in
	// /proc/<pid>/maps and its basename IS the id.
	id, ferr := idFromProcMaps(g.procRoot, pid)
	if ferr != nil {
		readProblem := "contents are not a numeric id"
		if lastErr != nil {
			readProblem = lastErr.Error()
		}
		return "", fmt.Errorf("pid map for %s unusable (%s) and /proc fallback failed: %w", pid, readProblem, ferr)
	}
	slog.Info("Resolved GPU-CR id from /proc/<pid>/maps fallback", "pid", pid, "id", id)
	return id, nil
}

func isAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// idFromProcMaps scans <procRoot>/<pid>/maps for the GPU-CR dump buffer
// mapping (a file named huge-ckpt/<id>, all digits) and returns the id.
func idFromProcMaps(procRoot, pid string) (string, error) {
	data, err := os.ReadFile(filepath.Join(procRoot, pid, "maps"))
	if err != nil {
		return "", err
	}
	for _, line := range strings.Split(string(data), "\n") {
		idx := strings.IndexByte(line, '/')
		if idx < 0 {
			continue
		}
		path := line[idx:]
		if !strings.Contains(path, "huge-ckpt/") {
			continue
		}
		base := filepath.Base(path)
		if isAllDigits(base) {
			return base, nil
		}
	}
	return "", fmt.Errorf("no huge-ckpt/<id> mapping found in %s/%s/maps", procRoot, pid)
}

// opTimeout bounds a single cr_client invocation. Without it, a workload
// dying mid-operation leaves cr_client polling the shared-memory control file
// forever and the job wedged in TRANSITIONING (observed in Phase 0).
// cr_client now enforces the same deadline internally; this is
// the outer belt to its braces.
func opTimeout() time.Duration {
	if v := os.Getenv("GPU_CR_OP_TIMEOUT_SEC"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return time.Duration(n) * time.Second
		}
	}
	return 120 * time.Second
}
