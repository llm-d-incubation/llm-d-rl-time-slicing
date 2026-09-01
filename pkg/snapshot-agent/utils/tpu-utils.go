package utils

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// TPU process discovery. There is no NVML equivalent for TPUs; a process is
// "on the accelerator" when it (a) runs libtpu with checkpointing enabled —
// visible as a control thread named "libtpu{RRRRSSSS}" in /proc/<pid>/task —
// and (b) holds an open /dev/vfio/<group> fd (libtpu attaches chips through
// their vfio iommu groups). A checkpointed process keeps its control thread
// but releases its vfio fds, and a process that never initialized the TPU has
// neither, so requiring both yields exactly the RUNNING set.
//
// Discovery runs on the host (the agent DaemonSet is privileged with
// hostPID), so PIDs are host-namespace PIDs — the same namespace the
// tpucheckpoint CLI targets.

var tpuControlThreadRe = regexp.MustCompile(`^libtpu[0-9a-fA-F]{8}$`)

// procRoot is a package var so tests can point discovery at a fixture tree.
var procRoot = "/proc"

// vfioRoot is a package var so tests can point the vfio gate at a fixture
// tree.
var vfioRoot = "/dev/vfio"

// openVfioGroup probes whether a vfio iommu group is openable (i.e. released
// by its previous holder). Overridable in tests, where real group chardevs
// don't exist.
var openVfioGroup = func(path string) error {
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		return err
	}
	return f.Close()
}

// GetPodTpuPIDs returns the host PIDs of all TPU-attached processes belonging
// to the specified pod. Drop-in replacement for GetPodPIDs (assigned over it
// when the agent runs with ACCELERATOR_TYPE=tpu).
func GetPodTpuPIDs(ctx context.Context, podName, namespace string) ([]int, error) {
	podUID, err := getPodUID(ctx, podName, namespace)
	if err != nil {
		return nil, fmt.Errorf("failed to get pod UID: %w", err)
	}

	candidates, err := listTpuProcesses()
	if err != nil {
		return nil, err
	}

	var pids []int
	for _, pid := range candidates {
		inCgroup, err := IsPIDInPodCgroupInternal(fmt.Sprintf("%s/%d/cgroup", procRoot, pid), podUID)
		if err != nil || !inCgroup {
			continue
		}
		if len(vfioFdsHeld(pid)) > 0 {
			pids = append(pids, pid)
		}
	}
	return pids, nil
}

// HasTpuProcesses reports whether any process on the node is attached to the
// TPU. Drop-in replacement for HasGPUProcesses on TPU nodes.
func HasTpuProcesses(ctx context.Context) (bool, error) {
	candidates, err := listTpuProcesses()
	if err != nil {
		return false, err
	}
	for _, pid := range candidates {
		if len(vfioFdsHeld(pid)) > 0 {
			return true, nil
		}
	}
	return false, nil
}

// listTpuProcesses returns every PID with a libtpu control thread.
func listTpuProcesses() ([]int, error) {
	procs, err := os.ReadDir(procRoot)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s: %w", procRoot, err)
	}

	var pids []int
	for _, proc := range procs {
		pid, err := strconv.Atoi(proc.Name())
		if err != nil {
			continue
		}
		tasks, err := os.ReadDir(filepath.Join(procRoot, proc.Name(), "task"))
		if err != nil {
			continue // process gone or not ours to read
		}
		for _, task := range tasks {
			comm, err := os.ReadFile(filepath.Join(procRoot, proc.Name(), "task", task.Name(), "comm"))
			if err != nil {
				continue
			}
			if tpuControlThreadRe.MatchString(strings.TrimSpace(string(comm))) {
				pids = append(pids, pid)
				break
			}
		}
	}
	return pids, nil
}

// vfioFdsHeld returns the /dev/vfio/<group> paths the process holds open.
func vfioFdsHeld(pid int) []string {
	fdDir := fmt.Sprintf("%s/%d/fd", procRoot, pid)
	fds, err := os.ReadDir(fdDir)
	if err != nil {
		return nil
	}
	var held []string
	for _, fd := range fds {
		target, err := os.Readlink(filepath.Join(fdDir, fd.Name()))
		if err != nil {
			continue
		}
		if strings.HasPrefix(target, "/dev/vfio/") {
			held = append(held, target)
		}
	}
	return held
}

// VfioGroupHolders is a host-wide scan: which processes hold which
// /dev/vfio group fds, as {"/dev/vfio/N": ["pid(comm)", ...]}. Runs on the
// host (hostPID), so on a shared node this attributes holds across ALL
// time-sliced jobs — the in-band answer to "who is holding the vfio lock"
// when the restore gate waits or a checkpoint leaves groups busy. A full
// /proc fd sweep of a fat libtpu process costs O(100ms); callers rate-limit
// to every few seconds.
func VfioGroupHolders() map[string][]string {
	holders := make(map[string][]string)
	procs, err := os.ReadDir(procRoot)
	if err != nil {
		return holders
	}
	for _, proc := range procs {
		pid, err := strconv.Atoi(proc.Name())
		if err != nil {
			continue
		}
		held := vfioFdsHeld(pid)
		if len(held) == 0 {
			continue
		}
		comm := "?"
		if b, err := os.ReadFile(filepath.Join(procRoot, proc.Name(), "comm")); err == nil {
			comm = strings.TrimSpace(string(b))
		}
		seen := make(map[string]bool)
		for _, g := range held {
			if seen[g] {
				continue
			}
			seen[g] = true
			holders[g] = append(holders[g], fmt.Sprintf("%d(%s)", pid, comm))
		}
	}
	return holders
}

// WaitVfioFree blocks until every /dev/vfio iommu group on this host is
// openable, polling every 250ms up to timeout.
//
// The previous occupant's chip release lags its logical yield by seconds
// (host-side hold by the HAL server / tpu-plugin daemon). Gating here keeps
// Restore from firing into EBUSY groups — which fails AND poisons the driver
// ("Reattach failed during Restore") — and measures the release latency.
// Returns an error after timeout rather than letting Restore poison the
// driver. The gate must run ONCE per job before any process enters Restore:
// early mesh members' Restore opens their own vfio groups while parked
// waiting for siblings, so gating per-process would deadlock against the
// job's own mesh.
func WaitVfioFree(ctx context.Context, timeout time.Duration) error {
	entries, err := os.ReadDir(vfioRoot)
	if err != nil {
		slog.InfoContext(ctx, "vfio gate: cannot read vfio dir; skipping", "dir", vfioRoot, "error", err)
		return nil
	}
	var groups []string
	for _, e := range entries {
		if _, err := strconv.Atoi(e.Name()); err == nil {
			groups = append(groups, filepath.Join(vfioRoot, e.Name()))
		}
	}
	if len(groups) == 0 {
		slog.InfoContext(ctx, "vfio gate: no vfio groups visible on this host; skipping", "dir", vfioRoot)
		return nil
	}

	t0 := time.Now()
	deadline := t0.Add(timeout)
	var lastHolderLog time.Time
	for {
		var busy []string
		for _, g := range groups {
			if err := openVfioGroup(g); err != nil {
				busy = append(busy, g)
			}
		}
		if len(busy) == 0 {
			if waited := time.Since(t0); waited > 500*time.Millisecond {
				slog.InfoContext(ctx, "vfio gate: waited for group release", "waited", waited)
			}
			return nil
		}
		now := time.Now()
		// Attribute the hold while we wait (first hit, then every ~5s):
		// which PID/comm is pinning each busy group.
		if now.Sub(lastHolderLog) >= 5*time.Second {
			slog.InfoContext(ctx, "vfio gate: waiting for busy groups",
				"waited", now.Sub(t0), "busy", busy, "holders", VfioGroupHolders())
			lastHolderLog = now
		}
		if now.After(deadline) {
			return fmt.Errorf("vfio gate: groups still busy after %s: %v (holders: %v) - refusing to Restore into busy chips",
				timeout, busy, VfioGroupHolders())
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("vfio gate: %w", ctx.Err())
		case <-time.After(250 * time.Millisecond):
		}
	}
}

// ClearTpuLockfiles removes each target container's /tmp/libtpu_lockfile
// (via /proc/<pid>/root). A stale lockfile blocks the next libtpu init in
// that container. Running on the host, the container's /tmp is only
// reachable through the process's root fs view. Best-effort: failures are
// logged, never fatal.
func ClearTpuLockfiles(ctx context.Context, pids []string) {
	seen := make(map[string]bool)
	for _, pid := range pids {
		path := fmt.Sprintf("%s/%s/root/tmp/libtpu_lockfile", procRoot, pid)
		resolved, err := filepath.EvalSymlinks(path)
		if err != nil {
			resolved = path
		}
		if seen[resolved] {
			continue
		}
		seen[resolved] = true
		if _, err := os.Stat(path); err != nil {
			continue
		}
		if err := os.Remove(path); err != nil {
			slog.WarnContext(ctx, "could not remove stale libtpu lockfile", "path", path, "error", err)
			continue
		}
		slog.InfoContext(ctx, "removed stale libtpu lockfile", "path", path)
	}
}
