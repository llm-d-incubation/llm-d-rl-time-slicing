package utils

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"syscall"
	"testing"
	"time"
)

// deadPID is a PID that will not exist on any test host (max pid is
// typically 4194304 on Linux).
const deadPID = "999999999"

func hasProcfs() bool {
	_, err := os.Stat("/proc/self/stat")
	return err == nil
}

// writeAged creates a GC-candidate file, backdated past gcMinAge when aged.
func writeAged(t *testing.T, dir, name string, aged bool) {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte("x"), 0o600); err != nil {
		t.Fatalf("WriteFile(%s): %v", name, err)
	}
	if aged {
		old := time.Now().Add(-time.Hour)
		if err := os.Chtimes(path, old, old); err != nil {
			t.Fatalf("Chtimes(%s): %v", name, err)
		}
	}
}

func assertGone(t *testing.T, dir, name string) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(dir, name)); !os.IsNotExist(err) {
		t.Errorf("%s should have been removed (stat err: %v)", name, err)
	}
}

func assertKept(t *testing.T, dir, name string) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
		t.Errorf("%s should have been kept: %v", name, err)
	}
}

func TestGCFilePatterns(t *testing.T) {
	tests := []struct {
		name      string
		file      string
		dumpMatch bool
		pidMatch  bool
	}{
		{name: "dump id", file: "123", dumpMatch: true},
		{name: "host staging", file: "123-host", dumpMatch: true},
		{name: "file-backend dump", file: "ckpt-123.data", dumpMatch: true},
		{name: "file-backend staging", file: "ckpt-123-host.data", dumpMatch: true},
		{name: "file-backend prefix without suffix", file: "ckpt-123"},
		{name: "suffix without prefix", file: "123.data"},
		{name: "non-numeric file-backend id", file: "ckpt-abc.data"},
		{name: "control channel", file: "control-4567", pidMatch: true},
		{name: "pid map", file: "pid_map_4567", pidMatch: true},
		{name: "ctl readiness advertisement", file: "ctl-ready-4567", pidMatch: true},
		{name: "bare control counter untouched", file: "control"},
		{name: "unrelated file", file: "somefile.txt"},
		{name: "non-numeric id", file: "abc-host"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isDumpFile(tc.file); got != tc.dumpMatch {
				t.Errorf("isDumpFile(%q) = %v, want %v", tc.file, got, tc.dumpMatch)
			}
			if got := pidFileRe.MatchString(tc.file); got != tc.pidMatch {
				t.Errorf("pidFileRe.MatchString(%q) = %v, want %v", tc.file, got, tc.pidMatch)
			}
		})
	}
}

func TestSweep(t *testing.T) {
	ctl := t.TempDir()
	t.Setenv("EXPORT_FILE_PATH", ctl)  // keep group-store sweep inside the tempdir
	t.Setenv("GPU_CR_CTL_PATH", "")    // legacy layout: ctl files share the data dir
	t.Setenv("GPU_CR_GROUP_STORE", "") // default <data>/groups

	// Deterministic maps scan: a fixture procfs whose one process maps the
	// two "live" dumps. Scanning the real /proc would make the test depend
	// on host privileges (an unprivileged runner cannot read other users'
	// maps, which correctly marks the scan incomplete and skips deletion).
	// The maps entries are written after the files exist, from their real
	// device:inode, under pathnames from a foreign mount namespace — the
	// scan must match by inode, never by name.
	fakeProc := t.TempDir()
	pidDir := filepath.Join(fakeProc, "1")
	if err := os.MkdirAll(pidDir, 0o755); err != nil {
		t.Fatal(err)
	}
	oldRoot := procfsRoot
	procfsRoot = fakeProc
	defer func() { procfsRoot = oldRoot }()

	writeAged(t, ctl, "123", true)           // stale dump, no live mapping -> removed
	writeAged(t, ctl, "123-host", true)      // stale staging -> removed
	writeAged(t, ctl, "456", false)          // fresh dump -> kept (min-age guard)
	writeAged(t, ctl, "789", true)           // stale dump, live mapping -> kept
	writeAged(t, ctl, "ckpt-321.data", true) // stale file-backend dump -> removed
	writeAged(t, ctl, "ckpt-321-host.data", true)
	writeAged(t, ctl, "ckpt-790.data", true)      // stale file-backend dump, live mapping -> kept
	writeAged(t, ctl, "control-"+deadPID, true)   // dead pid -> removed
	writeAged(t, ctl, "pid_map_"+deadPID, true)   // dead pid -> removed
	writeAged(t, ctl, "ctl-ready-"+deadPID, true) // dead pid -> removed
	writeAged(t, ctl, "control", true)            // bare counter -> kept
	writeAged(t, ctl, "unrelated.bin", true)      // unknown file -> kept

	// The live-PID guard needs a procfs (Linux); skip that piece elsewhere.
	ownPidMap := "pid_map_" + strconv.Itoa(os.Getpid())
	if hasProcfs() {
		writeAged(t, ctl, ownPidMap, true) // live pid -> kept
	}

	mapsLines := mapsEntry(t, filepath.Join(ctl, "789"), "/workload/ns/buf0") +
		mapsEntry(t, filepath.Join(ctl, "ckpt-790.data"), "/workload/ns/buf1")
	if err := os.WriteFile(filepath.Join(pidDir, "maps"), []byte(mapsLines), 0o600); err != nil {
		t.Fatal(err)
	}

	sweep(ctl)

	assertGone(t, ctl, "123")
	assertGone(t, ctl, "123-host")
	assertGone(t, ctl, "ckpt-321.data")
	assertGone(t, ctl, "ckpt-321-host.data")
	assertGone(t, ctl, "control-"+deadPID)
	assertGone(t, ctl, "pid_map_"+deadPID)
	assertGone(t, ctl, "ctl-ready-"+deadPID)
	assertKept(t, ctl, "456")
	assertKept(t, ctl, "789")
	assertKept(t, ctl, "ckpt-790.data")
	assertKept(t, ctl, "control")
	assertKept(t, ctl, "unrelated.bin")
	if hasProcfs() {
		assertKept(t, ctl, ownPidMap)
	}
}

// TestSweepCtlDir covers the second sweep target: per-PID control-plane
// files on the ctl tmpfs, swept with the shorter ctlMinAge.
func TestSweepCtlDir(t *testing.T) {
	data := t.TempDir()
	ctl := t.TempDir()
	t.Setenv("EXPORT_FILE_PATH", data)
	t.Setenv("GPU_CR_CTL_PATH", ctl)
	t.Setenv("GPU_CR_GROUP_STORE", "")

	writeAged(t, ctl, "control-"+deadPID, true)   // dead pid -> removed
	writeAged(t, ctl, "pid_map_"+deadPID, true)   // dead pid -> removed
	writeAged(t, ctl, "ctl-ready-"+deadPID, true) // dead pid -> removed
	writeAged(t, ctl, "pid_map_"+deadPID+"0", false)
	writeAged(t, ctl, "unrelated.bin", true) // unknown file -> kept

	sweep(data)

	assertGone(t, ctl, "control-"+deadPID)
	assertGone(t, ctl, "pid_map_"+deadPID)
	assertGone(t, ctl, "ctl-ready-"+deadPID)
	assertKept(t, ctl, "pid_map_"+deadPID+"0") // fresh -> kept (ctlMinAge guard)
	assertKept(t, ctl, "unrelated.bin")
}

// TestPidGoneStarttime covers the recycled-PID guard: a ctl-ready file whose
// advertised starttime differs from the live process's is stale even though
// /proc/<pid> exists.
func TestPidGoneStarttime(t *testing.T) {
	if !hasProcfs() {
		t.Skip("no procfs on this host")
	}
	dir := t.TempDir()
	pid := strconv.Itoa(os.Getpid())
	st, err := ProcStarttime(pid)
	if err != nil {
		t.Fatal(err)
	}

	stale := filepath.Join(dir, "ctl-ready-"+pid)
	if err := os.WriteFile(stale, []byte(fmt.Sprintf("pid=%s starttime=%d\n", pid, st+1)), 0o600); err != nil {
		t.Fatal(err)
	}
	if !pidGone(pid, stale) {
		t.Error("mismatched starttime must mark the advertisement stale")
	}

	current := filepath.Join(dir, "ctl-ready-"+pid)
	if err := os.WriteFile(current, []byte(fmt.Sprintf("pid=%s starttime=%d\n", pid, st)), 0o600); err != nil {
		t.Fatal(err)
	}
	if pidGone(pid, current) {
		t.Error("matching starttime must keep the advertisement")
	}

	// Files without an advertised starttime fall back to plain liveness.
	plain := filepath.Join(dir, "control-"+pid)
	if err := os.WriteFile(plain, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if pidGone(pid, plain) {
		t.Error("live pid without advertisement must be kept")
	}
	if !pidGone(deadPID, filepath.Join(dir, "control-"+deadPID)) {
		t.Error("dead pid must be reported gone")
	}
}

func TestOwnerGone(t *testing.T) {
	if !hasProcfs() {
		t.Skip("no procfs on this host")
	}
	pid := strconv.Itoa(os.Getpid())
	st, err := ProcStarttime(pid)
	if err != nil {
		t.Fatal(err)
	}
	if ownerGone(pid, st) {
		t.Error("live owner with matching starttime must count as alive")
	}
	if !ownerGone(pid, st+1) {
		t.Error("recycled pid (starttime mismatch) must count as gone")
	}
	if !ownerGone("999999999", st) {
		t.Error("nonexistent pid must count as gone")
	}
}

func TestAdvertisedStarttime(t *testing.T) {
	dir := t.TempDir()

	good := filepath.Join(dir, "ctl-ready-1")
	if err := os.WriteFile(good, []byte("pid=1 starttime=42 uid=0\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	st, ok := advertisedStarttime(good)
	if !ok || st != 42 {
		t.Errorf("advertisedStarttime() = %d, %v, want 42, true", st, ok)
	}

	bad := filepath.Join(dir, "ctl-ready-2")
	if err := os.WriteFile(bad, []byte("no starttime here\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, ok := advertisedStarttime(bad); ok {
		t.Error("file without starttime must not parse")
	}

	if _, ok := advertisedStarttime(filepath.Join(dir, "missing")); ok {
		t.Error("missing file must not parse")
	}
}

// TestSweepGroupStore covers the owner-liveness reap of destination slots.
// Ownership is path-encoded in <pid>-<starttime> subdirectory names:
// deletion requires every so-recorded owner dead (or recycled) AND the
// grace period passed; slots whose ownership cannot be fully parsed from
// their subdirectories — including pre-owner-dir legacy slots with flat
// dump files — are never deleted.
func TestSweepGroupStore(t *testing.T) {
	data := t.TempDir()
	t.Setenv("EXPORT_FILE_PATH", data)
	t.Setenv("GPU_CR_CTL_PATH", "")
	t.Setenv("GPU_CR_GROUP_STORE", "")
	t.Setenv("GPU_CR_GROUP_GRACE_HOURS", "")
	store := filepath.Join(data, "groups")

	// makeGroup creates a slot whose children are owner dirs (each holding
	// one dump file) and/or flat files, then optionally backdates the slot
	// past the default 1h grace.
	makeGroup := func(name string, ownerDirs, files []string, aged bool) {
		t.Helper()
		dir := filepath.Join(store, name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		for _, od := range ownerDirs {
			if err := os.MkdirAll(filepath.Join(dir, od), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, od, "17"), []byte("dump"), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		for _, f := range files {
			if err := os.WriteFile(filepath.Join(dir, f), []byte("x"), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		if aged {
			old := time.Now().Add(-2 * time.Hour) // default grace is 1h
			if err := os.Chtimes(dir, old, old); err != nil {
				t.Fatal(err)
			}
		}
	}

	makeGroup("dead-owner", []string{deadPID + "-123"}, nil, true) // all owners gone -> removed
	makeGroup("two-dead-owners", []string{deadPID + "-123", deadPID + "1-99"}, nil, true)
	makeGroup("empty", nil, nil, true)                                // no owner dirs -> NEVER removed
	makeGroup("fresh", []string{deadPID + "-123"}, nil, false)        // inside grace -> kept
	makeGroup("garbage-dir", []string{"not-an-owner"}, nil, true)     // unparseable dir -> kept
	makeGroup("mixed", []string{deadPID + "-123", "junk"}, nil, true) // partial parse -> kept
	makeGroup("legacy-flat", nil, []string{"17", ".owners"}, true)    // pre-owner-dir slot -> kept
	if hasProcfs() {
		pid := strconv.Itoa(os.Getpid())
		st, err := ProcStarttime(pid)
		if err != nil {
			t.Fatal(err)
		}
		// live owner -> kept
		makeGroup("live-owner", []string{fmt.Sprintf("%s-%d", pid, st)}, nil, true)
		// pid alive but starttime mismatched = recycled -> removed
		makeGroup("recycled-owner", []string{fmt.Sprintf("%s-%d", pid, st+1)}, nil, true)
	}

	sweepGroupStore(time.Now())

	assertGone(t, store, "dead-owner")
	assertGone(t, store, "two-dead-owners")
	assertKept(t, store, "empty")
	assertKept(t, store, "fresh")
	assertKept(t, store, "garbage-dir")
	assertKept(t, store, "mixed")
	assertKept(t, store, "legacy-flat")
	if hasProcfs() {
		assertKept(t, store, "live-owner")
		assertGone(t, store, "recycled-owner")
	}
}

func TestSlotOwners(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "123-456"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "789-42"), 0o755); err != nil {
		t.Fatal(err)
	}
	// Flat files are not ownership records and must be skipped, not fail
	// the parse.
	if err := os.WriteFile(filepath.Join(dir, "17"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	owners, ok := slotOwners(dir)
	if !ok || len(owners) != 2 || owners["123"] != 456 || owners["789"] != 42 {
		t.Errorf("slotOwners() = %v, %v; want both owners parsed", owners, ok)
	}

	// One unparseable SUBDIRECTORY poisons the whole slot: ownership is
	// not fully known, so there is no license to delete.
	if err := os.MkdirAll(filepath.Join(dir, "not-an-owner"), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, ok := slotOwners(dir); ok {
		t.Error("slotOwners() must report unknown ownership on an unparseable subdirectory")
	}
}

// TestSweepIncompleteProcScan: when a maps file is unreadable for a reason
// other than process exit, the scan cannot prove any dump is unmapped, so
// dump deletion is skipped; per-PID files are still reaped (their liveness
// check does not depend on the maps scan).
func TestSweepIncompleteProcScan(t *testing.T) {
	ctl := t.TempDir()
	t.Setenv("EXPORT_FILE_PATH", ctl)
	t.Setenv("GPU_CR_CTL_PATH", "")
	t.Setenv("GPU_CR_GROUP_STORE", "")

	fakeProc := t.TempDir()
	pidDir := filepath.Join(fakeProc, "42")
	if err := os.MkdirAll(pidDir, 0o755); err != nil {
		t.Fatal(err)
	}
	// A self-referencing symlink makes ReadFile fail with ELOOP — neither
	// IsNotExist nor ESRCH, so the scan must report incomplete.
	loop := filepath.Join(pidDir, "maps")
	if err := os.Symlink(loop, loop); err != nil {
		t.Skipf("cannot create symlink loop: %v", err)
	}
	oldRoot := procfsRoot
	procfsRoot = fakeProc
	defer func() { procfsRoot = oldRoot }()

	if _, complete := liveMappedInodes(); complete {
		t.Fatal("liveMappedInodes() reported a complete scan despite an unreadable maps file")
	}

	writeAged(t, ctl, "123", true)              // stale dump, but scan incomplete -> kept
	writeAged(t, ctl, "ckpt-123.data", true)    // file-backend dump, same protection -> kept
	writeAged(t, ctl, "control-"+deadPID, true) // dead pid -> still removed

	sweep(ctl)

	assertKept(t, ctl, "123")
	assertKept(t, ctl, "ckpt-123.data")
	assertGone(t, ctl, "control-"+deadPID)
}

// TestLiveMappedInodesExitedProcess: a numeric dir with no maps file is a
// process that exited between enumeration and read — benign churn, the
// scan stays complete.
func TestLiveMappedInodesExitedProcess(t *testing.T) {
	fakeProc := t.TempDir()
	if err := os.MkdirAll(filepath.Join(fakeProc, "7"), 0o755); err != nil {
		t.Fatal(err)
	}
	oldRoot := procfsRoot
	procfsRoot = fakeProc
	defer func() { procfsRoot = oldRoot }()

	if _, complete := liveMappedInodes(); !complete {
		t.Error("missing maps file (exited process) must not mark the scan incomplete")
	}
}

// mapsEntry renders a /proc/<pid>/maps line for path from its real
// device:inode, under a pathname the agent has never mounted.
func mapsEntry(t *testing.T, path, nsPath string) string {
	t.Helper()
	var st syscall.Stat_t
	if err := syscall.Stat(path, &st); err != nil {
		t.Fatal(err)
	}
	nums := devMajMin(st.Dev)
	return fmt.Sprintf("7f0000000000-7f0000200000 rw-s 00000000 %02x:%02x %d %s\n",
		nums.major, nums.minor, st.Ino, nsPath)
}
