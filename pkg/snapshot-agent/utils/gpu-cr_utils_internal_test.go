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

	old := time.Now().Add(-time.Hour)

	writeAged := func(name string, aged bool) {
		t.Helper()
		path := filepath.Join(ctl, name)
		if err := os.WriteFile(path, []byte("x"), 0o600); err != nil {
			t.Fatalf("WriteFile(%s): %v", name, err)
		}
		if aged {
			if err := os.Chtimes(path, old, old); err != nil {
				t.Fatalf("Chtimes(%s): %v", name, err)
			}
		}
	}

	writeAged("123", true)           // stale dump, no live mapping -> removed
	writeAged("123-host", true)      // stale staging -> removed
	writeAged("456", false)          // fresh dump -> kept (min-age guard)
	writeAged("789", true)           // stale dump, live mapping -> kept
	writeAged("ckpt-321.data", true) // stale file-backend dump -> removed
	writeAged("ckpt-321-host.data", true)
	writeAged("ckpt-790.data", true)    // stale file-backend dump, live mapping -> kept
	writeAged("control-"+deadPID, true) // dead pid -> removed
	writeAged("pid_map_"+deadPID, true) // dead pid -> removed
	writeAged("control", true)          // bare counter -> kept
	writeAged("unrelated.bin", true)    // unknown file -> kept

	// The live-PID guard needs a procfs (Linux); skip that piece elsewhere.
	_, procErr := os.Stat("/proc/self")
	ownPidMap := "pid_map_" + strconv.Itoa(os.Getpid())
	if procErr == nil {
		writeAged(ownPidMap, true) // live pid -> kept
	}

	mapsLines := mapsEntry(t, filepath.Join(ctl, "789"), "/workload/ns/buf0") +
		mapsEntry(t, filepath.Join(ctl, "ckpt-790.data"), "/workload/ns/buf1")
	if err := os.WriteFile(filepath.Join(pidDir, "maps"), []byte(mapsLines), 0o600); err != nil {
		t.Fatal(err)
	}

	sweep(ctl)

	assertGone := func(name string) {
		t.Helper()
		if _, err := os.Stat(filepath.Join(ctl, name)); !os.IsNotExist(err) {
			t.Errorf("%s should have been removed (stat err: %v)", name, err)
		}
	}
	assertKept := func(name string) {
		t.Helper()
		if _, err := os.Stat(filepath.Join(ctl, name)); err != nil {
			t.Errorf("%s should have been kept: %v", name, err)
		}
	}

	assertGone("123")
	assertGone("123-host")
	assertGone("ckpt-321.data")
	assertGone("ckpt-321-host.data")
	assertGone("control-" + deadPID)
	assertGone("pid_map_" + deadPID)
	assertKept("456")
	assertKept("789")
	assertKept("ckpt-790.data")
	assertKept("control")
	assertKept("unrelated.bin")
	if procErr == nil {
		assertKept(ownPidMap)
	}
}

// TestSweepIncompleteProcScan: when a maps file is unreadable for a reason
// other than process exit, the scan cannot prove any dump is unmapped, so
// dump deletion is skipped; per-PID files are still reaped (their liveness
// check does not depend on the maps scan).
func TestSweepIncompleteProcScan(t *testing.T) {
	ctl := t.TempDir()
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

	old := time.Now().Add(-time.Hour)
	write := func(name string) {
		t.Helper()
		path := filepath.Join(ctl, name)
		if err := os.WriteFile(path, []byte("x"), 0o600); err != nil {
			t.Fatalf("WriteFile(%s): %v", name, err)
		}
		if err := os.Chtimes(path, old, old); err != nil {
			t.Fatalf("Chtimes(%s): %v", name, err)
		}
	}
	write("123")           // stale dump, but scan incomplete -> kept
	write("ckpt-123.data") // file-backend dump, same protection -> kept
	write("control-" + deadPID)

	sweep(ctl)

	if _, err := os.Stat(filepath.Join(ctl, "123")); err != nil {
		t.Errorf("dump should have been kept on incomplete scan: %v", err)
	}
	if _, err := os.Stat(filepath.Join(ctl, "ckpt-123.data")); err != nil {
		t.Errorf("file-backend dump should have been kept on incomplete scan: %v", err)
	}
	if _, err := os.Stat(filepath.Join(ctl, "control-"+deadPID)); !os.IsNotExist(err) {
		t.Errorf("dead-pid control file should still be removed (stat err: %v)", err)
	}
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
