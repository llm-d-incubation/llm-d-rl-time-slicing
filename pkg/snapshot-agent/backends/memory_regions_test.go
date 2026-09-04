package backends_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/utils"
)

func region(pid int32, addr, size uint64) *pb.MemoryRegion {
	return &pb.MemoryRegion{Pid: pid, Address: addr, SizeBytes: size}
}

func memoryRegionsConfig(slot string, regions ...*pb.MemoryRegion) *pb.BackendConfig {
	return &pb.BackendConfig{
		Backend: &pb.BackendConfig_MemoryRegions{
			MemoryRegions: &pb.MemoryRegionsBackendConfig{
				Regions:      regions,
				SnapshotName: slot,
			},
		},
	}
}

// testStarttime is the starttime the test seam reports for every pid, so
// expected owner dirnames (<pid>-<testStarttime>) are deterministic.
const testStarttime = int64(555)

// newMemoryRegions returns a backend pointed at a tempdir layout via env.
// Destination-slot dumps land under the returned store dir (the -o paths),
// inside per-owner <pid>-<starttime> dirs with a pinned fake starttime
// (test pids have no procfs entry).
//
//nolint:gocritic // The project configuration bans named returns, conflicting with unnamedResult
func newMemoryRegions(t *testing.T) (*backends.MemoryRegions, string, string) {
	t.Helper()
	ctlDir := t.TempDir()
	t.Setenv("EXPORT_FILE_PATH", ctlDir)
	t.Setenv("GPU_CR_CTL_PATH", "")
	t.Setenv("GPU_CR_GROUP_STORE", "")
	mr := backends.NewMemoryRegions()
	mr.SetStarttimeFunc(func(string) (int64, error) { return testStarttime, nil })
	return mr, ctlDir, filepath.Join(ctlDir, "groups")
}

// writePidMap maps a PID to its dump-buffer id in the ctl dir, as the
// workload's preloader does at startup.
func writePidMap(t *testing.T, dir, pid, id string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "pid_map_"+pid), []byte(id+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
}

// expandStore substitutes the per-test group-store dir into expected args
// ("<store>" placeholder), so tables can spell out full -o paths.
func expandStore(args [][]string, store string) [][]string {
	out := make([][]string, len(args))
	for i, call := range args {
		out[i] = make([]string, len(call))
		for j, a := range call {
			out[i][j] = strings.ReplaceAll(a, "<store>", store)
		}
	}
	return out
}

func TestNewMemoryRegions(t *testing.T) {
	mr := backends.NewMemoryRegions()
	if mr == nil {
		t.Fatal("NewMemoryRegions returned nil")
	}
}

func TestMemoryRegionsSnapshot(t *testing.T) {
	tests := []struct {
		name        string
		config      *pb.BackendConfig
		jobID       string
		execErr     error
		expectedErr bool
		expectNoRun bool       // cr_client must not be invoked at all
		expectArgs  [][]string // checked on success; "<store>" expands to the group-store dir
	}{
		{
			name:   "SingleRegion",
			config: memoryRegionsConfig("slot-a", region(123, 0x7f00, 1024)),
			jobID:  "test-job",
			expectArgs: [][]string{
				{"-c", "-p", "123", "-s", "0x7f00:1024", "-o", "<store>/slot-a/123-555/42"},
			},
		},
		{
			name:   "RegionsOfOnePIDJoined",
			config: memoryRegionsConfig("slot-a", region(123, 0x7f00, 1024), region(123, 0x8f00, 2048)),
			jobID:  "test-job",
			expectArgs: [][]string{
				{"-c", "-p", "123", "-s", "0x7f00:1024,0x8f00:2048", "-o", "<store>/slot-a/123-555/42"},
			},
		},
		{
			name:   "AddressFormattedAsHex",
			config: memoryRegionsConfig("slot-a", region(123, 139637976727552, 1073741824)),
			jobID:  "test-job",
			expectArgs: [][]string{
				{"-c", "-p", "123", "-s", "0x7f0000000000:1073741824", "-o", "<store>/slot-a/123-555/42"},
			},
		},
		{
			name:   "EmptySnapshotNameFallsBackToJobID",
			config: memoryRegionsConfig("", region(123, 0x7f00, 1024)),
			jobID:  "job-1",
			expectArgs: [][]string{
				{"-c", "-p", "123", "-s", "0x7f00:1024", "-o", "<store>/job-1/123-555/42"},
			},
		},
		{
			name:        "NilConfig",
			config:      nil,
			jobID:       "test-job",
			expectedErr: true,
			expectNoRun: true,
		},
		{
			name:        "NoRegions",
			config:      memoryRegionsConfig("slot-a"),
			jobID:       "test-job",
			expectedErr: true,
			expectNoRun: true,
		},
		{
			name:        "ZeroPID",
			config:      memoryRegionsConfig("slot-a", region(0, 0x7f00, 1024)),
			jobID:       "test-job",
			expectedErr: true,
			expectNoRun: true,
		},
		{
			name:        "NegativePID",
			config:      memoryRegionsConfig("slot-a", region(-5, 0x7f00, 1024)),
			jobID:       "test-job",
			expectedErr: true,
			expectNoRun: true,
		},
		{
			name:        "ZeroSize",
			config:      memoryRegionsConfig("slot-a", region(123, 0x7f00, 0)),
			jobID:       "test-job",
			expectedErr: true,
			expectNoRun: true,
		},
		{
			name:        "PathTraversalSnapshotName",
			config:      memoryRegionsConfig("../../etc", region(123, 0x7f00, 1024)),
			jobID:       "test-job",
			expectedErr: true,
			expectNoRun: true,
		},
		{
			name:        "NestedSnapshotName",
			config:      memoryRegionsConfig("job/slot-a", region(123, 0x7f00, 1024)),
			jobID:       "test-job",
			expectedErr: true,
			expectNoRun: true,
		},
		{
			name:        "ExecFailure",
			config:      memoryRegionsConfig("slot-a", region(123, 0x7f00, 1024)),
			jobID:       "test-job",
			execErr:     fmt.Errorf("exec error"),
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mr, ctlDir, storeDir := newMemoryRegions(t)
			writePidMap(t, ctlDir, "123", "42")
			var calledArgs [][]string
			mr.SetExecCommand(func(_ context.Context, name string, args ...string) ([]byte, error) {
				if name != backends.CrClientPath {
					t.Errorf("exec binary = %q, want %q", name, backends.CrClientPath)
				}
				calledArgs = append(calledArgs, args)
				return nil, tt.execErr
			})

			err := mr.Snapshot(context.Background(), backends.Request{JobID: tt.jobID, Config: tt.config})
			if (err != nil) != tt.expectedErr {
				t.Errorf("Snapshot() error = %v, expectedErr %v", err, tt.expectedErr)
			}
			if tt.expectNoRun && len(calledArgs) != 0 {
				t.Errorf("Snapshot() invoked cr_client with %v despite invalid request", calledArgs)
			}
			if !tt.expectedErr {
				want := expandStore(tt.expectArgs, storeDir)
				if !reflect.DeepEqual(calledArgs, want) {
					t.Errorf("Snapshot() calledArgs = %v, expected %v", calledArgs, want)
				}
				// The slot dir must exist for the preloader to dump into.
				if _, err := os.Stat(filepath.Dir(want[0][len(want[0])-1])); err != nil {
					t.Errorf("slot dir not created: %v", err)
				}
			}
		})
	}
}

func TestMemoryRegionsRestore(t *testing.T) {
	tests := []struct {
		name        string
		config      *pb.BackendConfig
		makeSlot    string // "": no slot; "empty": bare slot dir; "owned": owner dir + dump
		execErr     error
		expectedErr bool
		expectNoRun bool
		expectArgs  [][]string
	}{
		{
			name:     "Success",
			config:   memoryRegionsConfig("slot-a", region(123, 0x7f00, 1024)),
			makeSlot: "owned",
			expectArgs: [][]string{
				{"-r", "-p", "123", "-s", "0x7f00:1024", "-o", "<store>/slot-a/123-555/42"},
			},
		},
		{
			name:        "MissingSlot",
			config:      memoryRegionsConfig("no-such-slot", region(123, 0x7f00, 1024)),
			expectedErr: true,
			expectNoRun: true,
		},
		{
			// A slot that exists but holds nothing for this pid+starttime:
			// a different (e.g. restarted) process must find nothing.
			name:        "SlotWithoutThisOwner",
			config:      memoryRegionsConfig("slot-a", region(123, 0x7f00, 1024)),
			makeSlot:    "empty",
			expectedErr: true,
			expectNoRun: true,
		},
		{
			name:        "NilConfig",
			config:      nil,
			expectedErr: true,
			expectNoRun: true,
		},
		{
			name:        "ExecFailure",
			config:      memoryRegionsConfig("slot-a", region(123, 0x7f00, 1024)),
			makeSlot:    "owned",
			execErr:     fmt.Errorf("exec error"),
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mr, ctlDir, storeDir := newMemoryRegions(t)
			writePidMap(t, ctlDir, "123", "42")
			switch tt.makeSlot {
			case "empty":
				if err := os.MkdirAll(filepath.Join(storeDir, "slot-a"), 0o755); err != nil {
					t.Fatal(err)
				}
			case "owned":
				ownerDir := filepath.Join(storeDir, "slot-a", "123-555")
				if err := os.MkdirAll(ownerDir, 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(ownerDir, "42"), []byte("dump"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			var calledArgs [][]string
			mr.SetExecCommand(func(_ context.Context, name string, args ...string) ([]byte, error) {
				if name != backends.CrClientPath {
					t.Errorf("exec binary = %q, want %q", name, backends.CrClientPath)
				}
				calledArgs = append(calledArgs, args)
				return nil, tt.execErr
			})

			err := mr.Restore(context.Background(), backends.Request{JobID: "test-job", Config: tt.config})
			if (err != nil) != tt.expectedErr {
				t.Errorf("Restore() error = %v, expectedErr %v", err, tt.expectedErr)
			}
			if tt.expectNoRun && len(calledArgs) != 0 {
				t.Errorf("Restore() invoked cr_client with %v despite invalid request", calledArgs)
			}
			if !tt.expectedErr {
				want := expandStore(tt.expectArgs, storeDir)
				if !reflect.DeepEqual(calledArgs, want) {
					t.Errorf("Restore() calledArgs = %v, expected %v", calledArgs, want)
				}
			}
		})
	}
}

// TestMemoryRegionsLazyInit covers the pid->id fallback: destination-path
// ops need the dump-buffer id before the first cr_client signal, but the
// preloader only writes pid_map inside init_CR, so an unresolvable PID is
// driven through cr_client -i (idempotent) and re-resolved.
func TestMemoryRegionsLazyInit(t *testing.T) {
	tests := []struct {
		name           string
		execErr        error
		writeMapOnInit bool
		wantErr        string // substring the error must contain; unset means success
		expectArgs     [][]string
	}{
		{
			name:           "InitThenResolve",
			writeMapOnInit: true,
			expectArgs: [][]string{
				{"-i", "-p", "123"},
				{"-c", "-p", "123", "-s", "0x7f00:1024", "-o", "<store>/slot-a/123-555/42"},
			},
		},
		{
			name:       "InitExecFailureSurfacesBothErrors",
			execErr:    fmt.Errorf("no such process"),
			wantErr:    "preloader init failed",
			expectArgs: [][]string{{"-i", "-p", "123"}},
		},
		{
			name:       "StillUnresolvableAfterInit",
			wantErr:    "failed to resolve PID 123",
			expectArgs: [][]string{{"-i", "-p", "123"}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mr, ctlDir, storeDir := newMemoryRegions(t)
			mr.SetProcRoot(t.TempDir()) // no /proc/<pid>/maps fallback either
			var calledArgs [][]string
			mr.SetExecCommand(func(_ context.Context, _ string, args ...string) ([]byte, error) {
				calledArgs = append(calledArgs, args)
				if tt.writeMapOnInit && args[0] == "-i" {
					writePidMap(t, ctlDir, "123", "42")
				}
				return nil, tt.execErr
			})

			err := mr.Snapshot(context.Background(), backends.Request{
				JobID:  "test-job",
				Config: memoryRegionsConfig("slot-a", region(123, 0x7f00, 1024)),
			})
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("Snapshot() unexpected error: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("Snapshot() error = %v, want substring %q", err, tt.wantErr)
			}
			want := expandStore(tt.expectArgs, storeDir)
			if !reflect.DeepEqual(calledArgs, want) {
				t.Errorf("Snapshot() calledArgs = %v, expected %v", calledArgs, want)
			}
		})
	}
}

// TestMemoryRegionsPidResolution covers the pid_map read paths, observable
// through the id in the -o destination the backend hands cr_client.
func TestMemoryRegionsPidResolution(t *testing.T) {
	tests := []struct {
		name   string
		setup  func(t *testing.T, mr *backends.MemoryRegions, ctlDir string)
		wantID string
	}{
		{
			name: "NULPaddedPidMap", // an mmap-written map file has a zero-padded tail
			setup: func(t *testing.T, _ *backends.MemoryRegions, ctlDir string) {
				t.Helper()
				content := append([]byte("77"), make([]byte, 30)...)
				if err := os.WriteFile(filepath.Join(ctlDir, "pid_map_123"), content, 0o600); err != nil {
					t.Fatal(err)
				}
			},
			wantID: "77",
		},
		{
			// The data dir is not a pid_map location: with the ctl tmpfs
			// configured, a map file on the data mount — even a valid-looking
			// one — must be ignored, not read as a fallback.
			name: "DataDirMapIgnoredWhenCtlTmpfsSet",
			setup: func(t *testing.T, _ *backends.MemoryRegions, ctlDir string) {
				t.Helper()
				tmpfsDir := t.TempDir()
				t.Setenv("GPU_CR_CTL_PATH", tmpfsDir)
				writePidMap(t, ctlDir, "123", "55") // stale map on the data mount
				writePidMap(t, tmpfsDir, "123", "91")
			},
			wantID: "91",
		},
		{
			name: "EmptyPidMapFallsBackToProcMaps", // lost bookkeeping, mapping is kernel truth
			setup: func(t *testing.T, mr *backends.MemoryRegions, ctlDir string) {
				t.Helper()
				if err := os.WriteFile(filepath.Join(ctlDir, "pid_map_123"), nil, 0o600); err != nil {
					t.Fatal(err)
				}
				procRoot := t.TempDir()
				mr.SetProcRoot(procRoot)
				if err := os.MkdirAll(filepath.Join(procRoot, "123"), 0o755); err != nil {
					t.Fatal(err)
				}
				maps := "7f0000000000-7f0040000000 rw-s 00000000 00:0f 12345 /var/tmp/huge-ckpt/88\n"
				if err := os.WriteFile(filepath.Join(procRoot, "123", "maps"), []byte(maps), 0o600); err != nil {
					t.Fatal(err)
				}
			},
			wantID: "88",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mr, ctlDir, storeDir := newMemoryRegions(t)
			tt.setup(t, mr, ctlDir)
			var calledArgs [][]string
			mr.SetExecCommand(func(_ context.Context, _ string, args ...string) ([]byte, error) {
				calledArgs = append(calledArgs, args)
				return nil, nil
			})

			err := mr.Snapshot(context.Background(), backends.Request{
				JobID:  "test-job",
				Config: memoryRegionsConfig("slot-a", region(123, 0x7f00, 1024)),
			})
			if err != nil {
				t.Fatalf("Snapshot() unexpected error: %v", err)
			}
			want := [][]string{{"-c", "-p", "123", "-s", "0x7f00:1024", "-o", filepath.Join(storeDir, "slot-a", "123-555", tt.wantID)}}
			if !reflect.DeepEqual(calledArgs, want) {
				t.Errorf("Snapshot() calledArgs = %v, expected %v", calledArgs, want)
			}
		})
	}
}

// TestMemoryRegionsSnapshotOwnerDir verifies the path-encoded ownership
// record GC's owner-liveness sweep relies on: the dump lands inside a
// <pid>-<starttime> dir whose starttime is the real procfs value. Needs a
// real procfs, so it uses the test process itself as the owner and the
// production starttime reader.
func TestMemoryRegionsSnapshotOwnerDir(t *testing.T) {
	if _, err := os.Stat("/proc/self/stat"); err != nil {
		t.Skip("no procfs on this host")
	}
	mr, ctlDir, storeDir := newMemoryRegions(t)
	mr.SetStarttimeFunc(utils.ProcStarttime)
	mr.SetExecCommand(func(context.Context, string, ...string) ([]byte, error) { return nil, nil })
	pid := strconv.Itoa(os.Getpid())
	writePidMap(t, ctlDir, pid, "42")

	pidNum, err := strconv.ParseInt(pid, 10, 32)
	if err != nil {
		t.Fatal(err)
	}
	err = mr.Snapshot(context.Background(), backends.Request{
		JobID:  "test-job",
		Config: memoryRegionsConfig("slot-a", region(int32(pidNum), 0x7f00, 1024)),
	})
	if err != nil {
		t.Fatalf("Snapshot() unexpected error: %v", err)
	}

	st, err := utils.ProcStarttime(pid)
	if err != nil {
		t.Fatal(err)
	}
	ownerDir := filepath.Join(storeDir, "slot-a", fmt.Sprintf("%s-%d", pid, st))
	if _, err := os.Stat(ownerDir); err != nil {
		t.Fatalf("owner dir %s not created: %v", ownerDir, err)
	}
}

// A failed starttime read means the owner dir — the slot's ownership record
// — cannot be named, so the snapshot must fail before any checkpoint runs.
func TestMemoryRegionsSnapshotStarttimeFailure(t *testing.T) {
	mr, ctlDir, _ := newMemoryRegions(t)
	mr.SetStarttimeFunc(func(pid string) (int64, error) {
		return 0, fmt.Errorf("no such process %s", pid)
	})
	writePidMap(t, ctlDir, "123", "42")
	var calls int
	mr.SetExecCommand(func(context.Context, string, ...string) ([]byte, error) {
		calls++
		return nil, nil
	})

	err := mr.Snapshot(context.Background(), backends.Request{
		JobID:  "test-job",
		Config: memoryRegionsConfig("slot-a", region(123, 0x7f00, 1024)),
	})
	if err == nil || !strings.Contains(err.Error(), "starttime of owner pid 123") {
		t.Fatalf("Snapshot() error = %v, want starttime failure", err)
	}
	if calls != 0 {
		t.Errorf("cr_client invoked %d times despite unrecordable owner", calls)
	}
}

func TestMemoryRegionsSnapshotPartialFailureKeepsOwners(t *testing.T) {
	mr, ctlDir, storeDir := newMemoryRegions(t)
	const pidA, pidB = "123", "456"
	writePidMap(t, ctlDir, pidA, "42")
	writePidMap(t, ctlDir, pidB, "43")
	// The second PID's checkpoint fails after the first one succeeded.
	mr.SetExecCommand(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		if len(args) > 2 && args[0] == "-c" && args[2] == pidB {
			return []byte("injected failure"), fmt.Errorf("boom")
		}
		return nil, nil
	})

	err := mr.Snapshot(context.Background(), backends.Request{
		JobID: "test-job",
		Config: memoryRegionsConfig("slot-partial",
			region(123, 0x1000, 64), region(456, 0x2000, 64)),
	})
	if err == nil {
		t.Fatal("Snapshot() must fail when a later PID's checkpoint fails")
	}

	// The slot holds PID A's dump even though the request failed. Every
	// dump's owner dir is created before its checkpoint runs — it is the
	// sweeper's only license to reclaim the slot once the owners die;
	// without it a failed multi-PID snapshot would pin its dump pages
	// forever.
	for _, pid := range []string{pidA, pidB} {
		ownerDir := filepath.Join(storeDir, "slot-partial", fmt.Sprintf("%s-%d", pid, testStarttime))
		if _, err := os.Stat(ownerDir); err != nil {
			t.Errorf("owner dir for pid %s missing after partial failure: %v", pid, err)
		}
	}
}

func TestMemoryRegionsHealthCheck(t *testing.T) {
	tests := []struct {
		name        string
		statErr     error
		expectedErr bool
	}{
		{
			name:        "Installed",
			statErr:     nil,
			expectedErr: false,
		},
		{
			name:        "Missing",
			statErr:     fmt.Errorf("no stat"),
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mr := backends.NewMemoryRegions()
			mr.SetStatFunc(func(path string) (os.FileInfo, error) {
				if path != backends.CrClientPath {
					t.Errorf("stat path = %q, want %q", path, backends.CrClientPath)
				}
				if tt.statErr != nil {
					return nil, tt.statErr
				}
				return os.Stat(".")
			})

			err := mr.HealthCheck(context.Background())
			if (err != nil) != tt.expectedErr {
				t.Errorf("HealthCheck() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}

func TestMemoryRegionsOpTimeout(t *testing.T) {
	t.Setenv("GPU_CR_OP_TIMEOUT_SEC", "1")

	mr, ctlDir, storeDir := newMemoryRegions(t)
	writePidMap(t, ctlDir, "123", "42")
	if err := os.MkdirAll(filepath.Join(storeDir, "slot-a"), 0o755); err != nil {
		t.Fatal(err)
	}
	mr.SetExecCommand(func(ctx context.Context, _ string, _ ...string) ([]byte, error) {
		// Simulate cr_client hanging on a dead workload's control channel:
		// block until the per-operation deadline cancels the context.
		<-ctx.Done()
		return nil, ctx.Err()
	})

	req := backends.Request{JobID: "test-job", Config: memoryRegionsConfig("slot-a", region(123, 0x7f00, 1024))}
	start := time.Now()
	if err := mr.Snapshot(context.Background(), req); err == nil {
		t.Fatal("Snapshot() expected timeout error, got nil")
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("Snapshot() took %v; the 1s GPU_CR_OP_TIMEOUT_SEC deadline was not applied", elapsed)
	}

	if err := mr.Restore(context.Background(), req); err == nil {
		t.Fatal("Restore() expected timeout error, got nil")
	}
}
