package backends_test

import (
	"context"
	"fmt"
	"os"
	"slices"
	"sync"
	"testing"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
)

func tpuConfig(pids ...int32) *pb.BackendConfig {
	return &pb.BackendConfig{
		Backend: &pb.BackendConfig_Tpu{
			Tpu: &pb.TpuBackendConfig{
				ExplicitTarget: &pb.ProcessTarget{Pids: pids},
			},
		},
	}
}

// newTestTpuCheckpoint returns a backend with the vfio gate and lockfile
// cleanup stubbed out and no retry backoff, plus a thread-safe recorder of
// CLI invocations (the backend batches all PIDs into one invocation).
func newTestTpuCheckpoint(execErr error) (*backends.TpuCheckpoint, *invocationLog) {
	c := backends.NewTpuCheckpoint()
	c.SetRetryBackoff(0)
	c.SetWaitVfioFree(func(context.Context) error { return nil })
	c.SetClearLocks(func(context.Context, []string) {})
	log := &invocationLog{}
	c.SetExecCommand(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		log.record(args)
		return nil, execErr
	})
	return c, log
}

type invocationLog struct {
	mu    sync.Mutex
	calls [][]string
}

func (l *invocationLog) record(args []string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.calls = append(l.calls, slices.Clone(args))
}

func (l *invocationLog) count() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return len(l.calls)
}

func (l *invocationLog) sorted() [][]string {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := slices.Clone(l.calls)
	slices.SortFunc(out, slices.Compare)
	return out
}

func TestNewTpuCheckpoint(t *testing.T) {
	if backends.NewTpuCheckpoint() == nil {
		t.Fatal("NewTpuCheckpoint returned nil")
	}
}

func TestTpuSnapshot(t *testing.T) {
	tests := []struct {
		name        string
		config      *pb.BackendConfig
		execErr     error
		expectedErr bool
		// The backend retries a failed checkpoint once; a failed restore
		// is never retried (see TestTpuRestore).
		wantCalls int
	}{
		{
			name:      "Success",
			config:    tpuConfig(123, 456),
			wantCalls: 1, // all PIDs batched into one invocation
		},
		{
			name:        "ExecFailure",
			config:      tpuConfig(123),
			execErr:     fmt.Errorf("exec error"),
			expectedErr: true,
			wantCalls:   2, // one attempt + one retry
		},
		{
			name:        "NoPIDs",
			config:      tpuConfig(),
			expectedErr: true,
		},
		{
			name:        "NilConfig",
			config:      nil,
			expectedErr: true,
		},
		{
			name:        "CudaConfigRejected",
			config:      cudaConfig(123),
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, log := newTestTpuCheckpoint(tt.execErr)

			err := c.Snapshot(context.Background(), backends.Request{JobID: "test-job", Config: tt.config})
			if (err != nil) != tt.expectedErr {
				t.Errorf("Snapshot() error = %v, expectedErr %v", err, tt.expectedErr)
			}
			if log.count() != tt.wantCalls {
				t.Errorf("Snapshot() invoked the CLI %d times, want %d", log.count(), tt.wantCalls)
			}
		})
	}
}

// TestTpuCliArgs pins the CLI contract: the gVisor tpucheckpoint CLI is
// batch-native, so a job with N processes gets ONE invocation carrying all
// PIDs comma-separated plus the per-process control timeout. For restore the
// single batched invocation is also the rendezvous contract: the CLI fans
// out concurrently, which is what lets every mesh member's RESTORE be
// pending at once, with the long rendezvous timeout.
func TestTpuCliArgs(t *testing.T) {
	tests := []struct {
		name string
		call func(*backends.TpuCheckpoint) error
		want [][]string
	}{
		{
			name: "Snapshot",
			call: func(c *backends.TpuCheckpoint) error {
				return c.Snapshot(context.Background(), backends.Request{JobID: "j", Config: tpuConfig(11, 22)})
			},
			want: [][]string{
				{"--action", "checkpoint", "--pid", "11,22", "--timeout", "120"},
			},
		},
		{
			name: "Restore",
			call: func(c *backends.TpuCheckpoint) error {
				return c.Restore(context.Background(), backends.Request{JobID: "j", Config: tpuConfig(11, 22)})
			},
			want: [][]string{
				{"--action", "restore", "--pid", "11,22", "--timeout", "600"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, log := newTestTpuCheckpoint(nil)

			if err := tt.call(c); err != nil {
				t.Fatalf("%s() error = %v", tt.name, err)
			}
			got := log.sorted()
			if len(got) != len(tt.want) {
				t.Fatalf("%s() made %d invocations, want %d: %v", tt.name, len(got), len(tt.want), got)
			}
			for i := range tt.want {
				if !slices.Equal(got[i], tt.want[i]) {
					t.Errorf("%s() invocation %d args = %v, want %v", tt.name, i, got[i], tt.want[i])
				}
			}
		})
	}
}

// TestTpuSnapshotPartialRetry pins the retry contract: when the CLI reports
// per-PID failures on stderr ("tpucheckpoint: pid <N>: ..."), the retry
// carries ONLY the failed PIDs — the rest are parked and must not receive a
// second CHECKPOINT.
func TestTpuSnapshotPartialRetry(t *testing.T) {
	c, log := newTestTpuCheckpoint(nil)
	calls := 0
	c.SetExecCommand(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		log.record(args)
		calls++
		if calls == 1 {
			return []byte("pid 11: checkpoint complete (1.234s)\ntpucheckpoint: pid 22: device busy\n"), fmt.Errorf("exit status 1")
		}
		return nil, nil
	})

	if err := c.Snapshot(context.Background(), backends.Request{JobID: "j", Config: tpuConfig(11, 22)}); err != nil {
		t.Fatalf("Snapshot() error = %v", err)
	}
	want := [][]string{
		{"--action", "checkpoint", "--pid", "11,22", "--timeout", "120"},
		{"--action", "checkpoint", "--pid", "22", "--timeout", "120"},
	}
	got := log.sorted()
	if len(got) != len(want) {
		t.Fatalf("Snapshot() made %d invocations, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if !slices.Equal(got[i], want[i]) {
			t.Errorf("Snapshot() invocation %d args = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestTpuRestore(t *testing.T) {
	tests := []struct {
		name        string
		config      *pb.BackendConfig
		execErr     error
		expectedErr bool
	}{
		{
			name:   "Success",
			config: tpuConfig(123),
		},
		{
			name:        "NoPIDs",
			config:      tpuConfig(),
			expectedErr: true,
		},
		{
			name:        "ExecFailure",
			config:      tpuConfig(123),
			execErr:     fmt.Errorf("exec error"),
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, log := newTestTpuCheckpoint(tt.execErr)

			err := c.Restore(context.Background(), backends.Request{JobID: "test-job", Config: tt.config})
			if (err != nil) != tt.expectedErr {
				t.Errorf("Restore() error = %v, expectedErr %v", err, tt.expectedErr)
			}
			for _, args := range log.sorted() {
				if args[0] != "--action" || args[1] != "restore" {
					t.Errorf("Restore() args = %v, want --action restore first", args)
				}
			}
			// A failed restore must never be retried: a duplicate RESTORE
			// request wedges libtpu's state machine.
			if tt.execErr != nil && log.count() != 1 {
				t.Errorf("Restore() invoked the CLI %d times after failure, want exactly 1 (never retry)", log.count())
			}
		})
	}
}

// TestTpuRestoreVfioGate pins the ordering contract: the vfio gate runs
// before any RESTORE is issued, and a gate failure aborts the restore with
// zero CLI invocations (so the restore stays safe to retry).
func TestTpuRestoreVfioGate(t *testing.T) {
	t.Run("GateBeforeRestore", func(t *testing.T) {
		c, log := newTestTpuCheckpoint(nil)
		gated := false
		c.SetWaitVfioFree(func(context.Context) error {
			if log.count() != 0 {
				t.Error("vfio gate ran after a CLI invocation; it must run before any RESTORE")
			}
			gated = true
			return nil
		})

		if err := c.Restore(context.Background(), backends.Request{JobID: "j", Config: tpuConfig(11)}); err != nil {
			t.Fatalf("Restore() error = %v", err)
		}
		if !gated {
			t.Error("vfio gate never ran")
		}
	})

	t.Run("GateFailureAborts", func(t *testing.T) {
		c, log := newTestTpuCheckpoint(nil)
		c.SetWaitVfioFree(func(context.Context) error { return fmt.Errorf("groups busy") })

		if err := c.Restore(context.Background(), backends.Request{JobID: "j", Config: tpuConfig(11)}); err == nil {
			t.Fatal("Restore() = nil, want error when the vfio gate fails")
		}
		if log.count() != 0 {
			t.Errorf("Restore() issued %d CLI invocations after a gate failure, want 0", log.count())
		}
	})
}

func TestTpuHealthCheck(t *testing.T) {
	tests := []struct {
		name        string
		lookPathErr error
		statErr     error
		expectedErr bool
	}{
		{
			name: "Success",
		},
		{
			name:        "MissingCLI",
			lookPathErr: fmt.Errorf("not found"),
			expectedErr: true,
		},
		{
			name:        "MissingVfio",
			statErr:     os.ErrNotExist,
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := backends.NewTpuCheckpoint()
			c.SetLookPath(func(path string) (string, error) {
				return path, tt.lookPathErr
			})
			c.SetStatPath(func(string) (os.FileInfo, error) {
				return nil, tt.statErr
			})

			err := c.HealthCheck(context.Background())
			if (err != nil) != tt.expectedErr {
				t.Errorf("HealthCheck() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}
