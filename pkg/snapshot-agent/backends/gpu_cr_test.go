package backends_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
)

func TestNewGpuCr(t *testing.T) {
	g := backends.NewGpuCr()
	if g == nil {
		t.Fatal("NewGpuCr returned nil")
	}
}

func TestGpuCrSnapshot(t *testing.T) {
	tests := []struct {
		name        string
		pids        []string
		execErr     error
		expectedErr bool
	}{
		{
			name:        "Success",
			pids:        []string{"123", "456"},
			execErr:     nil,
			expectedErr: false,
		},
		{
			name:        "ExecFailure",
			pids:        []string{"123"},
			execErr:     fmt.Errorf("exec error"),
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := backends.NewGpuCr()
			g.SetExecCommand(func(ctx context.Context, name string, args ...string) ([]byte, error) {
				if name != "/usr/local/bin/cr_client" {
					t.Errorf("expected command /usr/local/bin/cr_client, got %s", name)
				}
				return nil, tt.execErr
			})

			err := g.Snapshot(context.Background(), tt.pids)
			if (err != nil) != tt.expectedErr {
				t.Errorf("Snapshot() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}

func TestGpuCrRestore(t *testing.T) {
	tests := []struct {
		name        string
		pids        []string
		execErr     error
		expectedErr bool
	}{
		{
			name:        "Success",
			pids:        []string{"123"},
			execErr:     nil,
			expectedErr: false,
		},
		{
			name:        "NoPIDs",
			pids:        []string{},
			expectedErr: true,
		},
		{
			name:        "ExecFailure",
			pids:        []string{"123"},
			execErr:     fmt.Errorf("exec error"),
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := backends.NewGpuCr()
			g.SetExecCommand(func(ctx context.Context, name string, args ...string) ([]byte, error) {
				if name != "/usr/local/bin/cr_client" {
					t.Errorf("expected command /usr/local/bin/cr_client, got %s", name)
				}
				return nil, tt.execErr
			})

			err := g.Restore(context.Background(), tt.pids)
			if (err != nil) != tt.expectedErr {
				t.Errorf("Restore() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}

func TestGpuCrHealthCheck(t *testing.T) {
	tests := []struct {
		name        string
		lookPathErr error
		expectedErr bool
	}{
		{
			name:        "Success",
			lookPathErr: nil,
			expectedErr: false,
		},
		{
			name:        "ExecutableNotFound",
			lookPathErr: fmt.Errorf("not found"),
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := backends.NewGpuCr()
			g.SetLookPath(func(path string) (string, error) {
				return path, tt.lookPathErr
			})

			err := g.HealthCheck(context.Background())
			if (err != nil) != tt.expectedErr {
				t.Errorf("HealthCheck() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}
