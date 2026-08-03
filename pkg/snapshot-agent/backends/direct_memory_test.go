package backends_test

import (
	"context"
	"fmt"
	"os"
	"reflect"
	"testing"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
)

func directMemoryConfig(pids ...int32) *pb.BackendConfig {
	return &pb.BackendConfig{
		Backend: &pb.BackendConfig_DirectMemory{
			DirectMemory: &pb.DirectMemoryBackendConfig{
				ExplicitTarget: &pb.ProcessTarget{Pids: pids},
			},
		},
	}
}

func TestNewDirectMemory(t *testing.T) {
	dm := backends.NewDirectMemory()
	if dm == nil {
		t.Fatal("NewDirectMemory returned nil")
	}
}

func TestDirectMemorySnapshot(t *testing.T) {
	tests := []struct {
		name        string
		config      *pb.BackendConfig
		execErr     error
		expectedErr bool
		expectArgs  [][]string
	}{
		{
			name:   "SuccessMultiplePIDs",
			config: directMemoryConfig(123, 456),
			expectArgs: [][]string{
				{"-c", "-p", "123"},
				{"-c", "-p", "456"},
			},
		},
		{
			name:        "ExecFailure",
			config:      directMemoryConfig(123),
			execErr:     fmt.Errorf("exec error"),
			expectedErr: true,
			expectArgs: [][]string{
				{"-c", "-p", "123"},
			},
		},
		{
			name:        "NoPIDs",
			config:      directMemoryConfig(),
			expectedErr: true,
		},
		{
			name:        "NilConfig",
			config:      nil,
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dm := backends.NewDirectMemory()
			var calledArgs [][]string
			dm.SetExecCommand(func(_ context.Context, _ string, args ...string) ([]byte, error) {
				calledArgs = append(calledArgs, args)
				return nil, tt.execErr
			})

			err := dm.Snapshot(context.Background(), backends.Request{JobID: "test-job", Config: tt.config})
			if (err != nil) != tt.expectedErr {
				t.Errorf("Snapshot() error = %v, expectedErr %v", err, tt.expectedErr)
			}
			if !tt.expectedErr && !reflect.DeepEqual(calledArgs, tt.expectArgs) {
				t.Errorf("Snapshot() calledArgs = %v, expected %v", calledArgs, tt.expectArgs)
			}
		})
	}
}

func TestDirectMemoryRestore(t *testing.T) {
	tests := []struct {
		name        string
		config      *pb.BackendConfig
		execErr     error
		expectedErr bool
		expectArgs  [][]string
	}{
		{
			name:   "SuccessMultiplePIDs",
			config: directMemoryConfig(123, 456),
			expectArgs: [][]string{
				{"-r", "-p", "123"},
				{"-r", "-p", "456"},
			},
		},
		{
			name:        "NoPIDs",
			config:      directMemoryConfig(),
			expectedErr: true,
		},
		{
			name:        "NilConfig",
			config:      nil,
			expectedErr: true,
		},
		{
			name:        "ExecFailure",
			config:      directMemoryConfig(123),
			execErr:     fmt.Errorf("exec error"),
			expectedErr: true,
			expectArgs: [][]string{
				{"-r", "-p", "123"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dm := backends.NewDirectMemory()
			var calledArgs [][]string
			dm.SetExecCommand(func(_ context.Context, _ string, args ...string) ([]byte, error) {
				calledArgs = append(calledArgs, args)
				return nil, tt.execErr
			})

			err := dm.Restore(context.Background(), backends.Request{JobID: "test-job", Config: tt.config})
			if (err != nil) != tt.expectedErr {
				t.Errorf("Restore() error = %v, expectedErr %v", err, tt.expectedErr)
			}
			if !tt.expectedErr && !reflect.DeepEqual(calledArgs, tt.expectArgs) {
				t.Errorf("Restore() calledArgs = %v, expected %v", calledArgs, tt.expectArgs)
			}
		})
	}
}

func TestDirectMemoryHealthCheck(t *testing.T) {
	tests := []struct {
		name        string
		lookErr     error
		statErr     error
		expectedErr bool
	}{
		{
			name:        "SuccessInPath",
			lookErr:     nil,
			statErr:     nil,
			expectedErr: false,
		},
		{
			name:        "SuccessViaStat",
			lookErr:     fmt.Errorf("not in path"),
			statErr:     nil,
			expectedErr: false,
		},
		{
			name:        "NotFoundAnywhere",
			lookErr:     fmt.Errorf("not in path"),
			statErr:     fmt.Errorf("no stat"),
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dm := backends.NewDirectMemory()
			dm.SetLookPath(func(path string) (string, error) {
				if tt.lookErr != nil {
					return "", tt.lookErr
				}
				return path, nil
			})
			dm.SetStatFunc(func(path string) (os.FileInfo, error) {
				if tt.statErr != nil {
					return nil, tt.statErr
				}
				return os.Stat(".")
			})

			err := dm.HealthCheck(context.Background())
			if (err != nil) != tt.expectedErr {
				t.Errorf("HealthCheck() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}

func TestDirectMemoryConfigHelpers(t *testing.T) {
	pids := []string{"100", "200"}
	cfg, err := backends.BuildDirectMemoryConfig(pids)
	if err != nil {
		t.Fatalf("BuildDirectMemoryConfig() unexpected error: %v", err)
	}
	extracted := backends.ExtractDirectMemoryPIDStrings(cfg)
	if !reflect.DeepEqual(extracted, pids) {
		t.Errorf("ExtractDirectMemoryPIDStrings() = %v, want %v", extracted, pids)
	}

	if len(backends.ExtractDirectMemoryPIDStrings(nil)) != 0 {
		t.Errorf("Expected nil when extracting from nil config")
	}

	_, err = backends.BuildDirectMemoryConfig([]string{"100", "not-a-pid"})
	if err == nil {
		t.Errorf("BuildDirectMemoryConfig() expected error for invalid PID string, got nil")
	}
}
