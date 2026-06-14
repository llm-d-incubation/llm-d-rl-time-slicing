package backends_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
)

type mockNvmlClient struct {
	initRet        nvml.Return
	shutdownRet    nvml.Return
	deviceCount    int
	deviceCountRet nvml.Return
}

func (m *mockNvmlClient) Init() nvml.Return { return m.initRet }
func (m *mockNvmlClient) Shutdown() nvml.Return { return m.shutdownRet }
func (m *mockNvmlClient) DeviceGetCount() (int, nvml.Return) { return m.deviceCount, m.deviceCountRet }

func TestNewCudaCheckpoint(t *testing.T) {
	c := backends.NewCudaCheckpoint()
	if c == nil {
		t.Fatal("NewCudaCheckpoint returned nil")
	}
}

func TestSnapshot(t *testing.T) {
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
			c := backends.NewCudaCheckpoint()
			c.SetExecCommand(func(ctx context.Context, name string, args ...string) ([]byte, error) {
				return nil, tt.execErr
			})

			err := c.Snapshot(context.Background(), tt.pids)
			if (err != nil) != tt.expectedErr {
				t.Errorf("Snapshot() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}

func TestRestore(t *testing.T) {
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
			execErr:     nil,
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
			c := backends.NewCudaCheckpoint()
			c.SetExecCommand(func(ctx context.Context, name string, args ...string) ([]byte, error) {
				return nil, tt.execErr
			})

			err := c.Restore(context.Background(), tt.pids)
			if (err != nil) != tt.expectedErr {
				t.Errorf("Restore() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}

func TestHealthCheck(t *testing.T) {
	tests := []struct {
		name           string
		initRet        nvml.Return
		deviceCount    int
		deviceCountRet nvml.Return
		expectedErr    bool
	}{
		{
			name:           "Success",
			initRet:        nvml.SUCCESS,
			deviceCount:    1,
			deviceCountRet: nvml.SUCCESS,
			expectedErr:    false,
		},
		{
			name:           "NVMLInitFailure",
			initRet:        nvml.ERROR_LIBRARY_NOT_FOUND,
			expectedErr:    true,
		},
		{
			name:           "NoGPUs",
			initRet:        nvml.SUCCESS,
			deviceCount:    0,
			deviceCountRet: nvml.SUCCESS,
			expectedErr:    true,
		},
		{
			name:           "DeviceCountFailure",
			initRet:        nvml.SUCCESS,
			deviceCount:    0,
			deviceCountRet: nvml.ERROR_UNKNOWN,
			expectedErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := backends.NewCudaCheckpoint()
			c.SetLookPath(func(path string) (string, error) {
				return path, nil
			})
			c.SetNvmlClient(&mockNvmlClient{
				initRet:        tt.initRet,
				shutdownRet:    nvml.SUCCESS,
				deviceCount:    tt.deviceCount,
				deviceCountRet: tt.deviceCountRet,
			})

			err := c.HealthCheck(context.Background())
			if (err != nil) != tt.expectedErr {
				t.Errorf("HealthCheck() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}

func TestInternalFunctions(t *testing.T) {
	c := backends.NewCudaCheckpoint()

	t.Run("GetCudaCheckpointPath", func(t *testing.T) {
		path := backends.GetCudaCheckpointPath(c)
		if path == "" {
			t.Error("GetCudaCheckpointPath returned empty string")
		}
	})

	t.Run("RunSudoCommand", func(t *testing.T) {
		c.SetExecCommand(func(ctx context.Context, name string, args ...string) ([]byte, error) {
			if name != "test-cmd" {
				return nil, fmt.Errorf("unexpected command: %s", name)
			}
			return []byte("output"), nil
		})
		err := backends.RunSudoCommand(c, context.Background(), "test-cmd", "arg1")
		if err != nil {
			t.Errorf("RunSudoCommand failed: %v", err)
		}
	})

	t.Run("CheckpointPIDs", func(t *testing.T) {
		commands := []string{}
		c.SetExecCommand(func(ctx context.Context, name string, args ...string) ([]byte, error) {
			commands = append(commands, args[1]) // the action
			return nil, nil
		})
		err := backends.CheckpointPIDs(c, context.Background(), []string{"123"})
		if err != nil {
			t.Errorf("CheckpointPIDs failed: %v", err)
		}
		if len(commands) != 2 || commands[0] != "lock" || commands[1] != "checkpoint" {
			t.Errorf("Unexpected commands: %v", commands)
		}
	})

	t.Run("RestorePIDs", func(t *testing.T) {
		toggled := false
		c.SetExecCommand(func(ctx context.Context, name string, args ...string) ([]byte, error) {
			for _, arg := range args {
				if arg == "--toggle" {
					toggled = true
				}
			}
			return nil, nil
		})
		err := backends.RestorePIDs(c, context.Background(), []string{"123"})
		if err != nil {
			t.Errorf("RestorePIDs failed: %v", err)
		}
		if !toggled {
			t.Error("Expected --toggle in RestorePIDs")
		}
	})
}
