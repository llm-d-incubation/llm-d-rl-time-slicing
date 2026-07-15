package backends_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
)

func cudaConfig(pids ...int32) *pb.BackendConfig {
	return &pb.BackendConfig{
		Backend: &pb.BackendConfig_Cuda{
			Cuda: &pb.CudaBackendConfig{
				ExplicitTarget: &pb.ProcessTarget{Pids: pids},
			},
		},
	}
}

type mockNvmlClient struct {
	initRet        nvml.Return
	shutdownRet    nvml.Return
	deviceCount    int
	deviceCountRet nvml.Return
}

func (m *mockNvmlClient) Init() nvml.Return                  { return m.initRet }
func (m *mockNvmlClient) Shutdown() nvml.Return              { return m.shutdownRet }
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
		config      *pb.BackendConfig
		execErr     error
		expectedErr bool
	}{
		{
			name:   "Success",
			config: cudaConfig(123, 456),
		},
		{
			name:        "ExecFailure",
			config:      cudaConfig(123),
			execErr:     fmt.Errorf("exec error"),
			expectedErr: true,
		},
		{
			name:        "NoPIDs",
			config:      cudaConfig(),
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
			c := backends.NewCudaCheckpoint()
			c.SetExecCommand(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
				return nil, tt.execErr
			})

			err := c.Snapshot(context.Background(), backends.Request{JobID: "test-job", Config: tt.config})
			if (err != nil) != tt.expectedErr {
				t.Errorf("Snapshot() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}

func TestRestore(t *testing.T) {
	tests := []struct {
		name        string
		config      *pb.BackendConfig
		execErr     error
		expectedErr bool
	}{
		{
			name:   "Success",
			config: cudaConfig(123),
		},
		{
			name:        "NoPIDs",
			config:      cudaConfig(),
			expectedErr: true,
		},
		{
			name:        "ExecFailure",
			config:      cudaConfig(123),
			execErr:     fmt.Errorf("exec error"),
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := backends.NewCudaCheckpoint()
			c.SetExecCommand(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
				return nil, tt.execErr
			})

			err := c.Restore(context.Background(), backends.Request{JobID: "test-job", Config: tt.config})
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
		},
		{
			name:        "NVMLInitFailure",
			initRet:     nvml.ERROR_LIBRARY_NOT_FOUND,
			expectedErr: true,
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
