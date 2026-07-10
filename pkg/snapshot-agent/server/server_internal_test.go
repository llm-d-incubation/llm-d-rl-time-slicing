package server

import (
	"context"
	"log/slog"
	"net"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/utils"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	fakek8s "k8s.io/client-go/kubernetes/fake"
)

const bufSize = 1024 * 1024

var (
	lis          *bufconn.Listener
	testServer   *Server
	fakeClient   *fakek8s.Clientset
	mockedPIDs   map[string][]int
	mockedPIDsMu sync.RWMutex
)

type FailingBackend struct {
	backends.NoopBackend
}

func (b *FailingBackend) HealthCheck(ctx context.Context) error {
	return context.DeadlineExceeded
}

func initGRPCServer() {
	if lis != nil {
		return // Already initialized
	}
	os.Setenv("NODE_NAME", "test-node")

	lis = bufconn.Listen(bufSize)
	s := grpc.NewServer()

	noopBackend := backends.NewNoopBackend()
	failingBackend := &FailingBackend{}
	backendsMap := map[backends.BackendType]backends.Backend{
		backends.BackendNoop: noopBackend,
		backends.BackendCuda: noopBackend,
		"failing":            failingBackend,
	}

	// Default to BackendCuda (matching production) so that requests without a
	// BackendConfig take the k8s CUDA discovery path; the registered
	// implementation is still the noop backend.
	testServer = NewServer(backendsMap, backends.BackendCuda, "k8s")
	pb.RegisterSnapshotAgentServiceServer(s, testServer)
	grpc_health_v1.RegisterHealthServer(s, NewHealthServer(backendsMap, backends.BackendNoop))
	go func() {
		if err := s.Serve(lis); err != nil {
			slog.Error("Server exited with error", "error", err)
			os.Exit(1)
		}
	}()

	// Set up mocks
	fakeClient = fakek8s.NewSimpleClientset()
	utils.GetK8sClient = func() (kubernetes.Interface, error) {
		return fakeClient, nil
	}

	mockedPIDs = make(map[string][]int)
	utils.GetPodPIDs = func(ctx context.Context, podName, namespace string) ([]int, error) {
		mockedPIDsMu.RLock()
		defer mockedPIDsMu.RUnlock()
		if pids, ok := mockedPIDs[podName]; ok {
			return pids, nil
		}
		return nil, nil
	}
}

func bufDialer(context.Context, string) (net.Conn, error) {
	return lis.Dial()
}

func createFakePod(ctx context.Context, t *testing.T, jobID, podName string) {
	t.Helper()

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: "default",
			Labels: map[string]string{
				utils.JobIDLabel: jobID,
			},
		},
		Spec: corev1.PodSpec{
			NodeName: "test-node",
		},
	}
	_, err := fakeClient.CoreV1().Pods("default").Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create fake pod: %v", err)
	}
}

// prepareSavedJob is a helper that registers a job, transitions it to RUNNING,
// triggers a snapshot, and waits until the job state becomes SAVED.
func prepareSavedJob(
	t *testing.T,
	client pb.SnapshotAgentServiceClient,
	ctx context.Context,
	jobID, group, podName string,
	pids []int,
) {
	t.Helper()
	createFakePod(ctx, t, jobID, podName)
	mockedPIDsMu.Lock()
	mockedPIDs[podName] = pids
	mockedPIDsMu.Unlock()

	testServer.state.RegisterJob(jobID, group)

	if err := testServer.state.TransitionToRunning(jobID, pids); err != nil {
		t.Fatalf("Failed to transition job to RUNNING: %v", err)
	}

	_, err := client.Snapshot(ctx, &pb.SnapshotRequest{
		JobId: jobID,
		Group: group,
	})
	if err != nil {
		t.Fatalf("Failed to snapshot: %v", err)
	}

	// Wait for job to become SAVED
	deadline := time.Now().Add(2 * time.Second)
	saved := false
	for time.Now().Before(deadline) {
		statuses := testServer.state.GetJobStatus()
		for _, s := range statuses {
			if s.JobId == jobID && s.State == pb.JobState_JOB_STATE_SAVED {
				saved = true
				break
			}
		}
		if saved {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !saved {
		t.Fatalf("Timeout waiting for job %s to become SAVED", jobID)
	}
}

func TestServer_Snapshot(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := grpc.NewClient("passthrough://bufnet",
		grpc.WithContextDialer(bufDialer),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewSnapshotAgentServiceClient(conn)

	tests := []struct {
		name          string
		jobID         string
		group         string
		podName       string
		pids          []int
		backendConfig *pb.BackendConfig
	}{
		{
			name:    "With Group",
			jobID:   "test-job-snapshot-group",
			group:   "test-group",
			podName: "pod-snapshot-group",
			pids:    []int{123},
		},
		{
			name:    "Without Group",
			jobID:   "test-job-snapshot-nogroup",
			group:   "",
			podName: "pod-snapshot-nogroup",
			pids:    []int{456},
		},
		{
			name:    "With BackendConfig",
			jobID:   "test-job-snapshot-backendconfig",
			group:   "",
			podName: "pod-snapshot-backendconfig",
			pids:    []int{789},
			backendConfig: &pb.BackendConfig{
				Backend: &pb.BackendConfig_Cuda{
					Cuda: &pb.CudaBackendConfig{},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			createFakePod(ctx, t, tc.jobID, tc.podName)
			mockedPIDsMu.Lock()
			mockedPIDs[tc.podName] = tc.pids
			mockedPIDsMu.Unlock()

			testServer.state.RegisterJob(tc.jobID, tc.group)

			if err := testServer.state.TransitionToRunning(tc.jobID, tc.pids); err != nil {
				t.Fatalf("Failed to transition job to RUNNING: %v", err)
			}

			_, err = client.Snapshot(ctx, &pb.SnapshotRequest{
				JobId:         tc.jobID,
				Group:         tc.group,
				BackendConfig: tc.backendConfig,
			})
			if err != nil {
				t.Errorf("Expected success (using default noop backend), got error: %v", err)
			}
		})
	}
}

func TestServer_Restore(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := grpc.NewClient("passthrough://bufnet",
		grpc.WithContextDialer(bufDialer),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewSnapshotAgentServiceClient(conn)

	tests := []struct {
		name          string
		jobID         string
		group         string
		podName       string
		pids          []int
		backendConfig *pb.BackendConfig
	}{
		{
			name:    "With Group",
			jobID:   "test-job-restore-group",
			group:   "test-group",
			podName: "pod-restore-group",
			pids:    []int{123},
		},
		{
			name:    "Without Group",
			jobID:   "test-job-restore-nogroup",
			group:   "",
			podName: "pod-restore-nogroup",
			pids:    []int{456},
		},
		{
			name:    "With BackendConfig",
			jobID:   "test-job-restore-backendconfig",
			group:   "",
			podName: "pod-restore-backendconfig",
			pids:    []int{789},
			backendConfig: &pb.BackendConfig{
				Backend: &pb.BackendConfig_Cuda{
					Cuda: &pb.CudaBackendConfig{},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			prepareSavedJob(t, client, ctx, tc.jobID, tc.group, tc.podName, tc.pids)

			_, err = client.Restore(ctx, &pb.RestoreRequest{
				JobId:         tc.jobID,
				Group:         tc.group,
				BackendConfig: tc.backendConfig,
			})
			if err != nil {
				t.Errorf("Expected success (using default noop backend), got error: %v", err)
			}
		})
	}
}

func TestServer_GetOperation(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := grpc.NewClient("passthrough://bufnet",
		grpc.WithContextDialer(bufDialer),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewSnapshotAgentServiceClient(conn)

	_, err = client.GetOperation(ctx, &pb.GetOperationRequest{
		OperationId: "test-op",
	})
	if status.Code(err) != codes.NotFound {
		t.Errorf("Expected NotFound error, got: %v", err)
	}
}

func TestServer_Status(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := grpc.NewClient("passthrough://bufnet",
		grpc.WithContextDialer(bufDialer),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewSnapshotAgentServiceClient(conn)

	tests := []struct {
		name          string
		jobID         string
		setup         func(t *testing.T, jobID string)
		expectedState pb.JobState
	}{
		{
			name:  "Job is IDLE",
			jobID: "job-idle",
			setup: func(t *testing.T, jobID string) {
				t.Helper()
				testServer.state.RegisterJob(jobID, "test-group")
			},
			expectedState: pb.JobState_JOB_STATE_IDLE,
		},
		{
			name:  "Job is RUNNING",
			jobID: "job-running",
			setup: func(t *testing.T, jobID string) {
				t.Helper()
				testServer.state.RegisterJob(jobID, "test-group")
				if err := testServer.state.TransitionToRunning(jobID, []int{123}); err != nil {
					t.Fatalf("Failed to transition job to RUNNING: %v", err)
				}
			},
			expectedState: pb.JobState_JOB_STATE_RUNNING,
		},
		{
			name:  "Job is SAVED",
			jobID: "job-saved",
			setup: func(t *testing.T, jobID string) {
				t.Helper()
				prepareSavedJob(t, client, ctx, jobID, "test-group", "pod-status-saved", []int{456})
			},
			expectedState: pb.JobState_JOB_STATE_SAVED,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.setup(t, tc.jobID)

			resp, err := client.Status(ctx, &pb.StatusRequest{})
			if err != nil {
				t.Errorf("Expected success, got error: %v", err)
			}

			found := false
			for _, js := range resp.JobStatuses {
				if js.JobId == tc.jobID {
					found = true
					if js.State != tc.expectedState {
						t.Errorf("Expected job state %v, got %v", tc.expectedState, js.State)
					}
					break
				}
			}
			if !found {
				t.Errorf("Job %s not found in status response", tc.jobID)
			}
		})
	}
}

func TestServer_Health(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := grpc.NewClient("passthrough://bufnet",
		grpc.WithContextDialer(bufDialer),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := grpc_health_v1.NewHealthClient(conn)

	// Test default (noop) backend
	resp, err := client.Check(ctx, &grpc_health_v1.HealthCheckRequest{})
	if err != nil {
		t.Fatalf("Expected success for default backend, got error: %v", err)
	}
	if resp.Status != grpc_health_v1.HealthCheckResponse_SERVING {
		t.Errorf("Expected status=SERVING for default backend, got %v", resp.Status)
	}

	resp, err = client.Check(ctx, &grpc_health_v1.HealthCheckRequest{Service: ""})
	if err != nil {
		t.Fatalf("Expected success for empty service (default), got error: %v", err)
	}
	if resp.Status != grpc_health_v1.HealthCheckResponse_SERVING {
		t.Errorf("Expected status=SERVING for empty service, got %v", resp.Status)
	}

	resp, err = client.Check(ctx, &grpc_health_v1.HealthCheckRequest{Service: string(backends.BackendNoop)})
	if err != nil {
		t.Fatalf("Expected success for noop backend, got error: %v", err)
	}
	if resp.Status != grpc_health_v1.HealthCheckResponse_SERVING {
		t.Errorf("Expected status=SERVING for noop backend, got %v", resp.Status)
	}

	// Test missing backend (Cuda is now in backendsMap in initGRPCServer, use a non-existent one)
	_, err = client.Check(ctx, &grpc_health_v1.HealthCheckRequest{Service: "missing-backend"})
	if status.Code(err) != codes.NotFound {
		t.Errorf("Expected NotFound error for missing backend, got: %v", err)
	}

	// Test failing backend
	resp, err = client.Check(ctx, &grpc_health_v1.HealthCheckRequest{Service: "failing"})
	if err != nil {
		t.Fatalf("Expected success for failing backend (Check call itself should succeed), got error: %v", err)
	}
	if resp.Status != grpc_health_v1.HealthCheckResponse_NOT_SERVING {
		t.Errorf("Expected status=NOT_SERVING for failing backend, got %v", resp.Status)
	}
}

//nolint:nonamedreturns // Conflict between gocritic's unnamedResult and nonamedreturns
func newModeTestServer(
	t *testing.T,
	backendsMap map[backends.BackendType]backends.Backend,
	defaultBackend backends.BackendType,
	gpuOccupied bool,
	mode string,
) (client pb.SnapshotAgentServiceClient, cleanup func()) {
	t.Helper()

	origHasGPU := utils.HasGPUProcesses
	utils.HasGPUProcesses = func(_ context.Context) (bool, error) {
		return gpuOccupied, nil
	}

	lisLocal := bufconn.Listen(bufSize)
	s := grpc.NewServer()
	srv := NewServer(backendsMap, defaultBackend, mode)
	pb.RegisterSnapshotAgentServiceServer(s, srv)
	go func() {
		if err := s.Serve(lisLocal); err != nil {
			return
		}
	}()

	conn, err := grpc.NewClient("passthrough://bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return lisLocal.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial: %v", err)
	}

	cleanup = func() {
		conn.Close()
		s.GracefulStop()
		utils.HasGPUProcesses = origHasGPU
	}
	client = pb.NewSnapshotAgentServiceClient(conn)
	return client, cleanup
}

func TestServer_Snapshot_StandaloneAutoTransition(t *testing.T) {
	noopBackend := backends.NewNoopBackend()
	backendsMap := map[backends.BackendType]backends.Backend{
		backends.BackendNoop: noopBackend,
		backends.BackendCuda: noopBackend,
	}

	t.Run("CUDA with PIDs auto-transitions from IDLE", func(t *testing.T) {
		client, cleanup := newModeTestServer(t, backendsMap, backends.BackendNoop, true, "standalone")
		defer cleanup()

		resp, err := client.Snapshot(context.Background(), &pb.SnapshotRequest{
			JobId: "auto-cuda",
			BackendConfig: &pb.BackendConfig{
				Backend: &pb.BackendConfig_Cuda{
					Cuda: &pb.CudaBackendConfig{
						ExplicitTarget: &pb.ProcessTarget{Pids: []int32{123}},
					},
				},
			},
		})
		if err != nil {
			t.Fatalf("Expected success with GPU occupied, got: %v", err)
		}
		if resp.OperationId == "" {
			t.Error("Expected operation ID")
		}
	})

	t.Run("K8s mode does not auto-transition", func(t *testing.T) {
		// Even with the GPU occupied, k8s mode must not bootstrap unknown
		// jobs — the watcher is the single source of state transitions.
		client, cleanup := newModeTestServer(t, backendsMap, backends.BackendNoop, true, "k8s")
		defer cleanup()

		_, err := client.Snapshot(context.Background(), &pb.SnapshotRequest{
			JobId: "watcher-unknown",
			BackendConfig: &pb.BackendConfig{
				Backend: &pb.BackendConfig_Cuda{
					Cuda: &pb.CudaBackendConfig{
						ExplicitTarget: &pb.ProcessTarget{Pids: []int32{123}},
					},
				},
			},
		})
		if err == nil {
			t.Fatal("Expected failure for watcher-unknown job in k8s mode")
		}
	})

	t.Run("Rejects when GPU not occupied", func(t *testing.T) {
		client, cleanup := newModeTestServer(t, backendsMap, backends.BackendNoop, false, "standalone")
		defer cleanup()

		_, err := client.Snapshot(context.Background(), &pb.SnapshotRequest{
			JobId: "no-gpu",
			BackendConfig: &pb.BackendConfig{
				Backend: &pb.BackendConfig_Cuda{
					Cuda: &pb.CudaBackendConfig{
						ExplicitTarget: &pb.ProcessTarget{Pids: []int32{123}},
					},
				},
			},
		})
		if err == nil {
			t.Fatal("Expected failure when GPU not occupied")
		}
		if status.Code(err) != codes.FailedPrecondition {
			t.Errorf("Expected FailedPrecondition, got: %v", err)
		}
	})
}

func TestServer_Snapshot_StandaloneMode(t *testing.T) {
	lisDev := bufconn.Listen(bufSize)
	s := grpc.NewServer()
	noopBackend := backends.NewNoopBackend()
	backendsMap := map[backends.BackendType]backends.Backend{
		backends.BackendNoop: noopBackend,
		backends.BackendCuda: noopBackend,
	}
	// enable standalone mode
	standaloneServer := NewServer(backendsMap, backends.BackendNoop, "standalone")
	pb.RegisterSnapshotAgentServiceServer(s, standaloneServer)
	go func() {
		if err := s.Serve(lisDev); err != nil {
			return
		}
	}()
	defer s.GracefulStop()

	ctx := context.Background()
	conn, err := grpc.NewClient("passthrough://bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return lisDev.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewSnapshotAgentServiceClient(conn)

	// Register the job and transition it to RUNNING in the state machine
	standaloneServer.state.RegisterJob("test-job-standalone", "")
	if err := standaloneServer.state.TransitionToRunning("test-job-standalone", []int{123}); err != nil {
		t.Fatalf("Failed to transition job to RUNNING: %v", err)
	}

	// Test with PIDs in standalone mode (should succeed — backend validates)
	_, err = client.Snapshot(ctx, &pb.SnapshotRequest{
		JobId: "test-job-standalone",
		BackendConfig: &pb.BackendConfig{
			Backend: &pb.BackendConfig_Cuda{
				Cuda: &pb.CudaBackendConfig{
					ExplicitTarget: &pb.ProcessTarget{
						Pids: []int32{123},
					},
				},
			},
		},
	})
	if err != nil {
		t.Errorf("Expected success, got error: %v", err)
	}
}

// TestServer_Snapshot_StandaloneMode_BackendValidation verifies that in
// standalone mode the server passes the config through untouched and backend
// validation failures (here: a CUDA config with no PIDs) surface as a FAILED
// operation rather than a synchronous RPC error.
func TestServer_Snapshot_StandaloneMode_BackendValidation(t *testing.T) {
	lisDev := bufconn.Listen(bufSize)
	s := grpc.NewServer()
	backendsMap := map[backends.BackendType]backends.Backend{
		backends.BackendCuda: backends.NewCudaCheckpoint(),
	}
	standaloneServer := NewServer(backendsMap, backends.BackendCuda, "standalone")
	pb.RegisterSnapshotAgentServiceServer(s, standaloneServer)
	go func() {
		if err := s.Serve(lisDev); err != nil {
			return
		}
	}()
	defer s.GracefulStop()

	ctx := context.Background()
	conn, err := grpc.NewClient("passthrough://bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return lisDev.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewSnapshotAgentServiceClient(conn)

	standaloneServer.state.RegisterJob("test-job-standalone-nopids", "")
	if err := standaloneServer.state.TransitionToRunning("test-job-standalone-nopids", []int{123}); err != nil {
		t.Fatalf("Failed to transition job to RUNNING: %v", err)
	}

	// CUDA config with no PIDs: the RPC is accepted (validation is the
	// backend's job and runs in the background operation).
	resp, err := client.Snapshot(ctx, &pb.SnapshotRequest{
		JobId: "test-job-standalone-nopids",
		BackendConfig: &pb.BackendConfig{
			Backend: &pb.BackendConfig_Cuda{
				Cuda: &pb.CudaBackendConfig{},
			},
		},
	})
	if err != nil {
		t.Fatalf("Expected RPC to be accepted, got error: %v", err)
	}

	// The operation must end FAILED with the backend's validation error.
	var opResp *pb.GetOperationResponse
	for range 50 {
		opResp, err = client.GetOperation(ctx, &pb.GetOperationRequest{OperationId: resp.GetOperationId()})
		if err != nil {
			t.Fatalf("GetOperation failed: %v", err)
		}
		if opResp.GetStatus() != pb.OperationStatus_OPERATION_STATUS_PENDING {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if opResp.GetStatus() != pb.OperationStatus_OPERATION_STATUS_FAILED {
		t.Fatalf("Expected operation status FAILED, got %v", opResp.GetStatus())
	}
	if !strings.Contains(opResp.GetError(), "PID") {
		t.Errorf("Expected backend PID validation error, got: %q", opResp.GetError())
	}
}
