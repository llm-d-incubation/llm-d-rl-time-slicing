package server

import (
	"context"
	"log/slog"
	"net"
	"os"
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
		"failing":            failingBackend,
	}

	testServer = NewServer(backendsMap, backends.BackendNoop, "k8s")
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
		JobId:   jobID,
		Group:   group,
		Backend: pb.Backend_BACKEND_UNSPECIFIED,
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
		name    string
		jobID   string
		group   string
		podName string
		pids    []int
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
				JobId:   tc.jobID,
				Group:   tc.group,
				Backend: pb.Backend_BACKEND_UNSPECIFIED,
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
		name    string
		jobID   string
		group   string
		podName string
		pids    []int
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
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			prepareSavedJob(t, client, ctx, tc.jobID, tc.group, tc.podName, tc.pids)

			_, err = client.Restore(ctx, &pb.RestoreRequest{
				JobId:   tc.jobID,
				Group:   tc.group,
				Backend: pb.Backend_BACKEND_UNSPECIFIED,
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

	// Test missing backend (Cuda is not in backendsMap in initGRPCServer)
	_, err = client.Check(ctx, &grpc_health_v1.HealthCheckRequest{Service: string(backends.BackendCuda)})
	if status.Code(err) != codes.NotFound {
		t.Errorf("Expected NotFound error for missing CUDA backend, got: %v", err)
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
func newStandaloneTestServer(
	t *testing.T,
	backendsMap map[backends.BackendType]backends.Backend,
	defaultBackend backends.BackendType,
	gpuOccupied bool,
) (client pb.SnapshotAgentServiceClient, cleanup func()) {
	t.Helper()

	origHasGPU := utils.HasGPUProcesses
	utils.HasGPUProcesses = func(_ context.Context) (bool, error) {
		return gpuOccupied, nil
	}

	lisLocal := bufconn.Listen(bufSize)
	s := grpc.NewServer()
	srv := NewServer(backendsMap, defaultBackend, "standalone")
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
		client, cleanup := newStandaloneTestServer(t, backendsMap, backends.BackendNoop, true)
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

	t.Run("Rejects when GPU not occupied", func(t *testing.T) {
		client, cleanup := newStandaloneTestServer(t, backendsMap, backends.BackendNoop, false)
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

// mockAppAwareBackend implements AppAwareBackend for testing the server dispatch.
type mockAppAwareBackend struct {
	backends.NoopBackend
}

func (m *mockAppAwareBackend) SnapshotApp(_ context.Context, _ backends.AppConfig) error {
	return nil
}

func (m *mockAppAwareBackend) RestoreApp(_ context.Context, _ backends.AppConfig) error {
	return nil
}

func TestServer_Snapshot_AppAwareBackend(t *testing.T) {
	appBackend := &mockAppAwareBackend{}
	noopBackend := backends.NewNoopBackend()
	backendsMap := map[backends.BackendType]backends.Backend{
		backends.BackendNoop: noopBackend,
		backends.BackendVLLM: appBackend,
	}

	t.Run("App-aware skips PID requirement in standalone", func(t *testing.T) {
		client, cleanup := newStandaloneTestServer(
			t, backendsMap, backends.BackendNoop, true)
		defer cleanup()

		resp, err := client.Snapshot(context.Background(), &pb.SnapshotRequest{
			JobId: "app-test",
			BackendConfig: &pb.BackendConfig{
				Backend: &pb.BackendConfig_Vllm{
					Vllm: &pb.VLLMBackendConfig{
						Endpoints:  []string{"http://localhost:8000"},
						SleepLevel: 1,
					},
				},
			},
		})
		if err != nil {
			t.Fatalf("Expected success, got: %v", err)
		}
		if resp.OperationId == "" {
			t.Error("Expected operation ID")
		}
	})

	t.Run("App-aware rejects when GPU not occupied", func(t *testing.T) {
		client, cleanup := newStandaloneTestServer(
			t, backendsMap, backends.BackendNoop, false)
		defer cleanup()

		_, err := client.Snapshot(context.Background(), &pb.SnapshotRequest{
			JobId: "app-no-gpu",
			BackendConfig: &pb.BackendConfig{
				Backend: &pb.BackendConfig_Vllm{
					Vllm: &pb.VLLMBackendConfig{
						Endpoints:  []string{"http://localhost:8000"},
						SleepLevel: 1,
					},
				},
			},
		})
		if err == nil {
			t.Fatal("Expected failure when GPU not occupied")
		}
	})

	t.Run("App-aware restore after snapshot", func(t *testing.T) {
		client, cleanup := newStandaloneTestServer(
			t, backendsMap, backends.BackendNoop, true)
		defer cleanup()

		resp, err := client.Snapshot(context.Background(), &pb.SnapshotRequest{
			JobId: "app-roundtrip",
			BackendConfig: &pb.BackendConfig{
				Backend: &pb.BackendConfig_Vllm{
					Vllm: &pb.VLLMBackendConfig{
						Endpoints:  []string{"http://localhost:8000"},
						SleepLevel: 1,
					},
				},
			},
		})
		if err != nil {
			t.Fatalf("Snapshot failed: %v", err)
		}

		deadline := time.Now().Add(5 * time.Second)
		for time.Now().Before(deadline) {
			op, opErr := client.GetOperation(context.Background(),
				&pb.GetOperationRequest{OperationId: resp.OperationId})
			if opErr != nil {
				t.Fatalf("GetOperation failed: %v", opErr)
			}
			if op.Status == pb.OperationStatus_OPERATION_STATUS_COMPLETE {
				break
			}
			if op.Status == pb.OperationStatus_OPERATION_STATUS_FAILED {
				t.Fatalf("Snapshot operation failed: %s", op.GetError())
			}
			time.Sleep(50 * time.Millisecond)
		}

		resp2, err := client.Restore(context.Background(), &pb.RestoreRequest{
			JobId: "app-roundtrip",
			BackendConfig: &pb.BackendConfig{
				Backend: &pb.BackendConfig_Vllm{
					Vllm: &pb.VLLMBackendConfig{
						Endpoints: []string{"http://localhost:8000"},
					},
				},
			},
		})
		if err != nil {
			t.Fatalf("Restore failed: %v", err)
		}
		if resp2.OperationId == "" {
			t.Error("Expected operation ID for restore")
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

	// Test without PIDs in standalone mode (should fail)
	_, err = client.Snapshot(ctx, &pb.SnapshotRequest{
		JobId: "test-job-standalone",
	})
	if err == nil {
		t.Errorf("Expected failure when PIDs are not provided in standalone mode")
	} else if status.Code(err) != codes.InvalidArgument {
		t.Errorf("Expected InvalidArgument error, got: %v", err)
	}

	// Register the job and transition it to RUNNING in the state machine
	standaloneServer.state.RegisterJob("test-job-standalone", "")
	if err := standaloneServer.state.TransitionToRunning("test-job-standalone", []int{123}); err != nil {
		t.Fatalf("Failed to transition job to RUNNING: %v", err)
	}

	// Test with PIDs in standalone mode (should succeed)
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
