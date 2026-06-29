// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package server_test

import (
	"context"
	"log/slog"
	"net"
	"os"
	"testing"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/server"
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
	testServer   *server.Server
	fakeClient   *fakek8s.Clientset
	mockedPIDs   map[string][]int
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

	testServer = server.NewServer(backendsMap, backends.BackendNoop)
	pb.RegisterSnapshotAgentServiceServer(s, testServer)
	grpc_health_v1.RegisterHealthServer(s, server.NewHealthServer(backendsMap, backends.BackendNoop))
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
		if pids, ok := mockedPIDs[podName]; ok {
			return pids, nil
		}
		return nil, nil
	}
}

func bufDialer(context.Context, string) (net.Conn, error) {
	return lis.Dial()
}

func createFakePod(t *testing.T, jobID, podName string) {
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
	_, err := fakeClient.CoreV1().Pods("default").Create(context.Background(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create fake pod: %v", err)
	}
}

// prepareSavedJob is a helper that registers a job, transitions it to RUNNING,
// triggers a snapshot, and waits until the job state becomes SAVED.
func prepareSavedJob(t *testing.T, client pb.SnapshotAgentServiceClient, ctx context.Context, jobID, group string, podName string, pids []int) {
	t.Helper()
	createFakePod(t, jobID, podName)
	mockedPIDs[podName] = pids

	testServer.InternalState().RegisterJob(jobID, group)
	if err := testServer.InternalState().TransitionToRunning(jobID, pids); err != nil {
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
		statuses := testServer.InternalState().GetJobStatus()
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
			createFakePod(t, tc.jobID, tc.podName)
			mockedPIDs[tc.podName] = tc.pids

			testServer.InternalState().RegisterJob(tc.jobID, tc.group)
			if err := testServer.InternalState().TransitionToRunning(tc.jobID, tc.pids); err != nil {
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
				testServer.InternalState().RegisterJob(jobID, "test-group")
			},
			expectedState: pb.JobState_JOB_STATE_IDLE,
		},
		{
			name:  "Job is RUNNING",
			jobID: "job-running",
			setup: func(t *testing.T, jobID string) {
				testServer.InternalState().RegisterJob(jobID, "test-group")
				if err := testServer.InternalState().TransitionToRunning(jobID, []int{123}); err != nil {
					t.Fatalf("Failed to transition job to RUNNING: %v", err)
				}
			},
			expectedState: pb.JobState_JOB_STATE_RUNNING,
		},
		{
			name:  "Job is SAVED",
			jobID: "job-saved",
			setup: func(t *testing.T, jobID string) {
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
