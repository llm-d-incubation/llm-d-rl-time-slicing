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
	podutils "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/utils"
	googlegrpc "google.golang.org/grpc"
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
	s := googlegrpc.NewServer()

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
	podutils.GetK8sClient = func() (kubernetes.Interface, error) {
		return fakeClient, nil
	}

	mockedPIDs = make(map[string][]int)
	podutils.GetPodPIDs = func(ctx context.Context, podName, namespace string) ([]int, error) {
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
				podutils.JobIDLabel: jobID,
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

func TestServer_Snapshot(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := googlegrpc.NewClient("passthrough://bufnet",
		googlegrpc.WithContextDialer(bufDialer),
		googlegrpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewSnapshotAgentServiceClient(conn)

	// Test with group
	jobID := "test-job"
	podName := "test-pod"
	createFakePod(t, jobID, podName)
	mockedPIDs[podName] = []int{123}

	testServer.InternalState().RegisterJob(jobID, "test-group")
	if err := testServer.InternalState().TransitionToRunning(jobID, []int{123}); err != nil {
		t.Fatalf("Failed to transition job to RUNNING: %v", err)
	}

	_, err = client.Snapshot(ctx, &pb.SnapshotRequest{
		JobId:   jobID,
		Group:   "test-group",
		Backend: pb.Backend_BACKEND_UNSPECIFIED,
	})
	if err != nil {
		t.Errorf("Expected success (using default noop backend), got error: %v", err)
	}

	// Test without group
	jobIDNoGroup := "test-job-no-group"
	podNameNoGroup := "test-pod-no-group"
	createFakePod(t, jobIDNoGroup, podNameNoGroup)
	mockedPIDs[podNameNoGroup] = []int{456}

	testServer.InternalState().RegisterJob(jobIDNoGroup, "")
	if err := testServer.InternalState().TransitionToRunning(jobIDNoGroup, []int{456}); err != nil {
		t.Fatalf("Failed to transition job to RUNNING: %v", err)
	}

	_, err = client.Snapshot(ctx, &pb.SnapshotRequest{
		JobId:   jobIDNoGroup,
		Backend: pb.Backend_BACKEND_UNSPECIFIED,
	})
	if err != nil {
		t.Errorf("Expected success with empty group, got error: %v", err)
	}
}

func TestServer_Restore(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := googlegrpc.NewClient("passthrough://bufnet",
		googlegrpc.WithContextDialer(bufDialer),
		googlegrpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewSnapshotAgentServiceClient(conn)

	// Helper to snapshot and wait until SAVED
	prepareSavedJob := func(jobID, group string, podName string, pids []int) {
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

	// Test with group
	jobID := "test-job-restore"
	prepareSavedJob(jobID, "test-group", "pod-restore", []int{123})

	_, err = client.Restore(ctx, &pb.RestoreRequest{
		JobId:   jobID,
		Group:   "test-group",
		Backend: pb.Backend_BACKEND_UNSPECIFIED,
	})
	if err != nil {
		t.Errorf("Expected success (using default noop backend), got error: %v", err)
	}

	// Test without group
	jobIDNoGroup := "test-job-no-group-restore"
	prepareSavedJob(jobIDNoGroup, "", "pod-no-group-restore", []int{456})

	_, err = client.Restore(ctx, &pb.RestoreRequest{
		JobId:   jobIDNoGroup,
		Backend: pb.Backend_BACKEND_UNSPECIFIED,
	})
	if err != nil {
		t.Errorf("Expected success with empty group, got error: %v", err)
	}
}

func TestServer_GetOperation(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := googlegrpc.NewClient("passthrough://bufnet",
		googlegrpc.WithContextDialer(bufDialer),
		googlegrpc.WithTransportCredentials(insecure.NewCredentials()))
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
	conn, err := googlegrpc.NewClient("passthrough://bufnet",
		googlegrpc.WithContextDialer(bufDialer),
		googlegrpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewSnapshotAgentServiceClient(conn)

	// Clean up state from previous tests if any, or just check what's there.
	initialResp, err := client.Status(ctx, &pb.StatusRequest{})
	if err != nil {
		t.Errorf("Expected success, got error: %v", err)
	}

	jobID := "test-job-status"
	podName := "pod-status"
	createFakePod(t, jobID, podName)
	mockedPIDs[podName] = []int{789}

	testServer.InternalState().RegisterJob(jobID, "test-group")
	if err := testServer.InternalState().TransitionToRunning(jobID, []int{789}); err != nil {
		t.Fatalf("Failed to transition job to RUNNING: %v", err)
	}

	_, err = client.Snapshot(ctx, &pb.SnapshotRequest{
		JobId:   jobID,
		Group:   "test-group",
		Backend: pb.Backend_BACKEND_UNSPECIFIED,
	})
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}

	resp, err := client.Status(ctx, &pb.StatusRequest{})
	if err != nil {
		t.Errorf("Expected success, got error: %v", err)
	}

	if len(resp.JobStatuses) <= len(initialResp.JobStatuses) {
		t.Errorf("Expected more jobs than initial %d, got %d", len(initialResp.JobStatuses), len(resp.JobStatuses))
	}

	found := false
	for _, js := range resp.JobStatuses {
		if js.JobId == jobID {
			found = true
			if js.State != pb.JobState_JOB_STATE_FAULTED && js.State != pb.JobState_JOB_STATE_SAVED && js.State != pb.JobState_JOB_STATE_TRANSITIONING {
				t.Errorf("Unexpected job state: %v", js.State)
			}
			break
		}
	}
	if !found {
		t.Errorf("Job %s not found in status response", jobID)
	}
}

func TestServer_Health(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := googlegrpc.NewClient("passthrough://bufnet",
		googlegrpc.WithContextDialer(bufDialer),
		googlegrpc.WithTransportCredentials(insecure.NewCredentials()))
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
