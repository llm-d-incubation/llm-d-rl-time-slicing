package server_test

import (
	"context"
	"net"
	"testing"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/server"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	"k8s.io/klog/v2"
)

const bufSize = 1024 * 1024

var lis *bufconn.Listener

func initGRPCServer() {
	lis = bufconn.Listen(bufSize)
	s := grpc.NewServer()

	noopBackend := backends.NewNoopBackend()
	backendsMap := map[backends.BackendType]backends.Backend{
		backends.BackendNoop: noopBackend,
	}

	pb.RegisterSnapshotAgentServiceServer(s, server.NewServer(backendsMap, backends.BackendNoop))
	go func() {
		if err := s.Serve(lis); err != nil {
			klog.Fatalf("Server exited with error: %v", err)
		}
	}()
}

func bufDialer(context.Context, string) (net.Conn, error) {
	return lis.Dial()
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

	_, err = client.Snapshot(ctx, &pb.SnapshotRequest{
		JobId:   "test-job",
		Group:   "test-group",
		Backend: pb.Backend_BACKEND_UNSPECIFIED,
	})
	if err != nil {
		t.Errorf("Expected success (using default noop backend), got error: %v", err)
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

	_, err = client.Restore(ctx, &pb.RestoreRequest{
		JobId:   "test-job",
		Group:   "test-group",
		Backend: pb.Backend_BACKEND_UNSPECIFIED,
	})
	if err != nil {
		t.Errorf("Expected success (using default noop backend), got error: %v", err)
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

	resp, err := client.Status(ctx, &pb.StatusRequest{})
	if err != nil {
		t.Errorf("Expected success, got error: %v", err)
	}

	if len(resp.JobStatuses) != 0 {
		t.Errorf("Expected no jobs, got %d", len(resp.JobStatuses))
	}

	jobID := "test-job-status"
	_, err = client.Snapshot(ctx, &pb.SnapshotRequest{
		JobId:   jobID,
		Group:   "test-group",
		Backend: pb.Backend_BACKEND_UNSPECIFIED,
	})
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}

	resp, err = client.Status(ctx, &pb.StatusRequest{})
	if err != nil {
		t.Errorf("Expected success, got error: %v", err)
	}

	found := false
	for _, js := range resp.JobStatuses {
		if js.JobId == jobID {
			found = true
			if js.State != pb.JobState_JOB_STATE_FAULTED && js.State != pb.JobState_JOB_STATE_SAVED {
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
	conn, err := grpc.NewClient("passthrough://bufnet",
		grpc.WithContextDialer(bufDialer),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewSnapshotAgentServiceClient(conn)

	_, err = client.Health(ctx, &pb.HealthRequest{})
	if err != nil {
		t.Errorf("Expected success, got error: %v", err)
	}
}
