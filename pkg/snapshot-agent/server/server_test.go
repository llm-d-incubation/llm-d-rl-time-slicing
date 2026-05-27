package server_test

import (
	"context"
	"log"
	"net"
	"testing"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/server"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

const bufSize = 1024 * 1024

var lis *bufconn.Listener

func initGRPCServer() {
	lis = bufconn.Listen(bufSize)
	s := grpc.NewServer()
	pb.RegisterSnapshotAgentServiceServer(s, server.NewServer())
	go func() {
		if err := s.Serve(lis); err != nil {
			log.Fatalf("Server exited with error: %v", err)
		}
	}()
}

func bufDialer(context.Context, string) (net.Conn, error) {
	return lis.Dial()
}

func TestServer_Snapshot(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(bufDialer),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewSnapshotAgentServiceClient(conn)

	_, err = client.Snapshot(ctx, &pb.SnapshotRequest{
		JobId: "test-job",
		Group: "test-group",
	})
	if status.Code(err) != codes.Unimplemented {
		t.Errorf("Expected Unimplemented error, got: %v", err)
	}
}

func TestServer_Restore(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(bufDialer),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewSnapshotAgentServiceClient(conn)

	_, err = client.Restore(ctx, &pb.RestoreRequest{
		JobId: "test-job",
		Group: "test-group",
	})
	if status.Code(err) != codes.Unimplemented {
		t.Errorf("Expected Unimplemented error, got: %v", err)
	}
}

func TestServer_GetOperation(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(bufDialer),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewSnapshotAgentServiceClient(conn)

	_, err = client.GetOperation(ctx, &pb.GetOperationRequest{
		OperationId: "test-op",
	})
	if status.Code(err) != codes.Unimplemented {
		t.Errorf("Expected Unimplemented error, got: %v", err)
	}
}

func TestServer_Status(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(bufDialer),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewSnapshotAgentServiceClient(conn)

	_, err = client.Status(ctx, &pb.StatusRequest{})
	if status.Code(err) != codes.Unimplemented {
		t.Errorf("Expected Unimplemented error, got: %v", err)
	}
}

func TestServer_Health(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(bufDialer),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewSnapshotAgentServiceClient(conn)

	_, err = client.Health(ctx, &pb.HealthRequest{})
	if status.Code(err) != codes.Unimplemented {
		t.Errorf("Expected Unimplemented error, got: %v", err)
	}
}
