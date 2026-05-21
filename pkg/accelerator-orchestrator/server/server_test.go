package server

import (
	"context"
	"log"
	"net"
	"testing"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
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
	pb.RegisterAcceleratorOrchestratorServiceServer(s, NewServer())
	go func() {
		if err := s.Serve(lis); err != nil {
			log.Fatalf("Server exited with error: %v", err)
		}
	}()
}

func bufDialer(context.Context, string) (net.Conn, error) {
	return lis.Dial()
}

func TestServer_Acquire(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := grpc.DialContext(ctx, "bufnet", grpc.WithContextDialer(bufDialer), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewAcceleratorOrchestratorServiceClient(conn)

	_, err = client.Acquire(ctx, &pb.AcquireRequest{
		JobId:   "test-job",
		GroupId: "test-group",
	})
	if status.Code(err) != codes.Unimplemented {
		t.Errorf("Expected Unimplemented error, got: %v", err)
	}
}

func TestServer_Yield(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := grpc.DialContext(ctx, "bufnet", grpc.WithContextDialer(bufDialer), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewAcceleratorOrchestratorServiceClient(conn)

	_, err = client.Yield(ctx, &pb.YieldRequest{
		JobId:   "test-job",
		GroupId: "test-group",
	})
	if status.Code(err) != codes.Unimplemented {
		t.Errorf("Expected Unimplemented error, got: %v", err)
	}
}

func TestServer_Heartbeat(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := grpc.DialContext(ctx, "bufnet", grpc.WithContextDialer(bufDialer), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewAcceleratorOrchestratorServiceClient(conn)

	_, err = client.Heartbeat(ctx, &pb.HeartbeatRequest{
		JobId:   "test-job",
		GroupId: "test-group",
	})
	if status.Code(err) != codes.Unimplemented {
		t.Errorf("Expected Unimplemented error, got: %v", err)
	}
}

func TestServer_ListGroups(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := grpc.DialContext(ctx, "bufnet", grpc.WithContextDialer(bufDialer), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewAcceleratorOrchestratorServiceClient(conn)

	_, err = client.ListGroups(ctx, &pb.ListGroupsRequest{})
	if status.Code(err) != codes.Unimplemented {
		t.Errorf("Expected Unimplemented error, got: %v", err)
	}
}

func TestServer_GetGroupStatus(t *testing.T) {
	initGRPCServer()
	ctx := context.Background()
	conn, err := grpc.DialContext(ctx, "bufnet", grpc.WithContextDialer(bufDialer), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	defer conn.Close()
	client := pb.NewAcceleratorOrchestratorServiceClient(conn)

	_, err = client.GetGroupStatus(ctx, &pb.GetGroupStatusRequest{
		GroupId: "test-group",
	})
	if status.Code(err) != codes.Unimplemented {
		t.Errorf("Expected Unimplemented error, got: %v", err)
	}
}
