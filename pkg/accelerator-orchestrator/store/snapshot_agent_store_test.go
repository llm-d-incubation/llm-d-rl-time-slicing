package store_test

import (
	"context"
	"net"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type fakeAgentServer struct {
	agentpb.UnimplementedSnapshotAgentServiceServer
	statusResponse *agentpb.StatusResponse
	statusErr      error
	callCount      int32
}

func (s *fakeAgentServer) Status(ctx context.Context, req *agentpb.StatusRequest) (*agentpb.StatusResponse, error) {
	atomic.AddInt32(&s.callCount, 1)
	if s.statusErr != nil {
		return nil, s.statusErr
	}
	return s.statusResponse, nil
}

func TestGRPCSnapshotAgentStore_GetStatus(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	defer lis.Close()

	addr := lis.Addr().String()

	fakeServer := &fakeAgentServer{
		statusResponse: &agentpb.StatusResponse{
			JobStatuses: []*agentpb.JobStatus{
				{JobId: "job-1", State: agentpb.JobState_JOB_STATE_RUNNING},
			},
			AcceleratorStatuses: []*agentpb.AcceleratorStatus{
				{Id: "acc-1", MemoryUsedBytes: 100, MemoryTotalBytes: 1000},
			},
		},
	}

	grpcServer := grpc.NewServer()
	agentpb.RegisterSnapshotAgentServiceServer(grpcServer, fakeServer)

	go func() {
		_ = grpcServer.Serve(lis)
	}()
	defer grpcServer.GracefulStop()

	// Test with 5s TTL
	s := store.NewGRPCSnapshotAgentStore(5 * time.Second)
	resp, err := s.GetStatus(context.Background(), addr)
	if err != nil {
		t.Fatalf("GetStatus failed: %v", err)
	}

	if len(resp.JobStatuses) != 1 || resp.JobStatuses[0].JobId != "job-1" {
		t.Errorf("Unexpected job statuses: %v", resp.JobStatuses)
	}
	if len(resp.AcceleratorStatuses) != 1 || resp.AcceleratorStatuses[0].Id != "acc-1" {
		t.Errorf("Unexpected accelerator statuses: %v", resp.AcceleratorStatuses)
	}

	fakeServer.statusErr = status.Error(codes.Internal, "internal error")
	// Should return cached response even if server returns error now, because TTL (5s) hasn't expired.
	respCached, err := s.GetStatus(context.Background(), addr)
	if err != nil {
		t.Fatalf("Expected cached response, got error: %v", err)
	}
	if len(respCached.JobStatuses) != 1 || respCached.JobStatuses[0].JobId != "job-1" {
		t.Errorf("Unexpected cached job statuses: %v", respCached.JobStatuses)
	}
}

func TestGRPCSnapshotAgentStore_GetStatus_Caching(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	defer lis.Close()

	addr := lis.Addr().String()

	fakeServer := &fakeAgentServer{
		statusResponse: &agentpb.StatusResponse{
			JobStatuses: []*agentpb.JobStatus{
				{JobId: "job-1", State: agentpb.JobState_JOB_STATE_RUNNING},
			},
		},
	}

	grpcServer := grpc.NewServer()
	agentpb.RegisterSnapshotAgentServiceServer(grpcServer, fakeServer)

	go func() {
		_ = grpcServer.Serve(lis)
	}()
	defer grpcServer.GracefulStop()

	// Create store with 100ms TTL
	s := store.NewGRPCSnapshotAgentStore(100 * time.Millisecond)

	// 1. First call - cache miss, calls server
	resp1, err := s.GetStatus(context.Background(), addr)
	if err != nil {
		t.Fatalf("GetStatus failed: %v", err)
	}
	if atomic.LoadInt32(&fakeServer.callCount) != 1 {
		t.Errorf("Expected 1 call to server, got %d", fakeServer.callCount)
	}

	// 2. Second call (immediate) - cache hit
	resp2, err := s.GetStatus(context.Background(), addr)
	if err != nil {
		t.Fatalf("GetStatus failed: %v", err)
	}
	if atomic.LoadInt32(&fakeServer.callCount) != 1 {
		t.Errorf("Expected still 1 call to server, got %d", fakeServer.callCount)
	}

	if resp1.JobStatuses[0].JobId != resp2.JobStatuses[0].JobId {
		t.Errorf("Expected cached response to match, got %v and %v", resp1, resp2)
	}

	// 3. Wait for TTL to expire
	time.Sleep(150 * time.Millisecond)

	// 4. Third call - cache miss, calls server again
	_, err = s.GetStatus(context.Background(), addr)
	if err != nil {
		t.Fatalf("GetStatus failed: %v", err)
	}
	if atomic.LoadInt32(&fakeServer.callCount) != 2 {
		t.Errorf("Expected 2 calls to server, got %d", fakeServer.callCount)
	}
}

func TestGRPCSnapshotAgentStore_GetStatus_NoCaching(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	defer lis.Close()

	addr := lis.Addr().String()

	fakeServer := &fakeAgentServer{
		statusResponse: &agentpb.StatusResponse{
			JobStatuses: []*agentpb.JobStatus{
				{JobId: "job-1", State: agentpb.JobState_JOB_STATE_RUNNING},
			},
		},
	}

	grpcServer := grpc.NewServer()
	agentpb.RegisterSnapshotAgentServiceServer(grpcServer, fakeServer)

	go func() {
		_ = grpcServer.Serve(lis)
	}()
	defer grpcServer.GracefulStop()

	// Create store with 0 TTL (no caching)
	s := store.NewGRPCSnapshotAgentStore(0)

	// 1. First call - calls server
	_, err = s.GetStatus(context.Background(), addr)
	if err != nil {
		t.Fatalf("GetStatus failed: %v", err)
	}
	if atomic.LoadInt32(&fakeServer.callCount) != 1 {
		t.Errorf("Expected 1 call to server, got %d", fakeServer.callCount)
	}

	// 2. Second call - should call server again because TTL is 0
	_, err = s.GetStatus(context.Background(), addr)
	if err != nil {
		t.Fatalf("GetStatus failed: %v", err)
	}
	if atomic.LoadInt32(&fakeServer.callCount) != 2 {
		t.Errorf("Expected 2 calls to server, got %d", fakeServer.callCount)
	}
}
