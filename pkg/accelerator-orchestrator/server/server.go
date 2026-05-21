package server

import (
	"context"
	"fmt"
	"log"
	"net"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Server implements the AcceleratorOrchestratorService gRPC server.
type Server struct {
	pb.UnimplementedAcceleratorOrchestratorServiceServer
}

// NewServer creates a new Server instance.
func NewServer() *Server {
	return &Server{}
}

// Acquire implements AcceleratorOrchestratorService.Acquire.
func (s *Server) Acquire(ctx context.Context, req *pb.AcquireRequest) (*pb.AcquireResponse, error) {
	log.Printf("Acquire called: JobID=%s, GroupID=%s", req.GetJobId(), req.GetGroupId())
	return nil, status.Errorf(codes.Unimplemented, "method Acquire not implemented")
}

// Yield implements AcceleratorOrchestratorService.Yield.
func (s *Server) Yield(ctx context.Context, req *pb.YieldRequest) (*pb.YieldResponse, error) {
	log.Printf("Yield called: JobID=%s, GroupID=%s", req.GetJobId(), req.GetGroupId())
	return nil, status.Errorf(codes.Unimplemented, "method Yield not implemented")
}

// Heartbeat implements AcceleratorOrchestratorService.Heartbeat.
func (s *Server) Heartbeat(ctx context.Context, req *pb.HeartbeatRequest) (*pb.HeartbeatResponse, error) {
	log.Printf("Heartbeat called: JobID=%s, GroupID=%s", req.GetJobId(), req.GetGroupId())
	return nil, status.Errorf(codes.Unimplemented, "method Heartbeat not implemented")
}

// ListGroups implements AcceleratorOrchestratorService.ListGroups.
func (s *Server) ListGroups(ctx context.Context, req *pb.ListGroupsRequest) (*pb.ListGroupsResponse, error) {
	log.Printf("ListGroups called")
	return nil, status.Errorf(codes.Unimplemented, "method ListGroups not implemented")
}

// GetGroupStatus implements AcceleratorOrchestratorService.GetGroupStatus.
func (s *Server) GetGroupStatus(ctx context.Context, req *pb.GetGroupStatusRequest) (*pb.GetGroupStatusResponse, error) {
	log.Printf("GetGroupStatus called: GroupID=%s", req.GetGroupId())
	return nil, status.Errorf(codes.Unimplemented, "method GetGroupStatus not implemented")
}

// StartServer starts the gRPC server on the specified port.
func StartServer(port int) error {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return fmt.Errorf("failed to listen: %v", err)
	}

	s := grpc.NewServer()
	pb.RegisterAcceleratorOrchestratorServiceServer(s, NewServer())

	log.Printf("Starting gRPC server on port %d...", port)
	if err := s.Serve(lis); err != nil {
		return fmt.Errorf("failed to serve: %v", err)
	}
	return nil
}
