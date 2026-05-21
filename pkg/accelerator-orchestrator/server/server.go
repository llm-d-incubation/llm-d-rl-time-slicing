package server

import (
	"context"
	"errors"
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

// StartServer starts the gRPC server on the specified port and handles graceful shutdown when the context is canceled.
func StartServer(ctx context.Context, port int) error {
	lis, err := (&net.ListenConfig{}).Listen(ctx, "tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return fmt.Errorf("failed to listen: %w", err)
	}

	s := grpc.NewServer()
	pb.RegisterAcceleratorOrchestratorServiceServer(s, NewServer())

	errChan := make(chan error, 1)
	go func() {
		log.Printf("Starting gRPC server on port %d...", port)
		if err := s.Serve(lis); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
			errChan <- fmt.Errorf("failed to serve: %w", err)
		}
		close(errChan)
	}()

	select {
	case err := <-errChan:
		if err != nil {
			return err
		}
	case <-ctx.Done():
		log.Println("Context canceled, shutting down gRPC server gracefully...")
		s.GracefulStop()
		<-errChan
		log.Println("Server stopped")
	}

	return nil
}
