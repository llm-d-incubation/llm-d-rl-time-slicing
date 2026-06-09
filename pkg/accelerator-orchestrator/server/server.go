package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/controller"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/logging"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Server implements the AcceleratorOrchestratorService gRPC server.
type Server struct {
	pb.UnimplementedAcceleratorOrchestratorServiceServer
	ctrl *controller.Controller
}

// NewServer creates a new Server instance.
func NewServer(ctrl *controller.Controller) *Server {
	return &Server{
		ctrl: ctrl,
	}
}

// Acquire implements AcceleratorOrchestratorService.Acquire.
func (s *Server) Acquire(ctx context.Context, req *pb.AcquireRequest) (*pb.AcquireResponse, error) {
	ctx = logging.WithServerMethod(ctx, "Acquire")
	ctx = logging.WithJobID(ctx, req.GetJobId())
	ctx = logging.WithGroupID(ctx, req.GetGroupId())
	slog.InfoContext(ctx, "Acquire called")
	return nil, status.Errorf(codes.Unimplemented, "method Acquire not implemented")
}

// Yield implements AcceleratorOrchestratorService.Yield.
func (s *Server) Yield(ctx context.Context, req *pb.YieldRequest) (*pb.YieldResponse, error) {
	ctx = logging.WithServerMethod(ctx, "Yield")
	ctx = logging.WithJobID(ctx, req.GetJobId())
	ctx = logging.WithGroupID(ctx, req.GetGroupId())
	slog.InfoContext(ctx, "Yield called")
	return nil, status.Errorf(codes.Unimplemented, "method Yield not implemented")
}

// ListGroups implements AcceleratorOrchestratorService.ListGroups.
func (s *Server) ListGroups(ctx context.Context, req *pb.ListGroupsRequest) (*pb.ListGroupsResponse, error) {
	ctx = logging.WithServerMethod(ctx, "ListGroups")
	slog.InfoContext(ctx, "ListGroups called")
	return nil, status.Errorf(codes.Unimplemented, "method ListGroups not implemented")
}

// GetGroupStatus implements AcceleratorOrchestratorService.GetGroupStatus.
func (s *Server) GetGroupStatus(ctx context.Context, req *pb.GetGroupStatusRequest) (*pb.GetGroupStatusResponse, error) {
	ctx = logging.WithServerMethod(ctx, "GetGroupStatus")
	ctx = logging.WithGroupID(ctx, req.GetGroupId())
	slog.InfoContext(ctx, "GetGroupStatus called")
	return nil, status.Errorf(codes.Unimplemented, "method GetGroupStatus not implemented")
}

// StartServer starts the gRPC server on the specified port and handles graceful shutdown when the context is canceled.
// It also starts the controller in the background.
func StartServer(ctx context.Context, port int, ctrl *controller.Controller, workers int) error {
	lis, err := (&net.ListenConfig{}).Listen(ctx, "tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return fmt.Errorf("failed to listen: %w", err)
	}

	// Start controller in background
	go func() {
		slog.InfoContext(ctx, "Starting controller from server", "workers", workers)
		if err := ctrl.Run(ctx, workers); err != nil {
			slog.ErrorContext(ctx, "Error running controller", "error", err)
		}
	}()

	s := grpc.NewServer()
	pb.RegisterAcceleratorOrchestratorServiceServer(s, NewServer(ctrl))

	errChan := make(chan error, 1)
	go func() {
		slog.InfoContext(ctx, "Starting gRPC server", "port", port)
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
		slog.InfoContext(ctx, "Context canceled, shutting down gRPC server gracefully")
		s.GracefulStop()
		<-errChan
		slog.InfoContext(ctx, "Server stopped")
	}

	return nil
}
