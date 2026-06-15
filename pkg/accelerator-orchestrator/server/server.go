package server

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/controller"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// GroupStore defines the interface for group store operations needed by the server.
type GroupStore interface {
	Get(ctx context.Context, id string) (*store.Group, error)
	List(ctx context.Context) ([]*store.Group, error)
}

// JobStore defines the interface for job store operations needed by the server.
type JobStore interface {
	Get(ctx context.Context, groupID, jobID string) (*store.Job, error)
	ListByGroup(ctx context.Context, groupID string) ([]*store.Job, error)
}

// Server implements the AcceleratorOrchestratorService gRPC server.
type Server struct {
	pb.UnimplementedAcceleratorOrchestratorServiceServer
	ctrl       *controller.Controller
	groupStore GroupStore
	jobStore   JobStore
}

// NewServer creates a new Server instance.
func NewServer(ctrl *controller.Controller, groupStore GroupStore, jobStore JobStore) *Server {
	return &Server{
		ctrl:       ctrl,
		groupStore: groupStore,
		jobStore:   jobStore,
	}
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

// ListGroups implements AcceleratorOrchestratorService.ListGroups.
func (s *Server) ListGroups(ctx context.Context, req *pb.ListGroupsRequest) (*pb.ListGroupsResponse, error) {
	log.Printf("ListGroups called")
	groups, err := s.groupStore.List(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list groups: %v", err)
	}
	var ids []string
	for _, g := range groups {
		ids = append(ids, g.ID())
	}
	return &pb.ListGroupsResponse{GroupIds: ids}, nil
}

// GetGroupStatus implements AcceleratorOrchestratorService.GetGroupStatus.
func (s *Server) GetGroupStatus(ctx context.Context, req *pb.GetGroupStatusRequest) (*pb.GetGroupStatusResponse, error) {
	log.Printf("GetGroupStatus called: GroupID=%s", req.GetGroupId())

	group, err := s.groupStore.Get(ctx, req.GetGroupId())
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return nil, status.Errorf(codes.NotFound, "group %s not found", req.GetGroupId())
		}
		return nil, status.Errorf(codes.Internal, "failed to get group: %v", err)
	}

	snap := group.Snapshot()

	if snap.State == pb.GroupStatus_STATE_UNKNOWN {
		return nil, status.Errorf(codes.Unavailable, "group %s state is unknown", req.GetGroupId())
	}

	groupStatus := &pb.GroupStatus{
		GroupId:          snap.ID,
		GroupState:       snap.State,
		StateTimestamp:   timestamppb.New(snap.StateTimestamp),
		LockingJob:       snap.LockingJob,
		ActiveJob:        snap.ActiveJob,
		WaiterQueueDepth: int64(snap.WaiterQueueDepth),
	}

	jobs, err := s.jobStore.ListByGroup(ctx, group.ID())
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list jobs for group %s: %v", group.ID(), err)
	}

	var agentJobStates []*pb.SnapshotAgentJobState
	for _, job := range jobs {
		for agent, jobState := range job.ContextState() {
			agentJobStates = append(agentJobStates, &pb.SnapshotAgentJobState{
				Agent:    agent,
				JobState: jobState,
				JobId:    job.JobID(),
			})
		}
	}

	return &pb.GetGroupStatusResponse{
		Group:          groupStatus,
		AgentJobStates: agentJobStates,
	}, nil
}

// StartServer starts the gRPC server on the specified port and handles graceful shutdown when the context is canceled.
// It also starts the controller in the background.
func StartServer(
	ctx context.Context,
	port int,
	ctrl *controller.Controller,
	groupStore GroupStore,
	jobStore JobStore,
	workers int,
) error {
	lis, err := (&net.ListenConfig{}).Listen(ctx, "tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return fmt.Errorf("failed to listen: %w", err)
	}

	// Start controller in background
	go func() {
		log.Printf("Starting controller from server with %d workers...", workers)
		if err := ctrl.Run(ctx, workers); err != nil {
			log.Printf("Error running controller: %v", err)
		}
	}()

	s := grpc.NewServer()
	pb.RegisterAcceleratorOrchestratorServiceServer(s, NewServer(ctrl, groupStore, jobStore))

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
