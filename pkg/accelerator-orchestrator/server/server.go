package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/controller"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/logging"
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

const acquirePollInterval = 1 * time.Second

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
	ctx = logging.WithServerMethod(ctx, "Acquire")
	ctx = logging.WithJobID(ctx, req.GetJobId())
	ctx = logging.WithGroupID(ctx, req.GetGroupId())
	slog.InfoContext(ctx, "Acquire called")

	groupID := req.GetGroupId()
	jobID := req.GetJobId()
	startTime := time.Now()

	// 1. Get Group
	group, err := s.groupStore.Get(ctx, groupID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return nil, status.Errorf(codes.NotFound, "group %s not found", groupID)
		}
		return nil, status.Errorf(codes.Internal, "failed to get group: %v", err)
	}

	// 2. Request Lock
	group.Spec().RequestLock(jobID)
	if s.ctrl != nil {
		s.ctrl.EnqueueWork(groupID)
	}

	// 3. Wait Loop
	ticker := time.NewTicker(acquirePollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			slog.InfoContext(ctx, "Acquire context cancelled", "error", ctx.Err())
			return nil, status.FromContextError(ctx.Err()).Err()
		case <-ticker.C:
			// Check if group is faulted
			faulted, err := s.isGroupFaulted(ctx, groupID)
			if err != nil {
				return nil, status.Errorf(codes.Internal, "failed to check if group is faulted: %v", err)
			}
			if faulted {
				return nil, status.Errorf(codes.Unavailable, "group %s is faulted", groupID)
			}

			// Check if loaded job is ours
			if group.Status().LoadedJob() == jobID {
				slog.InfoContext(ctx, "Acquire succeeded, job loaded")
				return &pb.AcquireResponse{
					Success:         true,
					ContextRestored: true, // Default to true, as we don't have enough info to determine if it was zero-overhead
					WaitedMs:        time.Since(startTime).Milliseconds(),
				}, nil
			}
		}
	}
}

func (s *Server) isGroupFaulted(ctx context.Context, groupID string) (bool, error) {
	jobs, err := s.jobStore.ListByGroup(ctx, groupID)
	if err != nil {
		return false, err
	}
	for _, job := range jobs {
		for _, state := range job.ContextState() {
			if state == pb.SnapshotAgentJobState_STATE_FAULTED {
				return true, nil
			}
		}
	}
	return false, nil
}

// Yield implements AcceleratorOrchestratorService.Yield.
func (s *Server) Yield(ctx context.Context, req *pb.YieldRequest) (*pb.YieldResponse, error) {
	ctx = logging.WithServerMethod(ctx, "Yield")
	ctx = logging.WithJobID(ctx, req.GetJobId())
	ctx = logging.WithGroupID(ctx, req.GetGroupId())
	slog.InfoContext(ctx, "Yield called")

	groupID := req.GetGroupId()
	jobID := req.GetJobId()

	// 1. Get Group
	group, err := s.groupStore.Get(ctx, groupID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return nil, status.Errorf(codes.NotFound, "group %s not found", groupID)
		}
		return nil, status.Errorf(codes.Internal, "failed to get group: %v", err)
	}

	// 2. Take Snapshot BEFORE Yield
	snap := group.Snapshot()

	// 3. Call Yield
	err = group.Spec().Yield(ctx, jobID)
	if err != nil {
		if errors.Is(err, store.ErrNotLockHolder) {
			return nil, status.Errorf(codes.PermissionDenied, "job %s does not hold lock for group %s", jobID, groupID)
		}
		return nil, status.Errorf(codes.Internal, "failed to yield: %v", err)
	}

	if s.ctrl != nil {
		s.ctrl.EnqueueWork(groupID)
	}

	// 4. Construct Response from Snapshot
	numWaiters := snap.WaiterQueueDepth
	return &pb.YieldResponse{
		Success:          true,
		PendingWaiters:   int64(numWaiters),
		SnapshotDeferred: numWaiters == 0,
	}, nil
}

// ListGroups implements AcceleratorOrchestratorService.ListGroups.
func (s *Server) ListGroups(ctx context.Context, req *pb.ListGroupsRequest) (*pb.ListGroupsResponse, error) {
	ctx = logging.WithServerMethod(ctx, "ListGroups")
	slog.InfoContext(ctx, "ListGroups called")
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
	ctx = logging.WithServerMethod(ctx, "GetGroupStatus")
	ctx = logging.WithGroupID(ctx, req.GetGroupId())
	slog.InfoContext(ctx, "GetGroupStatus called")

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
		slog.InfoContext(ctx, "Starting controller from server", "workers", workers)
		if err := ctrl.Run(ctx, workers); err != nil {
			slog.ErrorContext(ctx, "Error running controller", "error", err)
		}
	}()

	s := grpc.NewServer()
	pb.RegisterAcceleratorOrchestratorServiceServer(s, NewServer(ctrl, groupStore, jobStore))

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
