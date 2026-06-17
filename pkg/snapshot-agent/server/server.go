package server

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"strconv"
	"time"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/logging"
	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
	sm "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/state-machine"
	podutils "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/utils"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
	v1 "k8s.io/api/core/v1"
)

// Server implements the SnapshotAgentService gRPC server.
type Server struct {
	pb.UnimplementedSnapshotAgentServiceServer
	state          *sm.StateManager
	backendMap     map[backends.BackendType]backends.Backend
	defaultBackend backends.BackendType
}

// NewServer creates a new Server instance.
func NewServer(backendMap map[backends.BackendType]backends.Backend, defaultBackend backends.BackendType) *Server {
	return &Server{
		state:          sm.NewStateManager(),
		backendMap:     backendMap,
		defaultBackend: defaultBackend,
	}
}

func (s *Server) getBackendType(backend pb.Backend) backends.BackendType {
	switch backend {
	case pb.Backend_BACKEND_CUDA:
		return backends.BackendCuda
	default:
		return s.defaultBackend
	}
}

// Snapshot triggers an asynchronous snapshot of the accelerator context for a job.
func (s *Server) Snapshot(ctx context.Context, req *pb.SnapshotRequest) (*pb.SnapshotResponse, error) {
	ctx = logging.WithServerMethod(ctx, "Snapshot")
	ctx = logging.WithJobID(ctx, req.GetJobId())
	ctx = logging.WithGroupID(ctx, req.GetGroup())

	slog.InfoContext(ctx, "Snapshot called", "backend", req.GetBackend())

	backendType := s.getBackendType(req.GetBackend())

	backend, ok := s.backendMap[backendType]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "backend %s not found", backendType)
	}

	bgCtx := context.WithoutCancel(ctx)
	opID, err := s.state.StartSnapshot(req.GetJobId(), req.GetGroup(), func() error {
		slog.InfoContext(bgCtx, "Background: Starting snapshot", "backend", backendType)
		pods, err := podutils.GetLocalPods(bgCtx, req.GetJobId())
		if err != nil {
			return fmt.Errorf("failed to get local pods: %w", err)
		}

		if len(pods) == 0 {
			return fmt.Errorf("no pods found for job %s", req.GetJobId())
		}

		allPIDs, allPIDStrings, err := getPIDsFromPods(bgCtx, pods)
		if err != nil {
			return fmt.Errorf("failed to get PIDs from pods: %w", err)
		}

		if len(allPIDStrings) == 0 {
			return fmt.Errorf("no GPU PIDs found for job %s", req.GetJobId())
		}

		err = backend.Snapshot(bgCtx, allPIDStrings)
		if err != nil {
			return fmt.Errorf("failed to snapshot job %s: %w", req.GetJobId(), err)
		}

		s.state.UpdateJobPIDs(req.GetJobId(), allPIDs)
		slog.InfoContext(bgCtx, "PIDs for job", "pids", allPIDs)
		return nil
	})
	if err != nil {
		slog.ErrorContext(ctx, "Failed to start snapshot", "error", err)
		return nil, err
	}

	return &pb.SnapshotResponse{OperationId: opID}, nil
}

// Restore triggers an asynchronous restoration of the accelerator context for a job.
func (s *Server) Restore(ctx context.Context, req *pb.RestoreRequest) (*pb.RestoreResponse, error) {
	ctx = logging.WithServerMethod(ctx, "Restore")
	ctx = logging.WithJobID(ctx, req.GetJobId())
	ctx = logging.WithGroupID(ctx, req.GetGroup())

	slog.InfoContext(ctx, "Restore called", "backend", req.GetBackend())

	backendType := s.getBackendType(req.GetBackend())

	backend, ok := s.backendMap[backendType]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "backend %s not found", backendType)
	}

	bgCtx := context.WithoutCancel(ctx)
	opID, err := s.state.StartRestore(req.GetJobId(), req.GetGroup(), func() error {
		slog.InfoContext(bgCtx, "Background: Starting restore", "backend", backendType)

		pids, err := s.state.GetJobPIDs(req.GetJobId())
		if err != nil {
			return fmt.Errorf("failed to get PIDs for job %s: %w", req.GetJobId(), err)
		}

		var pidStrings []string
		for _, pid := range pids {
			pidStrings = append(pidStrings, strconv.Itoa(pid))
		}

		slog.InfoContext(bgCtx, "Restoring PIDs", "pids", pidStrings, "backend", backendType)
		if err := backend.Restore(bgCtx, pidStrings); err != nil {
			return fmt.Errorf("failed to restore job %s: %w", req.GetJobId(), err)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	return &pb.RestoreResponse{OperationId: opID}, nil
}

// GetOperation polls the status of a long-running snapshot or restore operation.
func (s *Server) GetOperation(ctx context.Context, req *pb.GetOperationRequest) (*pb.GetOperationResponse, error) {
	ctx = logging.WithServerMethod(ctx, "GetOperation")
	slog.InfoContext(ctx, "GetOperation called", "operationID", req.GetOperationId())

	op, ok := s.state.GetOperation(req.GetOperationId())
	if !ok {
		return nil, status.Errorf(codes.NotFound, "operation %s not found", req.GetOperationId())
	}

	elapsed := time.Since(op.StartedAt).Milliseconds()
	if !op.FinishedAt.IsZero() {
		elapsed = op.FinishedAt.Sub(op.StartedAt).Milliseconds()
	}

	resp := &pb.GetOperationResponse{
		Status:    op.Status,
		ElapsedMs: elapsed,
	}

	if op.Status == pb.OperationStatus_OPERATION_STATUS_COMPLETE {
		storageBytes := op.StorageBytes
		deviceBytes := op.SnapshotDeviceBytes
		resp.StorageBytes = &storageBytes
		resp.SnapshotDeviceBytes = &deviceBytes
	}

	if op.Status == pb.OperationStatus_OPERATION_STATUS_FAILED {
		errStr := op.Error
		resp.Error = &errStr
	}

	return resp, nil
}

// Status returns the current state of jobs and accelerators on the node.
func (s *Server) Status(ctx context.Context, req *pb.StatusRequest) (*pb.StatusResponse, error) {
	ctx = logging.WithServerMethod(ctx, "Status")
	slog.InfoContext(ctx, "Status called")
	return &pb.StatusResponse{
		JobStatuses: s.state.GetJobStatus(),
		// TODO: Implement accelerator status discovery
		AcceleratorStatuses: nil,
	}, nil
}

// HealthServer implements the standard gRPC health service.
type HealthServer struct {
	grpc_health_v1.UnimplementedHealthServer
	backendMap     map[backends.BackendType]backends.Backend
	defaultBackend backends.BackendType
}

func NewHealthServer(backendMap map[backends.BackendType]backends.Backend, defaultBackend backends.BackendType) *HealthServer {
	return &HealthServer{
		backendMap:     backendMap,
		defaultBackend: defaultBackend,
	}
}

func (h *HealthServer) Check(
	ctx context.Context,
	req *grpc_health_v1.HealthCheckRequest,
) (*grpc_health_v1.HealthCheckResponse, error) {
	ctx = logging.WithServerMethod(ctx, "Check")
	backendType := backends.BackendType(req.Service)
	if req.Service == "" {
		backendType = h.defaultBackend
	}

	backend, ok := h.backendMap[backendType]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "backend %s not found", backendType)
	}

	if err := backend.HealthCheck(ctx); err != nil {
		slog.ErrorContext(ctx, "HealthCheck failed", "backend", backendType, "error", err)
		return &grpc_health_v1.HealthCheckResponse{
			Status: grpc_health_v1.HealthCheckResponse_NOT_SERVING,
		}, nil
	}

	return &grpc_health_v1.HealthCheckResponse{
		Status: grpc_health_v1.HealthCheckResponse_SERVING,
	}, nil
}

func (h *HealthServer) Watch(req *grpc_health_v1.HealthCheckRequest, stream grpc_health_v1.Health_WatchServer) error {
	return status.Errorf(codes.Unimplemented, "method Watch not implemented")
}

// StartServer starts the gRPC server on the specified port.
func StartServer(
	ctx context.Context,
	port int,
	backendMap map[backends.BackendType]backends.Backend,
	defaultBackend backends.BackendType,
) error {
	lc := net.ListenConfig{}
	lis, err := lc.Listen(ctx, "tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return fmt.Errorf("failed to listen: %w", err)
	}

	s := grpc.NewServer()
	pb.RegisterSnapshotAgentServiceServer(s, NewServer(backendMap, defaultBackend))
	grpc_health_v1.RegisterHealthServer(s, NewHealthServer(backendMap, defaultBackend))

	slog.InfoContext(ctx, "Starting gRPC server", "port", port)
	if err := s.Serve(lis); err != nil {
		return fmt.Errorf("failed to serve: %w", err)
	}
	return nil
}

//nolint:gocritic // The project configuration bans named returns, conflicting with this rule
func getPIDsFromPods(ctx context.Context, pods []v1.Pod) ([]int, []string, error) {
	var allPIDs []int
	var allPIDStrings []string
	for i := range pods {
		pod := &pods[i]
		pids, err := podutils.GetPodPIDs(ctx, pod.Name, pod.Namespace)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to get pod PIDs: %w", err)
		}
		allPIDs = append(allPIDs, pids...)
		for _, pid := range pids {
			allPIDStrings = append(allPIDStrings, strconv.Itoa(pid))
		}
	}
	return allPIDs, allPIDStrings, nil
}
