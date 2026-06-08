package server

import (
	"context"
	"fmt"
	"log"
	"net"
	"strconv"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
	sm "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/state-machine"
	podutils "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/utils"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Server implements the SnapshotAgentService gRPC server.
type Server struct {
	pb.UnimplementedSnapshotAgentServiceServer
	state          *sm.StateManager
	backendMap     map[string]backends.Backend
	defaultBackend string
}

// NewServer creates a new Server instance.
func NewServer(backendMap map[string]backends.Backend, defaultBackend string) *Server {
	return &Server{
		state:          sm.NewStateManager(),
		backendMap:     backendMap,
		defaultBackend: defaultBackend,
	}
}

// Snapshot triggers an asynchronous snapshot of the accelerator context for a job.
func (s *Server) Snapshot(ctx context.Context, req *pb.SnapshotRequest) (*pb.SnapshotResponse, error) {
	log.Printf("Snapshot called: JobID=%s, Group=%s, Backend=%s", req.GetJobId(), req.GetGroup(), req.GetBackend())

	backendName := req.GetBackend()
	if backendName == "" {
		backendName = s.defaultBackend
	}

	backend, ok := s.backendMap[backendName]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "backend %s not found", backendName)
	}

	bgCtx := context.WithoutCancel(ctx)
	opID, err := s.state.StartSnapshot(req.GetJobId(), req.GetGroup(), func() error {
		log.Printf("Background: Starting snapshot for %s using backend %s", req.GetJobId(), backendName)
		pods, err := podutils.GetLocalPods(bgCtx, req.GetJobId())
		if err != nil {
			return fmt.Errorf("failed to get local pods: %w", err)
		}

		if len(pods) == 0 {
			return fmt.Errorf("no pods found for job %s", req.GetJobId())
		}

		var allPIDs []int
		var allPIDStrings []string
		log.Printf("Pods found for job %s: %v", req.GetJobId(), pods)
		for i := range pods {
			pod := &pods[i]
			pids, err := podutils.GetPodPIDs(bgCtx, pod.Name, pod.Namespace)
			log.Printf("Pod %s has PIDs: %v", pod.Name, pids)
			if err != nil {
				return fmt.Errorf("failed to get pod PIDs: %w", err)
			}
			allPIDs = append(allPIDs, pids...)
			for _, pid := range pids {
				allPIDStrings = append(allPIDStrings, strconv.Itoa(pid))
			}
		}

		if len(allPIDStrings) == 0 {
			return fmt.Errorf("no GPU PIDs found for job %s", req.GetJobId())
		}

		err = backend.Snapshot(bgCtx, allPIDStrings)
		if err != nil {
			return fmt.Errorf("failed to snapshot job %s: %w", req.GetJobId(), err)
		}

		s.state.UpdateJobPIDs(req.GetJobId(), allPIDs)
		log.Printf("PIDs for job %s: %v", req.GetJobId(), allPIDs)
		return nil
	})
	if err != nil {
		log.Printf("Failed to start snapshot for job %s: %v", req.GetJobId(), err)
		return nil, err
	}

	return &pb.SnapshotResponse{OperationId: opID}, nil
}

// Restore triggers an asynchronous restoration of the accelerator context for a job.
func (s *Server) Restore(ctx context.Context, req *pb.RestoreRequest) (*pb.RestoreResponse, error) {
	log.Printf("Restore called: JobID=%s, Group=%s, Backend=%s", req.GetJobId(), req.GetGroup(), req.GetBackend())

	backendName := req.GetBackend()
	if backendName == "" {
		backendName = s.defaultBackend
	}

	backend, ok := s.backendMap[backendName]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "backend %s not found", backendName)
	}

	bgCtx := context.WithoutCancel(ctx)
	opID, err := s.state.StartRestore(req.GetJobId(), req.GetGroup(), func() error {
		log.Printf("Background: Starting restore for %s using backend %s", req.GetJobId(), backendName)

		pids, err := s.state.GetJobPIDs(req.GetJobId())
		if err != nil {
			return fmt.Errorf("failed to get PIDs for job %s: %w", req.GetJobId(), err)
		}

		var pidStrings []string
		for _, pid := range pids {
			pidStrings = append(pidStrings, strconv.Itoa(pid))
		}

		log.Printf("Restoring PIDs %v using backend %s", pidStrings, backendName)
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
	log.Printf("GetOperation called: OperationID=%s", req.GetOperationId())

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
	log.Printf("Status called")
	return nil, status.Errorf(codes.Unimplemented, "method Status not implemented")
}

// Health returns the health status of the agent.
func (s *Server) Health(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
	log.Printf("Health called")
	return &pb.HealthResponse{Healthy: true}, nil
}

// StartServer starts the gRPC server on the specified port.
func StartServer(port int, backendMap map[string]backends.Backend, defaultBackend string) error {
	lc := net.ListenConfig{}
	lis, err := lc.Listen(context.Background(), "tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return fmt.Errorf("failed to listen: %w", err)
	}

	s := grpc.NewServer()
	pb.RegisterSnapshotAgentServiceServer(s, NewServer(backendMap, defaultBackend))

	log.Printf("Starting gRPC server on port %d...", port)
	if err := s.Serve(lis); err != nil {
		return fmt.Errorf("failed to serve: %w", err)
	}
	return nil
}
