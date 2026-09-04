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
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/features"
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
	state           *sm.StateManager
	backendMap      map[backends.BackendType]backends.Backend
	defaultBackend  backends.BackendType
	deploymentMode  string
	channelRegistry *backends.ChannelRegistry
	featureGates    features.Gates
}

// NewServer creates a new Server instance. channelRegistry is shared with
// the app-channel backend so workloads registered through the
// WorkloadChannel RPC are reachable by Snapshot/Restore. featureGates
// selects which experimental capabilities are enabled; nil means every
// gate at its default (experimental capabilities off).
func NewServer(
	backendMap map[backends.BackendType]backends.Backend,
	defaultBackend backends.BackendType,
	deploymentMode string,
	channelRegistry *backends.ChannelRegistry,
	featureGates features.Gates,
) *Server {
	return &Server{
		state:           sm.NewStateManager(),
		backendMap:      backendMap,
		defaultBackend:  defaultBackend,
		deploymentMode:  deploymentMode,
		channelRegistry: channelRegistry,
		featureGates:    featureGates,
	}
}

// checkFeatureGates rejects configs that select an experimental backend
// whose feature gate is off. It runs before backend routing so a gated
// config never silently falls through to another backend.
func (s *Server) checkFeatureGates(config *pb.BackendConfig) error {
	if config.GetDirectMemory() != nil && !s.featureGates.Enabled(features.DirectMemoryBackend) {
		return status.Errorf(codes.FailedPrecondition,
			"the direct_memory backend is experimental (driven by GPU-CR, which carries deployment "+
				"requirements and operational caveats) and is disabled by default; restart the agent "+
				"with --feature-gates=%s=true (or the FEATURE_GATES env var) to enable it",
			features.DirectMemoryBackend)
	}
	if config.GetMemoryRegions() != nil && !s.featureGates.Enabled(features.MemoryRegionsBackend) {
		return status.Errorf(codes.FailedPrecondition,
			"the memory_regions backend is experimental (driven by GPU-CR, which carries deployment "+
				"requirements and operational caveats) and is disabled by default; restart the agent "+
				"with --feature-gates=%s=true (or the FEATURE_GATES env var) to enable it",
			features.MemoryRegionsBackend)
	}
	return nil
}

// Snapshot triggers an asynchronous snapshot of the accelerator context for a job.
func (s *Server) Snapshot(ctx context.Context, req *pb.SnapshotRequest) (*pb.SnapshotResponse, error) {
	ctx = logging.WithServerMethod(ctx, "Snapshot")
	ctx = logging.WithJobID(ctx, req.GetJobId())
	ctx = logging.WithGroupID(ctx, req.GetGroup())

	if err := s.checkFeatureGates(req.GetBackendConfig()); err != nil {
		return nil, err
	}

	backendType := s.getSnapshotBackendType(req.GetBackendConfig())
	slog.InfoContext(ctx, "Snapshot called", "backend", backendType)

	backend, ok := s.backendMap[backendType]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "backend %s not found", backendType)
	}

	s.ensureJobRunningIfGPUOccupied(ctx, req.GetJobId(), req.GetGroup())

	bgCtx := context.WithoutCancel(ctx)
	config := req.GetBackendConfig()

	snapshotFn, fnErr := s.buildSnapshotFn(bgCtx, req.GetJobId(), backendType, backend, config)
	if fnErr != nil {
		return nil, fnErr
	}

	slot := memoryRegionsSlot(config, req.GetJobId())
	opID, err := s.state.StartSnapshotSlot(req.GetJobId(), req.GetGroup(), slot, snapshotFn)
	if err != nil {
		slog.ErrorContext(ctx, "Failed to start snapshot", "error", err)
		return nil, err
	}

	return &pb.SnapshotResponse{OperationId: opID}, nil
}

func (s *Server) getSnapshotBackendType(config *pb.BackendConfig) backends.BackendType {
	if config == nil {
		return s.defaultBackend
	}
	if config.GetCuda() != nil {
		return backends.BackendCuda
	}
	if config.GetDirectMemory() != nil {
		return backends.BackendDirectMemory
	}
	if config.GetAppEndpoint() != nil {
		return backends.BackendAppEndpoint
	}
	if config.GetAppChannel() != nil {
		return backends.BackendAppChannel
	}
	if config.GetMemoryRegions() != nil {
		return backends.BackendMemoryRegions
	}
	return s.defaultBackend
}

// ensureJobRunningIfGPUOccupied registers the job and, if it is IDLE while
// the GPU has running compute processes, transitions it to RUNNING.
//
// Standalone mode only: in k8s mode the watcher is the single source of
// state-machine transitions (and additionally binds jobs to their targets,
// e.g. PIDs for the CUDA backend — a backend-specific concern that a future
// discovery interface will own per backend).
func (s *Server) ensureJobRunningIfGPUOccupied(ctx context.Context, jobID, group string) {
	if s.deploymentMode != "standalone" {
		return
	}
	s.state.RegisterJob(jobID, group)
	statuses := s.state.GetJobStatus()
	for _, js := range statuses {
		if js.JobId == jobID && js.State == pb.JobState_JOB_STATE_IDLE {
			occupied, err := podutils.HasGPUProcesses(ctx)
			if err != nil {
				slog.WarnContext(ctx, "NVML check failed, skipping auto-transition", "error", err)
				return
			}
			if occupied {
				slog.InfoContext(ctx, "GPU occupied, transitioning job to RUNNING", "jobID", jobID)
				if err := s.state.TransitionToRunning(jobID, nil); err != nil {
					slog.WarnContext(ctx, "Failed to auto-transition job", "jobID", jobID, "error", err)
				}
			}
			break
		}
	}
}

// buildSnapshotFn returns the background snapshot function for the given
// deployment mode and backend. In standalone mode the caller-provided
// BackendConfig is passed through to the backend as-is. In k8s mode, the CUDA
// backend needs PID discovery (resolve PIDs from pods via the watcher's job
// labels, then cache them for restore); application-aware backends resolve
// their targets themselves (HTTP endpoints from the config, or the workload
// channel registered under the job ID), so the config is passed through.
func (s *Server) buildSnapshotFn(
	bgCtx context.Context,
	jobID string,
	backendType backends.BackendType,
	backend backends.Backend,
	config *pb.BackendConfig,
) (func() error, error) {
	switch s.deploymentMode {
	case "standalone":
		return func() error {
			slog.InfoContext(bgCtx, "Background: Starting snapshot", "backend", backendType)
			return backend.Snapshot(bgCtx, backends.Request{JobID: jobID, Config: config})
		}, nil
	case "k8s":
		switch backendType {
		case backends.BackendCuda, backends.BackendDirectMemory:
			explicitPIDs := extractExplicitPIDs(config)
			return func() error {
				slog.InfoContext(bgCtx, "Background: Starting snapshot", "backend", backendType)
				allPIDs, allPIDStrings, pidErr := resolvePIDs(bgCtx, jobID, explicitPIDs)
				if pidErr != nil {
					return pidErr
				}
				var reqConfig *pb.BackendConfig
				var cfgErr error
				if backendType == backends.BackendDirectMemory {
					reqConfig, cfgErr = backends.BuildDirectMemoryConfig(allPIDStrings)
				} else {
					reqConfig = backends.BuildCudaConfig(allPIDStrings)
				}
				if cfgErr != nil {
					return fmt.Errorf("failed to build backend config for job %s: %w", jobID, cfgErr)
				}
				req := backends.Request{JobID: jobID, Config: reqConfig}
				if err := backend.Snapshot(bgCtx, req); err != nil {
					return fmt.Errorf("failed to snapshot job %s: %w", jobID, err)
				}
				s.state.UpdateJobPIDs(jobID, allPIDs)
				return nil
			}, nil
		case backends.BackendAppEndpoint, backends.BackendAppChannel:
			return func() error {
				slog.InfoContext(bgCtx, "Background: Starting snapshot", "backend", backendType)
				return backend.Snapshot(bgCtx, backends.Request{JobID: jobID, Config: config})
			}, nil
		case backends.BackendMemoryRegions:
			// Regions are explicit in the config; no PID discovery. The
			// config is passed through, and the region PIDs are cached for
			// Status parity with the CUDA backend.
			return func() error {
				slog.InfoContext(bgCtx, "Background: Starting snapshot", "backend", backendType)
				if err := backend.Snapshot(bgCtx, backends.Request{JobID: jobID, Config: config}); err != nil {
					return fmt.Errorf("failed to snapshot job %s: %w", jobID, err)
				}
				s.state.UpdateJobPIDs(jobID, pidsFromMemoryRegions(config))
				return nil
			}, nil
		default:
			return nil, status.Errorf(codes.InvalidArgument, "backend %q is not supported in k8s mode", backendType)
		}
	default:
		return nil, status.Errorf(codes.InvalidArgument, "unknown deployment mode %q", s.deploymentMode)
	}
}

// memoryRegionsSlot returns the snapshot slot a memory-regions request
// targets (snapshot_name, defaulting to the job ID), or "" for any other
// backend config. Slot-aware state transitions (live slot swap, fault
// recovery) only apply when this is non-empty.
func memoryRegionsSlot(config *pb.BackendConfig, jobID string) string {
	cfg := config.GetMemoryRegions()
	if cfg == nil {
		return ""
	}
	if s := cfg.GetSnapshotName(); s != "" {
		return s
	}
	return jobID
}

// pidsFromMemoryRegions returns the deduplicated PIDs referenced by a
// memory-regions BackendConfig, in first-appearance order.
func pidsFromMemoryRegions(config *pb.BackendConfig) []int {
	cfg := config.GetMemoryRegions()
	if cfg == nil {
		return nil
	}
	seen := make(map[int32]bool)
	var pids []int
	for _, r := range cfg.GetRegions() {
		if !seen[r.GetPid()] {
			seen[r.GetPid()] = true
			pids = append(pids, int(r.GetPid()))
		}
	}
	return pids
}

// extractExplicitPIDs returns the explicitly targeted PIDs from a CUDA
// or DirectMemory BackendConfig, or nil if none were provided.
func extractExplicitPIDs(config *pb.BackendConfig) []int32 {
	if config == nil {
		return nil
	}
	if cuda := config.GetCuda(); cuda != nil {
		if target := cuda.GetExplicitTarget(); target != nil {
			return target.GetPids()
		}
	}
	if dm := config.GetDirectMemory(); dm != nil {
		if target := dm.GetExplicitTarget(); target != nil {
			return target.GetPids()
		}
	}
	return nil
}

// buildRestoreFn returns the background restore function for the given
// deployment mode and backend. In standalone mode the caller-provided
// BackendConfig is passed through as-is. In k8s mode, the CUDA backend
// restores from the PIDs cached at snapshot time (NVML cannot re-discover
// them after checkpoint frees the GPU); application-aware backends resolve
// their targets themselves (HTTP endpoints from the config, or the workload
// channel registered under the job ID), so the config is passed through.
func (s *Server) buildRestoreFn(
	bgCtx context.Context,
	jobID string,
	backendType backends.BackendType,
	backend backends.Backend,
	config *pb.BackendConfig,
) (func() error, error) {
	switch s.deploymentMode {
	case "standalone":
		return func() error {
			slog.InfoContext(bgCtx, "Background: Starting restore", "backend", backendType)
			return backend.Restore(bgCtx, backends.Request{JobID: jobID, Config: config})
		}, nil
	case "k8s":
		switch backendType {
		case backends.BackendCuda, backends.BackendDirectMemory:
			return func() error {
				slog.InfoContext(bgCtx, "Background: Starting restore", "backend", backendType)
				pids, pidErr := s.state.GetJobPIDs(jobID)
				if pidErr != nil {
					return fmt.Errorf("failed to get PIDs for job %s: %w", jobID, pidErr)
				}
				var pidStrings []string
				for _, pid := range pids {
					pidStrings = append(pidStrings, strconv.Itoa(pid))
				}
				slog.InfoContext(bgCtx, "Restoring PIDs", "pids", pidStrings, "backend", backendType)
				var reqConfig *pb.BackendConfig
				var cfgErr error
				if backendType == backends.BackendDirectMemory {
					reqConfig, cfgErr = backends.BuildDirectMemoryConfig(pidStrings)
				} else {
					reqConfig = backends.BuildCudaConfig(pidStrings)
				}
				if cfgErr != nil {
					return fmt.Errorf("failed to build backend config for job %s: %w", jobID, cfgErr)
				}
				return backend.Restore(bgCtx, backends.Request{JobID: jobID, Config: reqConfig})
			}, nil
		case backends.BackendAppEndpoint, backends.BackendAppChannel, backends.BackendMemoryRegions:
			// App backends resolve their own targets; the memory-regions
			// backend restores the explicit regions in the config. Both are
			// pass-through.
			return func() error {
				slog.InfoContext(bgCtx, "Background: Starting restore", "backend", backendType)
				return backend.Restore(bgCtx, backends.Request{JobID: jobID, Config: config})
			}, nil
		default:
			return nil, status.Errorf(codes.InvalidArgument, "backend %q is not supported in k8s mode", backendType)
		}
	default:
		return nil, status.Errorf(codes.InvalidArgument, "unknown deployment mode %q", s.deploymentMode)
	}
}

// Restore triggers an asynchronous restoration of the accelerator context for a job.
func (s *Server) Restore(ctx context.Context, req *pb.RestoreRequest) (*pb.RestoreResponse, error) {
	ctx = logging.WithServerMethod(ctx, "Restore")
	ctx = logging.WithJobID(ctx, req.GetJobId())
	ctx = logging.WithGroupID(ctx, req.GetGroup())

	if err := s.checkFeatureGates(req.GetBackendConfig()); err != nil {
		return nil, err
	}

	backendType := s.getSnapshotBackendType(req.GetBackendConfig())
	slog.InfoContext(ctx, "Restore called", "backend", backendType)

	backend, ok := s.backendMap[backendType]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "backend %s not found", backendType)
	}

	bgCtx := context.WithoutCancel(ctx)
	restoreConfig := req.GetBackendConfig()

	restoreFn, fnErr := s.buildRestoreFn(bgCtx, req.GetJobId(), backendType, backend, restoreConfig)
	if fnErr != nil {
		return nil, fnErr
	}

	slot := memoryRegionsSlot(restoreConfig, req.GetJobId())
	opID, err := s.state.StartRestoreSlot(req.GetJobId(), req.GetGroup(), slot, restoreFn)
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

// StartServer starts the gRPC server on the specified port. featureGates
// may be nil, which leaves every gate at its default.
func StartServer(
	ctx context.Context,
	port int,
	backendMap map[backends.BackendType]backends.Backend,
	defaultBackend backends.BackendType,
	deploymentMode string,
	channelRegistry *backends.ChannelRegistry,
	featureGates features.Gates,
) error {
	lc := net.ListenConfig{}
	lis, err := lc.Listen(ctx, "tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return fmt.Errorf("failed to listen: %w", err)
	}

	// 1. Initialize K8s Client
	k8sClient, err := podutils.GetK8sClient()
	if err != nil {
		return fmt.Errorf("failed to get K8s client: %w", err)
	}

	// 2. Create Server (which creates StateManager internally)
	srv := NewServer(backendMap, defaultBackend, deploymentMode, channelRegistry, featureGates)

	// 3. Start the Watcher internally
	watcher, err := NewWatcher(k8sClient, srv.state)
	if err != nil {
		return fmt.Errorf("failed to create watcher: %w", err)
	}
	watcher.Start(ctx)

	s := grpc.NewServer()
	pb.RegisterSnapshotAgentServiceServer(s, srv)
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

//nolint:nonamedreturns // Conflict between gocritic's unnamedResult and nonamedreturns
func resolvePIDs(ctx context.Context, jobID string, reqPIDs []int32) (allPIDs []int, allPIDStrings []string, err error) {
	if len(reqPIDs) > 0 {
		allPIDs = make([]int, 0, len(reqPIDs))
		allPIDStrings = make([]string, 0, len(reqPIDs))
		for _, pid := range reqPIDs {
			allPIDs = append(allPIDs, int(pid))
			allPIDStrings = append(allPIDStrings, strconv.Itoa(int(pid)))
		}
		return allPIDs, allPIDStrings, nil
	}

	pods, err := podutils.GetLocalPods(ctx, jobID)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get local pods: %w", err)
	}

	if len(pods) == 0 {
		return nil, nil, fmt.Errorf("no pods found for job %s", jobID)
	}

	allPIDs, allPIDStrings, errPIDs := getPIDsFromPods(ctx, pods)
	if errPIDs != nil {
		return nil, nil, fmt.Errorf("failed to get PIDs from pods: %w", errPIDs)
	}

	if len(allPIDStrings) == 0 {
		return nil, nil, fmt.Errorf("no GPU PIDs found for job %s", jobID)
	}

	return allPIDs, allPIDStrings, nil
}
