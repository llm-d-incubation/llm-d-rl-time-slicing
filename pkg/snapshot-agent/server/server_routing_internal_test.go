package server

import (
	"context"
	"errors"
	"net"
	"sync/atomic"
	"testing"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
	podutils "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/utils"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/proto"
)

// TestBackendResolver_Resolve covers the resolution chain source by source:
// explicit configs always win and pass through untouched; config-less
// requests prefer the job's live registered workload channel, then the
// backend declared on the job's pods, then the configured default. Malformed
// or unknown pod declarations fail the resolution instead of falling
// through.
func TestBackendResolver_Resolve(t *testing.T) {
	cudaConfig := &pb.BackendConfig{
		Backend: &pb.BackendConfig_Cuda{Cuda: &pb.CudaBackendConfig{}},
	}
	endpointConfig := &pb.BackendConfig{
		Backend: &pb.BackendConfig_AppEndpoint{AppEndpoint: &pb.AppEndpointConfig{}},
	}
	channelConfig := &pb.BackendConfig{
		Backend: &pb.BackendConfig_AppChannel{AppChannel: &pb.AppChannelConfig{Mode: pb.SuspendMode_SUSPEND_MODE_DISCARD}},
	}
	emptyConfig := &pb.BackendConfig{}
	cudaAnnotation := &podutils.JobBackendAnnotation{Backend: "cuda"}

	tests := []struct {
		name           string
		config         *pb.BackendConfig
		registeredJobs []string
		annotation     *podutils.JobBackendAnnotation
		annotationErr  error
		jobID          string
		wantType       backends.BackendType
		wantSource     resolutionSource
		wantConfig     *pb.BackendConfig // expected pass-through config (pointer identity)
		wantProto      *pb.BackendConfig // expected built config (proto equality)
		wantNilConfig  bool              // expect no config at all
		wantCode       codes.Code        // expected resolution error; OK means success
	}{
		{
			name:           "explicit config wins over a registered channel and an annotation",
			config:         cudaConfig,
			registeredJobs: []string{"job-1"},
			annotation:     &podutils.JobBackendAnnotation{Backend: "app_endpoint"},
			jobID:          "job-1",
			wantType:       backends.BackendCuda,
			wantSource:     sourceRequest,
			wantConfig:     cudaConfig,
		},
		{
			name:       "explicit app-endpoint config passes through",
			config:     endpointConfig,
			jobID:      "job-1",
			wantType:   backends.BackendAppEndpoint,
			wantSource: sourceRequest,
			wantConfig: endpointConfig,
		},
		{
			name:       "explicit app-channel config passes through",
			config:     channelConfig,
			jobID:      "job-1",
			wantType:   backends.BackendAppChannel,
			wantSource: sourceRequest,
			wantConfig: channelConfig,
		},
		{
			name:           "empty non-nil config selects the default backend",
			config:         emptyConfig,
			registeredJobs: []string{"job-1"},
			jobID:          "job-1",
			wantType:       backends.BackendCuda,
			wantSource:     sourceRequest,
			wantConfig:     emptyConfig,
		},
		{
			name:          "no config, channel, or annotation selects the default backend",
			jobID:         "job-1",
			wantType:      backends.BackendCuda,
			wantSource:    sourceDefault,
			wantNilConfig: true,
		},
		{
			name:           "live channel wins over an annotation",
			registeredJobs: []string{"job-1"},
			annotation:     cudaAnnotation,
			jobID:          "job-1",
			wantType:       backends.BackendAppChannel,
			wantSource:     sourceWorkloadChannel,
			wantProto: &pb.BackendConfig{
				Backend: &pb.BackendConfig_AppChannel{AppChannel: &pb.AppChannelConfig{}},
			},
		},
		{
			name:           "channel for another job falls through to the annotation",
			registeredJobs: []string{"other-job"},
			annotation:     cudaAnnotation,
			jobID:          "job-1",
			wantType:       backends.BackendCuda,
			wantSource:     sourcePodAnnotation,
			wantProto:      cudaConfig,
		},
		{
			name:       "cuda annotation selects the cuda backend",
			annotation: cudaAnnotation,
			jobID:      "job-1",
			wantType:   backends.BackendCuda,
			wantSource: sourcePodAnnotation,
			wantProto:  cudaConfig,
		},
		{
			name: "app_endpoint annotation with config JSON builds the app-endpoint config",
			annotation: &podutils.JobBackendAnnotation{
				Backend: "app_endpoint",
				Config:  `{"app": "APP_VLLM", "endpoints": ["http://localhost:8000"], "mode": "SUSPEND_MODE_DISCARD"}`,
			},
			jobID:      "job-1",
			wantType:   backends.BackendAppEndpoint,
			wantSource: sourcePodAnnotation,
			wantProto: &pb.BackendConfig{
				Backend: &pb.BackendConfig_AppEndpoint{AppEndpoint: &pb.AppEndpointConfig{
					App:       pb.App_APP_VLLM,
					Endpoints: []string{"http://localhost:8000"},
					Mode:      pb.SuspendMode_SUSPEND_MODE_DISCARD,
				}},
			},
		},
		{
			name:          "noop annotation selects the noop backend without a config",
			annotation:    &podutils.JobBackendAnnotation{Backend: "noop"},
			jobID:         "job-1",
			wantType:      backends.BackendNoop,
			wantSource:    sourcePodAnnotation,
			wantNilConfig: true,
		},
		{
			// direct_memory is not routed yet: like an explicit direct_memory
			// config, it resolves to the default backend, and the feature
			// gate check rejects the resolved config at the RPC layer unless
			// the gate is enabled.
			name:       "direct_memory annotation builds the config and falls to the default backend",
			annotation: &podutils.JobBackendAnnotation{Backend: "direct_memory"},
			jobID:      "job-1",
			wantType:   backends.BackendCuda,
			wantSource: sourcePodAnnotation,
			wantProto: &pb.BackendConfig{
				Backend: &pb.BackendConfig_DirectMemory{DirectMemory: &pb.DirectMemoryBackendConfig{}},
			},
		},
		{
			name:       "unknown annotation backend fails the resolution",
			annotation: &podutils.JobBackendAnnotation{Backend: "warp-drive"},
			jobID:      "job-1",
			wantCode:   codes.FailedPrecondition,
		},
		{
			name: "malformed annotation config JSON fails the resolution",
			annotation: &podutils.JobBackendAnnotation{
				Backend: "app_endpoint",
				Config:  `{"endpoints": not-json`,
			},
			jobID:    "job-1",
			wantCode: codes.FailedPrecondition,
		},
		{
			name: "config JSON on the config-less noop backend fails the resolution",
			annotation: &podutils.JobBackendAnnotation{
				Backend: "noop",
				Config:  `{}`,
			},
			jobID:    "job-1",
			wantCode: codes.FailedPrecondition,
		},
		{
			name:          "failed annotation lookup fails the resolution",
			annotationErr: errors.New("pod list failed"),
			jobID:         "job-1",
			wantCode:      codes.Internal,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			registry := backends.NewChannelRegistry()
			for _, jobID := range tc.registeredJobs {
				registry.Register(jobID, nil, func(*pb.AgentCommand) error { return nil })
			}
			resolver := newBackendResolver(backends.BackendCuda, registry, "k8s")
			resolver.lookupPodBackend = func(context.Context, string) (*podutils.JobBackendAnnotation, error) {
				return tc.annotation, tc.annotationErr
			}

			gotType, gotConfig, gotSource, err := resolver.resolve(context.Background(), tc.jobID, tc.config)
			if tc.wantCode != codes.OK {
				if status.Code(err) != tc.wantCode {
					t.Fatalf("Expected resolution to fail with %v, got type=%s err=%v", tc.wantCode, gotType, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("Expected resolution to succeed, got: %v", err)
			}
			if gotType != tc.wantType {
				t.Errorf("Expected backend %s, got %s", tc.wantType, gotType)
			}
			if gotSource != tc.wantSource {
				t.Errorf("Expected source %s, got %s", tc.wantSource, gotSource)
			}
			switch {
			case tc.wantNilConfig:
				if gotConfig != nil {
					t.Errorf("Expected no config, got %v", gotConfig)
				}
			case tc.wantConfig != nil:
				if gotConfig != tc.wantConfig {
					t.Errorf("Expected the request config to pass through unchanged, got %v", gotConfig)
				}
			default:
				if !proto.Equal(gotConfig, tc.wantProto) {
					t.Errorf("Expected config %v, got %v", tc.wantProto, gotConfig)
				}
			}
		})
	}
}

// TestBackendResolver_StandaloneSkipsPodAnnotations verifies that standalone
// mode, which has no pods to consult, resolves config-less requests straight
// to the default backend.
func TestBackendResolver_StandaloneSkipsPodAnnotations(t *testing.T) {
	resolver := newBackendResolver(backends.BackendCuda, backends.NewChannelRegistry(), "standalone")
	if resolver.lookupPodBackend != nil {
		t.Fatal("Expected the pod-annotation source to be disabled in standalone mode")
	}

	gotType, gotConfig, gotSource, err := resolver.resolve(context.Background(), "job-1", nil)
	if err != nil {
		t.Fatalf("Expected resolution to succeed, got: %v", err)
	}
	if gotType != backends.BackendCuda || gotSource != sourceDefault || gotConfig != nil {
		t.Errorf("Expected the default backend with no config, got %s from %s with %v", gotType, gotSource, gotConfig)
	}
}

// recordingBackend counts invocations; the routing tests use it to prove a
// backend was bypassed (or hit).
type recordingBackend struct {
	backends.NoopBackend
	snapshots atomic.Int32
	restores  atomic.Int32
}

func (b *recordingBackend) Snapshot(ctx context.Context, req backends.Request) error {
	b.snapshots.Add(1)
	return b.NoopBackend.Snapshot(ctx, req)
}

func (b *recordingBackend) Restore(ctx context.Context, req backends.Request) error {
	b.restores.Add(1)
	return b.NoopBackend.Restore(ctx, req)
}

// newRoutingTestServer starts a server whose backend map holds the given
// backends (the default type must be among them) plus a real app-channel
// backend sharing the registry with the WorkloadChannel handler, and returns
// a connected client. The server's pod-annotation lookup is stubbed to find
// nothing; tests override srv.resolver.lookupPodBackend as needed.
func newRoutingTestServer(
	t *testing.T,
	defaultType backends.BackendType,
	backendsByType map[backends.BackendType]backends.Backend,
	mode string,
) (*Server, *backends.ChannelRegistry, pb.SnapshotAgentServiceClient) {
	t.Helper()
	lisRoute := bufconn.Listen(bufSize)
	s := grpc.NewServer()
	registry := backends.NewChannelRegistry()
	backendsMap := map[backends.BackendType]backends.Backend{
		backends.BackendAppChannel: backends.NewAppChannelBackend(registry),
	}
	for backendType, backend := range backendsByType {
		backendsMap[backendType] = backend
	}
	srv := NewServer(backendsMap, defaultType, mode, registry, nil)
	srv.resolver.lookupPodBackend = func(context.Context, string) (*podutils.JobBackendAnnotation, error) {
		return nil, nil
	}
	pb.RegisterSnapshotAgentServiceServer(s, srv)
	go func() {
		if err := s.Serve(lisRoute); err != nil {
			return
		}
	}()
	t.Cleanup(s.GracefulStop)

	conn, err := grpc.NewClient("passthrough://bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return lisRoute.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	t.Cleanup(func() { conn.Close() })
	return srv, registry, pb.NewSnapshotAgentServiceClient(conn)
}

// TestServer_NoConfig_RoutesToWorkloadChannel verifies the orchestrator
// scenario end to end: Snapshot/Restore requests without a BackendConfig
// reach a registered workload through its channel instead of the configured
// default (CUDA) backend.
func TestServer_NoConfig_RoutesToWorkloadChannel(t *testing.T) {
	defaultBackend := &recordingBackend{}
	srv, registry, client := newRoutingTestServer(t, backends.BackendCuda,
		map[backends.BackendType]backends.Backend{backends.BackendCuda: defaultBackend}, "k8s")
	ctx := context.Background()

	registerWorkload(ctx, t, client, registry, "routed-job", &pb.WorkloadCapabilities{
		SupportedModes: []pb.SuspendMode{pb.SuspendMode_SUSPEND_MODE_OFFLOAD},
		DefaultMode:    pb.SuspendMode_SUSPEND_MODE_OFFLOAD,
	})
	if err := srv.state.TransitionToRunning("routed-job", nil); err != nil {
		t.Fatalf("Failed to transition job to RUNNING: %v", err)
	}

	snapResp, err := client.Snapshot(ctx, &pb.SnapshotRequest{JobId: "routed-job"})
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	opResp := waitForOperation(ctx, t, client, snapResp.GetOperationId())
	if opResp.GetStatus() != pb.OperationStatus_OPERATION_STATUS_COMPLETE {
		t.Fatalf("Expected snapshot COMPLETE, got %v (error: %q)", opResp.GetStatus(), opResp.GetError())
	}

	restoreResp, err := client.Restore(ctx, &pb.RestoreRequest{JobId: "routed-job"})
	if err != nil {
		t.Fatalf("Restore failed: %v", err)
	}
	opResp = waitForOperation(ctx, t, client, restoreResp.GetOperationId())
	if opResp.GetStatus() != pb.OperationStatus_OPERATION_STATUS_COMPLETE {
		t.Fatalf("Expected restore COMPLETE, got %v (error: %q)", opResp.GetStatus(), opResp.GetError())
	}

	if got := defaultBackend.snapshots.Load(); got != 0 {
		t.Errorf("Expected the default backend to be bypassed for snapshot, got %d calls", got)
	}
	if got := defaultBackend.restores.Load(); got != 0 {
		t.Errorf("Expected the default backend to be bypassed for restore, got %d calls", got)
	}
}

// TestServer_NoConfig_PodAnnotationRoutesBackend verifies the pod-annotation
// source end to end: a config-less Snapshot for a job whose pods declare
// timeslice.io/backend=app_endpoint reaches the app-endpoint backend instead
// of the configured default (CUDA) backend.
func TestServer_NoConfig_PodAnnotationRoutesBackend(t *testing.T) {
	defaultBackend := &recordingBackend{}
	annotatedBackend := &recordingBackend{}
	srv, _, client := newRoutingTestServer(t, backends.BackendCuda,
		map[backends.BackendType]backends.Backend{
			backends.BackendCuda:        defaultBackend,
			backends.BackendAppEndpoint: annotatedBackend,
		}, "k8s")
	srv.resolver.lookupPodBackend = func(context.Context, string) (*podutils.JobBackendAnnotation, error) {
		return &podutils.JobBackendAnnotation{
			Backend: "app_endpoint",
			Config:  `{"app": "APP_VLLM", "endpoints": ["http://localhost:8000"]}`,
		}, nil
	}
	ctx := context.Background()

	srv.state.RegisterJob("annotated-job", "")
	if err := srv.state.TransitionToRunning("annotated-job", nil); err != nil {
		t.Fatalf("Failed to transition job to RUNNING: %v", err)
	}

	snapResp, err := client.Snapshot(ctx, &pb.SnapshotRequest{JobId: "annotated-job"})
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	opResp := waitForOperation(ctx, t, client, snapResp.GetOperationId())
	if opResp.GetStatus() != pb.OperationStatus_OPERATION_STATUS_COMPLETE {
		t.Fatalf("Expected snapshot COMPLETE, got %v (error: %q)", opResp.GetStatus(), opResp.GetError())
	}

	if got := annotatedBackend.snapshots.Load(); got != 1 {
		t.Errorf("Expected the annotated backend to handle the snapshot, got %d calls", got)
	}
	if got := defaultBackend.snapshots.Load(); got != 0 {
		t.Errorf("Expected the default backend to be bypassed, got %d calls", got)
	}
}

// TestServer_MalformedPodAnnotationFailsLoudly verifies that an unknown pod
// backend declaration fails both RPCs with FAILED_PRECONDITION instead of
// silently falling through to the default backend.
func TestServer_MalformedPodAnnotationFailsLoudly(t *testing.T) {
	defaultBackend := &recordingBackend{}
	srv, _, client := newRoutingTestServer(t, backends.BackendCuda,
		map[backends.BackendType]backends.Backend{backends.BackendCuda: defaultBackend}, "k8s")
	srv.resolver.lookupPodBackend = func(context.Context, string) (*podutils.JobBackendAnnotation, error) {
		return &podutils.JobBackendAnnotation{Backend: "warp-drive"}, nil
	}
	ctx := context.Background()

	srv.state.RegisterJob("misdeclared-job", "")
	if err := srv.state.TransitionToRunning("misdeclared-job", nil); err != nil {
		t.Fatalf("Failed to transition job to RUNNING: %v", err)
	}

	_, err := client.Snapshot(ctx, &pb.SnapshotRequest{JobId: "misdeclared-job"})
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("Expected Snapshot to fail with FAILED_PRECONDITION, got: %v", err)
	}
	_, err = client.Restore(ctx, &pb.RestoreRequest{JobId: "misdeclared-job"})
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("Expected Restore to fail with FAILED_PRECONDITION, got: %v", err)
	}
	if got := defaultBackend.snapshots.Load() + defaultBackend.restores.Load(); got != 0 {
		t.Errorf("Expected no backend calls for a misdeclared job, got %d", got)
	}
}

// TestServer_NoConfig_NoChannelUsesDefault verifies that config-less requests
// for jobs without a registered workload channel still hit the configured
// default backend (standalone mode passes the config through untouched).
func TestServer_NoConfig_NoChannelUsesDefault(t *testing.T) {
	defaultBackend := &recordingBackend{}
	srv, _, client := newRoutingTestServer(t, backends.BackendNoop,
		map[backends.BackendType]backends.Backend{backends.BackendNoop: defaultBackend}, "standalone")
	ctx := context.Background()

	srv.state.RegisterJob("plain-job", "")
	if err := srv.state.TransitionToRunning("plain-job", nil); err != nil {
		t.Fatalf("Failed to transition job to RUNNING: %v", err)
	}

	snapResp, err := client.Snapshot(ctx, &pb.SnapshotRequest{JobId: "plain-job"})
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	opResp := waitForOperation(ctx, t, client, snapResp.GetOperationId())
	if opResp.GetStatus() != pb.OperationStatus_OPERATION_STATUS_COMPLETE {
		t.Fatalf("Expected snapshot COMPLETE, got %v (error: %q)", opResp.GetStatus(), opResp.GetError())
	}
	if got := defaultBackend.snapshots.Load(); got != 1 {
		t.Errorf("Expected the default backend to handle the snapshot, got %d calls", got)
	}
}
