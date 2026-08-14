package server

import (
	"context"
	"fmt"
	"log/slog"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
	podutils "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/utils"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

// resolutionSource identifies which link of the resolution chain decided a
// request's backend.
type resolutionSource string

const (
	// sourceRequest is an explicit BackendConfig carried by the request.
	sourceRequest resolutionSource = "request"
	// sourceWorkloadChannel is a live workload channel registered for the job.
	sourceWorkloadChannel resolutionSource = "workload-channel"
	// sourcePodAnnotation is a backend declared on the workload's pods via
	// the timeslice.io/backend annotation.
	sourcePodAnnotation resolutionSource = "pod-annotation"
	// sourceDefault is the agent's configured default backend.
	sourceDefault resolutionSource = "default"
)

// backendResolver picks the backend and effective config for every Snapshot
// and Restore request. The sources are consulted in fixed precedence order —
// the first one that applies wins:
//
//  1. an explicit BackendConfig on the request, honored as-is;
//  2. a live workload channel registered for the job (a workload that
//     registered a channel expects suspend/resume pushes, and checkpointing
//     its PIDs behind its back would bypass them);
//  3. a backend declared on the workload's pods through the
//     timeslice.io/backend annotation (with optional config in
//     timeslice.io/backend-config) — malformed or unknown declarations fail
//     the operation rather than falling through silently;
//  4. the agent's configured default backend.
type backendResolver struct {
	defaultBackend backends.BackendType
	registry       *backends.ChannelRegistry
	// lookupPodBackend reads the backend declaration from the job's local
	// pods. Nil disables the pod-annotation source (standalone mode has no
	// pods to consult).
	lookupPodBackend func(ctx context.Context, jobID string) (*podutils.JobBackendAnnotation, error)
}

// newBackendResolver wires the resolution chain for the deployment mode: in
// k8s mode pod annotations are read through the same Kubernetes client that
// job discovery uses; standalone mode has no pods, so that source is
// disabled.
func newBackendResolver(
	defaultBackend backends.BackendType,
	registry *backends.ChannelRegistry,
	deploymentMode string,
) *backendResolver {
	resolver := &backendResolver{
		defaultBackend: defaultBackend,
		registry:       registry,
	}
	if deploymentMode == "k8s" {
		resolver.lookupPodBackend = podutils.GetJobBackendAnnotation
	}
	return resolver
}

// Resolve runs the resolution chain for one request and logs a single line
// stating the deciding source and the chosen backend.
func (r *backendResolver) Resolve(
	ctx context.Context, jobID string, explicit *pb.BackendConfig,
) (backends.BackendType, *pb.BackendConfig, error) {
	backendType, config, source, err := r.resolve(ctx, jobID, explicit)
	if err != nil {
		return "", nil, err
	}
	slog.InfoContext(ctx, "Resolved backend", "jobID", jobID, "source", source, "backend", backendType)
	return backendType, config, nil
}

// resolve is the chain itself; Resolve adds the logging.
//
//nolint:nonamedreturns // Conflict between gocritic's unnamedResult and nonamedreturns
func (r *backendResolver) resolve(
	ctx context.Context, jobID string, explicit *pb.BackendConfig,
) (backendType backends.BackendType, config *pb.BackendConfig, source resolutionSource, err error) {
	if explicit != nil {
		return r.typeForConfig(explicit), explicit, sourceRequest, nil
	}
	if r.hasLiveChannel(ctx, jobID) {
		// The mode is left unspecified so the app-channel backend resolves it
		// from the workload's registered capabilities (declared default, then
		// OFFLOAD).
		return backends.BackendAppChannel, &pb.BackendConfig{
			Backend: &pb.BackendConfig_AppChannel{AppChannel: &pb.AppChannelConfig{}},
		}, sourceWorkloadChannel, nil
	}
	backendType, config, found, err := r.fromPodAnnotation(ctx, jobID)
	if err != nil {
		return "", nil, "", err
	}
	if found {
		return backendType, config, sourcePodAnnotation, nil
	}
	return r.defaultBackend, nil, sourceDefault, nil
}

// typeForConfig maps an explicit BackendConfig to the backend that serves
// it. Configs that select no known backend fall through to the default.
//
// NOTE: direct_memory is not routed yet and falls through to the default
// backend. It is unreachable unless the DirectMemoryBackend feature gate is
// enabled (checkFeatureGates rejects it on the resolved config).
func (r *backendResolver) typeForConfig(config *pb.BackendConfig) backends.BackendType {
	switch {
	case config.GetCuda() != nil:
		return backends.BackendCuda
	case config.GetAppEndpoint() != nil:
		return backends.BackendAppEndpoint
	case config.GetAppChannel() != nil:
		return backends.BackendAppChannel
	default:
		return r.defaultBackend
	}
}

// hasLiveChannel reports whether the job has a live registered workload
// channel. A registered-but-disconnected channel does not count: resolution
// continues down the chain with a warning.
func (r *backendResolver) hasLiveChannel(ctx context.Context, jobID string) bool {
	if r.registry == nil {
		return false
	}
	session, err := r.registry.Session(jobID)
	if err != nil {
		return false
	}
	if session.Closed() {
		slog.WarnContext(ctx, "Workload channel is registered but disconnected, continuing backend resolution without it",
			"jobID", jobID)
		return false
	}
	return true
}

// fromPodAnnotation resolves the backend declared on the workload's local
// pods via the timeslice.io/backend annotation. A declaration that cannot be
// resolved — unknown backend name, unparsable config, conflicting pods, or a
// failed pod lookup — fails the operation loudly; it never silently falls
// through to the default.
//
//nolint:nonamedreturns // Conflict between gocritic's unnamedResult and nonamedreturns
func (r *backendResolver) fromPodAnnotation(
	ctx context.Context, jobID string,
) (backendType backends.BackendType, config *pb.BackendConfig, found bool, err error) {
	if r.lookupPodBackend == nil {
		return "", nil, false, nil
	}
	annotation, err := r.lookupPodBackend(ctx, jobID)
	if err != nil {
		return "", nil, false, status.Errorf(codes.Internal,
			"failed to read the %s annotation for job %s: %v", podutils.BackendAnnotation, jobID, err)
	}
	if annotation == nil {
		return "", nil, false, nil
	}
	backendType, config, err = r.parseAnnotation(annotation)
	if err != nil {
		return "", nil, false, status.Errorf(codes.FailedPrecondition,
			"invalid backend declaration on the pods of job %s: %v", jobID, err)
	}
	return backendType, config, true, nil
}

// parseAnnotation maps a pod backend declaration to a backend and config.
// The annotation value is a BackendConfig field name (cuda, app_endpoint,
// app_channel, direct_memory) or noop; the optional companion annotation
// carries the protojson encoding of that backend's config message, e.g.
// {"endpoints": ["http://localhost:8000"]} for app_endpoint.
func (r *backendResolver) parseAnnotation(
	annotation *podutils.JobBackendAnnotation,
) (backends.BackendType, *pb.BackendConfig, error) {
	var message proto.Message
	var config *pb.BackendConfig
	switch annotation.Backend {
	case "cuda":
		cuda := &pb.CudaBackendConfig{}
		message = cuda
		config = &pb.BackendConfig{Backend: &pb.BackendConfig_Cuda{Cuda: cuda}}
	case "app_endpoint":
		endpoint := &pb.AppEndpointConfig{}
		message = endpoint
		config = &pb.BackendConfig{Backend: &pb.BackendConfig_AppEndpoint{AppEndpoint: endpoint}}
	case "app_channel":
		channel := &pb.AppChannelConfig{}
		message = channel
		config = &pb.BackendConfig{Backend: &pb.BackendConfig_AppChannel{AppChannel: channel}}
	case "direct_memory":
		direct := &pb.DirectMemoryBackendConfig{}
		message = direct
		config = &pb.BackendConfig{Backend: &pb.BackendConfig_DirectMemory{DirectMemory: direct}}
	case string(backends.BackendNoop):
		if annotation.Config != "" {
			return "", nil, fmt.Errorf("backend %q takes no %s annotation",
				annotation.Backend, podutils.BackendConfigAnnotation)
		}
		return backends.BackendNoop, nil, nil
	default:
		return "", nil, fmt.Errorf(
			"unknown backend %q in the %s annotation (known: cuda, app_endpoint, app_channel, direct_memory, noop)",
			annotation.Backend, podutils.BackendAnnotation)
	}
	if annotation.Config != "" {
		if err := protojson.Unmarshal([]byte(annotation.Config), message); err != nil {
			return "", nil, fmt.Errorf("failed to parse the %s annotation as a %s config: %w",
				podutils.BackendConfigAnnotation, annotation.Backend, err)
		}
	}
	return r.typeForConfig(config), config, nil
}
