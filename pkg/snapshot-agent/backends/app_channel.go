package backends

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/google/uuid"
	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
)

// defaultCommandTimeout bounds how long a dispatched command waits for the
// workload's CommandResult. It covers the workload executing the suspend or
// resume (e.g. offloading weights to host memory), not just the round trip.
const defaultCommandTimeout = 2 * time.Minute

// WorkloadSession is one registered workload stream. It routes commands to
// the stream's send function and matches CommandResults to in-flight
// commands by command_id.
type WorkloadSession struct {
	jobID string
	caps  *pb.WorkloadCapabilities

	// sendMu serializes writes to the underlying gRPC stream, which does not
	// allow concurrent Sends.
	sendMu sync.Mutex
	send   func(*pb.AgentCommand) error

	mu      sync.Mutex
	pending map[string]chan *pb.CommandResult

	closeOnce sync.Once
	closed    chan struct{}
}

// Capabilities returns the capabilities the workload declared at registration.
func (s *WorkloadSession) Capabilities() *pb.WorkloadCapabilities {
	return s.caps
}

// HandleResult routes a CommandResult from the stream to the in-flight
// command waiting on it. Results for unknown command IDs are dropped (the
// command may have timed out).
func (s *WorkloadSession) HandleResult(res *pb.CommandResult) {
	s.mu.Lock()
	ch, ok := s.pending[res.GetCommandId()]
	if ok {
		delete(s.pending, res.GetCommandId())
	}
	s.mu.Unlock()
	if ok {
		ch <- res
	}
}

// close marks the session as gone, failing any in-flight dispatches.
func (s *WorkloadSession) close() {
	s.closeOnce.Do(func() { close(s.closed) })
}

// Dispatch assigns the command a unique ID, sends it once, and waits for the
// matching CommandResult, the session to drop, or ctx to expire.
func (s *WorkloadSession) Dispatch(ctx context.Context, cmd *pb.AgentCommand) error {
	cmd.CommandId = uuid.New().String()

	ch := make(chan *pb.CommandResult, 1)
	s.mu.Lock()
	s.pending[cmd.CommandId] = ch
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		delete(s.pending, cmd.CommandId)
		s.mu.Unlock()
	}()

	s.sendMu.Lock()
	err := s.send(cmd)
	s.sendMu.Unlock()
	if err != nil {
		return fmt.Errorf("failed to send command to workload %s: %w", s.jobID, err)
	}

	select {
	case res := <-ch:
		if !res.GetOk() {
			return fmt.Errorf("workload %s failed command: %s", s.jobID, res.GetError())
		}
		return nil
	case <-s.closed:
		return fmt.Errorf("workload %s disconnected before completing command", s.jobID)
	case <-ctx.Done():
		return fmt.Errorf("timed out waiting for workload %s to complete command: %w", s.jobID, ctx.Err())
	}
}

// ChannelRegistry tracks the workload session registered for each job. It is
// shared between the WorkloadChannel RPC handler (which registers streams)
// and the AppChannelBackend (which dispatches commands to them).
type ChannelRegistry struct {
	mu       sync.Mutex
	sessions map[string]*WorkloadSession
}

// NewChannelRegistry creates an empty registry.
func NewChannelRegistry() *ChannelRegistry {
	return &ChannelRegistry{sessions: make(map[string]*WorkloadSession)}
}

// Register creates a session for jobID backed by the given send function.
// A previously registered session for the same job is replaced and closed.
func (r *ChannelRegistry) Register(
	jobID string,
	caps *pb.WorkloadCapabilities,
	send func(*pb.AgentCommand) error,
) *WorkloadSession {
	session := &WorkloadSession{
		jobID:   jobID,
		caps:    caps,
		send:    send,
		pending: make(map[string]chan *pb.CommandResult),
		closed:  make(chan struct{}),
	}
	r.mu.Lock()
	old := r.sessions[jobID]
	r.sessions[jobID] = session
	r.mu.Unlock()
	if old != nil {
		slog.Info("Replacing existing workload channel registration", "jobID", jobID)
		old.close()
	}
	return session
}

// Unregister removes the session if it is still the current one for its job
// (a replaced session must not evict its replacement) and closes it.
func (r *ChannelRegistry) Unregister(session *WorkloadSession) {
	r.mu.Lock()
	if r.sessions[session.jobID] == session {
		delete(r.sessions, session.jobID)
	}
	r.mu.Unlock()
	session.close()
}

// Session returns the current session for jobID, or an error if no workload
// is registered.
func (r *ChannelRegistry) Session(jobID string) (*WorkloadSession, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	session, ok := r.sessions[jobID]
	if !ok {
		return nil, fmt.Errorf("no workload channel registered for job %q", jobID)
	}
	return session, nil
}

// resolveSuspendMode resolves the effective suspend mode from the request and
// the workload's registered capabilities: explicit request mode, then the
// workload's default, then OFFLOAD. The result is validated against the
// workload's supported modes when it declared any.
func resolveSuspendMode(requested pb.SuspendMode, caps *pb.WorkloadCapabilities) (pb.SuspendMode, error) {
	mode := requested
	if mode == pb.SuspendMode_SUSPEND_MODE_UNSPECIFIED {
		mode = caps.GetDefaultMode()
	}
	if mode == pb.SuspendMode_SUSPEND_MODE_UNSPECIFIED {
		mode = pb.SuspendMode_SUSPEND_MODE_OFFLOAD
	}
	supported := caps.GetSupportedModes()
	if len(supported) == 0 {
		return mode, nil
	}
	for _, m := range supported {
		if m == mode {
			return mode, nil
		}
	}
	return pb.SuspendMode_SUSPEND_MODE_UNSPECIFIED,
		fmt.Errorf("workload does not support suspend mode %s (supports %v)", mode, supported)
}

// AppChannelBackend implements Backend for application-aware workloads
// reached through their registered workload channels (the app_channel
// transport). The target is resolved from the job ID via the registry.
type AppChannelBackend struct {
	registry       *ChannelRegistry
	commandTimeout time.Duration
}

// NewAppChannelBackend creates the app-channel backend around a registry.
func NewAppChannelBackend(registry *ChannelRegistry) *AppChannelBackend {
	return &AppChannelBackend{
		registry:       registry,
		commandTimeout: defaultCommandTimeout,
	}
}

// Snapshot pushes a SNAPSHOT command to the workload registered for jobID
// and waits for it to complete.
func (b *AppChannelBackend) Snapshot(ctx context.Context, req Request) error {
	channelCfg := req.Config.GetAppChannel()
	if channelCfg == nil {
		return fmt.Errorf("app_channel config is required")
	}
	session, err := b.registry.Session(req.JobID)
	if err != nil {
		return err
	}
	mode, err := resolveSuspendMode(channelCfg.GetMode(), session.Capabilities())
	if err != nil {
		return fmt.Errorf("cannot snapshot job %s: %w", req.JobID, err)
	}

	ctx, cancel := context.WithTimeout(ctx, b.commandTimeout)
	defer cancel()

	slog.InfoContext(ctx, "Dispatching snapshot command to workload", "jobID", req.JobID, "mode", mode)
	start := time.Now()
	if err := session.Dispatch(ctx, &pb.AgentCommand{
		Command: &pb.AgentCommand_Snapshot{
			Snapshot: &pb.SnapshotCommand{Mode: mode, Tags: channelCfg.GetTags()},
		},
	}); err != nil {
		return err
	}
	slog.InfoContext(ctx, "Workload completed snapshot command", "jobID", req.JobID, "duration", time.Since(start))
	return nil
}

// Restore pushes a RESTORE command to the workload registered for jobID and
// waits for it to complete.
func (b *AppChannelBackend) Restore(ctx context.Context, req Request) error {
	channelCfg := req.Config.GetAppChannel()
	if channelCfg == nil {
		return fmt.Errorf("app_channel config is required")
	}
	session, err := b.registry.Session(req.JobID)
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(ctx, b.commandTimeout)
	defer cancel()

	slog.InfoContext(ctx, "Dispatching restore command to workload", "jobID", req.JobID)
	start := time.Now()
	if err := session.Dispatch(ctx, &pb.AgentCommand{
		Command: &pb.AgentCommand_Restore{
			Restore: &pb.RestoreCommand{Tags: channelCfg.GetTags()},
		},
	}); err != nil {
		return err
	}
	slog.InfoContext(ctx, "Workload completed restore command", "jobID", req.JobID, "duration", time.Since(start))
	return nil
}

// HealthCheck returns nil — sessions come and go with workload lifecycles,
// so there is nothing to probe at the backend level.
func (b *AppChannelBackend) HealthCheck(_ context.Context) error {
	return nil
}
