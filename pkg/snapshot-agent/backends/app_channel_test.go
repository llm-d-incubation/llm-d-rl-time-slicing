package backends_test

import (
	"context"
	"strings"
	"testing"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
)

// echoWorkload registers a session that immediately acknowledges every
// command, recording it. Returns the channel of observed commands.
func echoWorkload(registry *backends.ChannelRegistry, jobID string, caps *pb.WorkloadCapabilities) chan *pb.AgentCommand {
	commands := make(chan *pb.AgentCommand, 16)
	var session *backends.WorkloadSession
	session = registry.Register(jobID, caps, func(cmd *pb.AgentCommand) error {
		commands <- cmd
		go session.HandleResult(&pb.CommandResult{CommandId: cmd.GetCommandId(), Ok: true})
		return nil
	})
	return commands
}

func channelReq(jobID string, mode pb.SuspendMode, tags ...string) backends.Request {
	return backends.Request{
		JobID: jobID,
		Config: &pb.BackendConfig{
			Backend: &pb.BackendConfig_AppChannel{
				AppChannel: &pb.AppChannelConfig{Mode: mode, Tags: tags},
			},
		},
	}
}

func TestAppChannelSnapshotAndRestore(t *testing.T) {
	registry := backends.NewChannelRegistry()
	backend := backends.NewAppChannelBackend(registry)
	commands := echoWorkload(registry, "job-1", nil)

	err := backend.Snapshot(context.Background(),
		channelReq("job-1", pb.SuspendMode_SUSPEND_MODE_DISCARD, "weights"))
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	cmd := <-commands
	if cmd.GetCommandId() == "" {
		t.Error("Expected a non-empty command_id")
	}
	if cmd.GetSnapshot() == nil {
		t.Fatalf("Expected a snapshot command, got %v", cmd)
	}
	if cmd.GetSnapshot().GetMode() != pb.SuspendMode_SUSPEND_MODE_DISCARD {
		t.Errorf("Expected DISCARD mode, got %v", cmd.GetSnapshot().GetMode())
	}
	if len(cmd.GetSnapshot().GetTags()) != 1 || cmd.GetSnapshot().GetTags()[0] != "weights" {
		t.Errorf("Expected tags [weights], got %v", cmd.GetSnapshot().GetTags())
	}

	err = backend.Restore(context.Background(),
		channelReq("job-1", pb.SuspendMode_SUSPEND_MODE_UNSPECIFIED, "weights"))
	if err != nil {
		t.Fatalf("Restore failed: %v", err)
	}
	cmd = <-commands
	if cmd.GetRestore() == nil {
		t.Fatalf("Expected a restore command, got %v", cmd)
	}
	if len(cmd.GetRestore().GetTags()) != 1 || cmd.GetRestore().GetTags()[0] != "weights" {
		t.Errorf("Expected tags [weights], got %v", cmd.GetRestore().GetTags())
	}
}

func TestAppChannelSnapshotErrors(t *testing.T) {
	tests := []struct {
		name         string
		setup        func(*backends.ChannelRegistry, *backends.AppChannelBackend) chan *pb.AgentCommand
		req          backends.Request
		wantErr      string
		wantDispatch int // expected commands dispatched; -1 = don't check
	}{
		{
			name:    "unregistered job fails fast",
			setup:   func(*backends.ChannelRegistry, *backends.AppChannelBackend) chan *pb.AgentCommand { return nil },
			req:     channelReq("ghost", pb.SuspendMode_SUSPEND_MODE_UNSPECIFIED),
			wantErr: "no workload channel registered",
		},
		{
			name: "missing app_channel config",
			setup: func(r *backends.ChannelRegistry, _ *backends.AppChannelBackend) chan *pb.AgentCommand {
				echoWorkload(r, "job-1", nil)
				return nil
			},
			req:     backends.Request{JobID: "job-1", Config: &pb.BackendConfig{}},
			wantErr: "app_channel config is required",
		},
		{
			name: "workload failure propagates",
			setup: func(r *backends.ChannelRegistry, _ *backends.AppChannelBackend) chan *pb.AgentCommand {
				var session *backends.WorkloadSession
				session = r.Register("job-1", nil, func(cmd *pb.AgentCommand) error {
					go session.HandleResult(&pb.CommandResult{CommandId: cmd.GetCommandId(), Ok: false, Error: "engine exploded"})
					return nil
				})
				return nil
			},
			req:     channelReq("job-1", pb.SuspendMode_SUSPEND_MODE_OFFLOAD),
			wantErr: "engine exploded",
		},
		{
			name: "command timeout",
			setup: func(r *backends.ChannelRegistry, b *backends.AppChannelBackend) chan *pb.AgentCommand {
				b.SetCommandTimeout(50 * time.Millisecond)
				r.Register("job-1", nil, func(*pb.AgentCommand) error { return nil })
				return nil
			},
			req:     channelReq("job-1", pb.SuspendMode_SUSPEND_MODE_OFFLOAD),
			wantErr: "timed out",
		},
		{
			name: "disconnect fails inflight command",
			setup: func(r *backends.ChannelRegistry, _ *backends.AppChannelBackend) chan *pb.AgentCommand {
				var session *backends.WorkloadSession
				session = r.Register("job-1", nil, func(*pb.AgentCommand) error {
					go r.Unregister(session)
					return nil
				})
				return nil
			},
			req:     channelReq("job-1", pb.SuspendMode_SUSPEND_MODE_OFFLOAD),
			wantErr: "disconnected",
		},
		{
			name: "unsupported mode rejected before dispatch",
			setup: func(r *backends.ChannelRegistry, _ *backends.AppChannelBackend) chan *pb.AgentCommand {
				return echoWorkload(r, "trainer-1", &pb.WorkloadCapabilities{
					SupportedModes: []pb.SuspendMode{pb.SuspendMode_SUSPEND_MODE_OFFLOAD},
				})
			},
			req:          channelReq("trainer-1", pb.SuspendMode_SUSPEND_MODE_DISCARD),
			wantErr:      "does not support suspend mode",
			wantDispatch: 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			registry := backends.NewChannelRegistry()
			backend := backends.NewAppChannelBackend(registry)
			commands := tt.setup(registry, backend)
			err := backend.Snapshot(context.Background(), tt.req)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("Expected error containing %q, got: %v", tt.wantErr, err)
			}
			if tt.wantDispatch >= 0 && commands != nil && len(commands) != tt.wantDispatch {
				t.Errorf("Expected %d dispatched commands, got %d", tt.wantDispatch, len(commands))
			}
		})
	}
}

func TestAppChannelReregistrationReplacesSession(t *testing.T) {
	registry := backends.NewChannelRegistry()
	backend := backends.NewAppChannelBackend(registry)

	// First registration never replies; its replacement acknowledges.
	old := registry.Register("job-1", nil, func(*pb.AgentCommand) error { return nil })
	commands := echoWorkload(registry, "job-1", nil)

	// Unregistering the replaced session must not evict the replacement.
	registry.Unregister(old)

	err := backend.Snapshot(context.Background(),
		channelReq("job-1", pb.SuspendMode_SUSPEND_MODE_OFFLOAD))
	if err != nil {
		t.Fatalf("Snapshot failed after re-registration: %v", err)
	}
	if len(commands) != 1 {
		t.Errorf("Expected the replacement session to receive the command")
	}
}

func TestWorkloadSessionClosed(t *testing.T) {
	registry := backends.NewChannelRegistry()
	send := func(*pb.AgentCommand) error { return nil }

	s1 := registry.Register("job-1", nil, send)
	if s1.Closed() {
		t.Error("Expected a freshly registered session to not be closed")
	}

	s2 := registry.Register("job-1", nil, send)
	if !s1.Closed() {
		t.Error("Expected a replaced session to be closed")
	}
	if s2.Closed() {
		t.Error("Expected the replacement session to not be closed")
	}

	registry.Unregister(s2)
	if !s2.Closed() {
		t.Error("Expected an unregistered session to be closed")
	}
}

func TestAppChannelModeResolution(t *testing.T) {
	offloadOnly := &pb.WorkloadCapabilities{
		SupportedModes: []pb.SuspendMode{pb.SuspendMode_SUSPEND_MODE_OFFLOAD},
	}
	discardDefault := &pb.WorkloadCapabilities{
		SupportedModes: []pb.SuspendMode{pb.SuspendMode_SUSPEND_MODE_OFFLOAD, pb.SuspendMode_SUSPEND_MODE_DISCARD},
		DefaultMode:    pb.SuspendMode_SUSPEND_MODE_DISCARD,
	}
	tests := []struct {
		name      string
		requested pb.SuspendMode
		caps      *pb.WorkloadCapabilities
		want      pb.SuspendMode
		wantErr   bool
	}{
		{
			"unspecified defaults to offload",
			pb.SuspendMode_SUSPEND_MODE_UNSPECIFIED, nil,
			pb.SuspendMode_SUSPEND_MODE_OFFLOAD, false,
		},
		{
			"unspecified uses workload default",
			pb.SuspendMode_SUSPEND_MODE_UNSPECIFIED, discardDefault,
			pb.SuspendMode_SUSPEND_MODE_DISCARD, false,
		},
		{
			"explicit mode overrides default",
			pb.SuspendMode_SUSPEND_MODE_OFFLOAD, discardDefault,
			pb.SuspendMode_SUSPEND_MODE_OFFLOAD, false,
		},
		{
			"unsupported mode rejected",
			pb.SuspendMode_SUSPEND_MODE_DISCARD, offloadOnly,
			pb.SuspendMode_SUSPEND_MODE_UNSPECIFIED, true,
		},
		{
			"no declared modes skips validation",
			pb.SuspendMode_SUSPEND_MODE_DISCARD, nil,
			pb.SuspendMode_SUSPEND_MODE_DISCARD, false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := backends.ResolveSuspendMode(tt.requested, tt.caps)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ResolveSuspendMode() error = %v, wantErr %v", err, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("ResolveSuspendMode() = %v, want %v", got, tt.want)
			}
		})
	}
}
