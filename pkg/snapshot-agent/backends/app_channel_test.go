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

func TestAppChannelUnregisteredJobFailsFast(t *testing.T) {
	backend := backends.NewAppChannelBackend(backends.NewChannelRegistry())
	err := backend.Snapshot(context.Background(),
		channelReq("ghost", pb.SuspendMode_SUSPEND_MODE_UNSPECIFIED))
	if err == nil || !strings.Contains(err.Error(), "no workload channel registered") {
		t.Errorf("Expected unregistered-job error, got: %v", err)
	}
}

func TestAppChannelMissingConfig(t *testing.T) {
	backend := backends.NewAppChannelBackend(backends.NewChannelRegistry())
	err := backend.Snapshot(context.Background(), backends.Request{JobID: "job-1", Config: &pb.BackendConfig{}})
	if err == nil || !strings.Contains(err.Error(), "app_channel config is required") {
		t.Errorf("Expected missing-config error, got: %v", err)
	}
}

func TestAppChannelWorkloadFailurePropagates(t *testing.T) {
	registry := backends.NewChannelRegistry()
	backend := backends.NewAppChannelBackend(registry)
	var session *backends.WorkloadSession
	session = registry.Register("job-1", nil, func(cmd *pb.AgentCommand) error {
		go session.HandleResult(&pb.CommandResult{
			CommandId: cmd.GetCommandId(),
			Ok:        false,
			Error:     "engine exploded",
		})
		return nil
	})

	err := backend.Snapshot(context.Background(),
		channelReq("job-1", pb.SuspendMode_SUSPEND_MODE_OFFLOAD))
	if err == nil || !strings.Contains(err.Error(), "engine exploded") {
		t.Errorf("Expected workload error to propagate, got: %v", err)
	}
}

func TestAppChannelCommandTimeout(t *testing.T) {
	registry := backends.NewChannelRegistry()
	backend := backends.NewAppChannelBackend(registry)
	backend.SetCommandTimeout(50 * time.Millisecond)
	// Workload accepts commands but never replies.
	registry.Register("job-1", nil, func(*pb.AgentCommand) error { return nil })

	err := backend.Snapshot(context.Background(),
		channelReq("job-1", pb.SuspendMode_SUSPEND_MODE_OFFLOAD))
	if err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Errorf("Expected timeout error, got: %v", err)
	}
}

func TestAppChannelDisconnectFailsInflightCommand(t *testing.T) {
	registry := backends.NewChannelRegistry()
	backend := backends.NewAppChannelBackend(registry)
	var session *backends.WorkloadSession
	// Workload drops the connection right after receiving the command.
	session = registry.Register("job-1", nil, func(*pb.AgentCommand) error {
		go registry.Unregister(session)
		return nil
	})

	err := backend.Snapshot(context.Background(),
		channelReq("job-1", pb.SuspendMode_SUSPEND_MODE_OFFLOAD))
	if err == nil || !strings.Contains(err.Error(), "disconnected") {
		t.Errorf("Expected disconnect error, got: %v", err)
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

// TestAppChannelUnsupportedModeFailsBeforeDispatch verifies capability
// validation happens before any command reaches the workload.
func TestAppChannelUnsupportedModeFailsBeforeDispatch(t *testing.T) {
	registry := backends.NewChannelRegistry()
	backend := backends.NewAppChannelBackend(registry)
	commands := echoWorkload(registry, "trainer-1", &pb.WorkloadCapabilities{
		SupportedModes: []pb.SuspendMode{pb.SuspendMode_SUSPEND_MODE_OFFLOAD},
	})

	err := backend.Snapshot(context.Background(),
		channelReq("trainer-1", pb.SuspendMode_SUSPEND_MODE_DISCARD))
	if err == nil || !strings.Contains(err.Error(), "does not support suspend mode") {
		t.Errorf("Expected unsupported-mode error, got: %v", err)
	}
	if len(commands) != 0 {
		t.Errorf("Expected no command to be dispatched, got %d", len(commands))
	}
}
