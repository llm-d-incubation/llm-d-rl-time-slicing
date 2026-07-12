package server

import (
	"context"
	"net"
	"strings"
	"testing"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

// newChannelTestServer starts a server whose app-channel backend shares a
// registry with the WorkloadChannel handler, and returns a connected client.
func newChannelTestServer(t *testing.T) (*Server, *backends.ChannelRegistry, pb.SnapshotAgentServiceClient) {
	t.Helper()
	lisChan := bufconn.Listen(bufSize)
	s := grpc.NewServer()
	registry := backends.NewChannelRegistry()
	backendsMap := map[backends.BackendType]backends.Backend{
		backends.BackendAppChannel: backends.NewAppChannelBackend(registry),
	}
	srv := NewServer(backendsMap, backends.BackendAppChannel, "k8s", registry)
	pb.RegisterSnapshotAgentServiceServer(s, srv)
	go func() {
		if err := s.Serve(lisChan); err != nil {
			return
		}
	}()
	t.Cleanup(s.GracefulStop)

	conn, err := grpc.NewClient("passthrough://bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return lisChan.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial bufnet: %v", err)
	}
	t.Cleanup(func() { conn.Close() })
	return srv, registry, pb.NewSnapshotAgentServiceClient(conn)
}

// registerWorkload opens a WorkloadChannel stream, registers jobID, and
// starts an echo loop acknowledging every command. It blocks until the
// registration is visible in the registry.
func registerWorkload(
	ctx context.Context,
	t *testing.T,
	client pb.SnapshotAgentServiceClient,
	registry *backends.ChannelRegistry,
	jobID string,
	caps *pb.WorkloadCapabilities,
) {
	t.Helper()
	stream, err := client.WorkloadChannel(ctx)
	if err != nil {
		t.Fatalf("Failed to open workload channel: %v", err)
	}
	err = stream.Send(&pb.WorkloadMessage{
		Message: &pb.WorkloadMessage_Register{
			Register: &pb.RegisterWorkload{JobId: jobID, Capabilities: caps},
		},
	})
	if err != nil {
		t.Fatalf("Failed to send registration: %v", err)
	}
	for range 100 {
		if _, sErr := registry.Session(jobID); sErr == nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, sErr := registry.Session(jobID); sErr != nil {
		t.Fatalf("Workload registration never became visible: %v", sErr)
	}

	go func() {
		for {
			cmd, rErr := stream.Recv()
			if rErr != nil {
				return
			}
			sErr := stream.Send(&pb.WorkloadMessage{
				Message: &pb.WorkloadMessage_Result{
					Result: &pb.CommandResult{CommandId: cmd.GetCommandId(), Ok: true},
				},
			})
			if sErr != nil {
				return
			}
		}
	}()
}

func waitForOperation(
	ctx context.Context,
	t *testing.T,
	client pb.SnapshotAgentServiceClient,
	opID string,
) *pb.GetOperationResponse {
	t.Helper()
	var opResp *pb.GetOperationResponse
	var err error
	for range 100 {
		opResp, err = client.GetOperation(ctx, &pb.GetOperationRequest{OperationId: opID})
		if err != nil {
			t.Fatalf("GetOperation failed: %v", err)
		}
		if opResp.GetStatus() != pb.OperationStatus_OPERATION_STATUS_PENDING {
			return opResp
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("Operation %s never left PENDING", opID)
	return nil
}

func TestServer_WorkloadChannel_SnapshotRestoreRoundTrip(t *testing.T) {
	srv, registry, client := newChannelTestServer(t)
	ctx := context.Background()

	registerWorkload(ctx, t, client, registry, "chan-job", &pb.WorkloadCapabilities{
		SupportedModes: []pb.SuspendMode{
			pb.SuspendMode_SUSPEND_MODE_OFFLOAD,
			pb.SuspendMode_SUSPEND_MODE_DISCARD,
		},
		DefaultMode: pb.SuspendMode_SUSPEND_MODE_OFFLOAD,
	})

	// Registration binds the stream but does not transition state.
	if err := srv.state.TransitionToRunning("chan-job", nil); err != nil {
		t.Fatalf("Failed to transition job to RUNNING: %v", err)
	}

	channelCfg := &pb.BackendConfig{
		Backend: &pb.BackendConfig_AppChannel{AppChannel: &pb.AppChannelConfig{}},
	}
	snapResp, err := client.Snapshot(ctx, &pb.SnapshotRequest{JobId: "chan-job", BackendConfig: channelCfg})
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	opResp := waitForOperation(ctx, t, client, snapResp.GetOperationId())
	if opResp.GetStatus() != pb.OperationStatus_OPERATION_STATUS_COMPLETE {
		t.Fatalf("Expected snapshot COMPLETE, got %v (error: %q)", opResp.GetStatus(), opResp.GetError())
	}

	restoreResp, err := client.Restore(ctx, &pb.RestoreRequest{JobId: "chan-job", BackendConfig: channelCfg})
	if err != nil {
		t.Fatalf("Restore failed: %v", err)
	}
	opResp = waitForOperation(ctx, t, client, restoreResp.GetOperationId())
	if opResp.GetStatus() != pb.OperationStatus_OPERATION_STATUS_COMPLETE {
		t.Fatalf("Expected restore COMPLETE, got %v (error: %q)", opResp.GetStatus(), opResp.GetError())
	}
}

func TestServer_WorkloadChannel_UnregisteredJobFails(t *testing.T) {
	srv, _, client := newChannelTestServer(t)
	ctx := context.Background()

	srv.state.RegisterJob("no-channel-job", "")
	if err := srv.state.TransitionToRunning("no-channel-job", nil); err != nil {
		t.Fatalf("Failed to transition job to RUNNING: %v", err)
	}

	snapResp, err := client.Snapshot(ctx, &pb.SnapshotRequest{
		JobId: "no-channel-job",
		BackendConfig: &pb.BackendConfig{
			Backend: &pb.BackendConfig_AppChannel{AppChannel: &pb.AppChannelConfig{}},
		},
	})
	if err != nil {
		t.Fatalf("Snapshot RPC should be accepted, got: %v", err)
	}
	opResp := waitForOperation(ctx, t, client, snapResp.GetOperationId())
	if opResp.GetStatus() != pb.OperationStatus_OPERATION_STATUS_FAILED {
		t.Fatalf("Expected FAILED for unregistered job, got %v", opResp.GetStatus())
	}
	if !strings.Contains(opResp.GetError(), "no workload channel registered") {
		t.Errorf("Expected unregistered-channel error, got: %q", opResp.GetError())
	}
}

func TestServer_WorkloadChannel_RegistrationValidation(t *testing.T) {
	_, _, client := newChannelTestServer(t)
	ctx := context.Background()

	// First message must be a registration.
	stream, err := client.WorkloadChannel(ctx)
	if err != nil {
		t.Fatalf("Failed to open workload channel: %v", err)
	}
	err = stream.Send(&pb.WorkloadMessage{
		Message: &pb.WorkloadMessage_Result{Result: &pb.CommandResult{CommandId: "x"}},
	})
	if err != nil {
		t.Fatalf("Send failed: %v", err)
	}
	_, err = stream.Recv()
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("Expected InvalidArgument for non-registration first message, got: %v", err)
	}

	// job_id is required.
	stream, err = client.WorkloadChannel(ctx)
	if err != nil {
		t.Fatalf("Failed to open workload channel: %v", err)
	}
	err = stream.Send(&pb.WorkloadMessage{
		Message: &pb.WorkloadMessage_Register{Register: &pb.RegisterWorkload{}},
	})
	if err != nil {
		t.Fatalf("Send failed: %v", err)
	}
	_, err = stream.Recv()
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("Expected InvalidArgument for missing job_id, got: %v", err)
	}
}
