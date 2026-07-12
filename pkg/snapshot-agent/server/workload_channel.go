package server

import (
	"errors"
	"io"
	"log/slog"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/logging"
	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// WorkloadChannel handles a workload's long-lived registration stream. The
// first message must be a RegisterWorkload; the handler then routes
// CommandResults back to in-flight commands until the stream ends.
// Registration only binds the job to this stream — state-machine transitions
// stay with their existing owners (the watcher in k8s mode).
func (s *Server) WorkloadChannel(stream pb.SnapshotAgentService_WorkloadChannelServer) error {
	ctx := logging.WithServerMethod(stream.Context(), "WorkloadChannel")

	msg, err := stream.Recv()
	if err != nil {
		return status.Errorf(codes.InvalidArgument, "failed to receive registration message: %v", err)
	}
	reg := msg.GetRegister()
	if reg == nil {
		return status.Errorf(codes.InvalidArgument, "first message on the workload channel must be a RegisterWorkload")
	}
	if reg.GetJobId() == "" {
		return status.Errorf(codes.InvalidArgument, "job_id is required to register a workload")
	}

	ctx = logging.WithJobID(ctx, reg.GetJobId())
	ctx = logging.WithGroupID(ctx, reg.GetGroup())
	slog.InfoContext(ctx, "Workload registered", "capabilities", reg.GetCapabilities())

	session := s.channelRegistry.Register(reg.GetJobId(), reg.GetCapabilities(), stream.Send)
	defer func() {
		s.channelRegistry.Unregister(session)
		slog.InfoContext(ctx, "Workload channel closed")
	}()

	// Make the job visible to Status and the state machine.
	s.state.RegisterJob(reg.GetJobId(), reg.GetGroup())

	for {
		msg, err := stream.Recv()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
		if res := msg.GetResult(); res != nil {
			session.HandleResult(res)
			continue
		}
		if msg.GetRegister() != nil {
			return status.Errorf(codes.InvalidArgument, "workload is already registered on this stream")
		}
	}
}
