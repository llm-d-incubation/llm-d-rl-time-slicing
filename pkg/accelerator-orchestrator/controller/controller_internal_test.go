package controller

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
)

type mockAgentStoreForInternal struct {
	store.SnapshotAgentStore
	getStatusFunc func(ctx context.Context, nodeName string) (*agentpb.StatusResponse, error)
	snapshotFunc  func(ctx context.Context, nodeName, jobID, groupID string) (*agentpb.SnapshotResponse, error)
	operationFunc func(ctx context.Context, nodeName, operationID string) (*agentpb.GetOperationResponse, error)
}

func (m *mockAgentStoreForInternal) GetStatus(ctx context.Context, nodeName string) (*agentpb.StatusResponse, error) {
	if m.getStatusFunc != nil {
		return m.getStatusFunc(ctx, nodeName)
	}
	return &agentpb.StatusResponse{}, nil
}

func (m *mockAgentStoreForInternal) CloseClient(nodeName string) error {
	return nil
}

func (m *mockAgentStoreForInternal) Snapshot(
	ctx context.Context, nodeName, jobID, groupID string,
) (*agentpb.SnapshotResponse, error) {
	if m.snapshotFunc != nil {
		return m.snapshotFunc(ctx, nodeName, jobID, groupID)
	}
	return &agentpb.SnapshotResponse{}, nil
}

func (m *mockAgentStoreForInternal) GetOperation(
	ctx context.Context, nodeName, operationID string,
) (*agentpb.GetOperationResponse, error) {
	if m.operationFunc != nil {
		return m.operationFunc(ctx, nodeName, operationID)
	}
	return &agentpb.GetOperationResponse{}, nil
}

func TestController_WaitForOperation(t *testing.T) {
	nodeName := "node-1"
	opID := "op-123"

	pendingCalls := 0

	tests := []struct {
		name          string
		operationFunc func(ctx context.Context, node, operationID string) (*agentpb.GetOperationResponse, error)
		ctx           func() (context.Context, context.CancelFunc)
		wantErr       error
		wantErrMsg    string
		verify        func(t *testing.T, duration time.Duration)
	}{
		{
			name: "Success Immediate",
			operationFunc: func(ctx context.Context, node, operationID string) (*agentpb.GetOperationResponse, error) {
				return &agentpb.GetOperationResponse{
					Status: agentpb.OperationStatus_OPERATION_STATUS_COMPLETE,
				}, nil
			},
			ctx: func() (context.Context, context.CancelFunc) {
				return context.Background(), func() {}
			},
		},
		{
			name: "Failure Immediate",
			operationFunc: func(ctx context.Context, node, operationID string) (*agentpb.GetOperationResponse, error) {
				errMsg := "something went wrong"
				return &agentpb.GetOperationResponse{
					Status: agentpb.OperationStatus_OPERATION_STATUS_FAILED,
					Error:  &errMsg,
				}, nil
			},
			ctx: func() (context.Context, context.CancelFunc) {
				return context.Background(), func() {}
			},
			wantErrMsg: "operation op-123 failed: something went wrong",
		},
		{
			name: "Pending Then Success",
			operationFunc: func(ctx context.Context, node, operationID string) (*agentpb.GetOperationResponse, error) {
				pendingCalls++
				if pendingCalls < 3 {
					return &agentpb.GetOperationResponse{
						Status: agentpb.OperationStatus_OPERATION_STATUS_PENDING,
					}, nil
				}
				return &agentpb.GetOperationResponse{
					Status: agentpb.OperationStatus_OPERATION_STATUS_COMPLETE,
				}, nil
			},
			ctx: func() (context.Context, context.CancelFunc) {
				return context.Background(), func() {}
			},
			verify: func(t *testing.T, duration time.Duration) {
				t.Helper()
				if pendingCalls != 3 {
					t.Errorf("Expected 3 calls, got %d", pendingCalls)
				}
				if duration < 2*time.Second {
					t.Errorf("Expected test to take at least 2 seconds, took %v", duration)
				}
			},
		},
		{
			name: "Context Timeout",
			operationFunc: func(ctx context.Context, node, operationID string) (*agentpb.GetOperationResponse, error) {
				return &agentpb.GetOperationResponse{
					Status: agentpb.OperationStatus_OPERATION_STATUS_PENDING,
				}, nil
			},
			ctx: func() (context.Context, context.CancelFunc) {
				return context.WithTimeout(context.Background(), 500*time.Millisecond)
			},
			wantErr: context.DeadlineExceeded,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mockAgent := &mockAgentStoreForInternal{
				operationFunc: tc.operationFunc,
			}
			c := &Controller{agentStore: mockAgent}

			ctx, cancel := tc.ctx()
			defer cancel()

			start := time.Now()
			err := c.waitForOperation(ctx, nodeName, opID)
			duration := time.Since(start)

			switch {
			case tc.wantErr != nil:
				if err == nil {
					t.Errorf("Expected error %v, got nil", tc.wantErr)
				} else if !errors.Is(err, tc.wantErr) && !errors.Is(errors.Unwrap(err), tc.wantErr) {
					t.Errorf("Expected error to wrap %v, got %v", tc.wantErr, err)
				}
			case tc.wantErrMsg != "":
				if err == nil {
					t.Errorf("Expected error message %q, got nil", tc.wantErrMsg)
				} else if err.Error() != tc.wantErrMsg {
					t.Errorf("Expected error message %q, got %q", tc.wantErrMsg, err.Error())
				}
			default:
				if err != nil {
					t.Errorf("Expected no error, got %v", err)
				}
			}

			if tc.verify != nil {
				tc.verify(t, duration)
			}
		})
	}
}
