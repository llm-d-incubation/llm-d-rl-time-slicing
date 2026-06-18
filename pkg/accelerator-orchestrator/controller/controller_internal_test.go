package controller

import (
	"context"
	"errors"
	"testing"
	"time"

	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
)



func TestController_WaitForOperation(t *testing.T) {
	nodeName := "node-1"
	opID := "op-123"

	tests := []struct {
		name               string
		operationFunc      func(ctx context.Context, node, operationID string) (*agentpb.GetOperationResponse, error)
		operationResponses []*agentpb.GetOperationResponse
		ctx                func() (context.Context, context.CancelFunc)
		wantErr            error
		wantErrMsg         string
		verify             func(t *testing.T, duration time.Duration, mock *MockSnapshotAgentStore)
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
			operationResponses: []*agentpb.GetOperationResponse{
				{Status: agentpb.OperationStatus_OPERATION_STATUS_PENDING},
				{Status: agentpb.OperationStatus_OPERATION_STATUS_PENDING},
				{Status: agentpb.OperationStatus_OPERATION_STATUS_COMPLETE},
			},
			ctx: func() (context.Context, context.CancelFunc) {
				return context.Background(), func() {}
			},
			verify: func(t *testing.T, duration time.Duration, mock *MockSnapshotAgentStore) {
				t.Helper()
				if mock.OperationIndex != 3 {
					t.Errorf("Expected 3 calls, got %d", mock.OperationIndex)
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
			mockAgent := &MockSnapshotAgentStore{
				OperationFunc:      tc.operationFunc,
				OperationResponses: tc.operationResponses,
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
				tc.verify(t, duration, mockAgent)
			}
		})
	}
}
