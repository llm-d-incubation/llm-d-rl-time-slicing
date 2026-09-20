package controller

import (
	"context"
	"errors"
	"testing"
	"time"

	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
)

// TestController_WaitForOperation verifies the polling behavior of waitForOperation,
// including immediate success/failure, adaptive backoff over pending statuses, context cancellation,
// and bounding of consecutive GetOperation request failures.
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
				if duration < 30*time.Millisecond {
					t.Errorf("Expected test to take at least 30ms, took %v", duration)
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
		{
			name: "Exceeds Consecutive Failure Limit",
			operationFunc: func(ctx context.Context, node, operationID string) (*agentpb.GetOperationResponse, error) {
				return nil, errors.New("connection refused")
			},
			ctx: func() (context.Context, context.CancelFunc) {
				return context.Background(), func() {}
			},
			wantErrMsg: "operation op-123 status check failed 10 consecutive times: connection refused",
		},
		{
			name: "Transient Errors Reset Consecutive Failure Count",
			operationFunc: func() func(ctx context.Context, node, operationID string) (*agentpb.GetOperationResponse, error) {
				calls := 0
				return func(ctx context.Context, node, operationID string) (*agentpb.GetOperationResponse, error) {
					calls++
					// 5 failures, 1 pending success, 5 failures, then complete
					if (calls >= 1 && calls <= 5) || (calls >= 7 && calls <= 11) {
						return nil, errors.New("transient error")
					}
					if calls == 6 {
						return &agentpb.GetOperationResponse{
							Status: agentpb.OperationStatus_OPERATION_STATUS_PENDING,
						}, nil
					}
					return &agentpb.GetOperationResponse{
						Status: agentpb.OperationStatus_OPERATION_STATUS_COMPLETE,
					}, nil
				}
			}(),
			ctx: func() (context.Context, context.CancelFunc) {
				return context.Background(), func() {}
			},
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
			err := c.waitForOperation(ctx, "test-group", "test-job", nodeName, opID, "snapshot")
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
