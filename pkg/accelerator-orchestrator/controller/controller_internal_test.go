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

	t.Run("Success Immediate", func(t *testing.T) {
		mockAgent := &mockAgentStoreForInternal{
			operationFunc: func(ctx context.Context, node, operationID string) (*agentpb.GetOperationResponse, error) {
				return &agentpb.GetOperationResponse{
					Status: agentpb.OperationStatus_OPERATION_STATUS_COMPLETE,
				}, nil
			},
		}
		c := &Controller{agentStore: mockAgent}
		err := c.waitForOperation(context.Background(), nodeName, opID)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	t.Run("Failure Immediate", func(t *testing.T) {
		errMsg := "something went wrong"
		mockAgent := &mockAgentStoreForInternal{
			operationFunc: func(ctx context.Context, node, operationID string) (*agentpb.GetOperationResponse, error) {
				return &agentpb.GetOperationResponse{
					Status: agentpb.OperationStatus_OPERATION_STATUS_FAILED,
					Error:  &errMsg,
				}, nil
			},
		}
		c := &Controller{agentStore: mockAgent}
		err := c.waitForOperation(context.Background(), nodeName, opID)
		if err == nil {
			t.Error("Expected error, got nil")
		} else if err.Error() != "operation op-123 failed: something went wrong" {
			t.Errorf("Unexpected error message: %v", err)
		}
	})

	t.Run("Pending Then Success", func(t *testing.T) {
		calls := 0
		mockAgent := &mockAgentStoreForInternal{
			operationFunc: func(ctx context.Context, node, operationID string) (*agentpb.GetOperationResponse, error) {
				calls++
				if calls < 3 {
					return &agentpb.GetOperationResponse{
						Status: agentpb.OperationStatus_OPERATION_STATUS_PENDING,
					}, nil
				}
				return &agentpb.GetOperationResponse{
					Status: agentpb.OperationStatus_OPERATION_STATUS_COMPLETE,
				}, nil
			},
		}
		c := &Controller{agentStore: mockAgent}

		start := time.Now()
		err := c.waitForOperation(context.Background(), nodeName, opID)
		duration := time.Since(start)

		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if calls != 3 {
			t.Errorf("Expected 3 calls, got %d", calls)
		}
		if duration < 2*time.Second {
			t.Errorf("Expected test to take at least 2 seconds, took %v", duration)
		}
	})

	t.Run("Context Timeout", func(t *testing.T) {
		mockAgent := &mockAgentStoreForInternal{
			operationFunc: func(ctx context.Context, node, operationID string) (*agentpb.GetOperationResponse, error) {
				return &agentpb.GetOperationResponse{
					Status: agentpb.OperationStatus_OPERATION_STATUS_PENDING,
				}, nil
			},
		}
		c := &Controller{agentStore: mockAgent}

		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
		defer cancel()

		err := c.waitForOperation(ctx, nodeName, opID)
		if err == nil {
			t.Error("Expected error due to timeout, got nil")
		} else if !errors.Is(err, context.DeadlineExceeded) && !errors.Is(errors.Unwrap(err), context.DeadlineExceeded) {
			t.Errorf("Expected context deadline exceeded error, got %v", err)
		}
	})
}
