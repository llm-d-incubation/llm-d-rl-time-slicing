package controller

import (
	"context"
	"errors"
	"testing"
	"time"

	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/timeslice-orchestrator/store"
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

// TestIsJobLoaded_EmptyGroupVacuouslyLoaded verifies that a group with zero
// member nodes (created by a first Acquire before any pods are deployed) has
// nothing to restore, so its locking job counts as loaded — otherwise the
// initial Acquire of the lock-then-deploy pattern would wait forever.
func TestIsJobLoaded_EmptyGroupVacuouslyLoaded(t *testing.T) {
	ctx := context.Background()
	groupStore := store.NewGroupStore(store.NewMemLockStore())
	jobStore := store.NewJobStore()
	c := NewController(groupStore, jobStore, nil, nil, nil)

	g, _, err := groupStore.GetOrCreate(ctx, "empty-group")
	if err != nil {
		t.Fatalf("failed to create group: %v", err)
	}

	loaded, err := c.isJobLoaded(ctx, g, "job-1")
	if err != nil {
		t.Fatalf("isJobLoaded failed: %v", err)
	}
	if !loaded {
		t.Errorf("expected job to be vacuously loaded on a zero-node group")
	}

	// Sanity check: the empty-jobID short-circuit is unaffected.
	loaded, err = c.isJobLoaded(ctx, g, "")
	if err != nil || loaded {
		t.Errorf("expected empty jobID to be not loaded, got loaded=%v err=%v", loaded, err)
	}
}

// TestTryDeduceActiveJob_SkipsEmptyGroup verifies that active-job deduction
// does nothing for a zero-node group: with isJobLoaded vacuously true on
// empty groups, deduction would otherwise pick an arbitrary job as active
// while its pods are still Pending.
func TestTryDeduceActiveJob_SkipsEmptyGroup(t *testing.T) {
	ctx := context.Background()
	groupStore := store.NewGroupStore(store.NewMemLockStore())
	jobStore := store.NewJobStore()
	c := NewController(groupStore, jobStore, nil, nil, nil)

	g, _, err := groupStore.GetOrCreate(ctx, "empty-group")
	if err != nil {
		t.Fatalf("failed to create group: %v", err)
	}
	// A known job with unscheduled pods only (no context state anywhere).
	if err := jobStore.Put(ctx, store.NewJob("empty-group", "job-1")); err != nil {
		t.Fatalf("failed to put job: %v", err)
	}

	if err := c.tryDeduceActiveJob(ctx, g); err != nil {
		t.Fatalf("tryDeduceActiveJob failed: %v", err)
	}
	if got := g.Spec().ActiveJob(); got != "" {
		t.Errorf("expected no active job deduced for zero-node group, got %q", got)
	}
}
