package store_test

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
)

func TestGroup_GettersAndSetters(t *testing.T) {
	tests := []struct {
		name       string
		groupID    string
		nodes      []string
		lockingJob string
		activeJob  string
		state      pb.GroupStatus_State
	}{
		{
			name:       "empty/nil values",
			groupID:    "",
			nodes:      nil,
			lockingJob: "",
			activeJob:  "",
			state:      pb.GroupStatus_STATE_UNSPECIFIED,
		},
		{
			name:       "populated values",
			groupID:    "group-123",
			nodes:      []string{"node-a", "node-b"},
			lockingJob: "job-a",
			activeJob:  "job-b",
			state:      pb.GroupStatus_STATE_IDLE,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			group := store.NewGroup(tc.groupID, nil)
			group.SetNodes(tc.nodes)

			if group.ID() != tc.groupID {
				t.Errorf("ID() = %q, want %q", group.ID(), tc.groupID)
			}

			if !reflect.DeepEqual(group.Nodes(), tc.nodes) {
				t.Errorf("Nodes() = %+v, want %+v", group.Nodes(), tc.nodes)
			}

			// Verify deep copy/mutation protection for Nodes
			if len(tc.nodes) > 0 {
				mutatedNodes := group.Nodes()
				mutatedNodes[0] = "mutated"
				if reflect.DeepEqual(group.Nodes(), mutatedNodes) {
					t.Errorf("mutating returned Nodes slice modified the internal state")
				}
			}

			if tc.lockingJob != "" {
				if err := group.Lock(context.Background(), tc.lockingJob); err != nil {
					t.Fatalf("failed to lock: %v", err)
				}
			}
			if group.LockingJob() != tc.lockingJob {
				t.Errorf("LockingJob() = %q, want %q", group.LockingJob(), tc.lockingJob)
			}

			group.SetActiveJob(tc.activeJob)
			if group.ActiveJob() != tc.activeJob {
				t.Errorf("ActiveJob() = %q, want %q", group.ActiveJob(), tc.activeJob)
			}

			_, initialTimestamp := group.State()
			beforeSet := time.Now()
			group.SetState(tc.state)
			gotState, gotTimestamp := group.State()

			if gotState != tc.state {
				t.Errorf("State() state = %v, want %v", gotState, tc.state)
			}
			if tc.state != pb.GroupStatus_STATE_UNSPECIFIED {
				if gotTimestamp.Before(beforeSet) || gotTimestamp.After(time.Now()) {
					t.Errorf("State() timestamp %v is not close to set time", gotTimestamp)
				}
			} else {
				if !gotTimestamp.Equal(initialTimestamp) {
					t.Errorf("State() timestamp %v changed from %v, want unmodified", gotTimestamp, initialTimestamp)
				}
			}
		})
	}
}

func TestGroup_LockAndUnlock(t *testing.T) {
	ctx := context.Background()

	type step struct {
		op          string // "lock" or "unlock"
		jobID       string
		expectedErr error
		wantLocked  string
	}

	tests := []struct {
		name      string
		lockStore store.LockStore
		steps     []step
	}{
		{
			name:      "lock/unlock sequence without lockStore",
			lockStore: nil,
			steps: []step{
				{op: "lock", jobID: "job-a", expectedErr: nil, wantLocked: "job-a"},
				// Without lockStore, memory locks are overwritten directly in Lock()
				// since it doesn't check local LockingJob.
				{op: "lock", jobID: "job-b", expectedErr: nil, wantLocked: "job-b"},
				{op: "unlock", jobID: "job-b", expectedErr: nil, wantLocked: ""},
			},
		},
		{
			name:      "lock/unlock sequence with MemLockStore",
			lockStore: store.NewMemLockStore(),
			steps: []step{
				{op: "lock", jobID: "job-a", expectedErr: nil, wantLocked: "job-a"},
				{op: "lock", jobID: "job-b", expectedErr: store.ErrAlreadyLocked, wantLocked: "job-a"},
				{op: "lock", jobID: "job-a", expectedErr: nil, wantLocked: "job-a"}, // idempotent
				{op: "unlock", jobID: "job-b", expectedErr: store.ErrNotLockHolder, wantLocked: "job-a"},
				{op: "unlock", jobID: "job-a", expectedErr: nil, wantLocked: ""},
				{op: "unlock", jobID: "job-a", expectedErr: nil, wantLocked: ""}, // idempotent unlock
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			groupStore := store.NewGroupStore(tc.lockStore)
			group, _, err := groupStore.GetOrCreate(ctx, "group-1")
			if err != nil {
				t.Fatalf("failed to get or create group: %v", err)
			}

			for i, s := range tc.steps {
				var err error
				switch s.op {
				case "lock":
					err = group.Lock(ctx, s.jobID)
				case "unlock":
					err = group.Unlock(ctx, s.jobID)
				}

				if !errors.Is(err, s.expectedErr) {
					t.Fatalf("step %d: %s with %q returned error %v, want %v", i, s.op, s.jobID, err, s.expectedErr)
				}

				if group.LockingJob() != s.wantLocked {
					t.Errorf("step %d: after %s, LockingJob() = %q, want %q", i, s.op, group.LockingJob(), s.wantLocked)
				}
			}
		})
	}
}
