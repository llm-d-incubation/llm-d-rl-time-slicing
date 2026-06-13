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
		state      pb.GroupStatus_State
	}{
		{
			name:       "empty/nil values",
			groupID:    "",
			nodes:      nil,
			lockingJob: "",
			state:      pb.GroupStatus_STATE_UNSPECIFIED,
		},
		{
			name:       "populated values",
			groupID:    "group-123",
			nodes:      []string{"node-a", "node-b"},
			lockingJob: "job-a",
			state:      pb.GroupStatus_STATE_IDLE,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var wrappedLockStore store.GroupLockStore
			if tc.lockingJob != "" {
				lockStore := store.NewMemLockStore()
				lockStore.Lock(context.Background(), tc.groupID, tc.lockingJob)
				wrappedLockStore = store.NewGroupLockStoreWrapper(lockStore, tc.groupID)
			}
			group, err := store.NewGroup(context.Background(), tc.groupID, wrappedLockStore)
			if err != nil {
				t.Fatalf("NewGroup failed: %v", err)
			}
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

			if group.Spec().LockingJob() != tc.lockingJob {
				t.Errorf("LockingJob() = %q, want %q", group.Spec().LockingJob(), tc.lockingJob)
			}

			if group.Spec().ActiveJob() != tc.lockingJob {
				t.Errorf("ActiveJob() = %q, want %q", group.Spec().ActiveJob(), tc.lockingJob)
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

func TestGroup_Delete(t *testing.T) {
	ctx := context.Background()

	t.Run("delete with MemLockStore", func(t *testing.T) {
		lockStore := store.NewMemLockStore()
		groupStore := store.NewGroupStore(lockStore)
		groupID := "group-1"
		jobID := "job-a"

		// 1. Initialize group with job-a holding the lock
		if err := lockStore.Lock(ctx, groupID, jobID); err != nil {
			t.Fatalf("failed to lock: %v", err)
		}
		group, _, err := groupStore.GetOrCreate(ctx, groupID)
		if err != nil {
			t.Fatalf("failed to get or create group: %v", err)
		}

		// 2. Delete should succeed and unlock
		err = group.Spec().Delete(ctx)
		if err != nil {
			t.Fatalf("delete failed: %v", err)
		}
		if group.Spec().LockingJob() != "" {
			t.Errorf("expected group to be unlocked, got %q", group.Spec().LockingJob())
		}

		// 3. Delete again should be idempotent (succeed)
		err = group.Spec().Delete(ctx)
		if err != nil {
			t.Fatalf("subsequent delete failed: %v", err)
		}
	})
}

func TestNewGroup_InitializeLock(t *testing.T) {
	ctx := context.Background()
	lockStore := store.NewMemLockStore()
	groupID := "group-test"
	jobID := "job-lock"

	// 1. Lock the group in the lockStore directly
	if err := lockStore.Lock(ctx, groupID, jobID); err != nil {
		t.Fatalf("failed to setup lock in lockStore: %v", err)
	}

	// 2. Create NewGroup and check if it initialized lockingJob
	wrappedLockStore := store.NewGroupLockStoreWrapper(lockStore, groupID)
	group, err := store.NewGroup(ctx, groupID, wrappedLockStore)
	if err != nil {
		t.Fatalf("NewGroup failed: %v", err)
	}

	if group.Spec().LockingJob() != jobID {
		t.Errorf("NewGroup did not initialize lockingJob from lockStore, got %q, want %q", group.Spec().LockingJob(), jobID)
	}
	if group.Spec().ActiveJob() != jobID {
		t.Errorf("NewGroup did not initialize activeJob from lockStore, got %q, want %q", group.Spec().ActiveJob(), jobID)
	}
}

func TestGroupSpec_RequestLock(t *testing.T) {
	ctx := context.Background()
	
	// Start with job-1 holding the lock
	lockStore := store.NewMemLockStore()
	lockStore.Lock(ctx, "group-1", "job-1")
	wrapped := store.NewGroupLockStoreWrapper(lockStore, "group-1")
	group, err := store.NewGroup(ctx, "group-1", wrapped)
	if err != nil {
		t.Fatalf("failed to create group: %v", err)
	}
	spec := group.Spec()
	queue := spec.GetWaitingJobQueue()

	// 1. Request lock for job-1 (already holds lock) -> should NOT enqueue
	spec.RequestLock("job-1")
	if queue.Len() != 0 {
		t.Errorf("Expected queue to be empty, got %d", queue.Len())
	}

	// 2. Request lock for job-2 (does not hold lock) -> should enqueue
	spec.RequestLock("job-2")
	if queue.Len() != 1 {
		t.Errorf("Expected queue len to be 1, got %d", queue.Len())
	}
	if !queue.Exists("job-2") {
		t.Errorf("Expected job-2 to be in queue")
	}

	// 3. Request lock for job-2 again -> should be no-op (no duplicate)
	spec.RequestLock("job-2")
	if queue.Len() != 1 {
		t.Errorf("Expected queue len to remain 1, got %d", queue.Len())
	}
}

func TestGroupSpec_Yield(t *testing.T) {
	ctx := context.Background()

	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	groupID := "group-1"
	
	// Pre-lock job-1
	if err := lockStore.Lock(ctx, groupID, "job-1"); err != nil {
		t.Fatalf("failed to lock: %v", err)
	}
	
	group, _, err := groupStore.GetOrCreate(ctx, groupID)
	if err != nil {
		t.Fatalf("failed to get or create group: %v", err)
	}
	spec := group.Spec()
	queue := spec.GetWaitingJobQueue()

	// Now job-1 is locked.
	// 1. Enqueue job-2
	spec.RequestLock("job-2")
	// 2. Enqueue job-3
	spec.RequestLock("job-3")

	// Verify queue
	queueList := queue.List()
	if len(queueList) != 2 {
		t.Fatalf("expected queue len 2, got %d", len(queueList))
	}
	if queueList[0].JobID != "job-2" || queueList[1].JobID != "job-3" {
		t.Errorf("unexpected queue content: %+v", queueList)
	}

	// 3. Yield by non-holder (job-2) should fail
	err = spec.Yield(ctx, "job-2")
	if !errors.Is(err, store.ErrNotLockHolder) {
		t.Fatalf("expected ErrNotLockHolder, got %v", err)
	}
	if spec.LockingJob() != "job-1" {
		t.Errorf("expected group to remain locked by job-1, got %q", spec.LockingJob())
	}

	// 4. Yield by holder (job-1) should succeed and grant to job-2
	err = spec.Yield(ctx, "job-1")
	if err != nil {
		t.Fatalf("yield failed: %v", err)
	}
	if spec.LockingJob() != "job-2" {
		t.Errorf("expected lockingJob to be job-2, got %q", spec.LockingJob())
	}
	
	// Verify queue (should only have job-3)
	queueList = queue.List()
	if len(queueList) != 1 {
		t.Fatalf("expected queue len 1, got %d", len(queueList))
	}
	if queueList[0].JobID != "job-3" {
		t.Errorf("unexpected queue content: %+v", queueList)
	}

	// 5. Yield by holder (job-2) should succeed and grant to job-3
	err = spec.Yield(ctx, "job-2")
	if err != nil {
		t.Fatalf("yield failed: %v", err)
	}
	if spec.LockingJob() != "job-3" {
		t.Errorf("expected lockingJob to be job-3, got %q", spec.LockingJob())
	}
	if queue.Len() != 0 {
		t.Errorf("expected queue to be empty, got %d", queue.Len())
	}

	// 6. Yield by holder (job-3) should succeed and group becomes unlocked
	err = spec.Yield(ctx, "job-3")
	if err != nil {
		t.Fatalf("yield failed: %v", err)
	}
	if spec.LockingJob() != "" {
		t.Errorf("expected lockingJob to be empty, got %q", spec.LockingJob())
	}
}

func TestGroup_Snapshot(t *testing.T) {
	ctx := context.Background()
	lockStore := store.NewMemLockStore()
	if err := lockStore.Lock(ctx, "group-1", "job-lock"); err != nil {
		t.Fatalf("failed to lock: %v", err)
	}
	wrappedLockStore := store.NewGroupLockStoreWrapper(lockStore, "group-1")
	group, err := store.NewGroup(ctx, "group-1", wrappedLockStore)
	if err != nil {
		t.Fatalf("failed to create group: %v", err)
	}

	group.SetNodes([]string{"node-a", "node-b"})
	group.SetState(pb.GroupStatus_STATE_IDLE)
	group.Spec().SetActiveJob("job-active")
	group.Spec().GetWaitingJobQueue().Enqueue("job-wait-1")
	group.Spec().GetWaitingJobQueue().Enqueue("job-wait-2")

	// Take snapshot
	snap := group.Snapshot()

	// Verify snapshot values
	if snap.ID != "group-1" {
		t.Errorf("Snap ID = %q, want %q", snap.ID, "group-1")
	}
	if !reflect.DeepEqual(snap.Nodes, []string{"node-a", "node-b"}) {
		t.Errorf("Snap Nodes = %+v, want %+v", snap.Nodes, []string{"node-a", "node-b"})
	}
	if snap.State != pb.GroupStatus_STATE_IDLE {
		t.Errorf("Snap State = %v, want %v", snap.State, pb.GroupStatus_STATE_IDLE)
	}
	if snap.LockingJob != "job-lock" {
		t.Errorf("Snap LockingJob = %q, want %q", snap.LockingJob, "job-lock")
	}
	if snap.ActiveJob != "job-active" {
		t.Errorf("Snap ActiveJob = %q, want %q", snap.ActiveJob, "job-active")
	}
	if snap.WaiterQueueDepth != 2 {
		t.Errorf("Snap WaiterQueueDepth = %d, want %d", snap.WaiterQueueDepth, 2)
	}

	// Modify original group
	group.SetNodes([]string{"node-c"})
	group.SetState(pb.GroupStatus_STATE_LOCKED)

	if reflect.DeepEqual(snap.Nodes, group.Nodes()) {
		t.Errorf("Snap Nodes changed after original changed (deep copy failed)")
	}
	state, _ := group.State()
	if snap.State == state {
		t.Errorf("Snap State changed after original changed")
	}
}
