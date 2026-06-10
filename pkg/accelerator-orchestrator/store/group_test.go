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
			group, err := store.NewGroup(context.Background(), tc.groupID, nil)
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

			if tc.lockingJob != "" {
				if err := group.Spec().Lock(context.Background(), tc.lockingJob); err != nil {
					t.Fatalf("failed to lock: %v", err)
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
					err = group.Spec().Lock(ctx, s.jobID)
				case "unlock":
					err = group.Spec().Unlock(ctx, s.jobID)
				}

				if !errors.Is(err, s.expectedErr) {
					t.Fatalf("step %d: %s with %q returned error %v, want %v", i, s.op, s.jobID, err, s.expectedErr)
				}

				if group.Spec().LockingJob() != s.wantLocked {
					t.Errorf("step %d: after %s, LockingJob() = %q, want %q", i, s.op, group.Spec().LockingJob(), s.wantLocked)
				}
			}
		})
	}
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
	group, err := store.NewGroup(ctx, "group-1", nil)
	if err != nil {
		t.Fatalf("failed to create group: %v", err)
	}
	spec := group.Spec()

	// 1. Request lock when queue is empty and no one has lock
	spec.RequestLock("job-1")
	queue := spec.GetWaitingJobQueue()
	if queue.Len() != 1 {
		t.Errorf("Expected queue len to be 1, got %d", queue.Len())
	}
	if !queue.Exists("job-1") {
		t.Errorf("Expected job-1 to be in queue")
	}

	// 2. Request lock for the same job again (should be no-op/no duplicate)
	spec.RequestLock("job-1")
	if queue.Len() != 1 {
		t.Errorf("Expected queue len to remain 1, got %d", queue.Len())
	}

	// 3. Request lock for a different job
	spec.RequestLock("job-2")
	if queue.Len() != 2 {
		t.Errorf("Expected queue len to be 2, got %d", queue.Len())
	}
	if !queue.Exists("job-2") {
		t.Errorf("Expected job-2 to be in queue")
	}

	// 4. Grant lock to job-1, and request lock for job-1 again.
	// Since it now has the lock, it should NOT be enqueued again.
	err = spec.Lock(ctx, "job-1")
	if err != nil {
		t.Fatalf("failed to lock: %v", err)
	}
	// Dequeue it from queue to simulate it being processed
	val, ok := queue.Dequeue()
	if !ok || val != "job-1" {
		t.Fatalf("failed to dequeue job-1")
	}

	// Now job-1 has lock, and is NOT in queue.
	// RequestLock for job-1 should do nothing (not enqueue it).
	spec.RequestLock("job-1")
	if queue.Exists("job-1") {
		t.Errorf("Expected job-1 NOT to be enqueued since it already holds the lock")
	}
}

func TestGroupSpec_Yield(t *testing.T) {
	ctx := context.Background()

	type step struct {
		op          string // "lock", "enqueue", "yield"
		jobID       string
		expectedErr error
		wantLocked  string
		wantQueue   []string
	}

	tests := []struct {
		name      string
		lockStore store.LockStore
		steps     []step
	}{
		{
			name:      "yield without lockStore",
			lockStore: nil,
			steps: []step{
				// Setup: job-1 locks, job-2 and job-3 enqueue
				{op: "lock", jobID: "job-1", expectedErr: nil, wantLocked: "job-1", wantQueue: []string{}},
				{op: "enqueue", jobID: "job-2", expectedErr: nil, wantLocked: "job-1", wantQueue: []string{"job-2"}},
				{op: "enqueue", jobID: "job-3", expectedErr: nil, wantLocked: "job-1", wantQueue: []string{"job-2", "job-3"}},
				// Yield job-1 -> job-2 should get lock, job-3 remains in queue
				{op: "yield", jobID: "job-1", expectedErr: nil, wantLocked: "job-2", wantQueue: []string{"job-3"}},
				// Yield job-2 -> job-3 should get lock, queue becomes empty
				{op: "yield", jobID: "job-2", expectedErr: nil, wantLocked: "job-3", wantQueue: []string{}},
				// Yield job-3 -> queue is empty, group becomes unlocked
				{op: "yield", jobID: "job-3", expectedErr: nil, wantLocked: "", wantQueue: []string{}},
			},
		},
		{
			name:      "yield with MemLockStore",
			lockStore: store.NewMemLockStore(),
			steps: []step{
				// Setup
				{op: "lock", jobID: "job-1", expectedErr: nil, wantLocked: "job-1", wantQueue: []string{}},
				{op: "enqueue", jobID: "job-2", expectedErr: nil, wantLocked: "job-1", wantQueue: []string{"job-2"}},
				// Yield by non-lock holder should fail and not change anything
				{op: "yield", jobID: "job-2", expectedErr: store.ErrNotLockHolder, wantLocked: "job-1", wantQueue: []string{"job-2"}},
				// Yield by lock holder succeeds
				{op: "yield", jobID: "job-1", expectedErr: nil, wantLocked: "job-2", wantQueue: []string{}},
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
			spec := group.Spec()

			for stepIdx, stepInfo := range tc.steps {
				var err error
				switch stepInfo.op {
				case "lock":
					err = spec.Lock(ctx, stepInfo.jobID)
				case "enqueue":
					spec.RequestLock(stepInfo.jobID)
				case "yield":
					err = spec.Yield(ctx, stepInfo.jobID)
				}

				if !errors.Is(err, stepInfo.expectedErr) {
					t.Fatalf("step %d: %s with %q returned error %v, want %v", stepIdx, stepInfo.op, stepInfo.jobID, err, stepInfo.expectedErr)
				}

				if spec.LockingJob() != stepInfo.wantLocked {
					t.Errorf("step %d: after %s, LockingJob() = %q, want %q", stepIdx, stepInfo.op, spec.LockingJob(), stepInfo.wantLocked)
				}

				queueList := spec.GetWaitingJobQueue().List()
				queueIDs := make([]string, 0, len(queueList))
				for _, qj := range queueList {
					queueIDs = append(queueIDs, qj.JobID)
				}
				if !reflect.DeepEqual(queueIDs, stepInfo.wantQueue) {
					t.Errorf("step %d: after %s, queue = %v, want %v", stepIdx, stepInfo.op, queueIDs, stepInfo.wantQueue)
				}
			}
		})
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
