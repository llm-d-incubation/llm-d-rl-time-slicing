package controller_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/controller"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"k8s.io/client-go/util/workqueue"
)

type mockInfrastructureOrchestrator struct {
	initFunc    func(ctx context.Context) error
	observeFunc func(ctx context.Context, groupID string) error
}

func (m *mockInfrastructureOrchestrator) Init(ctx context.Context) error {
	if m.initFunc != nil {
		return m.initFunc(ctx)
	}
	return nil
}

func (m *mockInfrastructureOrchestrator) ObserveGroupState(ctx context.Context, groupID string) error {
	if m.observeFunc != nil {
		return m.observeFunc(ctx, groupID)
	}
	return nil
}

type mockSnapshotAgentStore struct {
	getStatusFunc func(ctx context.Context, nodeName string) (*agentpb.StatusResponse, error)
	snapshotFunc  func(ctx context.Context, nodeName, jobID, groupID string) (*agentpb.SnapshotResponse, error)
}

func (m *mockSnapshotAgentStore) GetStatus(ctx context.Context, nodeName string) (*agentpb.StatusResponse, error) {
	if m.getStatusFunc != nil {
		return m.getStatusFunc(ctx, nodeName)
	}
	return &agentpb.StatusResponse{}, nil
}

func (m *mockSnapshotAgentStore) CloseClient(nodeName string) error {
	return nil
}

func (m *mockSnapshotAgentStore) Snapshot(
	ctx context.Context, nodeName, jobID, groupID string,
) (*agentpb.SnapshotResponse, error) {
	if m.snapshotFunc != nil {
		return m.snapshotFunc(ctx, nodeName, jobID, groupID)
	}
	return &agentpb.SnapshotResponse{}, nil
}

// TestController_ReconcileSuccess verifies that the controller calls ObserveGroupState
// and successfully processes the item.
func TestController_ReconcileSuccess(t *testing.T) {
	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()
	queue := workqueue.NewTypedRateLimitingQueueWithConfig(
		workqueue.DefaultTypedControllerRateLimiter[string](),
		workqueue.TypedRateLimitingQueueConfig[string]{Name: "test"},
	)

	observeCalled := make(chan string, 1)
	mockOrch := &mockInfrastructureOrchestrator{
		observeFunc: func(ctx context.Context, groupID string) error {
			_, _, err := groupStore.GetOrCreate(ctx, groupID)
			if err != nil {
				return err
			}
			observeCalled <- groupID
			return nil
		},
	}

	mockAgentStore := &mockSnapshotAgentStore{}
	c := controller.NewController(groupStore, jobStore, queue, mockOrch, mockAgentStore)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start the controller
	go func() {
		if err := c.Run(ctx, 1); err != nil {
			t.Errorf("Controller Run failed: %v", err)
		}
	}()

	// Add an item to the queue
	queue.Add("group-1")

	// Wait for ObserveGroupState to be called
	select {
	case groupID := <-observeCalled:
		if groupID != "group-1" {
			t.Errorf("Expected ObserveGroupState to be called for group-1, got %s", groupID)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Timed out waiting for ObserveGroupState to be called")
	}

	// Verify the item was marked Done (queue length should become 0)
	time.Sleep(100 * time.Millisecond)
	if queue.Len() != 0 {
		t.Errorf("Expected queue to be empty, got length %d", queue.Len())
	}
}

// TestController_ReconcileFailure_Retries verifies that if ObserveGroupState fails,
// the controller retries by adding the item back to the queue (rate limited).
func TestController_ReconcileFailure_Retries(t *testing.T) {
	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()

	testQueue := &trackQueue{
		TypedRateLimitingInterface: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "test"},
		),
	}

	observeCalled := make(chan struct{})
	mockOrch := &mockInfrastructureOrchestrator{
		observeFunc: func(ctx context.Context, groupID string) error {
			close(observeCalled)
			return errors.New("observe failed")
		},
	}

	mockAgentStore := &mockSnapshotAgentStore{}
	c := controller.NewController(groupStore, jobStore, testQueue, mockOrch, mockAgentStore)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		if err := c.Run(ctx, 1); err != nil {
			t.Errorf("Controller Run failed: %v", err)
		}
	}()

	testQueue.Add("group-1")

	<-observeCalled

	// Wait for the item to be re-added
	err := waitWithTimeout(func() bool {
		return testQueue.getAddRateLimitedCount() > 0
	}, 2*time.Second)
	if err != nil {
		t.Fatal("Timed out waiting for item to be re-queued")
	}

	if testQueue.getAddRateLimitedCount() != 1 {
		t.Errorf("Expected AddRateLimited to be called once, got %d", testQueue.getAddRateLimitedCount())
	}
}

type trackQueue struct {
	workqueue.TypedRateLimitingInterface[string]
	mu                  sync.Mutex
	addRateLimitedCount int
}

func (t *trackQueue) AddRateLimited(item string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.addRateLimitedCount++
	t.TypedRateLimitingInterface.AddRateLimited(item)
}

func (t *trackQueue) getAddRateLimitedCount() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.addRateLimitedCount
}

func waitWithTimeout(f func() bool, timeout time.Duration) error {
	ch := make(chan struct{})
	go func() {
		for {
			if f() {
				close(ch)
				return
			}
			time.Sleep(50 * time.Millisecond)
		}
	}()
	select {
	case <-ch:
		return nil
	case <-time.After(timeout):
		return errors.New("timeout")
	}
}

func TestController_Reconcile_TwoJobsTakeLockTurns(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()
	queue := workqueue.NewTypedRateLimitingQueueWithConfig(
		workqueue.DefaultTypedControllerRateLimiter[string](),
		workqueue.TypedRateLimitingQueueConfig[string]{Name: "test"},
	)

	groupID := "group-1"

	// Mock ObserveGroupState (no-op, we don't need to sync lock state because
	// in-memory and lockStore are kept in sync by GroupSpec methods).
	mockOrch := &mockInfrastructureOrchestrator{
		observeFunc: func(ctx context.Context, gID string) error {
			return nil
		},
	}

	mockAgentStore := &mockSnapshotAgentStore{}
	c := controller.NewController(groupStore, jobStore, queue, mockOrch, mockAgentStore)

	// Start the controller
	go func() {
		if err := c.Run(ctx, 1); err != nil {
			t.Errorf("Controller Run failed: %v", err)
		}
	}()

	// 1. Pre-lock the group to "job-1" in lockStore, then create it.
	// This simulates starting with a locked group.
	if err := lockStore.Lock(ctx, groupID, "job-1"); err != nil {
		t.Fatalf("failed to lock in store: %v", err)
	}
	testGroup, _, err := groupStore.GetOrCreate(ctx, groupID)
	if err != nil {
		t.Fatalf("failed to create group: %v", err)
	}

	// Reconcile Phase 1 (job-1 locked)
	queue.Add(groupID)
	err = waitWithTimeout(func() bool { return queue.Len() == 0 }, 2*time.Second)
	if err != nil {
		t.Fatalf("Timed out waiting for Phase 1 reconcile: %v", err)
	}
	if testGroup.Spec().LockingJob() != "job-1" || testGroup.Spec().ActiveJob() != "job-1" {
		t.Errorf("Phase 1: expected lockingJob=job-1, activeJob=job-1; got lockingJob=%q, activeJob=%q",
			testGroup.Spec().LockingJob(), testGroup.Spec().ActiveJob())
	}

	// 2. job-2 requests lock and gets in queue
	testGroup.Spec().RequestLock("job-2")

	// Reconcile Phase 2 (job-2 enqueued, job-1 still locked)
	queue.Add(groupID)
	err = waitWithTimeout(func() bool { return queue.Len() == 0 }, 2*time.Second)
	if err != nil {
		t.Fatalf("Timed out waiting for Phase 2 reconcile: %v", err)
	}
	if testGroup.Spec().LockingJob() != "job-1" || testGroup.Spec().ActiveJob() != "job-1" {
		t.Errorf("Phase 2: expected lockingJob=job-1, activeJob=job-1; got lockingJob=%q, activeJob=%q",
			testGroup.Spec().LockingJob(), testGroup.Spec().ActiveJob())
	}
	if !testGroup.Spec().GetWaitingJobQueue().Exists("job-2") {
		t.Errorf("Phase 2: expected job-2 to be in queue")
	}

	// 3. job-1 yields the lock -> job-2 should get the lock
	err = testGroup.Spec().Yield(ctx, "job-1")
	if err != nil {
		t.Fatalf("Yield failed: %v", err)
	}

	// Reconcile Phase 3 (job-2 should be active/locking)
	queue.Add(groupID)
	err = waitWithTimeout(func() bool {
		return testGroup.Spec().LockingJob() == "job-2" && testGroup.Spec().ActiveJob() == "job-2"
	}, 3*time.Second)
	if err != nil {
		t.Fatalf("Timed out waiting for Phase 3 reconcile "+
			"(expected lockingJob=job-2, activeJob=job-2): %v. "+
			"Current state: lockingJob=%q, activeJob=%q",
			err, testGroup.Spec().LockingJob(), testGroup.Spec().ActiveJob())
	}
	if testGroup.Spec().GetWaitingJobQueue().Exists("job-2") {
		t.Errorf("Phase 3: expected job-2 to be dequeued")
	}

	// 4. job-1 requests lock again (gets in queue)
	testGroup.Spec().RequestLock("job-1")

	// Reconcile Phase 4 (job-1 enqueued, job-2 still locked)
	queue.Add(groupID)
	err = waitWithTimeout(func() bool { return queue.Len() == 0 }, 2*time.Second)
	if err != nil {
		t.Fatalf("Timed out waiting for Phase 4 reconcile: %v", err)
	}
	if testGroup.Spec().LockingJob() != "job-2" || testGroup.Spec().ActiveJob() != "job-2" {
		t.Errorf("Phase 4: expected lockingJob=job-2, activeJob=job-2; got lockingJob=%q, activeJob=%q",
			testGroup.Spec().LockingJob(), testGroup.Spec().ActiveJob())
	}
	if !testGroup.Spec().GetWaitingJobQueue().Exists("job-1") {
		t.Errorf("Phase 4: expected job-1 to be in queue")
	}

	// 5. job-2 yields the lock -> job-1 should get the lock again
	err = testGroup.Spec().Yield(ctx, "job-2")
	if err != nil {
		t.Fatalf("Yield failed: %v", err)
	}

	// Reconcile Phase 5 (job-1 should be active/locking again)
	queue.Add(groupID)
	err = waitWithTimeout(func() bool {
		return testGroup.Spec().LockingJob() == "job-1" && testGroup.Spec().ActiveJob() == "job-1"
	}, 3*time.Second)
	if err != nil {
		t.Fatalf("Timed out waiting for Phase 5 reconcile "+
			"(expected lockingJob=job-1, activeJob=job-1): %v. "+
			"Current state: lockingJob=%q, activeJob=%q",
			err, testGroup.Spec().LockingJob(), testGroup.Spec().ActiveJob())
	}
	if testGroup.Spec().GetWaitingJobQueue().Exists("job-1") {
		t.Errorf("Phase 5: expected job-1 to be dequeued")
	}
}

func TestController_Reconcile_OneJobLoopRemainsActive(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()
	queue := workqueue.NewTypedRateLimitingQueueWithConfig(
		workqueue.DefaultTypedControllerRateLimiter[string](),
		workqueue.TypedRateLimitingQueueConfig[string]{Name: "test"},
	)

	groupID := "group-1"

	// Mock ObserveGroupState to notify when reconcile runs.
	observeCalled := make(chan struct{}, 10)
	mockOrch := &mockInfrastructureOrchestrator{
		observeFunc: func(ctx context.Context, gID string) error {
			observeCalled <- struct{}{}
			return nil
		},
	}

	mockAgentStore := &mockSnapshotAgentStore{}
	c := controller.NewController(groupStore, jobStore, queue, mockOrch, mockAgentStore)

	// Start the controller
	go func() {
		if err := c.Run(ctx, 1); err != nil {
			t.Errorf("Controller Run failed: %v", err)
		}
	}()

	// 1. Pre-lock the group to "job-1" in lockStore, then create it.
	if err := lockStore.Lock(ctx, groupID, "job-1"); err != nil {
		t.Fatalf("failed to lock job-1: %v", err)
	}
	testGroup, _, err := groupStore.GetOrCreate(ctx, groupID)
	if err != nil {
		t.Fatalf("failed to create group: %v", err)
	}

	// Reconcile Lock
	queue.Add(groupID)
	select {
	case <-observeCalled:
	case <-time.After(2 * time.Second):
		t.Fatalf("Timed out waiting for Lock reconcile")
	}

	// Verify Locked State
	if testGroup.Spec().LockingJob() != "job-1" || testGroup.Spec().ActiveJob() != "job-1" {
		t.Errorf("Expected lockingJob=job-1, activeJob=job-1; got lockingJob=%q, activeJob=%q",
			testGroup.Spec().LockingJob(), testGroup.Spec().ActiveJob())
	}

	// 2. Yield job-1 (no waiters, so it just unlocks)
	err = testGroup.Spec().Yield(ctx, "job-1")
	if err != nil {
		t.Fatalf("Yield failed: %v", err)
	}

	// Reconcile Yield
	queue.Add(groupID)
	select {
	case <-observeCalled:
	case <-time.After(2 * time.Second):
		t.Fatalf("Timed out waiting for Yield reconcile")
	}

	// Verify Yielded State: lockingJob is "", but activeJob REMAINS "job-1" (warm!)
	if testGroup.Spec().LockingJob() != "" {
		t.Errorf("Expected lockingJob to be empty, got %q", testGroup.Spec().LockingJob())
	}
	if testGroup.Spec().ActiveJob() != "job-1" {
		t.Errorf("Expected activeJob to remain job-1, got %q", testGroup.Spec().ActiveJob())
	}
}

func TestController_Reconcile_TriggersSnapshot(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()
	queue := workqueue.NewTypedRateLimitingQueueWithConfig(
		workqueue.DefaultTypedControllerRateLimiter[string](),
		workqueue.TypedRateLimitingQueueConfig[string]{Name: "test"},
	)

	groupID := "group-1"
	nodeName := "node-1"

	// 1. Setup group and nodes
	testGroup, _, err := groupStore.GetOrCreate(ctx, groupID)
	if err != nil {
		t.Fatalf("failed to create group: %v", err)
	}
	testGroup.Status().SetNodes([]string{nodeName})
	// Set active job to job-1
	testGroup.Spec().SetActiveJob("job-1")

	// 2. Setup jobs in store
	job1 := store.NewJob(groupID, "job-1")
	job2 := store.NewJob(groupID, "job-2")
	// job-2 is RUNNING on node-1, which is NOT the active job (job-1)
	job2.UpdateContextState(nodeName, pb.SnapshotAgentJobState_STATE_RUNNING)

	if err := jobStore.Put(ctx, job1); err != nil {
		t.Fatalf("failed to put job1: %v", err)
	}
	if err := jobStore.Put(ctx, job2); err != nil {
		t.Fatalf("failed to put job2: %v", err)
	}

	// 3. Mock SnapshotAgentStore to track calls
	snapshotCalled := make(chan string, 1)
	mockAgentStore := &mockSnapshotAgentStore{
		snapshotFunc: func(ctx context.Context, node, jobID, gID string) (*agentpb.SnapshotResponse, error) {
			if node == nodeName && jobID == "job-2" && gID == groupID {
				snapshotCalled <- jobID
			}
			return &agentpb.SnapshotResponse{OperationId: "op-123"}, nil
		},
	}

	mockOrch := &mockInfrastructureOrchestrator{
		observeFunc: func(ctx context.Context, gID string) error {
			// Do nothing, we already set up the state
			return nil
		},
	}

	c := controller.NewController(groupStore, jobStore, queue, mockOrch, mockAgentStore)

	// Start the controller
	go func() {
		if err := c.Run(ctx, 1); err != nil {
			t.Errorf("Controller Run failed: %v", err)
		}
	}()

	// Trigger reconcile
	queue.Add(groupID)

	// Verify Snapshot was called
	select {
	case jobID := <-snapshotCalled:
		if jobID != "job-2" {
			t.Errorf("Expected snapshot to be called for job-2, got %s", jobID)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Timed out waiting for Snapshot to be called")
	}
}

func TestController_Reconcile_ActiveJobFaultedFails(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()
	testQueue := &trackQueue{
		TypedRateLimitingInterface: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "test"},
		),
	}

	groupID := "group-1"
	nodeName := "node-1"

	// 1. Setup group and nodes
	testGroup, _, err := groupStore.GetOrCreate(ctx, groupID)
	if err != nil {
		t.Fatalf("failed to create group: %v", err)
	}
	testGroup.Status().SetNodes([]string{nodeName})
	testGroup.Spec().SetActiveJob("job-1")

	// 2. Setup jobs in store
	job1 := store.NewJob(groupID, "job-1")
	// job-1 (active) is FAULTED on node-1
	job1.UpdateContextState(nodeName, pb.SnapshotAgentJobState_STATE_FAULTED)

	if err := jobStore.Put(ctx, job1); err != nil {
		t.Fatalf("failed to put job1: %v", err)
	}

	mockAgentStore := &mockSnapshotAgentStore{}
	mockOrch := &mockInfrastructureOrchestrator{
		observeFunc: func(ctx context.Context, gID string) error {
			return nil
		},
	}

	c := controller.NewController(groupStore, jobStore, testQueue, mockOrch, mockAgentStore)

	// Start the controller
	go func() {
		if err := c.Run(ctx, 1); err != nil {
			t.Errorf("Controller Run failed: %v", err)
		}
	}()

	// Trigger reconcile
	testQueue.Add(groupID)

	// Verify that it failed and was re-queued (AddRateLimited called)
	err = waitWithTimeout(func() bool {
		return testQueue.getAddRateLimitedCount() > 0
	}, 2*time.Second)
	if err != nil {
		t.Fatal("Timed out waiting for item to be re-queued (reconciliation should have failed)")
	}

	if testQueue.getAddRateLimitedCount() < 1 {
		t.Errorf("Expected AddRateLimited to be called at least once, got %d", testQueue.getAddRateLimitedCount())
	}
}

func TestController_ObserveJobContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()

	groupID := "group-1"
	nodeName := "node-1"

	// 1. Setup group and nodes in store
	g, _, err := groupStore.GetOrCreate(ctx, groupID)
	if err != nil {
		t.Fatalf("failed to create group: %v", err)
	}
	g.Status().SetNodes([]string{nodeName})

	// 2. Setup job in store (must exist for context to be updated)
	job := store.NewJob(groupID, "job-1")
	if err := jobStore.Put(ctx, job); err != nil {
		t.Fatalf("failed to put job: %v", err)
	}

	// 3. Mock SnapshotAgentStore to return status
	mockAgentStore := &mockSnapshotAgentStore{
		getStatusFunc: func(ctx context.Context, node string) (*agentpb.StatusResponse, error) {
			if node == nodeName {
				return &agentpb.StatusResponse{
					JobStatuses: []*agentpb.JobStatus{
						{JobId: "job-1", State: agentpb.JobState_JOB_STATE_RUNNING},
					},
				}, nil
			}
			return &agentpb.StatusResponse{}, nil
		},
	}

	c := controller.NewController(groupStore, jobStore, nil, nil, mockAgentStore)

	// 4. Call ObserveJobContext
	err = c.ObserveJobContext(ctx, groupID)
	if err != nil {
		t.Fatalf("ObserveJobContext failed: %v", err)
	}

	// 5. Verify job context state is updated
	updatedJob, err := jobStore.Get(ctx, groupID, "job-1")
	if err != nil {
		t.Fatalf("failed to get job: %v", err)
	}
	state, ok := updatedJob.ContextState()[nodeName]
	if !ok {
		t.Fatalf("Expected context state for job-1 on node-1 to exist")
	}
	if state != pb.SnapshotAgentJobState_STATE_RUNNING {
		t.Errorf("Expected job-1 state to be RUNNING, got %v", state)
	}
}
