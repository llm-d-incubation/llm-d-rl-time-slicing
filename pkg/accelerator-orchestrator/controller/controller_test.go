package controller_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/controller"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
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

	c := controller.NewController(groupStore, jobStore, queue, mockOrch)

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

	c := controller.NewController(groupStore, jobStore, testQueue, mockOrch)

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

	c := controller.NewController(groupStore, jobStore, queue, mockOrch)

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

	c := controller.NewController(groupStore, jobStore, queue, mockOrch)

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
