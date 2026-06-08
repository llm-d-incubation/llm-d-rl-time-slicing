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
