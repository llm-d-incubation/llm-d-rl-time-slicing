package controller

import (
	"context"
	"fmt"
	"log/slog"
	"runtime/debug"
	"time"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/logging"
)

// handleCrash is a helper that recovers from panics, logs the panic and stack trace.
// It is intended to be used in `defer` statements in goroutines.
func handleCrash(ctx context.Context) {
	if r := recover(); r != nil {
		slog.ErrorContext(ctx, "Observed a panic", "panic", r, "stack", string(debug.Stack()))
	}
}

// until runs the provided function repeatedly with a period sleep between runs.
// It stops when the context is cancelled. It recovers from panics in the function
// using handleCrash to ensure the worker loop continues running.
func until(ctx context.Context, f func(context.Context), period time.Duration) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		func() {
			defer handleCrash(ctx)
			f(ctx)
		}()

		timer := time.NewTimer(period)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
	}
}

// WorkQueue defines the interface for the work queue.
// It contains only the methods used by the controller.
type WorkQueue interface {
	// Add enqueues the group ID for reconciliation.
	Add(groupID string)
	// AddRateLimited enqueues the group ID using a rate limiter.
	// This is typically used to requeue a group ID after a reconciliation failure.
	AddRateLimited(groupID string)
	// Forget resets the rate limit tracking for the group ID,
	// usually called after a successful reconciliation.
	Forget(groupID string)
	// Done signals that the reconciliation cycle for this group ID is complete.
	// This must be called for every item retrieved from Get() to unlock it for future processing.
	Done(groupID string)
	// Get retrieves the next group ID to process. It blocks until an item is available.
	// If the queue is shut down, it returns shutdown=true.
	Get() (groupID string, shutdown bool)
	// ShutDown shuts down the queue, preventing new items from being added and
	// notifying all blocked readers.
	ShutDown()
}

// InfrastructureOrchestrator defines the interface for interacting with the underlying infrastructure.
type InfrastructureOrchestrator interface {
	// Init initializes the infrastructure orchestrator.
	// It should block until the orchestrator is ready or return an error.
	Init(ctx context.Context) error
	// ObserveGroupState observes the current state of the infrastructure for the given group
	// and updates the groupStore and jobStore accordingly.
	ObserveGroupState(ctx context.Context, groupID string) error
}

// Controller coordinates the reconciliation loop for slice groups.
// It listens for changes in the infrastructure (via WorkQueue), observes the current state,
// determines the desired state, and takes actions to align the actual state with the desired state.
type Controller struct {
	queue             WorkQueue
	groupStore        *store.GroupStore
	jobStore          *store.JobStore
	infraOrchestrator InfrastructureOrchestrator
}

// NewController creates a new Controller with the provided stores, queue, and infrastructure orchestrator.
func NewController(
	groupStore *store.GroupStore,
	jobStore *store.JobStore,
	queue WorkQueue,
	infraOrchestrator InfrastructureOrchestrator,
) *Controller {
	return &Controller{
		queue:             queue,
		groupStore:        groupStore,
		jobStore:          jobStore,
		infraOrchestrator: infraOrchestrator,
	}
}

// Run starts the controller's reconciliation loop.
// It initializes the infrastructure orchestrator, then starts the specified number of worker goroutines.
// It blocks until the provided context is cancelled.
func (c *Controller) Run(ctx context.Context, workers int) error {
	defer handleCrash(ctx)
	defer c.queue.ShutDown()

	slog.InfoContext(ctx, "Starting Group controller")

	slog.InfoContext(ctx, "Initializing infrastructure")
	if err := c.infraOrchestrator.Init(ctx); err != nil {
		return fmt.Errorf("failed to initialize infrastructure: %w", err)
	}

	slog.InfoContext(ctx, "Starting workers")
	for i := 0; i < workers; i++ {
		workerID := i
		go until(ctx, func(ctx context.Context) {
			c.runWorker(ctx, workerID)
		}, time.Second)
	}

	slog.InfoContext(ctx, "Started workers")
	<-ctx.Done()
	slog.InfoContext(ctx, "Shutting down workers")

	return nil
}

// runWorker is the entry point for a worker goroutine.
// It continuously processes work items from the queue until the queue is shut down.
func (c *Controller) runWorker(ctx context.Context, workerID int) {
	ctx = logging.WithWorkerID(ctx, workerID)
	for c.processNextWorkItem(ctx) {
	}
}

// processNextWorkItem retrieves and processes a single work item (group ID) from the queue.
// It returns false if the queue is shut down, signaling the worker to exit.
// It wraps the reconciliation of a single group with error handling, rate limiting, and queue management.
func (c *Controller) processNextWorkItem(ctx context.Context) bool {
	groupID, shutdown := c.queue.Get()
	if shutdown {
		return false
	}

	err := func(groupID string) error {
		defer c.queue.Done(groupID)

		cycleCtx := logging.WithGroupID(ctx, groupID)

		if err := c.reconcileGroup(cycleCtx, groupID); err != nil {
			c.queue.AddRateLimited(groupID)
			return fmt.Errorf("error syncing '%s': %s, requeuing", groupID, err.Error())
		}
		c.queue.Forget(groupID)
		slog.InfoContext(cycleCtx, "Successfully synced group")
		return nil
	}(groupID)
	if err != nil {
		slog.ErrorContext(ctx, "Error processing work item", "error", err)
		return true
	}

	return true
}

// reconcileGroup performs the actual reconciliation for a single group.
// It observes the current state of the group from the infrastructure and updates the stores.
// Expects to be the only thread reconciling that particular group at any time.
func (c *Controller) reconcileGroup(ctx context.Context, groupID string) error {
	slog.InfoContext(ctx, "Reconciling group")

	// 1. Observe Current State and update stores
	if err := c.infraOrchestrator.ObserveGroupState(ctx, groupID); err != nil {
		return fmt.Errorf("failed to observe group state: %w", err)
	}

	// 2. Determine Desired State (TODO)

	// 3. Act (TODO)

	// 4. Update Status (TODO)

	return nil
}
