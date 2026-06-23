package controller

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"runtime/debug"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/logging"
	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
)

const (
	operationPollInterval = 1 * time.Second
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
	agentStore        store.SnapshotAgentStore
}

// NewController creates a new Controller with the provided stores, queue, and infrastructure orchestrator.
func NewController(
	groupStore *store.GroupStore,
	jobStore *store.JobStore,
	queue WorkQueue,
	infraOrchestrator InfrastructureOrchestrator,
	agentStore store.SnapshotAgentStore,
) *Controller {
	return &Controller{
		queue:             queue,
		groupStore:        groupStore,
		jobStore:          jobStore,
		infraOrchestrator: infraOrchestrator,
		agentStore:        agentStore,
	}
}

// EnqueueWork enqueues the group ID for reconciliation.
func (c *Controller) EnqueueWork(groupID string) {
	c.queue.Add(groupID)
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

	if err := c.ObserveJobContext(ctx, groupID); err != nil {
		return fmt.Errorf("failed to observe job context: %w", err)
	}

	// TODO: deduce activejob for restart case when group is in STATE_IDLE_YIELDED

	// 2. Determine Desired State
	g, err := c.groupStore.Get(ctx, groupID)
	if err != nil {
		return fmt.Errorf("failed to get group %s from store: %w", groupID, err)
	}

	if _, err := g.Spec().TryPromote(ctx); err != nil {
		return fmt.Errorf("failed to promote next job: %w", err)
	}

	activeJob := g.Spec().ActiveJob()

	// 3. Act
	// TODO: add optional fan out parallism for node reconciliation
	for _, node := range g.Status().Nodes() {
		if err := c.reconcileNode(ctx, g.ID(), node, activeJob); err != nil {
			return fmt.Errorf("failed to reconcile node %s: %w", node, err)
		}
	}

	// 4. Update Status
	if err := c.updateGroupStatus(ctx, g); err != nil {
		return fmt.Errorf("failed to update group status: %w", err)
	}

	return nil
}

// reconcileNode reconciles the state of a single node for the active job.
func (c *Controller) reconcileNode(ctx context.Context, groupID, nodeName, activeJobID string) error {
	ctx = logging.WithNodeName(ctx, nodeName)
	jobs, err := c.jobStore.ListByGroup(ctx, groupID)
	if err != nil {
		return fmt.Errorf("failed to list jobs for group %s: %w", groupID, err)
	}

	agentJobStates := make(map[string]pb.SnapshotAgentJobState_State)
	for _, job := range jobs {
		state, ok := job.ContextState()[nodeName]
		if !ok {
			state = pb.SnapshotAgentJobState_STATE_UNSPECIFIED
		}
		agentJobStates[job.JobID()] = state
	}

	slog.DebugContext(ctx, "Reconciling node", "activeJobID", activeJobID, "agentJobStates", agentJobStates)

	// 1. Early exits for no reconciliation work needed/possible
	if activeJobID != "" {
		// Already running
		if state, ok := agentJobStates[activeJobID]; ok && state == pb.SnapshotAgentJobState_STATE_RUNNING {
			slog.InfoContext(ctx, "Active job is already running, exiting early", "activeJobID", activeJobID)
			return nil
		}

		// Faulted group
		if state, ok := agentJobStates[activeJobID]; ok && state == pb.SnapshotAgentJobState_STATE_FAULTED {
			return fmt.Errorf("active job %s is in FAULTED state on node %s, requires human intervention", activeJobID, nodeName)
		}
	}

	// 2. Wait out any existing transitions
	// TODO: crash recovery detect that an agent is still in the middle of transistioning. We will not have the operation
	// number but we can still wait till it leaves that state.

	// 3. Ensure no other jobs have their context loaded
	for jobID, state := range agentJobStates {
		if jobID == activeJobID || state != pb.SnapshotAgentJobState_STATE_RUNNING {
			continue
		}

		slog.InfoContext(ctx, "Triggering snapshot for job", "jobID", jobID, "state", state)
		resp, err := c.agentStore.Snapshot(ctx, nodeName, jobID, groupID)
		if err != nil {
			return fmt.Errorf("failed to trigger snapshot for job %s on node %s: %w", jobID, nodeName, err)
		}
		if err := c.waitForOperation(ctx, nodeName, resp.OperationId); err != nil {
			return fmt.Errorf("failed while waiting for snapshot operation %s for job %s on node %s: %w",
				resp.OperationId, jobID, nodeName, err)
		}
		if err := c.observeNodeJobContext(ctx, groupID, nodeName); err != nil {
			return fmt.Errorf("failed to refresh agent state after snapshot: %w", err)
		}
	}

	if activeJobID == "" {
		return nil
	}

	// 4. Ensure active job is loaded where there is available context
	state, ok := agentJobStates[activeJobID]
	if !ok || state != pb.SnapshotAgentJobState_STATE_SAVED {
		return nil
	}

	slog.InfoContext(ctx, "Triggering restore for active job", "jobID", activeJobID, "state", state)
	resp, err := c.agentStore.Restore(ctx, nodeName, activeJobID, groupID)
	if err != nil {
		return fmt.Errorf("failed to trigger restore for active job %s on node %s: %w",
			activeJobID, nodeName, err)
	}
	if err := c.waitForOperation(ctx, nodeName, resp.OperationId); err != nil {
		return fmt.Errorf("failed waiting for restore op %s for job %s on %s: %w",
			resp.OperationId, activeJobID, nodeName, err)
	}
	if err := c.observeNodeJobContext(ctx, groupID, nodeName); err != nil {
		return fmt.Errorf("failed to refresh agent state after restore: %w", err)
	}

	return nil
}

// isJobLoaded checks if a specific job is currently loaded on the nodes of the group.
// A job J is considered loaded if, for every node N in the group, the job's state on N
// is either STATE_RUNNING, or STATE_UNSPECIFIED and no other job is running on N.
// If multiple jobs are running on the same node, it returns an error (impossible state).
func (c *Controller) isJobLoaded(ctx context.Context, group *store.Group, jobID string) (bool, error) {
	if jobID == "" {
		return false, nil
	}
	jobs, err := c.jobStore.ListByGroup(ctx, group.ID())
	if err != nil {
		return false, fmt.Errorf("failed to list jobs for group %s: %w", group.ID(), err)
	}

	nodes := group.Status().Nodes()
	if len(nodes) == 0 {
		return false, nil
	}

	// Map of node -> jobID of the job running on it.
	// If multiple jobs are running on the same node, we error out.
	nodeRunningJob := make(map[string]string)
	for _, job := range jobs {
		for node, state := range job.ContextState() {
			if state == pb.SnapshotAgentJobState_STATE_RUNNING {
				if current, ok := nodeRunningJob[node]; ok && current != job.JobID() {
					return false, fmt.Errorf("impossible state: multiple jobs running on node %s: %s and %s", node, current, job.JobID())
				}
				nodeRunningJob[node] = job.JobID()
			}
		}
	}

	targetJob, err := c.jobStore.Get(ctx, group.ID(), jobID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			// To support RL init where the lock is needed before deploying pods,
			// assume jobs that do not exist (no pods) are loaded
			// provided no other job is running. We represent this
			// by using a dummy job with empty context state (unspecified).
			targetJob = store.NewJob(group.ID(), jobID)
		} else {
			return false, fmt.Errorf("failed to get job %s: %w", jobID, err)
		}
	}

	contextState := targetJob.ContextState()
	for _, node := range nodes {
		state, ok := contextState[node]
		if !ok {
			state = pb.SnapshotAgentJobState_STATE_UNSPECIFIED
		}

		switch state {
		case pb.SnapshotAgentJobState_STATE_RUNNING:
			// Safe, checked for conflicts already
		case pb.SnapshotAgentJobState_STATE_UNSPECIFIED:
			if nodeRunningJob[node] != "" {
				// Another job is running on this node
				return false, nil
			}
		default:
			// STATE_SAVED, STATE_FAULTED, etc.
			return false, nil
		}
	}

	return true, nil
}

// determineGroupState deduces the high-level group state based on locking, active, and loaded job IDs.
func determineGroupState(lockingJobID, activeJobID, loadedJobID string) pb.GroupStatus_State {
	activeJobLoaded := (activeJobID != "" && activeJobID == loadedJobID)

	if lockingJobID != "" {
		if activeJobLoaded {
			return pb.GroupStatus_STATE_LOCKED
		}
		return pb.GroupStatus_STATE_SWITCHING
	}

	// Unlocked
	if activeJobID != "" {
		return pb.GroupStatus_STATE_IDLE_YIELDED
	}

	return pb.GroupStatus_STATE_IDLE
}

// updateGroupStatus deduces the group status based on the current state and updates it in the store.
func (c *Controller) updateGroupStatus(ctx context.Context, group *store.Group) error {
	activeJobID := group.Spec().ActiveJob()
	activeJobLoaded := false
	if activeJobID != "" {
		var err error
		activeJobLoaded, err = c.isJobLoaded(ctx, group, activeJobID)
		if err != nil {
			return fmt.Errorf("failed to check if active job %s is loaded: %w", activeJobID, err)
		}
	}

	if activeJobLoaded {
		group.Status().SetLoadedJob(activeJobID)
	} else {
		group.Status().SetLoadedJob("")
	}

	lockingJobID := group.Spec().LockingJob()
	loadedJobID := group.Status().LoadedJob()

	state := determineGroupState(lockingJobID, activeJobID, loadedJobID)
	group.Status().SetState(state)

	return nil
}

// ObserveJobContext queries snapshot agents for all nodes in the group and updates job context states.
func (c *Controller) ObserveJobContext(ctx context.Context, groupID string) error {
	g, err := c.groupStore.Get(ctx, groupID)
	if err != nil {
		return fmt.Errorf("failed to get group %s from store: %w", groupID, err)
	}
	groupNodes := g.Status().Nodes()

	for _, nodeName := range groupNodes {
		if err := c.observeNodeJobContext(ctx, groupID, nodeName); err != nil {
			return err
		}
	}
	return nil
}

// observeNodeJobContext queries the snapshot agent for a single node and updates job context states in the store.
// It logs and ignores agent communication errors (non-fatal), but returns store errors (fatal).
func (c *Controller) observeNodeJobContext(ctx context.Context, groupID, nodeName string) error {
	resp, err := c.agentStore.GetStatus(ctx, nodeName)
	if err != nil {
		slog.ErrorContext(ctx, "Failed to get status from snapshot agent", "error", err, "node", nodeName)
		return nil
	}

	for _, js := range resp.JobStatuses {
		// Only update if the job is known in this group
		_, err := c.jobStore.Get(ctx, groupID, js.JobId)
		if errors.Is(err, store.ErrNotFound) {
			continue
		} else if err != nil {
			return fmt.Errorf("failed to get job %s from store: %w", js.JobId, err)
		}

		state := translateJobState(js.State)
		if err := c.jobStore.UpdateContextState(ctx, groupID, js.JobId, nodeName, state); err != nil {
			return fmt.Errorf("failed to update job context state for job %s on node %s: %w", js.JobId, nodeName, err)
		}
		slog.DebugContext(ctx, "Updated job context state", "job", js.JobId, "node", nodeName, "state", state)
	}
	return nil
}

func translateJobState(s agentpb.JobState) pb.SnapshotAgentJobState_State {
	switch s {
	case agentpb.JobState_JOB_STATE_IDLE:
		return pb.SnapshotAgentJobState_STATE_IDLE
	case agentpb.JobState_JOB_STATE_RUNNING:
		return pb.SnapshotAgentJobState_STATE_RUNNING
	case agentpb.JobState_JOB_STATE_TRANSITIONING:
		return pb.SnapshotAgentJobState_STATE_TRANSITIONING
	case agentpb.JobState_JOB_STATE_SAVED:
		return pb.SnapshotAgentJobState_STATE_SAVED
	case agentpb.JobState_JOB_STATE_FAULTED:
		return pb.SnapshotAgentJobState_STATE_FAULTED
	default:
		return pb.SnapshotAgentJobState_STATE_UNSPECIFIED
	}
}

// waitForOperation blocks until the given operation on the node completes or fails.
func (c *Controller) waitForOperation(ctx context.Context, nodeName, operationID string) error {
	ctx = logging.WithNodeName(ctx, nodeName)
	ctx = logging.WithOperationID(ctx, operationID)

	ticker := time.NewTicker(operationPollInterval)
	defer ticker.Stop()

	slog.InfoContext(ctx, "Waiting for agent operation to complete")

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("context cancelled while waiting for operation %s: %w", operationID, ctx.Err())
		case <-ticker.C:
			resp, err := c.agentStore.GetOperation(ctx, nodeName, operationID)
			if err != nil {
				slog.WarnContext(ctx, "Failed to get operation status, will retry", "error", err)
				continue
			}

			switch resp.Status {
			case agentpb.OperationStatus_OPERATION_STATUS_COMPLETE:
				slog.InfoContext(ctx, "Operation completed successfully", "elapsedMs", resp.ElapsedMs)
				return nil
			case agentpb.OperationStatus_OPERATION_STATUS_FAILED:
				errStr := "unknown error"
				if resp.Error != nil {
					errStr = *resp.Error
				}
				return fmt.Errorf("operation %s failed: %s", operationID, errStr)
			case agentpb.OperationStatus_OPERATION_STATUS_PENDING:
				slog.DebugContext(ctx, "Operation still pending", "elapsedMs", resp.ElapsedMs)
			default:
				slog.WarnContext(ctx, "Unknown operation status", "status", resp.Status)
			}
		}
	}
}
