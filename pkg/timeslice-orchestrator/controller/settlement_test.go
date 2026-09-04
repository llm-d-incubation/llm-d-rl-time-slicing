package controller_test

import (
	"context"
	"testing"
	"time"

	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/timeslice-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/timeslice-orchestrator/controller"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/timeslice-orchestrator/store"
	"k8s.io/client-go/util/workqueue"
)

// settlementFixture builds the scenario behind the grant-settlement gate:
// job-b was granted the samplers lock, deployed its pods (context IDLE) and
// yielded before its engine was observed on the accelerator; job-a is SAVED
// and queued for promotion. Promoting now would restore job-a onto a device
// job-b's engine is about to occupy.
func settlementFixture(t *testing.T, ctx context.Context, agentStore *controller.MockSnapshotAgentStore) (
	*controller.Controller, *store.Group, *store.Job, *trackQueue,
) {
	t.Helper()

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
	group, _, err := groupStore.GetOrCreate(ctx, groupID)
	if err != nil {
		t.Fatalf("failed to create group: %v", err)
	}
	group.Status().SetNodes([]string{"node-1"})

	// job-b: granted earlier (active), lock since yielded, engine never observed.
	group.Spec().SetActiveJob("job-b")
	jobB := store.NewJob(groupID, "job-b")
	jobB.UpdateContextState("node-1", pb.SnapshotAgentJobState_STATE_IDLE)
	if err := jobStore.Put(ctx, jobB); err != nil {
		t.Fatalf("failed to put job-b: %v", err)
	}

	// job-a: checkpointed off-device, waiting for promotion.
	jobA := store.NewJob(groupID, "job-a")
	jobA.UpdateContextState("node-1", pb.SnapshotAgentJobState_STATE_SAVED)
	if err := jobStore.Put(ctx, jobA); err != nil {
		t.Fatalf("failed to put job-a: %v", err)
	}
	group.Spec().RequestLock("job-a")

	mockOrch := &mockInfrastructureOrchestrator{
		observeFunc: func(ctx context.Context, gID string) error { return nil },
	}
	c := controller.NewController(groupStore, jobStore, testQueue, mockOrch, agentStore)
	return c, group, jobB, testQueue
}

// TestController_Reconcile_SettlementHoldsPromotion: while job-b's grant is
// unconsumed, job-a must not be promoted and no snapshot/restore may be
// issued — the reconcile requeues instead.
func TestController_Reconcile_SettlementHoldsPromotion(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	agentStore := &controller.MockSnapshotAgentStore{
		SnapshotFunc: func(ctx context.Context, nodeName, jobID, groupID string) (*agentpb.SnapshotResponse, error) {
			t.Errorf("unexpected Snapshot(%s) while promotion is held", jobID)
			return &agentpb.SnapshotResponse{}, nil
		},
		RestoreFunc: func(ctx context.Context, nodeName, jobID, groupID string) (*agentpb.RestoreResponse, error) {
			t.Errorf("unexpected Restore(%s) while promotion is held", jobID)
			return &agentpb.RestoreResponse{}, nil
		},
	}
	c, group, _, testQueue := settlementFixture(t, ctx, agentStore)

	go func() {
		if err := c.Run(ctx, 1); err != nil {
			t.Errorf("Controller Run failed: %v", err)
		}
	}()
	testQueue.Add("group-1")

	// Let several reconcile cycles (initial + rate-limited requeues) pass.
	if err := waitWithTimeout(func() bool { return testQueue.getDoneCount() >= 3 }, 3*time.Second); err != nil {
		t.Fatalf("Timed out waiting for reconciles: %v", err)
	}

	if got := group.Spec().LockingJob(); got != "" {
		t.Errorf("Expected promotion to be held (no locking job), got %q", got)
	}
	if depth := group.Spec().GetWaitingJobQueue().Len(); depth != 1 {
		t.Errorf("Expected job-a to remain queued (depth 1), got %d", depth)
	}
}

// TestController_Reconcile_SettlementReleasesOnRunning: the moment job-b is
// observed RUNNING, the hold releases and job-a is promoted; job-b is then
// evictable through the normal preemption path.
func TestController_Reconcile_SettlementReleasesOnRunning(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	agentStore := &controller.MockSnapshotAgentStore{
		OperationResponses: []*agentpb.GetOperationResponse{
			{Status: agentpb.OperationStatus_OPERATION_STATUS_COMPLETE},
		},
	}
	c, group, jobB, testQueue := settlementFixture(t, ctx, agentStore)

	go func() {
		if err := c.Run(ctx, 1); err != nil {
			t.Errorf("Controller Run failed: %v", err)
		}
	}()
	testQueue.Add("group-1")

	// First observe the hold...
	if err := waitWithTimeout(func() bool { return testQueue.getDoneCount() >= 1 }, 2*time.Second); err != nil {
		t.Fatalf("Timed out waiting for first reconcile: %v", err)
	}
	if got := group.Spec().LockingJob(); got != "" {
		t.Fatalf("Expected promotion to be held before job-b runs, got %q", got)
	}

	// ...then job-b's engine lands (the watcher reports RUNNING).
	jobB.UpdateContextState("node-1", pb.SnapshotAgentJobState_STATE_RUNNING)

	if err := waitWithTimeout(func() bool { return group.Spec().LockingJob() == "job-a" }, 3*time.Second); err != nil {
		t.Fatalf("Expected job-a to be promoted once job-b was observed RUNNING: %v", err)
	}
}

// TestController_Reconcile_SettlementTimeoutUnblocks: a grant that never
// produces device activity (crashed engine, job exited right after grant)
// stops holding promotion once SettleTimeout expires.
func TestController_Reconcile_SettlementTimeoutUnblocks(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	agentStore := &controller.MockSnapshotAgentStore{
		OperationResponses: []*agentpb.GetOperationResponse{
			{Status: agentpb.OperationStatus_OPERATION_STATUS_COMPLETE},
		},
	}
	c, group, _, testQueue := settlementFixture(t, ctx, agentStore)
	c.SettleTimeout = 100 * time.Millisecond

	go func() {
		if err := c.Run(ctx, 1); err != nil {
			t.Errorf("Controller Run failed: %v", err)
		}
	}()
	testQueue.Add("group-1")

	if err := waitWithTimeout(func() bool { return group.Spec().LockingJob() == "job-a" }, 5*time.Second); err != nil {
		t.Fatalf("Expected promotion to proceed after SettleTimeout: %v", err)
	}
}
