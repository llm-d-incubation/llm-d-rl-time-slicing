package simulate_test

import (
	"context"
	"sync"
	"testing"
	"time"

	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/timeslice-orchestrator/controller"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/timeslice-orchestrator/store"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/tests/integration/orchestrator/simulate"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/workqueue"
)

// noopInfraOrchestrator satisfies controller.InfrastructureOrchestrator for tests
// that set up group/job state in the stores directly.
type noopInfraOrchestrator struct{}

func (n *noopInfraOrchestrator) Init(ctx context.Context) error { return nil }
func (n *noopInfraOrchestrator) ObserveGroupState(ctx context.Context, groupID string) error {
	return nil
}

// TestSimulated_MultiNodeGroup_RendezvousSwitch drives the controller through two full
// job switches on a 3-node group whose fake snapshot agents enforce slice-wide
// rendezvous semantics (as multi-host TPU/libtpu does): a checkpoint or restore
// operation only completes once the same operation is in flight on every node of the
// group. If the controller issued the operations serially and blocked per node, the
// first node's operation would stay pending forever and this test would time out.
func TestSimulated_MultiNodeGroup_RendezvousSwitch(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	groupID := "trainers"
	nodes := []string{"node-1", "node-2", "node-3"}

	// 1. Stores with a 3-node group.
	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()

	group, _, err := groupStore.GetOrCreate(ctx, groupID)
	if err != nil {
		t.Fatalf("failed to create group: %v", err)
	}
	group.Status().SetNodes(nodes)

	// 2. Fake agents with multi-host rendezvous on both operation types.
	fakeAgentStore := simulate.NewFakeSnapshotAgentStore()
	fakeAgentStore.RestoreRendezvous = len(nodes)
	fakeAgentStore.SnapshotRendezvous = len(nodes)

	// job-1 is loaded (RUNNING) on the whole slice; job-2 is parked (SAVED).
	for _, node := range nodes {
		fakeAgentStore.SetJobState(node, "job-1", agentpb.JobState_JOB_STATE_RUNNING)
		fakeAgentStore.SetJobState(node, "job-2", agentpb.JobState_JOB_STATE_SAVED)
	}

	// Jobs must exist in the store for ObserveJobContext to track their states.
	if err := jobStore.Put(ctx, store.NewJob(groupID, "job-1")); err != nil {
		t.Fatalf("failed to put job-1: %v", err)
	}
	if err := jobStore.Put(ctx, store.NewJob(groupID, "job-2")); err != nil {
		t.Fatalf("failed to put job-2: %v", err)
	}

	// Record which nodes each operation type was issued on.
	var mu sync.Mutex
	snapshotNodes := make(map[string]int)
	restoreNodes := make(map[string]int)
	fakeAgentStore.OnSnapshot = func(node, jobID string) {
		mu.Lock()
		defer mu.Unlock()
		snapshotNodes[node]++
	}
	fakeAgentStore.OnRestore = func(node, jobID string) {
		mu.Lock()
		defer mu.Unlock()
		restoreNodes[node]++
	}

	// 3. Controller.
	queue := &simulate.TrackQueue{
		TypedRateLimitingInterface: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "test-multinode-rendezvous"},
		),
	}
	ctrl := controller.NewController(groupStore, jobStore, queue, &noopInfraOrchestrator{}, fakeAgentStore)
	go func() {
		if err := ctrl.Run(ctx, 1); err != nil {
			t.Errorf("Controller Run failed: %v", err)
		}
	}()

	waitForStates := func(jobID string, want agentpb.JobState) error {
		return wait.PollUntilContextTimeout(ctx, 200*time.Millisecond, 20*time.Second, true,
			func(ctx context.Context) (bool, error) {
				for _, node := range nodes {
					if fakeAgentStore.GetJobState(node, jobID) != want {
						return false, nil
					}
				}
				return true, nil
			})
	}

	// The OnSnapshot/OnRestore hooks fire asynchronously, so give the counters a
	// moment to catch up before asserting exact values.
	// A timeout here just means the exact-count asserts below fail with details.
	waitForCounts := func(counts map[string]int, want int) {
		_ = wait.PollUntilContextTimeout(ctx, 50*time.Millisecond, 2*time.Second, true,
			func(ctx context.Context) (bool, error) {
				mu.Lock()
				defer mu.Unlock()
				for _, node := range nodes {
					if counts[node] != want {
						return false, nil
					}
				}
				return true, nil
			})
	}

	// 4. Switch 1: job-2 becomes active. The controller must checkpoint job-1 on all
	// three nodes concurrently, then restore job-2 on all three nodes concurrently.
	group.Spec().SetActiveJob("job-2")
	queue.Add(groupID)

	if err := waitForStates("job-1", agentpb.JobState_JOB_STATE_SAVED); err != nil {
		t.Fatalf("switch 1: job-1 was not checkpointed on all nodes (rendezvous deadlock?): %v", err)
	}
	if err := waitForStates("job-2", agentpb.JobState_JOB_STATE_RUNNING); err != nil {
		t.Fatalf("switch 1: job-2 was not restored on all nodes (rendezvous deadlock?): %v", err)
	}

	waitForCounts(snapshotNodes, 1)
	waitForCounts(restoreNodes, 1)
	mu.Lock()
	for _, node := range nodes {
		if snapshotNodes[node] != 1 {
			t.Errorf("switch 1: expected exactly 1 snapshot on %s, got %d", node, snapshotNodes[node])
		}
		if restoreNodes[node] != 1 {
			t.Errorf("switch 1: expected exactly 1 restore on %s, got %d", node, restoreNodes[node])
		}
	}
	mu.Unlock()

	// 5. Switch 2: back to job-1, proving repeated rendezvous switches converge.
	group.Spec().SetActiveJob("job-1")
	queue.Add(groupID)

	if err := waitForStates("job-2", agentpb.JobState_JOB_STATE_SAVED); err != nil {
		t.Fatalf("switch 2: job-2 was not checkpointed on all nodes: %v", err)
	}
	if err := waitForStates("job-1", agentpb.JobState_JOB_STATE_RUNNING); err != nil {
		t.Fatalf("switch 2: job-1 was not restored on all nodes: %v", err)
	}

	waitForCounts(snapshotNodes, 2)
	waitForCounts(restoreNodes, 2)
	mu.Lock()
	defer mu.Unlock()
	for _, node := range nodes {
		if snapshotNodes[node] != 2 {
			t.Errorf("switch 2: expected 2 snapshots total on %s, got %d", node, snapshotNodes[node])
		}
		if restoreNodes[node] != 2 {
			t.Errorf("switch 2: expected 2 restores total on %s, got %d", node, restoreNodes[node])
		}
	}
}
