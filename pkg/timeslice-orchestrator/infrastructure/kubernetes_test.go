package infrastructure_test

import (
	"context"
	"errors"
	"testing"
	"time"

	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/timeslice-orchestrator/infrastructure"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/timeslice-orchestrator/store"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
)

type fakeSnapshotAgentStore struct {
	statusFunc    func(ctx context.Context, nodeName string) (*agentpb.StatusResponse, error)
	closeFunc     func(nodeName string) error
	snapshotFunc  func(ctx context.Context, nodeName, jobID, groupID string) (*agentpb.SnapshotResponse, error)
	operationFunc func(ctx context.Context, nodeName, operationID string) (*agentpb.GetOperationResponse, error)
	restoreFunc   func(ctx context.Context, nodeName, jobID, groupID string) (*agentpb.RestoreResponse, error)
}

func (f *fakeSnapshotAgentStore) GetStatus(ctx context.Context, nodeName string) (*agentpb.StatusResponse, error) {
	if f.statusFunc != nil {
		return f.statusFunc(ctx, nodeName)
	}
	return &agentpb.StatusResponse{}, nil
}

func (f *fakeSnapshotAgentStore) CloseClient(nodeName string) error {
	if f.closeFunc != nil {
		return f.closeFunc(nodeName)
	}
	return nil
}

func (f *fakeSnapshotAgentStore) Snapshot(
	ctx context.Context, nodeName, jobID, groupID string,
) (*agentpb.SnapshotResponse, error) {
	if f.snapshotFunc != nil {
		return f.snapshotFunc(ctx, nodeName, jobID, groupID)
	}
	return &agentpb.SnapshotResponse{}, nil
}

func (f *fakeSnapshotAgentStore) GetOperation(
	ctx context.Context, nodeName, operationID string,
) (*agentpb.GetOperationResponse, error) {
	if f.operationFunc != nil {
		return f.operationFunc(ctx, nodeName, operationID)
	}
	return &agentpb.GetOperationResponse{}, nil
}

func (f *fakeSnapshotAgentStore) Restore(
	ctx context.Context, nodeName, jobID, groupID string,
) (*agentpb.RestoreResponse, error) {
	if f.restoreFunc != nil {
		return f.restoreFunc(ctx, nodeName, jobID, groupID)
	}
	return &agentpb.RestoreResponse{}, nil
}

func TestObserveGroupState_Cleanup(t *testing.T) {
	clientset := fake.NewClientset()
	informerFactory := informers.NewSharedInformerFactory(clientset, 0)
	nodeInformer := informerFactory.Core().V1().Nodes()
	podInformer := informerFactory.Core().V1().Pods()

	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()
	fakeAgentStore := &fakeSnapshotAgentStore{}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	infraOrch := infrastructure.NewKubernetesOrchestrator(nodeInformer, podInformer, groupStore, jobStore, fakeAgentStore)

	// Start informers
	informerFactory.Start(ctx.Done())
	if err := infraOrch.Init(ctx); err != nil {
		t.Fatalf("Failed to initialize infra orchestrator: %v", err)
	}

	// Pre-populate store with a group and a job to test cleanup
	g, _, err := groupStore.GetOrCreate(ctx, "group-a")
	if err != nil {
		t.Fatalf("Failed to create group: %v", err)
	}
	g.Status().SetNodes([]string{"node-1"})

	job := store.NewJob("group-a", "job-1")
	if err := jobStore.Put(ctx, job); err != nil {
		t.Fatalf("Failed to put job: %v", err)
	}

	// Call ObserveGroupState when there are no nodes and no pods in K8s
	err = infraOrch.ObserveGroupState(ctx, "group-a")
	if err != nil {
		t.Fatalf("ObserveGroupState failed: %v", err)
	}

	// Verify group and job are deleted from store
	_, err = groupStore.Get(ctx, "group-a")
	if !errors.Is(err, store.ErrNotFound) {
		t.Errorf("Expected group-a to be deleted (ErrNotFound), got: %v", err)
	}

	_, err = jobStore.Get(ctx, "group-a", "job-1")
	if !errors.Is(err, store.ErrNotFound) {
		t.Errorf("Expected job-a to be deleted (ErrNotFound), got: %v", err)
	}
}

// newInfra builds a KubernetesOrchestrator wired to a fake clientset with the
// given objects, with informer caches synced. Shared by the membership tests.
func newInfra(
	t *testing.T, ctx context.Context, groupStore *store.GroupStore, jobStore *store.JobStore, objs ...runtime.Object,
) *infrastructure.KubernetesOrchestrator {
	t.Helper()
	clientset := fake.NewClientset(objs...)
	informerFactory := informers.NewSharedInformerFactory(clientset, 0)
	nodeInformer := informerFactory.Core().V1().Nodes()
	podInformer := informerFactory.Core().V1().Pods()
	infraOrch := infrastructure.NewKubernetesOrchestrator(
		nodeInformer, podInformer, groupStore, jobStore, &fakeSnapshotAgentStore{})
	informerFactory.Start(ctx.Done())
	if err := infraOrch.Init(ctx); err != nil {
		t.Fatalf("Failed to initialize infra orchestrator: %v", err)
	}
	return infraOrch
}

func groupPod(name, group, job, nodeName string, phase corev1.PodPhase) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
			Labels: map[string]string{
				"timeslice.io/group":  group,
				"timeslice.io/job-id": job,
			},
		},
		Spec:   corev1.PodSpec{NodeName: nodeName},
		Status: corev1.PodStatus{Phase: phase},
	}
}

// TestObserveGroupState_NodesFromPods verifies that a group's nodes are
// exactly the distinct nodes of its live, scheduled member pods. Unscheduled pods contribute nothing (no node
// yet); terminal pods contribute nothing (their accelerator processes are
// gone); duplicates collapse.
func TestObserveGroupState_NodesFromPods(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	groupStore := store.NewGroupStore(store.NewMemLockStore())
	jobStore := store.NewJobStore()

	infraOrch := newInfra(t, ctx, groupStore, jobStore,
		groupPod("pod-running-1", "group-1", "job-a", "node-1", corev1.PodRunning),
		groupPod("pod-running-2", "group-1", "job-b", "node-1", corev1.PodRunning), // same node: dedup
		groupPod("pod-pending", "group-1", "job-a", "", corev1.PodPending),         // unscheduled: ignored
		groupPod("pod-done", "group-1", "job-a", "node-2", corev1.PodSucceeded),    // terminal: ignored
		groupPod("pod-other-group", "group-2", "job-c", "node-3", corev1.PodRunning),
	)

	if err := infraOrch.ObserveGroupState(ctx, "group-1"); err != nil {
		t.Fatalf("ObserveGroupState failed: %v", err)
	}
	g, err := groupStore.Get(ctx, "group-1")
	if err != nil {
		t.Fatalf("Expected group-1 to exist in store: %v", err)
	}
	if nodes := g.Status().Nodes(); len(nodes) != 1 || nodes[0] != "node-1" {
		t.Errorf("Expected group-1 nodes to be [node-1], got %v", nodes)
	}
}

// TestObserveGroupState_AllPodsTerminal_Cleanup verifies that terminal pods
// count as absent for the cleanup decision too: when every member pod has
// succeeded or failed (and there is no lock activity), the group and its jobs
// are removed from the store instead of lingering until Kubernetes
// garbage-collects the finished pod objects.
func TestObserveGroupState_AllPodsTerminal_Cleanup(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	groupStore := store.NewGroupStore(store.NewMemLockStore())
	jobStore := store.NewJobStore()

	infraOrch := newInfra(t, ctx, groupStore, jobStore,
		groupPod("pod-done-1", "group-a", "job-1", "node-1", corev1.PodSucceeded),
		groupPod("pod-done-2", "group-a", "job-1", "node-2", corev1.PodFailed),
	)

	// Pre-populate the store as if the group and job had been observed live.
	g, _, err := groupStore.GetOrCreate(ctx, "group-a")
	if err != nil {
		t.Fatalf("Failed to create group: %v", err)
	}
	g.Status().SetNodes([]string{"node-1", "node-2"})
	if err := jobStore.Put(ctx, store.NewJob("group-a", "job-1")); err != nil {
		t.Fatalf("Failed to put job: %v", err)
	}

	if err := infraOrch.ObserveGroupState(ctx, "group-a"); err != nil {
		t.Fatalf("ObserveGroupState failed: %v", err)
	}

	if _, err := groupStore.Get(ctx, "group-a"); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("Expected group-a to be cleaned up when all pods are terminal, got: %v", err)
	}
	if _, err := jobStore.Get(ctx, "group-a", "job-1"); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("Expected job-1 to be cleaned up when all pods are terminal, got: %v", err)
	}
}

// TestObserveGroupState_TerminalJobRemoved verifies the per-job consequence of
// the liveness rule: a job whose pods have all finished is dropped from the
// store (so stale context state, e.g. FAULTED, cannot outlive its pods),
// while jobs with live pods are retained.
func TestObserveGroupState_TerminalJobRemoved(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	groupStore := store.NewGroupStore(store.NewMemLockStore())
	jobStore := store.NewJobStore()

	infraOrch := newInfra(t, ctx, groupStore, jobStore,
		groupPod("pod-live", "group-a", "job-live", "node-1", corev1.PodRunning),
		groupPod("pod-done", "group-a", "job-done", "node-2", corev1.PodSucceeded),
	)

	if err := jobStore.Put(ctx, store.NewJob("group-a", "job-done")); err != nil {
		t.Fatalf("Failed to put job: %v", err)
	}

	if err := infraOrch.ObserveGroupState(ctx, "group-a"); err != nil {
		t.Fatalf("ObserveGroupState failed: %v", err)
	}

	g, err := groupStore.Get(ctx, "group-a")
	if err != nil {
		t.Fatalf("Expected group-a to exist: %v", err)
	}
	if nodes := g.Status().Nodes(); len(nodes) != 1 || nodes[0] != "node-1" {
		t.Errorf("Expected group-a nodes to be [node-1], got %v", nodes)
	}
	if _, err := jobStore.Get(ctx, "group-a", "job-live"); err != nil {
		t.Errorf("Expected job-live to be retained, got: %v", err)
	}
	if _, err := jobStore.Get(ctx, "group-a", "job-done"); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("Expected job-done to be removed once its pods are terminal, got: %v", err)
	}
}

// TestObserveGroupState_NodeLabelsIgnored verifies that group.timeslice.io/*
// node labels have no effect: a labeled node with no member pods must NOT
// create or populate a group.
func TestObserveGroupState_NodeLabelsIgnored(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	groupStore := store.NewGroupStore(store.NewMemLockStore())
	jobStore := store.NewJobStore()

	labeledNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-1",
			Labels: map[string]string{"group.timeslice.io/group-1": "true"},
		},
	}
	infraOrch := newInfra(t, ctx, groupStore, jobStore, labeledNode)

	if err := infraOrch.ObserveGroupState(ctx, "group-1"); err != nil {
		t.Fatalf("ObserveGroupState failed: %v", err)
	}
	if _, err := groupStore.Get(ctx, "group-1"); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("Expected group-1 to not exist (labels are ignored), got: %v", err)
	}
}

// TestObserveGroupState_GCGuard verifies that a group with lock activity but
// no pods yet is NOT garbage-collected: in the lock-then-deploy pattern,
// Acquire creates the group before any pods exist, and cleanup must not
// delete it out from under the lock holder.
func TestObserveGroupState_GCGuard(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	groupStore := store.NewGroupStore(store.NewMemLockStore())
	jobStore := store.NewJobStore()

	infraOrch := newInfra(t, ctx, groupStore, jobStore)

	g, _, err := groupStore.GetOrCreate(ctx, "group-a")
	if err != nil {
		t.Fatalf("Failed to create group: %v", err)
	}
	// Simulate a job that acquired the (empty) group and holds the lock.
	g.Spec().RequestLock("job-1")
	if _, err := g.Spec().TryPromote(ctx); err != nil {
		t.Fatalf("TryPromote failed: %v", err)
	}

	if err := infraOrch.ObserveGroupState(ctx, "group-a"); err != nil {
		t.Fatalf("ObserveGroupState failed: %v", err)
	}
	if _, err := groupStore.Get(ctx, "group-a"); err != nil {
		t.Errorf("Expected group-a to survive cleanup while lock is held, got: %v", err)
	}
}

func TestObserveGroupState_UpdateJobsAndPods(t *testing.T) {
	clientset := fake.NewClientset()
	informerFactory := informers.NewSharedInformerFactory(clientset, 0)
	nodeInformer := informerFactory.Core().V1().Nodes()
	podInformer := informerFactory.Core().V1().Pods()

	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()

	fakeAgentStore := &fakeSnapshotAgentStore{}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	infraOrch := infrastructure.NewKubernetesOrchestrator(nodeInformer, podInformer, groupStore, jobStore, fakeAgentStore)

	// Start informers
	informerFactory.Start(ctx.Done())
	if err := infraOrch.Init(ctx); err != nil {
		t.Fatalf("Failed to initialize infra orchestrator: %v", err)
	}

	// Add pod to fake clientset (jobs are tracked from labeled pods alone).
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod-1",
			Namespace: "default",
			Labels: map[string]string{
				"timeslice.io/group":  "group-1",
				"timeslice.io/job-id": "job-a",
			},
		},
	}
	_, err := clientset.CoreV1().Pods(pod.Namespace).Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// Wait for caches to sync
	err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 3*time.Second, true, func(ctx context.Context) (bool, error) {
		pods, err := podInformer.Lister().List(labels.Everything())
		if err != nil {
			return false, err
		}
		return len(pods) == 1, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for caches to sync: %v", err)
	}

	// Call ObserveGroupState
	err = infraOrch.ObserveGroupState(ctx, "group-1")
	if err != nil {
		t.Fatalf("ObserveGroupState failed: %v", err)
	}

	// Verify job-a is in store with correct pods
	job, err := jobStore.Get(ctx, "group-1", "job-a")
	if err != nil {
		t.Fatalf("Expected job-a to exist in store: %v", err)
	}
	if len(job.Pods()) != 1 || job.Pods()[0] != string(pod.UID) {
		t.Errorf("Expected job-a to have pod %s, got %v", pod.UID, job.Pods())
	}

	// Verify job-b (which has no pods) is NOT in store
	_, err = jobStore.Get(ctx, "group-1", "job-b")
	if !errors.Is(err, store.ErrNotFound) {
		t.Errorf("Expected job-b to not exist in store, got: %v", err)
	}
}
