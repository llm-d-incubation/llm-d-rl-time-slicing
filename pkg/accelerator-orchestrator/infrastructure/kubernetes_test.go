package infrastructure_test

import (
	"context"
	"errors"
	"testing"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/infrastructure"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
)

type fakeSnapshotAgentStore struct {
	statusFunc func(ctx context.Context, nodeName string) (*agentpb.StatusResponse, error)
	closeFunc  func(nodeName string) error
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
	g.SetNodes([]string{"node-1"})

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

func TestObserveGroupState_UpdateNodes(t *testing.T) {
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

	// Add node to fake clientset
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-1",
			Labels: map[string]string{"group.timeslice.io/group-1": "true"},
		},
	}
	_, err := clientset.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// Wait for caches to sync
	err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 3*time.Second, true, func(ctx context.Context) (bool, error) {
		nodes, err := nodeInformer.Lister().List(labels.Everything())
		if err != nil {
			return false, err
		}
		return len(nodes) == 1, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for caches to sync: %v", err)
	}

	// Call ObserveGroupState
	err = infraOrch.ObserveGroupState(ctx, "group-1")
	if err != nil {
		t.Fatalf("ObserveGroupState failed: %v", err)
	}

	// Verify group is created in store with correct nodes
	g, err := groupStore.Get(ctx, "group-1")
	if err != nil {
		t.Fatalf("Expected group-1 to exist in store: %v", err)
	}
	if len(g.Nodes()) != 1 || g.Nodes()[0] != "node-1" {
		t.Errorf("Expected group-1 to have node-1, got %v", g.Nodes())
	}
}

func TestObserveGroupState_UpdateJobsAndContext(t *testing.T) {
	clientset := fake.NewClientset()
	informerFactory := informers.NewSharedInformerFactory(clientset, 0)
	nodeInformer := informerFactory.Core().V1().Nodes()
	podInformer := informerFactory.Core().V1().Pods()

	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()

	fakeAgentStore := &fakeSnapshotAgentStore{
		statusFunc: func(ctx context.Context, nodeName string) (*agentpb.StatusResponse, error) {
			if nodeName == "node-1" {
				return &agentpb.StatusResponse{
					JobStatuses: []*agentpb.JobStatus{
						{JobId: "job-a", State: agentpb.JobState_JOB_STATE_RUNNING},
						{JobId: "job-b", State: agentpb.JobState_JOB_STATE_IDLE}, // Should be ignored because job-b has no pods
					},
				}, nil
			}
			return &agentpb.StatusResponse{}, nil
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	infraOrch := infrastructure.NewKubernetesOrchestrator(nodeInformer, podInformer, groupStore, jobStore, fakeAgentStore)

	// Start informers
	informerFactory.Start(ctx.Done())
	if err := infraOrch.Init(ctx); err != nil {
		t.Fatalf("Failed to initialize infra orchestrator: %v", err)
	}

	// Add node and pod to fake clientset
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-1",
			Labels: map[string]string{"group.timeslice.io/group-1": "true"},
		},
	}
	_, err := clientset.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

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
	_, err = clientset.CoreV1().Pods(pod.Namespace).Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// Wait for caches to sync
	err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 3*time.Second, true, func(ctx context.Context) (bool, error) {
		nodes, err := nodeInformer.Lister().List(labels.Everything())
		if err != nil {
			return false, err
		}
		pods, err := podInformer.Lister().List(labels.Everything())
		if err != nil {
			return false, err
		}
		return len(nodes) == 1 && len(pods) == 1, nil
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

	// Verify job-a context state is updated from snapshot agent
	ctxState, ok := job.ContextState()["node-1"]
	if !ok {
		t.Fatalf("Expected context state for job-a on node-1 to exist")
	}
	if ctxState != pb.SnapshotAgentJobState_STATE_RUNNING {
		t.Errorf("Expected job-a state to be RUNNING, got %v", ctxState)
	}

	// Verify job-b (which was in snapshot agent but not observed as pod) is NOT in store
	_, err = jobStore.Get(ctx, "group-1", "job-b")
	if !errors.Is(err, store.ErrNotFound) {
		t.Errorf("Expected job-b to not exist in store, got: %v", err)
	}
}
