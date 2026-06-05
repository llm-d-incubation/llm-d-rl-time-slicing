package controller

import (
	"context"
	"testing"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
)

func TestControllerReconciliation(t *testing.T) {
	clientset := fake.NewSimpleClientset()
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
					},
				}, nil
			}
			return &agentpb.StatusResponse{}, nil
		},
	}
	c := NewController(clientset, nodeInformer, podInformer, groupStore, jobStore, fakeAgentStore)

	// No mocks, running inline placeholders

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start informers
	informerFactory.Start(ctx.Done())

	// Start the controller in background
	go func() {
		if err := c.Run(ctx, 1); err != nil {
			t.Logf("Controller stopped: %v", err)
		}
	}()

	// Wait for cache sync (controller does this, but we need to make sure it started)
	time.Sleep(100 * time.Millisecond)

	// Test Case 1: Add Nodes with group label
	node1 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-1",
			Labels: map[string]string{
				"group.timeslice.io/group-a": "true",
			},
		},
	}
	_, err := clientset.CoreV1().Nodes().Create(context.TODO(), node1, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node-1: %v", err)
	}

	node2 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-2",
			Labels: map[string]string{
				"group.timeslice.io/group-a": "false",
			},
		},
	}
	_, err = clientset.CoreV1().Nodes().Create(context.TODO(), node2, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node-2: %v", err)
	}

	// Test Case 2: Add a Pod with group label
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod-1",
			Namespace: "default",
			Labels: map[string]string{
				"timeslice.io/group": "group-b",
			},
		},
	}
	_, err = clientset.CoreV1().Pods("default").Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// Test Case 3: Add a Pod with group and job label for group-a
	pod2 := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod-2",
			Namespace: "default",
			Labels: map[string]string{
				"timeslice.io/group": "group-a",
			},
			Annotations: map[string]string{
				"timeslice.io/job-id": "job-a",
			},
		},
	}
	_, err = clientset.CoreV1().Pods("default").Create(context.TODO(), pod2, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// Wait for reconciliation to happen
	time.Sleep(500 * time.Millisecond)

	// Assertions
	// 1. Verify groupStore has correct nodes for group-a
	g, err := groupStore.Get(context.Background(), "group-a")
	if err != nil {
		t.Fatalf("Failed to get group-a: %v", err)
	}
	nodesList := g.Nodes()
	if len(nodesList) != 1 || nodesList[0] != "node-1" {
		t.Errorf("Expected group-a to have nodes [node-1], got %v", nodesList)
	}

	// 2. Verify jobStore has job-a in group-a
	job, err := jobStore.Get(context.Background(), "group-a", "job-a")
	if err != nil {
		t.Fatalf("Failed to get job-a: %v", err)
	}
	if len(job.Pods()) != 1 || job.Pods()[0] != string(pod2.UID) {
		t.Errorf("Expected job-a to have pod-2 UID, got %v", job.Pods())
	}

	// 3. Verify job-a has correct context state on node-1
	ctxStates := job.ContextState()
	if ctxStates["node-1"] != pb.SnapshotAgentJobState_STATE_RUNNING {
		t.Errorf("Expected job-a on node-1 to be RUNNING, got %v", ctxStates["node-1"])
	}
}

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

func TestControllerReconciliation_GroupLivesWithNodes(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(clientset, 0)
	nodeInformer := informerFactory.Core().V1().Nodes()
	podInformer := informerFactory.Core().V1().Pods()

	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()
	fakeAgentStore := &fakeSnapshotAgentStore{}
	c := NewController(clientset, nodeInformer, podInformer, groupStore, jobStore, fakeAgentStore)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	informerFactory.Start(ctx.Done())

	go func() {
		if err := c.Run(ctx, 1); err != nil {
			t.Logf("Controller stopped: %v", err)
		}
	}()

	time.Sleep(100 * time.Millisecond)

	// 1. Add node and pod to establish group-a and job-a
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-1",
			Labels: map[string]string{
				"group.timeslice.io/group-a": "true",
			},
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
				"timeslice.io/group": "group-a",
			},
			Annotations: map[string]string{
				"timeslice.io/job-id": "job-a",
			},
		},
	}
	_, err = clientset.CoreV1().Pods("default").Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// Wait for reconciliation
	time.Sleep(500 * time.Millisecond)

	// Verify they are in store
	g, err := groupStore.Get(ctx, "group-a")
	if err != nil || len(g.Nodes()) != 1 {
		t.Fatalf("Setup failed, group-a not populated correctly: %v", err)
	}
	_, err = jobStore.Get(ctx, "group-a", "job-a")
	if err != nil {
		t.Fatalf("Setup failed, job-a not populated: %v", err)
	}

	// 2. Delete the pod -> job-a should disappear (since it was the only pod)
	err = clientset.CoreV1().Pods("default").Delete(ctx, "pod-1", metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete pod: %v", err)
	}

	// Wait for reconciliation
	time.Sleep(500 * time.Millisecond)

	// Verify job-a is deleted
	_, err = jobStore.Get(ctx, "group-a", "job-a")
	if err != store.ErrNotFound {
		t.Errorf("Expected job-a to be deleted (ErrNotFound), got: %v", err)
	}
	// Group should still exist because node-1 is still there
	g, err = groupStore.Get(ctx, "group-a")
	if err != nil || len(g.Nodes()) != 1 {
		t.Errorf("Expected group-a to still exist with 1 node, got error: %v", err)
	}

	// 3. Delete the node -> group-a should disappear
	err = clientset.CoreV1().Nodes().Delete(ctx, "node-1", metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete node: %v", err)
	}

	// Wait for reconciliation
	time.Sleep(500 * time.Millisecond)

	// Verify group-a is deleted
	_, err = groupStore.Get(ctx, "group-a")
	if err != store.ErrNotFound {
		t.Errorf("Expected group-a to be deleted (ErrNotFound), got: %v", err)
	}
}

func TestControllerReconciliation_GroupLivesWithPods(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(clientset, 0)
	nodeInformer := informerFactory.Core().V1().Nodes()
	podInformer := informerFactory.Core().V1().Pods()

	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()
	fakeAgentStore := &fakeSnapshotAgentStore{}
	c := NewController(clientset, nodeInformer, podInformer, groupStore, jobStore, fakeAgentStore)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	informerFactory.Start(ctx.Done())

	go func() {
		if err := c.Run(ctx, 1); err != nil {
			t.Logf("Controller stopped: %v", err)
		}
	}()

	time.Sleep(100 * time.Millisecond)

	// 1. Add node and pod to establish group-a and job-a
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-1",
			Labels: map[string]string{
				"group.timeslice.io/group-a": "true",
			},
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
				"timeslice.io/group": "group-a",
			},
			Annotations: map[string]string{
				"timeslice.io/job-id": "job-a",
			},
		},
	}
	_, err = clientset.CoreV1().Pods("default").Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// Wait for reconciliation
	time.Sleep(500 * time.Millisecond)

	// 2. Delete the node -> group-a should STILL EXIST because pod-1 is still there
	err = clientset.CoreV1().Nodes().Delete(ctx, "node-1", metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete node: %v", err)
	}

	// Wait for reconciliation
	time.Sleep(500 * time.Millisecond)

	// Verify group-a still exists but has 0 nodes
	g, err := groupStore.Get(ctx, "group-a")
	if err != nil {
		t.Errorf("Expected group-a to still exist, got error: %v", err)
	} else if len(g.Nodes()) != 0 {
		t.Errorf("Expected group-a to have 0 nodes, got %v", g.Nodes())
	}

	// Verify job-a still exists
	_, err = jobStore.Get(ctx, "group-a", "job-a")
	if err != nil {
		t.Errorf("Expected job-a to still exist, got error: %v", err)
	}

	// 3. Delete the pod -> group-a should now disappear (0 nodes and 0 pods)
	err = clientset.CoreV1().Pods("default").Delete(ctx, "pod-1", metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete pod: %v", err)
	}

	// Wait for reconciliation
	time.Sleep(500 * time.Millisecond)

	// Verify group-a is deleted
	_, err = groupStore.Get(ctx, "group-a")
	if err != store.ErrNotFound {
		t.Errorf("Expected group-a to be deleted (ErrNotFound), got: %v", err)
	}
}
