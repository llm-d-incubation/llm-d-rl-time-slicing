//nolint:testpackage // Integration E2E tests require package-level access to unexported helper trackQueue
package acceleratororchestrator

import (
	"context"
	"fmt"
	"net"
	"testing"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/controller"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/infrastructure"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/server"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	google_grpc "google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/util/workqueue"
)

// TestE2E_SingleRLJob tests a single RL job running to completion using FakeRLJob.
func TestE2E_SingleRLJob(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// 1. Initialize Fake Kubernetes Clientset
	clientset := fake.NewClientset()
	go StartFakeScheduler(ctx, clientset)

	// Populate Fake Kubernetes with Nodes BEFORE starting informers to avoid startup races
	nodeSampler := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-sampler-1",
			Labels: map[string]string{"group.timeslice.io/samplers": "true"},
		},
	}
	_, err := clientset.CoreV1().Nodes().Create(ctx, nodeSampler, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node-sampler-1: %v", err)
	}

	nodeTrainer := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-trainer-1",
			Labels: map[string]string{"group.timeslice.io/trainers": "true"},
		},
	}
	_, err = clientset.CoreV1().Nodes().Create(ctx, nodeTrainer, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node-trainer-1: %v", err)
	}

	informerFactory := informers.NewSharedInformerFactory(clientset, 0)
	nodeInformer := informerFactory.Core().V1().Nodes()
	podInformer := informerFactory.Core().V1().Pods()

	// 2. Initialize Stores
	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()

	// 3. Setup Agent State Simulator (Fake)
	fakeAgentStore := NewFakeSnapshotAgentStore()

	// 4. Initialize Infrastructure Orchestrator and Controller
	testQueue := &trackQueue{
		TypedRateLimitingInterface: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "test-e2e-single-job"},
		),
	}

	infraOrch := infrastructure.NewKubernetesOrchestrator(
		nodeInformer,
		podInformer,
		groupStore,
		jobStore,
		fakeAgentStore,
	)

	ctrl := controller.NewController(
		groupStore,
		jobStore,
		testQueue,
		infraOrch,
		fakeAgentStore,
	)

	// Start Informers and Infrastructure Orchestrator
	informerFactory.Start(ctx.Done())
	if err := infraOrch.Start(ctx, testQueue); err != nil {
		t.Fatalf("Failed to start infra orchestrator: %v", err)
	}

	// Start Controller
	go func() {
		if err := ctrl.Run(ctx, 1); err != nil {
			t.Errorf("Controller Run failed: %v", err)
		}
	}()

	// 5. Start gRPC Server
	lc := net.ListenConfig{}
	lis, err := lc.Listen(ctx, "tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Failed to listen: %v", err)
	}
	grpcServer := google_grpc.NewServer()
	srv := server.NewServer(ctrl, groupStore, jobStore)
	pb.RegisterAcceleratorOrchestratorServiceServer(grpcServer, srv)
	go func() {
		if err := grpcServer.Serve(lis); err != nil {
			t.Errorf("gRPC server failed: %v", err)
		}
	}()
	defer grpcServer.Stop()

	// Create gRPC Client
	conn, err := google_grpc.NewClient(lis.Addr().String(), google_grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to create gRPC client: %v", err)
	}
	defer conn.Close()
	client := pb.NewAcceleratorOrchestratorServiceClient(conn)

	// Nodes already populated at startup

	// Wait for K8s caches to sync and infra orchestrator to populate stores
	err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		gs, err := groupStore.Get(ctx, "samplers")
		if err != nil {
			return false, nil //nolint:nilerr // Intentional to allow polling to retry on transient "not found" errors
		}
		gt, err := groupStore.Get(ctx, "trainers")
		if err != nil {
			return false, nil //nolint:nilerr // Intentional to allow polling to retry on transient "not found" errors
		}
		return len(gs.Status().Nodes()) == 1 && len(gt.Status().Nodes()) == 1, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for store initialization: %v", err)
	}

	t.Log("Store initialized with samplers and trainers groups")

	// Run Scenario
	if err := RunSingleRLJobScenario(ctx, clientset, client, t, "", ""); err != nil {
		t.Fatalf("Scenario failed: %v", err)
	}
}

// TestE2E_QueuedRLJobs tests multiple RL jobs contending for locks and queuing.
func TestE2E_QueuedRLJobs(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// 1. Initialize Fake Kubernetes Clientset
	clientset := fake.NewClientset()
	go StartFakeScheduler(ctx, clientset)

	// Populate Fake Kubernetes with Nodes BEFORE starting informers to avoid startup races
	nodeSampler := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-sampler-1",
			Labels: map[string]string{"group.timeslice.io/samplers": "true"},
		},
	}
	_, err := clientset.CoreV1().Nodes().Create(ctx, nodeSampler, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node-sampler-1: %v", err)
	}

	nodeTrainer := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-trainer-1",
			Labels: map[string]string{"group.timeslice.io/trainers": "true"},
		},
	}
	_, err = clientset.CoreV1().Nodes().Create(ctx, nodeTrainer, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node-trainer-1: %v", err)
	}

	informerFactory := informers.NewSharedInformerFactory(clientset, 0)
	nodeInformer := informerFactory.Core().V1().Nodes()
	podInformer := informerFactory.Core().V1().Pods()

	// 2. Initialize Stores
	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()

	// 3. Setup Agent State Simulator (Fake)
	fakeAgentStore := NewFakeSnapshotAgentStore()

	// 4. Initialize Infrastructure Orchestrator and Controller
	testQueue := &trackQueue{
		TypedRateLimitingInterface: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "test-e2e-queued-jobs"},
		),
	}

	infraOrch := infrastructure.NewKubernetesOrchestrator(
		nodeInformer,
		podInformer,
		groupStore,
		jobStore,
		fakeAgentStore,
	)

	ctrl := controller.NewController(
		groupStore,
		jobStore,
		testQueue,
		infraOrch,
		fakeAgentStore,
	)

	// Start Informers and Infrastructure Orchestrator
	informerFactory.Start(ctx.Done())
	if err := infraOrch.Start(ctx, testQueue); err != nil {
		t.Fatalf("Failed to start infra orchestrator: %v", err)
	}

	// Start Controller
	go func() {
		if err := ctrl.Run(ctx, 1); err != nil {
			t.Errorf("Controller Run failed: %v", err)
		}
	}()

	// 5. Start gRPC Server
	lc := net.ListenConfig{}
	lis, err := lc.Listen(ctx, "tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Failed to listen: %v", err)
	}
	grpcServer := google_grpc.NewServer()
	srv := server.NewServer(ctrl, groupStore, jobStore)
	pb.RegisterAcceleratorOrchestratorServiceServer(grpcServer, srv)
	go func() {
		if err := grpcServer.Serve(lis); err != nil {
			t.Errorf("gRPC server failed: %v", err)
		}
	}()
	defer grpcServer.Stop()

	// Create gRPC Client
	conn, err := google_grpc.NewClient(lis.Addr().String(), google_grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to create gRPC client: %v", err)
	}
	defer conn.Close()
	client := pb.NewAcceleratorOrchestratorServiceClient(conn)

	// Nodes already populated at startup

	// Wait for K8s caches to sync and infra orchestrator to populate stores
	err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		gs, err := groupStore.Get(ctx, "samplers")
		if err != nil {
			return false, nil //nolint:nilerr // Intentional to allow polling to retry on transient "not found" errors
		}
		gt, err := groupStore.Get(ctx, "trainers")
		if err != nil {
			return false, nil //nolint:nilerr // Intentional to allow polling to retry on transient "not found" errors
		}
		return len(gs.Status().Nodes()) == 1 && len(gt.Status().Nodes()) == 1, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for store initialization: %v", err)
	}

	t.Log("Store initialized with samplers and trainers groups")

	// Run Scenario
	if err := RunQueuedRLJobsScenario(ctx, clientset, client, t, "", ""); err != nil {
		t.Fatalf("Scenario failed: %v", err)
	}
}

// StartFakeScheduler simulates a K8s scheduler and DRA driver for the fake clientset.
// It watches for unscheduled pods, binds them to a matching node, and allocates their ResourceClaims.
func StartFakeScheduler(ctx context.Context, clientset *fake.Clientset) {
	podWatch, err := clientset.CoreV1().Pods("default").Watch(ctx, metav1.ListOptions{})
	if err != nil {
		return
	}
	defer podWatch.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case event, ok := <-podWatch.ResultChan():
			if !ok {
				return
			}
			if event.Type != watch.Added && event.Type != watch.Modified {
				continue
			}
			pod, ok := event.Object.(*corev1.Pod)
			if !ok {
				continue
			}

			// If already scheduled, skip
			if pod.Spec.NodeName != "" {
				continue
			}

			// Find matching node based on group label
			groupID := pod.Labels["timeslice.io/group"]
			if groupID == "" {
				continue
			}

			nodes, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
			if err != nil {
				continue
			}

			var targetNode *corev1.Node
			selectorKey := fmt.Sprintf("group.timeslice.io/%s", groupID)
			for i := range nodes.Items {
				node := &nodes.Items[i]
				if node.Labels[selectorKey] == "true" {
					targetNode = node
					break
				}
			}

			if targetNode == nil {
				continue
			}

			// 1. Update pod with NodeName, Running status, and Ready condition (Simulate Scheduler & Kubelet)
			podCopy := pod.DeepCopy()
			podCopy.Spec.NodeName = targetNode.Name
			podCopy.Status.Phase = corev1.PodRunning
			podCopy.Status.Conditions = []corev1.PodCondition{
				{
					Type:   corev1.PodReady,
					Status: corev1.ConditionTrue,
				},
			}

			_, err = clientset.CoreV1().Pods("default").Update(ctx, podCopy, metav1.UpdateOptions{})
			if err != nil {
				continue
			}

			// 2. Allocate the ResourceClaim if referenced (Simulate DRA Driver)
			for _, pc := range pod.Spec.ResourceClaims {
				if pc.ResourceClaimName == nil {
					continue
				}
				claimName := *pc.ResourceClaimName
				claim, err := clientset.ResourceV1().ResourceClaims("default").Get(ctx, claimName, metav1.GetOptions{})
				if err != nil {
					continue
				}

				claimCopy := claim.DeepCopy()
				claimCopy.Status = resourcev1.ResourceClaimStatus{
					Allocation: &resourcev1.AllocationResult{
						Devices: resourcev1.DeviceAllocationResult{
							Results: []resourcev1.DeviceRequestAllocationResult{
								{
									Request: "gpu",
									Driver:  "gpu.nvidia.com",
									Pool:    targetNode.Name,
									Device:  "gpu-0",
								},
							},
						},
					},
					ReservedFor: []resourcev1.ResourceClaimConsumerReference{
						{
							Resource: "pods",
							Name:     pod.Name,
							UID:      pod.UID,
						},
					},
				}

				_, err = clientset.ResourceV1().ResourceClaims("default").UpdateStatus(ctx, claimCopy, metav1.UpdateOptions{})
				if err != nil {
					// Fallback to regular update if UpdateStatus fails in fake client
					//nolint:errcheck // Ignore error from fallback update in fake client
					_, _ = clientset.ResourceV1().ResourceClaims("default").Update(ctx, claimCopy, metav1.UpdateOptions{})
				}
			}
		}
	}
}
