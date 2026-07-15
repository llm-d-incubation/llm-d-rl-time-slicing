package acceleratororchestrator

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
)

// Logger defines a simple logging interface compatible with testing.T and custom CLI loggers.
type Logger interface {
	Log(args ...interface{})
	Logf(format string, args ...interface{})
	Error(args ...interface{})
	Errorf(format string, args ...interface{})
}

// FakeRLJob simulates a Reinforcement Learning job that orchestrates samplers and trainers.
type FakeRLJob struct {
	name                   string
	client                 pb.AcceleratorOrchestratorServiceClient
	clientset              kubernetes.Interface
	iterations             int
	t                      Logger
	createdPods            []string // track created pod names for cleanup
	mu                     sync.Mutex
	podFactory             *PodFactory
	samplerTemplateKey     string
	trainerTemplateKey     string
	samplerSharedClaimName string
	trainerSharedClaimName string

	// Callbacks to control sampling/training behavior/duration
	OnSampling func(ctx context.Context)
	OnTraining func(ctx context.Context)
}

func NewFakeRLJob(
	name string,
	client pb.AcceleratorOrchestratorServiceClient,
	clientset kubernetes.Interface,
	iterations int,
	t Logger,
	samplerTemplateKey string,
	trainerTemplateKey string,
	samplerSharedClaimName string,
	trainerSharedClaimName string,
) *FakeRLJob {
	return &FakeRLJob{
		name:                   name,
		client:                 client,
		clientset:              clientset,
		iterations:             iterations,
		t:                      t,
		podFactory:             NewPodFactory(),
		samplerTemplateKey:     samplerTemplateKey,
		trainerTemplateKey:     trainerTemplateKey,
		samplerSharedClaimName: samplerSharedClaimName,
		trainerSharedClaimName: trainerSharedClaimName,
	}
}

// RegisterPodTemplate registers a pod template for a specific group in the job's factory.
func (f *FakeRLJob) RegisterPodTemplate(groupID string, pod *corev1.Pod) {
	f.podFactory.Register(groupID, pod)
}

func (f *FakeRLJob) acquireWithRetry(ctx context.Context, groupID string) (*pb.AcquireResponse, error) {
	maxRetries := 20
	retryInterval := 500 * time.Millisecond
	for i := 0; i < maxRetries; i++ {
		resp, err := f.client.Acquire(ctx, &pb.AcquireRequest{
			GroupId: groupID,
			JobId:   f.name,
		})
		if err == nil {
			return resp, nil
		}

		st, ok := status.FromError(err)
		if ok && (st.Code() == codes.Unavailable || st.Code() == codes.Internal) {
			f.t.Logf("[Job %s] WARNING: Acquire for group %s returned transient error (%v), retrying in %v (attempt %d/%d)...",
				f.name, groupID, err, retryInterval, i+1, maxRetries)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(retryInterval):
				continue
			}
		}
		return nil, err
	}
	return nil, fmt.Errorf("failed to acquire lock for %s after retries", groupID)
}

func (f *FakeRLJob) yieldWithRetry(ctx context.Context, groupID string) error {
	maxRetries := 20
	retryInterval := 500 * time.Millisecond
	for i := 0; i < maxRetries; i++ {
		_, err := f.client.Yield(ctx, &pb.YieldRequest{
			GroupId: groupID,
			JobId:   f.name,
		})
		if err == nil {
			return nil
		}

		st, ok := status.FromError(err)
		if ok && (st.Code() == codes.Unavailable || st.Code() == codes.Internal) {
			f.t.Logf("[Job %s] WARNING: Yield for group %s returned transient error (%v), retrying in %v (attempt %d/%d)...",
				f.name, groupID, err, retryInterval, i+1, maxRetries)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(retryInterval):
				continue
			}
		}
		return err
	}
	return fmt.Errorf("failed to yield lock for %s after retries", groupID)
}

func (f *FakeRLJob) Run(ctx context.Context) error {
	f.t.Logf("[Job %s] Starting RL Job", f.name)

	// 1. INIT PHASE
	if err := f.init(ctx); err != nil {
		return fmt.Errorf("init failed: %w", err)
	}

	// 2. LOOP PHASE
	if err := f.loop(ctx); err != nil {
		return fmt.Errorf("loop failed: %w", err)
	}

	// 3. CLEANUP PHASE
	if err := f.cleanup(ctx); err != nil {
		return fmt.Errorf("cleanup failed: %w", err)
	}

	f.t.Logf("[Job %s] RL Job completed successfully", f.name)
	return nil
}

func (f *FakeRLJob) init(ctx context.Context) error {
	f.t.Logf("[Job %s] Entering Init Phase", f.name)

	// Acquire lock for samplers
	f.t.Logf("[Job %s] Acquiring lock for samplers...", f.name)
	resp, err := f.acquireWithRetry(ctx, "samplers")
	if err != nil {
		return err
	}
	if !resp.Success {
		return fmt.Errorf("failed to acquire samplers lock")
	}
	f.t.Logf("[Job %s] Acquired samplers lock. Deploying sampler pods...", f.name)

	// Deploy sampler pods
	if err := f.deployPods(ctx, "samplers"); err != nil {
		return err
	}

	// Yield samplers
	f.t.Logf("[Job %s] Yielding samplers lock...", f.name)
	if err := f.yieldWithRetry(ctx, "samplers"); err != nil {
		return err
	}

	// Acquire lock for trainers
	f.t.Logf("[Job %s] Acquiring lock for trainers...", f.name)
	resp, err = f.acquireWithRetry(ctx, "trainers")
	if err != nil {
		return err
	}
	if !resp.Success {
		return fmt.Errorf("failed to acquire trainers lock")
	}
	f.t.Logf("[Job %s] Acquired trainers lock. Deploying trainer pods...", f.name)

	// Deploy trainer pods
	if err := f.deployPods(ctx, "trainers"); err != nil {
		return err
	}

	// Yield trainers
	f.t.Logf("[Job %s] Yielding trainers lock...", f.name)
	if err := f.yieldWithRetry(ctx, "trainers"); err != nil {
		return err
	}

	f.t.Logf("[Job %s] Init Phase complete", f.name)
	return nil
}

func (f *FakeRLJob) loop(ctx context.Context) error {
	f.t.Logf("[Job %s] Entering Loop Phase (%d iterations)", f.name, f.iterations)

	for i := 0; i < f.iterations; i++ {
		f.t.Logf("[Job %s] Iteration %d/%d", f.name, i+1, f.iterations)

		// 1. Lock samplers
		f.t.Logf("[Job %s] Acquiring lock for samplers...", f.name)
		resp, err := f.acquireWithRetry(ctx, "samplers")
		if err != nil {
			return err
		}
		if !resp.Success {
			return fmt.Errorf("failed to acquire samplers lock in loop")
		}

		// Sampling happening here
		f.t.Logf("[Job %s] >>> SAMPLING HAPPENING HERE <<<", f.name)
		if f.OnSampling != nil {
			f.OnSampling(ctx)
		} else {
			time.Sleep(50 * time.Millisecond) // default
		}

		// Yield samplers
		f.t.Logf("[Job %s] Yielding samplers lock...", f.name)
		if err := f.yieldWithRetry(ctx, "samplers"); err != nil {
			return err
		}

		// 2. Lock trainers
		f.t.Logf("[Job %s] Acquiring lock for trainers...", f.name)
		resp, err = f.acquireWithRetry(ctx, "trainers")
		if err != nil {
			return err
		}
		if !resp.Success {
			return fmt.Errorf("failed to acquire trainers lock in loop")
		}

		// Training happening here
		f.t.Logf("[Job %s] >>> TRAINING HAPPENING HERE <<<", f.name)
		if f.OnTraining != nil {
			f.OnTraining(ctx)
		} else {
			time.Sleep(50 * time.Millisecond) // default
		}

		// Yield trainers
		f.t.Logf("[Job %s] Yielding trainers lock...", f.name)
		if err := f.yieldWithRetry(ctx, "trainers"); err != nil {
			return err
		}
	}

	f.t.Logf("[Job %s] Loop Phase complete", f.name)
	return nil
}

func (f *FakeRLJob) cleanup(ctx context.Context) error {
	f.t.Logf("[Job %s] Entering Cleanup Phase", f.name)
	return f.cleanupPods(ctx)
}

func (f *FakeRLJob) deployPods(ctx context.Context, groupID string) error {
	// Find nodes for group
	selector := fmt.Sprintf("group.timeslice.io/%s=true", groupID)
	nodes, err := f.clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{LabelSelector: selector})
	if err != nil {
		return fmt.Errorf("failed to list nodes: %w", err)
	}

	if len(nodes.Items) == 0 {
		return fmt.Errorf("no nodes found for group %s", groupID)
	}

	for range nodes.Items {
		podName := fmt.Sprintf("pod-%s-%s-%s", f.name, groupID, uuid.NewString()[:8])

		var sharedClaimName string
		switch groupID {
		case "samplers":
			sharedClaimName = f.samplerSharedClaimName
		case "trainers":
			sharedClaimName = f.trainerSharedClaimName
		}

		// Pull pod definition from factory using the correct template key
		var templateKey string
		switch groupID {
		case "samplers":
			templateKey = f.samplerTemplateKey
		case "trainers":
			templateKey = f.trainerTemplateKey
		}
		pod := f.podFactory.GetPod(templateKey)

		// Customize for this run
		pod.Name = podName
		pod.Namespace = "default"
		pod.Spec.NodeName = "" // Let scheduler handle it

		// Set NodeSelector to target the group
		if pod.Spec.NodeSelector == nil {
			pod.Spec.NodeSelector = make(map[string]string)
		}
		pod.Spec.NodeSelector[fmt.Sprintf("group.timeslice.io/%s", groupID)] = "true"

		// Add tolerations for timeslice.io/shared and default GKE GPU taints
		pod.Spec.Tolerations = append(pod.Spec.Tolerations,
			corev1.Toleration{
				Key:      "timeslice.io/shared",
				Operator: corev1.TolerationOpEqual,
				Value:    "true",
				Effect:   corev1.TaintEffectNoSchedule,
			},
			corev1.Toleration{
				Key:      "nvidia.com/gpu",
				Operator: corev1.TolerationOpEqual,
				Value:    "present",
				Effect:   corev1.TaintEffectNoSchedule,
			},
		)

		// Reference the ResourceClaim in Pod Spec
		pod.Spec.Containers[0].Resources.Claims = []corev1.ResourceClaim{
			{
				Name: "gpu-resource",
			},
		}
		pod.Spec.ResourceClaims = []corev1.PodResourceClaim{
			{
				Name:              "gpu-resource",
				ResourceClaimName: &sharedClaimName,
			},
		}

		// Inject timeslice labels (must remain in fake_rl_job.go)
		if pod.Labels == nil {
			pod.Labels = make(map[string]string)
		}
		pod.Labels["timeslice.io/group"] = groupID
		pod.Labels["timeslice.io/job-id"] = f.name

		_, err = f.clientset.CoreV1().Pods("default").Create(ctx, pod, metav1.CreateOptions{})
		if err != nil {
			if !k8serrors.IsAlreadyExists(err) {
				return fmt.Errorf("failed to create pod %s: %w", podName, err)
			}
			f.t.Logf("[Job %s] Pod %s already exists (pre-deployed)", f.name, podName)
		} else {
			f.t.Logf("[Job %s] Deployed pod %s (pending scheduling)", f.name, podName)
		}

		// Wait for scheduling, validate node, and wait for readiness
		f.t.Logf("[Job %s] Waiting for pod %s to be scheduled and ready...", f.name, podName)
		err = wait.PollUntilContextTimeout(
			ctx, 500*time.Millisecond, 10*time.Minute, true,
			func(ctx context.Context) (bool, error) {
				pod, err := f.clientset.CoreV1().Pods("default").Get(ctx, podName, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				if pod.Spec.NodeName == "" {
					return false, nil // still pending scheduling
				}

				// Verify it is one of the expected nodes for the group
				isExpectedNode := false
				for j := range nodes.Items {
					if pod.Spec.NodeName == nodes.Items[j].Name {
						isExpectedNode = true
						break
					}
				}
				if !isExpectedNode {
					return false, fmt.Errorf(
						"pod %s scheduled to unexpected node %q (expected one of group %s nodes)",
						podName, pod.Spec.NodeName, groupID,
					)
				}

				// Verify pod is ready
				isReady := false
				for _, cond := range pod.Status.Conditions {
					if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
						isReady = true
						break
					}
				}
				if !isReady {
					return false, nil // scheduled but not ready yet
				}

				f.t.Logf("[Job %s] Pod %s is scheduled and ready on node %s", f.name, podName, pod.Spec.NodeName)
				return true, nil
			})
		if err != nil {
			return fmt.Errorf("failed to verify pod scheduling for %s: %w", podName, err)
		}

		// Always track for cleanup
		f.mu.Lock()
		f.createdPods = append(f.createdPods, podName)
		f.mu.Unlock()
	}

	return nil
}

func (f *FakeRLJob) cleanupPods(ctx context.Context) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	// 1. Delete Pods
	for _, podName := range f.createdPods {
		err := f.clientset.CoreV1().Pods("default").Delete(ctx, podName, metav1.DeleteOptions{})
		if err != nil {
			if !k8serrors.IsNotFound(err) {
				f.t.Errorf("[Job %s] Failed to delete pod %s: %v", f.name, podName, err)
			}
		} else {
			f.t.Logf("[Job %s] Deleted pod %s", f.name, podName)
		}
	}
	f.createdPods = nil

	return nil
}
