package acceleratororchestrator

import (
	"context"
	"fmt"
	"sync"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
	name        string
	client      pb.AcceleratorOrchestratorServiceClient
	clientset   kubernetes.Interface
	iterations  int
	t           Logger
	createdPods []string // track created pod names for cleanup
	mu          sync.Mutex
	podFactory  *PodFactory

	// Callbacks to control sampling/training behavior/duration
	OnSampling func(ctx context.Context)
	OnTraining func(ctx context.Context)
}

func NewFakeRLJob(name string, client pb.AcceleratorOrchestratorServiceClient, clientset kubernetes.Interface, iterations int, t Logger) *FakeRLJob {
	return &FakeRLJob{
		name:       name,
		client:     client,
		clientset:  clientset,
		iterations: iterations,
		t:          t,
		podFactory: NewPodFactory(),
	}
}

// RegisterPodTemplate registers a pod template for a specific group in the job's factory.
func (f *FakeRLJob) RegisterPodTemplate(groupID string, pod *corev1.Pod) {
	f.podFactory.Register(groupID, pod)
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
	resp, err := f.client.Acquire(ctx, &pb.AcquireRequest{
		GroupId: "samplers",
		JobId:   f.name,
	})
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
	_, err = f.client.Yield(ctx, &pb.YieldRequest{
		GroupId: "samplers",
		JobId:   f.name,
	})
	if err != nil {
		return err
	}

	// Acquire lock for trainers
	f.t.Logf("[Job %s] Acquiring lock for trainers...", f.name)
	resp, err = f.client.Acquire(ctx, &pb.AcquireRequest{
		GroupId: "trainers",
		JobId:   f.name,
	})
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
	_, err = f.client.Yield(ctx, &pb.YieldRequest{
		GroupId: "trainers",
		JobId:   f.name,
	})
	if err != nil {
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
		resp, err := f.client.Acquire(ctx, &pb.AcquireRequest{
			GroupId: "samplers",
			JobId:   f.name,
		})
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
		_, err = f.client.Yield(ctx, &pb.YieldRequest{
			GroupId: "samplers",
			JobId:   f.name,
		})
		if err != nil {
			return err
		}

		// 2. Lock trainers
		f.t.Logf("[Job %s] Acquiring lock for trainers...", f.name)
		resp, err = f.client.Acquire(ctx, &pb.AcquireRequest{
			GroupId: "trainers",
			JobId:   f.name,
		})
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
		_, err = f.client.Yield(ctx, &pb.YieldRequest{
			GroupId: "trainers",
			JobId:   f.name,
		})
		if err != nil {
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

	for _, node := range nodes.Items {
		podName := fmt.Sprintf("pod-%s-%s-%s", f.name, groupID, node.Name)
		
		// Pull pod definition from factory
		pod := f.podFactory.GetPod(groupID)
		
		// Customize for this run
		pod.Name = podName
		pod.Namespace = "default"
		pod.Spec.NodeName = node.Name
		
		// Inject timeslice labels (must remain in fake_rl_job.go)
		if pod.Labels == nil {
			pod.Labels = make(map[string]string)
		}
		pod.Labels["timeslice.io/group"] = groupID
		pod.Labels["timeslice.io/job-id"] = f.name

		_, err := f.clientset.CoreV1().Pods("default").Create(ctx, pod, metav1.CreateOptions{})
		if err != nil {
			if !k8serrors.IsAlreadyExists(err) {
				return fmt.Errorf("failed to create pod %s: %w", podName, err)
			}
			f.t.Logf("[Job %s] Pod %s already exists (pre-deployed)", f.name, podName)
		} else {
			f.t.Logf("[Job %s] Deployed pod %s on node %s", f.name, podName, node.Name)
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
