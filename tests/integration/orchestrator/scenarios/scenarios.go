// Copyright 2026 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package scenarios

import (
	"context"
	"fmt"
	"sync"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/timeslice-orchestrator/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
)

// RunSingleRLJobScenario runs the single RL job E2E scenario.
func RunSingleRLJobScenario(
	ctx context.Context,
	clientset kubernetes.Interface,
	client pb.TimeSliceOrchestratorServiceClient,
	logger Logger,
	samplerTemplateKey string,
	trainerTemplateKey string,
) error {
	logger.Log("Starting Single RL Job Scenario")

	samplerClaim := "claim-shared-single-samplers"
	trainerClaim := "claim-shared-single-trainers"

	if err := createSharedClaim(ctx, clientset, samplerClaim); err != nil {
		return err
	}
	defer func() {
		if err := deleteSharedClaim(ctx, clientset, samplerClaim); err != nil {
			logger.Errorf("Failed to delete shared claim %s: %v", samplerClaim, err)
		}
	}()

	if err := createSharedClaim(ctx, clientset, trainerClaim); err != nil {
		return err
	}
	defer func() {
		if err := deleteSharedClaim(ctx, clientset, trainerClaim); err != nil {
			logger.Errorf("Failed to delete shared claim %s: %v", trainerClaim, err)
		}
	}()

	// Run Fake RL Job
	job := NewFakeRLJob(
		"my-rl-job", client, clientset, 2, logger,
		samplerTemplateKey, trainerTemplateKey,
		samplerClaim, trainerClaim,
	)

	// Set custom work durations
	job.OnSampling = func(ctx context.Context) {
		logger.Log("Custom sampling work (10ms)...")
		time.Sleep(10 * time.Millisecond)
	}
	job.OnTraining = func(ctx context.Context) {
		logger.Log("Custom training work (10ms)...")
		time.Sleep(10 * time.Millisecond)
	}

	// Run the job. It should complete without error.
	if err := job.Run(ctx); err != nil {
		return fmt.Errorf("fake RL Job failed: %w", err)
	}

	// Verify Post-Cleanup State: All pods created by this job should be deleted
	err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
		pods, err := clientset.CoreV1().Pods("default").List(ctx, metav1.ListOptions{
			LabelSelector: "timeslice.io/job-id=my-rl-job",
		})
		if err != nil {
			return false, err
		}
		return len(pods.Items) == 0, nil
	})
	if err != nil {
		return fmt.Errorf("timed out waiting for pods cleanup: %w", err)
	}

	logger.Log("Single RL Job Scenario completed successfully")
	return nil
}

// RunQueuedRLJobsScenario runs the queued RL jobs contention scenario.
func RunQueuedRLJobsScenario(
	ctx context.Context,
	clientset kubernetes.Interface,
	client pb.TimeSliceOrchestratorServiceClient,
	logger Logger,
	samplerTemplateKey string,
	trainerTemplateKey string,
) error {
	logger.Log("Starting Queued RL Jobs Scenario")

	samplerClaim := "claim-shared-queued-samplers"
	trainerClaim := "claim-shared-queued-trainers"

	if err := createSharedClaim(ctx, clientset, samplerClaim); err != nil {
		return err
	}
	defer func() {
		if err := deleteSharedClaim(ctx, clientset, samplerClaim); err != nil {
			logger.Errorf("Failed to delete shared claim %s: %v", samplerClaim, err)
		}
	}()

	if err := createSharedClaim(ctx, clientset, trainerClaim); err != nil {
		return err
	}
	defer func() {
		if err := deleteSharedClaim(ctx, clientset, trainerClaim); err != nil {
			logger.Errorf("Failed to delete shared claim %s: %v", trainerClaim, err)
		}
	}()

	jobA := NewFakeRLJob(
		"job-a", client, clientset, 2, logger,
		samplerTemplateKey, trainerTemplateKey,
		samplerClaim, trainerClaim,
	)
	jobB := NewFakeRLJob(
		"job-b", client, clientset, 2, logger,
		samplerTemplateKey, trainerTemplateKey,
		samplerClaim, trainerClaim,
	)

	// Channels for coordination
	jobASampling := make(chan struct{})
	unblockJobA := make(chan struct{})

	var coordOnce sync.Once
	// Configure Job A callbacks to block during sampling
	jobA.OnSampling = func(ctx context.Context) {
		coordOnce.Do(func() {
			logger.Log("[Test] Job A is sampling, notifying test and blocking...")
			close(jobASampling) // Notify test
			select {
			case <-unblockJobA:
				logger.Log("[Test] Job A unblocked, finishing sampling...")
			case <-ctx.Done():
				logger.Log("[Test] Job A context cancelled while blocked")
			}
		})
	}
	jobA.OnTraining = func(ctx context.Context) {
		logger.Log("[Test] Job A training (10ms)...")
		time.Sleep(10 * time.Millisecond)
	}

	// Configure Job B callbacks to just run quickly
	jobB.OnSampling = func(ctx context.Context) {
		logger.Log("[Test] Job B sampling (10ms)...")
		time.Sleep(10 * time.Millisecond)
	}
	jobB.OnTraining = func(ctx context.Context) {
		logger.Log("[Test] Job B training (10ms)...")
		time.Sleep(10 * time.Millisecond)
	}

	// Start Job A in background
	jobAErr := make(chan error, 1)
	go func() {
		err := jobA.Run(ctx)
		if err != nil {
			logger.Errorf("[Test] Job A exited with error: %v", err)
		}
		jobAErr <- err
	}()

	// Wait for Job A to reach sampling phase (holding samplers lock)
	select {
	case <-jobASampling:
		logger.Log("[Test] Confirmed Job A is holding samplers lock")
	case <-time.After(10 * time.Minute):
		return fmt.Errorf("timed out waiting for Job A to start sampling")
	}

	// Start Job B in background. It should block in Init trying to acquire samplers lock.
	jobBErr := make(chan error, 1)
	go func() {
		err := jobB.Run(ctx)
		if err != nil {
			logger.Errorf("[Test] Job B exited with error: %v", err)
		}
		jobBErr <- err
	}()

	// Give Job B a moment to run and block on the lock
	time.Sleep(1 * time.Second)

	// Verify that Job B is queued behind Job A in the samplers group
	resp, err := client.GetGroupStatus(ctx, &pb.GetGroupStatusRequest{GroupId: "samplers"})
	if err != nil {
		return fmt.Errorf("failed to get samplers group status: %w", err)
	}
	g := resp.Group
	if g.LockingJob != "job-a" {
		return fmt.Errorf("expected lockingJob to be job-a, got %q", g.LockingJob)
	}
	if g.WaiterQueueDepth != 1 {
		return fmt.Errorf("expected waiter queue depth to be 1 (job-b waiting), got %d", g.WaiterQueueDepth)
	}
	if g.LoadedJob != "job-a" {
		return fmt.Errorf("expected loadedJob to be job-a, got %q", g.LoadedJob)
	}
	logger.Log("[Test] Confirmed Job B is queued behind Job A")

	// Unblock Job A. This should allow Job A to finish, yield, and Job B to acquire the lock.
	logger.Log("[Test] Unblocking Job A...")
	close(unblockJobA)

	// Wait for both jobs to complete
	select {
	case err := <-jobAErr:
		if err != nil {
			return fmt.Errorf("job A failed: %w", err)
		}
	case <-time.After(10 * time.Minute):
		return fmt.Errorf("timed out waiting for Job A to complete")
	}

	select {
	case err := <-jobBErr:
		if err != nil {
			return fmt.Errorf("job B failed: %w", err)
		}
	case <-time.After(10 * time.Minute):
		return fmt.Errorf("timed out waiting for Job B to complete")
	}

	// Verify Post-Cleanup State (only pods for these jobs)
	err = wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
		pods, err := clientset.CoreV1().Pods("default").List(ctx, metav1.ListOptions{
			LabelSelector: "timeslice.io/job-id in (job-a, job-b)",
		})
		if err != nil {
			return false, err
		}
		return len(pods.Items) == 0, nil
	})
	if err != nil {
		return fmt.Errorf("timed out waiting for pods cleanup: %w", err)
	}

	logger.Log("Queued RL Jobs Scenario completed successfully")
	return nil
}

//nolint:gocritic
func createSharedClaim(ctx context.Context, clientset kubernetes.Interface, name string) error {
	claim := &resourcev1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
		},
		Spec: resourcev1.ResourceClaimSpec{
			Devices: resourcev1.DeviceClaim{
				Requests: []resourcev1.DeviceRequest{
					{
						Name: "gpu",
						Exactly: &resourcev1.ExactDeviceRequest{
							DeviceClassName: "gpu.nvidia.com",
							AllocationMode:  resourcev1.DeviceAllocationModeExactCount,
							Count:           1,
						},
					},
				},
			},
		},
	}
	_, err := clientset.ResourceV1().ResourceClaims("default").Create(ctx, claim, metav1.CreateOptions{})
	if err != nil && !k8serrors.IsAlreadyExists(err) {
		return fmt.Errorf("failed to create shared claim %s: %w", name, err)
	}
	return nil
}

//nolint:gocritic
func deleteSharedClaim(ctx context.Context, clientset kubernetes.Interface, name string) error {
	err := clientset.ResourceV1().ResourceClaims("default").Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil && !k8serrors.IsNotFound(err) {
		return fmt.Errorf("failed to delete shared claim %s: %w", name, err)
	}
	return nil
}

// ExecInPod runs a command inside a scenario pod's container; wired to the
// harness's SPDY exec helper by the test that drives the scenario.
type ExecInPod func(ctx context.Context, podName, container string, command ...string) (string, error)

// killableBurnerPod is the default GPU burner restructured so the GPU
// process can be killed WITHOUT the pod dying: a shell stays as PID 1,
// records the python PID, and keeps the pod alive after it exits. This
// reproduces a real workload teardown (Ray drivers exit, worker pods
// linger) where the accelerator processes are gone but the pod - and with
// it the platform's job bookkeeping - remains.
func killableBurnerPod() *corev1.Pod {
	gracePeriodSec := int64(2)
	script := `cat > /tmp/burn.py <<'PYEOF'
import torch, time
total = torch.cuda.get_device_properties(0).total_memory
alloc = int(total * 0.5)
print(f"Allocating 50% VRAM: {alloc / 1e9:.2f} GB")
t = torch.zeros(alloc // 4, dtype=torch.float32, device='cuda')
torch.cuda.synchronize()
print("holding VRAM")
while True:
    time.sleep(1)
PYEOF
python3 /tmp/burn.py &
echo $! > /tmp/gpu.pid
wait $(cat /tmp/gpu.pid)
echo "GPU process exited; keeping pod alive"
sleep infinity`
	return &corev1.Pod{
		Spec: corev1.PodSpec{
			TerminationGracePeriodSeconds: &gracePeriodSec,
			Containers: []corev1.Container{
				{
					Name:            "pytorch-container",
					Image:           "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
					Command:         []string{"sh", "-c"},
					Args:            []string{script},
					ImagePullPolicy: corev1.PullIfNotPresent,
				},
			},
		},
	}
}

// waitAgentJobState polls GetGroupStatus until the given job reports the
// wanted state on at least one agent of the group.
func waitAgentJobState(
	ctx context.Context,
	client pb.TimeSliceOrchestratorServiceClient,
	groupID, jobID string,
	want pb.SnapshotAgentJobState_State,
	timeout time.Duration,
) error {
	return wait.PollUntilContextTimeout(ctx, 2*time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		resp, err := client.GetGroupStatus(ctx, &pb.GetGroupStatusRequest{GroupId: groupID})
		if err != nil {
			//nolint:nilerr // the group may not exist yet (NotFound); keep polling until timeout
			return false, nil
		}
		for _, s := range resp.AgentJobStates {
			if s.JobId == jobID && s.JobState == want {
				return true, nil
			}
		}
		return false, nil
	})
}

// RunTeardownRaceScenario reproduces the end-of-run teardown race
// (issue #164): a job releases its group lock with no waiters (the snapshot
// is deferred, so the agent keeps reporting it RUNNING), then its
// accelerator processes exit while its pods linger. The next job's Acquire
// triggers lazy eviction of this ghost - a checkpoint of processes that no
// longer exist. Without the fix, PID discovery fails the snapshot, the dead
// job is marked FAULTED, and the survivor's Acquire fails with
// "group samplers is faulted"; with the fix, the eviction completes, the
// ghost settles to IDLE, and the survivor is granted the lock.
func RunTeardownRaceScenario(
	ctx context.Context,
	clientset kubernetes.Interface,
	client pb.TimeSliceOrchestratorServiceClient,
	logger Logger,
	execInPod ExecInPod,
) error {
	logger.Log("Starting Teardown Race Scenario")

	const ghostJob = "ghost-job"
	const survivorJob = "survivor-job"
	samplerClaim := "claim-shared-teardown-samplers"

	if err := createSharedClaim(ctx, clientset, samplerClaim); err != nil {
		return err
	}
	defer func() {
		if err := deleteSharedClaim(ctx, clientset, samplerClaim); err != nil {
			logger.Errorf("Failed to delete shared claim %s: %v", samplerClaim, err)
		}
	}()

	jobA := NewFakeRLJob(
		ghostJob, client, clientset, 0, logger,
		"killable", "", samplerClaim, "",
	)
	jobA.RegisterPodTemplate("killable", killableBurnerPod())
	defer func() {
		if err := jobA.cleanupPods(context.WithoutCancel(ctx)); err != nil {
			logger.Errorf("Failed to clean up ghost job pods: %v", err)
		}
	}()

	// 1. The ghost job acquires samplers and deploys its GPU workload.
	logger.Logf("[Test] %s acquiring samplers...", ghostJob)
	resp, err := jobA.acquireWithRetry(ctx, "samplers")
	if err != nil {
		return fmt.Errorf("ghost job failed to acquire samplers: %w", err)
	}
	if !resp.Success {
		return fmt.Errorf("ghost job samplers acquire was not successful")
	}
	if err := jobA.deployPods(ctx, "samplers"); err != nil {
		return fmt.Errorf("ghost job failed to deploy sampler pod: %w", err)
	}

	// 2. Wait until the agent observes the workload on the accelerator.
	logger.Logf("[Test] Waiting for %s to be observed RUNNING by the agent...", ghostJob)
	if err := waitAgentJobState(ctx, client, "samplers", ghostJob,
		pb.SnapshotAgentJobState_STATE_RUNNING, 6*time.Minute); err != nil {
		return fmt.Errorf("ghost job never observed RUNNING: %w", err)
	}

	// 3. Final release with NO waiters: the snapshot is deferred, so the
	// agent keeps reporting the job RUNNING.
	logger.Logf("[Test] %s releasing samplers (no waiters - snapshot deferred)...", ghostJob)
	if err := jobA.yieldWithRetry(ctx, "samplers"); err != nil {
		return fmt.Errorf("ghost job failed to yield samplers: %w", err)
	}

	// 4. The workload exits (killed) while the pod - and the platform's job
	// bookkeeping - lingers. This is the moment a real job's driver exits.
	jobA.mu.Lock()
	podName := jobA.createdPods[0]
	jobA.mu.Unlock()
	logger.Logf("[Test] Killing the GPU process in pod %s (pod stays alive)...", podName)
	out, err := execInPod(ctx, podName, "pytorch-container",
		"sh", "-c", "kill -9 $(cat /tmp/gpu.pid) && echo killed")
	if err != nil {
		return fmt.Errorf("failed to kill GPU process in %s: %w (output: %s)", podName, err, out)
	}

	// 5. Confirm the ghost: the agent still reports the dead job RUNNING.
	time.Sleep(5 * time.Second)
	statusResp, err := client.GetGroupStatus(ctx, &pb.GetGroupStatusRequest{GroupId: "samplers"})
	if err != nil {
		return fmt.Errorf("failed to get samplers status after kill: %w", err)
	}
	ghostSeen := false
	for _, s := range statusResp.AgentJobStates {
		if s.JobId == ghostJob && s.JobState == pb.SnapshotAgentJobState_STATE_RUNNING {
			ghostSeen = true
		}
	}
	if !ghostSeen {
		return fmt.Errorf("expected stale RUNNING record for %s after its processes exited (ghost precondition)", ghostJob)
	}
	logger.Logf("[Test] Confirmed ghost: %s processes are dead but the agent still reports RUNNING", ghostJob)

	// 6. The survivor acquires: this triggers lazy eviction of the ghost.
	// Without the fix this fails with "group samplers is faulted".
	logger.Logf("[Test] %s acquiring samplers (triggers lazy eviction of the ghost)...", survivorJob)
	acquireCtx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()
	if _, err := client.Acquire(acquireCtx, &pb.AcquireRequest{
		GroupId: "samplers",
		JobId:   survivorJob,
	}); err != nil {
		return fmt.Errorf(
			"teardown race: survivor's Acquire failed after the ghost job's processes exited "+
				"(lazy eviction of the dead job likely FAULTED the group): %w", err)
	}
	logger.Logf("[Test] %s acquired samplers - ghost eviction handled", survivorJob)

	// 7. The ghost must have settled to IDLE (evicted with nothing to save).
	if err := waitAgentJobState(ctx, client, "samplers", ghostJob,
		pb.SnapshotAgentJobState_STATE_IDLE, time.Minute); err != nil {
		return fmt.Errorf("ghost job did not settle to IDLE after eviction: %w", err)
	}

	// 8. Release the survivor's lock so later scenarios start clean.
	if _, err := client.Yield(ctx, &pb.YieldRequest{GroupId: "samplers", JobId: survivorJob}); err != nil {
		logger.Errorf("Failed to yield survivor lock: %v", err)
	}

	logger.Log("Teardown Race Scenario completed successfully")
	return nil
}
