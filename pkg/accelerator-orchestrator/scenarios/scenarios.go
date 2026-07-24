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

// Package scenarios contains the accelerator-orchestrator E2E scenario
// drivers. It is the single implementation shared by the rlts CLI
// (`rlts test orchestrator`) and the Go test suites: scenarios take a gRPC
// client and a Kubernetes clientset, report progress through the Logger
// interface, and return errors rather than printing or exiting.
package scenarios

import (
	"context"
	"fmt"
	"sync"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
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
	client pb.AcceleratorOrchestratorServiceClient,
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
	err := wait.PollUntilContextTimeout(ctx, 200*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
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
	client pb.AcceleratorOrchestratorServiceClient,
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
	err = wait.PollUntilContextTimeout(ctx, 200*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
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
