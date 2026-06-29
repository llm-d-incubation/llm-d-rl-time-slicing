// Copyright 2025 The llm-d Authors.
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

package server

import (
	"context"
	"os"
	"testing"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	sm "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/state-machine"
	podutils "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/utils"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	fakek8s "k8s.io/client-go/kubernetes/fake"
)

func TestWatcher_RegisterJobOnAdd(t *testing.T) {
	os.Setenv("NODE_NAME", "test-node")
	defer os.Unsetenv("NODE_NAME")

	fakeClient := fakek8s.NewSimpleClientset()
	podutils.GetK8sClient = func() (kubernetes.Interface, error) {
		return fakeClient, nil
	}

	state := sm.NewStateManager()
	watcher, err := NewWatcher(fakeClient, state)
	if err != nil {
		t.Fatalf("Failed to create watcher: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	watcher.Start(ctx)

	// Create a pod with the job label
	jobID := "test-job-watcher"
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "default",
			Labels: map[string]string{
				podutils.JobIDLabel: jobID,
				"timeslice.io/group": "test-group",
			},
		},
		Spec: corev1.PodSpec{
			NodeName: "test-node",
		},
	}

	_, err = fakeClient.CoreV1().Pods("default").Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// Verify job is registered as IDLE
	deadline := time.Now().Add(2 * time.Second)
	registered := false
	for time.Now().Before(deadline) {
		statuses := state.GetJobStatus()
		for _, s := range statuses {
			if s.JobId == jobID {
				if s.State != pb.JobState_JOB_STATE_IDLE {
					t.Errorf("Expected job state IDLE, got %v", s.State)
				}
				registered = true
				break
			}
		}
		if registered {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if !registered {
		t.Error("Timeout waiting for job to be registered")
	}
}

func TestWatcher_DetectionLoop(t *testing.T) {
	os.Setenv("NODE_NAME", "test-node")
	defer os.Unsetenv("NODE_NAME")

	fakeClient := fakek8s.NewSimpleClientset()
	podutils.GetK8sClient = func() (kubernetes.Interface, error) {
		return fakeClient, nil
	}

	// Mock GetPodPIDs
	mockedPIDs := []int{123, 456}
	podutils.GetPodPIDs = func(ctx context.Context, podName, namespace string) ([]int, error) {
		return mockedPIDs, nil
	}

	state := sm.NewStateManager()
	watcher, err := NewWatcher(fakeClient, state)
	if err != nil {
		t.Fatalf("Failed to create watcher: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	watcher.Start(ctx)

	// Create a pod with the job label
	jobID := "test-job-detect"
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod-detect",
			Namespace: "default",
			Labels: map[string]string{
				podutils.JobIDLabel: jobID,
			},
		},
		Spec: corev1.PodSpec{
			NodeName: "test-node",
		},
	}

	_, err = fakeClient.CoreV1().Pods("default").Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// Wait for job to transition to RUNNING
	deadline := time.Now().Add(3 * time.Second)
	running := false
	for time.Now().Before(deadline) {
		statuses := state.GetJobStatus()
		for _, s := range statuses {
			if s.JobId == jobID && s.State == pb.JobState_JOB_STATE_RUNNING {
				running = true
				break
			}
		}
		if running {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	if !running {
		t.Error("Timeout waiting for job to transition to RUNNING")
	}

	// Verify PIDs are associated
	pids, err := state.GetJobPIDs(jobID)
	if err != nil {
		t.Fatalf("Failed to get job PIDs: %v", err)
	}

	if len(pids) != 2 || pids[0] != 123 || pids[1] != 456 {
		t.Errorf("Unexpected PIDs: %v", pids)
	}
}
