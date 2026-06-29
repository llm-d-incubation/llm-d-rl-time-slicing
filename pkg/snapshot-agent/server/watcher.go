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
	"fmt"
	"log/slog"
	"os"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	sm "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/state-machine"
	podutils "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/utils"
)

// Watcher watches pods on the local node and manages job state transitions.
type Watcher struct {
	clientset kubernetes.Interface
	state     *sm.StateManager
	nodeName  string
	informer  cache.SharedIndexInformer
}

// NewWatcher creates a new Watcher instance.
func NewWatcher(clientset kubernetes.Interface, state *sm.StateManager) (*Watcher, error) {
	nodeName := os.Getenv("NODE_NAME")
	if nodeName == "" {
		return nil, fmt.Errorf("NODE_NAME environment variable not set")
	}

	// Create an informer filtered by node name
	factory := informers.NewSharedInformerFactoryWithOptions(clientset, 0,
		informers.WithTweakListOptions(func(options *metav1.ListOptions) {
			options.FieldSelector = fmt.Sprintf("spec.nodeName=%s", nodeName)
		}),
	)
	podInformer := factory.Core().V1().Pods().Informer()

	w := &Watcher{
		clientset: clientset,
		state:     state,
		nodeName:  nodeName,
		informer:  podInformer,
	}

	_, err := podInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: w.handlePodAdd,
		UpdateFunc: w.handlePodUpdate,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to add event handler to informer: %w", err)
	}

	return w, nil
}

// Start starts the watcher and the GPU detection loop.
func (w *Watcher) Start(ctx context.Context) {
	slog.Info("Starting pod watcher", "nodeName", w.nodeName)
	go w.informer.Run(ctx.Done())
	if !cache.WaitForCacheSync(ctx.Done(), w.informer.HasSynced) {
		slog.Error("Failed to sync informer cache")
		return
	}

	// Start the GPU detection loop
	go w.detectionLoop(ctx)
}

func (w *Watcher) handlePodAdd(obj interface{}) {
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		return
	}
	w.registerPodJob(pod)
}

func (w *Watcher) handlePodUpdate(oldObj, newObj interface{}) {
	pod, ok := newObj.(*corev1.Pod)
	if !ok {
		return
	}
	w.registerPodJob(pod)
}

func (w *Watcher) registerPodJob(pod *corev1.Pod) {
	jobID, hasJob := pod.Labels[podutils.JobIDLabel]
	if !hasJob {
		return
	}

	// Group can be optional, check if there is a group label
	group := pod.Labels["timeslice.io/group"]

	slog.Info("Detected pod for job", "pod", pod.Name, "jobID", jobID, "group", group)
	w.state.RegisterJob(jobID, group)
}

func (w *Watcher) detectionLoop(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			w.checkIdleJobs(ctx)
		}
	}
}

func (w *Watcher) checkIdleJobs(ctx context.Context) {
	statuses := w.state.GetJobStatus()
	for _, status := range statuses {
		if status.State != pb.JobState_JOB_STATE_IDLE {
			continue
		}

		// Find pods for this job on this node
		pods := w.getLocalPodsForJob(status.JobId)
		for _, pod := range pods {
			pids, err := podutils.GetPodPIDs(ctx, pod.Name, pod.Namespace)
			if err != nil {
				// Don't log spammy errors if NVML is not initialized or no devices
				slog.Debug("Failed to get pod PIDs", "pod", pod.Name, "error", err)
				continue
			}

			if len(pids) > 0 {
				slog.Info("Detected GPU activity for job, transitioning to RUNNING", "jobID", status.JobId, "pids", pids)
				if err := w.state.TransitionToRunning(status.JobId, pids); err != nil {
					slog.Error("Failed to transition job to RUNNING", "jobID", status.JobId, "error", err)
				}
				break // Transitioned, no need to check other pods for this job
			}
		}
	}
}

func (w *Watcher) getLocalPodsForJob(jobID string) []*corev1.Pod {
	var localPods []*corev1.Pod
	objs := w.informer.GetStore().List()
	for _, obj := range objs {
		pod, ok := obj.(*corev1.Pod)
		if !ok {
			continue
		}
		if pod.Labels[podutils.JobIDLabel] == jobID {
			localPods = append(localPods, pod)
		}
	}
	return localPods
}
