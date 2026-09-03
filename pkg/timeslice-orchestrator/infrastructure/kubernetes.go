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

package infrastructure

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/logging"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/timeslice-orchestrator/controller"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/timeslice-orchestrator/store"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	corev1informers "k8s.io/client-go/informers/core/v1"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
)

const (
	// NodeLabelPrefix is deprecated: node labels no longer define group
	// membership and are ignored by the orchestrator. Kept only until the
	// rlts CLI diagnostics are migrated to the orchestrator API.
	NodeLabelPrefix = "group.timeslice.io/"
	PodLabelKey     = "timeslice.io/group"
	JobLabelKey     = "timeslice.io/job-id"
)

// PodInfo contains simplified information about a pod.
type PodInfo struct {
	UID   string
	JobID string
}

// KubernetesOrchestrator implements controller.InfrastructureOrchestrator for Kubernetes.
type KubernetesOrchestrator struct {
	nodeInformer       corev1informers.NodeInformer
	podInformer        corev1informers.PodInformer
	nodeLister         corev1listers.NodeLister
	podLister          corev1listers.PodLister
	nodeSynced         cache.InformerSynced
	podSynced          cache.InformerSynced
	groupStore         *store.GroupStore
	jobStore           *store.JobStore
	snapshotAgentStore store.SnapshotAgentStore
}

// NewKubernetesOrchestrator creates a new KubernetesOrchestrator.
func NewKubernetesOrchestrator(
	nodeInformer corev1informers.NodeInformer,
	podInformer corev1informers.PodInformer,
	groupStore *store.GroupStore,
	jobStore *store.JobStore,
	snapshotAgentStore store.SnapshotAgentStore,
) *KubernetesOrchestrator {
	return &KubernetesOrchestrator{
		nodeInformer:       nodeInformer,
		podInformer:        podInformer,
		nodeLister:         nodeInformer.Lister(),
		podLister:          podInformer.Lister(),
		nodeSynced:         nodeInformer.Informer().HasSynced,
		podSynced:          podInformer.Informer().HasSynced,
		groupStore:         groupStore,
		jobStore:           jobStore,
		snapshotAgentStore: snapshotAgentStore,
	}
}

// Init initializes the KubernetesOrchestrator by waiting for informer caches to sync.
func (k *KubernetesOrchestrator) Init(ctx context.Context) error {
	if !cache.WaitForCacheSync(ctx.Done(), k.nodeSynced, k.podSynced) {
		return fmt.Errorf("failed to wait for informer caches to sync")
	}
	return nil
}

// isTerminalPod reports whether the pod has run to completion (successfully
// or not). Terminal pods no longer host accelerator processes, so both group
// node membership and job/pod bookkeeping must ignore them; the pod objects
// themselves may linger until Kubernetes garbage-collects them.
func isTerminalPod(pod *corev1.Pod) bool {
	return pod.Status.Phase == corev1.PodSucceeded || pod.Status.Phase == corev1.PodFailed
}

// getNodesForGroup returns the names of the nodes that belong to the given
// group: a node is a member iff a live pod labeled timeslice.io/group=<group>
// is bound to it. Placement is the scheduler's decision (shared DRA claims);
// the orchestrator observes the outcome. Node labels are not consulted.
func (k *KubernetesOrchestrator) getNodesForGroup(groupID string) ([]string, error) {
	selector := labels.SelectorFromSet(labels.Set{PodLabelKey: groupID})
	pods, err := k.podLister.List(selector)
	if err != nil {
		return nil, err
	}
	nodeSet := make(map[string]bool)
	var groupNodes []string
	for _, pod := range pods {
		// Skip pods that are not scheduled yet: they contribute no hardware
		// to the group until the scheduler binds them to a node.
		if pod.Spec.NodeName == "" {
			continue
		}
		// Skip terminal pods: their accelerator processes are gone, so the
		// node no longer hosts this group's context through them.
		if isTerminalPod(pod) {
			continue
		}
		if !nodeSet[pod.Spec.NodeName] {
			nodeSet[pod.Spec.NodeName] = true
			groupNodes = append(groupNodes, pod.Spec.NodeName)
		}
	}
	return groupNodes, nil
}

// getPodsForGroup returns the live pods that are tied to the given group.
// It applies the same liveness rule as getNodesForGroup: terminal pods are
// excluded, so a job whose pods have all finished drops out of the store and
// a group whose pods have all finished becomes eligible for cleanup instead
// of lingering until Kubernetes garbage-collects the pod objects. (Unscheduled
// pods ARE included: a pending pod is a live member whose job must be
// tracked; it just contributes no node yet.)
func (k *KubernetesOrchestrator) getPodsForGroup(groupID string) ([]PodInfo, error) {
	selector := labels.SelectorFromSet(labels.Set{PodLabelKey: groupID})
	pods, err := k.podLister.List(selector)
	if err != nil {
		return nil, err
	}
	var podInfos []PodInfo
	for _, pod := range pods {
		if isTerminalPod(pod) {
			continue
		}
		jobID := pod.Labels[JobLabelKey]
		if jobID == "" {
			continue
		}
		podInfos = append(podInfos, PodInfo{
			UID:   string(pod.UID),
			JobID: jobID,
		})
	}
	return podInfos, nil
}

// ObserveGroupState observes the current state of the infrastructure for the given group
// and updates the groupStore and jobStore accordingly.
func (k *KubernetesOrchestrator) ObserveGroupState(ctx context.Context, groupID string) error {
	ctx = logging.WithGroupID(ctx, groupID)
	// 1. Find nodes belonging to the group
	groupNodes, err := k.getNodesForGroup(groupID)
	if err != nil {
		return fmt.Errorf("failed to get nodes for group %s: %w", groupID, err)
	}

	// 2. Find pods tied to the group
	pods, err := k.getPodsForGroup(groupID)
	if err != nil {
		return fmt.Errorf("failed to get pods for group %s: %w", groupID, err)
	}

	// If no nodes and no pods, clean up the group and its jobs — unless the
	// group has lock activity. Groups are created on demand by the first
	// Acquire, which in the lock-then-deploy pattern happens BEFORE any
	// member pods exist; deleting such a group would pull it out from under
	// its lock holder.
	if len(groupNodes) == 0 && len(pods) == 0 {
		if g, err := k.groupStore.Get(ctx, groupID); err == nil && g != nil {
			snap := g.Snapshot()
			if snap.LockingJob != "" || snap.ActiveJob != "" || snap.WaiterQueueDepth > 0 {
				slog.InfoContext(ctx, "Group has no nodes or pods but has lock activity; skipping cleanup",
					"lockingJob", snap.LockingJob, "activeJob", snap.ActiveJob, "waiters", snap.WaiterQueueDepth)
				// Still record the (empty) node membership so reconciliation
				// sees an up-to-date view.
				return k.updateGroupNodes(ctx, groupID, groupNodes)
			}
		}
		if err := k.cleanupGroup(ctx, groupID); err != nil {
			return fmt.Errorf("failed to cleanup group %s: %w", groupID, err)
		}
		return nil
	}

	// 3. Update group nodes in store
	if err := k.updateGroupNodes(ctx, groupID, groupNodes); err != nil {
		return err
	}

	// 4. Update jobs and their pods in store
	if err := k.updateJobsAndPods(ctx, groupID, pods); err != nil {
		return err
	}

	return nil
}

func (k *KubernetesOrchestrator) updateGroupNodes(ctx context.Context, groupID string, groupNodes []string) error {
	g, _, err := k.groupStore.GetOrCreate(ctx, groupID)
	if err != nil {
		return fmt.Errorf("failed to get or create group %s in store: %w", groupID, err)
	}

	// Clean up clients for nodes that were removed from the group
	oldNodes := g.Status().Nodes()
	removedNodes := findRemovedNodes(oldNodes, groupNodes)
	for _, nodeName := range removedNodes {
		if err := k.snapshotAgentStore.CloseClient(nodeName); err != nil {
			slog.ErrorContext(ctx, "Failed to close snapshot agent client for removed node", "error", err, "node", nodeName)
		}
	}

	g.Status().SetNodes(groupNodes)
	slog.InfoContext(ctx, "Updated nodes for group", "nodes", groupNodes)
	return nil
}

func (k *KubernetesOrchestrator) updateJobsAndPods(ctx context.Context, groupID string, pods []PodInfo) error {
	jobPods := make(map[string][]string)
	for _, pod := range pods {
		jobPods[pod.JobID] = append(jobPods[pod.JobID], pod.UID)
	}

	// Update or create jobs
	for jobID, uids := range jobPods {
		job, err := k.jobStore.Get(ctx, groupID, jobID)
		if err != nil {
			if errors.Is(err, store.ErrNotFound) {
				job = store.NewJob(groupID, jobID)
			} else {
				return err
			}
		}
		job.SetPods(uids)
		if err := k.jobStore.Put(ctx, job); err != nil {
			return err
		}
	}

	// Delete jobs that no longer have pods
	existingJobs, err := k.jobStore.ListByGroup(ctx, groupID)
	if err != nil {
		return err
	}
	for _, ej := range existingJobs {
		if _, ok := jobPods[ej.JobID()]; !ok {
			if err := k.jobStore.Delete(ctx, groupID, ej.JobID()); err != nil {
				return err
			}
			slog.InfoContext(ctx, "Deleted job from store because it has no pods", "job", ej.JobID())
		}
	}
	return nil
}

func (k *KubernetesOrchestrator) cleanupGroup(ctx context.Context, groupID string) error {
	existingJobs, err := k.jobStore.ListByGroup(ctx, groupID)
	if err != nil {
		return err
	}
	for _, ej := range existingJobs {
		if err := k.jobStore.Delete(ctx, groupID, ej.JobID()); err != nil {
			return err
		}
		slog.InfoContext(ctx, "Deleted job from store because group has no nodes and no pods", "job", ej.JobID())
	}

	// Close clients for all nodes that were in the group
	oldGroup, err := k.groupStore.Get(ctx, groupID)
	if err == nil && oldGroup != nil {
		for _, nodeName := range oldGroup.Status().Nodes() {
			if err := k.snapshotAgentStore.CloseClient(nodeName); err != nil {
				slog.ErrorContext(ctx, "Failed to close snapshot agent client on group deletion", "error", err, "node", nodeName)
			}
		}
	}

	if err := k.groupStore.Delete(ctx, groupID); err != nil {
		return err
	}
	slog.InfoContext(ctx, "Deleted group from store because it has no nodes and no pods")
	return nil
}

// Start registers event handlers on the informers.
func (k *KubernetesOrchestrator) Start(ctx context.Context, queue controller.WorkQueue) error {
	if err := k.setupNodeInformer(ctx, queue); err != nil {
		return fmt.Errorf("failed to setup node informer: %w", err)
	}
	if err := k.setupPodInformer(ctx, queue); err != nil {
		return fmt.Errorf("failed to setup pod informer: %w", err)
	}
	return nil
}

func (k *KubernetesOrchestrator) setupNodeInformer(ctx context.Context, queue controller.WorkQueue) error {
	_, err := k.nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			k.enqueueNode(ctx, obj, queue)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			k.enqueueNode(ctx, newObj, queue)
			k.enqueueNode(ctx, oldObj, queue)
		},
		DeleteFunc: func(obj interface{}) {
			k.enqueueNode(ctx, obj, queue)
		},
	})
	return err
}

func (k *KubernetesOrchestrator) enqueueNode(ctx context.Context, obj interface{}, queue controller.WorkQueue) {
	var node *corev1.Node
	var ok bool
	if node, ok = obj.(*corev1.Node); !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("error decoding object, invalid type"))
			return
		}
		node, ok = tombstone.Obj.(*corev1.Node)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("error decoding object tombstone, invalid type"))
			return
		}
	}

	slog.InfoContext(ctx, "Enqueue Node", "node", node.Name)

	groups := k.getGroupsFromNode(node)
	for _, group := range groups {
		queue.Add(group)
	}
}

// getGroupsFromNode returns the groups that have member pods bound to the
// node, so node events (e.g. deletion) requeue exactly the groups whose
// membership they affect.
func (k *KubernetesOrchestrator) getGroupsFromNode(node *corev1.Node) []string {
	// List all pods carrying the timeslice group label, then filter to this node.
	req, err := labels.NewRequirement(PodLabelKey, selection.Exists, nil)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to build pod label requirement: %w", err))
		return nil
	}
	pods, err := k.podLister.List(labels.NewSelector().Add(*req))
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to list timeslice pods for node %s: %w", node.Name, err))
		return nil
	}
	groupSet := make(map[string]bool)
	var groups []string
	for _, pod := range pods {
		if pod.Spec.NodeName != node.Name {
			continue
		}
		group := pod.Labels[PodLabelKey]
		if group != "" && !groupSet[group] {
			groupSet[group] = true
			groups = append(groups, group)
		}
	}
	return groups
}

func (k *KubernetesOrchestrator) setupPodInformer(ctx context.Context, queue controller.WorkQueue) error {
	_, err := k.podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			k.enqueuePod(ctx, obj, queue)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			k.enqueuePod(ctx, newObj, queue)
			k.enqueuePod(ctx, oldObj, queue)
		},
		DeleteFunc: func(obj interface{}) {
			k.enqueuePod(ctx, obj, queue)
		},
	})
	return err
}

func (k *KubernetesOrchestrator) enqueuePod(ctx context.Context, obj interface{}, queue controller.WorkQueue) {
	var pod *corev1.Pod
	var ok bool
	if pod, ok = obj.(*corev1.Pod); !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("error decoding object, invalid type"))
			return
		}
		pod, ok = tombstone.Obj.(*corev1.Pod)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("error decoding object tombstone, invalid type"))
			return
		}
	}

	slog.InfoContext(ctx, "Enqueue Pod", "pod", fmt.Sprintf("%s/%s", pod.Namespace, pod.Name))

	group := k.getGroupFromPod(pod)
	if group != "" {
		queue.Add(group)
	}
}

func (k *KubernetesOrchestrator) getGroupFromPod(pod *corev1.Pod) string {
	if pod.Labels == nil {
		return ""
	}
	return pod.Labels[PodLabelKey]
}

func findRemovedNodes(oldNodes, newNodes []string) []string {
	newSet := make(map[string]bool)
	for _, n := range newNodes {
		newSet[n] = true
	}
	var removed []string
	for _, o := range oldNodes {
		if !newSet[o] {
			removed = append(removed, o)
		}
	}
	return removed
}
