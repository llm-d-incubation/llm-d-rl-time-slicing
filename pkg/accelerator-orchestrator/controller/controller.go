package controller

import (
	"context"
	"fmt"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	corev1informers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

const (
	NodeLabelPrefix = "group.timeslice.io/"
	PodLabelKey     = "timeslice.io/group"
)

// DesiredState represents the desired state of the group.
type DesiredState struct {
	Name string
	// Add other fields as needed
}

// ActualState represents the actual state of the group.
type ActualState struct {
	Name  string
	// Add other fields as needed (e.g., associated nodes, pods)
}

// Controller implements the reconciliation loop for groups.
type Controller struct {
	clientset kubernetes.Interface
	queue     workqueue.RateLimitingInterface

	groupStore         *store.GroupStore
	jobStore           *store.JobStore
	snapshotAgentStore store.SnapshotAgentStore

	nodeLister corev1listers.NodeLister
	nodeSynced cache.InformerSynced
	podLister  corev1listers.PodLister
	podSynced  cache.InformerSynced
}

// NewController creates a new Controller.
func NewController(
	clientset kubernetes.Interface,
	nodeInformer corev1informers.NodeInformer,
	podInformer corev1informers.PodInformer,
	groupStore *store.GroupStore,
	jobStore *store.JobStore,
	snapshotAgentStore store.SnapshotAgentStore,
) *Controller {
	c := &Controller{
		clientset:          clientset,
		queue:              workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "groups"),
		groupStore:         groupStore,
		jobStore:           jobStore,
		snapshotAgentStore: snapshotAgentStore,
		nodeLister:         nodeInformer.Lister(),
		nodeSynced:         nodeInformer.Informer().HasSynced,
		podLister:          podInformer.Lister(),
		podSynced:          podInformer.Informer().HasSynced,
	}

	c.setupNodeInformer(nodeInformer)
	c.setupPodInformer(podInformer)

	return c
}

// Run starts the controller.
func (c *Controller) Run(ctx context.Context, workers int) error {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	logger := klog.FromContext(ctx)
	logger.Info("Starting Group controller")

	logger.Info("Waiting for informer caches to sync")
	if ok := cache.WaitForCacheSync(ctx.Done(), c.nodeSynced, c.podSynced); !ok {
		return fmt.Errorf("failed to wait for caches to sync")
	}

	logger.Info("Starting workers")
	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, c.runWorker, time.Second)
	}

	logger.Info("Started workers")
	<-ctx.Done()
	logger.Info("Shutting down workers")

	return nil
}

func (c *Controller) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *Controller) processNextWorkItem(ctx context.Context) bool {
	obj, shutdown := c.queue.Get()
	if shutdown {
		return false
	}

	logger := klog.FromContext(ctx)

	err := func(obj interface{}) error {
		defer c.queue.Done(obj)
		var key string
		var ok bool
		if key, ok = obj.(string); !ok {
			c.queue.Forget(obj)
			utilruntime.HandleError(fmt.Errorf("expected string in workqueue but got %#v", obj))
			return nil
		}

		cycleLogger := logger.WithValues("group", key)
		cycleCtx := klog.NewContext(ctx, cycleLogger)

		if err := c.reconcileGroup(cycleCtx, key); err != nil {
			c.queue.AddRateLimited(obj)
			return fmt.Errorf("error syncing '%s': %s, requeuing", key, err.Error())
		}
		c.queue.Forget(obj)
		cycleLogger.Info("Successfully synced group")
		return nil
	}(obj)

	if err != nil {
		utilruntime.HandleError(err)
		return true
	}

	return true
}

func (c *Controller) reconcileGroup(ctx context.Context, key string) error {
	logger := klog.FromContext(ctx)
	logger.Info("Reconciling group")

	// 1. Observe Current State
	if err := c.observeGroupState(ctx, key); err != nil {
		return fmt.Errorf("failed to observe group state: %w", err)
	}

	// 2. Determine Desired State (TODO)

	// 3. Act (TODO)

	// 4. Update Status (TODO)

	return nil
}

func (c *Controller) observeGroupState(ctx context.Context, groupName string) error {
	logger := klog.FromContext(ctx)

	// 1. Find nodes belonging to the group
	selector := labels.SelectorFromSet(labels.Set{NodeLabelPrefix + groupName: "true"})
	nodes, err := c.nodeLister.List(selector)
	if err != nil {
		return fmt.Errorf("failed to list nodes for group %s: %w", groupName, err)
	}

	var groupNodes []string
	for _, node := range nodes {
		groupNodes = append(groupNodes, node.Name)
	}

	// 2. Find pods tied to the group
	selector = labels.SelectorFromSet(labels.Set{PodLabelKey: groupName})
	pods, err := c.podLister.List(selector)
	if err != nil {
		return fmt.Errorf("failed to list pods for group %s: %w", groupName, err)
	}

	// If no nodes and no pods, we clean up the group and its jobs and return early.
	if len(groupNodes) == 0 && len(pods) == 0 {
		existingJobs, err := c.jobStore.ListByGroup(ctx, groupName)
		if err != nil {
			return err
		}
		for _, ej := range existingJobs {
			if err := c.jobStore.Delete(ctx, groupName, ej.JobID()); err != nil {
				return err
			}
			logger.Info("Deleted job from store because group has no nodes and no pods", "job", ej.JobID())
		}

		if err := c.groupStore.Delete(ctx, groupName); err != nil {
			return fmt.Errorf("failed to delete group %s: %w", groupName, err)
		}
		logger.Info("Deleted group from store because it has no nodes and no pods", "group", groupName)
		return nil
	}

	g, _, err := c.groupStore.GetOrCreate(ctx, groupName)
	if err != nil {
		return fmt.Errorf("failed to get or create group in store: %w", err)
	}
	g.SetNodes(groupNodes)
	logger.Info("Updated nodes for group", "nodes", groupNodes)

	jobPods := make(map[string][]string)
	for _, pod := range pods {
		jobID := pod.Annotations["timeslice.io/job-id"]
		if jobID == "" {
			jobID = pod.Labels["timeslice.io/job-id"]
		}
		if jobID == "" {
			continue
		}
		jobPods[jobID] = append(jobPods[jobID], string(pod.UID))
	}

	// Update or create jobs
	for jobID, uids := range jobPods {
		job, err := c.jobStore.Get(ctx, groupName, jobID)
		if err != nil {
			if err == store.ErrNotFound {
				job = store.NewJob(groupName, jobID)
			} else {
				return err
			}
		}
		job.SetPods(uids)
		if err := c.jobStore.Put(ctx, job); err != nil {
			return err
		}
	}

	// Delete jobs that no longer have pods
	existingJobs, err := c.jobStore.ListByGroup(ctx, groupName)
	if err != nil {
		return err
	}
	for _, ej := range existingJobs {
		if _, ok := jobPods[ej.JobID()]; !ok {
			if err := c.jobStore.Delete(ctx, groupName, ej.JobID()); err != nil {
				return err
			}
			logger.Info("Deleted job from store because it has no pods", "job", ej.JobID())
		}
	}

	// 3. Call snapshotagentstore status for every node in the group & populate job context
	for _, nodeName := range groupNodes {
		resp, err := c.snapshotAgentStore.GetStatus(ctx, nodeName)
		if err != nil {
			logger.Error(err, "Failed to get status from snapshot agent", "node", nodeName)
			continue
		}

		for _, js := range resp.JobStatuses {
			// Only update if the job is known in this group
			_, err := c.jobStore.Get(ctx, groupName, js.JobId)
			if err == store.ErrNotFound {
				continue
			} else if err != nil {
				return err
			}

			state := translateJobState(js.State)
			if err := c.jobStore.UpdateContextState(ctx, groupName, js.JobId, nodeName, state); err != nil {
				return err
			}
			logger.Info("Updated job context state", "job", js.JobId, "node", nodeName, "state", state)
		}
	}

	return nil
}

func translateJobState(s agentpb.JobState) pb.SnapshotAgentJobState_State {
	switch s {
	case agentpb.JobState_JOB_STATE_IDLE:
		return pb.SnapshotAgentJobState_STATE_IDLE
	case agentpb.JobState_JOB_STATE_RUNNING:
		return pb.SnapshotAgentJobState_STATE_RUNNING
	case agentpb.JobState_JOB_STATE_TRANSITIONING:
		return pb.SnapshotAgentJobState_STATE_TRANSITIONING
	case agentpb.JobState_JOB_STATE_SAVED:
		return pb.SnapshotAgentJobState_STATE_SAVED
	case agentpb.JobState_JOB_STATE_FAULTED:
		return pb.SnapshotAgentJobState_STATE_FAULTED
	default:
		return pb.SnapshotAgentJobState_STATE_UNSPECIFIED
	}
}
