package controller

import (
	"context"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	corev1informers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

func (c *Controller) setupNodeInformer(nodeInformer corev1informers.NodeInformer) {
	_, err := nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.enqueueNode(obj)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			c.enqueueNode(newObj)
			c.enqueueNode(oldObj)
		},
		DeleteFunc: func(obj interface{}) {
			c.enqueueNode(obj)
			c.cleanupNodeClient(obj)
		},
	})
	if err != nil {
		panic(fmt.Errorf("failed to add node event handler: %w", err))
	}
}

func (c *Controller) enqueueNode(obj interface{}) {
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

	logger := klog.FromContext(context.Background())
	logger.Info("Enqueue Node", "node", node.Name)

	groups := getGroupsFromNode(node)
	for _, group := range groups {
		c.queue.Add(group)
	}
}

func getGroupsFromNode(node *corev1.Node) []string {
	var groups []string
	for k := range node.Labels {
		if strings.HasPrefix(k, NodeLabelPrefix) {
			group := strings.TrimPrefix(k, NodeLabelPrefix)
			if group != "" {
				groups = append(groups, group)
			}
		}
	}
	return groups
}

func (c *Controller) cleanupNodeClient(obj interface{}) {
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

	logger := klog.FromContext(context.Background())
	logger.Info("Node deleted, cleaning up snapshot agent client", "node", node.Name)
	if err := c.snapshotAgentStore.CloseClient(node.Name); err != nil {
		logger.Error(err, "Failed to close snapshot agent client", "node", node.Name)
	}
}
