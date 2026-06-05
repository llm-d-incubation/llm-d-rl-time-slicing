package controller

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	corev1informers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

func (c *Controller) setupPodInformer(podInformer corev1informers.PodInformer) {
	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.enqueuePod(obj)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			c.enqueuePod(newObj)
			c.enqueuePod(oldObj)
		},
		DeleteFunc: func(obj interface{}) {
			c.enqueuePod(obj)
		},
	})
}

func (c *Controller) enqueuePod(obj interface{}) {
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

	logger := klog.FromContext(context.Background())
	logger.Info("Enqueue Pod", "pod", fmt.Sprintf("%s/%s", pod.Namespace, pod.Name))

	group := getGroupFromPod(pod)
	if group != "" {
		c.queue.Add(group)
	}
}

func getGroupFromPod(pod *corev1.Pod) string {
	if pod.Labels == nil {
		return ""
	}
	return pod.Labels[PodLabelKey]
}
