//go:build integration

// Package harness provides the generic in-cluster pieces shared by the
// integration suites: client setup, node selection, pod lifecycle, exec and
// HTTP polling helpers, and VRAM checks. Suites compose a Cluster with their
// own workload-specific helpers (see tests/integration/snapshot-agent).
//
// Everything here runs INSIDE the cluster (see the suite launchers): the
// client is built from the in-cluster config of the test-runner pod.
package harness

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
)

// Cluster is an in-cluster Kubernetes client plus the namespace the suite
// deploys its pods into.
type Cluster struct {
	Client    *kubernetes.Clientset
	Config    *rest.Config
	Namespace string
}

// NewCluster connects to the cluster via the in-cluster config.
func NewCluster(t *testing.T, namespace string) *Cluster {
	t.Helper()

	cfg, err := rest.InClusterConfig()
	if err != nil {
		t.Fatalf("in-cluster config: %v", err)
	}
	client, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		t.Fatalf("k8s client: %v", err)
	}
	return &Cluster{Client: client, Config: cfg, Namespace: namespace}
}

// --- Node selection ---

// PickNode returns TEST_NODE when set (pinning the suite to a node known to
// be otherwise idle), otherwise the first node with a free GPU.
func (c *Cluster) PickNode(t *testing.T) string {
	t.Helper()
	if node := os.Getenv("TEST_NODE"); node != "" {
		return node
	}
	return c.PickGPUNode(t)
}

// RequiredNode returns TEST_NODE and fails the test if it is unset. Suites
// that attach to pre-deployed, node-pinned components (e.g. Helm charts)
// cannot pick a node themselves.
func RequiredNode(t *testing.T) string {
	t.Helper()
	node := os.Getenv("TEST_NODE")
	if node == "" {
		t.Fatal("TEST_NODE env var is required")
	}
	return node
}

// PickGPUNode returns the first node with at least one FREE GPU (allocatable
// minus what running pods have requested).
func (c *Cluster) PickGPUNode(t *testing.T) string {
	t.Helper()
	nodes, err := c.Client.CoreV1().Nodes().List(context.Background(), metav1.ListOptions{
		LabelSelector: "cloud.google.com/gke-accelerator",
	})
	if err != nil {
		t.Fatalf("listing GPU nodes: %v", err)
	}
	for i := range nodes.Items {
		node := &nodes.Items[i]
		alloc := node.Status.Allocatable[corev1.ResourceName("nvidia.com/gpu")]
		if alloc.Value() < 1 {
			continue
		}
		used, err := c.GPUsRequestedOnNode(node.Name)
		if err != nil {
			t.Fatalf("counting GPU requests on %s: %v", node.Name, err)
		}
		if alloc.Value()-used >= 1 {
			return node.Name
		}
		t.Logf("node %s has no free GPUs (%d/%d requested), skipping", node.Name, used, alloc.Value())
	}
	t.Fatal("no node with a free GPU found")
	return ""
}

// GPUsRequestedOnNode sums nvidia.com/gpu requests of non-terminated pods on
// the node.
func (c *Cluster) GPUsRequestedOnNode(node string) (int64, error) {
	pods, err := c.Client.CoreV1().Pods("").List(context.Background(), metav1.ListOptions{
		FieldSelector: "spec.nodeName=" + node,
	})
	if err != nil {
		return 0, err
	}
	var used int64
	for i := range pods.Items {
		pod := &pods.Items[i]
		if pod.Status.Phase == corev1.PodSucceeded || pod.Status.Phase == corev1.PodFailed {
			continue
		}
		for j := range pod.Spec.Containers {
			req := pod.Spec.Containers[j].Resources.Requests[corev1.ResourceName("nvidia.com/gpu")]
			used += req.Value()
		}
	}
	return used, nil
}

// --- Pod lifecycle ---

// WaitPodReady waits until the named pod in the Cluster namespace is Ready
// and returns its IP.
func (c *Cluster) WaitPodReady(t *testing.T, name string, timeout time.Duration) string {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		pod, err := c.Client.CoreV1().Pods(c.Namespace).Get(context.Background(), name, metav1.GetOptions{})
		if err == nil {
			for _, cond := range pod.Status.Conditions {
				if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
					return pod.Status.PodIP
				}
			}
		}
		time.Sleep(3 * time.Second)
	}
	t.Fatalf("timeout waiting for pod %s to become ready", name)
	return ""
}

// WaitPodReadyByLabel waits for a Ready pod matching labelSelector on the
// given node in the given namespace and returns its IP. It attaches to pods
// deployed by something other than the suite (e.g. a Helm chart DaemonSet).
func (c *Cluster) WaitPodReadyByLabel(t *testing.T, namespace, labelSelector, node string, timeout time.Duration) string {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		pods, err := c.Client.CoreV1().Pods(namespace).List(context.Background(), metav1.ListOptions{
			LabelSelector: labelSelector,
			FieldSelector: "spec.nodeName=" + node,
		})
		if err == nil {
			for i := range pods.Items {
				pod := &pods.Items[i]
				for _, cond := range pod.Status.Conditions {
					if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
						return pod.Status.PodIP
					}
				}
			}
		}
		time.Sleep(3 * time.Second)
	}
	t.Fatalf("timeout waiting for pod matching %q on node %s in namespace %s", labelSelector, node, namespace)
	return ""
}

// DeletePodAndWait force-deletes the named pod in the Cluster namespace and
// waits until it is gone. Missing pods are not an error.
func (c *Cluster) DeletePodAndWait(t *testing.T, name string) {
	t.Helper()
	zero := int64(0)
	err := c.Client.CoreV1().Pods(c.Namespace).Delete(context.Background(), name, metav1.DeleteOptions{
		GracePeriodSeconds: &zero,
	})
	if apierrors.IsNotFound(err) {
		return
	}
	if err != nil {
		t.Fatalf("deleting pod %s: %v", name, err)
	}
	deadline := time.Now().Add(2 * time.Minute)
	for time.Now().Before(deadline) {
		_, err := c.Client.CoreV1().Pods(c.Namespace).Get(context.Background(), name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return
		}
		time.Sleep(2 * time.Second)
	}
	t.Fatalf("timeout waiting for pod %s to be deleted", name)
}

// WaitHTTP polls url until it returns 200. Pod readiness only means the
// container started; servers may need additional time (e.g. model load).
func (c *Cluster) WaitHTTP(t *testing.T, url string, timeout time.Duration) {
	t.Helper()
	client := &http.Client{Timeout: 5 * time.Second}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := client.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return
			}
		}
		time.Sleep(5 * time.Second)
	}
	t.Fatalf("timeout waiting for %s", url)
}

// --- Exec helpers ---

// ExecPod runs a command in a pod container in the Cluster namespace and
// returns stdout.
func (c *Cluster) ExecPod(pod, container string, timeout time.Duration, command ...string) (string, error) {
	return c.ExecPodStdin(pod, container, nil, timeout, command...)
}

// ExecPodStdin runs a command in a pod container, streaming stdin into it if
// non-nil, and returns stdout.
func (c *Cluster) ExecPodStdin(pod, container string, stdin io.Reader, timeout time.Duration, command ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req := c.Client.CoreV1().RESTClient().Post().
		Resource("pods").Name(pod).Namespace(c.Namespace).SubResource("exec").
		VersionedParams(&corev1.PodExecOptions{
			Container: container,
			Command:   command,
			Stdin:     stdin != nil,
			Stdout:    true,
			Stderr:    true,
		}, scheme.ParameterCodec)

	exec, err := remotecommand.NewSPDYExecutor(c.Config, "POST", req.URL())
	if err != nil {
		return "", fmt.Errorf("creating executor: %w", err)
	}
	var stdout, stderr bytes.Buffer
	err = exec.StreamWithContext(ctx, remotecommand.StreamOptions{
		Stdin:  stdin,
		Stdout: &stdout,
		Stderr: &stderr,
	})
	if err != nil {
		return "", fmt.Errorf("exec failed: %w (stderr: %s)", err, stderr.String())
	}
	return stdout.String(), nil
}

// PodVRAMMiB returns the GPU memory used (MiB) as seen from the given pod.
func (c *Cluster) PodVRAMMiB(t *testing.T, pod, container string, timeout time.Duration) int {
	t.Helper()
	out, err := c.ExecPod(pod, container, timeout,
		"nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits")
	if err != nil {
		t.Fatalf("querying VRAM on %s: %v", pod, err)
	}
	mib, err := strconv.Atoi(strings.TrimSpace(strings.Split(out, "\n")[0]))
	if err != nil {
		t.Fatalf("parsing VRAM from %q: %v", out, err)
	}
	return mib
}
