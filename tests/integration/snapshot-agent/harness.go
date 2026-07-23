//go:build integration

// Package integration contains end-to-end tests for snapshot-agent backends.
//
// The harness runs INSIDE the cluster (see run.sh): it deploys the
// snapshot-agent and inference engine pods via the Kubernetes API. All
// snapshot/restore calls go through the Python client (via agentctl.py) —
// the production path for workloads — never a Go gRPC client. The harness
// talks to the engines over HTTP directly for inference checks.
//
// Test cases live in standalone_test.go and k8s_test.go. To add a test, add
// a t.Run(...) inside the engine group that provides the pods it needs.
package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
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

const (
	namespace     = "default"
	agentPodName  = "snapshot-agent-test"
	agentPort     = 9001
	podTimeout    = 5 * time.Minute
	healthTimeout = 5 * time.Minute
	// opTimeout bounds one agentctl.py invocation (RPC + operation polling);
	// it also guards against the client's unbounded wait_for_operation.
	opTimeout = 120 * time.Second
	// vramFreedMiB is the threshold below which we consider GPU memory freed.
	vramFreedMiB = 5000
)

// Harness manages the test stack for one deployment mode.
type Harness struct {
	Client *kubernetes.Clientset
	Config *rest.Config
	Node   string
	Image  string
	Model  string
	Mode   string // "standalone" or "k8s"

	AgentIP string
}

// NewHarness connects to the cluster, picks a GPU node, and deploys the
// snapshot-agent in the given mode. The agent is deleted via t.Cleanup.
func NewHarness(t *testing.T, mode string) *Harness {
	t.Helper()

	cfg, err := rest.InClusterConfig()
	if err != nil {
		t.Fatalf("in-cluster config: %v", err)
	}
	client, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		t.Fatalf("k8s client: %v", err)
	}

	image := os.Getenv("AGENT_IMAGE")
	if image == "" {
		t.Fatal("AGENT_IMAGE env var is required")
	}
	model := os.Getenv("MODEL")
	if model == "" {
		model = "Qwen/Qwen2.5-0.5B"
	}

	h := &Harness{Client: client, Config: cfg, Image: image, Model: model, Mode: mode}
	// TEST_NODE pins the suite to a specific node (e.g. one known to be
	// otherwise idle); by default the first node with a free GPU is used.
	if node := os.Getenv("TEST_NODE"); node != "" {
		h.Node = node
	} else {
		h.Node = h.pickGPUNode(t)
	}
	t.Logf("using node %s, image %s, mode %s", h.Node, h.Image, h.Mode)

	h.deployAgent(t)
	return h
}

// pickGPUNode returns the first node with at least one FREE GPU
// (allocatable minus what running pods have requested). Engines run one at a
// time, so a single free GPU is enough.
func (h *Harness) pickGPUNode(t *testing.T) string {
	t.Helper()
	nodes, err := h.Client.CoreV1().Nodes().List(context.Background(), metav1.ListOptions{
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
		used, err := h.gpusRequestedOnNode(node.Name)
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

// gpusRequestedOnNode sums nvidia.com/gpu requests of non-terminated pods on
// the node.
func (h *Harness) gpusRequestedOnNode(node string) (int64, error) {
	pods, err := h.Client.CoreV1().Pods("").List(context.Background(), metav1.ListOptions{
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

func (h *Harness) deployAgent(t *testing.T) {
	t.Helper()
	h.deletePodAndWait(t, agentPodName)

	pod := agentPod(h.Image, h.Node, h.Mode)
	if _, err := h.Client.CoreV1().Pods(namespace).Create(context.Background(), pod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("creating agent pod: %v", err)
	}
	t.Cleanup(func() { h.deletePodAndWait(t, agentPodName) })

	h.AgentIP = h.waitPodReady(t, agentPodName)
	t.Logf("agent ready at %s:%d", h.AgentIP, agentPort)
}

// WithEngine deploys an inference engine, waits until its HTTP server has
// loaded the model, runs fn, and deletes the engine (freeing the GPU).
func (h *Harness) WithEngine(t *testing.T, spec EngineSpec, fn func(t *testing.T, e *Engine)) {
	t.Helper()
	podName := spec.Name + "-test"
	h.deletePodAndWait(t, podName)

	pod := spec.BuildPod(h)
	if _, err := h.Client.CoreV1().Pods(namespace).Create(context.Background(), pod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("creating %s pod: %v", spec.Name, err)
	}
	defer h.deletePodAndWait(t, podName)

	ip := h.waitPodReady(t, podName)
	t.Logf("%s pod ready at %s, waiting for model load...", spec.Name, ip)
	h.waitHTTP(t, fmt.Sprintf("http://%s:%d/health", ip, spec.Port))

	e := &Engine{Spec: spec, IP: ip, PodName: podName}
	if h.Mode == "standalone" {
		e.PID = h.findPID(t, spec.PIDPattern)
		t.Logf("%s PID: %d", spec.Name, e.PID)
	} else {
		// Give the watcher time to spot the labeled pod's GPU activity and
		// register the job.
		t.Log("waiting 10s for watcher to register the job...")
		time.Sleep(10 * time.Second)
	}

	fn(t, e)
}

// Engine is a running inference engine instance.
type Engine struct {
	Spec    EngineSpec
	IP      string
	PodName string
	PID     int32 // standalone mode only
}

// Endpoint returns the engine's HTTP base URL as seen from inside the cluster.
func (e *Engine) Endpoint() string {
	return fmt.Sprintf("http://%s:%d", e.IP, e.Spec.Port)
}

// --- Pod helpers ---

func (h *Harness) waitPodReady(t *testing.T, name string) string {
	t.Helper()
	deadline := time.Now().Add(podTimeout)
	for time.Now().Before(deadline) {
		pod, err := h.Client.CoreV1().Pods(namespace).Get(context.Background(), name, metav1.GetOptions{})
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

func (h *Harness) deletePodAndWait(t *testing.T, name string) {
	t.Helper()
	zero := int64(0)
	err := h.Client.CoreV1().Pods(namespace).Delete(context.Background(), name, metav1.DeleteOptions{
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
		_, err := h.Client.CoreV1().Pods(namespace).Get(context.Background(), name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return
		}
		time.Sleep(2 * time.Second)
	}
	t.Fatalf("timeout waiting for pod %s to be deleted", name)
}

// waitHTTP polls url until it returns 200. Pod readiness only means the
// container started; engines need additional time to load the model.
func (h *Harness) waitHTTP(t *testing.T, url string) {
	t.Helper()
	client := &http.Client{Timeout: 5 * time.Second}
	deadline := time.Now().Add(healthTimeout)
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

// execPod runs a command in a pod container and returns stdout.
func (h *Harness) execPod(pod, container string, command ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), opTimeout)
	defer cancel()
	req := h.Client.CoreV1().RESTClient().Post().
		Resource("pods").Name(pod).Namespace(namespace).SubResource("exec").
		VersionedParams(&corev1.PodExecOptions{
			Container: container,
			Command:   command,
			Stdout:    true,
			Stderr:    true,
		}, scheme.ParameterCodec)

	exec, err := remotecommand.NewSPDYExecutor(h.Config, "POST", req.URL())
	if err != nil {
		return "", fmt.Errorf("creating executor: %w", err)
	}
	var stdout, stderr bytes.Buffer
	err = exec.StreamWithContext(ctx, remotecommand.StreamOptions{
		Stdout: &stdout,
		Stderr: &stderr,
	})
	if err != nil {
		return "", fmt.Errorf("exec failed: %w (stderr: %s)", err, stderr.String())
	}
	return stdout.String(), nil
}

// findPID locates the engine process on the node via the agent pod (hostPID).
func (h *Harness) findPID(t *testing.T, pattern string) int32 {
	t.Helper()
	script := fmt.Sprintf(
		`for p in /proc/[0-9]*/cmdline; do grep -ql '%s' $p 2>/dev/null && echo $p | cut -d/ -f3 && break; done`,
		pattern)
	out, err := h.execPod(agentPodName, "snapshot-agent", "sh", "-c", script)
	if err != nil {
		t.Fatalf("finding PID for %q: %v", pattern, err)
	}
	pid, err := strconv.ParseInt(strings.TrimSpace(out), 10, 32)
	if err != nil {
		t.Fatalf("parsing PID from %q: %v", out, err)
	}
	return int32(pid)
}

// VRAMMiB returns the GPU memory used (MiB) as seen from the engine pod.
func (h *Harness) VRAMMiB(t *testing.T, e *Engine) int {
	t.Helper()
	out, err := h.execPod(e.PodName, e.Spec.Name,
		"nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits")
	if err != nil {
		t.Fatalf("querying VRAM on %s: %v", e.PodName, err)
	}
	mib, err := strconv.Atoi(strings.TrimSpace(strings.Split(out, "\n")[0]))
	if err != nil {
		t.Fatalf("parsing VRAM from %q: %v", out, err)
	}
	return mib
}

// --- Agent call helpers ---
//
// All snapshot/restore calls go through the Python client via agentctl.py —
// the production path for workloads. The Go tests never dial the agent
// directly. BackendArgs describes a config with primitives; agentctl.py
// constructs the actual BackendConfig proto in Python, the same way a real
// workload does.

// BackendArgs are the agentctl.py flags describing a backend config.
type BackendArgs []string

func cudaConfig(pids ...int32) BackendArgs {
	args := BackendArgs{"--backend", "cuda"}
	if len(pids) > 0 {
		strs := make([]string, len(pids))
		for i, pid := range pids {
			strs[i] = strconv.Itoa(int(pid))
		}
		args = append(args, "--pids", strings.Join(strs, ","))
	}
	return args
}

// appConfig targets an application-aware workload via its HTTP API.
// mode may be "" (application default), "offload", or "discard".
func appConfig(app, endpoint, mode string) BackendArgs {
	args := BackendArgs{"--backend", "app", "--app", app, "--endpoints", endpoint}
	if mode != "" {
		args = append(args, "--mode", mode)
	}
	return args
}

// SnapshotOK snapshots via the Python client and fails the test if the
// operation does not complete.
func (h *Harness) SnapshotOK(t *testing.T, jobID string, cfg BackendArgs) {
	t.Helper()
	h.agentctl(t, "snapshot", jobID, cfg)
}

// RestoreOK restores via the Python client and fails the test if the
// operation does not complete.
func (h *Harness) RestoreOK(t *testing.T, jobID string, cfg BackendArgs) {
	t.Helper()
	h.agentctl(t, "restore", jobID, cfg)
}

func (h *Harness) agentctl(t *testing.T, action, jobID string, cfg BackendArgs) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), opTimeout)
	defer cancel()

	// go test runs with the package directory as the working directory.
	args := []string{"agentctl.py", action,
		"--agent", fmt.Sprintf("%s:%d", h.AgentIP, agentPort),
		"--job-id", jobID}
	args = append(args, cfg...)

	cmd := exec.CommandContext(ctx, "python3", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s(%s) via python client failed: %v\n%s", action, jobID, err, string(out))
	}
	t.Logf("%s", strings.TrimSpace(string(out)))
}

// --- Inference helpers ---

// Inference runs a fixed deterministic prompt against the engine's
// OpenAI-compatible API and returns the completion text.
func (h *Harness) Inference(t *testing.T, e *Engine) string {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"model":       h.Model,
		"prompt":      "The capital of France is",
		"max_tokens":  15,
		"temperature": 0,
	})
	if err != nil {
		t.Fatalf("marshaling inference request: %v", err)
	}
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Post(e.Endpoint()+"/v1/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("inference request to %s: %v", e.Endpoint(), err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("reading inference response: %v", err)
	}
	var parsed struct {
		Choices []struct {
			Text string `json:"text"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil || len(parsed.Choices) == 0 {
		t.Fatalf("unexpected inference response (status %d): %s", resp.StatusCode, string(raw))
	}
	return parsed.Choices[0].Text
}

// RequireFreedAndCorrect asserts VRAM was actually freed while the engine was
// asleep and that inference output is identical after restore.
func RequireFreedAndCorrect(t *testing.T, vramWhileAsleep int, before, after string) {
	t.Helper()
	if vramWhileAsleep >= vramFreedMiB {
		t.Errorf("VRAM not freed: %d MiB (want < %d)", vramWhileAsleep, vramFreedMiB)
	}
	if before != after {
		t.Errorf("inference changed after restore: before=%q after=%q", before, after)
	}
}

