//go:build integration

// Package integration contains end-to-end tests for snapshot-agent backends.
//
// The harness runs INSIDE the cluster (see run.sh): it deploys the
// snapshot-agent and inference engine pods via the Kubernetes API. All
// snapshot/restore calls go through the Python client (via agentctl.py) —
// the production path for workloads — never a Go gRPC client. The harness
// talks to the engines over HTTP directly for inference checks.
//
// Generic cluster helpers (client setup, node selection, pod lifecycle,
// exec) live in tests/integration/harness; this package adds the
// snapshot-agent specifics: agent deployment, engine specs, and the
// agentctl.py call path.
//
// Test cases live in standalone_test.go, k8s_test.go and
// orchestrator_test.go. To add a test, add a t.Run(...) inside the engine
// group that provides the pods it needs.
package integration

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/tests/integration/harness"
)

const (
	namespace    = "default"
	agentPodName = "snapshot-agent-test"
	agentPort    = 9001
	// chartNamespace is where the official Helm charts install components:
	// their templates pin the namespace rather than using the release's.
	chartNamespace = "timeslice-system"
	podTimeout     = 5 * time.Minute
	healthTimeout  = 5 * time.Minute
	// opTimeout bounds one agentctl.py invocation (RPC + operation polling);
	// it also guards against the client's unbounded wait_for_operation.
	opTimeout = 120 * time.Second
	// vramFreedMiB is the threshold below which we consider GPU memory freed.
	vramFreedMiB = 5000
)

// Harness manages the test stack for one deployment mode.
type Harness struct {
	*harness.Cluster
	Node  string
	Model string
	Mode  string // "standalone" or "k8s"

	AgentIP string
	// AgentPort is the agent's gRPC port: agentPort for harness-deployed
	// agents, overridable via CHART_AGENT_PORT for chart-deployed ones (so
	// the suite can coexist with an unrelated agent on the default port).
	AgentPort int
}

// NewHarness connects to the cluster, picks a GPU node, and deploys the
// standalone agent from the `make standalone` artifacts (run.sh builds them
// in the test-runner pod). The agent is deleted via t.Cleanup.
func NewHarness(t *testing.T) *Harness {
	t.Helper()

	model := os.Getenv("MODEL")
	if model == "" {
		model = "Qwen/Qwen2.5-0.5B"
	}

	h := &Harness{
		Cluster:   harness.NewCluster(t, namespace),
		Model:     model,
		Mode:      "standalone",
		AgentPort: agentPort,
	}
	// TEST_NODE pins the suite to a specific node (e.g. one known to be
	// otherwise idle); by default the first node with a free GPU is used.
	h.Node = h.PickNode(t)
	t.Logf("using node %s, mode %s", h.Node, h.Mode)

	h.deployAgent(t)
	return h
}

// NewChartHarness attaches to the snapshot-agent deployed by the official
// Helm chart (run.sh installs it for the k8s phase) instead of deploying an
// agent pod. The chart's DaemonSet is pinned to TEST_NODE via nodeSelector,
// so TEST_NODE is required.
func NewChartHarness(t *testing.T) *Harness {
	t.Helper()

	node := harness.RequiredNode(t)
	model := os.Getenv("MODEL")
	if model == "" {
		model = "Qwen/Qwen2.5-0.5B"
	}

	h := &Harness{
		Cluster:   harness.NewCluster(t, namespace),
		Model:     model,
		Mode:      "k8s",
		Node:      node,
		AgentPort: agentPort,
	}
	if p := os.Getenv("CHART_AGENT_PORT"); p != "" {
		port, err := strconv.Atoi(p)
		if err != nil {
			t.Fatalf("invalid CHART_AGENT_PORT %q: %v", p, err)
		}
		h.AgentPort = port
	}
	// The chart runs the agent on hostNetwork, so the pod IP is the node IP —
	// also how the orchestrator reaches agents.
	h.AgentIP = h.WaitPodReadyByLabel(t, chartNamespace,
		"app.kubernetes.io/name=snapshot-agent,app.kubernetes.io/instance=sa-chart-test",
		h.Node, podTimeout)
	t.Logf("using node %s, chart-deployed agent at %s:%d", h.Node, h.AgentIP, h.AgentPort)
	return h
}

func (h *Harness) deployAgent(t *testing.T) {
	t.Helper()
	h.DeletePodAndWait(t, agentPodName)

	pod := agentPod(h.Node)
	if _, err := h.Client.CoreV1().Pods(namespace).Create(context.Background(), pod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("creating agent pod: %v", err)
	}
	t.Cleanup(func() { h.DeletePodAndWait(t, agentPodName) })

	h.AgentIP = h.WaitPodReady(t, agentPodName, podTimeout)
	h.installAgentBinaries(t)
	h.waitAgentUp(t)
	t.Logf("agent (make-standalone artifacts) ready at %s:%d", h.AgentIP, h.AgentPort)
}

// standaloneBinDir is where run.sh builds the standalone artifacts in the
// test-runner pod (`make standalone` → bin/), relative to this package's
// directory (go test's working directory).
const standaloneBinDir = "../../../bin"

// installAgentBinaries streams the make-standalone artifacts into the waiting
// agent pod and releases it (the container execs the agent once /opt/rlts is
// populated and the .ready marker exists).
func (h *Harness) installAgentBinaries(t *testing.T) {
	t.Helper()

	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	for _, name := range []string{"snapshot-agent", "cuda-checkpoint"} {
		data, err := os.ReadFile(standaloneBinDir + "/" + name)
		if err != nil {
			t.Fatalf("reading make-standalone artifact (run.sh builds them with `make standalone`): %v", err)
		}
		if err := tw.WriteHeader(&tar.Header{Name: "bin/" + name, Mode: 0o755, Size: int64(len(data))}); err != nil {
			t.Fatalf("writing tar header for %s: %v", name, err)
		}
		if _, err := tw.Write(data); err != nil {
			t.Fatalf("writing tar data for %s: %v", name, err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("closing tar stream: %v", err)
	}

	if _, err := h.ExecPodStdin(agentPodName, "snapshot-agent", &buf, opTimeout, "tar", "-xf", "-", "-C", "/opt/rlts"); err != nil {
		t.Fatalf("copying standalone artifacts into agent pod: %v", err)
	}
	if _, err := h.ExecPod(agentPodName, "snapshot-agent", opTimeout, "touch", "/opt/rlts/.ready"); err != nil {
		t.Fatalf("releasing agent pod: %v", err)
	}
}

// waitAgentUp waits until the agent's gRPC port accepts connections; the
// standalone agent starts only after its binaries are copied in.
func (h *Harness) waitAgentUp(t *testing.T) {
	t.Helper()
	addr := net.JoinHostPort(h.AgentIP, strconv.Itoa(h.AgentPort))
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 2*time.Second)
		if err == nil {
			conn.Close()
			return
		}
		time.Sleep(2 * time.Second)
	}
	t.Fatalf("timeout waiting for the standalone agent to listen on %s", addr)
}

// WithEngine deploys an inference engine, waits until its HTTP server has
// loaded the model, runs fn, and deletes the engine (freeing the GPU).
func (h *Harness) WithEngine(t *testing.T, spec EngineSpec, fn func(t *testing.T, e *Engine)) {
	t.Helper()
	podName := spec.Name + "-test"
	h.DeletePodAndWait(t, podName)

	pod := spec.BuildPod(h)
	if _, err := h.Client.CoreV1().Pods(namespace).Create(context.Background(), pod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("creating %s pod: %v", spec.Name, err)
	}
	defer h.DeletePodAndWait(t, podName)

	ip := h.WaitPodReady(t, podName, podTimeout)
	t.Logf("%s pod ready at %s, waiting for model load...", spec.Name, ip)
	h.WaitHTTP(t, fmt.Sprintf("http://%s:%d/health", ip, spec.Port), healthTimeout)

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

// findPID locates the engine process on the node via the agent pod (hostPID).
func (h *Harness) findPID(t *testing.T, pattern string) int32 {
	t.Helper()
	script := fmt.Sprintf(
		`for p in /proc/[0-9]*/cmdline; do grep -ql '%s' $p 2>/dev/null && echo $p | cut -d/ -f3 && break; done`,
		pattern)
	out, err := h.ExecPod(agentPodName, "snapshot-agent", opTimeout, "sh", "-c", script)
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
	return h.PodVRAMMiB(t, e.PodName, e.Spec.Name, opTimeout)
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
		"--agent", fmt.Sprintf("%s:%d", h.AgentIP, h.AgentPort),
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
