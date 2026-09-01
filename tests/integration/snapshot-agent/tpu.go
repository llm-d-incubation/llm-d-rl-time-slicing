//go:build integration

package integration

import (
	"archive/tar"
	"bytes"
	"context"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/tests/integration/harness"
)

// TPU test pieces (see TestTpu in tpu_test.go).
//
// The layout mirrors the CUDA standalone phase: a hostPID agent pod on the
// TPU node runs the make-standalone snapshot-agent plus the gVisor
// tpucheckpoint CLI, and a separate workload pod runs one single-chip JAX
// process per TPU chip with a checkpointing-enabled libtpu
// (LIBTPU_CHECKPOINTING_ENABLED=true spawns the libtpu control threads the
// CLI drives). Snapshot/restore go through agentctl.py with explicit host
// PIDs, exactly like the CUDA flow.
//
// The workload must issue NO TPU ops while checkpointed (libtpu contract:
// the driver quiesces before yielding its slice). The test enforces this
// with a flag file the workers poll: TpuQuiesce parks every worker on pure
// CPU polling before the snapshot, TpuResume lets them compute again after
// the restore — and their step counters advancing past the parked values is
// the proof that restore actually brought the chips back.
const (
	tpuWorkloadPodName   = "tpu-workload-test"
	tpuConfigMapName     = "tpu-workload-src"
	tpuWorkloadContainer = "workload"
	// tpuStateDir is the emptyDir where the workers publish their status
	// files and watch for the quiesce flag.
	tpuStateDir = "/workload-state"
)

// tpuChips returns the number of TPU chips the workload requests (one
// worker process per chip), from TPU_CHIPS (default 8, a v5e-8 host).
func tpuChips(t *testing.T) int {
	t.Helper()
	s := os.Getenv("TPU_CHIPS")
	if s == "" {
		return 8
	}
	n, err := strconv.Atoi(s)
	if err != nil || n <= 0 {
		t.Fatalf("invalid TPU_CHIPS %q", s)
	}
	return n
}

func tpuConfig(pids ...int32) BackendArgs {
	strs := make([]string, len(pids))
	for i, pid := range pids {
		strs[i] = strconv.Itoa(int(pid))
	}
	return BackendArgs{"--backend", "tpu", "--pids", strings.Join(strs, ",")}
}

// NewTpuHarness connects to the cluster and deploys the standalone agent on
// the TPU node (TEST_NODE) with the tpucheckpoint CLI alongside it.
func NewTpuHarness(t *testing.T) *Harness {
	t.Helper()

	h := &Harness{
		Cluster:   harness.NewCluster(t, namespace),
		Mode:      "standalone",
		Node:      harness.RequiredNode(t),
		AgentPort: agentPort,
	}
	t.Logf("using TPU node %s", h.Node)

	h.DeletePodAndWait(t, agentPodName)
	pod := tpuAgentPod(h.Node)
	if _, err := h.Client.CoreV1().Pods(namespace).Create(context.Background(), pod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("creating TPU agent pod: %v", err)
	}
	t.Cleanup(func() { h.DeletePodAndWait(t, agentPodName) })

	h.AgentIP = h.WaitPodReady(t, agentPodName, podTimeout)
	h.installTpuAgentBinaries(t)
	h.waitAgentUp(t)
	t.Logf("agent (make-standalone artifacts + tpucheckpoint) ready at %s:%d", h.AgentIP, h.AgentPort)
	return h
}

// installTpuAgentBinaries streams the agent and the tpucheckpoint CLI into
// the waiting agent pod and releases it (same handshake as
// installAgentBinaries; the TPU backend finds tpucheckpoint on PATH under
// /opt/rlts/bin).
func (h *Harness) installTpuAgentBinaries(t *testing.T) {
	t.Helper()

	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	for _, name := range []string{"snapshot-agent", "tpucheckpoint"} {
		data, err := os.ReadFile(standaloneBinDir + "/" + name)
		if err != nil {
			t.Fatalf("reading artifact %s (run.sh builds snapshot-agent with `make standalone` and fetches tpucheckpoint from TPU_CHECKPOINT_URI): %v", name, err)
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
		t.Fatalf("copying artifacts into agent pod: %v", err)
	}
	if _, err := h.ExecPod(agentPodName, "snapshot-agent", opTimeout, "touch", "/opt/rlts/.ready"); err != nil {
		t.Fatalf("releasing agent pod: %v", err)
	}
}

// tpuAgentPod is the TPU flavor of agentPod: same wait-for-binaries
// handshake, hostPID and privileged (the CLI drives host PIDs via /proc and
// the vfio gate opens /dev/vfio groups), no NVIDIA pieces, and no TPU chips
// requested — the agent only needs the node's /dev and /proc.
func tpuAgentPod(node string) *corev1.Pod {
	privileged := true
	hostPathDir := corev1.HostPathDirectory
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      agentPodName,
			Namespace: namespace,
			Labels: map[string]string{
				"app":        agentPodName,
				"test-suite": "snapshot-agent-integration",
			},
		},
		Spec: corev1.PodSpec{
			ServiceAccountName: "snapshot-agent-test",
			RestartPolicy:      corev1.RestartPolicyNever,
			NodeName:           node,
			HostPID:            true,
			Tolerations:        tpuTolerations(),
			Containers: []corev1.Container{{
				Name:  "snapshot-agent",
				Image: "debian:bookworm-slim",
				Command: []string{"sh", "-c",
					`until [ -f /opt/rlts/.ready ]; do sleep 1; done; exec /opt/rlts/bin/snapshot-agent --port=9001 --deployment-mode=standalone`},
				Env: []corev1.EnvVar{
					// /opt/rlts/bin provides the tpucheckpoint CLI.
					{Name: "PATH", Value: "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/rlts/bin"},
					{Name: "NODE_NAME", ValueFrom: &corev1.EnvVarSource{
						FieldRef: &corev1.ObjectFieldSelector{FieldPath: "spec.nodeName"},
					}},
				},
				Ports:           []corev1.ContainerPort{{ContainerPort: agentPort}},
				SecurityContext: &corev1.SecurityContext{Privileged: &privileged},
				VolumeMounts: []corev1.VolumeMount{
					{Name: "rlts", MountPath: "/opt/rlts"},
					{Name: "dev", MountPath: "/dev"},
				},
			}},
			Volumes: []corev1.Volume{
				{Name: "rlts", VolumeSource: corev1.VolumeSource{
					EmptyDir: &corev1.EmptyDirVolumeSource{},
				}},
				{Name: "dev", VolumeSource: corev1.VolumeSource{
					HostPath: &corev1.HostPathVolumeSource{Path: "/dev", Type: &hostPathDir},
				}},
			},
		},
	}
}

func tpuTolerations() []corev1.Toleration {
	return []corev1.Toleration{{
		Key:      "google.com/tpu",
		Operator: corev1.TolerationOpExists,
		Effect:   corev1.TaintEffectNoSchedule,
	}}
}

// TpuWorkload is a running multi-process TPU job.
type TpuWorkload struct {
	PodName string
	Chips   int
	// PIDs are the workers' HOST PIDs (as the agent's TPU backend targets
	// them), discovered via their libtpu control threads.
	PIDs []int32
}

// WithTpuWorkload deploys the TPU workload pod (one single-chip JAX process
// per chip, see tpu_workload.py), waits until every worker computes, runs
// fn, and deletes the pod (freeing the chips).
func (h *Harness) WithTpuWorkload(t *testing.T, image string, fn func(t *testing.T, w *TpuWorkload)) {
	t.Helper()
	chips := tpuChips(t)

	h.createTpuSourceConfigMap(t)
	defer func() {
		if err := h.DeleteConfigMap(tpuConfigMapName); err != nil {
			t.Logf("warning: failed to delete ConfigMap %s: %v", tpuConfigMapName, err)
		}
	}()

	h.DeletePodAndWait(t, tpuWorkloadPodName)
	pod := tpuWorkloadPod(h, image, chips, h.tpuNodeSelector(t))
	if _, err := h.Client.CoreV1().Pods(namespace).Create(context.Background(), pod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("creating TPU workload pod: %v", err)
	}
	defer h.DeletePodAndWait(t, tpuWorkloadPodName)

	// The readiness probe covers TPU init and the first step of every
	// worker.
	h.WaitPodReady(t, tpuWorkloadPodName, podTimeout)

	w := &TpuWorkload{PodName: tpuWorkloadPodName, Chips: chips}
	w.PIDs = h.findTpuPIDs(t, chips)
	t.Logf("TPU workload ready, %d worker PIDs: %v", chips, w.PIDs)

	fn(t, w)
}

// createTpuSourceConfigMap packages tpu_workload.py into a ConfigMap
// mounted by the workload pod.
func (h *Harness) createTpuSourceConfigMap(t *testing.T) {
	t.Helper()
	script, err := os.ReadFile("tpu_workload.py")
	if err != nil {
		t.Fatalf("reading tpu_workload.py: %v", err)
	}
	if err := h.DeleteConfigMap(tpuConfigMapName); err != nil {
		t.Logf("warning: pre-create ConfigMap cleanup failed: %v", err)
	}
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      tpuConfigMapName,
			Namespace: namespace,
			Labels:    map[string]string{"test-suite": "snapshot-agent-integration"},
		},
		Data: map[string]string{"tpu_workload.py": string(script)},
	}
	if _, err := h.Client.CoreV1().ConfigMaps(namespace).Create(context.Background(), cm, metav1.CreateOptions{}); err != nil {
		t.Fatalf("creating ConfigMap %s: %v", tpuConfigMapName, err)
	}
}

// tpuNodeSelector pins the workload to the TPU node the GKE-approved way:
// pods requesting google.com/tpu must select on the accelerator and
// topology labels (the tpu-accelerator-topology-constraints admission
// webhook rejects a bare nodeName pin), plus the hostname to stay on
// TEST_NODE.
func (h *Harness) tpuNodeSelector(t *testing.T) map[string]string {
	t.Helper()
	node, err := h.Client.CoreV1().Nodes().Get(context.Background(), h.Node, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("reading node %s: %v", h.Node, err)
	}
	selector := map[string]string{"kubernetes.io/hostname": h.Node}
	for _, label := range []string{
		"cloud.google.com/gke-tpu-accelerator",
		"cloud.google.com/gke-tpu-topology",
	} {
		v, ok := node.Labels[label]
		if !ok {
			t.Fatalf("node %s has no %s label (not a TPU node?)", h.Node, label)
		}
		selector[label] = v
	}
	return selector
}

// tpuWorkloadPod runs the multi-process JAX workload. TPU_LIBTPU_URI (a
// gs:// object) optionally provides a checkpointing-enabled libtpu build
// fetched by an init container; without it the image's libtpu must support
// LIBTPU_CHECKPOINTING_ENABLED.
func tpuWorkloadPod(h *Harness, image string, chips int, nodeSelector map[string]string) *corev1.Pod {
	env := []corev1.EnvVar{
		{Name: "LIBTPU_CHECKPOINTING_ENABLED", Value: "true"},
		{Name: "TPU_NPROC", Value: strconv.Itoa(chips)},
		{Name: "TPU_STATE_DIR", Value: tpuStateDir},
	}
	volumes := []corev1.Volume{
		{Name: "src", VolumeSource: corev1.VolumeSource{
			ConfigMap: &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: tpuConfigMapName},
			},
		}},
		{Name: "state", VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{},
		}},
	}
	mounts := []corev1.VolumeMount{
		{Name: "src", MountPath: "/opt/src"},
		{Name: "state", MountPath: tpuStateDir},
	}
	var initContainers []corev1.Container
	if uri := os.Getenv("TPU_LIBTPU_URI"); uri != "" {
		volumes = append(volumes, corev1.Volume{Name: "libtpu", VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{},
		}})
		mounts = append(mounts, corev1.VolumeMount{Name: "libtpu", MountPath: "/libtpu"})
		env = append(env, corev1.EnvVar{Name: "TPU_LIBRARY_PATH", Value: "/libtpu/libtpu.so"})
		initContainers = []corev1.Container{{
			Name:    "fetch-libtpu",
			Image:   "google/cloud-sdk:slim",
			Command: []string{"bash", "-c", fmt.Sprintf("gcloud storage cp %q /libtpu/libtpu.so", uri)},
			VolumeMounts: []corev1.VolumeMount{
				{Name: "libtpu", MountPath: "/libtpu"},
			},
		}}
	}

	tpuQty := resource.MustParse(strconv.Itoa(chips))
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      tpuWorkloadPodName,
			Namespace: namespace,
			Labels: map[string]string{
				"app":        tpuWorkloadPodName,
				"test-suite": "snapshot-agent-integration",
			},
		},
		Spec: corev1.PodSpec{
			ServiceAccountName: "snapshot-agent-test",
			RestartPolicy:      corev1.RestartPolicyNever,
			NodeSelector:       nodeSelector,
			Tolerations:        tpuTolerations(),
			InitContainers:     initContainers,
			Volumes:            volumes,
			Containers: []corev1.Container{{
				Name:         tpuWorkloadContainer,
				Image:        image,
				Command:      []string{"python3", "/opt/src/tpu_workload.py"},
				Env:          env,
				VolumeMounts: mounts,
				ReadinessProbe: &corev1.Probe{
					ProbeHandler: corev1.ProbeHandler{
						Exec: &corev1.ExecAction{Command: []string{"test", "-f", tpuStateDir + "/ready"}},
					},
					InitialDelaySeconds: 20,
					PeriodSeconds:       5,
					FailureThreshold:    60,
				},
				Resources: corev1.ResourceRequirements{
					Limits:   corev1.ResourceList{"google.com/tpu": tpuQty},
					Requests: corev1.ResourceList{"google.com/tpu": tpuQty},
				},
			}},
		},
	}
}

// findTpuPIDs locates the workers' host PIDs via the agent pod (hostPID):
// exactly the processes carrying a libtpu control thread — the same
// discovery the tpucheckpoint CLI performs.
func (h *Harness) findTpuPIDs(t *testing.T, want int) []int32 {
	t.Helper()
	script := `for c in /proc/[0-9]*/task/*/comm; do ` +
		`read -r name < "$c" 2>/dev/null || continue; ` +
		`case "$name" in libtpu*) echo "$c" | cut -d/ -f3 ;; esac; ` +
		`done | sort -un`
	deadline := time.Now().Add(2 * time.Minute)
	for {
		out, err := h.ExecPod(agentPodName, "snapshot-agent", opTimeout, "sh", "-c", script)
		if err != nil {
			t.Fatalf("scanning for libtpu control threads: %v", err)
		}
		var pids []int32
		for _, line := range strings.Fields(out) {
			pid, err := strconv.ParseInt(line, 10, 32)
			if err != nil {
				t.Fatalf("parsing PID from %q: %v", out, err)
			}
			pids = append(pids, int32(pid))
		}
		if len(pids) == want {
			sort.Slice(pids, func(i, j int) bool { return pids[i] < pids[j] })
			return pids
		}
		// Control threads appear once libtpu finishes initializing; the
		// readiness probe makes this window short.
		if time.Now().After(deadline) {
			t.Fatalf("found %d processes with libtpu control threads, want %d: %v "+
				"(is the libtpu build checkpointing-enabled?)", len(pids), want, pids)
		}
		time.Sleep(3 * time.Second)
	}
}

// TpuQuiesce sets the quiesce flag and waits until every worker reports
// parked — required before a checkpoint (a checkpointed process must issue
// no TPU ops until restored).
func (h *Harness) TpuQuiesce(t *testing.T, w *TpuWorkload) {
	t.Helper()
	if _, err := h.ExecPod(w.PodName, tpuWorkloadContainer, opTimeout, "touch", tpuStateDir+"/quiesce"); err != nil {
		t.Fatalf("setting quiesce flag: %v", err)
	}
	h.waitTpuState(t, w, "parked")
	t.Log("all TPU workers parked")
}

// TpuResume clears the quiesce flag; workers go back to issuing TPU ops.
func (h *Harness) TpuResume(t *testing.T, w *TpuWorkload) {
	t.Helper()
	if _, err := h.ExecPod(w.PodName, tpuWorkloadContainer, opTimeout, "rm", "-f", tpuStateDir+"/quiesce"); err != nil {
		t.Fatalf("clearing quiesce flag: %v", err)
	}
}

// TpuSteps returns each worker's current step counter.
func (h *Harness) TpuSteps(t *testing.T, w *TpuWorkload) []int {
	t.Helper()
	states, steps := h.readTpuStatus(t, w)
	_ = states
	return steps
}

// WaitTpuAdvancing waits until every worker's step counter moves past its
// value in before — the proof that the chips compute again after restore.
func (h *Harness) WaitTpuAdvancing(t *testing.T, w *TpuWorkload, before []int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Minute)
	for {
		_, steps := h.readTpuStatus(t, w)
		advancing := 0
		for i := range steps {
			if steps[i] > before[i] {
				advancing++
			}
		}
		if advancing == w.Chips {
			t.Logf("all %d TPU workers computing again (steps %v -> %v)", w.Chips, before, steps)
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("only %d/%d workers advanced after restore (steps before=%v now=%v)",
				advancing, w.Chips, before, steps)
		}
		time.Sleep(2 * time.Second)
	}
}

// waitTpuState waits until every worker's status file reports the given
// state ("run" or "parked").
func (h *Harness) waitTpuState(t *testing.T, w *TpuWorkload, state string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Minute)
	for {
		states, _ := h.readTpuStatus(t, w)
		matching := 0
		for _, s := range states {
			if s == state {
				matching++
			}
		}
		if matching == w.Chips {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("only %d/%d workers reached state %q: %v", matching, w.Chips, state, states)
		}
		time.Sleep(2 * time.Second)
	}
}

// readTpuStatus reads every worker's status file ("<state> <step>").
func (h *Harness) readTpuStatus(t *testing.T, w *TpuWorkload) (states []string, steps []int) {
	t.Helper()
	out, err := h.ExecPod(w.PodName, tpuWorkloadContainer, opTimeout, "sh", "-c",
		fmt.Sprintf(`for i in $(seq 0 %d); do cat %s/status-$i 2>/dev/null || echo "missing 0"; echo; done`,
			w.Chips-1, tpuStateDir))
	if err != nil {
		t.Fatalf("reading worker status files: %v", err)
	}
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 {
			t.Fatalf("malformed worker status line %q (full output: %q)", line, out)
		}
		step, err := strconv.Atoi(fields[1])
		if err != nil {
			t.Fatalf("malformed step in status line %q: %v", line, err)
		}
		states = append(states, fields[0])
		steps = append(steps, step)
	}
	if len(states) != w.Chips {
		t.Fatalf("got %d worker status lines, want %d: %q", len(states), w.Chips, out)
	}
	return states, steps
}
