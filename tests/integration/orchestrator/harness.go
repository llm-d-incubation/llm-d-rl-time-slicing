//go:build integration

// Package orchestrator_test provides the composed orchestrator integration
// suite harness. It attaches to chart-deployed orchestrator and snapshot-agent
// instances (installed by run.sh), sets up dedicated integration test groups
// (integ-samplers, integ-trainers) on TEST_NODE, and provides workload hooks
// and assertion helpers.
//
// Like the snapshot-agent harness, everything runs INSIDE the cluster from
// the test-runner pod.
package orchestrator_test

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/tests/integration/harness"
)

const (
	// integSamplers and integTrainers are the group labels the suite applies
	// to TEST_NODE. The scenario code (scenarios/) hardcodes these group IDs
	// in its Acquire/Yield calls and node lookups, so they must match.
	// Safety on shared clusters comes from labeling ONLY TEST_NODE, not from
	// using distinct group names (the preflight check in run.sh ensures
	// TEST_NODE is set).
	integSamplers = "samplers"
	integTrainers = "trainers"

	// chartNamespace is where the official Helm charts install components.
	chartNamespace = "timeslice-system"

	orchPodTimeout = 5 * time.Minute
)

// ComposedHarness manages the composed orchestrator + snapshot-agent test
// stack. Both components are deployed by run.sh via their official Helm
// charts; this harness attaches to them and sets up the test environment.
//
// The canonical topology uses separate nodes for each group (one GPU per
// node): TEST_NODE_SAMPLERS for the samplers group, TEST_NODE_TRAINERS
// for the trainers group.
type ComposedHarness struct {
	*harness.Cluster
	SamplerNode string
	TrainerNode string
	OrchAddr    string // host:port of the orchestrator gRPC service
}

// NewComposedHarness connects to the cluster, attaches to the chart-deployed
// orchestrator (via its ClusterIP Service) and snapshot-agent, labels the
// group nodes, and registers cleanup.
//
// Requires TEST_NODE_SAMPLERS and TEST_NODE_TRAINERS (separate GPU nodes,
// one per group — the canonical deployment topology).
func NewComposedHarness(t *testing.T) *ComposedHarness {
	t.Helper()

	samplerNode := os.Getenv("TEST_NODE_SAMPLERS")
	trainerNode := os.Getenv("TEST_NODE_TRAINERS")
	if samplerNode == "" || trainerNode == "" {
		t.Fatal("TEST_NODE_SAMPLERS and TEST_NODE_TRAINERS are required (one GPU node per group)")
	}
	if samplerNode == trainerNode {
		t.Fatal("TEST_NODE_SAMPLERS and TEST_NODE_TRAINERS must be different nodes")
	}

	c := harness.NewCluster(t, "default")

	h := &ComposedHarness{
		Cluster:     c,
		SamplerNode: samplerNode,
		TrainerNode: trainerNode,
	}

	// Resolve the orchestrator Service address.
	orchPort := 50051
	if p := os.Getenv("ORCH_PORT"); p != "" {
		port, err := strconv.Atoi(p)
		if err != nil {
			t.Fatalf("invalid ORCH_PORT %q: %v", p, err)
		}
		orchPort = port
	}
	h.OrchAddr = fmt.Sprintf("orch-chart-test.%s.svc.cluster.local:%d", chartNamespace, orchPort)

	// Wait for the orchestrator pod to be Ready.
	orchSelector := "app.kubernetes.io/name=timesliceorchestrator,app.kubernetes.io/instance=orch-chart-test"
	h.WaitPodReadyByLabel(t, chartNamespace, orchSelector, "", orchPodTimeout)
	t.Logf("orchestrator chart pod ready, reachable at %s", h.OrchAddr)

	// Wait for snapshot-agent chart pods to be Ready on BOTH group nodes.
	saSelector := "app.kubernetes.io/name=snapshot-agent,app.kubernetes.io/instance=sa-chart-test"
	agentPort := 9001
	if p := os.Getenv("CHART_AGENT_PORT"); p != "" {
		port, err := strconv.Atoi(p)
		if err != nil {
			t.Fatalf("invalid CHART_AGENT_PORT %q: %v", p, err)
		}
		agentPort = port
	}
	for _, node := range []string{samplerNode, trainerNode} {
		ip := h.WaitPodReadyByLabel(t, chartNamespace, saSelector, node, orchPodTimeout)
		t.Logf("snapshot-agent chart pod ready at %s:%d on %s", ip, agentPort, node)
	}

	// Label each group node exclusively — unless the run validates label-free
	// membership (INTEG_EMERGENT set), in which case NO node labels may
	// exist: the orchestrator must create groups on first Acquire and
	// discover their nodes from scheduled pods.
	if os.Getenv("INTEG_EMERGENT") == "" {
		h.exclusiveLabel(t, samplerNode, integSamplers)
		h.exclusiveLabel(t, trainerNode, integTrainers)
	} else {
		t.Log("INTEG_EMERGENT set: skipping group node labeling (label-free membership mode)")
	}

	// Pre-clean: remove any leaked pods from a previous failed run.
	h.cleanLeakedPods(t)

	return h
}

// exclusiveLabel ensures TEST_NODE is the ONLY node with the given group
// label: it removes the label from all other nodes (restoring on cleanup)
// and adds it to TEST_NODE (removing on cleanup).
func (h *ComposedHarness) exclusiveLabel(t *testing.T, testNode, group string) {
	t.Helper()
	ctx := context.Background()
	labelKey := fmt.Sprintf("group.timeslice.io/%s", group)

	nodes, err := h.Client.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("listing nodes: %v", err)
	}
	for i := range nodes.Items {
		n := &nodes.Items[i]
		if n.Name == testNode {
			continue
		}
		if n.Labels[labelKey] != "true" {
			continue
		}
		delete(n.Labels, labelKey)
		if _, err := h.Client.CoreV1().Nodes().Update(ctx, n, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("removing label %s from node %s: %v", labelKey, n.Name, err)
		}
		t.Logf("temporarily removed label %s from node %s", labelKey, n.Name)
		nodeName := n.Name
		t.Cleanup(func() {
			nn, err := h.Client.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
			if err != nil {
				t.Logf("WARNING: could not restore label %s on %s: %v", labelKey, nodeName, err)
				return
			}
			nn.Labels[labelKey] = "true"
			if _, err := h.Client.CoreV1().Nodes().Update(ctx, nn, metav1.UpdateOptions{}); err != nil {
				t.Logf("WARNING: could not restore label %s on %s: %v", labelKey, nodeName, err)
			} else {
				t.Logf("restored label %s on node %s", labelKey, nodeName)
			}
		})
	}
	h.labelNode(t, testNode, group)
}

// labelNode adds a group.timeslice.io/<group>=true label to the node and
// registers cleanup to remove it.
func (h *ComposedHarness) labelNode(t *testing.T, node, group string) {
	t.Helper()
	ctx := context.Background()
	labelKey := fmt.Sprintf("group.timeslice.io/%s", group)

	n, err := h.Client.CoreV1().Nodes().Get(ctx, node, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("getting node %s: %v", node, err)
	}
	if n.Labels == nil {
		n.Labels = make(map[string]string)
	}
	n.Labels[labelKey] = "true"
	if _, err := h.Client.CoreV1().Nodes().Update(ctx, n, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("labeling node %s with %s: %v", node, labelKey, err)
	}
	t.Logf("labeled node %s: %s=true", node, labelKey)

	t.Cleanup(func() {
		n, err := h.Client.CoreV1().Nodes().Get(ctx, node, metav1.GetOptions{})
		if err != nil {
			t.Logf("WARNING: could not get node %s for label cleanup: %v", node, err)
			return
		}
		delete(n.Labels, labelKey)
		if _, err := h.Client.CoreV1().Nodes().Update(ctx, n, metav1.UpdateOptions{}); err != nil {
			t.Logf("WARNING: could not remove label %s from node %s: %v", labelKey, node, err)
		} else {
			t.Logf("removed label %s from node %s", labelKey, node)
		}
	})
}

// cleanLeakedPods deletes any pods from a previous failed run that carry the
// timeslice.io/group label in the default namespace.
func (h *ComposedHarness) cleanLeakedPods(t *testing.T) {
	t.Helper()
	ctx := context.Background()
	for _, group := range []string{integSamplers, integTrainers} {
		pods, err := h.Client.CoreV1().Pods("default").List(ctx, metav1.ListOptions{
			LabelSelector: fmt.Sprintf("timeslice.io/group=%s", group),
		})
		if err != nil {
			t.Logf("WARNING: listing leaked pods for group %s: %v", group, err)
			continue
		}
		for i := range pods.Items {
			name := pods.Items[i].Name
			if err := h.Client.CoreV1().Pods("default").Delete(ctx, name, metav1.DeleteOptions{}); err != nil {
				t.Logf("WARNING: deleting leaked pod %s: %v", name, err)
			} else {
				t.Logf("cleaned leaked pod %s", name)
			}
		}
	}
}
