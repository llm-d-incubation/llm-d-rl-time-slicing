//go:build integration

package orchestrator

import (
	"context"
	"os"
	"testing"
	"time"

	orchpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/scenarios"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/tests/integration/harness"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// TestOrchestrator runs the shared E2E scenarios — the same code
// `rlts test orchestrator` runs — against the orchestrator deployed by the
// official Helm chart. run.sh installs the chart and labels TEST_NODE with
// the samplers/trainers groups the scenarios drive.
//
// The scenarios use fake RL jobs with pause pods (no GPU work), so run.sh
// points the chart at an agent port with no listener: job context stays
// unknown and the full lock protocol is exercised — the same mode
// `rlts test orchestrator` targets on orchestrator-only deployments. With a
// reachable agent, jobs without GPU activity are (correctly) reported idle
// and never considered loaded, so lock handoffs cannot progress; scenarios
// with agent-driven C/R need real GPU workloads and are future work.
func TestOrchestrator(t *testing.T) {
	if os.Getenv("ORCH_CHART_DEPLOYED") == "" {
		t.Skip("requires the chart-deployed orchestrator (run.sh --phase orchestrator)")
	}

	addr := os.Getenv("ORCH_ADDR")
	if addr == "" {
		t.Fatal("ORCH_ADDR env var is required")
	}
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dialing orchestrator at %s: %v", addr, err)
	}
	defer conn.Close()
	client := orchpb.NewAcceleratorOrchestratorServiceClient(conn)

	c := harness.NewCluster(t, "default")
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

	// Pre-clean leaked locks from previous runs (mirrors the rlts CLI);
	// Yield fails harmlessly when the job does not hold the lock.
	for _, groupID := range []string{"samplers", "trainers"} {
		for _, jobID := range []string{"my-rl-job", "job-a", "job-b"} {
			//nolint:errcheck // best-effort pre-clean
			_, _ = client.Yield(ctx, &orchpb.YieldRequest{GroupId: groupID, JobId: jobID})
		}
	}

	if err := scenarios.RunSingleRLJobScenario(ctx, c.Client, client, t, "pause", "pause"); err != nil {
		t.Fatalf("Single RL Job scenario: %v", err)
	}
	if err := scenarios.RunQueuedRLJobsScenario(ctx, c.Client, client, t, "pause", "pause"); err != nil {
		t.Fatalf("Queued RL Jobs scenario: %v", err)
	}
}
