//go:build integration

package orchestrator_test

import (
	"context"
	"os"
	"testing"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/timeslice-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/tests/integration/orchestrator/scenarios"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// TestOrchestrator is the composed orchestrator integration suite. It runs
// against BOTH official Helm charts deployed by run.sh (snapshot-agent +
// timeslice-orchestrator) on a real cluster with real GPU workloads.
//
// The orchestrator chart is installed with snapshotAgentPort matching
// CHART_AGENT_PORT so it commands the suite's own agent, not whatever
// might be running on the default port on a shared node.
//
// Scenario workloads deploy pods into dedicated integration test groups
// (integ-samplers, integ-trainers) labeled only on TEST_NODE.
func TestOrchestrator(t *testing.T) {
	if os.Getenv("ORCH_CHART_DEPLOYED") == "" {
		t.Skip("requires chart-deployed orchestrator (run.sh --phase orchestrator)")
	}

	h := NewComposedHarness(t)

	t.Run("SingleRLJob", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		defer cancel()

		conn, err := grpc.NewClient(
			h.OrchAddr,
			grpc.WithTransportCredentials(insecure.NewCredentials()),
		)
		if err != nil {
			t.Fatalf("dialing orchestrator at %s: %v", h.OrchAddr, err)
		}
		defer conn.Close()

		client := pb.NewTimeSliceOrchestratorServiceClient(conn)

		err = scenarios.RunSingleRLJobScenario(
			ctx,
			h.Client,
			client,
			t,
			"vllm",  // sampler template key
			"vllm",  // trainer template key
		)
		if err != nil {
			t.Fatalf("SingleRLJob scenario failed: %v", err)
		}
	})

	// QueuedRLJobs is omitted: the orchestrator has a cold-start bug where
	// groups initialise in IDLE state and the second job's Acquire deadlocks
	// waiting for a state transition that never fires. Tracked upstream.
}
