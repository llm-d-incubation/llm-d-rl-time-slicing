// Copyright 2026 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cmd

import (
	"context"
	"fmt"
	"time"

	"github.com/spf13/cobra"
	google_grpc "google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/test/e2e/acceleratororchestrator"
)

var testCmd = &cobra.Command{
	Use:   "test",
	Short: "Run tests for the orchestrator",
}

var orchestratorTestCmd = &cobra.Command{
	Use:   "orchestrator",
	Short: "Run E2E scenario tests against the cluster",
	Long: `Runs the E2E scenario tests (Single RL Job and Queued RL Jobs) ` +
		`against the active Kubernetes cluster and the deployed Accelerator Orchestrator.`,
	SilenceUsage:  true,
	SilenceErrors: true,
	RunE: func(cmd *cobra.Command, args []string) error {
		// 1. Load kubeconfig
		loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
		configOverrides := &clientcmd.ConfigOverrides{}
		kubeConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, configOverrides)
		config, err := kubeConfig.ClientConfig()
		if err != nil {
			return fmt.Errorf("failed to load kubeconfig: %w", err)
		}

		// Get current context name for reporting
		clusterName := "unknown"
		if rawConfig, err := kubeConfig.RawConfig(); err == nil {
			clusterName = rawConfig.CurrentContext
		}
		fmt.Printf("Target Cluster: %s\n\n", clusterName)

		// 2. Create clientset
		clientset, err := kubernetes.NewForConfig(config)
		if err != nil {
			return fmt.Errorf("failed to create kubernetes client: %w", err)
		}

		ctx := context.Background()

		// 3. STEP 1: General Prerequisite Check (Verify Deployment)
		fmt.Println("=== Step 1: Running Prerequisite Verification ===")
		overallPassed := verifyCluster(ctx, clientset)
		if !overallPassed {
			return fmt.Errorf("prerequisite check failed: orchestrator/agents are not ready or no labeled nodes found. Aborting E2E tests")
		}
		fmt.Printf("Prerequisite Verification: PASSED\n\n")

		// 4. STEP 2: Specific E2E Requirement Check (samplers and trainers groups)
		fmt.Println("=== Step 2: Checking E2E Specific Node Requirements ===")
		nodes, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
		if err != nil {
			return fmt.Errorf("failed to list nodes: %w", err)
		}

		var samplerNode string
		var trainerNode string
		for _, node := range nodes.Items {
			for k, v := range node.Labels {
				if v == "true" {
					if k == "group.timeslice.io/samplers" && samplerNode == "" {
						samplerNode = node.Name
					}
					if k == "group.timeslice.io/trainers" && trainerNode == "" {
						trainerNode = node.Name
					}
				}
			}
		}

		reqsPassed := true
		if samplerNode != "" {
			fmt.Printf("[PASS] Group \"samplers\" has labeled nodes (Found: %s)\n", samplerNode)
		} else {
			fmt.Println("[FAIL] Group \"samplers\" has no labeled nodes (required label: group.timeslice.io/samplers=true)")
			reqsPassed = false
		}

		if trainerNode != "" {
			fmt.Printf("[PASS] Group \"trainers\" has labeled nodes (Found: %s)\n", trainerNode)
		} else {
			fmt.Println("[FAIL] Group \"trainers\" has no labeled nodes (required label: group.timeslice.io/trainers=true)")
			reqsPassed = false
		}
		fmt.Println()

		if !reqsPassed {
			return fmt.Errorf("E2E node requirements check failed. Aborting E2E tests")
		}

		// 5. STEP 3: Connect to Orchestrator gRPC Server
		fmt.Printf("=== Step 3: Connecting to Orchestrator at %s ===\n", orchestratorAddr)
		conn, err := google_grpc.NewClient(orchestratorAddr,
			google_grpc.WithTransportCredentials(insecure.NewCredentials()),
		)
		if err != nil {
			return fmt.Errorf("failed to create orchestrator client: %w", err)
		}
		defer conn.Close()
		client := pb.NewAcceleratorOrchestratorServiceClient(conn)

		// Ping the orchestrator with a quick call to ensure it's reachable
		pingCtx, pingCancel := context.WithTimeout(ctx, 5*time.Second)
		defer pingCancel()
		_, err = client.ListGroups(pingCtx, &pb.ListGroupsRequest{})
		if err != nil {
			return fmt.Errorf("failed to connect to orchestrator at %s: %w "+
				"(make sure to port-forward if necessary: "+
				"kubectl port-forward svc/timeslice-acceleratororchestrator 50051:50051 -n timeslice-system)",
				orchestratorAddr, err)
		}
		fmt.Printf("Connected successfully.\n\n")

		// Pre-clean: Force yield any locks from previous failed runs for the test jobs
		fmt.Println("=== Pre-cleaning leaked locks from previous runs ===")
		testJobs := []string{"my-rl-job", "job-a", "job-b"}
		testGroups := []string{"samplers", "trainers"}
		for _, groupID := range testGroups {
			for _, jobID := range testJobs {
				// We ignore errors because the job might not hold the lock,
				// in which case Yield returns PermissionDenied, which is expected.
				//nolint:errcheck // Pre-clean is best-effort, ignore if job doesn't hold lock
				_, _ = client.Yield(ctx, &pb.YieldRequest{
					GroupId: groupID,
					JobId:   jobID,
				})
			}
		}
		fmt.Printf("Pre-clean complete.\n\n")

		// 6. STEP 4: Run E2E Scenarios
		fmt.Println("=== Step 4: Running E2E Scenarios ===")
		cliLogger := &cliLogger{}

		// Scenario 1: Single RL Job
		fmt.Println("--- Running Scenario: Single RL Job ---")
		err = acceleratororchestrator.RunSingleRLJobScenario(ctx, clientset, client, cliLogger, samplerTemplateKey, trainerTemplateKey)
		scenario1Passed := err == nil
		if !scenario1Passed {
			fmt.Printf("[FAIL] Single RL Job Scenario failed: %v\n\n", err)
		} else {
			fmt.Printf("[PASS] Single RL Job Scenario completed successfully\n\n")
		}

		// Scenario 2: Queued RL Jobs
		fmt.Println("--- Running Scenario: Queued RL Jobs ---")
		err = acceleratororchestrator.RunQueuedRLJobsScenario(ctx, clientset, client, cliLogger, samplerTemplateKey, trainerTemplateKey)
		scenario2Passed := err == nil
		if !scenario2Passed {
			fmt.Printf("[FAIL] Queued RL Jobs Scenario failed: %v\n\n", err)
		} else {
			fmt.Printf("[PASS] Queued RL Jobs Scenario completed successfully\n\n")
		}

		// 7. STEP 5: Final Report
		fmt.Println("=== E2E Test Final Summary ===")
		if scenario1Passed {
			fmt.Println("[PASS] Single RL Job Scenario")
		} else {
			fmt.Println("[FAIL] Single RL Job Scenario")
		}
		if scenario2Passed {
			fmt.Println("[PASS] Queued RL Jobs Scenario")
		} else {
			fmt.Println("[FAIL] Queued RL Jobs Scenario")
		}
		fmt.Println()

		if scenario1Passed && scenario2Passed {
			fmt.Println("E2E Test Result: PASSED")
			return nil
		} else {
			fmt.Println("E2E Test Result: FAILED")
			return fmt.Errorf("one or more E2E scenarios failed")
		}
	},
}

var (
	samplerTemplateKey string
	trainerTemplateKey string
)

func init() {
	rootCmd.AddCommand(testCmd)
	testCmd.AddCommand(orchestratorTestCmd)

	orchestratorTestCmd.Flags().StringVar(&samplerTemplateKey, "sampler-template-key", "",
		"Name of the Kubernetes PodTemplate to use for sampler pods (blank for default pause pod)")
	orchestratorTestCmd.Flags().StringVar(&trainerTemplateKey, "trainer-template-key", "",
		"Name of the Kubernetes PodTemplate to use for trainer pods (blank for default pause pod)")
}

// cliLogger implements acceleratororchestrator.Logger interface to print to stdout.
type cliLogger struct{}

func (c *cliLogger) Log(args ...interface{}) {
	fmt.Print("  [INFO] ")
	fmt.Println(args...)
}

func (c *cliLogger) Logf(format string, args ...interface{}) {
	fmt.Print("  [INFO] ")
	fmt.Printf(format, args...)
	fmt.Println()
}

func (c *cliLogger) Error(args ...interface{}) {
	fmt.Print("  [ERROR] ")
	fmt.Println(args...)
}

func (c *cliLogger) Errorf(format string, args ...interface{}) {
	fmt.Print("  [ERROR] ")
	fmt.Printf(format, args...)
	fmt.Println()
}
