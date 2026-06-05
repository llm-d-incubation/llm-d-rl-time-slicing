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
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
)

var orchestratorCmd = &cobra.Command{
	Use:   "orchestrator",
	Short: "Interact with the Accelerator Orchestrator",
	Long:  `Manage and query the Accelerator Orchestrator.`,
}

func init() {
	rootCmd.AddCommand(orchestratorCmd)
	orchestratorCmd.AddCommand(listGroupsCmd)
	orchestratorCmd.AddCommand(statusCmd)
	orchestratorCmd.AddCommand(acquireCmd)
	orchestratorCmd.AddCommand(yieldCmd)
}

func getClient() (pb.AcceleratorOrchestratorServiceClient, *grpc.ClientConn, error) {
	conn, err := grpc.NewClient(orchestratorAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, nil, fmt.Errorf("did not connect: %w", err)
	}
	return pb.NewAcceleratorOrchestratorServiceClient(conn), conn, nil
}

var listGroupsCmd = &cobra.Command{
	Use:   "list",
	Short: "List all active time-slice groups",
	RunE: func(cmd *cobra.Command, args []string) error {
		client, conn, err := getClient()
		if err != nil {
			return err
		}
		defer conn.Close()

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		resp, err := client.ListGroups(ctx, &pb.ListGroupsRequest{})
		if err != nil {
			return fmt.Errorf("failed to list groups: %w", err)
		}

		if len(resp.GroupIds) == 0 {
			fmt.Println("No active groups found.")
			return nil
		}

		fmt.Println("Active Groups:")
		for _, id := range resp.GroupIds {
			fmt.Printf("- %s\n", id)
		}
		return nil
	},
}

var statusCmd = &cobra.Command{
	Use:   "status [group-id]",
	Short: "Get status of a time-slice group",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		groupID := args[0]
		client, conn, err := getClient()
		if err != nil {
			return err
		}
		defer conn.Close()

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		resp, err := client.GetGroupStatus(ctx, &pb.GetGroupStatusRequest{GroupId: groupID})
		if err != nil {
			return fmt.Errorf("failed to get status: %w", err)
		}

<<<<<<< HEAD
		if resp.Group == nil {
			return fmt.Errorf("server returned success but group status is missing")
		}

=======
>>>>>>> 60e3183 (Add cli and update deployment)
		fmt.Printf("Group: %s\n", resp.Group.GroupId)
		fmt.Printf("  State: %s\n", resp.Group.GroupState)
		if resp.Group.StateTimestamp != nil {
			fmt.Printf("  State Timestamp: %s\n", resp.Group.StateTimestamp.AsTime())
		}
		fmt.Printf("  Locking Job: %s\n", resp.Group.LockingJob)
		fmt.Printf("  Active Job: %s\n", resp.Group.ActiveJob)
		fmt.Printf("  Waiter Queue Depth: %d\n", resp.Group.WaiterQueueDepth)

		if len(resp.AgentJobStates) > 0 {
			fmt.Println("  Agent Job States:")
			for _, state := range resp.AgentJobStates {
				fmt.Printf("    - Agent: %s, State: %s\n", state.Agent, state.JobState)
			}
		}
		return nil
	},
}

var acquireCmd = &cobra.Command{
	Use:   "acquire [group-id] [job-id]",
	Short: "Acquire exclusive access to a group (blocking)",
	Args:  cobra.ExactArgs(2),
	RunE: func(cmd *cobra.Command, args []string) error {
		groupID := args[0]
		jobID := args[1]

		client, conn, err := getClient()
		if err != nil {
			return err
		}
		defer conn.Close()

		// Acquire can block, so we use context.Background() to allow blocking.
		ctx := context.Background()

		fmt.Printf("Requesting access to group %s for job %s (blocking)...\n", groupID, jobID)
		resp, err := client.Acquire(ctx, &pb.AcquireRequest{GroupId: groupID, JobId: jobID})
		if err != nil {
			return fmt.Errorf("failed to acquire: %w", err)
		}

		fmt.Printf("Acquire result: Success=%t, Waited=%dms, ContextRestored=%t\n",
			resp.Success, resp.WaitedMs, resp.ContextRestored)
		return nil
	},
}

var yieldCmd = &cobra.Command{
	Use:   "yield [group-id] [job-id]",
	Short: "Yield exclusive access to a group",
	Args:  cobra.ExactArgs(2),
	RunE: func(cmd *cobra.Command, args []string) error {
		groupID := args[0]
		jobID := args[1]

		client, conn, err := getClient()
		if err != nil {
			return err
		}
		defer conn.Close()

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		resp, err := client.Yield(ctx, &pb.YieldRequest{GroupId: groupID, JobId: jobID})
		if err != nil {
			return fmt.Errorf("failed to yield: %w", err)
		}

		fmt.Printf("Yield result: Success=%t, PendingWaiters=%d, SnapshotDeferred=%t\n",
			resp.Success, resp.PendingWaiters, resp.SnapshotDeferred)
		return nil
	},
}
