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
	"strings"

	"github.com/spf13/cobra"
	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
)

var inspectCmd = &cobra.Command{
	Use:   "inspect",
	Short: "Inspect orchestrator and snapshot-agent deployment status",
	Long: `Checks the Kubernetes cluster to ensure both the accelerator-orchestrator deployment ` +
		`and snapshot-agent daemonset are deployed and running, and returns their image versions.`,
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
		fmt.Printf("Checking cluster: %s\n\n", clusterName)

		// 2. Create clientset
		clientset, err := kubernetes.NewForConfig(config)
		if err != nil {
			return fmt.Errorf("failed to create kubernetes client: %w", err)
		}

		ctx := context.Background()
		overallPassed := verifyCluster(ctx, clientset)

		if overallPassed {
			fmt.Println("Inspection Result: PASSED")
			return nil
		} else {
			fmt.Println("Inspection Result: FAILED")
			return fmt.Errorf("inspection failed")
		}
	},
}

func init() {
	rootCmd.AddCommand(inspectCmd)
}

// verifyCluster runs the verification checks against the cluster and prints the results.
// Returns true if all checks passed, false otherwise.
func verifyCluster(ctx context.Context, clientset *kubernetes.Clientset) bool {
	namespace := "timeslice-system" // Default namespace as per helm charts
	overallPassed := true

	// Helper to print check results
	printResult := func(name string, passed bool, details []string) {
		status := "[PASS]"
		if !passed {
			status = "[FAIL]"
			overallPassed = false
		}
		fmt.Printf("%s %s\n", status, name)
		for _, detail := range details {
			fmt.Printf("       - %s\n", detail)
		}
	}

	// 1. Verify Accelerator Orchestrator Deployment
	var orchPassed bool
	var orchDetails []string
	deployments, err := clientset.AppsV1().Deployments(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: "app.kubernetes.io/name=acceleratororchestrator",
	})

	var orchDep *appsv1.Deployment
	if err == nil && len(deployments.Items) > 0 {
		orchDep = &deployments.Items[0]
	} else if err == nil {
		// Fallback: search by name containing "acceleratororchestrator"
		allDeps, err := clientset.AppsV1().Deployments(namespace).List(ctx, metav1.ListOptions{})
		if err == nil {
			for i := range allDeps.Items {
				d := &allDeps.Items[i]
				if strings.Contains(d.Name, "acceleratororchestrator") {
					orchDep = d
					break
				}
			}
		}
	}

	switch {
	case err != nil:
		orchPassed = false
		orchDetails = []string{fmt.Sprintf("Error listing deployments: %v", err)}
	case orchDep == nil:
		orchPassed = false
		orchDetails = []string{fmt.Sprintf("Deployment not found in namespace %s", namespace)}
	default:
		ready := orchDep.Status.ReadyReplicas
		desired := int32(1)
		if orchDep.Spec.Replicas != nil {
			desired = *orchDep.Spec.Replicas
		}

		if ready > 0 {
			orchPassed = true
			var image string
			if len(orchDep.Spec.Template.Spec.Containers) > 0 {
				image = orchDep.Spec.Template.Spec.Containers[0].Image
			}
			orchDetails = []string{
				fmt.Sprintf("Status: Running (%d/%d ready replicas)", ready, desired),
				fmt.Sprintf("Image: %s", image),
			}
		} else {
			orchPassed = false
			orchDetails = []string{
				fmt.Sprintf("Status: Not Ready (%d/%d ready replicas)", ready, desired),
			}
		}
	}
	printResult("Accelerator Orchestrator Deployment", orchPassed, orchDetails)
	fmt.Println()

	// 2. Verify Snapshot Agent DaemonSet
	var agentPassed bool
	var agentDetails []string
	daemonsets, err := clientset.AppsV1().DaemonSets(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: "app.kubernetes.io/name=snapshot-agent",
	})

	var agentDS *appsv1.DaemonSet
	if err == nil && len(daemonsets.Items) > 0 {
		agentDS = &daemonsets.Items[0]
	} else if err == nil {
		// Fallback: search by name containing "snapshot-agent"
		allDS, err := clientset.AppsV1().DaemonSets(namespace).List(ctx, metav1.ListOptions{})
		if err == nil {
			for i := range allDS.Items {
				ds := &allDS.Items[i]
				if strings.Contains(ds.Name, "snapshot-agent") {
					agentDS = ds
					break
				}
			}
		}
	}

	switch {
	case err != nil:
		agentPassed = false
		agentDetails = []string{fmt.Sprintf("Error listing daemonsets: %v", err)}
	case agentDS == nil:
		agentPassed = false
		agentDetails = []string{fmt.Sprintf("DaemonSet not found in namespace %s", namespace)}
	default:
		ready := agentDS.Status.NumberReady
		if ready > 0 {
			agentPassed = true
			var image string
			if len(agentDS.Spec.Template.Spec.Containers) > 0 {
				image = agentDS.Spec.Template.Spec.Containers[0].Image
			}
			agentDetails = []string{
				fmt.Sprintf("Status: Running (%d ready replicas, at least 1 required)", ready),
				fmt.Sprintf("Image: %s", image),
			}
		} else {
			agentPassed = false
			agentDetails = []string{
				fmt.Sprintf("Status: Not Ready (%d ready replicas, at least 1 required)", ready),
			}
		}
	}
	printResult("Snapshot Agent DaemonSet", agentPassed, agentDetails)
	fmt.Println()

	// 3. Verify Node Labels
	nodePassed, nodeDetails := verifyNodeLabels(ctx, clientset)
	printResult("Labeled GPU Nodes", nodePassed, nodeDetails)
	fmt.Println()

	return overallPassed
}

// verifyNodeLabels checks node labels and enforces the one-node-per-group restriction.
//
//nolint:gocritic // named returns are forbidden by nonamedreturns linter in this project
func verifyNodeLabels(ctx context.Context, clientset kubernetes.Interface) (bool, []string) {
	nodes, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return false, []string{fmt.Sprintf("Error listing nodes: %v", err)}
	}

	const nodeLabelPrefix = "group.timeslice.io/"
	type LabeledNodeInfo struct {
		Name  string
		Group string
	}
	var labeledNodes []LabeledNodeInfo

	for i := range nodes.Items {
		node := &nodes.Items[i]
		for k, v := range node.Labels {
			if strings.HasPrefix(k, nodeLabelPrefix) && v == "true" {
				group := strings.TrimPrefix(k, nodeLabelPrefix)
				labeledNodes = append(labeledNodes, LabeledNodeInfo{
					Name:  node.Name,
					Group: group,
				})
				break
			}
		}
	}

	if len(labeledNodes) == 0 {
		return false, []string{
			"Status: Found 0 labeled nodes",
			"No nodes found with timeslice group label (group.timeslice.io/<group>=true)",
		}
	}

	// Check one-node-per-group restriction
	groupToNodes := make(map[string][]string)
	for _, ln := range labeledNodes {
		groupToNodes[ln.Group] = append(groupToNodes[ln.Group], ln.Name)
	}

	var violationDetails []string
	for group, nodeNames := range groupToNodes {
		if len(nodeNames) > 1 {
			violationDetails = append(violationDetails, fmt.Sprintf(
				"Violation: Group %q has %d nodes: %s (max 1 node per group supported due to DRA)",
				group, len(nodeNames), strings.Join(nodeNames, ", "),
			))
		}
	}

	if len(violationDetails) > 0 {
		nodeDetails := make([]string, 0, 1+len(labeledNodes)+len(violationDetails))
		nodeDetails = append(nodeDetails, fmt.Sprintf("Status: Found %d labeled nodes with group violations", len(labeledNodes)))
		for _, ln := range labeledNodes {
			nodeDetails = append(nodeDetails, fmt.Sprintf("Node: %s (group: %s)", ln.Name, ln.Group))
		}
		nodeDetails = append(nodeDetails, violationDetails...)
		return false, nodeDetails
	}

	nodeDetails := make([]string, 0, 1+len(labeledNodes))
	nodeDetails = append(nodeDetails, fmt.Sprintf("Status: Found %d labeled nodes", len(labeledNodes)))
	for _, ln := range labeledNodes {
		nodeDetails = append(nodeDetails, fmt.Sprintf("Node: %s (group: %s)", ln.Name, ln.Group))
	}
	return true, nodeDetails
}
