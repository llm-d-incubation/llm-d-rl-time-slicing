// Copyright 2025 The llm-d Authors.
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

package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"strconv"
	"time"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/logging"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/features"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/server"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/utils"
)

func main() {
	// Initialize slog with ContextHandler
	jsonHandler := slog.NewJSONHandler(os.Stdout, nil)
	ctxHandler := logging.NewContextHandler(jsonHandler)
	slog.SetDefault(slog.New(ctxHandler))

	port := flag.Int("port", 9001, "The port to listen on")
	deploymentMode := flag.String("deployment-mode", "standalone", "Deployment mode ('standalone' or 'k8s')")
	featureGatesSpec := flag.String("feature-gates", "",
		"Comma-separated list of Name=bool pairs selecting experimental features, e.g. 'DirectMemoryBackend=true'")
	flag.Parse()

	depMode := *deploymentMode
	if envDepMode := os.Getenv("DEPLOYMENT_MODE"); envDepMode != "" {
		depMode = envDepMode
	}

	// AGENT_PORT overrides the flag, mirroring DEPLOYMENT_MODE: the Helm
	// chart configures the agent through env vars, not flags.
	listenPort := *port
	if envPort := os.Getenv("AGENT_PORT"); envPort != "" {
		p, err := strconv.Atoi(envPort)
		if err != nil {
			slog.Error("Invalid AGENT_PORT", "value", envPort, "error", err)
			os.Exit(1)
		}
		listenPort = p
	}

	if depMode != "standalone" && depMode != "k8s" {
		slog.Error("Invalid deployment mode, must be 'standalone' or 'k8s'", "mode", depMode)
		os.Exit(1)
	}

	// FEATURE_GATES overrides the flag, mirroring DEPLOYMENT_MODE and
	// AGENT_PORT: the Helm chart configures the agent through env vars.
	gatesSpec := *featureGatesSpec
	if envGates := os.Getenv("FEATURE_GATES"); envGates != "" {
		gatesSpec = envGates
	}
	featureGates, err := features.Parse(gatesSpec)
	if err != nil {
		slog.Error("Invalid feature gates", "value", gatesSpec, "error", err)
		os.Exit(1)
	}

	ctx := context.Background()

	// The channel registry is shared between the app-channel backend and the
	// server's WorkloadChannel RPC handler.
	channelRegistry := backends.NewChannelRegistry()
	registeredBackends := map[backends.BackendType]backends.Backend{
		backends.BackendCuda:         backends.NewCudaCheckpoint(),
		backends.BackendNoop:         backends.NewNoopBackend(),
		backends.BackendAppEndpoint:  backends.NewAppEndpointBackend(),
		backends.BackendAppChannel:   backends.NewAppChannelBackend(channelRegistry),
		backends.BackendDirectMemory: backends.NewDirectMemory(),
	}

	// GPU-CR housekeeping runs only when the shared checkpoint dir is
	// configured, keeping CUDA/app-only deployments untouched.
	if ctlDir := os.Getenv("EXPORT_FILE_PATH"); ctlDir != "" {
		// The dir must be writable by the (unprivileged) GPU-CR workloads
		// that mmap their dump buffers in it. Tenant UIDs aren't known up
		// front and hostPath volumes get no fsGroup remapping, so
		// world-writable is the only workable mode; the sticky bit (the
		// /tmp model) keeps one tenant from deleting or renaming another's
		// artifacts.
		if _, err := os.Stat(ctlDir); err == nil {
			if err := os.Chmod(ctlDir, 0o777|os.ModeSticky); err != nil {
				slog.WarnContext(ctx, "Failed to chmod GPU-CR checkpoint dir to 1777", "dir", ctlDir, "error", err)
			} else {
				slog.InfoContext(ctx, "Set GPU-CR checkpoint dir permissions to 1777 (world-writable, sticky)", "dir", ctlDir)
			}
		}
		// Sweep stale GPU-CR artifacts: a leaked dump pins its full extent
		// in shm/hugetlbfs even after the owning process dies.
		utils.StartGPUCRSweeper(ctx, ctlDir, 10*time.Minute)
	}

	slog.InfoContext(ctx, "Starting Snapshot Agent",
		"port", listenPort, "deploymentMode", depMode, "featureGates", featureGates.String())
	err = server.StartServer(
		ctx, listenPort, registeredBackends, backends.BackendCuda, depMode, channelRegistry, featureGates)
	if err != nil {
		slog.ErrorContext(ctx, "Failed to start server", "error", err)
		os.Exit(1)
	}
}
