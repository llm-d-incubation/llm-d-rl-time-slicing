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

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/logging"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/server"
)

func main() {
	// Initialize slog with ContextHandler
	jsonHandler := slog.NewJSONHandler(os.Stdout, nil)
	ctxHandler := logging.NewContextHandler(jsonHandler)
	slog.SetDefault(slog.New(ctxHandler))

	port := flag.Int("port", 9001, "The port to listen on")
	flag.Parse()

	ctx := context.Background()

	registeredBackends := map[backends.BackendType]backends.Backend{
		backends.BackendCuda: backends.NewCudaCheckpoint(),
		backends.BackendNoop: backends.NewNoopBackend(),
	}

	slog.InfoContext(ctx, "Starting Snapshot Agent", "port", *port)
	if err := server.StartServer(ctx, *port, registeredBackends, backends.BackendCuda); err != nil {
		slog.ErrorContext(ctx, "Failed to start server", "error", err)
		os.Exit(1)
	}
}
