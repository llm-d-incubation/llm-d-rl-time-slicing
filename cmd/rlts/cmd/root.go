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
	"github.com/spf13/cobra"
)

var orchestratorAddr string

var rootCmd = &cobra.Command{
	Use:   "rlts",
	Short: "rlts is a CLI for interacting with LLM-D RL Time Slicing",
	Long:  `A command line interface to interact with the LLM-D RL Time Slicing components, including the Accelerator Orchestrator.`,
}

func Execute() error {
	return rootCmd.Execute()
}

func init() {
	rootCmd.PersistentFlags().StringVar(&orchestratorAddr, "addr", "localhost:50051",
		"address of the accelerator orchestrator gRPC server")
}
