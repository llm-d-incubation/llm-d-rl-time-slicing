[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/llm-d-incubation/llm-d-rl-time-slicing/blob/main/LICENSE)
[![Join Slack](https://img.shields.io/badge/Join_Slack-blue?logo=slack)](https://llm-d.ai/slack)

# Time-Slicing for Reinforcement Learning Workloads
  > **Current Project Status & Roadmap:**
  > * **Snapshot Agent (GPU):** Available today for standalone integration.
  > * **Accelerator Orchestrator:** In active development.
## The Problem: Accelerator Underutilization
  Reinforcement learning (RL) workloads spend a significant fraction of their lifecycle idle—waiting on reward evaluation, generation stragglers, or synchronization steps. Across large-scale fleets, this leaves expensive accelerator hardware **underutilized 45–66% of the time**, even though the underlying RL math doesn't require it.
  
  ## The Solution: Platform-Level Sharing
  **llm-d-rl-time-slicing** moves the utilization fix from the application layer to the platform layer. Multiple independent RL jobs cooperatively share the same accelerator hardware, swapping during each job's natural blocking phases (generation, training, weight sync) rather than holding the accelerator idle. 
  
  **Your training loop stays exactly the same — no algorithmic rewrites required.**
  
  ## How It Works
  We introduce **collaborative, application-aware time-slicing**. Using a lightweight client library that pairs seamlessly with your existing training and inference frameworks, the system delivers two core capabilities:
  * **Intelligent Scheduling:** Dynamically coordinates accelerator access across concurrent jobs based on their execution phases.
  * **Fast Context Switching:** Performs fast, transparent state checkpointing and restoration under the hood.

For the full design rationale and preliminary benchmark results, see the [Platform-Native Time-Slicing proposal](https://github.com/llm-d/llm-d/blob/main/docs/proposals/rl-time-slicing-platform.md).

## Architecture

![Architecture](https://github.com/llm-d-incubation/llm-d-rl-time-slicing/blob/main/docs/diagrams/time-slicing-architecture-diagram.png?raw=true)

This architecture consists of the following foundational components:

- **Snapshot Agent**: A node-local daemon, deployed as a Kubernetes DaemonSet, that performs the actual checkpoint/restore of accelerator state for a job. It supports a pluggable backend model, with backends specific to the underlying accelerator and checkpoint mechanism.
- **Accelerator Orchestrator**: A central coordinator that manages exclusive accelerator access across co-located jobs. It persists lock state for crash recovery and exposes a gRPC API (`Acquire`/`Yield`) that frameworks invoke at natural phase boundaries.
- **`timeslice` client**: A lightweight library used by training and inference services to interact seamlessly with the Snapshot Agent and Accelerator Orchestrator without needing to manage raw gRPC calls directly.

## Modes of Operation

- **Cooperative Accelerator Time-Slicing**: The Accelerator Orchestrator coordinates multiple jobs sharing a cluster of accelerator nodes, granting and reclaiming hardware access at each job's natural yield points.
- **Standalone Snapshot Agent Integration**: Training services that already implement their own scheduling (e.g., tinker-style architectures) can interface directly with the Snapshot Agent's checkpoint/restore primitives, bypassing the orchestrator entirely.

For step-by-step instructions, installation walkthroughs, and API references, explore our [Documentation & Guides](./guides).

## Contributing

Start with the [llm-d organization contributing guide](https://github.com/llm-d/llm-d/blob/main/CONTRIBUTING.md) for project-wide guidelines, code of conduct, and community resources.

We currently use the llm-d Slack workspace for communication — join via [llm-d.ai/slack](https://llm-d.ai/slack).

For large changes, please [open an issue](https://github.com/llm-d-incubation/llm-d-rl-time-slicing/issues/new) first describing the change so maintainers can do an assessment. See [DEVELOPMENT.md](./DEVELOPMENT.md) for details on building, testing, and working with the codebase.

All commits must be signed off (DCO) — see [PR_SIGNOFF.md](./PR_SIGNOFF.md) for instructions.

Contributions are welcome!

## Security

To report a security vulnerability, please see [SECURITY.md](./SECURITY.md).

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](./LICENSE) for details.
