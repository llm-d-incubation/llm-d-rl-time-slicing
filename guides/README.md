# Documentation & Guides

## What is time-slicing?

Multiple jobs cooperatively share the same GPU hardware by taking turns at natural phase boundaries. When Job A finishes training and waits for generation, the platform checkpoints its GPU state, restores Job B, and lets it train — filling idle time that would otherwise waste expensive accelerator cycles.

## Real-World Use Cases

### Multi-Tenant RL-as-a-Service
Platforms like **open-rl** or **Tinker** serve multiple users fine-tuning LLMs with RL. Time-slicing lets the platform pack more concurrent fine-tuning jobs onto the same GPU fleet — each user's job runs at phase boundaries without needing dedicated hardware.

### Batch RL Experiment Queues
A research team wants to run **4 RL experiments on 2 GPU nodes**. Instead of running them sequentially (4× the wall time) or buying more GPUs, time-slicing queues all 4 jobs and swaps them in/out at phase boundaries, maximizing GPU throughput.

### Dev/Prod GPU Sharing
Development RL jobs share GPUs with production training during off-peak hours. The orchestrator ensures mutual exclusion and clean context switching — no manual coordination, no OOM risks.

### Inference + Training Co-tenancy
A vLLM/SGLang serving workload shares GPUs with an RL training job. When the training job yields during its idle phase, the serving workload resumes inference. Both workloads get useful GPU time without dedicated hardware for each.

## Sync vs Async RL

**Sync RL** — training and generation alternate strictly. The trainer waits for all samples before starting the next step. Both trainer and sampler GPUs have idle gaps. Time-slicing shares **both pools** — Job B trains while Job A generates, then they swap.

**Async RL** — generation runs continuously while training proceeds on available batches. Sampler GPUs are busy non-stop, but the trainer GPU has idle gaps waiting for the next batch. Time-slicing shares only the **trainer pool**; each job gets dedicated sampler GPUs (separate group, no contention).

---

## Guides

### Platform Components

| Component | What it does | Guide |
|-----------|-------------|-------|
| **Accelerator Orchestrator** | Central gRPC service managing exclusive GPU access via lock queues | [Guide](accelerator-orchestrator/) |
| **Snapshot Agent** | Node-local DaemonSet performing GPU state checkpoint/restore via cuda-checkpoint | [Guide](snapshot-agent/) |
| **`timeslice` client** | Python library for acquire/release calls to the orchestrator | [README](../pkg/client/python/) |
| **Platform deployment** | Helm chart installing all components | [Deploy guide](../deploy/) |

### Framework Integrations

| Framework | Sync | Async | Integration Guide |
|-----------|------|-------|-------------------|
| **Slime** | [Sync example](rl-frameworks/slime/sync/) | [Async example](rl-frameworks/slime/async/) | [Slime guide](rl-frameworks/slime/) |
| **verl** | Validated (sync colocated + disagg) | Validated (fully-async) | [verl guide](rl-frameworks/verl/) |

### Building a New Integration

If you're integrating a framework not listed above, see the [Framework Integration Guide](framework-integration/) for the generic pattern: phase boundaries, lock protocol, sync vs async patterns.
