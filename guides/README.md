# Documentation & Guides

## What is time-slicing?

Multiple RL jobs cooperatively share the same GPU hardware by taking turns at natural phase boundaries. When Job A finishes training and waits for generation, the platform checkpoints its GPU state, restores Job B, and lets it train — filling idle time that would otherwise waste expensive accelerator cycles.

## When does it help?

Time-slicing delivers value when your workload has **significant GPU idle time** between phases. In disaggregated RL (separate trainer and sampler GPUs), the trainer sits idle while the sampler generates — typically 30-70% of wall time. Time-slicing fills that gap with another job's work.

It does **not** help when GPUs are already fully utilized (e.g., colocated training where the same GPU runs both training and generation back-to-back with no gap).

## Sync vs Async RL

**Sync RL** — training and generation alternate strictly. The trainer waits for all samples before starting the next step. Clear phase boundaries, predictable lock patterns.

**Async RL** — generation runs continuously while training proceeds on available batches. The trainer GPU has idle gaps (waiting for the next batch), but the sampler GPUs are busy non-stop. For time-slicing, only the **trainer pool** needs to be shared; samplers can be dedicated (each job gets its own sampler group, no contention).

---

## User Journeys

### I want to understand the components

| Component | What it does | Guide |
|-----------|-------------|-------|
| **Accelerator Orchestrator** | Central gRPC service managing exclusive GPU access via lock queues | [Accelerator Orchestrator Guide](accelerator-orchestrator/) |
| **Snapshot Agent** | Node-local DaemonSet performing GPU state checkpoint/restore | [Snapshot Agent Guide](snapshot-agent/) |
| **`timeslice` client** | Python library for interacting with the orchestrator and snapshot agent | [Python Client README](../pkg/client/python/) |

### I want to set up time-slicing on my cluster

Follow the deployment guides for each component:
1. [Deploy the platform](../deploy/) (Helm chart with orchestrator + snapshot agent + NVIDIA DRA driver)
2. [Configure the orchestrator](accelerator-orchestrator/) (node labels, taints, DRA ResourceClaims)
3. [Configure the snapshot agent](snapshot-agent/) (verify DaemonSet on GPU nodes)

### I want to integrate my RL framework

Start with the generic integration guide, then follow the framework-specific guide:

1. [Framework Integration Guide](framework-integration/) — phase boundaries, lock protocol, the PhaseCallback interface, sync vs async patterns
2. Framework-specific guides:
   - [Slime](rl-frameworks/slime/) — PhaseCallback integration, zero code changes
   - [verl](rl-frameworks/verl/) — subclass + `@register_trainer` integration
   - [Other frameworks](rl-frameworks/) — comparison table and status

### I want to run two RL jobs on shared GPUs

Pick your framework and mode:
- **Slime sync** — [Sync disaggregated example](rl-frameworks/slime/sync/) (2 GRPO jobs sharing trainer + sampler pools)
- **Slime async** — [Async disaggregated example](rl-frameworks/slime/async/) (trainer pool shared, samplers dedicated)
- **verl** — [verl guide](rl-frameworks/verl/) (sync colocated, sync disagg, or fully-async)
