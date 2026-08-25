# Framework Integration Guide

Pre-packaged time-slicing integrations are available for supported frameworks — see [Slime](slime/) and [verl](verl/). If your framework is already supported, use the pre-packaged integration instead of following this guide.

This guide is for **framework owners and developers** who want to build a time-slicing integration for a framework that doesn't have one yet.

## Overview

Time-slicing works by acquiring and releasing GPU locks at **phase boundaries** in the training loop. Every phase that touches the GPU — initialization, training, generation, weight sync — must hold its pool's lock. Between phases (during idle waits), the lock is released so another job can use the GPU.

The platform handles GPU state transparently via [cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint): when a job yields its lock, the snapshot agent saves its GPU state to host memory; when it re-acquires, the state is restored. The framework doesn't need to manage GPU memory — it just signals when it starts and stops using the GPU.

## Step 1: Identify Phase Boundaries

Every RL training loop has these GPU-occupying phases:

| Phase | GPU Pool | What happens |
|-------|----------|-------------|
| **init** | both | Model loads onto GPU, NCCL groups created, initial weight sync |
| **generate** | sampler | Rollout/inference |
| **train** | trainer | Forward pass, backward pass, optimizer step |
| **weight_sync** | both | Updated weights broadcast from trainer to sampler (NCCL) |
| **save** | trainer | Checkpoint to disk |

## Step 2: Choose Your Integration Pattern

### Pattern A: PhaseCallback (recommended)

If the framework has a driver script that orchestrates phases, add a `PhaseCallback` interface:

```python
class PhaseCallback:
    def on_phase_begin(self, phase, role, context=None): pass
    def on_phase_end(self, phase, role, context=None): pass
    def close(self): pass
```

The driver calls `on_phase_begin` before each GPU phase and `on_phase_end` after. The time-slicing package provides a callback implementation that does lock acquire/release. The framework stays unaware of time-slicing.

**Example:** [Slime integration](slime/) uses `--phase-callback-path`

### Pattern B: Trainer subclass

If the framework has overridable lifecycle hooks, subclass the trainer and inject acquire/release:

```python
class TimeslicedTrainer(BaseTrainer):
    def __init__(self, config):
        self.locks = PhaseLocks.from_env()
        self.locks.ensure()           # acquire before init
        super().__init__(config)

    def on_step_begin(self):
        self.locks.ensure()           # acquire before step
        super().on_step_begin()

    def on_step_end(self):
        super().on_step_end()         # finish step (including weight sync)
        self.locks.drop_all()         # release after step
```

### Pattern C: Method patching (last resort)

If the framework has no hooks and no driver-level callback support, patch individual methods. Fragile — pins to specific internal method names.

## Step 3: Lock Protocol

### Roles and groups

- **Trainer pool** — GPUs running training. One orchestrator group shared by all jobs.
- **Sampler pool** — GPUs running generation. In sync mode, shared. In async mode, each job gets its own group (no contention).

### Lock ordering: trainer before sampler

When acquiring both locks, always acquire **trainer first, then sampler**. This prevents deadlocks.

### Sync RL pattern

Both pools time-sliced, whole-step turns.

### Async RL pattern

Trainer-only time-slicing, per-job sampler groups. See [Pattern C in the Orchestrator guide](../accelerator-orchestrator/#pattern-c-async-rl--trainer-only-time-slicing).

## Testing Checklist

1. Single job (no contention) — verify lock acquire/release in logs
2. Two jobs — verify lock handoff (`waited=Xms`, `context_restored=True`)
3. Confirm `update_weights` succeeds after context restore
4. Set `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0` as container-level env vars (not just runtime env) so they propagate to subprocesses
5. Ensure worker pod memory limits are large enough to hold checkpointed GPU state in host memory (at least 2× GPU memory footprint per time-sliced pool)

## Example: Manual Slime Integration

The following shows how the Slime framework was manually integrated before the pre-packaged `llm-d-timeslice-slime` solution was available. This pattern applies to any framework with a driver script.

### Step 1: Add Time-Slicing Command-Line Arguments
Add time-slicing configuration options to the framework's argument parser:

```python
parser.add_argument("--enable-timeslice", action="store_true", default=False)
parser.add_argument("--timeslice-orchestrator-addr", type=str,
    default="timeslice-acceleratororchestrator.timeslice-system.svc.cluster.local:50051")
parser.add_argument("--timeslice-job-id", type=str, default=None)
parser.add_argument("--timeslice-sampler-group", type=str, default="samplers")
parser.add_argument("--timeslice-trainer-group", type=str, default="trainers")
```

### Step 2: Initialize Client & Enforce Lock Order
Instantiate clients for both GPU groups. To prevent deadlocks, enforce a **Trainer-First lock hierarchy** — always acquire the Trainer lock before the Sampler lock:

```python
from timeslice import TimeSliceOrchestratorClient

sampler_client = TimeSliceOrchestratorClient(target=addr, job_id=job_id, group_id=sampler_group)
trainer_client = TimeSliceOrchestratorClient(target=addr, job_id=job_id, group_id=trainer_group)

# Acquire Trainer first (deadlock prevention)
trainer_client.acquire()
```

### Step 3: Wrap Rollout and Training Phases
Acquire and release GPU grants around each phase. Because weight synchronization (`update_weights`) uses GPU-to-GPU NCCL broadcast, **both locks must be held concurrently** during the transfer:

```python
for rollout_id in range(num_rollout):
    # Phase 1: Generation (Sampler GPU)
    rollout_data = ray.get(rollout_manager.generate.remote(rollout_id))
    sampler_client.release()

    # Phase 2: Training (Trainer GPU)
    trainer_client.acquire()
    actor_model.async_train(rollout_id, rollout_data)

    # Phase 3: Weight Sync (Both GPUs — NCCL broadcast)
    sampler_client.acquire()
    actor_model.update_weights()
    trainer_client.release()
```

> [!TIP]
> For a detailed walkthrough of all code changes in the manual Slime integration (categorized by scheduling, device fixes, and memory offloading), see **[manual-integration-example.md](manual-integration-example.md)**.
