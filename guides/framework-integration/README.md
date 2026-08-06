# Framework Integration Guide

How to integrate an RL training framework with GPU time-slicing. This guide is for **framework owners and developers** building a new integration — not for end users of already-integrated frameworks (see [Slime](../rl-frameworks/slime/)).

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

**Example:** [Slime integration](../rl-frameworks/slime/) uses `--phase-callback-path`

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

## Important: Do Not Use Cooperative Offloading

Framework-level GPU memory offloading (`--offload-train`, `torch_memory_saver`) **conflicts with cuda-checkpoint**. The platform handles GPU state at the OS level — the framework should not also try to manage GPU memory around phase boundaries.

## Testing Checklist

1. Single job (no contention) — verify lock acquire/release in logs
2. Two jobs — verify lock handoff (`waited=Xms`, `context_restored=True`)
3. Confirm `update_weights` succeeds after context restore
4. Set `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0`
5. No `--offload-train` or cooperative memory management flags

## Case Study: Manual Slime Integration

For a detailed example of manually integrating a framework (before the PhaseCallback approach existed), see [SLIME_CHANGES.md](examples/SLIME_CHANGES.md) — a line-by-line walkthrough of adding orchestrator client calls to Slime's driver scripts.
