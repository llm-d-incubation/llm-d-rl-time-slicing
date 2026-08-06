# Framework Integration Guide

How to integrate an RL training framework with GPU time-slicing. This guide is for **framework owners and developers** building a new time-slicing integration — not for end users of already-integrated frameworks (see the [framework-specific guides](../rl-frameworks/)).

## Overview

Time-slicing works by acquiring and releasing GPU locks at **phase boundaries** in the training loop. Every phase that touches the GPU — initialization, training, generation, weight sync — must hold its pool's lock. Between phases (during idle waits), the lock is released so another job can use the GPU.

The platform handles GPU state transparently via [cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint): when a job yields its lock, the snapshot agent saves its GPU state to host memory; when it re-acquires, the state is restored. The framework doesn't need to manage GPU memory — it just signals when it starts and stops using the GPU.

## Step 1: Identify Phase Boundaries

Every RL training loop has these GPU-occupying phases:

| Phase | GPU Pool | What happens |
|-------|----------|-------------|
| **init** | both | Model loads onto GPU, NCCL groups created, initial weight sync |
| **generate** | sampler | Rollout/inference (SGLang, vLLM, or colocated generation) |
| **train** | trainer | Forward pass, backward pass, optimizer step |
| **weight_sync** | both | Updated weights broadcast from trainer to sampler (NCCL) |
| **save** | trainer | Checkpoint to disk |
| **eval** | sampler | Validation inference |

**"both"** means the phase requires NCCL communication between trainer and sampler GPUs — both must be resident simultaneously.

## Step 2: Choose Your Integration Pattern

### Pattern A: PhaseCallback (recommended for new integrations)

If the framework has a driver script that orchestrates phases (like Slime's `train.py`), add a `PhaseCallback` interface:

```python
class PhaseCallback:
    def on_phase_begin(self, phase, role, context=None): pass
    def on_phase_end(self, phase, role, context=None): pass
    def close(self): pass
```

The driver calls `on_phase_begin` before each GPU phase and `on_phase_end` after. The time-slicing package provides a callback implementation that does lock acquire/release. The framework stays completely unaware of time-slicing.

**Used by:** [Slime integration](../rl-frameworks/slime/) (PhaseCallback fork)

### Pattern B: Trainer subclass (for frameworks with hook/registry systems)

If the framework has overridable lifecycle hooks (like verl's `on_step_begin`/`on_step_end`), subclass the trainer and inject acquire/release in the hooks:

```python
@register_trainer("sync_timesliced")
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

**Used by:** [verl integration](../rl-frameworks/verl/) (`trainer_mode=sync_timesliced`)

### Pattern C: Method patching (last resort)

If the framework has no hooks and no driver-level callback support, patch individual methods:

```python
original = Trainer._fit_update_weights
def patched(self, *args, **kwargs):
    locks.ensure()
    result = original(self, *args, **kwargs)
    locks.drop_all()
    return result
Trainer._fit_update_weights = patched
```

Fragile — pins to specific internal method names. Use only when Patterns A/B aren't possible.

**Used by:** verl fully-async experimental mode

## Step 3: Lock Protocol

### Roles and groups

- **Trainer pool** — the GPUs running training. One orchestrator group shared by all jobs (e.g., `trainers`).
- **Sampler pool** — the GPUs running generation. In sync mode, shared like the trainer pool. In async mode, each job gets its own group (no contention needed — samplers are always busy).

### Lock ordering: trainer before sampler

When acquiring both locks (for `init` or `weight_sync`), always acquire **trainer first, then sampler**. This prevents deadlocks — the only two wait shapes are "want trainer holding nothing" and "want sampler holding trainer."

### Sync RL pattern

```
on_step_begin   → ACQUIRE trainer + sampler
... generate + train + weight_sync ...
on_step_end     → RELEASE trainer + sampler
```

Whole-step turns. Simple but the GPU is locked for the entire step including generation.

### Async/disagg RL pattern

```
phase_begin("generate", "sampler")  → ACQUIRE sampler
... generation runs ...
phase_end("generate", "sampler")    → RELEASE sampler

phase_begin("train", "trainer")     → ACQUIRE trainer
... training runs ...
phase_end("train", "trainer")       → RELEASE trainer

phase_begin("weight_sync", "both")  → ACQUIRE trainer + sampler
... NCCL broadcast ...
phase_end("weight_sync", "both")    → RELEASE sampler (keep trainer)
```

Cross-pipeline: Job A trains while Job B generates. Only `weight_sync` serializes both pools.

### Async with dedicated samplers

For async RL where samplers are always busy, give each job its own sampler group:
- Job A: `TIMESLICE_TRAINER_GROUP=trainers TIMESLICE_SAMPLER_GROUP=samplers-a`
- Job B: `TIMESLICE_TRAINER_GROUP=trainers TIMESLICE_SAMPLER_GROUP=samplers-b`

The trainer pool is shared (time-sliced). The sampler pools have no contention — acquires return immediately.

## Step 4: Environment Variables

All time-slicing packages use the same env vars:

| Variable | Purpose |
|----------|---------|
| `TIMESLICE_JOB_ID` | Unique job identifier (e.g., `job-a`) |
| `TIMESLICE_ORCH_ADDR` | Orchestrator gRPC address |
| `TIMESLICE_GROUP` | Single group for colocated mode |
| `TIMESLICE_TRAINER_GROUP` | Trainer pool group for disagg mode |
| `TIMESLICE_SAMPLER_GROUP` | Sampler pool group for disagg mode |

When these are not set, the integration runs in **no-op mode** — zero overhead, same image works with and without the platform.

## Important: Do Not Use Cooperative Offloading

Framework-level GPU memory offloading (e.g., Slime's `--offload-train`, verl's `torch_memory_saver`) **conflicts with cuda-checkpoint**. The platform's snapshot agent handles GPU state save/restore at the OS level — the framework should not also try to manage GPU memory around phase boundaries.

If you use both, the framework's memory bookkeeping becomes stale after cuda-checkpoint restores the process, causing `cudaErrorIllegalAddress` crashes.

## Testing Checklist

1. Run a single job (no contention) — verify lock acquire/release in logs (`[timeslice]` lines)
2. Run two jobs — verify lock handoff (`waited=Xms`, `context_restored=True`)
3. Confirm `update_weights` (NCCL broadcast) succeeds after context restore
4. Check that `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0` are set (standard for cuda-checkpoint)
5. Verify no `--offload-train` or cooperative memory management flags

## Next Steps

- [Slime integration guide](../rl-frameworks/slime/) — PhaseCallback-based, zero code changes
- [verl integration guide](../rl-frameworks/verl/) — subclass-based, three modes
- [PhaseCallback formal specification](https://github.com/aishukamal/rl-time-slicing/blob/main/time-slicing-vertical-integrations/PHASE-CALLBACK-SPEC.md)
