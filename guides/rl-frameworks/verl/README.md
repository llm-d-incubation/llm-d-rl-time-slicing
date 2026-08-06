# verl Time-Slicing Integration

Integrate [verl](https://github.com/verl-project/verl) RL training with GPU time-slicing. Supports three modes covering verl's sync, disaggregated, and fully-async trainer architectures.

## Modes

| Mode | verl Trainer | Lock Type | Integration |
|------|-------------|-----------|-------------|
| **Sync colocated** | `sync` (v1) | Single group | Subclass + `@register_trainer("sync_timesliced")` |
| **Sync disaggregated** | `separate_async` (v1) | Dual role locks | Subclass + `@register_trainer("separate_async_timesliced")` |
| **Fully-async** | `fully_async_policy` (experimental) | Single group | Method patching via import hook |

All three modes ship in a single package: `pip install timeslice-verl`.

## Quick Start

### 1. Install the package

```bash
pip install timeslice-verl
```

The package auto-registers with verl via the `verl.plugins` entry point — importing `verl` loads it automatically.

### 2. Choose your mode

**Sync colocated** — both trainer and sampler on the same GPUs, whole-step turns:
```yaml
trainer:
  use_v1: true
  v1:
    trainer_mode: sync_timesliced
```

**Sync disaggregated** — separate trainer and sampler GPU pools:
```yaml
trainer:
  use_v1: true
  v1:
    trainer_mode: separate_async_timesliced
```

**Fully-async** — experimental streaming trainer:
```bash
export TIMESLICE_FULLY_ASYNC=1
```

### 3. Set environment variables

**Colocated (single group):**
```bash
TIMESLICE_JOB_ID=my-job
TIMESLICE_ORCH_ADDR=timeslice-acceleratororchestrator.timeslice-system:50051
TIMESLICE_GROUP=shared-gpu
```

**Disaggregated (dual groups):**
```bash
TIMESLICE_JOB_ID=my-job
TIMESLICE_ORCH_ADDR=timeslice-acceleratororchestrator.timeslice-system:50051
TIMESLICE_TRAINER_GROUP=trainers
TIMESLICE_SAMPLER_GROUP=samplers
```

When env vars are not set, the package runs in **no-op mode** — zero overhead, same image works with and without the platform.

### 4. Run normally

No changes to your training script, model config, or data pipeline. Just `pip install` + config key + env vars.

## How Each Mode Works

### Sync Colocated (`sync_timesliced`)

Subclasses `PPOTrainerSync`. Wraps each training step with a single group lock:

```
__init__      → acquire lock (before model init, weight sync, everything)
on_step_begin → re-acquire lock (blocks while other job runs its step)
on_step_end   → release lock (GPU checkpointed out, other job gets it)
```

The GPU is held for the entire step (generation + training + weight sync) and released between steps. Simple but the entire step is serialized.

### Sync Disaggregated (`separate_async_timesliced`)

Subclasses `PPOTrainerSeparateAsync`. Uses role-based dual locks (trainer-first ordering):

```
on_sample_begin (starved) → release TRAINER, acquire SAMPLER
  ... job's generation runs on sampler pool ...
  ... other job can train on trainer pool ...
on_sample_end             → release SAMPLER, acquire TRAINER
  ... training compute ...
on_step_end               → acquire SAMPLER (dual-lock weight sync), release SAMPLER
```

Enables cross-pipeline concurrency: Job A trains while Job B generates. Conditional yield-on-starvation probes the replay buffer — skips the TRAINER release if samples are already buffered.

### Fully-Async (method patching)

Patches `FullyAsyncTrainer` methods via a `sys.meta_path` import hook (activated by `TIMESLICE_FULLY_ASYNC=1`). Patches:
- `init_workers` — acquire before model load, yield after
- `_get_samples_from_queue` — acquire after batch is ready (the per-step resume point)
- `_fit_update_weights` — dual-lock weight sync span
- `_fit_save_checkpoint` — guard against saving while checkpointed out

Uses `asyncio.to_thread` for blocking gRPC calls (the fully-async trainer runs on an asyncio event loop).

## NCCL Requirements

Always set these environment variables for cuda-checkpoint compatibility:
```bash
NCCL_CUMEM_ENABLE=0
NCCL_NVLS_ENABLE=0
```

For disaggregated mode with NCCL weight sync, set `rebuild_group=true` in the verl config to destroy and recreate NCCL communicators around each sync (required for checkpoint/restore safety).

## Package

The `timeslice-verl` package source is at [`aishukamal/rl-time-slicing/time-slicing-vertical-integrations/verl/`](https://github.com/aishukamal/rl-time-slicing/tree/main/time-slicing-vertical-integrations/verl/).

Structure:
```
timeslice_verl/
  __init__.py          # Auto-registers both v1 trainer modes
  locks.py             # PhaseLocks, RoleLocks, PhaseTransitions
  trainer.py           # sync_timesliced mode
  trainer_disagg.py    # separate_async_timesliced mode
  fully_async.py       # Method-patching wrapper for experimental fully-async
```

## Next Steps

- [Framework Integration Guide](../../framework-integration/) — generic integration patterns and lock protocol
- [Accelerator Orchestrator Guide](../../accelerator-orchestrator/) — platform deployment and DRA setup
