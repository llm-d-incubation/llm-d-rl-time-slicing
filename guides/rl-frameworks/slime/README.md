# Slime Time-Slicing Integration

Share GPU hardware across multiple [Slime](https://github.com/THUDM/slime) RL jobs using the `timeslice-slime` package. Zero Slime source code changes — add one CLI flag and set environment variables.

## How It Works

The `timeslice-slime` package provides a `PhaseCallback` that acquires and releases orchestrator locks at Slime's natural phase boundaries (init, generate, train, weight sync). The platform's snapshot agent handles GPU state checkpoint/restore transparently via cuda-checkpoint.

```bash
# Add to your Slime launch command:
--phase-callback-path timeslice_slime.callback.TimesliceCallback
```

## Sync vs Async RL

| | Sync (`train.py`) | Async (`train_async.py`) |
|---|---|---|
| Generation | Sequential (gen → wait → train) | Pipelined (gen N+1 overlaps train N) |
| Sampler idle time | Yes (idle during training) | No (continuous generation) |
| Trainer idle time | Yes (idle during generation) | Yes (idle waiting for batch) |
| Time-sliced pools | Trainer + Sampler (shared groups) | Trainer only (per-job sampler groups) |

## Quick Start

### Prerequisites

1. Time-slicing platform deployed — [deployment guide](../../../deploy/)
2. GPU nodes labeled and tainted — [orchestrator guide](../../accelerator-orchestrator/)
3. KubeRay operator v1.6.0+ installed

### Install the package

The `timeslice-slime` package is in [`package/`](package/). Install it alongside the Slime fork with PhaseCallback support:

```bash
# Install the forked Slime with PhaseCallback support
pip install "slime @ git+https://github.com/aishukamal/slime.git@feat/phase-callbacks"

# Install the timeslice client
pip install "timeslice @ git+https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git#subdirectory=pkg/client/python"

# Install the timeslice-slime callback package
pip install ./guides/rl-frameworks/slime/package/
```

> **Note:** The Slime fork adds a `--phase-callback-path` CLI argument and driver-level phase emissions. Once this is upstreamed to [THUDM/slime](https://github.com/THUDM/slime), the fork install line above can be replaced with the mainline package.

### Set environment variables

**Sync RL** — both pools shared:
```bash
TIMESLICE_JOB_ID=my-job
TIMESLICE_ORCH_ADDR=timeslice-acceleratororchestrator.timeslice-system.svc.cluster.local:50051
TIMESLICE_TRAINER_GROUP=trainers
TIMESLICE_SAMPLER_GROUP=samplers
```

**Async RL** — trainer shared, per-job samplers:
```bash
TIMESLICE_JOB_ID=my-job
TIMESLICE_ORCH_ADDR=timeslice-acceleratororchestrator.timeslice-system.svc.cluster.local:50051
TIMESLICE_TRAINER_GROUP=trainers
TIMESLICE_SAMPLER_GROUP=samplers-${JOB_NAME}   # unique per job
```

### Run

```bash
# Sync
python3 train.py ... --phase-callback-path timeslice_slime.callback.TimesliceCallback

# Async
python3 train_async.py ... --phase-callback-path timeslice_slime.callback.TimesliceCallback
```

Do **not** use `--offload-train` or `--offload-rollout` — the platform handles GPU state via cuda-checkpoint. Framework-level cooperative offloading conflicts with cuda-checkpoint and causes crashes.

## Runnable Examples

| Mode | Description | Guide |
|------|-------------|-------|
| **Sync disaggregated** | Two sync GRPO jobs sharing trainer + sampler GPU pools | [sync/](sync/) |
| **Async disaggregated** | Two async GRPO jobs with shared trainer pool, dedicated samplers | [async/](async/) |

## Package

The `timeslice-slime` package source is in [`package/`](package/):

```
package/
├── pyproject.toml
├── timeslice_slime/
│   ├── __init__.py      # exports TimesliceCallback
│   ├── callback.py      # PhaseCallback implementation
│   └── locks.py         # RoleLocks with trainer-first ordering
└── tests/
    └── test_callback.py # 12 lock protocol tests (no GPU needed)
```

Run tests: `cd package && python3 -m unittest tests.test_callback -v`

## NCCL Requirements

Set these environment variables for cuda-checkpoint compatibility:
```bash
NCCL_CUMEM_ENABLE=0
NCCL_NVLS_ENABLE=0
```

## For Framework Developers

If you want to understand how this integration was built, or integrate a different framework, see the [Framework Integration Guide](../../framework-integration/).
