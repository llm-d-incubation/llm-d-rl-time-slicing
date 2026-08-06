# Async Time-Sliced Disaggregated GRPO Example

This directory contains the configuration for deploying two independent **async** RL jobs sharing physical GPU pools using the `timeslice-slime` PhaseCallback integration.

In async mode, generation N+1 runs concurrently with training N (pipelined). The key difference from sync: **sampler GPUs are always busy** — only the trainer pool needs time-slicing. Each job gets its own sampler group to avoid unnecessary contention.

---

## How Async Differs from Sync

| Aspect | Sync | Async |
|--------|------|-------|
| Driver | `train.py` | `train_async.py` |
| Generation | Sequential (gen → wait → train) | Pipelined (gen N+1 overlaps train N) |
| Sampler idle time | Yes (idle during training) | No (continuous generation) |
| Trainer idle time | Yes (idle during generation) | Yes (idle waiting for batch) |
| Sampler groups | Shared (both jobs contend) | Per-job (no contention) |
| Time-sliced pools | Trainer + Sampler | Trainer only |

## Prerequisites

1. Time-slicing platform deployed ([deployment guide](../../../deploy/))
2. KubeRay operator v1.6.0+ installed
3. GPU nodes labeled and tainted:
   ```bash
   kubectl label nodes <trainer-node> timeslice.io/enabled=true group.timeslice.io/trainers=true
   kubectl label nodes <sampler-node> timeslice.io/enabled=true group.timeslice.io/samplers=true
   kubectl taint nodes <trainer-node> <sampler-node> timeslice.io/shared=true:NoSchedule
   ```
4. DRA ResourceClaims applied:
   ```bash
   kubectl apply -f ../sync/resource-claims.yaml
   ```

## Quick Start

### 1. Install the `timeslice-slime` package in your container image

```dockerfile
RUN pip install timeslice-slime
```

Or install the package from `../package/`.

### 2. Update the launch script

Change the entrypoint from `train.py` to `train_async.py` and add the PhaseCallback:

```bash
python3 /root/slime/train_async.py \
    ... \
    --phase-callback-path timeslice_slime.callback.TimesliceCallback
```

Do **not** use `--offload-train` or `--offload-rollout` — the platform handles GPU state via cuda-checkpoint.

### 3. Set environment variables per job

Each job shares the trainer group but gets its own sampler group:

**Job A:**
```yaml
env_vars:
  TIMESLICE_JOB_ID: "async-job-a"
  TIMESLICE_ORCH_ADDR: "timeslice-acceleratororchestrator.timeslice-system.svc.cluster.local:50051"
  TIMESLICE_TRAINER_GROUP: "trainers"
  TIMESLICE_SAMPLER_GROUP: "samplers-a"
```

**Job B:**
```yaml
env_vars:
  TIMESLICE_JOB_ID: "async-job-b"
  TIMESLICE_ORCH_ADDR: "timeslice-acceleratororchestrator.timeslice-system.svc.cluster.local:50051"
  TIMESLICE_TRAINER_GROUP: "trainers"
  TIMESLICE_SAMPLER_GROUP: "samplers-b"
```

### 4. Submit both jobs

```bash
JOB_NAME=async-job-a envsubst < ray-job.yaml.template | kubectl apply -f -
JOB_NAME=async-job-b envsubst < ray-job.yaml.template | kubectl apply -f -
```

## Expected Behavior

- **Trainer GPU**: Alternates between jobs via lock handoff. When Job A finishes a training step and releases the trainer lock, the snapshot agent checkpoints its GPU state and restores Job B.
- **Sampler GPUs**: Run continuously for each job. No contention, no context switching — each job's sampler group has no competing job.
- **Weight sync**: Both trainer and sampler locks are acquired (dual-lock span) for the NCCL broadcast, then the sampler lock is released.

## Monitoring

```bash
# Watch for time-slicing events in the submitter logs
kubectl logs <submitter-pod> | grep "\[timeslice\]"

# Expected output:
# [timeslice] job=async-job-a ACQUIRE role=trainer group=trainers waited=1000ms
# [timeslice] job=async-job-a ACQUIRE role=sampler group=samplers-a waited=1000ms
# ...
# [timeslice] job=async-job-a RELEASE role=trainer group=trainers pending_waiters=1
# [timeslice] job=async-job-b ACQUIRE role=trainer group=trainers waited=Xms context_restored=True
```

The `waited=Xms` on the trainer acquire shows how long the job waited for the other job to finish its training step. `context_restored=True` confirms the snapshot agent restored the GPU state.
