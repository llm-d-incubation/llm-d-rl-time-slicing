# Async Time-Sliced Disaggregated GRPO Example

Deploy two independent **async RL** jobs sharing the trainer GPU pool using the `timeslice-slime` PhaseCallback package.

In async mode, generation N+1 runs concurrently with training N. Sampler GPUs are always busy — only the **trainer pool** is time-sliced. Each job gets its own sampler group to avoid unnecessary contention.

---

## What's Different from Sync

| Aspect | Sync ([sync/](../sync/)) | Async (this) |
|--------|------|-------|
| Entrypoint | `train.py` | `train_async.py` |
| Sampler groups | Shared (`samplers`) | Per-job (`samplers-${JOB_NAME}`) |
| Time-sliced pools | Trainer + Sampler | Trainer only |

---

## Files

* `resource-claims.yaml` — same DRA claims as sync (trainer + sampler pools)
* `ray-job.yaml.template` — uses `train_async.py` entrypoint, per-job `TIMESLICE_SAMPLER_GROUP`
* `run_async_grpo.sh` — launch script with `--phase-callback-path`
* `setup_node.sh` — same postStart hook as sync (installs fork, client, package)

---

## Quick Start

Assumes the time-slicing platform is deployed and GPU nodes are labeled.

### 1. Apply DRA Resource Claims

```bash
kubectl apply -f guides/rl-frameworks/slime/async/resource-claims.yaml
```

### 2. Create the ConfigMap

```bash
kubectl create configmap slime-async-script \
  --from-file=run_async_grpo.sh=guides/rl-frameworks/slime/async/run_async_grpo.sh \
  --from-file=setup_node.sh=guides/rl-frameworks/slime/async/setup_node.sh \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 3. Submit Two Jobs

```bash
JOB_NAME=slime-async-1 envsubst < guides/rl-frameworks/slime/async/ray-job.yaml.template | kubectl apply -f -
JOB_NAME=slime-async-2 envsubst < guides/rl-frameworks/slime/async/ray-job.yaml.template | kubectl apply -f -
```

Each job gets its own sampler group (`samplers-slime-async-1`, `samplers-slime-async-2`) while sharing the `trainers` group.

### 4. Monitor

```bash
kubectl get rayjobs
kubectl logs <submitter-pod> | grep "\[timeslice\]"
```

Expected: trainer lock handoffs between jobs (`ACQUIRE role=trainer waited=Xms`). Sampler acquires return immediately (~1s, no contention).

### 5. Cleanup

```bash
kubectl delete rayjob slime-async-1 slime-async-2
kubectl delete -f guides/rl-frameworks/slime/async/resource-claims.yaml
kubectl delete configmap slime-async-script
```
