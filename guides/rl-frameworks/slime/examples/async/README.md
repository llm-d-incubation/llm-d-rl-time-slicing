# Async Time-Sliced Disaggregated GRPO Example

Deploy two independent **async RL** jobs sharing the trainer GPU pool using the `llm-d-timeslice-slime` PhaseCallback package.

In async mode, generation N+1 runs concurrently with training N. Sampler GPUs are always busy — only the **trainer pool** is time-sliced. Each job gets its own sampler group to avoid unnecessary contention.

---

## What's Different from Sync

| Aspect | Sync ([sync/](../sync/)) | Async (this) |
|--------|------|-------|
| `SLIME_MODE` | `sync` | `async` |
| Entrypoint (auto-selected) | `train.py` | `train_async.py` |
| Sampler groups (auto-derived) | Shared (`samplers`) | Per-job (`samplers-${JOB_NAME}`) |
| Time-sliced pools | Trainer + Sampler | Trainer only |

All files (launch script, RayJob template, setup script, resource claims) are shared with the sync example — only `SLIME_MODE` differs.

**Note:** For async mode, the sampler node needs per-job group labels so the orchestrator can discover each job's sampler group:
```bash
kubectl label nodes <sampler-node> group.timeslice.io/samplers-slime-async-1=true group.timeslice.io/samplers-slime-async-2=true
```

---

## Quick Start

Assumes the time-slicing platform is deployed ([deployment guide](../../../../deploy/)) and GPU nodes are labeled ([main guide](../../README.md)).

### 1. Apply DRA Resource Claims

```bash
kubectl apply -f guides/rl-frameworks/slime/examples/resource-claims.yaml
```

### 2. Label Sampler Node for Per-Job Groups

```bash
kubectl label nodes <sampler-node> \
  group.timeslice.io/samplers-slime-async-1=true \
  group.timeslice.io/samplers-slime-async-2=true
```

### 3. Create the ConfigMap

```bash
kubectl create configmap slime-job-script \
  --from-file=run_grpo.sh=guides/rl-frameworks/slime/examples/run_grpo.sh \
  --from-file=setup_node.sh=guides/rl-frameworks/slime/examples/setup_node.sh \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 4. Submit Two Jobs

```bash
# Job 1
sed 's/${JOB_NAME}/slime-async-1/g; s/${SLIME_MODE}/async/g; s/${SAMPLER_TS_GROUP}/samplers-slime-async-1/g' \
  guides/rl-frameworks/slime/examples/ray-job.yaml.template | kubectl apply -f -

# Job 2 (stagger by ~30s to avoid init contention)
sleep 30
sed 's/${JOB_NAME}/slime-async-2/g; s/${SLIME_MODE}/async/g; s/${SAMPLER_TS_GROUP}/samplers-slime-async-2/g' \
  guides/rl-frameworks/slime/examples/ray-job.yaml.template | kubectl apply -f -
```

Each job gets its own sampler group (`samplers-slime-async-1`, `samplers-slime-async-2`) while sharing the `trainers` group.

### 5. Monitor

```bash
kubectl get rayjobs
kubectl logs <submitter-pod> | grep "\[timeslice\]"
```

Expected: trainer lock handoffs between jobs (`ACQUIRE role=trainer waited=Xms`). Sampler acquires return immediately (~1s, no contention).

### 6. Cleanup

```bash
kubectl delete rayjob slime-async-1 slime-async-2
kubectl delete -f guides/rl-frameworks/slime/examples/resource-claims.yaml
kubectl delete configmap slime-job-script
# Remove per-job sampler labels
kubectl label nodes <sampler-node> \
  group.timeslice.io/samplers-slime-async-1- \
  group.timeslice.io/samplers-slime-async-2-
```
