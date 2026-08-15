# Sync Time-Sliced Disaggregated GRPO Example

Deploy two independent **sync RL** jobs sharing physical GPU pools using the `llm-d-timeslice-slime` PhaseCallback package, DRA, and the Accelerator Orchestrator.

Both trainer and sampler pools are time-sliced — Job B trains while Job A generates, then they swap.

---

## Quick Start

Assumes the time-slicing platform is deployed ([deployment guide](../../../../deploy/)) and GPU nodes are labeled ([main guide](../../README.md)).

### 1. Apply DRA Resource Claims

```bash
kubectl apply -f guides/rl-frameworks/slime/examples/resource-claims.yaml
```

### 2. Create the ConfigMap

```bash
kubectl create configmap slime-job-script \
  --from-file=run_grpo.sh=guides/rl-frameworks/slime/examples/run_grpo.sh \
  --from-file=setup_node.sh=guides/rl-frameworks/slime/examples/setup_node.sh \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 3. Submit Two Jobs

```bash
# Job 1
sed 's/${JOB_NAME}/slime-sync-1/g; s/${SLIME_MODE}/sync/g; s/${SAMPLER_TS_GROUP}/samplers/g' \
  guides/rl-frameworks/slime/examples/ray-job.yaml.template | kubectl apply -f -

# Job 2 (stagger by ~30s to avoid init contention)
sleep 30
sed 's/${JOB_NAME}/slime-sync-2/g; s/${SLIME_MODE}/sync/g; s/${SAMPLER_TS_GROUP}/samplers/g' \
  guides/rl-frameworks/slime/examples/ray-job.yaml.template | kubectl apply -f -
```

### 4. Monitor

```bash
kubectl get rayjobs
kubectl get pods -o wide -l slime-role
# Check time-slicing events:
kubectl logs <submitter-pod> | grep "\[timeslice\]"
```

### 5. Cleanup

```bash
kubectl delete rayjob slime-sync-1 slime-sync-2
kubectl delete -f guides/rl-frameworks/slime/examples/resource-claims.yaml
kubectl delete configmap slime-job-script
```
