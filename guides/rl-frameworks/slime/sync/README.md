# Sync Time-Sliced Disaggregated GRPO Example

Deploy two independent **sync RL** jobs sharing physical GPU pools using the `timeslice-slime` PhaseCallback package, DRA, and the Accelerator Orchestrator.

Both trainer and sampler pools are time-sliced — Job B trains while Job A generates, then they swap.

---

## Files

* `resource-claims.yaml` — shared DRA `ResourceClaim` manifests for trainer and sampler pools
* `ray-job.yaml.template` — parameterized RayJob template with DRA claims, time-slicing labels, and postStart hooks
* `run_disaggregated_grpo.sh` — launch script with `--phase-callback-path timeslice_slime.callback.TimesliceCallback`
* `setup_node.sh` — postStart hook that installs the Slime fork, timeslice client, and timeslice-slime package at runtime

---

## Quick Start

Assumes the time-slicing platform is deployed ([deployment guide](../../../../deploy/)) and GPU nodes are labeled ([orchestrator guide](../../../accelerator-orchestrator/)).

### 1. Apply DRA Resource Claims

```bash
kubectl apply -f guides/rl-frameworks/slime/sync/resource-claims.yaml
```

### 2. Create the ConfigMap

```bash
kubectl create configmap slime-job-script \
  --from-file=run_disaggregated_grpo.sh=guides/rl-frameworks/slime/sync/run_disaggregated_grpo.sh \
  --from-file=setup_node.sh=guides/rl-frameworks/slime/sync/setup_node.sh \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 3. Submit Two Jobs

```bash
JOB_NAME=slime-sync-1 envsubst < guides/rl-frameworks/slime/sync/ray-job.yaml.template | kubectl apply -f -
JOB_NAME=slime-sync-2 envsubst < guides/rl-frameworks/slime/sync/ray-job.yaml.template | kubectl apply -f -
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
kubectl delete -f guides/rl-frameworks/slime/sync/resource-claims.yaml
kubectl delete configmap slime-job-script
```
