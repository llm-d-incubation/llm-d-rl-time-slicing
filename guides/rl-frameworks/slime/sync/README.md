# Multi-Cluster Time-Sliced Disaggregated GRPO Example

This directory contains the files required to deploy two independent sync RL jobs, each provisioned in its **own distinct Ray Cluster**, concurrently sharing physical GPU pools using Kubernetes **Dynamic Resource Allocation (DRA) Overloading** and the **Accelerator Orchestrator** time-slicing lock queues. This serves as a simple, quick example of the behavioral mechanisms of time slicing. 

Because physical GPU limits (`nvidia.com/gpu`) are replaced with DRA `ResourceClaim`s, Kubernetes allows worker pods from both Job 1 and Job 2 to co-locate and schedule onto the exact same physical GPU nodes simultaneously. At runtime, the workloads coordinate via the Accelerator Orchestrator to atomically snapshot and swap accelerator memory, eliminating idle waiting and maximizing GPU duty cycles.

---

## 1. Directory Files

*   `resource-claims.yaml`: Global shared DRA `ResourceClaim` manifests (`shared-trainers-gpu-claim` and `shared-samplers-gpu-claim`) referencing `gpu.nvidia.com`.
*   `ray-job.yaml.template`: Parameterized GKE `RayJob` resource template configured with DRA claim bindings, time-slicing pod labels (`timeslice.io/job-id`, `timeslice.io/group`), group node selectors (`group.timeslice.io/trainers` and `group.timeslice.io/samplers`), and container `lifecycle.postStart` hooks executing `setup_node.sh`.
*   `run_disaggregated_grpo.sh`: Launch script running disaggregated GRPO integrated with the time-slicing client library.
*   `setup_node.sh`: Node initialization script mounted via ConfigMap and executed by container `postStart` hooks. Handles checking out the custom `timeslice` branch, installing SDK clients, downloading HuggingFace models/datasets into `/tmp`, and pre-compiling FlashInfer JIT kernels to prevent startup memory-saver deadlocks.

---

## 2. Execution Guide

Assumes you have already deployed the time-slicing Helm chart on your cluster.

### Step 2.1: Apply Global DRA Resource Claims
Deploy the shared DRA resource claims that all jobs will reference across the trainer and sampler groups:

```bash
kubectl apply -f guides/rl-frameworks/slime/sync/resource-claims.yaml
```

### Step 2.2: Register the Launch Script ConfigMap
Package the local launch script and node setup script into a Kubernetes ConfigMap to mount into the Ray pods:

```bash
kubectl create configmap slime-job-script \
  --from-file=run_disaggregated_grpo.sh=guides/rl-frameworks/slime/sync/run_disaggregated_grpo.sh \
  --from-file=setup_node.sh=guides/rl-frameworks/slime/sync/setup_node.sh \
  --dry-run=client -o yaml | kubectl apply -f -
```

### Step 2.3: Submit the RayJobs Concurrently
Using the parameterized template, submit both jobs concurrently using `envsubst` to set distinct job names:

```bash
# Submit Job 1
JOB_NAME=slime-job1 envsubst < guides/rl-frameworks/slime/sync/ray-job.yaml.template | kubectl apply -f -

# Submit Job 2
JOB_NAME=slime-job2 envsubst < guides/rl-frameworks/slime/sync/ray-job.yaml.template | kubectl apply -f -
```

---

## 3. Monitoring Execution & Lock Contention

Monitor the RayJob resources and observe pod co-location across nodes:

```bash
# Check RayJob statuses
kubectl get rayjobs

# Verify that worker pods for both jobs are Running on the same physical nodes
kubectl get pods -o wide -l slime-role
```

---

## 4. Cleanup

The clusters tear down automatically 60 seconds after completion. To manually clean up all resources:

```bash
kubectl delete rayjob slime-job1 slime-job2
kubectl delete -f guides/rl-frameworks/slime/sync/resource-claims.yaml
kubectl delete configmap slime-job-script
```
