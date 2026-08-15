# Time-Slicing Integration Guide for Slime Workloads

This guide provides step-by-step instructions on how to integrate and deploy **Slime** (high-performance RL framework for LLMs) with the **llm-d-rl-time-slicing** platform using the pre-packaged `llm-d-timeslice-slime` integration.

### Motivation: Maximizing GPU Utilization
In disaggregated RL setups, GPUs sit idle whenever worker groups wait for another phase to complete. Cooperative time-slicing enables multiple independent Slime jobs to share physical GPU pools — when one job finishes a phase, its GPU context is checkpointed and evicted, allowing another job to immediately utilize the hardware.

**Sync RL** (`train.py`): Training and generation alternate strictly. Both trainer and sampler GPUs have idle gaps. Time-slicing shares **both pools** — Job B trains while Job A generates, then they swap.

**Async RL** (`train_async.py`): Generation N+1 runs concurrently with training N (pipelined). Sampler GPUs are busy non-stop. Only the **trainer GPU** has idle gaps waiting for the next batch. Time-slicing shares only the trainer pool; each job gets dedicated sampler GPUs (separate group per job, no contention).

For runnable examples, see:
* **[Sync GRPO Example](examples/sync/README.md)** — two sync RL jobs sharing trainer + sampler pools
* **[Async GRPO Example](examples/async/README.md)** — two async RL jobs with shared trainer pool, dedicated samplers

---

## Table of Contents
1. [Cluster Prerequisites](#1-cluster-prerequisites)
2. [Deploying the Time-Slicing Platform](#2-deploying-the-time-slicing-platform)
3. [Integrating with Slime](#3-integrating-with-slime)
4. [Deploying Time-Sliced Slime Jobs](#4-deploying-time-sliced-slime-jobs)
5. [Submitting and Observing Time-Sliced Jobs](#5-submitting-and-observing-time-sliced-jobs)
6. [Observing Convergence and Job Completion](#6-observing-convergence-and-job-completion)

---

## 1. Cluster Prerequisites

Before deploying cooperative time-slicing for Slime, ensure your Kubernetes cluster meets the following requirements:

### Kubernetes Version
* Kubernetes **v1.34** or later.

### KubeRay Version
* **KubeRay Operator v1.6.0 or later** (required for modern Dynamic Resource Allocation pod syntax).

### GPU Node Configuration
* GPU nodes must run **NVIDIA GPU Driver 565 or later**. This is a strict requirement to support **NVIDIA Dynamic Resource Allocation (DRA)**. On GKE, you can verify the driver version by checking the NVIDIA DRA driver pod logs or running `kubectl exec` into a GPU pod and running `nvidia-smi`.
* GPU memory capacity must be sufficient to hold the active working set of a single Slime job's trainer or sampler at any one time (since inactive jobs will have their GPU memory checkpointed and evicted).
* Sampler/trainer node host memory capacity must be sufficient to hold the GPU memory footprint of the trainers/samplers needed for the number of parallel Slime jobs. Worker pod memory limits must account for cuda-checkpoint saving GPU state to host memory — set the pod memory limit to at least 2× the GPU memory footprint of the workload (e.g., 128Gi for a sampler using ~33GB GPU memory with two time-sliced jobs).

### Node Labeling and Tainting for Time-Slice Pools
The `timeslice` platform relies on node labels and taints to identify resource pools (groups) and isolate time-sliced workloads. For disaggregated Slime executions, label and taint your GPU nodes accordingly:

* **Enable Time-Slicing & Taint Nodes**:
  ```bash
  kubectl label nodes <node-name> timeslice.io/enabled=true
  kubectl taint nodes <node-name> timeslice.io/shared=true:NoSchedule
  ```
* **Trainer Nodes**:
  ```bash
  kubectl label nodes <trainer-node> group.timeslice.io/trainers=true
  ```
* **Sampler Nodes**:
  ```bash
  kubectl label nodes <sampler-node> group.timeslice.io/samplers=true
  ```

### Shared DRA Resource Claims
Cooperative time-slicing leverages Kubernetes **Dynamic Resource Allocation (DRA)** so multiple jobs' worker pods can share physical GPU hardware without scheduler blocking. Before submitting jobs, create shared `ResourceClaim` manifests for both the trainer and sampler pools in your target namespace:

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: shared-trainers-gpu-claim
  namespace: default
spec:
  devices:
    requests:
    - name: gpu
      exactly:
        deviceClassName: gpu.nvidia.com
        allocationMode: ExactCount
        count: 1
---
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: shared-samplers-gpu-claim
  namespace: default
spec:
  devices:
    requests:
    - name: gpu
      exactly:
        deviceClassName: gpu.nvidia.com
        allocationMode: ExactCount
        count: 1
```

Apply these resource claims to your cluster:
```bash
kubectl apply -f guides/rl-frameworks/slime/examples/resource-claims.yaml
```

---

## 2. Deploying the Time-Slicing Platform

We deploy the core platform components—**TimeSlice Orchestrator**, **Snapshot Agent** (DaemonSet), and the **NVIDIA DRA Driver**—using the parent Helm chart.

### Step 1: Update Helm Chart Dependencies
From the root of your `llm-d-rl-time-slicing` workspace, navigate to the `deploy` directory and fetch the required subcharts:
```bash
cd deploy/
helm dependency update .
```

### Step 2: Configure `values.yaml`
Review or modify the parent `values.yaml` file to match your cluster environment:
```yaml
timesliceorchestrator:
  replicaCount: 1
  image:
    tag: latest

snapshot-agent:
  image:
    tag: latest
  nodeSelector:
    timeslice.io/enabled: "true"

nvidia-dra-driver-gpu:
  enabled: true
  # Use "/home/kubernetes/bin/nvidia/" for GKE Container-Optimized OS (COS) nodes.
  # Use "/opt/nvidia" for standard Ubuntu/Debian nodes.
  nvidiaDriverRoot: "/home/kubernetes/bin/nvidia/"
```

### Step 3: Install the Helm Chart
Install the chart into a dedicated namespace (`timeslice-system`). This ensures all service accounts, RBAC policies, and daemons are securely isolated:
```bash
helm install timeslice . -n timeslice-system --create-namespace
```

### Step 4: Verify Platform Health
Verify that the orchestrator and agents are running and healthy:
1. **Using kubectl**:
   Ensure all pods in the `timeslice-system` namespace are `Running`:
   ```bash
   kubectl get pods -n timeslice-system
   ```
2. **Using the `rlts` CLI** (optional, requires Go toolchain):
   ```bash
   go build -o bin/rlts ./cmd/rlts
   ./bin/rlts orchestrator verify
   ```

---

## 3. Integrating with Slime

The `llm-d-timeslice-slime` package provides a pre-built `PhaseCallback` that acquires and releases orchestrator locks at Slime's natural phase boundaries (init, generate, train, weight sync). No manual code changes to Slime are required.

### How It Works

1. A Slime fork adds a `--phase-callback-path` CLI argument and `on_phase_begin`/`on_phase_end` calls around each GPU phase in the training drivers (`train.py` and `train_async.py`).
2. The `llm-d-timeslice-slime` package provides `TimesliceCallback` — a `PhaseCallback` implementation that maps phase events to orchestrator lock acquire/release.
3. Configuration (job ID, orchestrator address, group IDs) comes from environment variables — Slime doesn't need to know about any of them.

> **Note:** The Slime fork is a temporary measure. We have already requested upstream Slime to add phase callback hooks. Once upstreamed, the only change needed is one pip install line in `setup_node.sh` (fork → mainline Slime).

### Installation

The `setup_node.sh` script (mounted via ConfigMap and executed by container `postStart` hooks) installs the required packages at runtime:

```bash
# Replace the base image's Slime with the fork (PhaseCallback support)
rm -rf /root/slime
git clone --depth 1 -b feat/phase-callbacks https://github.com/aishukamal/slime.git /root/slime
cd /root/slime && pip install --no-cache-dir --no-deps -e .

# Install the timeslice client library (--no-deps avoids protobuf version conflicts)
pip install --no-cache-dir --no-deps "timeslice @ git+https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git#subdirectory=pkg/client/python"

# Install the llm-d-timeslice-slime callback package
pip install --no-cache-dir --no-deps "llm-d-timeslice-slime @ git+https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git#subdirectory=pkg/integrations/slime"
```

> **Why `--no-deps`?** The Slime base image ships protobuf 6.x which Ray depends on. Without `--no-deps`, pip upgrades protobuf to 7.x, which corrupts the running Ray process.

### Usage

Add one flag to your Slime launch command:
```bash
# Sync
python3 train.py ... --phase-callback-path timeslice_slime.callback.TimesliceCallback

# Async
python3 train_async.py ... --phase-callback-path timeslice_slime.callback.TimesliceCallback
```

Set environment variables — these are set in `run_grpo.sh` (derived from `JOB_NAME` and `SLIME_MODE`) and `runtimeEnvYAML` in the RayJob manifest:
- `TIMESLICE_JOB_ID` — unique job identifier, derived from `JOB_NAME` (set in `run_grpo.sh`)
- `TIMESLICE_ORCH_ADDR` — orchestrator gRPC address (set in `runtimeEnvYAML`)
- `TIMESLICE_TRAINER_GROUP` — trainer pool group, shared across jobs (set in `run_grpo.sh`)
- `TIMESLICE_SAMPLER_GROUP` — sampler pool group: shared for sync, per-job for async (set in `run_grpo.sh` based on `SLIME_MODE`)
- `NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=0` — required for cuda-checkpoint compatibility. Must be set as **container-level env vars** in the RayJob worker pod spec (not just `runtimeEnvYAML`), so they propagate to subprocesses like the SGLang inference server.

> [!TIP]
> For a detailed walkthrough of how to manually integrate a framework (without the pre-packaged solution), see **[Manual Integration Example](../manual-integration-example.md)**.

---

## 4. Deploying Time-Sliced Slime Jobs

### Step 1: Package and Containerize
The examples use the standard `slimerl/slime` container image. The `setup_node.sh` script installs the Slime fork and timeslice packages at runtime via `postStart` hooks — no custom image build required.

### Step 2: Configure KubeRay `RayJob` with DRA Resource Claims
When deploying Slime across independent Ray clusters, use KubeRay `RayJob` manifests configured with **Kubernetes Dynamic Resource Allocation (DRA)** (`resourceClaims`). Binding containers to shared DRA claims (`shared-trainers-gpu-claim` and `shared-samplers-gpu-claim`) instead of static `nvidia.com/gpu` limits allows multiple jobs' worker pods to co-locate on the same physical GPU nodes without scheduler blocking.

A complete disaggregated Slime workload requires defining **two separate worker groups** under `workerGroupSpecs`: one for trainers and one for rollouts (samplers). For each group:
* **Custom Ray Resources**: Include custom resource counts in `rayStartParams` (`"{\"trainers\": 1}"` for trainers and `"{\"samplers\": 1}"` for rollouts) so that Ray placement groups can bind tasks to the appropriate worker pool.
* **Pool-Specific Identifiers**: Ensure each worker group is configured with its corresponding node selector (`group.timeslice.io/trainers: "true"` vs. `samplers: "true"`), pod labels (`timeslice.io/group: trainers` vs. `samplers`), and shared DRA claim (`shared-trainers-gpu-claim` vs. `shared-samplers-gpu-claim`).
* **Time-Slicing Environment Variables**: Set `TIMESLICE_JOB_ID`, `TIMESLICE_ORCH_ADDR`, `TIMESLICE_TRAINER_GROUP`, `TIMESLICE_SAMPLER_GROUP`, and NCCL env vars in `runtimeEnvYAML`.

For **async RL**, set `TIMESLICE_SAMPLER_GROUP` to a per-job value (e.g., `samplers-${JOB_NAME}`) so each job gets its own sampler group with no contention.

> [!TIP]
> For example KubeRay templates and initialization scripts, see:
> * **Sync:** [`examples/sync/ray-job.yaml.template`](examples/sync/ray-job.yaml.template) and [`examples/sync/setup_node.sh`](examples/sync/setup_node.sh)
> * **Async:** [`examples/async/ray-job.yaml.template`](examples/async/ray-job.yaml.template) and [`examples/async/setup_node.sh`](examples/async/setup_node.sh)

---

## 5. Submitting and Observing Time-Sliced Jobs

Once the platform is deployed and the Slime code is integrated, you can submit multiple jobs and observe them sharing the GPUs.

### Step 1: Submit Multiple Jobs
Deploy two independent Slime jobs to the cluster (e.g., `slime-job-a` and `slime-job-b`).
Ensure they have unique `TIMESLICE_JOB_ID` environment variables.

### Step 2: Port-Forward the Orchestrator
To monitor the orchestrator state from your local machine, port-forward the gRPC service:
```bash
kubectl port-forward svc/timeslice-timesliceorchestrator 50051:50051 -n timeslice-system
```

### Step 3: Observe Time-Slicing via the CLI
Use the `rlts` CLI tool to watch the active resource allocations in real-time.

1. **Watch the Samplers Pool**:
   ```bash
   watch -n 0.5 ./bin/rlts orchestrator status samplers
   ```
   **Expected Output:**
   You should see the `Active Job` and `Locking Job` alternate between `slime-job-a` and `slime-job-b`. When one job is sampling, the other job's status will show in the `Waiter Queue Depth` (depth = 1).

2. **Watch the Trainers Pool**:
   ```bash
   watch -n 0.5 ./bin/rlts orchestrator status trainers
   ```
   In a pipelined setup, you will observe the jobs interleaving: while `slime-job-a` is using the `trainers` pool, `slime-job-b` is using the `samplers` pool, and vice-versa.

### Step 4: Observe Context Switches in the Logs
You can inspect the platform logs to verify that the Snapshot Agent is actively saving and restoring GPU states during swaps.

1. **Orchestrator Logs (Scheduling Decisions)**:
   ```bash
   kubectl logs -n timeslice-system -l app.kubernetes.io/name=timesliceorchestrator --tail=100 -f
   ```
   Look for lines indicating lock transfers:
   ```text
   [INFO] Acquire request from job "slime-job-b" for group "samplers" - Queued (Lock held by "slime-job-a")
   [INFO] Yield received from job "slime-job-a" for group "samplers"
   [INFO] Granting lock to next waiter "slime-job-b" for group "samplers"
   ```

2. **Snapshot Agent Logs (State Checkpoint & Restore)**:
   ```bash
   kubectl logs -n timeslice-system -l app.kubernetes.io/name=snapshot-agent --tail=100 -f
   ```
   Look for lines showing the actual GPU context switching:
   ```text
   [INFO] Evicting/Snapshotting GPU state for job "slime-job-a" on node "gpu-node-1"
   [INFO] Snapshot completed in 142ms.
   [INFO] Restoring GPU state for job "slime-job-b" on node "gpu-node-1"
   [INFO] Restore completed in 158ms.
   ```

3. **TimesliceCallback Logs (Lock Acquire/Release)**:
   ```bash
   kubectl logs <submitter-pod> | grep "\[timeslice\]"
   ```
   Look for lock events:
   ```text
   [timeslice] job=slime-job-a ACQUIRE role=trainer group=trainers waited=1000ms context_restored=True
   [timeslice] job=slime-job-a ACQUIRE role=sampler group=samplers waited=1000ms context_restored=True
   [timeslice] job=slime-job-a RELEASE role=sampler group=samplers pending_waiters=1
   ```

---

## 6. Observing Convergence and Job Completion

Cooperative time-slicing shares the accelerator hardware transparently at the system level. While the wall-clock time per iteration will reflect the shared resource environment, the **algorithmic convergence** (how the model learns over training steps) remains completely unaffected.

### A. Monitoring Training Metrics & Convergence
Slime workloads typically log training metrics to **TensorBoard**, **Weights & Biases (W&B)**, or local stdout logs. You can observe convergence by monitoring standard RL metrics:
1. **Reward/Score Curves**: The mean reward should steadily increase over iterations, indicating the policy is successfully learning.
2. **Policy & Value Loss**: Megatron-LM's training loss curves (actor loss, critic/value loss) should stabilize or decrease as training progresses.
3. **KL Divergence**: Monitor the KL divergence between the active policy and the reference model to ensure it stays within target bounds (e.g., to prevent policy collapse).
4. **Step vs. Wall-Clock Time**:
   * **Step-wise Convergence**: The step-wise convergence graph (e.g., Reward vs. Training Steps) will align perfectly with a standalone (non-timesliced) run. The time-slicing process does not alter the mathematical state transitions.
   * **Wall-Clock Progress**: Because the trainers and samplers are disaggregated & RL jobs interleaved, wall-clock time saved to run N jobs will depend on how much time the jobs can run different phases in parallel minus accelerator context swap time.

### B. Observing Job Completion
When a Slime job completes its designated number of iterations:
1. **Graceful Exit**: The `TimesliceCallback.close()` method will release any remaining locks and clean up gRPC channels. The atexit safety net ensures locks are always released, even on crashes.
2. **Kubernetes Job Status**:
   If deployed as a Kubernetes `Job` or `PyTorchJob` (via the Kubeflow Training Operator), you can observe the status transition to `Completed` (or `Succeeded`):
   ```bash
   kubectl get jobs -w
   # or for Kubeflow Training Operator:
   kubectl get pytorchjobs -w
   ```
   **Expected Output:**
   ```text
   NAME             COMPLETION   STATUS      AGE
   slime-job-a      1/1          Succeeded   45m
   slime-job-b      0/1          Running     46m
   ```
3. **Release of Lock Pools**:
   Once `slime-job-a` completes and terminates, the orchestrator will notice the channel closure, and `slime-job-b` will get **exclusive, continuous access** to the GPU pools without any further time-slicing delays. You can verify this via:
   ```bash
   ./bin/rlts orchestrator status samplers
   ```
   The `Waiter Queue Depth` will drop to `0` and stay there, and the `Active Job` will remain permanently assigned to `slime-job-b` until it also completes.

