# Time-Slicing Integration Guide for Slime Workloads

This guide provides step-by-step instructions on how to integrate and deploy **Slime** (high-performance RL framework for LLMs) with the **llm-d-rl-time-slicing** platform.

### Motivation: Maximizing GPU Utilization
In traditional disaggregated RL setups, GPUs sit idle whenever worker groups wait for another phase to complete (e.g., trainer GPUs idling during rollout generation, or rollout GPUs idling during policy updates). Cooperative time-slicing enables multiple independent Slime jobs to multiplex physical GPU resource pools concurrently. When one job finishes a phase, its GPU context is checkpointed and evicted, allowing another job to immediately utilize the hardware—significantly driving up GPU duty cycle and overall cluster throughput.

For a step-by-step reproduction guide using RayCluster Kubernetes manifests and disaggregated GRPO launch scripts, see:
* **[Multi-Cluster Time-Sliced Disaggregated GRPO Guide](sync/README.md)**

---

## Table of Contents
1. [Cluster Prerequisites](#1-cluster-prerequisites)
2. [Deploying the Time-Slicing Platform](#2-deploying-the-time-slicing-platform)
3. [Code Integration with Slime](#3-code-integration-with-slime)
4. [Deploying the Modified Slime Variant](#4-deploying-the-modified-slime-variant)
5. [Submitting and Observing Time-Sliced Jobs](#5-submitting-and-observing-time-sliced-jobs)
6. [Observing Convergence and Job Completion](#6-observing-convergence-and-job-completion)

---

## 1. Cluster Prerequisites

Before deploying cooperative time-slicing for Slime, ensure your Kubernetes cluster meets the following requirements:

### Kubernetes Version
* Kubernetes **v1.26** or later.

### GPU Node Configuration
* GPU nodes must run **NVIDIA GPU Driver 565 or later**. This is a strict requirement to support **NVIDIA Dynamic Resource Allocation (DRA)**, which enables transparent context switching and snapshot/restore of GPU state.
* GPU memory capacity must be sufficient to hold the active working set of a single Slime job's trainer or sampler at any one time (since inactive jobs will have their GPU states checkpointed and evicted).

### Node Labeling for Time-Slice Pools
The `timeslice` platform relies on node labels to identify resource pools (groups). For disaggregated Slime executions, label your GPU nodes accordingly:
* **Sampler Nodes**:
  ```bash
  kubectl label nodes <node-name> group.timeslice.io/samplers=true
  ```
* **Trainer Nodes**:
  ```bash
  kubectl label nodes <node-name> group.timeslice.io/trainers=true
  ```

---

## 2. Deploying the Time-Slicing Platform

We deploy the core platform components—**Accelerator Orchestrator**, **Snapshot Agent** (DaemonSet), and the **NVIDIA DRA Driver**—using the parent Helm chart.

### Step 1: Update Helm Chart Dependencies
From the root of your `llm-d-rl-time-slicing` workspace, navigate to the `deploy` directory and fetch the required subcharts:
```bash
cd deploy/
helm dependency update .
```

### Step 2: Configure `values.yaml`
Review or modify the parent `values.yaml` file to match your cluster environment:
```yaml
acceleratororchestrator:
  replicaCount: 2
  image:
    tag: latest

snapshot-agent:
  image:
    tag: latest

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
1. **Using the `rlts` CLI**:
   Build the CLI tool and run the verify command:
   ```bash
   go build -o bin/rlts ./cmd/rlts
   ./bin/rlts orchestrator verify
   ```
2. **Using kubectl**:
   Ensure all pods in the `timeslice-system` namespace are `Running`:
   ```bash
   kubectl get pods -n timeslice-system
   ```

---

## 3. Code Integration with Slime

To participate in cooperative time-slicing, the Slime training loop driver requests and yields access to the GPU resource pools at its natural phase boundaries.

Because worker processes (SGLang engines and Megatron-LM trainer actors) run as background servers, **only the main RL loop driver script (`train.py`)** needs to communicate with the Accelerator Orchestrator via `OrchestratorClient`.

### Step 1: Add Time-Slicing Command-Line Arguments
Add time-slicing configuration options to `slime/utils/arguments.py`:

```python
parser.add_argument(
    "--enable-timeslice",
    action="store_true",
    default=False,
    help="Enable llm-d-rl-time-slicing cooperative GPU grant acquisition.",
)
parser.add_argument(
    "--timeslice-orchestrator-addr",
    type=str,
    default="timeslice-acceleratororchestrator.timeslice-system.svc.cluster.local:50051",
    help="Address of the Accelerator Orchestrator gRPC service.",
)
parser.add_argument(
    "--timeslice-job-id",
    type=str,
    default=None,
    help="Unique job identifier for the Accelerator Orchestrator.",
)
parser.add_argument(
    "--timeslice-sampler-group",
    type=str,
    default="group-slime-sampler",
    help="Accelerator Orchestrator time-slice group for rollout samplers.",
)
parser.add_argument(
    "--timeslice-trainer-group",
    type=str,
    default="group-slime-trainer",
    help="Accelerator Orchestrator time-slice group for trainer actors.",
)
```

### Step 2: Initialize Client & Allocate Placement Groups (`train.py`)
In `train.py`, instantiate clients for both the sampler and trainer GPU groups. To prevent cross-cluster circular wait deadlocks, enforce a **Trainer-First lock hierarchy**: acquire the Trainer lock before requesting placement groups from Ray, and create actor and rollout groups concurrently using a `ThreadPoolExecutor`:

```python
import os, concurrent.futures
from timeslice import OrchestratorClient

def train(args):
    sampler_client = None
    trainer_client = None
    job_id = getattr(args, "timeslice_job_id", None) or os.getenv("TIMESLICE_JOB_ID", "slime-job-default")

    if getattr(args, "enable_timeslice", False):
        addr = getattr(args, "timeslice_orchestrator_addr", "timeslice-acceleratororchestrator.timeslice-system.svc.cluster.local:50051")
        sampler_client = OrchestratorClient(target=addr, job_id=job_id, group_id=getattr(args, "timeslice_sampler_group", "group-slime-sampler"))
        trainer_client = OrchestratorClient(target=addr, job_id=job_id, group_id=getattr(args, "timeslice_trainer_group", "group-slime-trainer"))

    # DEADLOCK PREVENTION: Acquire Trainer grant first, then Sampler.
    if trainer_client:
        trainer_client.acquire()

    def _create_rollout_group():
        if sampler_client:
            sampler_client.acquire()
        return create_placement_groups(args, role="rollout")

    # Parallel deployment of the actor (ie. trainer) & sampler groups    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_actor = executor.submit(create_placement_groups, args, role="actor") 
        f_rollout = executor.submit(_create_rollout_group)
        pgs = f_actor.result()
        pgs.update(f_rollout.result())
```

### Step 3: Wrap Rollout and Training Phases
Acquire and release GPU grants around the rollout collection and policy training loops in `train.py`. Note that because weight synchronization (`update_weights`) executes via direct GPU-to-GPU NCCL broadcast, **both Trainer and Sampler locks must be held concurrently** during the transfer:

```python
    # Yield Trainer lock after initial weight sync, retaining Sampler lock for rollout
    if trainer_client:
        trainer_client.release()

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # ---------------------------------------------------------
        # Phase 1: Rollout Generation (Rollout GPU Group)
        # ---------------------------------------------------------
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        if sampler_client:
            sampler_client.release()

        # ---------------------------------------------------------
        # Phase 2: Megatron-LM Policy Training (Trainer GPU Group)
        # ---------------------------------------------------------
        if trainer_client:
            trainer_client.acquire()

        actor_model.async_train(rollout_id, rollout_data_ref)
        
        # Re-acquire Sampler lock (following Trainer-first order) for GPU-to-GPU NCCL broadcast
        if sampler_client:
            sampler_client.acquire()

        actor_model.update_weights()

        # Yield Trainer lock, retaining Sampler lock for next rollout epoch
        if trainer_client:
            trainer_client.release()

    if sampler_client:
        sampler_client.release()
        sampler_client.close()
    if trainer_client:
        trainer_client.close()
```

> [!NOTE]
> **Cooperative Memory Offloading is Optional:** Application-level offloading (`--offload-train` and `--offload-rollout`) is optional. The node-local Snapshot Agent can transparently context-switch the entire GPU VRAM state without code changes. Cooperative offloading simply optimizes PCIe swap latency by discarding transient SGLang KV caches and copying only static model weights (~1GB) to CPU memory before yielding locks.

> [!TIP]
> For a tested codebase reference branch containing these exact changes, see [jessicaochen/slime (timeslice branch)](https://github.com/jessicaochen/slime/tree/timeslice).
> For a detailed walkthrough of all codebase changes made in this fork (categorized by scheduling, device leak fixes, and memory offloading), see **[Detailed Fork Changes](sync/SLIME_CHANGES.md)**.

---

## Concrete Deployment & Benchmark Reference

For a step-by-step reproduction guide using RayCluster Kubernetes manifests, disaggregated GRPO launch scripts, and verified performance benchmark numbers, see:

* **[Disaggregated Time-Sliced Slime Deployment Guide](sync/README.md)**

---

## 4. Deploying the Modified Slime Variant

To run your modified Slime workload on the cluster, you must package the `timeslice` client library and configure the Kubernetes deployments.

### Step 1: Package and Containerize
Ensure the `timeslice` Python client is installed in your Slime container image. Add the following to your Slime `Dockerfile`:

<!-- TDB: Less than 98% confident in the exact base image or Dockerfile structure of the Slime workload. Customize this step to fit your existing Docker build process. -->
```dockerfile
# Copy the local timeslice Python client library into the image
COPY pkg/client/python /opt/timeslice-client

# Install the client library and its dependencies (grpcio, protobuf, etc.)
RUN pip install /opt/timeslice-client
```

### Step 2: Configure KubeRay `RayJob` with DRA Resource Claims
When deploying Slime across independent Ray clusters, use KubeRay `RayJob` manifests configured with **Kubernetes Dynamic Resource Allocation (DRA)** (`resourceClaims`). Bounding containers to shared DRA claims (`shared-trainers-gpu-claim` and `shared-samplers-gpu-claim`) instead of static `nvidia.com/gpu` limits allows multiple jobs' worker pods to co-locate on the same physical GPU nodes without scheduler blocking:

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: slime-job-a
spec:
  rayClusterSpec:
    workerGroupSpecs:
    - groupName: trainer-group
      template:
        metadata:
          labels:
            timeslice.io/group: trainers
            timeslice.io/job-id: slime-job-a
        spec:
          nodeSelector:
            group.timeslice.io/trainers: "true"
          containers:
          - name: ray-worker
            image: my-registry/slime-modified:latest
            env:
            - name: TIMESLICE_JOB_ID
              value: "slime-job-a"
            resources:
              claims:
              - name: gpu-claim
            lifecycle:
              postStart:
                exec:
                  command: ["/bin/sh", "-c", "/opt/scripts/setup_node.sh"]
          resourceClaims:
          - name: gpu-claim
            resourceClaimName: shared-trainers-gpu-claim
```

> [!TIP]
> Example KubeRay templates and initialization scripts, see **[`sync/ray-job.yaml.template`](sync/ray-job.yaml.template)** and **[`sync/setup_node.sh`](sync/setup_node.sh)**.

---

## 5. Submitting and Observing Time-Sliced Jobs

Once the platform is deployed and the Slime code is integrated, you can submit multiple jobs and observe them sharing the GPUs.

### Step 1: Submit Multiple Jobs
Deploy two independent Slime jobs to the cluster (e.g., `slime-job-a` and `slime-job-b`).
Ensure they have unique `TIMESLICE_JOB_ID` environment variables.

### Step 2: Port-Forward the Orchestrator
To monitor the orchestrator state from your local machine, port-forward the gRPC service:
```bash
kubectl port-forward svc/timeslice-acceleratororchestrator 50051:50051 -n timeslice-system
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
   kubectl logs -n timeslice-system -l app.kubernetes.io/name=acceleratororchestrator --tail=100 -f
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
   * **Wall-Clock Progress**: Because the GPUs are shared, the wall-clock time per step will increase by a factor of $N$ (where $N$ is the number of co-located jobs), minus any gains from overlapping CPU-heavy phases (like reward processing or data loading) of one job with the other job's GPU phases.

### B. Observing Job Completion
When a Slime job completes its designated number of iterations:
1. **Graceful Exit**: The `OrchestratorClient` context manager or the `.close()` method will clean up the gRPC channels and permanently release any remaining locks.
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

