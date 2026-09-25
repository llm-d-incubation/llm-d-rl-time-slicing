# Deploying Snapshot Agent

This directory contains the Helm chart for deploying the Snapshot Agent DaemonSet in a Kubernetes cluster.

Public images are published to `ghcr.io/llm-d-incubation/llm-d-rl-time-slicing/*` by CI: `latest` on every merge to main; versioned tags via a manual workflow run.

## Prerequisites

*   A Kubernetes cluster with GPU nodes (NVIDIA).
*   `kubectl` configured to connect to your cluster.
*   `helm` (v3+) installed.

## Deployment with Helm

> [!IMPORTANT]
> The Snapshot Agent is hardcoded to be deployed in the `timeslice-system` namespace. Consequently, the Helm chart creates resources specifically in the `timeslice-system` namespace.

To deploy the agent independently using the local Helm chart:

1.  **Install the chart**:
    From the `deploy` directory, install the chart into the `timeslice-system` namespace (creating it if it doesn't exist):
    ```bash
    helm install snapshot-agent ./snapshot-agent \
      --namespace timeslice-system \
      --create-namespace
    ```
    This will deploy the agent as a `DaemonSet` and set up the required RBAC permissions:
    *   Creating a `ServiceAccount` for the agent.
    *   Creating a `ClusterRole` and `ClusterRoleBinding` granting permissions to `get`, `list`, and `watch` pods and nodes, and `get` on `nodes/proxy`.
    *   Configuring the agent pods to use this `ServiceAccount`.

2.  **Verify the deployment**:
    ```bash
    kubectl get pods -n timeslice-system -l app.kubernetes.io/name=snapshot-agent
    ```

3.  **Uninstall the chart**:
    ```bash
    helm uninstall snapshot-agent --namespace timeslice-system
    ```

## Deployment on GKE GPU Clusters

### 1. Requirements

*   A GKE cluster with at least one GPU node pool.
*   The NVIDIA GPU device driver must be installed on the nodes (e.g., using the [GKE GPU driver installer](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#installing_drivers)).

### 2. Default Configuration for GKE

*   `nvidia.driver.hostPath`: `/home/kubernetes/bin/nvidia` (Standard path for GPU drivers on GKE COS).
*   `nvidia.devices.hostPath`: `/dev` (Standard path for device access).
*   `tolerations`: Includes `nvidia.com/gpu` to allow the agent to run on GPU-tainted nodes.

### 3. Installation on GKE

To install the chart on GKE, ensuring it only targets nodes with GPUs:

```bash
helm install snapshot-agent ./snapshot-agent \
  --namespace timeslice-system \
  --create-namespace \
  --set-string "nodeSelector.cloud\.google\.com/gke-gpu=true"
```

### 4. Customizing for Ubuntu Nodes on GKE

If your GKE nodes are using Ubuntu instead of COS, you may need to override the driver path:

```bash
helm install snapshot-agent ./snapshot-agent \
  --namespace timeslice-system \
  --create-namespace \
  --set-string "nodeSelector.cloud\.google\.com/gke-gpu=true" \
  --set nvidia.driver.hostPath=/usr/lib/nvidia
```

### 5. Deploying on Non-GKE GPU Clusters

If you are deploying the snapshot agent to a non-GKE cluster (e.g., EKS, AKS, or bare-metal), you will likely need to adjust the node selector and driver paths because they differ from GKE defaults.

#### A. Override GPU Node Selector
Non-GKE clusters typically use different labels to identify GPU nodes. For example, standard NVIDIA GPU nodes often use `nvidia.com/gpu=true` or `hardware=gpu`. 

You can override the GKE-default node selector during installation:

```bash
helm install snapshot-agent ./snapshot-agent \
  --namespace timeslice-system \
  --create-namespace \
  --set-string "nodeSelector.nvidia\.com/gpu=true"
```

*Note: You may need to escape the dots in the label key as shown above (`nodeSelector.nvidia\.com/gpu=true`).*

#### B. Override NVIDIA Driver Host Path
On non-GKE clusters, the NVIDIA driver libraries might be installed in different locations on the host. Common paths include:
*   `/usr/lib/nvidia`
*   `/usr/local/nvidia`
*   `/usr/lib/x86_64-linux-gnu`

You can override the host path using:

```bash
helm install snapshot-agent ./snapshot-agent \
  --namespace timeslice-system \
  --create-namespace \
  --set nvidia.driver.hostPath=/usr/lib/nvidia
```

#### C. Override Tolerations
If your GPU nodes have different taints than the default `nvidia.com/gpu=present:NoSchedule`, you must override the tolerations. For example, if your nodes are tainted with `sku=gpu:NoSchedule`:

```bash
helm install snapshot-agent ./snapshot-agent \
  --namespace timeslice-system \
  --create-namespace \
  --set tolerations[0].key=sku \
  --set tolerations[0].operator=Equal \
  --set tolerations[0].value=gpu \
  --set tolerations[0].effect=NoSchedule
```

## Development Workflow: Custom Images

During development, you will need to build your own container image containing your changes and push it to a custom registry.

### 1. Build and Push the Image

We use the provided `Makefile` targets to build and push the container image.

1.  Define your custom registry and version (tag) by setting them as environment variables:
    ```bash
    export REGISTRY=your-custom-registry.com/your-project
    export VERSION=dev-$(git rev-parse --short HEAD)
    ```
2.  Run the following make target from the repository root to build and push the image:
    ```bash
    make snapshot-agent-image-push
    ```
    This will build the image and push it to `your-custom-registry.com/your-project/llm-d-rl-time-slicing/snapshot-agent:dev-<hash>` (the repo name comes from `PROJECT_NAME`, also overridable).

### 2. Deploy with your Custom Image

Once your image is pushed, you can instruct Helm to use it.

#### Option A: Via Command Line Flags (Recommended for Development)

```bash
helm install snapshot-agent ./snapshot-agent \
  --namespace timeslice-system \
  --create-namespace \
  --set image.repository=your-custom-registry.com/your-project/snapshot-agent \
  --set image.tag=dev
```

#### Option B: Via `values.yaml`

Edit `deploy/snapshot-agent/values.yaml` directly:

```yaml
image:
  repository: your-custom-registry.com/your-project/snapshot-agent
  pullPolicy: IfNotPresent
  tag: "dev"
```

And then run:
```bash
helm install snapshot-agent ./snapshot-agent \
  --namespace timeslice-system \
  --create-namespace
```

## GPU-CR Direct Memory Backend (`directMemory.*`)

The `direct_memory` backend parks and resumes a workload's **entire GPU
state** via GPU-CR. The workload's preloader drains the device memory it
tracked into node-local hugepage-backed files over a pinned DMA path
*before* the CUDA context is frozen with `cuda-checkpoint`, so the bulk
bytes bypass that tool's slow pageable copy — park and resume run several
times faster for the same VRAM, and the parked state lives in files that
survive independently of the process's memory.

It is gated behind the `directMemory` values block and **disabled by
default** — with `directMemory.enabled=false` the rendered chart is
identical to a plain CUDA/app-backend deployment.

```bash
helm install snapshot-agent ./snapshot-agent \
  --namespace timeslice-system \
  --create-namespace \
  --set directMemory.enabled=true
```

Enabling it adds:

*   The checkpoint directory shared with workloads, mounted at
    `directMemory.ctlDir` (default `/mnt/huge-ckpt`) from
    `directMemory.hostCtlPath` (default `/var/tmp/huge-ckpt`), with
    `mountPropagation: HostToContainer` so the init containers' mounts are
    visible. The agent's `EXPORT_FILE_PATH` and per-operation deadline
    (`DIRECT_MEMORY_OP_TIMEOUT_SEC`, `opTimeoutSec`, default 120 s) are set
    from this block.
*   The `DirectMemoryBackend` feature gate, implied by
    `directMemory.enabled=true` (an explicit `featureGates` entry overrides
    the implied value).
*   A `PriorityClass` (`priorityClass.*`) so the pod — and in particular its
    hugepage bootstrap — wins node placement over GPU workloads.
*   Three privileged init containers that `nsenter` the host mount
    namespace, all idempotent per node boot (rendered while
    `directMemory.hugetlbfs.mount` is on, the default):
    *   `provision-hugepages` (`hugetlbfs.bootstrap.*`) — writes
        `vm.nr_hugepages` (`pages2Mi`, default 12288 = 24 Gi; size it to
        your workloads' dump buffers plus headroom) and restarts the
        kubelet once so the node publishes `hugepages-2Mi` capacity for
        WORKLOAD pods. No nodepool hugepage configuration or special node
        image is needed.
    *   `mount-hugetlbfs` — mounts hugetlbfs at `hostCtlPath`
        (`pagesize=2M,mode=0777`). GPU-CR pins its dump and staging files
        for DMA, which requires hugepage-backed files; without this mount
        every dump silently degrades to boot-disk page cache. Turn
        `hugetlbfs.mount` off only if the path already is a hugetlbfs.
    *   `mount-ctl-tmpfs` — mounts a small tmpfs (`ctlTmpfsSizeMi`, default
        64) nested at `<hostCtlPath>/ctl` for the control files through
        which the agent's `cr_client` and the workload's preloader
        coordinate. Both sides discover it through the store mount they
        already share — no configuration on either side — and keeping
        control files off hugetlbfs is what lets the agent run with no
        hugepage request.

The agent requests **no `hugepages-2Mi` at all**: dump bytes are written by
the workload's own process, and the agent's control traffic stays on the
tmpfs. That zero request is what lets the DaemonSet schedule on fresh nodes
*before* hugepage capacity exists and absorb the hugepage bootstrap as an
init container. (`directMemory.hugepagesResource` exists only for older
GPU-CR builds that keep control files on the store, on nodes whose pool is
already provisioned.)

Misconfigurations fail early:

*   An undersized or unallocatable hugepage pool makes `provision-hugepages`
    exit 1 **without** the kubelet restart — it surfaces as this pod
    CrashLooping, never as workload SIGBUS.
*   `hugetlbfs.bootstrap` combined with a pagesize other than `2M`, or with
    `hugepagesResource` (a pod that requests hugepages cannot schedule
    before its own bootstrap publishes capacity), fails at render time.

Workload pods need the GPU-CR preloader (`LD_PRELOAD=vGPU-NVIDIA.so`),
`hostPID`, the checkpoint dir hostPath mounted at `/mnt/huge-ckpt` with
`mountPropagation: HostToContainer`, and `hugepages-2Mi` resources sized for
their dump buffers — see the workload example in
[guides/snapshot-agent](../../guides/snapshot-agent/README.md).

There is no `cr_client` install step: the binary is built from
`third_party/gpu-cr` into the agent image at `/usr/local/bin/cr_client`, so
the agent, `cr_client`, and the preloader source always roll together.
`grpc.health.v1.Health/Check` with `service: "direct-memory"` reports
`NOT_SERVING` if the binary is missing.

> [!NOTE]
> Clusters that previously ran a manually deployed agent may already have a
> non-Helm `timeslice-snapshot-agent` PriorityClass; `helm install` refuses
> to adopt it. Delete it once before installing:
> `kubectl delete priorityclass timeslice-snapshot-agent`.

## GPU-CR Memory-Regions Backend (`memoryRegions.*`)

The `memory_regions` backend checkpoints and restores **explicit ranges** of
a workload's device memory into named snapshot slots, while the workload
keeps running — the use case is swapping one set of weights for another
(for example alternating LoRA adapters through a single vLLM slot) without
parking the whole process. `direct_memory` (above) parks everything;
`memory_regions` touches only the ranges the caller names.

It is gated behind the `memoryRegions` values block and **disabled by
default** — with `memoryRegions.enabled=false` the rendered chart is
identical to a plain CUDA/app-backend deployment.

```bash
helm install snapshot-agent ./snapshot-agent \
  --namespace timeslice-system \
  --create-namespace \
  --set memoryRegions.enabled=true
```

Both GPU-CR backends run on the same node plumbing, so **this block adds
almost nothing of its own**. Everything described under
[Direct Memory](#gpu-cr-direct-memory-backend-directmemory) — the shared
checkpoint store, the hugetlbfs mount, the control tmpfs, the hugepage
bootstrap, the PriorityClass, and the agent's zero hugepage request —
renders identically when `memoryRegions.enabled=true`, configured by the
same `directMemory.ctlDir` / `hostCtlPath` / `hugetlbfs.*` knobs. Enabling
this backend does **not** require `directMemory.enabled=true`; enabling
both renders one copy of the shared machinery, not two.

What is specific to this block:

*   The `MemoryRegionsBackend` feature gate, implied by
    `memoryRegions.enabled=true` (an explicit `featureGates` entry
    overrides the implied value).
*   `GPU_CR_OP_TIMEOUT_SEC` (`memoryRegions.opTimeoutSec`, default 120 s) —
    the per-`cr_client`-invocation deadline for this backend, separate from
    `direct_memory`'s so the two can be tuned independently.

Workload requirements are the same as for `direct_memory` (preloader,
`hostPID`, the shared checkpoint dir, hugepages sized for its buffers),
plus one addition: the caller must know the device addresses and sizes it
wants checkpointed, because this backend does no discovery. See the
backend section and the region/slot API in
[guides/snapshot-agent](../../guides/snapshot-agent/README.md).
`grpc.health.v1.Health/Check` with `service: "memory-regions"` reports
whether `cr_client` is available.
