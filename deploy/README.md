# Timeslice Parent Helm Chart

This is the parent Helm chart that coordinates the deployment of both the **TimeSlice Orchestrator** and the **Snapshot Agent**.

Public images are published to `ghcr.io/llm-d-incubation/llm-d-rl-time-slicing/*` by CI
(`.github/workflows/ci-release.yaml`): `timesliceorchestrator`, `snapshot-agent`,
`guest-kubelet`, `gpu-shadow-plugin`, `timeslice-webhook`, `donor-controller`,
`rl-integration`, and `verl` (verl with the timeslice integration built in,
`docker/verl/Dockerfile`). Tags:

*   `latest` on every merge to main;
*   on every push to a release branch (`demo`, `demo-*`, `demo/**`), or a manual
    run on any other branch without a version: an immutable `demo-<short sha>`
    and a moving tag named after the branch (`/` becomes `-`);
*   a manual run with a version: that version.

`global.imageTag` sets the tag of every platform image at once (a component's
own `image.tag` wins). On the `demo/batch` branch it defaults to `demo-batch`,
the moving tag of that branch's newest release, so an install from a clone of
the branch needs no tag (`""` means `latest`, which follows main). Pin it for a
reproducible install, for example to the commit you cloned:

```bash
helm install timeslice . -n timeslice-system --create-namespace \
  --set global.imageTag=demo-$(git rev-parse HEAD | cut -c1-7)
```

## Directory Structure

*   `Chart.yaml`: Defines the parent chart and its dependencies.
*   `values.yaml`: Allows overriding configuration for both subcharts.
*   `values-inject-verl.yaml`: Overlay for RL jobs on a stock verl image (turns on the webhook's package injection; the default assumes the prebuilt verl image).
*   `timesliceorchestrator/`: Subchart for the TimeSlice Orchestrator.
*   `snapshot-agent/`: Subchart for the Snapshot Agent DaemonSet.

The guest time-slicing subcharts `guest-kubelet/` (the virtual kubelet),
`timeslice-webhook/` (admission webhook and policies), `donor-controller/` and
`gpu-shadow/` (the GPU shadow device plugin for the guest kubelet's
`--gpu-mode=pooled`) are **on by default**: together they are the shadow path
described below. `gpu-metrics/` (a DCGM exporter with 1 s samples, including
SM activity and occupancy) is also on by default. The chart does not install a DRA driver. Each chart's `values.yaml` documents its
values. Set `createNamespace: false` to install into a namespace that already
exists.

## The default: shadow path (no DRA)

A normal GKE GPU cluster with the NVIDIA device plugin is enough:

```bash
helm dependency update .
helm upgrade --install timeslice . -n timeslice-system --create-namespace
```

The defaults place the snapshot agent on GPU nodes only, detect each host's
GPU model and memory in the guest kubelet, and pull a guest's images onto the
host as soon as the guest is admitted (`guest-kubelet --prepull-images`), so the
first lend does not wait for a pull.

What each team does after that:

*   **RL team** (trainer pods lend their GPUs while idle), in a normal KubeRay
    RayJob:
    *   label the trainer worker group `timeslice.io/donor: "true"`;
    *   add one rayStartParams `resources` line per worker group and
        `ray_pg_extra_resources=...` (Ray placement pinning, so verl's trainer
        lands on the donor group);
    *   set `async_training.trainer_name=timeslice`.
    At RayCluster creation the webhook (`timeslice-webhook.rayCluster`) labels
    the head and the other worker groups for the RL integration and sets
    `TIMESLICE_FULLY_ASYNC=1`, `TIMESLICE_JOB_ID`, `TIMESLICE_GROUP` and
    `TIMESLICE_ORCH_ADDR` in every group (values you set are kept). The RL
    team runs the prebuilt verl image
    (`ghcr.io/llm-d-incubation/llm-d-rl-time-slicing/verl`: stock
    `verlai/verl:vllm020.dev2` plus a pinned verl fork with the fully-async
    lifecycle hooks, verl 983cb0f2 + 2 commits, Apache-2.0, the timeslice client
    and verl plugin, TransferQueue and cupy-cuda12x). Do not `pip install` verl
    at pod start with that image; it would replace the fork. Donor pods also get their memory limit raised by one GPU's memory
    per GPU (`timeslice-webhook.flags.donor-gpu-memory`, default: by GPU model),
    because the trainer's checkpoint lands in pod memory.
*   **Batch team**: label guest pods `timeslice.io/guest: "true"`. The webhook
    steers them onto the virtual nodes, where the guest kubelet runs them on the
    pooled `timeslice.io/gpu-shadow` resource. By default
    (`timeslice-webhook.flags.guest-steering=preferred`) a guest prefers lent
    GPUs and, when none is lent, schedules on a regular GPU node and stays
    there (placement is decided once), so batch and RL jobs can be submitted
    in any order; a Pending guest may make the cluster autoscaler add a GPU
    node. `guest-steering=required` keeps guests on lent GPUs only: they stay
    Pending until a GPU is lent.

How it fits together:

*   The donor controller labels the nodes running donor pods
    (`timeslice.io/donor=true`); the `gpu-shadow` DaemonSet runs there and
    advertises `timeslice.io/gpu-shadow` for the GPUs the donor pod holds (0 when
    more than one pod holds `nvidia.com/gpu` on the node).
*   The guest kubelet registers a virtual node per donor node and runs guest pods
    on the lent GPUs while the trainer is idle; before the trainer resumes it
    taints the host `timeslice.io/gpu-fence` and drains the guests.
*   Only the guest kubelet's service account may request
    `timeslice.io/gpu-shadow*`: a ValidatingAdmissionPolicy enforces it
    cluster-wide (`timeslice-webhook.shadowPolicy`) and the webhook rejects the
    same requests with a clearer message. Privileged guests are refused.
*   The webhook serving certificate and CA are generated by the chart and kept
    across upgrades (`timeslice-webhook.tls.rotate=true` issues new ones).
*   Cross-references (orchestrator address, guest kubelet service account) are
    derived from the release name and namespace; trainers and guests may live in
    any namespace.

Stock verl image: instead of the prebuilt image, the RL team can keep its own
verl image. Install with `-f values-inject-verl.yaml`
(`timeslice-webhook.rlIntegration.enabled=true`): every pod of a Ray cluster
that has a donor group then gets an init container
(`timeslice-webhook.rlIntegration`) that copies the timeslice Python packages
and the same pinned verl fork (`rlIntegration.injectVerl`, on by default) in
front of `PYTHONPATH`. The image must provide Python >= 3.10, `grpcio` and
`protobuf` >= 6 (stock verl images do).

Images: CI publishes every image the chart uses (see the top of this page).
To use your own registry, set the repositories of `timesliceorchestrator`,
`snapshot-agent`, `guest-kubelet`, `timeslice-webhook`,
`timeslice-webhook.rlIntegration`, `donor-controller` and `gpu-shadow`
(Dockerfiles: `docker/timesliceorchestrator/`, `docker/snapshot-agent/`,
`guest-kubelet/Dockerfile`, `guest-kubelet/Dockerfile.gpu-shadow`,
`docker/timeslice-webhook/`, `deploy/donor-controller/`,
`docker/rl-integration/`; build context is the repository root).

## Prerequisites

*   Helm v3 installed.
*   Access to a Kubernetes cluster.
*   GPU nodes with the NVIDIA device plugin (`nvidia.com/gpu`).

## Usage

### 1. Initialize/Update Dependencies

Because this chart uses local subcharts as dependencies, you must build the dependencies before deploying. Run the following command from this directory (`deploy/`):

```bash
helm dependency update .
```

This will look at the `dependencies` section in `Chart.yaml`, package the local subcharts, and place them in a `charts/` directory (which should be ignored or will be created dynamically).

### 2. Configuration

You can configure the subcharts by modifying the parent `values.yaml` file. Values for each subchart must be nested under the subchart's name.

DRA is not required: the shadow path uses the device plugin, and the chart does not install a DRA driver. The DRA shared-claim path in `guides/rl-batch-interleaving` needs the NVIDIA DRA driver installed separately.

Example `values.yaml`:

```yaml
global:
  imageTag: demo-1a2b3c4   # every platform image

timesliceorchestrator:
  replicaCount: 2

snapshot-agent:
  image:
    tag: v0.1.0   # overrides global.imageTag for this component
```


### 3. Installation

The chart automatically creates the `timeslice-system` namespace, and **all** deployed resources (orchestrator, agent, RBAC, etc.) are forced into this namespace regardless of where the Helm release is installed.

To install or upgrade the chart:

```bash
helm upgrade --install timeslice .
```

This will install the Helm release metadata in your current default namespace, but all Kubernetes resources will be deployed to `timeslice-system`.

If you prefer to have the Helm release metadata also reside in the `timeslice-system` namespace, you must use the `--create-namespace` flag (or ensure the namespace exists beforehand):

```bash
helm upgrade --install timeslice . -n timeslice-system --create-namespace
```

### 4. Deploying on GKE GPU Clusters

The defaults already keep the snapshot agent on GPU nodes (GKE's `cloud.google.com/gke-accelerator` label or NVIDIA feature discovery's `nvidia.com/gpu.present=true`). `values-gke.yaml` is optional: it pins the agent with a `nodeSelector` instead.

The `values-gke.yaml` file contains:
*   **Target GKE GPU Nodes**: Targets nodes labeled with `cloud.google.com/gke-gpu=true`.

#### Example A: Deploying on GKE (Default Public Images)
```bash
helm upgrade --install timeslice . -n timeslice-system --create-namespace -f values-gke.yaml
```

#### Example B: Deploying on GKE with a Custom Registry (Development & Validation)
If you are developing and validating using a custom registry, you can combine the GKE infrastructure file with your custom registry settings.

##### Option 1: Chaining Multiple Values Files (Canonical)
Create a development-specific values file (e.g., `values-dev.yaml`) containing your registry overrides:
```yaml
# values-dev.yaml
timesliceorchestrator:
  image:
    repository: your-custom-registry.com/your-project/timesliceorchestrator
snapshot-agent:
  image:
    repository: your-custom-registry.com/your-project/snapshot-agent
```

Then deploy by chaining the values files in order (later files override earlier ones):
```bash
helm upgrade --install timeslice . -f values-gke.yaml -f values-dev.yaml
```

##### Option 2: Chaining Values File with Command-Line Overrides
Alternatively, you can pass the registry overrides via `--set` flags alongside the GKE values file:
```bash
helm upgrade --install timeslice . \
  -f values-gke.yaml \
  --set timesliceorchestrator.image.repository=your-custom-registry.com/your-project/timesliceorchestrator \
  --set snapshot-agent.image.repository=your-custom-registry.com/your-project/snapshot-agent
```

*Note: set the repositories of the guest time-slicing images (`guest-kubelet`, `timeslice-webhook`, `timeslice-webhook.rlIntegration.image`, `donor-controller`, `gpu-shadow`) the same way.*

### 5. Deploying on Non-GKE GPU Clusters

If you are deploying to a non-GKE cluster (e.g., EKS or bare-metal), you do not need the GKE-specific `values-gke.yaml` file. Instead, you can customize the node selector and driver paths for your specific environment. See the [Snapshot Agent README](./snapshot-agent/README.md) for detailed instructions.

### 6. Installation with custom values

If you have a custom values file:

```bash
helm upgrade --install timeslice . -f my-values.yaml
```

### 7. Uninstallation

To uninstall/delete the `timeslice` deployment:

If you installed it without specifying a namespace (default):
```bash
helm uninstall timeslice
```

If you installed it into the `timeslice-system` namespace:
```bash
helm uninstall timeslice -n timeslice-system
```

Uninstalling the release will automatically delete the `timeslice-system` namespace and all resources within it.
