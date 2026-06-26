# Timeslice Parent Helm Chart

This is the parent Helm chart that coordinates the deployment of both the **Accelerator Orchestrator** and the **Snapshot Agent**.

## Directory Structure

*   `Chart.yaml`: Defines the parent chart and its dependencies.
*   `values.yaml`: Allows overriding configuration for both subcharts.
*   `acceleratororchestrator/`: Subchart for the Accelerator Orchestrator.
*   `snapshot-agent/`: Subchart for the Snapshot Agent DaemonSet.

## Prerequisites

*   Helm v3 installed.
*   Access to a Kubernetes cluster.
*   GPU nodes must run **NVIDIA GPU Driver 565 or later** to support DRA.

## Usage

### 1. Initialize/Update Dependencies

Because this chart uses local subcharts as dependencies, you must build the dependencies before deploying. Run the following command from this directory (`deploy/`):

```bash
helm dependency update .
```

This will look at the `dependencies` section in `Chart.yaml`, package the local subcharts, and place them in a `charts/` directory (which should be ignored or will be created dynamically).

### 2. Configuration

You can configure the subcharts by modifying the parent `values.yaml` file. Values for each subchart must be nested under the subchart's name.

DRA is required for timeslicing. This helm file includes the **NVIDIA DRA Driver**, which is included by default for convenience. If your cluster already has the DRA driver installed, you can disable it by setting `nvidia-dra-driver-gpu.enabled` to `false`.

Example `values.yaml`:

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
  # Default is for COS nodes. For Ubuntu nodes, change to "/opt/nvidia"
  nvidiaDriverRoot: "/home/kubernetes/bin/nvidia/"
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

To deploy the system on a GKE cluster with GPU nodes, you should use the GKE-specific configuration file `values-gke.yaml` to apply GKE-specific defaults (such as node selectors).

The `values-gke.yaml` file contains:
*   **Target GKE GPU Nodes**: Targets nodes labeled with `cloud.google.com/gke-gpu=true`.

#### DRA (Dynamic Resource Allocation) Requirement
The `timeslice` system **requires DRA** to function. 
*   **Enabled by Default**: By default, this Helm chart will attempt to install the Nvidia DRA driver (`nvidia-dra-driver-gpu`) as a dependency.
*   **Disabling the Driver**: If you have **already installed the DRA driver via other means** (e.g., a cluster-wide operator), or if you are validating in an environment where you want to bypass the chart's driver installation, you can disable it by setting `nvidia-dra-driver-gpu.enabled=false`.

#### Example A: Deploying on GKE (Default Public Images, Installing DRA)
To deploy using the default public images and install the DRA driver:
```bash
helm upgrade --install timeslice . -f values-gke.yaml
```

#### Example B: Deploying on GKE (DRA Driver Already Installed / Disabled)
If you have already installed the DRA driver via other means, disable the chart's driver installation:
```bash
helm upgrade --install timeslice . -f values-gke.yaml --set nvidia-dra-driver-gpu.enabled=false
```

#### Example C: Deploying on GKE with a Custom Registry (Development & Validation)
If you are developing and validating using a custom registry, you can combine the GKE infrastructure file with your custom registry settings.

##### Option 1: Chaining Multiple Values Files (Canonical)
Create a development-specific values file (e.g., `values-dev.yaml`) containing your registry overrides:
```yaml
# values-dev.yaml
acceleratororchestrator:
  image:
    repository: your-custom-registry.com/your-project/acceleratororchestrator
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
  --set acceleratororchestrator.image.repository=your-custom-registry.com/your-project/acceleratororchestrator \
  --set snapshot-agent.image.repository=your-custom-registry.com/your-project/snapshot-agent
```

*Note: If you also need to disable the DRA driver (e.g., if already installed), just append `--set nvidia-dra-driver-gpu.enabled=false` to the commands above.*

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
