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

To install the chart:

```bash
helm install timeslice .
```

This will install the Helm release metadata in your current default namespace, but all Kubernetes resources will be deployed to `timeslice-system`.

If you prefer to have the Helm release metadata also reside in the `timeslice-system` namespace, you must use the `--create-namespace` flag (or ensure the namespace exists beforehand):

```bash
helm install timeslice . -n timeslice-system --create-namespace
```

To install with custom values:

```bash
helm install timeslice . -f my-values.yaml
```

### 4. Uninstallation

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
