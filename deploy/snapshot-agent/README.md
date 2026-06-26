# Deploying Snapshot Agent

This directory contains the Helm chart for deploying the Snapshot Agent DaemonSet in a Kubernetes cluster.

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
    make image-push-snapshot-agent
    ```
    This will build the image and push it to `your-custom-registry.com/your-project/snapshot-agent:dev-<hash>`.

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
