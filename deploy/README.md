# Deploying Accelerator Orchestrator

This directory contains the Helm chart for deploying the Accelerator Orchestrator in a Kubernetes cluster.

## Prerequisites

*   A Kubernetes cluster.
*   `kubectl` configured to connect to your cluster.
*   `helm` (v3+) installed.

## Deployment with Helm

To deploy the orchestrator using the local Helm chart:

1.  **Install the chart**:
    From the `deploy` directory:
    ```bash
    helm install acceleratororchestrator ./acceleratororchestrator
    ```
    This will deploy the orchestrator with the default configuration, which includes:
    *   Creating a `ServiceAccount` for the orchestrator.
    *   Creating a `ClusterRole` and `ClusterRoleBinding` granting the service account permissions to:
        *   Read (`get`, `list`, `watch`) `pods` and `nodes`.
        *   Read and Write (`create`, `update`, `patch`, `delete`) `configmaps`.
    *   Configuring the orchestrator pod to use this `ServiceAccount`.

2.  **Verify the deployment**:
    ```bash
    kubectl get pods -l app.kubernetes.io/name=acceleratororchestrator
    ```

3.  **Uninstall the chart**:
    ```bash
    helm uninstall acceleratororchestrator
    ```

## Development Workflow: Custom Images

During development, you will need to build your own container image containing your changes and push it to a custom registry (e.g., Google Container Registry, Artifact Registry, Docker Hub, or a local registry like Kind/Minikube).

### 1. Build and Push the Image

We use the provided `Makefile` targets to build and push the container image. The Makefile uses `docker buildx` under the hood to build multi-arch images (amd64/arm64) and push them to your registry.

1.  Define your custom registry and version (tag) by setting them as environment variables:
    ```bash
    export REGISTRY=your-custom-registry.com/your-project
    export VERSION=dev-$(git rev-parse --short HEAD)
    ```
2.  Run the following make target from the repository root to build and push the image:
    ```bash
    make image-push-orchestrator
    ```
    This will build the image using `docker/acceleratororchestrator/Dockerfile` and push it to `your-custom-registry.com/your-project/acceleratororchestrator:dev-<hash>`.

### 2. Deploy with your Custom Image

Once your image is pushed, you can instruct Helm to use it.

#### Option A: Via Command Line Flags (Recommended for Development)

This avoids modifying files in your git tree:

```bash
helm install acceleratororchestrator ./acceleratororchestrator \
  --set image.repository=your-custom-registry.com/your-project/acceleratororchestrator \
  --set image.tag=dev
```

#### Option B: Via `values.yaml`

Edit `deploy/acceleratororchestrator/values.yaml` directly:

```yaml
image:
  repository: your-custom-registry.com/your-project/acceleratororchestrator
  pullPolicy: IfNotPresent
  tag: "dev"
```

And then run:
```bash
helm install acceleratororchestrator ./acceleratororchestrator
```
