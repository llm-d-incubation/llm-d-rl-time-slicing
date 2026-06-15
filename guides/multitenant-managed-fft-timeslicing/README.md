# GPU Time-Slicing for Multi-Tenant Managed RL Services

This guide is for developers building a managed RL service who want to enable multiple tenants to share GPU nodes by time-slicing their RL workloads. This applies to any GPU-bound RL component — trainer, sampler, or both, each running as a separate pod. The service is responsible for deciding when to preempt a running job and when to resume it. The Snapshot Agent provides the GPU checkpoint/restore primitive the service calls at those boundaries. From the tenant's perspective, their RL job simply pauses and resumes transparently

## Overview

The Snapshot Agent is a gRPC service that runs as a DaemonSet on each GPU node. It provides a local interface for application pods to:
- **Snapshot:** Save the current state of a GPU job.
- **Restore:** Reload a previously saved state to the GPU.
- **Status:** Monitor the current state of jobs and accelerators.

By using the Snapshot Agent, you can implement time-slicing without modifying the core LLM engine code, simply by wrapping your execution loop with the `timeslice.SnapshotAgentClient` client library.

## Architecture

1.  **Snapshot Agent (DaemonSet):** Runs on every node with GPUs. It has privileged access to the GPU devices and host paths required for snapshotting.
2.  **Tenant's RL Workload:**: A sampler or trainer running on a pod.
3.  **Snapshot Agent Client:** A Python library used by the application pod to communicate with the local Snapshot Agent via gRPC (port 9001).

---

## 1. Deploying the Snapshot Agent

The Snapshot Agent must be deployed as a DaemonSet to ensure it is available on every node.

### Using Helm
```bash
cd ../../deploy/
helm install snapshot-agent ./snapshot-agent
```

### Key Configuration (`values.yaml`)
- `port`: The gRPC port (default: `9001`).
- `securityContext.privileged`: Must be `true` to access GPU registers.
- `nvidia.driver.hostPath`: Path to NVIDIA driver binaries on the host (e.g., `/home/kubernetes/bin/nvidia`).

See [helm-snapshot-agent.md](../../deploy/snapshot-agent/README.md) for more details on the Helm chart.

---

## 2. Integrating with Tenant's RL Workload

To enable a pod for time-slicing, you need to add specific labels and environment variables to your Deployment.

### Required Labels
The Snapshot Agent identifies pods that should be managed via labels:
- `timeslice.io/job-id: "<unique-job-id>"`: A unique identifier for the job.

### Environment Variables
The pod needs to know the IP of the node it is running on to connect to the local Snapshot Agent.

```yaml
env:
  - name: NODE_IP
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP
  - name: AGENT_ENDPOINT
    value: "$(NODE_IP):9001"
```

---

## 3. Using the Python Client

The `timeslice.SnapshotAgentClient` library provides a high-level Python API for interacting with the Snapshot Agent.

### Installation
```bash
cd ../../pkg/client/python
pip install .
```

### Choosing a Backend
The `backend` parameter controls how GPU state is checkpointed. It is an optional argument in the following `SnapshotAgentClient` methods:

*   `snapshot(job_id, group, backend=...)`
*   `restore(job_id, group, backend=...)`
*   `snapshot_and_wait(job_id, group, backend=...)`
*   `restore_and_wait(job_id, group, backend=...)`

Available backends:
*   **BACKEND_CUDA (default):** Full process-level GPU checkpoint via `cuda-checkpoint`. Suitable for any GPU-bound RL component (trainer, sampler) running full model weights.
*   Additional backends for lighter-weight workloads (e.g. LoRA adapters) are planned for a future release.

### Basic Workflow
The most common usage is to trigger a snapshot at the end of a "slice" and a restore at the beginning of the next one.

```python
from timeslice.snapshot_agent import SnapshotAgentClient

# AGENT_ENDPOINT is typically $(NODE_IP):9001
endpoint = os.getenv("AGENT_ENDPOINT", "localhost:9001")
job_id = "test-job"
group = "default"

with SnapshotAgentClient(endpoint) as client:
    # 1. Trigger Snapshot and wait for completion
    print("Snapshotting...")
    result = client.snapshot_and_wait(job_id, group)
    if result.status == "OPERATION_STATUS_COMPLETE":
        print(f"Snapshot success! Used {result.storage_bytes} bytes")

    # ... wait for your turn (orchestrated externally) ...

    # 2. Trigger Restore and wait for completion
    print("Restoring...")
    result = client.restore_and_wait(job_id, group)
    if result.status == "OPERATION_STATUS_COMPLETE":
        print("Restore success!")
```

### Advanced Usage
For more granular control, you can use asynchronous methods and poll for status manually.

```python
# Start snapshot
response = client.snapshot(job_id, group)
op_id = response.operation_id

# Do other work...

# Wait for completion
result = client.wait_for_operation(op_id)
```

---

## 4. Monitoring and Troubleshooting

### Checking Agent Status
You can query the agent for the status of all managed jobs:
```python
status = client.status()
for job in status.job_statuses:
    print(f"Job {job.job_id}: {job.state}")
```

### Direct gRPC Access
For debugging, you can use `grpcurl`:
```bash
# Get node IP first
NODE_IP=$(kubectl get pod <agent-pod> -o jsonpath='{.status.hostIP}')

grpcurl -plaintext -d '{"job_id": "test-job", "group": "default"}' \
  $NODE_IP:9001 snapshot_agent.v1alpha1.SnapshotAgentService/Snapshot
```

### Common Issues
- **Permission Denied:** Ensure the Snapshot Agent pod is running as `privileged: true`.
- **Connection Refused:** Verify the `AGENT_ENDPOINT` environment variable correctly points to `$(NODE_IP):9001`.
- **GPU Not Found:** Check that the `nvidia.driver.hostPath` in the agent's configuration matches your node's setup.
