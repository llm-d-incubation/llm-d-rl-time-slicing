# Snapshot Agent

The Snapshot Agent provides GPU checkpoint/restore primitives to enable efficient resource sharing for GPU-bound workloads. By allowing processes to save and reload their entire GPU state, it enables scenarios where multiple high-memory workloads can share the same physical GPU hardware.

It can be deployed in two modes:
1. **Standalone Mode (Primary / Default):** The agent runs directly on the host (or a VM). Workloads are targeted directly by specifying their Process IDs (PIDs) in the client requests.
2. **Kubernetes Mode (Optional Automation):** The agent runs as a DaemonSet. It automatically discovers target PIDs by querying the local Kubernetes API for pods matching specific job labels.

---

## 1. Running in Standalone Mode (Primary)

In standalone mode, you run the `snapshot-agent` binary directly on your host machine (e.g., a GCE VM). Workloads are targeted by specifying their Process IDs (PIDs) directly in the client request.

### Starting the Agent

By default, the agent starts in standalone mode on port `9001`:
```bash
# Run the binary
./snapshot-agent-linux

# Or explicitly set the port and mode:
./snapshot-agent-linux -deployment-mode=standalone -port=9001
```

### Triggering a Snapshot (with PIDs)

Since the agent is in standalone mode, it cannot auto-discover PIDs. You must explicitly provide the target PIDs.

#### Using `grpcurl`

Specify the PIDs under the `backend_config.cuda.explicit_target` payload:

```bash
grpcurl -plaintext \
  -import-path pkg/snapshot-agent/api/v1alpha1 \
  -proto pkg/snapshot-agent/api/v1alpha1/snapshot_agent.proto \
  -d '{
    "job_id": "test-job",
    "backend_config": {
      "cuda": {
        "explicit_target": {
          "pids": [1234]
        }
      }
    }
  }' \
  localhost:9001 \
  snapshot_agent.v1alpha1.SnapshotAgentService/Snapshot
```

#### Using the Python Client

Install the client:
```bash
pip install "git+https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git#subdirectory=pkg/client/python"
```

Trigger a snapshot by passing the target PIDs using a `BackendConfig` object:
```python
from timeslice.snapshot_agent import SnapshotAgentClient
from timeslice.snapshot_agent import snapshot_agent_pb2

# Connect to the local agent
with SnapshotAgentClient("localhost:9001") as client:
    # Define the backend config with target PIDs
    backend_config = snapshot_agent_pb2.BackendConfig(
        cuda=snapshot_agent_pb2.CudaBackendConfig(
            explicit_target=snapshot_agent_pb2.ProcessTarget(pids=[1234])
        )
    )

    # Trigger snapshot and wait for completion
    result = client.snapshot_and_wait(
        job_id="test-job",
        backend_config=backend_config,
    )
    if result.status == "OPERATION_STATUS_COMPLETE":
        print(f"Snapshot succeeded in {result.elapsed_ms} ms")
```

---

## 2. Running in Kubernetes Mode (Optional Automation)

If you are deploying workloads inside a Kubernetes cluster, the Snapshot Agent can run as a `DaemonSet` and automatically discover the GPU process PIDs of your pods.

### Deploying the Agent
The agent must be deployed as a privileged DaemonSet on every GPU node.

#### Using Helm
Follow the instructions in [deploy/snapshot-agent/README.md](../../deploy/snapshot-agent/README.md) to deploy.

Key settings in `values.yaml`:
* `port`: The gRPC port (default: `9001`).
* `securityContext.privileged`: Must be `true` to access GPU registers.
* `nvidia.driver.hostPath`: Path to NVIDIA driver binaries on the host (e.g., `/home/kubernetes/bin/nvidia`).

### Integrating with Workloads
Workload pods are identified using labels. The agent queries the local Kubelet API for pods matching the target `job-id` and extracts their GPU PIDs automatically.

#### Required Labels
Add this label to your workload pods:
* `timeslice.io/job-id: "<unique-job-id>"`

#### Environment Variables
Provide the local node's IP to your workload so it can connect to the agent:
```yaml
env:
  - name: NODE_IP
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP
  - name: AGENT_ENDPOINT
    value: "$(NODE_IP):9001"
```

#### Client Call (K8s Mode)
Since the agent automatically discovers the PIDs based on the `job_id`, you do not need to pass `pids` in the client call:

```python
result = client.snapshot_and_wait(job_id="my-k8s-job-id")
```

---

## 3. Backends

The Snapshot Agent supports multiple backends for different GPU memory management strategies. Each backend is selected per-request via the `backend_config` field.

| Backend | Config | How it works | VRAM Freed | Resume Time |
|---------|--------|-------------|------------|-------------|
| CUDA Checkpoint | `cuda` | Process-level CUDA state save/restore via `cuda-checkpoint` | ~100% | ~1-3s |
| Application-Aware | `app_endpoint` | Suspend/resume through the application's own HTTP API (vLLM, SGLang) | ~96% | ~50-100ms |

### CUDA Checkpoint

Saves and restores the entire CUDA context of a process. Works with any GPU workload regardless of framework.

```python
from timeslice.snapshot_agent import SnapshotAgentClient, snapshot_agent_pb2

with SnapshotAgentClient("localhost:9001") as client:
    # Checkpoint
    result = client.snapshot_and_wait(
        job_id="my-job",
        backend_config=snapshot_agent_pb2.BackendConfig(
            cuda=snapshot_agent_pb2.CudaBackendConfig(
                explicit_target=snapshot_agent_pb2.ProcessTarget(pids=[1234])
            )
        ),
    )

    # Restore
    result = client.restore_and_wait(
        job_id="my-job",
        backend_config=snapshot_agent_pb2.BackendConfig(
            cuda=snapshot_agent_pb2.CudaBackendConfig(
                explicit_target=snapshot_agent_pb2.ProcessTarget(pids=[1234])
            )
        ),
    )
```

In Kubernetes mode, PIDs are discovered automatically — omit `explicit_target`:
```python
result = client.snapshot_and_wait(
    job_id="my-k8s-job",
    backend_config=snapshot_agent_pb2.BackendConfig(
        cuda=snapshot_agent_pb2.CudaBackendConfig()
    ),
)
```

### Application-Aware (app_endpoint)

Suspends and resumes an application-aware workload through its own HTTP API. The `app` field selects the application; `endpoints` targets the server(s).

**Suspend mode** states what happens to the workload's durable state while suspended:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `SUSPEND_MODE_OFFLOAD` (default) | State preserved in host memory; Restore copies it back | Standard suspend/resume |
| `SUSPEND_MODE_DISCARD` | State dropped; the application re-provisions it after Restore | RL training — push new weights after resume |

**Tags** select memory regions (`weights`, `kv_cache`, ...). If omitted, the application's full region set is used. On Snapshot, tags select what to suspend (where the application supports it); on Restore, what to bring back.

#### vLLM

Server requirements:

```bash
VLLM_SERVER_DEV_MODE=1 python -m vllm.entrypoints.openai.api_server \
  --model <model> \
  --enable-sleep-mode
```

```python
with SnapshotAgentClient("localhost:9001") as client:
    # Suspend: offload weights to CPU, discard KV cache
    client.snapshot_and_wait(
        job_id="my-vllm-job",
        backend_config=snapshot_agent_pb2.BackendConfig(
            app_endpoint=snapshot_agent_pb2.AppEndpointConfig(
                app=snapshot_agent_pb2.APP_VLLM,
                endpoints=["http://localhost:8000"],
            )
        ),
    )

    # Resume: restore all
    client.restore_and_wait(
        job_id="my-vllm-job",
        backend_config=snapshot_agent_pb2.BackendConfig(
            app_endpoint=snapshot_agent_pb2.AppEndpointConfig(
                app=snapshot_agent_pb2.APP_VLLM,
                endpoints=["http://localhost:8000"],
            )
        ),
    )
```

Partial resume — bring back only specific regions:
```python
app_endpoint=snapshot_agent_pb2.AppEndpointConfig(
    app=snapshot_agent_pb2.APP_VLLM,
    endpoints=["http://localhost:8000"],
    tags=["weights"],
)
```

#### SGLang

Server requirements:

```bash
python -m sglang.launch_server \
  --model-path <model> \
  --enable-memory-saver \
  --enable-weights-cpu-backup
```

For SGLang the effective suspend mode is fixed by the server's launch flags, not per call: with `--enable-weights-cpu-backup` weights are preserved (OFFLOAD behavior); without it they are discarded (DISCARD behavior) and inference produces incorrect results after resume unless the application pushes new weights.

```python
with SnapshotAgentClient("localhost:9001") as client:
    # Suspend: release GPU memory
    client.snapshot_and_wait(
        job_id="my-sglang-job",
        backend_config=snapshot_agent_pb2.BackendConfig(
            app_endpoint=snapshot_agent_pb2.AppEndpointConfig(
                app=snapshot_agent_pb2.APP_SGLANG,
                endpoints=["http://localhost:30000"],
            )
        ),
    )

    # Resume: restore GPU memory
    client.restore_and_wait(
        job_id="my-sglang-job",
        backend_config=snapshot_agent_pb2.BackendConfig(
            app_endpoint=snapshot_agent_pb2.AppEndpointConfig(
                app=snapshot_agent_pb2.APP_SGLANG,
                endpoints=["http://localhost:30000"],
            )
        ),
    )
```

### Composing Backends

Application-aware suspend and CUDA checkpoint are separate operations that compose. Suspend first, then checkpoint; restore in reverse order:

```python
# 1. Application-level suspend (frees ~96% VRAM)
client.snapshot_and_wait(
    job_id="app-job",
    backend_config=snapshot_agent_pb2.BackendConfig(
        app_endpoint=snapshot_agent_pb2.AppEndpointConfig(
            app=snapshot_agent_pb2.APP_VLLM,
            endpoints=["http://localhost:8000"],
        )
    ),
)

# 2. CUDA checkpoint (frees the remaining CUDA context)
client.snapshot_and_wait(
    job_id="cuda-job",
    backend_config=snapshot_agent_pb2.BackendConfig(
        cuda=snapshot_agent_pb2.CudaBackendConfig(
            explicit_target=snapshot_agent_pb2.ProcessTarget(pids=[1234])
        )
    ),
)

# Restore: 3. CUDA restore, then 4. application-level resume
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
For debugging, you can use `grpcurl` directly against the agent.

**If running in Standalone Mode (on localhost):**
```bash
grpcurl -plaintext \
  -import-path pkg/snapshot-agent/api/v1alpha1 \
  -proto pkg/snapshot-agent/api/v1alpha1/snapshot_agent.proto \
  -d '{
    "job_id": "test-job",
    "backend_config": {
      "cuda": {
        "explicit_target": {
          "pids": [1234]
        }
      }
    }
  }' \
  localhost:9001 \
  snapshot_agent.v1alpha1.SnapshotAgentService/Snapshot
```

**If running in Kubernetes Mode (using the Node IP):**
```bash
# Get node IP first
NODE_IP=$(kubectl get pod <agent-pod> -o jsonpath='{.status.hostIP}')

grpcurl -plaintext \
  -import-path pkg/snapshot-agent/api/v1alpha1 \
  -proto pkg/snapshot-agent/api/v1alpha1/snapshot_agent.proto \
  -d '{"job_id": "test-job"}' \
  $NODE_IP:9001 \
  snapshot_agent.v1alpha1.SnapshotAgentService/Snapshot
```

### Common Issues
- **Permission Denied:** Ensure the Snapshot Agent pod is running as `privileged: true`.
- **Connection Refused:** Verify the `AGENT_ENDPOINT` environment variable correctly points to `$(NODE_IP):9001`.
- **GPU Not Found:** Check that the `nvidia.driver.hostPath` in the agent's configuration matches your node's setup.
- **Garbage inference after resume (vLLM):** The workload was suspended with `SUSPEND_MODE_DISCARD`, which drops weights. Use the default `SUSPEND_MODE_OFFLOAD`, or have the application push new weights after resume.
- **Garbage inference after SGLang resume:** The SGLang server was started without `--enable-weights-cpu-backup`. Restart with this flag.
