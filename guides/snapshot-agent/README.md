# Snapshot Agent

The Snapshot Agent provides GPU checkpoint/restore primitives to enable efficient resource sharing for GPU-bound workloads. By allowing processes to save and reload their entire GPU state, it enables scenarios where multiple high-memory workloads can share the same physical GPU hardware.

It can be deployed in two modes:
1. **Standalone Mode (Primary / Default):** The agent runs directly on the host (or a VM). Workloads are targeted directly by specifying their Process IDs (PIDs) in the client requests.
2. **Kubernetes Mode (Optional Automation):** The agent runs as a DaemonSet. It automatically discovers target PIDs by querying the local Kubernetes API for pods matching specific job labels.

---

## 1. Running in Standalone Mode (Primary)

In standalone mode, you run the `snapshot-agent` binary directly on your host machine (e.g., a GCE VM). Workloads are targeted by specifying their Process IDs (PIDs) directly in the client request.

### Installing

Two ways to get the agent onto a GPU host:

**Build from source** (requires Go and the NVIDIA driver; x86_64 Linux):
```bash
git clone https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git
cd llm-d-rl-time-slicing
make standalone     # bin/snapshot-agent (CGO — links against NVML)
                    # + bin/cuda-checkpoint (pinned NVIDIA binary the CUDA
                    #   backend execs; found via PATH)
```

**Or run the published container image** (self-contained: agent + `cuda-checkpoint`):
```bash
docker run -d --name snapshot-agent \
  --privileged --pid=host --gpus all \
  -p 9001:9001 \
  ghcr.io/llm-d-incubation/llm-d-rl-time-slicing/snapshot-agent:latest
```

`--pid=host` is required: standalone mode targets host PIDs, so the agent must
share the host PID namespace to checkpoint them.

### Starting the Agent

By default, the agent starts in standalone mode on port `9001`. Run it as root
(or as the same user as the workloads) — `cuda-checkpoint` toggles the CUDA
state of other processes:
```bash
sudo env PATH="$PWD/bin:$PATH" ./bin/snapshot-agent

# Or explicitly set the port and mode (also settable via the AGENT_PORT and
# DEPLOYMENT_MODE environment variables):
sudo env PATH="$PWD/bin:$PATH" ./bin/snapshot-agent -deployment-mode=standalone -port=9001
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
from timeslice.snapshot_agent import snapshot_agent_pb2 as snapshot

# Connect to the local agent
with SnapshotAgentClient("localhost:9001") as client:
    # Define the backend config with target PIDs
    backend_config = snapshot.BackendConfig(
        cuda=snapshot.CudaBackendConfig(
            explicit_target=snapshot.ProcessTarget(pids=[1234])
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

The VRAM Freed and Resume Time figures are illustrative, measured with a small model (Qwen2.5-0.5B) on an H100; actual numbers depend on the model size, hardware, and engine version.

### CUDA Checkpoint

Saves and restores the entire CUDA context of a process. Works with any GPU workload regardless of framework.

```python
from timeslice.snapshot_agent import SnapshotAgentClient
from timeslice.snapshot_agent import snapshot_agent_pb2 as snapshot

cuda_config = snapshot.BackendConfig(
    cuda=snapshot.CudaBackendConfig(
        explicit_target=snapshot.ProcessTarget(pids=[1234])
    )
)

with SnapshotAgentClient("localhost:9001") as client:
    # Checkpoint
    result = client.snapshot_and_wait(job_id="my-job", backend_config=cuda_config)

    # Restore
    result = client.restore_and_wait(job_id="my-job", backend_config=cuda_config)
```

In Kubernetes mode, PIDs are discovered automatically — omit `explicit_target`:
```python
result = client.snapshot_and_wait(
    job_id="my-k8s-job",
    backend_config=snapshot.BackendConfig(cuda=snapshot.CudaBackendConfig()),
)
```

### Application-Aware (app_endpoint)

This backend implements application-aware snapshot/restore: suspend and resume are HTTP calls to an endpoint on the running application, and the application itself offloads or drops its GPU state in response (vLLM's sleep API, SGLang's memory-occupation API). The `app` field selects the application; `endpoints` targets the server(s).

**Suspend mode** states what happens to the workload's durable state while suspended:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `SUSPEND_MODE_OFFLOAD` | State preserved in host memory; Restore copies it back | Standard suspend/resume |
| `SUSPEND_MODE_DISCARD` | State dropped; the application re-provisions it after Restore | RL training — push new weights after resume |

When the mode is unspecified (`SUSPEND_MODE_UNSPECIFIED`), the application's default behavior applies: for vLLM that is OFFLOAD; for SGLang the launch flags decide either way (see below).

**Tags** select memory regions (`weights`, `kv_cache`, ...). If omitted, the application's full region set is used. On Snapshot, tags select what to suspend (where the application supports it); on Restore, what to bring back.

#### vLLM

Server requirements:

```bash
VLLM_SERVER_DEV_MODE=1 python -m vllm.entrypoints.openai.api_server \
  --model <model> \
  --enable-sleep-mode
```

```python
vllm_config = snapshot.BackendConfig(
    app_endpoint=snapshot.AppEndpointConfig(
        app=snapshot.APP_VLLM,
        endpoints=["http://localhost:8000"],
    )
)

with SnapshotAgentClient("localhost:9001") as client:
    # Suspend: offload weights to CPU, discard KV cache
    client.snapshot_and_wait(job_id="my-vllm-job", backend_config=vllm_config)

    # Resume: restore all
    client.restore_and_wait(job_id="my-vllm-job", backend_config=vllm_config)
```

Partial resume — bring back only specific regions:
```python
app_endpoint=snapshot.AppEndpointConfig(
    app=snapshot.APP_VLLM,
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
sglang_config = snapshot.BackendConfig(
    app_endpoint=snapshot.AppEndpointConfig(
        app=snapshot.APP_SGLANG,
        endpoints=["http://localhost:30000"],
    )
)

with SnapshotAgentClient("localhost:9001") as client:
    # Suspend: release GPU memory
    client.snapshot_and_wait(job_id="my-sglang-job", backend_config=sglang_config)

    # Resume: restore GPU memory
    client.restore_and_wait(job_id="my-sglang-job", backend_config=sglang_config)
```

### Composing Backends

Application-aware suspend and CUDA checkpoint are separate operations that compose. Suspend first, then checkpoint; restore in reverse order:

```python
app_config = snapshot.BackendConfig(
    app_endpoint=snapshot.AppEndpointConfig(
        app=snapshot.APP_VLLM,
        endpoints=["http://localhost:8000"],
    )
)
cuda_config = snapshot.BackendConfig(
    cuda=snapshot.CudaBackendConfig(
        explicit_target=snapshot.ProcessTarget(pids=[1234])
    )
)

# 1. Application-level suspend (frees most VRAM)
client.snapshot_and_wait(job_id="app-job", backend_config=app_config)

# 2. CUDA checkpoint (frees the remaining CUDA context)
client.snapshot_and_wait(job_id="cuda-job", backend_config=cuda_config)

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
- **Garbage inference after resume (vLLM):** The workload was suspended with `SUSPEND_MODE_DISCARD`, which drops weights. Suspend with `SUSPEND_MODE_OFFLOAD` (vLLM's default when the mode is unspecified), or have the application push new weights after resume.
- **Garbage inference after SGLang resume:** The SGLang server was started without `--enable-weights-cpu-backup`. Restart with this flag.
