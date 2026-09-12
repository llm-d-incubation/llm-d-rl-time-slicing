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
make standalone
```

**Or run the published container image:**
```bash
docker run -d --name snapshot-agent \
  --privileged --pid=host --gpus all \
  -p 9001:9001 \
  ghcr.io/llm-d-incubation/llm-d-rl-time-slicing/snapshot-agent:latest
```

### Starting the Agent

By default, the agent starts in standalone mode on port `9001`:
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
| Application-Aware | `app_channel` | Suspend/resume pushed over a channel the workload registered (Python-API workloads, no HTTP server) | ~96% | ~50-100ms |
| Direct Memory | `direct_memory` | Full-process park/resume via GPU-CR `cr_client`: the workload's preloader dumps device state to node shared memory and the process stays alive | ~100% | ~0.5-2s |

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

### Application-Aware (app_channel)

For workloads that embed their engine in-process through a Python API — no HTTP
server for the agent to call (e.g. an RL sampler running vLLM via
`AsyncLLMEngine`). The connection is inverted: the workload registers with the
node-local agent once at startup, and the agent pushes suspend/resume commands
over that stream. Callers address the workload by `job_id` alone — no
endpoints, and no knowledge of which application is running.

**Workload side** — register once at startup with `register_workload`:

```python
from timeslice.snapshot_agent import register_workload

engine = AsyncLLMEngine.from_engine_args(args)   # enable_sleep_mode=True

handle = register_workload(
    "127.0.0.1:9001",       # the agent on this node (registration is node-scoped)
    job_id="my-sampler",    # must match the job_id used in Snapshot/Restore
                            # requests (in k8s: the timeslice.io/job-id pod label)
    group="samplers",
    workload=engine,        # vLLM engines are recognized by type
)
# ... run; the library services commands in the background ...
handle.close()              # on clean shutdown
```

The library owns the stream: a background thread, command dispatch and
acknowledgements, and reconnect with backoff (re-registering after agent
restarts). Recognized engines (vLLM `LLM`/`AsyncLLMEngine`/`AsyncLLM`) need
nothing else.

Workloads with no publicly known C/R API (e.g. hand-rolled FSDP offload) keep
their own mechanics and hand them to the library, either as an object with
`snapshot(mode, tags)`/`restore(tags)` methods:

```python
class TrainerWorkload:
    supported_modes = ["offload"]   # trainers can't reconstruct dropped state

    def snapshot(self, mode, tags):
        offload_model_and_optimizer_to_host()

    def restore(self, tags):
        reload_from_host()

register_workload("127.0.0.1:9001", job_id="my-trainer", group="trainers",
                  workload=TrainerWorkload())
```

or as plain callbacks:

```python
register_workload("127.0.0.1:9001", job_id="my-trainer", group="trainers",
                  on_snapshot=lambda mode, tags: trainer.offload(),
                  on_restore=lambda tags: trainer.reload(),
                  supported_modes=["offload"])
```

At registration the workload advertises its capabilities (`supported_modes`,
`default_mode`). The agent resolves each request as: explicit request mode →
registered default → `SUSPEND_MODE_OFFLOAD`, and rejects unsupported modes
before any command is sent (e.g. DISCARD against a trainer that only supports
OFFLOAD fails the operation immediately).

**Caller side** — the usual `Snapshot`/`Restore` with an `app_channel` config.
An empty config means "suspend however the workload declared at registration":

```python
channel_config = snapshot.BackendConfig(app_channel=snapshot.AppChannelConfig())

result = client.snapshot_and_wait(job_id="my-sampler", backend_config=channel_config)

result = client.restore_and_wait(job_id="my-sampler", backend_config=channel_config)
```

A request for a job with no registered channel fails fast with
`no workload channel registered for job "..."`. If the workload's suspend
raises, the operation fails with the workload's error text.

### Direct Memory (direct_memory)

Full-process GPU park/resume driven by GPU-CR's `cr_client`. Like the CUDA
Checkpoint backend it saves and restores the process's entire device state,
and it uses the same `cuda-checkpoint` toggle for the CUDA context — the
difference is who moves the bytes. Plain `cuda-checkpoint` copies all
device memory through the driver into the process's own pageable host RAM.
Here, the workload's GPU-CR preloader first drains the device memory it
tracked into hugepage-backed files on the node through a pinned DMA
pipeline and frees it; the context toggle then freezes what is by that
point a nearly empty context. The bulk bytes never cross the slow pageable
path — which is why park and resume are several times faster for
large-VRAM workloads — and the parked state lives in node-local files that
survive independently of the process's memory.

Requirements:

* The target workload runs under the GPU-CR vGPU preloader
  (`LD_PRELOAD=vGPU-NVIDIA.so`), built from the same `third_party/gpu-cr`
  tree as the `cr_client` shipped in the agent image — the two share
  compiled-in constants and are version-locked.
* Agent and workload share the GPU-CR checkpoint/control directory (the
  agent's `EXPORT_FILE_PATH`; in Kubernetes, the `directMemory` block in
  the Helm chart under `deploy/snapshot-agent` renders the shared mount
  plus, by default, init containers that mount hugetlbfs for the dump
  store, mount the control-file tmpfs nested at `<store>/ctl` (discovered
  by both sides with no configuration; it keeps the agent free of hugepage
  requests), and provision the node's 2Mi hugepage pool at deploy time —
  so no special node image or pre-sized nodepool is needed).
* Node hugepage capacity for whole-VRAM dumps, sized to the GPU-CR build's
  dump extent (the chart's bootstrap provisions this by default; size it
  via `directMemory.hugetlbfs.bootstrap.pages2Mi`).

The backend is experimental and gated off by default: requests fail with
`FAILED_PRECONDITION` unless the agent runs with
`--feature-gates=DirectMemoryBackend=true` (or the `FEATURE_GATES` env
var). The Helm chart sets the gate automatically when
`directMemory.enabled=true`.

```python
from timeslice.snapshot_agent import SnapshotAgentClient, direct_memory_config

with SnapshotAgentClient("localhost:9001") as client:
    # Park: device state is dumped node-locally, VRAM is freed,
    # the process stays alive.
    client.snapshot_and_wait(
        job_id="my-job",
        backend_config=direct_memory_config(pids=[1234]),
    )
    # Resume: parked state is mapped back and execution continues.
    client.restore_and_wait(
        job_id="my-job",
        backend_config=direct_memory_config(pids=[1234]),
    )
```

In Kubernetes mode PIDs are discovered from the `timeslice.io/job-id` pod
label — omit `pids` (i.e. `direct_memory_config()`).

#### Setting up a workload pod

Bake the preloader into your workload image, copied from the artifact image
that `third_party/gpu-cr/Dockerfile.build` produces (building both it and
the agent's `cr_client` from the same tree is what keeps them compatible):

```dockerfile
FROM <registry>/gpucr-so:<tag> AS gpucr
FROM vllm/vllm-openai:v0.22.0
COPY --from=gpucr /vGPU-NVIDIA.so /usr/local/lib/vGPU-NVIDIA.so
RUN chmod 755 /usr/local/lib/vGPU-NVIDIA.so
```

Then the pod needs the job-id label, the shared checkpoint-dir mount, a
hugepage allowance for its dumps, and a handful of env vars:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-sampler
  labels:
    timeslice.io/job-id: "my-job"   # how the agent finds this pod's PIDs
spec:
  # The preloader names its control file after the process's own PID; the
  # agent signals the HOST PID it discovered. These only match in the host
  # PID namespace.
  hostPID: true
  containers:
  - name: workload
    image: <your image with vGPU-NVIDIA.so baked in>
    securityContext:
      runAsUser: 0
    env:
    # Required: inject the preloader and point it at the dump-store ROOT
    # (the chart's directMemory.ctlDir value — /mnt/huge-ckpt by default).
    # GPU-CR discovers the control directory at <root>/ctl on its own;
    # never set EXPORT_FILE_PATH to the nested ctl path itself.
    - name: LD_PRELOAD
      value: "/usr/local/lib/vGPU-NVIDIA.so"
    - name: GPU_VENDOR
      value: "NVIDIA"
    - name: EXPORT_FILE_PATH
      value: "/mnt/huge-ckpt"
    # Dump-buffer size in GiB. Unset = the build default (25). Size it to
    # the VRAM working set you actually park.
    - name: GPU_CR_SHM_GB
      value: "8"
    # Part of the configuration all published direct_memory results were
    # measured with (vLLM in eager mode, caching allocator off); running
    # without them is untested.
    - name: PYTORCH_NO_CUDA_MEMORY_CACHING
      value: "1"
    - name: CUDA_LAUNCH_BLOCKING
      value: "1"
    resources:
      requests:
        nvidia.com/gpu: "1"
        memory: "6Gi"
        # The dump buffer plus headroom: ~12Gi for GPU_CR_SHM_GB=8,
        # ~28Gi for the unset (25 GiB) default. Kubernetes requires a
        # memory request alongside hugepages.
        hugepages-2Mi: "12Gi"
      limits:
        nvidia.com/gpu: "1"
        memory: "6Gi"
        hugepages-2Mi: "12Gi"
    volumeMounts:
    - name: huge-ckpt
      mountPath: /mnt/huge-ckpt
      # Pick up the hugetlbfs + control-tmpfs mounts the agent's init
      # containers made on the host, even if this pod started first.
      mountPropagation: HostToContainer
  volumes:
  - name: huge-ckpt
    hostPath:
      path: /var/tmp/huge-ckpt   # the chart's directMemory.hostCtlPath
      type: DirectoryOrCreate
```

No control-plane configuration is needed: the preloader discovers the
control-file tmpfs at `<store>/ctl` through this same mount.

Each `cr_client` invocation runs under a per-operation deadline
(`DIRECT_MEMORY_OP_TIMEOUT_SEC`, default 120 s): a workload that dies
mid-operation fails that operation instead of wedging the job in
TRANSITIONING. Health: `grpc.health.v1.Health/Check` with
`service: "direct-memory"` reports whether `cr_client` is available.

### Composing Backends

Application-aware suspend (either transport) and CUDA checkpoint are separate operations that compose. Suspend first, then checkpoint; restore in reverse order:

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
- **`cr_client not found at /usr/local/bin/cr_client` (direct_memory):** The agent image was built without the GPU-CR builder stage. Deploy the standard snapshot-agent image; there is no path override.
- **direct_memory operation times out:** `cr_client` talks to the workload's preloader over a shared-memory control channel; a timeout usually means the workload is not running under `LD_PRELOAD=vGPU-NVIDIA.so`, the preloader and `cr_client` were built from different GPU-CR trees, or the target process died mid-operation. The deadline is `DIRECT_MEMORY_OP_TIMEOUT_SEC` (default 120 s).
