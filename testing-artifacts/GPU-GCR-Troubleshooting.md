# GPU-GCR Integration Troubleshooting Guide (GKE)

This document summarizes the issues encountered and resolved during the integration of the `GPU-GCR` (GPU Checkpoint/Restore) backend with the `snapshot-agent` on a Google Kubernetes Engine (GKE) cluster.

---

## 1. Missing `libcudart.so.12` in Sampler Pod

### Issue
The sampler pod failed to start with the following error:
```
python3: error while loading shared libraries: libcudart.so.12: cannot open shared object file: No such file or directory
```

### Cause
The preloaded library `vGPU-NVIDIA.so` was compiled in the Dockerfile builder stage using `nvidia/cuda:12.2.0-devel-ubuntu22.04` (CUDA 12). However, the runtime image (`vllm/vllm-openai:v0.22.0`) uses **CUDA 13** (`torch 2.11.0+cu130`). At startup, the dynamic linker could not resolve the CUDA 12 dependency.

### Solution
1.  Updated [Dockerfile.example](file:///usr/local/google/home/edwinhernandez/go/time-slicing-pr-repo/testing-artifacts/Dockerfile.example) to use `nvidia/cuda:13.0.0-devel-ubuntu22.04` as the builder stage to match the runtime CUDA version.
2.  Added `LD_LIBRARY_PATH=/usr/local/cuda/lib64` to the sampler container env in [deployment.yaml](file:///usr/local/google/home/edwinhernandez/go/time-slicing-pr-repo/testing-artifacts/deployment.yaml) to ensure the runtime can locate `libcudart.so.13`.

---

## 2. Missing `libgcc_s.so.1` in Snapshot Agent Pod

### Issue
The snapshot operation failed with:
```
/opt/bin/cr_client: error while loading shared libraries: libgcc_s.so.1: cannot open shared object file: No such file or directory
```

### Cause
The `snapshot-agent` was using `gcr.io/distroless/base` as its runtime image. This image is extremely minimal and lacks standard C++ support libraries required by the compiled `cr_client` binary.

### Solution
Updated [Dockerfile](file:///usr/local/google/home/edwinhernandez/go/time-slicing-pr-repo/llm-d-rl-time-slicing/docker/snapshot-agent/Dockerfile) to use **`gcr.io/distroless/cc`** as the runtime base image, which includes `libgcc_s.so.1` and `libstdc++.so.6` by default.

---

## 3. Failed Mount of `/mnt/huge-ckpt` on GKE COS Nodes

### Issue
The `snapshot-agent` pod was stuck in `Pending` with the following event:
```
Warning  FailedMount  ...  kubelet  MountVolume.SetUp failed for volume "huge-ckpt" : mkdir /mnt/huge-ckpt: read-only file system
```

### Cause
GKE COS nodes use a read-only root filesystem. The directory `/mnt` on the host is read-only, so the kubelet failed to create `/mnt/huge-ckpt` using `DirectoryOrCreate`.

### Solution
Updated the `hostPath` in both [daemonset.yaml](file:///usr/local/google/home/edwinhernandez/go/time-slicing-pr-repo/llm-d-rl-time-slicing/deploy/snapshot-agent/templates/daemonset.yaml) and [deployment.yaml](file:///usr/local/google/home/edwinhernandez/go/time-slicing-pr-repo/testing-artifacts/deployment.yaml) to **`/var/tmp/huge-ckpt`** (which is writable on GKE COS). The container-side mount path remains `/mnt/huge-ckpt`.

---

## 4. Signal Delivery Hang (PID Namespace Mismatch)

### Issue
The snapshot request hung indefinitely without any activity in the workload.

### Cause
`GPU-CR` coordinates the checkpoint via a `mmap`ed control file named `/mnt/huge-ckpt/control-<PID>`. 
*   The workload (running inside the container namespace) saw its PID as `192` and created `control-192`.
*   The agent (running in the host PID namespace) saw the workload as PID `119608` and waited on `control-119608`.

Since the filenames did not match, they could not establish communication.

### Solution
Enabled **`hostPID: true`** in [deployment.yaml](file:///usr/local/google/home/edwinhernandez/go/time-slicing-pr-repo/testing-artifacts/deployment.yaml) for the sampler pod. This forces the workload to run in the host PID namespace, aligning the PIDs.

---

## 5. Signal Ignored/Overwritten (`SIGUSR1` Conflict)

### Issue
The snapshot request still hung, and the workload logs showed no sign of receiving the checkpoint signal.

### Cause
`GPU-CR` defaults to using `SIGUSR1` for checkpointing. In a complex Python/vLLM environment, **`SIGUSR1` is often intercepted, blocked, or ignored** by the Python runtime or other dependencies (like PyTorch or Ray), preventing our C++ handler from executing.

### Solution
Patched the `GPU-CR` source code during the build phase in both the agent and the sampler to use **Real-Time Signals** which are not used by the Python runtime:
*   Changed `CR_CKPT_SIGNAL` to **`(SIGRTMAX - 8)`** (Signal 56).
*   Changed `CR_RESTORE_SIGNAL` to **`(SIGRTMAX - 7)`** (Signal 57).

Applied via `sed` in [daemonset.yaml](file:///usr/local/google/home/edwinhernandez/go/time-slicing-pr-repo/llm-d-rl-time-slicing/deploy/snapshot-agent/templates/daemonset.yaml) and [Dockerfile.example](file:///usr/local/google/home/edwinhernandez/go/time-slicing-pr-repo/testing-artifacts/Dockerfile.example).

---

## 6. Permission Denied on Control File

### Issue
The workload received the signal but immediately crashed with:
```
open(): Permission denied
```

### Cause
The control file `control-<PID>` was created by the agent (`cr_client` running as `root`) with `0755` permissions. When the workload (running as the non-root `sampler` user) tried to open it with write access (`O_RDWR`), it was blocked.

### Solution
Patched `GPU-CR`'s `src/comm/share_mem.cpp` during the build phase to explicitly set the control file permissions to **`0777`** using `fchmod` immediately after creation.

---

## 7. Permission Denied on `/mnt/huge-ckpt` Directory

### Issue
The workload failed with `open(): Permission denied` when trying to create `/mnt/huge-ckpt/control` (without PID) during initialization (`get_id()`).

### Cause
The `/mnt/huge-ckpt` directory (created by `DirectoryOrCreate` on the host) is owned by `root:root` with `0755` permissions. The non-root `sampler` user did not have permission to create new files (like `control` or the VRAM dump files) in this directory.

### Solution
Updated the Snapshot Agent's [main.go](file:///usr/local/google/home/edwinhernandez/go/time-slicing-pr-repo/llm-d-rl-time-slicing/cmd/snapshot-agent/main.go) to explicitly **`chmod 777 /mnt/huge-ckpt`** at startup. Since the agent runs as `root`, it successfully opens up write access for the `sampler` user.
