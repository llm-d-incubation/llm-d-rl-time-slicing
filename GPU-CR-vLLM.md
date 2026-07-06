# GPU-CR with vLLM on Kubernetes: Integration & Troubleshooting Guide

This document summarizes the findings, architectural considerations, and bug fixes required to run `GPU-CR` (CUDA Checkpoint & Restore) with `vLLM` workloads inside Kubernetes containers (GKE).

---

## 1. Container & Deployment Setup

To execute `cr_client` from inside a workload container:

1. **Shared Binary Volume (`/opt/bin`)**:
   - Create an `emptyDir` volume (`bin-dir`) mounted at `/opt/bin` on both init containers and the main container.
   - Set `PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/bin` in the container environment.

2. **Init Containers**:
   - **`install-cuda-checkpoint`**: Fetches `cuda-checkpoint` from NVIDIA's repository and sets `/mnt/huge-ckpt` directory permissions to `0777`.
   - **`install-gpu-cr`**: Clones `https://github.com/Edwinhr716/GPU-CR.git` (branch `vllm-k8s-fix`), builds `cr_client`, `multi_cr_client`, and `vGPU-NVIDIA.so`, and places `cr_client` into `/opt/bin`.

3. **Permissions for Hugepages Shared Volume (`/mnt/huge-ckpt`)**:
   - `share_mem.cpp` creates IPC control files (`control-<PID>`) and memory dumps under `/mnt/huge-ckpt/`.
   - `/mnt/huge-ckpt` must have `0777` permissions (and `fchmod(fd_control, 0777)` in `share_mem.cpp`) so non-root container users (e.g. UID 1000) can open/write IPC control files without encountering `open(): Bad file descriptor` (Errno 9).

---

## 2. Process Identification in vLLM Workloads

When running `vllm serve`:
- The entry process is a Python server frontend.
- `vllm` spawns engine worker processes (`VLLM::EngineCore`) via `multiprocessing.spawn`.
- **Target PID Rule**: `cr_client` must be called with the PID of the **`VLLM::EngineCore` worker process**, not the parent `vllm serve` process. The worker process is the actual CUDA context owner that loads `vGPU-NVIDIA.so`.

---

## 3. Signal Remapping (`CR_CKPT_SIGNAL` & `CR_RESTORE_SIGNAL`)

- **Why `SIGUSR1` / `SIGUSR2` Fail**:
  - Python runtime, PyTorch, and Ray register their own signal handlers for `SIGUSR1` and `SIGUSR2` during process startup.
  - Standard `SIGUSR1`/`SIGUSR2` signals are swallowed or overridden by Python, preventing `vGPU-NVIDIA.so`'s C++ handler from receiving them.
- **Real-Time Signal Remapping**:
  - Remap `CR_CKPT_SIGNAL` to `(SIGRTMAX - 8)` (Signal 56).
  - Remap `CR_RESTORE_SIGNAL` to `(SIGRTMAX - 7)` (Signal 57).
- **Necessity Inside Same Container**:
  - Overwriting signals remains strictly necessary even inside the same container because signal handling is process-level (`VLLM::EngineCore`'s signal handler table), not container-level.

---

## 4. `cr_client -r` Crash & Restore Order Bug

### Root Cause
In the upstream `GPU-CR` repository, `cr_client.cpp` executed restore in the following order:
1. `cuda-checkpoint --toggle --pid <PID>` (un-froze GPU execution).
2. `comm->send_msg(RESTORE_MSG)` & `kill(pid, CR_RESTORE_SIGNAL)` (remapped physical VRAM).

Unpausing GPU execution before remapping physical VRAM caused active GPU threads to access unmapped VRAM virtual addresses (`releasePhysicalMemory`), triggering an immediate CUDA illegal memory access / segfault crash.

### The Fix (`vllm-k8s-fix` Branch)
The restore sequence in `coordinator/cr_client.cpp` was updated to:
1. **Send `CR_RESTORE_SIGNAL` first**: `vGPU`'s signal handler executes `remapPhysicalMemory` to restore physical VRAM page mappings.
2. **Execute `cuda-checkpoint --toggle` second**: Safely unpauses GPU kernel execution once physical memory pages are restored.

---

## 5. Verification Command Reference

```bash
# 1. Identify VLLM::EngineCore PID inside the container
kubectl exec <pod-name> -c sampler -- ps aux | grep multiprocessing

# 2. Checkpoint process
kubectl exec <pod-name> -c sampler -- cr_client -c -p <ENGINE_PID>

# 3. Restore process
kubectl exec <pod-name> -c sampler -- cr_client -r -p <ENGINE_PID>
```
