# Detailed Changes in the Slime Integration Fork (Multi-Cluster Architecture)

To enable cooperative time-slicing across independent KubeRay clusters (`sync` mode) using Kubernetes Dynamic Resource Allocation (DRA) overloading, several architectural changes and synchronization enhancements were implemented in our [`jessicaochen/slime (timeslice branch)`](https://github.com/jessicaochen/slime/tree/timeslice) fork. 

These changes are categorized by their implementation purpose below:

---

### 1. Time-Slicing Lock Coordination & Deadlock Prevention
* **Purpose:** Integrates the gRPC-based Accelerator Orchestrator client into Slime's main training loop to coordinate GPU residency and prevent cross-cluster circular wait deadlocks.
* **Key Files:** [`train.py`](https://github.com/jessicaochen/slime/blob/timeslice/train.py), [`slime/utils/arguments.py`](https://github.com/jessicaochen/slime/blob/timeslice/slime/utils/arguments.py).
* **Implementation Details:**
  * **CLI Flag Registration:** Added orchestrator configuration flags (`--enable-timeslice`, `--timeslice-orchestrator-addr`, `--timeslice-job-id`, `--timeslice-sampler-group`, `--timeslice-trainer-group`) to `arguments.py`.
  * **Parallel Grant Acquisition at Startup:** In `train.py`, placement group creation for Trainer actors and Rollout samplers is decoupled and executed concurrently using a 2-worker `ThreadPoolExecutor`.
  * **Strict Trainer-First Lock Ordering:** To guarantee deadlock freedom when multiple jobs initialize or sync weights simultaneously, lock acquisition strictly enforces a **Trainer lock first, then Sampler lock** hierarchy. The Trainer lock is acquired prior to requesting actor placement groups from Ray.
  * **Residency-Aligned Phase Toggling:** Aligns lock boundaries with GPU execution phases:
    * Holds the Sampler lock during rollout generation and yields it immediately after generation and KV cache offloading are complete.
    * Acquires the Trainer lock prior to policy backpropagation (`critic_model.async_train` and PPO steps) and yields it after offloading Trainer weights to CPU.
    * Re-acquires the Sampler lock while holding the Trainer lock during NCCL weight broadcasts (`update_weights`), yielding the Trainer lock once transfer finishes while retaining the Sampler lock for the next rollout epoch.
  * **Clean Teardown:** Added explicit lock release (`client.release()`) and socket teardown (`client.close()`) calls on loop completion to prevent lock leakage in the orchestrator.

---

### 2. Dynamic Placement Group Allocation & Custom Resource Routing
* **Purpose:** Allows independent Ray clusters to route actor tasks to specialized GKE GPU node pools (`trainer-gpu-pool` and `sampler-gpu-pool`) and allocate placement groups dynamically under lock grants.
* **Key Files:** [`slime/ray/placement_group.py`](https://github.com/jessicaochen/slime/blob/timeslice/slime/ray/placement_group.py).
* **Implementation Details:**
  * **Role-Decoupled Placement Group Creation:** Modified `create_placement_groups` to accept an explicit `role` parameter (`"actor"`, `"rollout"`, or `None`). This allows `train.py` to request Trainer placement groups independently from Rollout placement groups.
  * **Custom Resource Injections for DRA Routing:** Injects custom resource demands (`"trainers": 1` and `"samplers": 1`) into Ray placement group bundles. When combined with KubeRay container resource claims, this ensures Ray actors schedule onto pods backed by the correct DRA resource claims (`shared-trainers-gpu-claim` vs. `shared-samplers-gpu-claim`).
  * **Integer Bundles for DRA Co-location:** Allocates full integer bundles (`{"GPU": 1, "trainers": 1}`). Co-location across independent jobs is handled transparently at the Kubernetes scheduler level via DRA claim overloading.

---

### 3. Cooperative VRAM Offloading & Direct GPU Weight Sync
* **Purpose:** Minimizes memory swapping overhead over PCIe by coordinating PyTorch application-level offloading with gRPC lock transitions and NCCL broadcast synchronization.
* **Key Files:** [`train.py`](https://github.com/jessicaochen/slime/blob/timeslice/train.py), [`slime/ray/rollout.py`](https://github.com/jessicaochen/slime/blob/timeslice/slime/ray/rollout.py), [`slime/backends/megatron_utils/actor.py`](https://github.com/jessicaochen/slime/blob/timeslice/slime/backends/megatron_utils/actor.py).
* **Implementation Details:**
  * **Application-Level Swapping (`--offload-train` / `--offload-rollout`):** Coordinated with lock yielding so that SGLang discards transient KV caches and only swaps static model weights (~1GB) to CPU memory before yielding the Sampler lock. Similarly, Megatron Trainer offloads optimizer states and model weights before yielding the Trainer lock.
  * **Trainer Offload Wake-up Fix (Direct GPU Weight Sync):** Modified `update_weights` to wake up the Megatron Trainer model in VRAM before broadcasting updated weights to SGLang and immediately return it to sleep after transfer. Because distributed weight synchronization executes via direct **GPU-to-GPU NCCL broadcast**, both SGLang and Megatron must reside in active GPU VRAM (holding both orchestrator locks) during the transfer.
