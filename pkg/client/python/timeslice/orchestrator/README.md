# Accelerator Orchestrator Python Client

The `OrchestratorClient` coordinates access to shared accelerators (GPUs/TPUs) among multiple jobs in a time-slice group. It ensures that only one job has exclusive access to the accelerator at any given time, driving the snapshot/restore cycle automatically behind the scenes.

## Usage

### Explicit Acquire & Release

You can explicitly control the lock using `acquire()` and `release()`. It is highly recommended to use a `try...finally` block to ensure the lock is always released.

```python
from timeslice import OrchestratorClient
from timeslice.orchestrator import OrchestratorError

# Initialize client for a specific job in a shared GPU group
client = OrchestratorClient(
    target="orchestrator-service:50051",
    job_id="my-job-123",
    group_id="shared-gpu-group"
)

try:
    print("Waiting to acquire GPU access...")
    # Blocks until access is granted
    result = client.acquire(timeout_sec=60.0)
    print(f"Acquired! Waited {result.waited_ms}ms.")

    # --- Perform GPU intensive work here ---
    
finally:
    print("Releasing GPU access...")
    client.release()
    client.close()
```

### Context Manager (`lock`)

A cleaner and safer way to manage the lock lifecycle is using the `lock()` context manager, which automatically handles acquisition and release.

```python
from timeslice import OrchestratorClient

client = OrchestratorClient("orchestrator-service:50051", "my-job", "shared-gpu")

# Automatically acquires on enter, releases on exit (even if exceptions occur)
with client.lock(timeout_sec=30.0):
    print("GPU access secured!")
    # Run GPU kernels...
```

### Decorator Usage

The `lock()` context manager also supports decorator usage out of the box. The lock is acquired when the decorated function is called and released when it returns.

```python
from timeslice import OrchestratorClient

client = OrchestratorClient("orchestrator-service:50051", "my-job", "shared-gpu")

@client.lock(timeout_sec=10.0)
def run_gpu_kernel():
    print("Running GPU kernel under orchestrator lock...")
    # GPU work...

# Call the function; lock is managed automatically
run_gpu_kernel()
```

### Optional Initialization & Dynamic Overrides

The `job_id` and `group_id` are optional on construction. If omitted, they **must** be provided dynamically during the method calls. You can also override the constructor-configured values on a per-call basis.

```python
from timeslice import OrchestratorClient

# Initialize without job_id and group_id
client = OrchestratorClient(target="orchestrator-service:50051")

# Pass them dynamically during acquire/release
client.acquire(job_id="job-A", group_id="group-A")
client.release(job_id="job-A", group_id="group-A")

# You can also use the lock context manager with overrides
with client.lock(job_id="job-B", group_id="group-B"):
    # Runs with job-B on group-B
    pass
```

---

## Comprehensive Example: Reinforcement Learning (RL) Job (Decorator Pattern)

In a typical Reinforcement Learning setup, a single training job might need to coordinate access across two different GPU resource pools: one dedicated to **Sampling** (generating experience rollouts) and another dedicated to **Training** (updating model weights).

Using `OrchestratorClient` as a **decorator**, we can cleanly bind GPU lock management directly to the execution functions. This keeps the orchestration loop in `main()` extremely clean and readable.

```python
import time
from timeslice import OrchestratorClient

# Configuration
ORCHESTRATOR_TARGET = "orchestrator-service:50051"

# A single Job ID representing this specific RL run
RL_JOB_ID = "rl-run-123"

# Two different time-slice groups representing different GPU resource pools
SAMPLING_GROUP = "sampling-gpu-pool"
TRAINING_GROUP = "training-gpu-pool"

# Initialize clients at the module level so they can be used as decorators
sampler_client = OrchestratorClient(ORCHESTRATOR_TARGET, job_id=RL_JOB_ID, group_id=SAMPLING_GROUP)
trainer_client = OrchestratorClient(ORCHESTRATOR_TARGET, job_id=RL_JOB_ID, group_id=TRAINING_GROUP)


# Decorate functions to automatically manage GPU locks when they are invoked

@sampler_client.lock()
def deploy_sampler_pods():
    print(f"[Deployer] Launching sampler pods in group '{SAMPLING_GROUP}'...")
    time.sleep(1)

@trainer_client.lock()
def deploy_trainer_pods():
    print(f"[Deployer] Launching trainer pods in group '{TRAINING_GROUP}'...")
    time.sleep(1)

@sampler_client.lock()
def run_sampling_phase():
    print(f"[Samplers] Generating rollouts on GPUs in group '{SAMPLING_GROUP}'...")
    time.sleep(2)  # Simulating GPU load

@trainer_client.lock()
def run_training_phase():
    print(f"[Trainer] Updating policy weights on GPUs in group '{TRAINING_GROUP}'...")
    time.sleep(3)  # Simulating GPU load


def main():
    # 1. Deploy Sampler Pods
    # Calling this function automatically acquires the sampling group lock
    print("\n=== Phase 1: Deploying Samplers ===")
    deploy_sampler_pods()
    print("[System] Samplers deployed and initialized.")

    # 2. Deploy Trainer Pods
    # Calling this function automatically acquires the training group lock
    print("\n=== Phase 2: Deploying Trainers ===")
    deploy_trainer_pods()
    print("[System] Trainers deployed and initialized.")

    # 3. RL Orchestration Loop
    # Alternate GPU access between the sampling group and the training group
    print("\n=== Phase 3: Starting RL Training Loop ===")
    try:
        for iteration in range(1, 4):
            print(f"\n--- Iteration {iteration} ---")
            
            # --- Step A: Sampling ---
            # Calling this automatically locks the sampling group, runs, and releases
            print(f"[Orchestrator] Triggering Sampling on '{SAMPLING_GROUP}'...")
            run_sampling_phase()
            print(f"[Orchestrator] Sampling completed.")

            # --- Step B: Training ---
            # Calling this automatically locks the training group, runs, and releases
            print(f"[Orchestrator] Triggering Training on '{TRAINING_GROUP}'...")
            run_training_phase()
            print(f"[Orchestrator] Training completed.")
            
    finally:
        # Clean up client channels
        sampler_client.close()
        trainer_client.close()
        print("\n[System] RL Job completed and clients closed.")

if __name__ == "__main__":
    main()
```

---

## Development

To regenerate gRPC stubs for the Accelerator Orchestrator:

```bash
# Run from pkg/client/python directory
python3 -m grpc_tools.protoc \
    -I../../accelerator-orchestrator/api/v1alpha1 \
    --python_out=timeslice/orchestrator/_generated \
    --grpc_python_out=timeslice/orchestrator/_generated \
    ../../accelerator-orchestrator/api/v1alpha1/accelerator_orchestrator.proto
```
