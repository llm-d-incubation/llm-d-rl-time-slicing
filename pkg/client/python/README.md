# Timeslice Python SDK

This is the Python library for Timeslice.

## Installation

```bash
pip install .
```

For development (including gRPC tools):
```bash
pip install .[dev]
```

## Clients

This SDK provides two clients:

1.  **Snapshot Agent Client**: Used to trigger manual snapshots and restores on local nodes. See usage below.
2.  **Accelerator Orchestrator Client**: Used to coordinate shared GPU access between jobs in a time-slice group. Detailed documentation and examples can be found in the [Orchestrator README](timeslice/orchestrator/README.md).

---

## Snapshot Agent Client Usage

```python
from timeslice.snapshot_agent import SnapshotAgentClient

with SnapshotAgentClient(endpoint="localhost:9001") as client:
    # Trigger a snapshot and wait for it to complete
    from timeslice.snapshot_agent import snapshot_agent_pb2
    result = client.snapshot_and_wait(
        job_id="my-job", 
        group="default", 
        backend_config=snapshot_agent_pb2.BackendConfig(
            cuda=snapshot_agent_pb2.CudaBackendConfig()
        )
    )
    print(f"Snapshot finished with status: {result.status}")
```

---

## Development

To generate gRPC stubs for the Snapshot Agent:

```bash
# Run from pkg/client/python directory
python3 -m grpc_tools.protoc \
    -Itimeslice/snapshot_agent=../../snapshot-agent/api/v1alpha1 \
    --python_out=. \
    --grpc_python_out=. \
    timeslice/snapshot_agent/snapshot_agent.proto
```

To generate stubs for the Accelerator Orchestrator, see the [Orchestrator README](timeslice/orchestrator/README.md#development).