# Timeslice Python SDK

This is the Python library for Timeslice.

## Installation

```bash
pip install .
```

## Usage

### Snapshot Agent client

```python
from timeslice.snapshot_agent import SnapshotAgentClient

with SnapshotAgentClient(endpoint="localhost:9001") as client:
    # Trigger a snapshot and wait for it to complete
    result = client.snapshot_and_wait(job_id="my-job", group="default", backend="CUDA")
    print(f"Snapshot finished with status: {result.status}")
```

## Development

To generate gRPC stubs:

```bash
pip install grpcio-tools
python3 -m grpc_tools.protoc -I../../snapshot-agent/api/v1alpha1 --python_out=timeslice/snapshot_agent --grpc_python_out=timeslice/snapshot_agent ../../snapshot-agent/api/v1alpha1/snapshot_agent.proto
```

You will need to fix the imports in the generated files (e.g., `import snapshot_agent_pb2 as snapshot__agent__pb2` -> `from . import snapshot_agent_pb2 as snapshot__agent__pb2`).