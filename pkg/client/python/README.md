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
2.  **TimeSlice Orchestrator Client**: Used to coordinate shared GPU access between jobs in a time-slice group. Detailed documentation and examples can be found in the [Orchestrator README](timeslice/orchestrator/README.md).

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

### Memory-Regions Backend

To selectively snapshot/restore explicit device-memory ranges of a running
process (GPU-CR memory-regions backend), build the config with
`memory_regions_config`. Regions can be `MemoryRegion` dataclasses,
`(pid, address, size_bytes)` tuples, or `"pid:0xADDR:size"` strings;
hex and decimal addresses are both accepted.

`snapshot_name` names the agent-side snapshot slot (it defaults to the
`job_id` when empty). Distinct names let multiple snapshots of the same
process coexist and be swapped on demand. Use `snapshot_name` — not the
request's `group` — for slot naming; `group` is orchestrator-owned.

```python
from timeslice.snapshot_agent import (
    MemoryRegion,
    SnapshotAgentClient,
    memory_regions_config,
)

with SnapshotAgentClient(endpoint="localhost:9001") as client:
    regions = [MemoryRegion(pid=12345, address=0x7F0000000000, size_bytes=1 << 30)]

    # Save the regions into slot "slot-a"...
    result = client.snapshot_and_wait(
        job_id="my-job",
        backend_config=memory_regions_config(regions, snapshot_name="slot-a"),
    )
    print(f"Snapshot finished with status: {result.status}")

    # ...and load them back later (possibly alternating with other slots).
    result = client.restore_and_wait(
        job_id="my-job",
        backend_config=memory_regions_config(regions, snapshot_name="slot-a"),
    )
    print(f"Restore finished with status: {result.status}")
```

The three region forms are interchangeable (and can be mixed in one call):

```python
# MemoryRegion dataclass
config = memory_regions_config(
    [MemoryRegion(pid=12345, address=0x7F0000000000, size_bytes=1 << 30)]
)

# (pid, address, size_bytes) tuple
config = memory_regions_config([(12345, 0x7F0000000000, 1 << 30)])

# "pid:address:size" string; address and size accept hex or decimal
config = memory_regions_config(["12345:0x7F0000000000:1073741824"])
```

---

## Development

To generate gRPC stubs for the Snapshot Agent, first install the pinned codegen
toolchain (generated code sets the package's minimum supported grpcio/protobuf —
regenerating with a newer toolchain silently raises the floors, see `[dev]` in
`pyproject.toml`):

```bash
pip install "grpcio-tools==1.81.0"
```

Then:

```bash
# Run from pkg/client/python directory
python3 -m grpc_tools.protoc \
    -Itimeslice/snapshot_agent=../../snapshot-agent/api/v1alpha1 \
    --python_out=. \
    --grpc_python_out=. \
    timeslice/snapshot_agent/snapshot_agent.proto
```

To generate stubs for the TimeSlice Orchestrator, see the [Orchestrator README](timeslice/orchestrator/README.md#development).