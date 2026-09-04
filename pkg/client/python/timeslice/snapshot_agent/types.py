from dataclasses import dataclass
from typing import List, Optional

# Wire-format bounds: MemoryRegion.pid is an int32, address and size_bytes
# are uint64 (see snapshot_agent.proto). Checked here so callers get a
# clear error at construction rather than at protobuf assignment.
_INT32_MAX = 2**31 - 1
_UINT64_MAX = 2**64 - 1


@dataclass(frozen=True)
class MemoryRegion:
    """One device-memory range owned by a process.

    Addresses are plain integers; helpers accept hex strings ("0x7f...")
    and convert. On the wire the address travels as a uint64 (grpcurl JSON
    renders it as a decimal string).
    """

    pid: int
    address: int
    size_bytes: int

    def __post_init__(self):
        for name, value in (
            ("pid", self.pid),
            ("address", self.address),
            ("size_bytes", self.size_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"memory region {name} must be an int, got {type(value).__name__}"
                )
        if not 0 < self.pid <= _INT32_MAX:
            raise ValueError(
                f"memory region pid must be positive and fit in int32, got {self.pid}"
            )
        if not 0 <= self.address <= _UINT64_MAX:
            raise ValueError(
                "memory region address must be non-negative and fit in uint64, "
                f"got {self.address}"
            )
        if not 0 < self.size_bytes <= _UINT64_MAX:
            raise ValueError(
                "memory region size_bytes must be positive and fit in uint64, "
                f"got {self.size_bytes}"
            )

    @classmethod
    def from_spec(cls, spec: str) -> "MemoryRegion":
        """Parses a 'pid:0xADDR:size' spec string.

        Address and size accept hex ("0x...") or decimal literals.
        """
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"invalid memory region spec {spec!r}, expected 'pid:address:size'"
            )
        try:
            pid = int(parts[0], 0)
            address = int(parts[1], 0)
            size_bytes = int(parts[2], 0)
        except ValueError as e:
            raise ValueError(f"invalid memory region spec {spec!r}: {e}") from e
        return cls(pid=pid, address=address, size_bytes=size_bytes)


@dataclass(frozen=True)
class SnapshotResponse:
    """Response message for Snapshot RPC."""

    operation_id: str


@dataclass(frozen=True)
class RestoreResponse:
    """Response message for Restore RPC."""

    operation_id: str


@dataclass(frozen=True)
class HealthResponse:
    """Response message for Health Check RPC."""

    status: str


@dataclass(frozen=True)
class GetOperationResponse:
    """Response message for GetOperation RPC."""

    status: str
    elapsed_ms: int
    storage_bytes: Optional[int] = None
    snapshot_device_bytes: Optional[int] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class JobStatus:
    """Status information for a specific job."""

    job_id: str
    state: str


@dataclass(frozen=True)
class AcceleratorStatus:
    """Status information for an accelerator."""

    id: str
    memory_used_bytes: int
    memory_total_bytes: int


@dataclass(frozen=True)
class StatusResponse:
    """Response message for Status RPC."""

    job_statuses: List[JobStatus]
    accelerator_statuses: List[AcceleratorStatus]
