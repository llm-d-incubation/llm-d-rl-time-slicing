from dataclasses import dataclass
from typing import List, Optional


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
