from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List


class GroupLockState(str, Enum):
    UNSPECIFIED = "UNSPECIFIED"
    UNKNOWN = "UNKNOWN"
    IDLE = "IDLE"
    IDLE_YIELDED = "IDLE_YIELDED"
    LOCKED = "LOCKED"
    SWITCHING = "SWITCHING"


class AgentJobState(str, Enum):
    UNSPECIFIED = "UNSPECIFIED"
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    TRANSITIONING = "TRANSITIONING"
    SAVED = "SAVED"
    FAULTED = "FAULTED"


@dataclass(frozen=True)
class AcquireResult:
    success: bool
    waited_ms: int
    context_restored: bool


@dataclass(frozen=True)
class YieldResult:
    success: bool
    pending_waiters: int
    snapshot_deferred: bool


@dataclass(frozen=True)
class SnapshotAgentJobState:
    agent: str
    job_id: str
    job_state: AgentJobState


@dataclass(frozen=True)
class GroupStatus:
    group_id: str
    group_state: GroupLockState
    state_timestamp: datetime
    locking_job: str
    active_job: str
    waiter_queue_depth: int
    loaded_job: str


@dataclass(frozen=True)
class OrchestratorGroupStatus:
    group: GroupStatus
    agent_job_states: List[SnapshotAgentJobState]
