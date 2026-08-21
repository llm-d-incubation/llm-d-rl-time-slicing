"""PhaseLocks: idempotent group-lock helper around the timeslice client.

Configuration comes from the environment:
  TIMESLICE_JOB_ID    unique job identifier (e.g. "job-a")
  TIMESLICE_ORCH_ADDR node-local timeslice-orchestrator gRPC address (e.g. "127.0.0.1:50051")
  TIMESLICE_GROUP     time-slice group id shared by the co-scheduled jobs (e.g. "trainers")

If any variable is missing the helper runs in NO-OP mode (with a one-time warning)
so the same image works without the time-slicing platform.
"""

import atexit
import os
import sys
import time
from typing import Iterable, Optional

ENV_JOB_ID = "TIMESLICE_JOB_ID"
ENV_ORCH_ADDR = "TIMESLICE_ORCH_ADDR"
ENV_GROUP = "TIMESLICE_GROUP"


def _log(msg: str) -> None:
    # print+flush: verl configures the root logger at WARNING, and these lines
    # must always be visible in job logs for timeline analysis.
    print(f"[timeslice] {msg}", flush=True)


def _import_client():
    """Resolve the orchestrator client class (renamed upstream at some point:
    OrchestratorClient -> TimeSliceOrchestratorClient, API identical)."""
    try:
        from timeslice import OrchestratorClient  # lazy: keep import-safe without client

        return OrchestratorClient
    except ImportError:
        from timeslice import TimeSliceOrchestratorClient

        return TimeSliceOrchestratorClient


class PhaseLocks:
    """Idempotent acquire/release of orchestrator group locks.

    ensure(groups)  - blocking-acquire every group not already held (idempotent).
    drop_all()      - release every held group (idempotent, safe to call anytime).
    close()         - drop_all + close the gRPC channel.

    `client_factory(target, job_id, group_id)` is injectable for grpc-free tests.
    """

    def __init__(
        self,
        job_id: Optional[str],
        orch_addr: Optional[str],
        group: Optional[str],
        client_factory=None,
    ):
        self.job_id = job_id
        self.orch_addr = orch_addr
        self.group = group
        self.enabled = bool(job_id and orch_addr and group)
        self._held: set = set()
        self._client = None
        self._closed = False

        if not self.enabled:
            _log(
                "WARNING: time-slicing lock disabled (missing one of "
                f"{ENV_JOB_ID}/{ENV_ORCH_ADDR}/{ENV_GROUP}); PhaseLocks is a no-op."
            )
            return

        if client_factory is None:
            client_cls = _import_client()

            def client_factory(target, job_id, group_id):
                return client_cls(target=target, job_id=job_id, group_id=group_id)

        self._client = client_factory(self.orch_addr, self.job_id, self.group)
        _log(f"job={self.job_id} connected orchestrator={self.orch_addr} group={self.group}")
        # Best-effort safety net: never exit holding the group lock.
        atexit.register(self._atexit_cleanup)

    @classmethod
    def from_env(cls, client_factory=None) -> "PhaseLocks":
        return cls(
            job_id=os.environ.get(ENV_JOB_ID),
            orch_addr=os.environ.get(ENV_ORCH_ADDR),
            group=os.environ.get(ENV_GROUP),
            client_factory=client_factory,
        )

    # ------------------------------------------------------------------ core
    @property
    def held(self) -> bool:
        return self.group in self._held

    def ensure(self, groups: Optional[Iterable[str]] = None) -> None:
        """Blocking-acquire each group in `groups` (default: the configured group).

        Already-held groups are skipped, so calling this every step is cheap.
        Blocks until the orchestrator grants the lock. Acquire errors propagate
        (a job must not run unlocked because the orchestrator is unreachable).
        """
        if not self.enabled:
            return
        for g in groups if groups is not None else (self.group,):
            if g in self._held:
                continue
            t0 = time.monotonic()
            result = self._client.acquire(group_id=g)  # blocks until granted
            waited_ms = getattr(result, "waited_ms", None)
            if waited_ms is None:
                waited_ms = int((time.monotonic() - t0) * 1000)
            self._held.add(g)
            _log(
                f"job={self.job_id} ACQUIRE group={g} waited={waited_ms}ms "
                f"context_restored={getattr(result, 'context_restored', '?')}"
            )

    def drop_all(self) -> None:
        """Release every held group. Idempotent; release errors are logged, not raised."""
        if not self.enabled:
            return
        for g in list(self._held):
            try:
                result = self._client.release(group_id=g)
                _log(
                    f"job={self.job_id} RELEASE group={g} "
                    f"pending_waiters={getattr(result, 'pending_waiters', '?')} "
                    f"snapshot_deferred={getattr(result, 'snapshot_deferred', '?')}"
                )
            except Exception as e:  # noqa: BLE001 - never let a release error kill the job
                _log(f"job={self.job_id} RELEASE group={g} FAILED: {e}")
            finally:
                self._held.discard(g)

    def close(self) -> None:
        """Release everything and close the gRPC channel."""
        if not self.enabled or self._closed:
            return
        self.drop_all()
        try:
            self._client.close()
        except Exception as e:  # noqa: BLE001
            _log(f"job={self.job_id} client close FAILED: {e}")
        self._closed = True

    # ------------------------------------------------------------- internals
    def _atexit_cleanup(self) -> None:
        if self._closed or not self._held:
            return
        _log(f"job={self.job_id} atexit: releasing held groups {sorted(self._held)}")
        try:
            self.close()
        except Exception as e:  # noqa: BLE001
            print(f"[timeslice] atexit cleanup failed: {e}", file=sys.stderr, flush=True)
