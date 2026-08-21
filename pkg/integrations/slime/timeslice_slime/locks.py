"""RoleLocks: idempotent per-role lock helper for disaggregated Slime topologies.

Configuration comes from the environment:
  TIMESLICE_JOB_ID          unique job identifier (e.g. "job-a")
  TIMESLICE_ORCH_ADDR       orchestrator gRPC address (e.g. "localhost:50051")
  TIMESLICE_TRAINER_GROUP   trainer pool group id (e.g. "trainers")
  TIMESLICE_SAMPLER_GROUP   sampler pool group id (e.g. "samplers")

If any variable is missing the helper runs in NO-OP mode (with a one-time warning)
so the same image works without the time-slicing platform.

Global lock order (deadlock freedom): TRAINER before SAMPLER.
  - A job may hold SAMPLER alone, but must never REQUEST TRAINER while
    holding SAMPLER.
  - Dual-lock spans (weight sync, init) acquire SAMPLER while already
    holding TRAINER.
With only these two wait shapes a wait-for cycle is impossible.
"""

import atexit
import os
import sys
import time


def _log(msg: str) -> None:
    print(f"[timeslice] {msg}", flush=True)


ENV_JOB_ID = "TIMESLICE_JOB_ID"
ENV_ORCH_ADDR = "TIMESLICE_ORCH_ADDR"
ENV_TRAINER_GROUP = "TIMESLICE_TRAINER_GROUP"
ENV_SAMPLER_GROUP = "TIMESLICE_SAMPLER_GROUP"

TRAINER = "trainer"
SAMPLER = "sampler"


class RoleLocks:
    """Idempotent per-role orchestrator locks with enforced trainer-first order.

    One OrchestratorClient per (job_id, group_id), as required by the
    orchestrator client contract.  Roles map to groups:
      TRAINER -> trainer_group (the trainers pool)
      SAMPLER -> sampler_group (the samplers pool)

    `client_factory(orch_addr, job_id, group_id)` is injectable for tests.
    """

    def __init__(
        self,
        job_id: str | None,
        orch_addr: str | None,
        trainer_group: str | None,
        sampler_group: str | None,
        client_factory=None,
    ):
        self.job_id = job_id
        self.orch_addr = orch_addr
        self.groups = {TRAINER: trainer_group, SAMPLER: sampler_group}
        self.enabled = bool(job_id and orch_addr and trainer_group and sampler_group)
        self._held: set = set()
        self._clients: dict = {}
        self._closed = False

        if not self.enabled:
            _log(
                "WARNING: disagg time-slicing disabled (missing one of "
                f"{ENV_JOB_ID}/{ENV_ORCH_ADDR}/{ENV_TRAINER_GROUP}/{ENV_SAMPLER_GROUP}); "
                "RoleLocks is a no-op."
            )
            return
        if trainer_group == sampler_group:
            raise ValueError(
                f"trainer_group and sampler_group must differ (both {trainer_group!r}): "
                "the disaggregated topology places the pools on distinct groups/nodes."
            )

        if client_factory is None:
            from timeslice import TimeSliceOrchestratorClient

            def client_factory(target, job_id, group_id):
                return TimeSliceOrchestratorClient(target=target, job_id=job_id, group_id=group_id)

        # One client per (job_id, group_id).
        for role, group in self.groups.items():
            self._clients[role] = client_factory(self.orch_addr, self.job_id, group)
        _log(
            f"job={self.job_id} connected orchestrator={self.orch_addr} "
            f"trainer_group={trainer_group} sampler_group={sampler_group}"
        )
        atexit.register(self._atexit_cleanup)

    @classmethod
    def from_env(cls, client_factory=None) -> "RoleLocks":
        return cls(
            job_id=os.environ.get(ENV_JOB_ID),
            orch_addr=os.environ.get(ENV_ORCH_ADDR),
            trainer_group=os.environ.get(ENV_TRAINER_GROUP),
            sampler_group=os.environ.get(ENV_SAMPLER_GROUP),
            client_factory=client_factory,
        )

    # ------------------------------------------------------------------ core
    def held(self, role: str) -> bool:
        return role in self._held

    DEFAULT_ACQUIRE_TIMEOUT_SEC = 600

    def acquire(self, role: str, timeout_sec: float | None = None) -> None:
        """Blocking-acquire the group lock for `role` (idempotent).

        Raises RuntimeError on a lock-order violation: TRAINER must never be
        requested while SAMPLER is held.
        Raises TimeoutError if the lock is not granted within `timeout_sec`
        seconds (default: 600).
        """
        if role not in (TRAINER, SAMPLER):
            raise ValueError(f"unknown role {role!r}")
        if role == TRAINER and SAMPLER in self._held and TRAINER not in self._held:
            raise RuntimeError(
                "lock-order violation: acquiring TRAINER while holding SAMPLER "
                "(global order is trainer-first; release SAMPLER first)"
            )
        if not self.enabled or role in self._held:
            return
        if timeout_sec is None:
            timeout_sec = self.DEFAULT_ACQUIRE_TIMEOUT_SEC
        t0 = time.monotonic()
        result = self._clients[role].acquire(
            group_id=self.groups[role],
            timeout_sec=timeout_sec,
        )
        waited_ms = getattr(result, "waited_ms", None)
        if waited_ms is None:
            waited_ms = int((time.monotonic() - t0) * 1000)
        self._held.add(role)
        _log(
            f"job={self.job_id} ACQUIRE role={role} group={self.groups[role]} "
            f"waited={waited_ms}ms context_restored={getattr(result, 'context_restored', '?')}"
        )

    def release(self, role: str) -> None:
        """Release the group lock for `role` (idempotent; errors logged, not raised)."""
        if not self.enabled or role not in self._held:
            return
        try:
            result = self._clients[role].release(group_id=self.groups[role])
            _log(
                f"job={self.job_id} RELEASE role={role} group={self.groups[role]} "
                f"pending_waiters={getattr(result, 'pending_waiters', '?')} "
                f"snapshot_deferred={getattr(result, 'snapshot_deferred', '?')}"
            )
        except Exception as e:  # noqa: BLE001 - never let a release error kill the job
            _log(f"job={self.job_id} RELEASE role={role} FAILED: {e}")
        finally:
            self._held.discard(role)

    def release_all(self) -> None:
        """Release everything in reverse lock order (SAMPLER first, then TRAINER)."""
        for role in (SAMPLER, TRAINER):
            self.release(role)

    def close(self) -> None:
        if self._closed:
            return
        self.release_all()
        for role, client in self._clients.items():
            try:
                client.close()
            except Exception as e:  # noqa: BLE001
                _log(f"job={self.job_id} client close role={role} FAILED: {e}")
        self._closed = True

    # ------------------------------------------------------------- internals
    def _atexit_cleanup(self) -> None:
        if self._closed or not self._held:
            return
        _log(f"job={self.job_id} atexit: releasing held roles {sorted(self._held)}")
        try:
            self.close()
        except Exception as e:  # noqa: BLE001
            print(f"[timeslice] atexit cleanup failed: {e}", file=sys.stderr, flush=True)
