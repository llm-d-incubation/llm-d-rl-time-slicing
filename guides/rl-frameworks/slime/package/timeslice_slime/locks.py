"""PhaseLocks: idempotent group-lock helper around timeslice.TimeSliceOrchestratorClient.

Configuration comes from the environment:
  TIMESLICE_JOB_ID    unique job identifier (e.g. "job-a")
  TIMESLICE_ORCH_ADDR node-local accelerator-orchestrator gRPC address (e.g. "127.0.0.1:50051")
  TIMESLICE_GROUP     time-slice group id shared by the colocated jobs (e.g. "gpu0")

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
    # must always be visible in job logs for the PoC timeline analysis.
    print(f"[timeslice] {msg}", flush=True)


class PhaseLocks:
    """Idempotent acquire/release of orchestrator group locks.

    ensure(groups)  - blocking-acquire every group not already held (idempotent).
    drop_all()      - release every held group (idempotent, safe to call anytime).
    close()         - drop_all + close the gRPC channel.
    """

    def __init__(
        self,
        job_id: Optional[str],
        orch_addr: Optional[str],
        group: Optional[str],
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
                "WARNING: time-slicing disabled (missing one of "
                f"{ENV_JOB_ID}/{ENV_ORCH_ADDR}/{ENV_GROUP}); PhaseLocks is a no-op."
            )
            return

        try:
            from timeslice import OrchestratorClient
        except ImportError:
            from timeslice import TimeSliceOrchestratorClient as OrchestratorClient

        self._client = OrchestratorClient(
            target=self.orch_addr, job_id=self.job_id, group_id=self.group
        )
        _log(f"job={self.job_id} connected orchestrator={self.orch_addr} group={self.group}")
        # Best-effort safety net: never exit holding the group lock. Registered
        # here (not in the trainer) so any user of PhaseLocks gets it.
        atexit.register(self._atexit_cleanup)

    @classmethod
    def from_env(cls) -> "PhaseLocks":
        return cls(
            job_id=os.environ.get(ENV_JOB_ID),
            orch_addr=os.environ.get(ENV_ORCH_ADDR),
            group=os.environ.get(ENV_GROUP),
        )

    # ------------------------------------------------------------------ core
    def ensure(self, groups: Optional[Iterable[str]] = None) -> None:
        """Blocking-acquire each group in `groups` (default: the configured group).

        Already-held groups are skipped, so calling this every step is cheap.
        Blocks until the orchestrator grants the lock (whole-step turn-taking).
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


# ======================================================================
# v2: role-based locks for the disaggregated two-pool topology
# ======================================================================
#
# Two orchestrator groups per job:
#   TIMESLICE_TRAINER_GROUP  group id of the trainers pool (e.g. "trainers")
#   TIMESLICE_SAMPLER_GROUP  group id of the samplers pool (e.g. "samplers")
# plus the shared TIMESLICE_JOB_ID / TIMESLICE_ORCH_ADDR.
#
# Global lock order (deadlock freedom): TRAINER before SAMPLER.
#   - A job may hold SAMPLER alone (sample-wait span), but it must never
#     REQUEST TRAINER while holding SAMPLER.
#   - Dual-lock spans (init weight sync, per-step weight sync, eval) are
#     always entered by acquiring SAMPLER while already holding TRAINER.
# With only these two wait shapes ("want TRAINER holding nothing",
# "want SAMPLER holding TRAINER") a wait-for cycle is impossible.

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
        job_id: Optional[str],
        orch_addr: Optional[str],
        trainer_group: Optional[str],
        sampler_group: Optional[str],
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
            try:
                from timeslice import OrchestratorClient
            except ImportError:
                from timeslice import TimeSliceOrchestratorClient as OrchestratorClient

            def client_factory(target, job_id, group_id):
                return OrchestratorClient(target=target, job_id=job_id, group_id=group_id)

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

    def acquire(self, role: str) -> None:
        """Blocking-acquire the group lock for `role` (idempotent).

        Raises RuntimeError on a lock-order violation: TRAINER must never be
        requested while SAMPLER is held.
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
        t0 = time.monotonic()
        result = self._clients[role].acquire(group_id=self.groups[role])
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


class PhaseTransitions:
    """Maps verl v1 trainer hook points onto RoleLocks transitions.

    Phase model per training job (mode "separate_async_timesliced"):

      init_begin        + TRAINER, + SAMPLER   trainer.init(): worker groups,
                                               model load, on_init_end syncs
                                               weights to hybrid AND standalone
                                               replicas (dual-pool span).
      train_begin       - SAMPLER              keep TRAINER for feed/step 1.
      sample_begin(True)  - TRAINER, + SAMPLER yield-on-starvation: the trainer
                                               pool is free for the other job
                                               while this job's generation runs
                                               on the sampler pool.
      sample_begin(False) (no-op)              conditional-release refinement:
                                               replay buffer already has the
                                               batch; keep TRAINER, skip SAMPLER.
      sample_end        - SAMPLER, + TRAINER   release before re-acquiring so
                                               TRAINER is never requested while
                                               SAMPLER is held (lock order).
      weight_sync_begin + SAMPLER (TRAINER held: dual-lock span for the
                                   cross-pool NCCL weight broadcast)
      weight_sync_end   - SAMPLER
      validate_begin    + SAMPLER (TRAINER held; separate_async validation also
                                   wakes the hybrid replicas on trainer GPUs)
      validate_end      - SAMPLER
      train_end         - SAMPLER, - TRAINER, close
    """

    def __init__(self, locks: "RoleLocks"):
        self.locks = locks

    # -- init / shutdown
    def init_begin(self) -> None:
        self.locks.acquire(TRAINER)
        self.locks.acquire(SAMPLER)

    def train_begin(self) -> None:
        self.locks.release(SAMPLER)

    def train_end(self) -> None:
        self.locks.release_all()
        self.locks.close()

    # -- sampling
    def sample_begin(self, starved: bool = True) -> None:
        if not starved:
            # Batch already buffered: keep TRAINER, no need for the sampler pool.
            return
        # Yield the trainer pool while blocked on generation, then hold the
        # sampler pool for the span in which this job's generation executes.
        self.locks.release(TRAINER)
        self.locks.acquire(SAMPLER)

    def sample_end(self) -> None:
        # Order matters: SAMPLER must be released before TRAINER is requested.
        self.locks.release(SAMPLER)
        self.locks.acquire(TRAINER)

    # -- weight sync (cross-pool NCCL broadcast: both pools must be resident)
    def weight_sync_begin(self) -> None:
        if self.locks.enabled and not self.locks.held(TRAINER):
            raise RuntimeError("weight_sync_begin without TRAINER held (invariant violation)")
        self.locks.acquire(SAMPLER)

    def weight_sync_end(self) -> None:
        self.locks.release(SAMPLER)

    # -- validation
    def validate_begin(self) -> None:
        self.locks.acquire(TRAINER)  # idempotent; normally already held
        self.locks.acquire(SAMPLER)

    def validate_end(self) -> None:
        self.locks.release(SAMPLER)
