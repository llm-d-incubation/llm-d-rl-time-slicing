"""Lock protocol tests for TimesliceCallback (no GPU, no Slime, no Ray).

Tests the REAL TimesliceCallback class by stubbing the slime.utils.phase_callback
module so it can be imported without a Slime installation.

Asserts:
  * Trainer-first global order (TRAINER never requested while SAMPLER held)
  * No lock leaks (every acquire has a matching release)
  * Dual-lock during weight_sync and init (both held during NCCL broadcast)
  * No-op mode runs clean when env vars are absent
  * All GPU-touching phases (generate, train, weight_sync, offload, onload,
    eval, create) are covered by locks
"""

import importlib.util
import os
import sys
import types
import unittest

# Stub slime.utils.phase_callback so TimesliceCallback can be imported
# without a Slime installation.
_slime = types.ModuleType("slime")
_slime_utils = types.ModuleType("slime.utils")
_slime_phase_callback = types.ModuleType("slime.utils.phase_callback")
_slime_phase_callback.PhaseCallback = object
_slime.utils = _slime_utils
sys.modules.setdefault("slime", _slime)
sys.modules.setdefault("slime.utils", _slime_utils)
sys.modules.setdefault("slime.utils.phase_callback", _slime_phase_callback)

_LOCKS_PATH = os.path.join(os.path.dirname(__file__), "..", "timeslice_slime", "locks.py")
_spec = importlib.util.spec_from_file_location("timeslice_slime.locks", _LOCKS_PATH)
_locks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_locks)

# Stub timeslice_slime package so callback.py can do "from timeslice_slime.locks import ..."
_ts_pkg = types.ModuleType("timeslice_slime")
_ts_pkg.locks = _locks
sys.modules["timeslice_slime"] = _ts_pkg
sys.modules["timeslice_slime.locks"] = _locks

_CB_PATH = os.path.join(os.path.dirname(__file__), "..", "timeslice_slime", "callback.py")
_cb_spec = importlib.util.spec_from_file_location("timeslice_slime.callback", _CB_PATH)
_cb = importlib.util.module_from_spec(_cb_spec)
_cb_spec.loader.exec_module(_cb)

SAMPLER = _locks.SAMPLER
TRAINER = _locks.TRAINER
RoleLocks = _locks.RoleLocks
TimesliceCallback = _cb.TimesliceCallback

JOB = "job-a"
ADDR = "orch:50051"
TG = "trainers"
SG = "samplers"


class FakeClient:
    def __init__(self, events, job_id, group_id):
        self.events = events
        self.job_id = job_id
        self.group_id = group_id
        self.closed = False

    def acquire(self, group_id, timeout_sec=None):
        assert group_id == self.group_id
        self.events.append(("acquire", group_id))
        return type("R", (), {"waited_ms": 0, "context_restored": False})()

    def release(self, group_id):
        assert group_id == self.group_id
        self.events.append(("release", group_id))
        return type("R", (), {"pending_waiters": 0, "snapshot_deferred": False})()

    def close(self):
        self.closed = True


def make_locks(events):
    return RoleLocks(
        job_id=JOB,
        orch_addr=ADDR,
        trainer_group=TG,
        sampler_group=SG,
        client_factory=lambda target, job_id, group_id: FakeClient(events, job_id, group_id),
    )


def held_after(events):
    held = set()
    for op, group in events:
        if op == "acquire":
            assert group not in held, f"double acquire of {group}: {events}"
            held.add(group)
        else:
            assert group in held, f"release of un-held {group}: {events}"
            held.discard(group)
    return held


def make_callback(events):
    """Create a real TimesliceCallback with fake orchestrator clients."""
    locks = make_locks(events)
    cb = TimesliceCallback.__new__(TimesliceCallback)
    cb.locks = locks
    return cb, locks


class TestSyncDriverSequence(unittest.TestCase):
    """Replay the sync driver (train.py) phase sequence."""

    def _run_sync(self, n_steps, offload_rollout=False, release_train=False, with_eval=False):
        events = []
        cb, locks = make_callback(events)

        cb.on_phase_begin("init", "both")
        cb.on_phase_end("init", "both")

        for step in range(n_steps):
            ctx = {"rollout_id": step}

            cb.on_phase_begin("generate", "sampler", ctx)
            cb.on_phase_end("generate", "sampler", ctx)

            if offload_rollout:
                cb.on_phase_begin("offload", "sampler", ctx)
                cb.on_phase_end("offload", "sampler", ctx)

            if release_train:
                cb.on_phase_begin("create", "trainer", ctx)
                cb.on_phase_end("create", "trainer", ctx)

            cb.on_phase_begin("train", "trainer", ctx)
            self.assertTrue(locks.held(TRAINER))
            self.assertFalse(locks.held(SAMPLER))
            cb.on_phase_end("train", "trainer", ctx)

            if offload_rollout and not release_train:
                cb.on_phase_begin("onload", "sampler", ctx)
                cb.on_phase_end("onload", "sampler", ctx)

            cb.on_phase_begin("weight_sync", "both", ctx)
            self.assertTrue(locks.held(TRAINER) and locks.held(SAMPLER))
            cb.on_phase_end("weight_sync", "both", ctx)
            self.assertFalse(locks.held(TRAINER), "TRAINER must be released after weight_sync")
            self.assertFalse(locks.held(SAMPLER), "SAMPLER must be released after weight_sync")

            if offload_rollout:
                cb.on_phase_begin("onload", "sampler", ctx)
                cb.on_phase_end("onload", "sampler", ctx)

            if with_eval:
                cb.on_phase_begin("eval", "sampler", ctx)
                cb.on_phase_end("eval", "sampler", ctx)

        cb.close()
        return events, locks

    def test_basic_3_steps(self):
        events, _ = self._run_sync(3)
        self.assertEqual(held_after(events), set())

    def test_with_offload_rollout(self):
        events, _ = self._run_sync(2, offload_rollout=True)
        self.assertEqual(held_after(events), set())

    def test_with_release_train(self):
        events, _ = self._run_sync(2, release_train=True)
        self.assertEqual(held_after(events), set())

    def test_with_eval(self):
        events, _ = self._run_sync(2, with_eval=True)
        self.assertEqual(held_after(events), set())

    def test_all_options(self):
        events, _ = self._run_sync(2, offload_rollout=True, release_train=True, with_eval=True)
        self.assertEqual(held_after(events), set())

    def test_trainer_first_order(self):
        events, _ = self._run_sync(3, offload_rollout=True, with_eval=True)
        held = set()
        for op, group in events:
            if op == "acquire":
                if group == TG:
                    self.assertNotIn(SG, held, f"TRAINER acquired while SAMPLER held: {events}")
                held.add(group)
            else:
                held.discard(group)

    def test_no_locks_between_phases(self):
        """After weight_sync end, no locks are held going into the next iteration."""
        events = []
        cb, locks = make_callback(events)
        cb.on_phase_begin("init", "both")
        cb.on_phase_end("init", "both")
        for step in range(2):
            ctx = {"rollout_id": step}
            self.assertFalse(locks.held(TRAINER), f"TRAINER leaked into step {step}")
            self.assertFalse(locks.held(SAMPLER), f"SAMPLER leaked into step {step}")
            cb.on_phase_begin("generate", "sampler", ctx)
            cb.on_phase_end("generate", "sampler", ctx)
            cb.on_phase_begin("train", "trainer", ctx)
            cb.on_phase_end("train", "trainer", ctx)
            cb.on_phase_begin("weight_sync", "both", ctx)
            cb.on_phase_end("weight_sync", "both", ctx)
        cb.close()


class TestAsyncDriverSequence(unittest.TestCase):
    """Replay the async driver (train_async.py) phase sequence.

    In async mode, generate is dispatched before train — the sampler lock
    is acquired at dispatch and held through training until collection.
    With per-job sampler groups this is a no-op on the orchestrator side.
    """

    def _run_async(self, n_steps, sync_interval=1, with_eval=False):
        """Replay the async driver lock sequence.

        Lock order in async overlap: TRAINER first, then SAMPLER.
        Gen dispatch happens AFTER train begin (TRAINER held), so SAMPLER
        acquisition respects the global order.
        """
        events = []
        cb, locks = make_callback(events)

        cb.on_phase_begin("init", "both")
        cb.on_phase_end("init", "both")

        # Pre-loop: dispatch first generate
        cb.on_phase_begin("generate", "sampler", {"rollout_id": 0})

        for step in range(n_steps):
            ctx = {"rollout_id": step}

            # Collect pending generation
            cb.on_phase_end("generate", "sampler", ctx)

            # Acquire TRAINER first
            cb.on_phase_begin("train", "trainer", ctx)
            self.assertTrue(locks.held(TRAINER))

            # Then dispatch next gen (acquire SAMPLER — order respected)
            if step + 1 < n_steps:
                cb.on_phase_begin("generate", "sampler", {"rollout_id": step + 1})

            # Training runs with both TRAINER and SAMPLER held
            cb.on_phase_end("train", "trainer", ctx)

            if (step + 1) % sync_interval == 0:
                # Collect pending generate before weight sync
                if step + 1 < n_steps and locks.held(SAMPLER):
                    cb.on_phase_end("generate", "sampler", ctx)
                cb.on_phase_begin("weight_sync", "both", ctx)
                self.assertTrue(locks.held(TRAINER) and locks.held(SAMPLER))
                cb.on_phase_end("weight_sync", "both", ctx)

            if with_eval:
                cb.on_phase_begin("eval", "sampler", ctx)
                cb.on_phase_end("eval", "sampler", ctx)

        cb.close()
        return events, locks

    def test_every_step_sync(self):
        events, _ = self._run_async(4, sync_interval=1)
        self.assertEqual(held_after(events), set())

    def test_periodic_sync(self):
        events, _ = self._run_async(6, sync_interval=3)
        self.assertEqual(held_after(events), set())

    def test_with_eval(self):
        events, _ = self._run_async(3, sync_interval=1, with_eval=True)
        self.assertEqual(held_after(events), set())

    def test_trainer_first_order(self):
        events, _ = self._run_async(4, sync_interval=2)
        held = set()
        for op, group in events:
            if op == "acquire":
                if group == TG:
                    self.assertNotIn(SG, held, "TRAINER acquired while SAMPLER held")
                held.add(group)
            else:
                held.discard(group)

    def test_sampler_held_during_train(self):
        """In async mode, sampler lock is held during training (overlapping GPU work).

        Lock order: TRAINER acquired first (train begin), then SAMPLER
        (gen dispatch).  Both held during training execution.
        """
        events = []
        cb, locks = make_callback(events)
        cb.on_phase_begin("init", "both")
        cb.on_phase_end("init", "both")
        # Dispatch gen 0
        cb.on_phase_begin("generate", "sampler", {"rollout_id": 0})
        # Collect gen 0
        cb.on_phase_end("generate", "sampler", {"rollout_id": 0})
        # Acquire TRAINER first (train begin)
        cb.on_phase_begin("train", "trainer", {"rollout_id": 0})
        self.assertTrue(locks.held(TRAINER))
        # Dispatch gen 1 — acquire SAMPLER (order: TRAINER then SAMPLER)
        cb.on_phase_begin("generate", "sampler", {"rollout_id": 1})
        self.assertTrue(locks.held(SAMPLER), "SAMPLER must be held during async train")
        self.assertTrue(locks.held(TRAINER), "TRAINER must be held during async train")
        cb.on_phase_end("train", "trainer", {"rollout_id": 0})


class TestNoopMode(unittest.TestCase):
    def test_no_env_runs_clean(self):
        locks = RoleLocks(None, None, None, None)
        cb = TimesliceCallback.__new__(TimesliceCallback)
        cb.locks = locks
        cb.on_phase_begin("init", "both")
        cb.on_phase_end("init", "both")
        cb.on_phase_begin("generate", "sampler")
        cb.on_phase_end("generate", "sampler")
        cb.on_phase_begin("train", "trainer")
        cb.on_phase_end("train", "trainer")
        cb.on_phase_begin("weight_sync", "both")
        cb.on_phase_end("weight_sync", "both")
        cb.on_phase_begin("offload", "sampler")
        cb.on_phase_end("offload", "sampler")
        cb.on_phase_begin("onload", "sampler")
        cb.on_phase_end("onload", "sampler")
        cb.on_phase_begin("eval", "sampler")
        cb.on_phase_end("eval", "sampler")
        cb.on_phase_begin("create", "trainer")
        cb.on_phase_end("create", "trainer")
        cb.close()


class TestRoleLocks(unittest.TestCase):
    def test_idempotent_acquire(self):
        events = []
        locks = make_locks(events)
        locks.acquire(TRAINER)
        locks.acquire(TRAINER)
        self.assertEqual(events, [("acquire", TG)])

    def test_order_violation(self):
        events = []
        locks = make_locks(events)
        locks.acquire(SAMPLER)
        with self.assertRaises(RuntimeError):
            locks.acquire(TRAINER)

    def test_same_group_rejected(self):
        with self.assertRaises(ValueError):
            RoleLocks(JOB, ADDR, "g", "g", client_factory=lambda *a: FakeClient([], JOB, "g"))

    def test_close_releases_all(self):
        events = []
        locks = make_locks(events)
        locks.acquire(TRAINER)
        locks.acquire(SAMPLER)
        locks.close()
        self.assertEqual(held_after(events), set())

    def test_release_all_reverse_order(self):
        events = []
        locks = make_locks(events)
        locks.acquire(TRAINER)
        locks.acquire(SAMPLER)
        locks.release_all()
        self.assertEqual(events[-2:], [("release", SG), ("release", TG)])


if __name__ == "__main__":
    unittest.main()
