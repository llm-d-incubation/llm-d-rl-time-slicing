"""Lock protocol tests for TimesliceCallback (no GPU, no Slime, no Ray).

Simulates the exact phase sequences that Slime's sync and async drivers
produce and asserts:
  * Trainer-first global order (TRAINER never requested while SAMPLER held)
  * No lock leaks (every acquire has a matching release)
  * Dual-lock during weight_sync (both held during NCCL broadcast)
  * Weight sync keeps TRAINER after end (for next step's train phase)
  * No-op mode runs clean when env vars are absent
"""

import importlib.util
import os
import unittest

_LOCKS_PATH = os.path.join(os.path.dirname(__file__), "..", "timeslice_slime", "locks.py")
_spec = importlib.util.spec_from_file_location("_timeslice_locks", _LOCKS_PATH)
_locks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_locks)

SAMPLER = _locks.SAMPLER
TRAINER = _locks.TRAINER
RoleLocks = _locks.RoleLocks

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

    def acquire(self, group_id):
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


class CallbackHarness:
    """Simulate TimesliceCallback without importing Slime."""

    def __init__(self, locks):
        self.locks = locks

    def on_phase_begin(self, phase, role, context=None):
        if role in ("trainer", "both"):
            self.locks.acquire(TRAINER)
        if role in ("sampler", "both"):
            self.locks.acquire(SAMPLER)

    def on_phase_end(self, phase, role, context=None):
        if phase == "weight_sync":
            self.locks.release(SAMPLER)
            return
        if role in ("sampler", "both"):
            self.locks.release(SAMPLER)
        if role in ("trainer", "both"):
            self.locks.release(TRAINER)

    def close(self):
        self.locks.close()


class TestSyncDriverSequence(unittest.TestCase):
    """Replay the sync driver (train.py) phase sequence."""

    def _run_sync(self, n_steps, with_save=False, with_eval=False):
        events = []
        locks = make_locks(events)
        cb = CallbackHarness(locks)

        cb.on_phase_begin("init", "both")
        cb.on_phase_end("init", "both")

        for step in range(n_steps):
            ctx = {"rollout_id": step}
            cb.on_phase_begin("generate", "sampler", ctx)
            cb.on_phase_end("generate", "sampler", ctx)

            cb.on_phase_begin("train", "trainer", ctx)
            self.assertTrue(locks.held(TRAINER))
            self.assertFalse(locks.held(SAMPLER))
            cb.on_phase_end("train", "trainer", ctx)

            if with_save:
                cb.on_phase_begin("save", "trainer", ctx)
                cb.on_phase_end("save", "trainer", ctx)

            cb.on_phase_begin("weight_sync", "both", ctx)
            self.assertTrue(locks.held(TRAINER) and locks.held(SAMPLER))
            cb.on_phase_end("weight_sync", "both", ctx)
            self.assertTrue(locks.held(TRAINER), "TRAINER must be kept after weight_sync")
            self.assertFalse(locks.held(SAMPLER), "SAMPLER must be released after weight_sync")

            if with_eval:
                cb.on_phase_begin("eval", "sampler", ctx)
                cb.on_phase_end("eval", "sampler", ctx)

        cb.close()
        return events, locks

    def test_basic_3_steps(self):
        events, locks = self._run_sync(3)
        self.assertEqual(held_after(events), set())

    def test_with_save_and_eval(self):
        events, locks = self._run_sync(2, with_save=True, with_eval=True)
        self.assertEqual(held_after(events), set())

    def test_trainer_first_order(self):
        events, _ = self._run_sync(3, with_save=True, with_eval=True)
        held = set()
        for op, group in events:
            if op == "acquire":
                if group == TG:
                    self.assertNotIn(SG, held, f"TRAINER acquired while SAMPLER held: {events}")
                held.add(group)
            else:
                held.discard(group)

    def test_weight_sync_keeps_trainer(self):
        events = []
        locks = make_locks(events)
        cb = CallbackHarness(locks)
        cb.on_phase_begin("init", "both")
        cb.on_phase_end("init", "both")
        cb.on_phase_begin("generate", "sampler")
        cb.on_phase_end("generate", "sampler")
        cb.on_phase_begin("train", "trainer")
        cb.on_phase_end("train", "trainer")
        cb.on_phase_begin("weight_sync", "both")
        cb.on_phase_end("weight_sync", "both")
        self.assertTrue(locks.held(TRAINER))
        self.assertFalse(locks.held(SAMPLER))


class TestAsyncDriverSequence(unittest.TestCase):
    """Replay the async driver (train_async.py) phase sequence."""

    def _run_async(self, n_steps, sync_interval=1):
        events = []
        locks = make_locks(events)
        cb = CallbackHarness(locks)

        cb.on_phase_begin("init", "both")
        cb.on_phase_end("init", "both")

        for step in range(n_steps):
            ctx = {"rollout_id": step}
            cb.on_phase_begin("generate", "sampler", ctx)
            cb.on_phase_end("generate", "sampler", ctx)

            cb.on_phase_begin("train", "trainer", ctx)
            cb.on_phase_end("train", "trainer", ctx)

            if (step + 1) % sync_interval == 0:
                cb.on_phase_begin("weight_sync", "both", ctx)
                self.assertTrue(locks.held(TRAINER) and locks.held(SAMPLER))
                cb.on_phase_end("weight_sync", "both", ctx)

        cb.close()
        return events, locks

    def test_every_step_sync(self):
        events, _ = self._run_async(4, sync_interval=1)
        self.assertEqual(held_after(events), set())

    def test_periodic_sync(self):
        events, _ = self._run_async(6, sync_interval=3)
        self.assertEqual(held_after(events), set())
        # 6 generate phases acquire SAMPLER + init acquires SAMPLER + 2 weight syncs acquire SAMPLER = 9
        sampler_acquires = sum(1 for op, g in events if op == "acquire" and g == SG)
        self.assertEqual(sampler_acquires, 9)

    def test_trainer_first_order(self):
        events, _ = self._run_async(4, sync_interval=2)
        held = set()
        for op, group in events:
            if op == "acquire":
                if group == TG:
                    self.assertNotIn(SG, held)
                held.add(group)
            else:
                held.discard(group)


class TestNoopMode(unittest.TestCase):
    def test_no_env_runs_clean(self):
        locks = RoleLocks(None, None, None, None)
        cb = CallbackHarness(locks)
        cb.on_phase_begin("init", "both")
        cb.on_phase_end("init", "both")
        cb.on_phase_begin("generate", "sampler")
        cb.on_phase_end("generate", "sampler")
        cb.on_phase_begin("train", "trainer")
        cb.on_phase_end("train", "trainer")
        cb.on_phase_begin("weight_sync", "both")
        cb.on_phase_end("weight_sync", "both")
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
            RoleLocks(JOB, ADDR, "g", "g",
                      client_factory=lambda *a: FakeClient([], JOB, "g"))

    def test_close_releases_all(self):
        events = []
        locks = make_locks(events)
        locks.acquire(TRAINER)
        locks.acquire(SAMPLER)
        locks.close()
        self.assertEqual(held_after(events), set())


if __name__ == "__main__":
    unittest.main()
