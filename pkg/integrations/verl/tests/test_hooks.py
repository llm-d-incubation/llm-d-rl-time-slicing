"""Unit tests for timeslice_verl (pure python: no GPU, no verl, no grpc, no ray).

Run:  python3 -m pytest tests/ -v

What is covered:
  * inert-without-env: every hook is a no-op unless TIMESLICE_FULLY_ASYNC=1
  * lock transition order across the hook sequence the fully-async trainer
    fires (init_workers -> initial param sync -> N x (sample_end ->
    update_weights)):
      - on_init_workers_begin acquires, on_init_workers_end yields
      - on_sample_end acquires (per-step resume point)
      - on_update_weights_begin is an ensure-held no-op in steady state and a
        real acquire on the pre-fit initial-param-sync path
      - on_update_weights_end yields only when synced=True
      - on_save_checkpoint vetoes (returns False) when save_freq > 0
      - on_exception releases the lock (crash-release)
  * client-rename compat (OrchestratorClient -> TimeSliceOrchestratorClient)
  * experimental TIMESLICE_EMPTY_CACHE_BEFORE_YIELD (inert by default)
  * timeslice_verl.trainer registers under "timeslice" and wraps
    _fit_update_weights with crash-release (against a stubbed verl trainer)
"""

import asyncio
import importlib
import inspect
import os
import sys
import types
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import timeslice_verl.hooks as hooks_mod  # noqa: E402
import timeslice_verl.locks as locks_mod  # noqa: E402
from timeslice_verl.hooks import TimesliceHooksMixin  # noqa: E402

JOB = "job-a"
ADDR = "orch:50051"
GROUP = "trainers"


class FakeAcquireResult:
    def __init__(self, waited_ms=7, context_restored=True):
        self.success = True
        self.waited_ms = waited_ms
        self.context_restored = context_restored


class FakeYieldResult:
    success = True
    pending_waiters = 1
    snapshot_deferred = False


class FakeClient:
    """Records acquire/release into a shared event log."""

    def __init__(self, events, job_id, group_id):
        self.events = events
        self.job_id = job_id
        self.group_id = group_id
        self.closed = False

    def acquire(self, group_id):
        assert group_id == self.group_id
        self.events.append(("acquire", group_id))
        return FakeAcquireResult()

    def release(self, group_id):
        assert group_id == self.group_id
        self.events.append(("release", group_id))
        return FakeYieldResult()

    def close(self):
        self.closed = True


class FakeTrainer(TimesliceHooksMixin):
    """The hooks mixin over a stand-in for verl's FullyAsyncTrainer state."""

    def __init__(self, save_freq=-1, client_factory=None):
        super().__init__(client_factory=client_factory)
        self.current_param_version = 3
        self.metrics = {}
        self.config = SimpleNamespace(trainer=SimpleNamespace(save_freq=save_freq))


@pytest.fixture
def events():
    return []


@pytest.fixture
def ts_env(monkeypatch, events):
    """Full fully-async env + injected grpc-free client factory."""
    monkeypatch.setenv("TIMESLICE_FULLY_ASYNC", "1")
    monkeypatch.setenv("TIMESLICE_JOB_ID", JOB)
    monkeypatch.setenv("TIMESLICE_ORCH_ADDR", ADDR)
    monkeypatch.setenv("TIMESLICE_GROUP", GROUP)
    monkeypatch.delenv(hooks_mod.ENV_EMPTY_CACHE, raising=False)
    return SimpleNamespace(events=events)


def make_trainer(events, save_freq=-1):
    return FakeTrainer(
        save_freq=save_freq,
        client_factory=lambda target, job_id, group_id: FakeClient(events, job_id, group_id),
    )


# Hook sequences exactly as the fully-async trainer fires them ---------------
# (the trainer awaits awaitable hook results; on_save_checkpoint is sync)


def run_init_workers(t, events, fail=False):
    async def seq():
        await t.on_init_workers_begin()
        events.append("orig_init_workers")
        if fail:
            exc = RuntimeError("init failed")
            await t.on_exception("init_workers", exc)
            raise exc
        await t.on_init_workers_end()

    asyncio.run(seq())


def run_sample_end(t, events, batch_arrived=True):
    async def seq():
        events.append("orig_get_samples")
        if batch_arrived:
            # verl fires on_sample_end only after a real batch was assembled
            # (the queue-termination shutdown path raises before the hook).
            await t.on_sample_end()

    asyncio.run(seq())


def run_update_weights(t, events, result):
    async def seq():
        await t.on_update_weights_begin()
        events.append("orig_update_weights")
        await t.on_update_weights_end(synced=result is not None)

    asyncio.run(seq())


# ======================================================================
# inertness
# ======================================================================


class TestInert:
    def test_all_hooks_noop_without_gate(self, monkeypatch, events):
        monkeypatch.delenv("TIMESLICE_FULLY_ASYNC", raising=False)
        monkeypatch.setenv("TIMESLICE_JOB_ID", JOB)
        monkeypatch.setenv("TIMESLICE_ORCH_ADDR", ADDR)
        monkeypatch.setenv("TIMESLICE_GROUP", GROUP)
        t = make_trainer(events)
        run_init_workers(t, events)
        run_sample_end(t, events)
        run_update_weights(t, events, {"timing": 1.0})
        # matches the base trainer's default (proceed with the save)
        assert t.on_save_checkpoint(force=True) is True
        asyncio.run(t.on_exception("update_weights", RuntimeError()))
        assert [e for e in events if isinstance(e, tuple)] == []
        assert t._locks is None  # never even built the lock layer

    def test_gate_on_but_no_lock_env_is_noop_locks(self, ts_env, events, monkeypatch):
        monkeypatch.delenv("TIMESLICE_JOB_ID")
        t = make_trainer(events)
        run_init_workers(t, events)
        run_sample_end(t, events)
        run_update_weights(t, events, {"timing": 1.0})
        assert [e for e in events if isinstance(e, tuple)] == []


# ======================================================================
# lock transition order
# ======================================================================


class TestLockOrder:
    def test_init_workers_acquire_then_yield(self, ts_env, events):
        t = make_trainer(events)
        run_init_workers(t, events)
        assert events == [("acquire", GROUP), "orig_init_workers", ("release", GROUP)]

    def test_init_workers_failure_releases(self, ts_env, events):
        t = make_trainer(events)
        with pytest.raises(RuntimeError):
            run_init_workers(t, events, fail=True)
        assert events == [("acquire", GROUP), "orig_init_workers", ("release", GROUP)]

    def test_sample_end_acquires(self, ts_env, events):
        t = make_trainer(events)
        run_sample_end(t, events)
        assert events == ["orig_get_samples", ("acquire", GROUP)]

    def test_shutdown_no_acquire(self, ts_env, events):
        t = make_trainer(events)
        run_sample_end(t, events, batch_arrived=False)  # queue termination path
        assert events == ["orig_get_samples"]

    def test_update_weights_yields_only_on_real_sync(self, ts_env, events):
        t = make_trainer(events)
        run_sample_end(t, events)
        del events[:]
        run_update_weights(t, events, {"timing": 1.0})
        # ensure-held no-op, yield after synced=True
        assert events == ["orig_update_weights", ("release", GROUP)]

    def test_update_weights_unsynced_keeps_lock(self, ts_env, events):
        t = make_trainer(events)
        run_sample_end(t, events)
        del events[:]
        run_update_weights(t, events, None)
        assert events == ["orig_update_weights"]  # no release
        assert "trigger_gt_1" in t._warned

    def test_initial_param_sync_acquires_when_not_held(self, ts_env, events):
        # fully_async_main calls _fit_update_weights between init_workers
        # (after which we yielded) and fit(): ensure-held must acquire.
        t = make_trainer(events)
        run_init_workers(t, events)
        del events[:]
        run_update_weights(t, events, {"timing": 1.0})
        assert events == [("acquire", GROUP), "orig_update_weights", ("release", GROUP)]

    def test_full_step_sequence_alternates(self, ts_env, events):
        t = make_trainer(events)
        run_init_workers(t, events)
        run_update_weights(t, events, {"t": 1})  # initial param sync
        for _ in range(2):
            run_sample_end(t, events)
            run_update_weights(t, events, {"t": 1})
        lock_events = [e for e in events if isinstance(e, tuple)]
        held = False
        for op, group in lock_events:
            assert group == GROUP
            if op == "acquire":
                assert not held, f"double acquire: {lock_events}"
                held = True
            else:
                assert held, f"release while not held: {lock_events}"
                held = False
        assert not held, f"lock leaked at end: {lock_events}"
        assert len(lock_events) == 2 * 4  # init, initial sync, 2 steps

    def test_save_checkpoint_vetoed_when_save_freq_positive(self, ts_env, events):
        t = make_trainer(events, save_freq=5)
        assert t.on_save_checkpoint(force=True) is False
        assert events == []  # no lock traffic
        assert "save_checkpoint" in t._warned

    def test_save_checkpoint_allowed_when_disabled(self, ts_env, events):
        t = make_trainer(events, save_freq=-1)
        assert t.on_save_checkpoint(force=True) is True


# ======================================================================
# crash release
# ======================================================================


class TestCrashRelease:
    def test_releases_held_lock(self, ts_env, events):
        t = make_trainer(events)
        run_sample_end(t, events)
        exc = RuntimeError("NcclError: unhandled cuda error")
        asyncio.run(t.on_exception("update_weights", exc))
        assert events == ["orig_get_samples", ("acquire", GROUP), ("release", GROUP)]
        assert not t._locks.held

    def test_idempotent_when_not_held(self, ts_env, events):
        t = make_trainer(events)
        run_init_workers(t, events)  # ends released
        del events[:]
        asyncio.run(t.on_exception("fit", RuntimeError("boom")))
        assert events == []  # nothing held -> drop_all releases nothing

    def test_never_raises(self, ts_env, events):
        t = make_trainer(events)
        run_sample_end(t, events)

        def bad_release(group_id):
            raise ConnectionError("orchestrator gone")

        t._locks._client.release = bad_release
        asyncio.run(t.on_exception("fit", RuntimeError("orig")))  # must not raise
        assert not t._locks.held  # still marked released locally


# ======================================================================
# client-rename compat (PhaseLocks lazy import fallback)
# ======================================================================


def _fake_timeslice_module(class_name):
    mod = types.ModuleType("timeslice")

    class Client:
        def __init__(self, target, job_id, group_id):
            self.target = target
            self.job_id = job_id
            self.group_id = group_id

        def acquire(self, group_id):
            return FakeAcquireResult()

        def release(self, group_id):
            return FakeYieldResult()

        def close(self):
            pass

    Client.__name__ = class_name
    setattr(mod, class_name, Client)
    return mod, Client


class TestClientRenameCompat:
    def test_new_client_name_only(self, monkeypatch):
        mod, cls = _fake_timeslice_module("TimeSliceOrchestratorClient")
        assert not hasattr(mod, "OrchestratorClient")
        monkeypatch.setitem(sys.modules, "timeslice", mod)
        pl = locks_mod.PhaseLocks(job_id=JOB, orch_addr=ADDR, group=GROUP)
        assert pl.enabled and isinstance(pl._client, cls)
        assert (pl._client.target, pl._client.job_id, pl._client.group_id) == (ADDR, JOB, GROUP)
        pl.ensure()
        assert pl.held

    def test_old_client_name_still_works(self, monkeypatch):
        mod, cls = _fake_timeslice_module("OrchestratorClient")
        monkeypatch.setitem(sys.modules, "timeslice", mod)
        pl = locks_mod.PhaseLocks(job_id=JOB, orch_addr=ADDR, group=GROUP)
        assert pl.enabled and isinstance(pl._client, cls)


# ======================================================================
# experimental TIMESLICE_EMPTY_CACHE_BEFORE_YIELD (inert by default)
# ======================================================================


def _fake_torch(events, raise_on_call=False):
    mod = types.ModuleType("torch")

    def empty_cache():
        if raise_on_call:
            raise RuntimeError("no CUDA context")
        events.append("empty_cache")

    mod.cuda = SimpleNamespace(empty_cache=empty_cache)
    return mod


class TestEmptyCacheBeforeYield:
    def test_inert_by_default(self, ts_env, events, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(events))
        t = make_trainer(events)
        run_init_workers(t, events)
        run_sample_end(t, events)
        run_update_weights(t, events, {"t": 1})
        assert "empty_cache" not in events

    def test_enabled_calls_before_each_yields_release(self, ts_env, events, monkeypatch):
        monkeypatch.setenv(hooks_mod.ENV_EMPTY_CACHE, "1")
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(events))
        t = make_trainer(events)
        run_init_workers(t, events)
        assert events == [("acquire", GROUP), "orig_init_workers", "empty_cache", ("release", GROUP)]
        del events[:]
        run_sample_end(t, events)
        run_update_weights(t, events, {"t": 1})
        assert events == [
            "orig_get_samples",
            ("acquire", GROUP),
            "orig_update_weights",
            "empty_cache",
            ("release", GROUP),
        ]

    def test_enabled_without_torch_is_safe(self, ts_env, events, monkeypatch):
        monkeypatch.setenv(hooks_mod.ENV_EMPTY_CACHE, "1")
        monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
        t = make_trainer(events)
        run_init_workers(t, events)
        assert events == [("acquire", GROUP), "orig_init_workers", ("release", GROUP)]

    def test_empty_cache_failure_never_breaks_yield(self, ts_env, events, monkeypatch):
        monkeypatch.setenv(hooks_mod.ENV_EMPTY_CACHE, "1")
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(events, raise_on_call=True))
        t = make_trainer(events)
        run_init_workers(t, events)
        assert events == [("acquire", GROUP), "orig_init_workers", ("release", GROUP)]


# ======================================================================
# timeslice_verl.trainer (against a stubbed verl trainer module)
# ======================================================================


@pytest.fixture
def stub_verl_trainer_module(monkeypatch):
    """Install a stub verl fully_async_trainer module and (re)import
    timeslice_verl.trainer against it."""
    registered = {}

    class FakeFullyAsyncTrainer:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs

        async def _fit_update_weights(self):
            raise RuntimeError("param sync blew up")

        async def _dispatch_on_exception(self, point, exc):
            # mirror verl's helper: fire on_exception once, awaiting coroutines
            if exc is getattr(self, "_last_on_exception_exc", None):
                return
            self._last_on_exception_exc = exc
            result = self.on_exception(point, exc)
            if inspect.isawaitable(result):
                await result

        def on_exception(self, point, exc):
            return None

    def register_trainer(name):
        def decorator(cls):
            registered[name] = cls
            return cls

        return decorator

    for name in ("verl", "verl.experimental", "verl.experimental.fully_async_policy"):
        mod = types.ModuleType(name)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)
    fat = types.ModuleType("verl.experimental.fully_async_policy.fully_async_trainer")
    fat.FullyAsyncTrainer = FakeFullyAsyncTrainer
    fat.register_trainer = register_trainer
    monkeypatch.setitem(sys.modules, "verl.experimental.fully_async_policy.fully_async_trainer", fat)

    sys.modules.pop("timeslice_verl.trainer", None)
    trainer_mod = importlib.import_module("timeslice_verl.trainer")
    yield SimpleNamespace(module=trainer_mod, registered=registered, base=FakeFullyAsyncTrainer)
    sys.modules.pop("timeslice_verl.trainer", None)


class TestTrainerRegistration:
    def test_import_registers_timeslice_trainer(self, stub_verl_trainer_module):
        cls = stub_verl_trainer_module.registered.get("timeslice")
        assert cls is stub_verl_trainer_module.module.TimesliceFullyAsyncTrainer
        assert issubclass(cls, TimesliceHooksMixin)
        assert issubclass(cls, stub_verl_trainer_module.base)
        # mixin first in the MRO: its hook overrides win
        mro = cls.__mro__
        assert mro.index(TimesliceHooksMixin) < mro.index(stub_verl_trainer_module.base)

    def test_init_forwards_kwargs_and_sets_up_hooks_state(self, stub_verl_trainer_module):
        t = stub_verl_trainer_module.registered["timeslice"](config="cfg", tokenizer="tok")
        assert t.init_kwargs == {"config": "cfg", "tokenizer": "tok"}
        assert t._locks is None and t._warned == set()

    def test_update_weights_crash_fires_crash_release(self, stub_verl_trainer_module, ts_env, events):
        cls = stub_verl_trainer_module.registered["timeslice"]
        t = cls()
        t.current_param_version = 1
        t.metrics = {}
        t._client_factory = lambda target, job_id, group_id: FakeClient(events, job_id, group_id)
        # take the lock the way the pre-fit path does, then crash the sync
        asyncio.run(t.on_update_weights_begin())
        assert events == [("acquire", GROUP)]
        with pytest.raises(RuntimeError, match="param sync blew up"):
            asyncio.run(t._fit_update_weights())
        assert events == [("acquire", GROUP), ("release", GROUP)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
