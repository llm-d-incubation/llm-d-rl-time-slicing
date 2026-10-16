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
  * acquire timeout: PhaseLocks.ensure passes timeout_sec to the client
    (default DEFAULT_ACQUIRE_TIMEOUT_SEC=600); a client TimeoutError propagates
  * client-rename compat (OrchestratorClient -> TimeSliceOrchestratorClient)
  * experimental TIMESLICE_EMPTY_CACHE_BEFORE_YIELD (inert by default)
  * timeslice_verl.trainer registers under "timeslice" and wraps
    _fit_update_weights with crash-release (against a stubbed verl trainer)
  * app-channel offload (TIMESLICE_APP_OFFLOAD, experimental): inert by
    default; per-node proxies are spawned at the end of init_workers BEFORE
    the yield and closed on crash; the offload spec targets f"{train_role}_to";
    missing env degrades to warn-once; spawn failures propagate (against a
    stubbed timeslice.snapshot_agent.rayworker, no ray/grpc); once the proxies
    are up every release carries backend_config="app_channel" (the per-job
    backend the orchestrator forwards to the agents), degraded paths and the
    crash path yield without one
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
import timeslice_verl.offload as offload_mod  # noqa: E402
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

    def acquire(self, group_id, timeout_sec=None):
        assert group_id == self.group_id
        self.last_timeout_sec = timeout_sec
        self.events.append(("acquire", group_id))
        return FakeAcquireResult()

    def release(self, group_id, backend_config=None):
        assert group_id == self.group_id
        if backend_config is None:
            self.events.append(("release", group_id))
        else:
            self.events.append(("release", group_id, backend_config))
        return FakeYieldResult()

    def close(self):
        self.closed = True


class FakeTrainer(TimesliceHooksMixin):
    """The hooks mixin over a stand-in for verl's FullyAsyncTrainer state."""

    def __init__(self, save_freq=-1, client_factory=None, proxy_spawner=None):
        super().__init__(client_factory=client_factory, proxy_spawner=proxy_spawner)
        self.current_param_version = 3
        self.metrics = {}
        self.config = SimpleNamespace(trainer=SimpleNamespace(save_freq=save_freq))
        # what the fully-async trainer exposes after init_workers
        self.actor_wg = SimpleNamespace(workers=[object(), object()])
        self.train_role = "actor"


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
    monkeypatch.delenv(hooks_mod.ENV_APP_OFFLOAD, raising=False)
    monkeypatch.delenv(hooks_mod.ENV_AGENT_ADDR, raising=False)
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
# acquire timeout
# ======================================================================


def make_locks(events):
    return locks_mod.PhaseLocks(
        job_id=JOB,
        orch_addr=ADDR,
        group=GROUP,
        client_factory=lambda target, job_id, group_id: FakeClient(events, job_id, group_id),
    )


class TestAcquireTimeout:
    def test_default_timeout_passed_to_client(self, events):
        pl = make_locks(events)
        pl.ensure()
        assert pl._client.last_timeout_sec == locks_mod.PhaseLocks.DEFAULT_ACQUIRE_TIMEOUT_SEC == 600

    def test_explicit_timeout_passed_to_client(self, events):
        pl = make_locks(events)
        pl.ensure(timeout_sec=42)
        assert pl._client.last_timeout_sec == 42

    def test_already_held_skips_client_call(self, events):
        pl = make_locks(events)
        pl.ensure()
        pl.ensure(timeout_sec=1)  # held: no second acquire, timeout untouched
        assert events == [("acquire", GROUP)]
        assert pl._client.last_timeout_sec == 600

    def test_client_timeout_error_propagates(self, events):
        pl = make_locks(events)

        def timed_out_acquire(group_id, timeout_sec=None):
            raise TimeoutError(f"lock not granted within {timeout_sec}s")

        pl._client.acquire = timed_out_acquire
        with pytest.raises(TimeoutError):
            pl.ensure()
        assert not pl.held  # never marked held on a failed acquire

    def test_timeout_error_propagates_through_hooks(self, ts_env, events):
        t = make_trainer(events)

        def timed_out_acquire(group_id, timeout_sec=None):
            raise TimeoutError("lock not granted within 600s")

        # Build the lock layer, then make the next acquire time out.
        run_init_workers(t, events)
        t._locks._client.acquire = timed_out_acquire
        with pytest.raises(TimeoutError):
            asyncio.run(t.on_sample_end())


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

        def acquire(self, group_id, timeout_sec=None):
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


# ======================================================================
# app-channel offload (TIMESLICE_APP_OFFLOAD) — stubbed relay, no ray/grpc
# ======================================================================

AGENT = "10.0.0.1:9001"


class FakeOffloadSpec:
    """Stand-in for timeslice.snapshot_agent.rayworker.OffloadSpec."""

    def __init__(self, method, snapshot_kwargs=None, restore_kwargs=None, **extra):
        self.method = method
        self.snapshot_kwargs = snapshot_kwargs
        self.restore_kwargs = restore_kwargs
        self.extra = extra


@pytest.fixture
def stub_rayworker(monkeypatch):
    """Install a stub timeslice.snapshot_agent.rayworker so offload.py's lazy
    imports resolve without the client package (or ray) installed."""
    spawns = []

    def spawn_offload_proxies(handles, spec, **kwargs):
        spawns.append((list(handles), spec, kwargs))
        return SimpleNamespace(nodes=["node-1"], actors=[], close=lambda timeout=5.0: None)

    for name in ("timeslice", "timeslice.snapshot_agent"):
        mod = types.ModuleType(name)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)
    rw = types.ModuleType("timeslice.snapshot_agent.rayworker")
    rw.OffloadSpec = FakeOffloadSpec
    rw.spawn_offload_proxies = spawn_offload_proxies
    monkeypatch.setitem(sys.modules, "timeslice.snapshot_agent.rayworker", rw)
    return SimpleNamespace(spawns=spawns, module=rw)


@pytest.fixture
def offload_env(ts_env, stub_rayworker, monkeypatch):
    monkeypatch.setenv(hooks_mod.ENV_APP_OFFLOAD, "1")
    monkeypatch.setenv(hooks_mod.ENV_AGENT_ADDR, AGENT)
    return ts_env


class FakeProxySet:
    def __init__(self, events, fail_close=False):
        self.events = events
        self.nodes = ["node-1"]
        self.actors = []
        self.fail_close = fail_close

    def close(self, timeout=5.0):
        self.events.append(("close_proxies",))
        if self.fail_close:
            raise RuntimeError("ray is gone")


def make_offload_trainer(events, fail_spawn=None, fail_close=False):
    def spawner(handles, spec, *, job_id, group):
        events.append(("spawn_proxies", spec.method, len(handles), job_id, group))
        if fail_spawn is not None:
            raise fail_spawn
        return FakeProxySet(events, fail_close=fail_close)

    return FakeTrainer(
        client_factory=lambda target, job_id, group_id: FakeClient(events, job_id, group_id),
        proxy_spawner=spawner,
    )


class RoleLike:
    """verl's Role enum stringifies to the worker prefix ("actor", "actor_rollout")."""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class TestAppOffload:
    def test_inert_without_flag(self, ts_env, events):
        t = make_offload_trainer(events)
        run_init_workers(t, events)
        assert events == [("acquire", GROUP), "orig_init_workers", ("release", GROUP)]
        assert t._offload_proxies is None

    def test_proxies_spawned_before_init_workers_yield(self, offload_env, events):
        t = make_offload_trainer(events)
        run_init_workers(t, events)
        assert events == [
            ("acquire", GROUP),
            "orig_init_workers",
            ("spawn_proxies", "actor_to", 2, JOB, GROUP),
            ("release", GROUP, "app_channel"),  # the init yield already names the backend
        ]
        assert t._offload_proxies is not None

    def test_yields_with_app_channel_after_proxies_spawn(self, offload_env, events):
        t = make_offload_trainer(events)
        run_init_workers(t, events)
        assert t._locks.backend_config == "app_channel"
        run_sample_end(t, events)
        run_update_weights(t, events, {"t": 1})
        releases = [e for e in events if e[0] == "release"]
        assert releases == [("release", GROUP, "app_channel")] * 2  # every later release repeats it

    def test_degraded_paths_yield_without_backend_config(self, offload_env, events, monkeypatch):
        monkeypatch.delenv(hooks_mod.ENV_AGENT_ADDR)
        t = make_offload_trainer(events)
        run_init_workers(t, events)
        assert events[-1] == ("release", GROUP)
        assert t._locks.backend_config is None

    def test_crash_close_resets_backend_config(self, offload_env, events):
        t = make_offload_trainer(events)
        run_init_workers(t, events)
        asyncio.run(t.on_exception("fit", RuntimeError("boom")))
        assert t._locks.backend_config is None
        run_sample_end(t, events)
        del events[:]
        run_update_weights(t, events, {"t": 1})
        assert events[-1] == ("release", GROUP)  # no channel left: plain release

    def test_spec_method_derived_from_train_role(self, offload_env, events):
        t = make_offload_trainer(events)
        t.train_role = RoleLike("actor_rollout")
        run_init_workers(t, events)
        assert ("spawn_proxies", "actor_rollout_to", 2, JOB, GROUP) in events

        spec = offload_mod.offload_spec_for_role(RoleLike("actor"))
        assert spec.method == "actor_to"
        assert spec.snapshot_kwargs == {"device": "cpu", "model": True, "optimizer": True, "grad": True}
        assert spec.restore_kwargs == {"device": "device", "model": True, "optimizer": True, "grad": False}
        with pytest.raises(ValueError):
            offload_mod.offload_spec_for_role("")

    def test_missing_agent_addr_warns_once_and_noops(self, offload_env, events, monkeypatch):
        monkeypatch.delenv(hooks_mod.ENV_AGENT_ADDR)
        t = make_offload_trainer(events)
        run_init_workers(t, events)
        run_init_workers(t, events)
        assert [e for e in events if e[0] == "spawn_proxies"] == []
        assert "app_offload_misconfigured" in t._warned
        assert events.count(("release", GROUP)) == 2  # the lock protocol is unaffected

    def test_missing_job_id_warns_once(self, offload_env, events, monkeypatch):
        monkeypatch.delenv("TIMESLICE_JOB_ID")
        t = make_offload_trainer(events)
        run_init_workers(t, events)
        assert [e for e in events if e[0] == "spawn_proxies"] == []
        assert "app_offload_misconfigured" in t._warned

    def test_missing_workers_warns_once(self, offload_env, events):
        t = make_offload_trainer(events)
        t.actor_wg = None
        run_init_workers(t, events)
        assert events == [("acquire", GROUP), "orig_init_workers", ("release", GROUP)]
        assert "app_offload_no_workers" in t._warned

    def test_spawn_failure_propagates_and_lock_released_on_exception(self, offload_env, events):
        t = make_offload_trainer(events, fail_spawn=TimeoutError("proxies did not connect"))
        with pytest.raises(TimeoutError):
            run_init_workers(t, events)
        # still holding the lock: verl's catch-all fires on_exception next
        assert events == [("acquire", GROUP), "orig_init_workers", ("spawn_proxies", "actor_to", 2, JOB, GROUP)]
        asyncio.run(t.on_exception("init_workers", TimeoutError()))
        assert events[-1] == ("release", GROUP)
        assert ("close_proxies",) not in events  # nothing was created
        assert t._offload_proxies is None

    def test_on_exception_closes_proxies(self, offload_env, events):
        t = make_offload_trainer(events)
        run_init_workers(t, events)
        asyncio.run(t.on_exception("fit", RuntimeError("boom")))
        assert events[-1] == ("close_proxies",)
        assert t._offload_proxies is None

    def test_on_exception_never_raises_when_close_fails(self, offload_env, events):
        t = make_offload_trainer(events, fail_close=True)
        run_init_workers(t, events)
        run_sample_end(t, events)  # hold the lock so the crash-release has work to do
        asyncio.run(t.on_exception("fit", RuntimeError("boom")))  # must not raise
        assert ("close_proxies",) in events
        assert not t._locks.held

    def test_spawn_is_idempotent(self, offload_env, events):
        t = make_offload_trainer(events)
        run_init_workers(t, events)
        run_init_workers(t, events)
        assert len([e for e in events if e[0] == "spawn_proxies"]) == 1

    def test_default_spawner_lazy_imports_client(self, stub_rayworker):
        spec = FakeOffloadSpec("actor_to")
        offload_mod.default_proxy_spawner(["h0", "h1"], spec, job_id="j", group="g")
        (handles, got_spec, kwargs) = stub_rayworker.spawns[0]
        assert handles == ["h0", "h1"] and got_spec is spec
        assert kwargs["agent"] is None  # resolved per node inside the proxy
        assert (kwargs["job_id"], kwargs["group"]) == ("j", "g")
        assert kwargs["ready_timeout"] == offload_mod.OFFLOAD_READY_TIMEOUT_SEC
        assert "log" not in kwargs  # nothing from the driver is shipped into the proxies


class TestBackendConfigOnRelease:
    """PhaseLocks passes backend_config to release() only when set, so
    orchestrator clients without the kwarg keep working."""

    def test_passed_only_when_set(self, events):
        locks = locks_mod.PhaseLocks(
            JOB, ADDR, GROUP, client_factory=lambda target, job_id, group_id: FakeClient(events, job_id, group_id)
        )
        locks.ensure()
        locks.drop_all()
        assert events[-1] == ("release", GROUP)
        locks.backend_config = offload_mod.YIELD_BACKEND_CONFIG
        locks.ensure()
        locks.drop_all()
        assert events[-1] == ("release", GROUP, "app_channel")

    def test_old_client_without_kwarg_still_works_when_unset(self, events):
        class OldClient(FakeClient):
            def release(self, group_id):  # no backend_config parameter at all
                self.events.append(("release", group_id))
                return FakeYieldResult()

        locks = locks_mod.PhaseLocks(
            JOB, ADDR, GROUP, client_factory=lambda target, job_id, group_id: OldClient(events, job_id, group_id)
        )
        locks.ensure()
        locks.drop_all()
        assert events == [("acquire", GROUP), ("release", GROUP)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
