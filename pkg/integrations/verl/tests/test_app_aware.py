"""Unit tests for timeslice_verl.app_aware and its hooks wiring (pure python:
no GPU, no verl, no ray, no grpc)."""

import asyncio
import os
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import timeslice_verl.hooks as hooks_mod  # noqa: E402
from timeslice_verl import app_aware  # noqa: E402
from timeslice_verl.hooks import TimesliceHooksMixin  # noqa: E402

JOB, ADDR, GROUP = "job-a", "orch:50051", "ns.job-a.trainer"


class FakeWG:
    """Stands in for verl's actor worker group (manual offload control)."""

    def __init__(self, events):
        self.events = events
        self.workers = ["w0", "w1"]

    def to(self, device, model=True, optimizer=True, grad=True):
        self.events.append(("to", device, model, optimizer, grad))


class FakeClient:
    def __init__(self, events, during_release=None):
        self.events = events
        # Runs inside the Yield RPC, before it returns: the orchestrator may
        # lend (and have the agent park the trainer) while Yield is in flight.
        self.during_release = during_release

    def acquire(self, group_id, timeout_sec=None):
        self.events.append(("acquire", group_id))
        return SimpleNamespace(success=True, waited_ms=1, context_restored=True)

    def release(self, group_id, **_kw):
        self.events.append(("release", group_id))
        if self.during_release is not None:
            self.during_release()
        return SimpleNamespace(success=True, pending_waiters=0, snapshot_deferred=False)

    def close(self):
        pass


class FakeTrainer(TimesliceHooksMixin):
    def __init__(self, events):
        super().__init__(client_factory=lambda target, job_id, group_id: FakeClient(events))
        self.config = SimpleNamespace(trainer=SimpleNamespace(save_freq=-1))
        self.actor_wg = FakeWG(events)


@pytest.fixture
def events():
    return []


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("TIMESLICE_FULLY_ASYNC", "1")
    monkeypatch.setenv("TIMESLICE_JOB_ID", JOB)
    monkeypatch.setenv("TIMESLICE_ORCH_ADDR", ADDR)
    monkeypatch.setenv("TIMESLICE_GROUP", GROUP)
    monkeypatch.setenv("TIMESLICE_DEVICE", "cuda")
    monkeypatch.delenv(app_aware.ENV_AGENT_ADDRS, raising=False)
    monkeypatch.delenv(app_aware.ENV_AGENT_PORT, raising=False)
    monkeypatch.delenv(app_aware.ENV_DONOR_BACKEND, raising=False)


# ----------------------------------------------------------------- TrainerPark


def test_park_offloads_optimizer_then_model_and_restores(env, events):
    park = app_aware.TrainerPark(FakeWG(events), is_safe=lambda: True)
    park.snapshot("offload", [])
    assert park.parked
    assert events == [("to", "cpu", False, True, False), ("to", "cpu", True, False, True)]
    park.snapshot("offload", [])  # repeated park: no-op
    assert len(events) == 2
    park.restore([])
    assert not park.parked
    assert events[-1] == ("to", "cuda", True, True, True)
    park.restore([])  # repeated restore: no-op
    assert len(events) == 3
    assert (park.parks, park.restores, park.self_restores) == (1, 1, 0)


def test_restore_defaults_to_verls_device_alias(env, events, monkeypatch):
    # verl's worker-level to() accepts only "cpu" and "device" (its alias for the accelerator).
    monkeypatch.delenv("TIMESLICE_DEVICE", raising=False)
    park = app_aware.TrainerPark(FakeWG(events), is_safe=lambda: True)
    park.snapshot("offload", [])
    park.restore([])
    assert events[-1] == ("to", "device", True, True, True)


def test_park_frees_worker_caches_after_the_offload(env, events, capsys):
    def free_cache(wg):
        events.append(("free", tuple(wg.workers)))
        return [{"torch_reserved": 0, "cupy_before": 4 << 30, "cupy_after": 0}, {}]

    park = app_aware.TrainerPark(FakeWG(events), is_safe=lambda: True, free_cache=free_cache)
    park.snapshot("offload", [])
    assert events == [("to", "cpu", False, True, False), ("to", "cpu", True, False, True), ("free", ("w0", "w1"))]
    out = capsys.readouterr().out
    assert "cupy_before" in out


def test_park_survives_a_failing_cache_free(env, events):
    def free_cache(_wg):
        raise RuntimeError("worker gone")

    park = app_aware.TrainerPark(FakeWG(events), is_safe=lambda: True, free_cache=free_cache)
    park.snapshot("offload", [])  # the agent's residual check decides; the park itself is done
    assert park.parked


def test_trim_before_sync_runs_only_while_resident(env, events, capsys):
    def trim_cache(wg):
        events.append(("trim", tuple(wg.workers)))
        return [{"reserved_before": 70 << 30, "reserved_after": 23 << 30}]

    park = app_aware.TrainerPark(FakeWG(events), is_safe=lambda: True, trim_cache=trim_cache)
    park.trim_before_sync()
    assert events == [("trim", ("w0", "w1"))]
    assert "pre-sync torch cache trim" in capsys.readouterr().out
    park.snapshot("offload", [])
    del events[:]
    park.trim_before_sync()  # parked: the workers hold nothing on the GPU to trim
    assert events == []


def test_trim_before_sync_survives_a_failing_trim(env, events):
    def trim_cache(_wg):
        raise RuntimeError("worker gone")

    park = app_aware.TrainerPark(FakeWG(events), is_safe=lambda: True, trim_cache=trim_cache)
    park.trim_before_sync()  # the sync itself decides
    app_aware.TrainerPark(FakeWG(events), is_safe=lambda: True).trim_before_sync()  # no trim: no-op


def test_trim_torch_cache_empties_the_cache(monkeypatch):
    reserved = [48 << 30]
    cuda = SimpleNamespace(
        is_initialized=lambda: True,
        memory_reserved=lambda: reserved[0],
        empty_cache=lambda: reserved.__setitem__(0, 23 << 30),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    assert app_aware._trim_torch_cache(None) == {"reserved_before": 48 << 30, "reserved_after": 23 << 30}
    cuda.is_initialized = lambda: False
    assert app_aware._trim_torch_cache(None) == {}
    monkeypatch.setitem(sys.modules, "torch", None)
    assert "error" in app_aware._trim_torch_cache(None)


def test_free_device_cache_frees_an_imported_cupy_pool(monkeypatch):
    class Pool:
        def __init__(self):
            self.total = 4 << 30

        def total_bytes(self):
            return self.total

        def free_all_blocks(self):
            self.total = 0

    pool, pinned = Pool(), Pool()
    fake_cupy = SimpleNamespace(get_default_memory_pool=lambda: pool, get_default_pinned_memory_pool=lambda: pinned)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "torch", None)  # no torch: still frees cupy
    out = app_aware._free_device_cache(None)
    assert out["cupy_before"] == 4 << 30 and out["cupy_after"] == 0
    assert pinned.total == 0


def test_free_device_cache_does_not_import_cupy(monkeypatch):
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    out = app_aware._free_device_cache(None)
    assert "cupy_before" not in out and "cupy" not in sys.modules


class _FakeHandle:
    def __init__(self, calls, name, orig=False, sharded=False):
        self._calls, self._name = calls, name
        self._use_orig_params, self.uses_sharded_strategy = orig, sharded

    def _use_unsharded_views(self, as_params):
        self._calls.append((self._name, as_params))


def test_resync_fsdp_views_repoints_unsharded_handles(monkeypatch):
    # FSDP1 on one GPU runs NO_SHARD, and with use_orig_params=False the module keeps
    # Tensor views of the flat parameter outside forward: moving the flat parameter to
    # the CPU leaves the views (and the whole model) on the GPU until they are re-pointed.
    calls = []
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp._flat_param", SimpleNamespace(FlatParamHandle=_FakeHandle))
    keep = [_FakeHandle(calls, "a"), _FakeHandle(calls, "b", orig=True), _FakeHandle(calls, "c", sharded=True)]
    assert app_aware._resync_fsdp_views() == 1
    assert calls == [("a", False)]
    del keep


def test_resync_fsdp_views_without_fsdp_loaded(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch.distributed.fsdp._flat_param", raising=False)
    assert app_aware._resync_fsdp_views() == 0


def test_free_device_cache_resyncs_fsdp_views(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp._flat_param", SimpleNamespace(FlatParamHandle=_FakeHandle))
    monkeypatch.setitem(sys.modules, "torch", None)
    keep = _FakeHandle(calls, "a")
    out = app_aware._free_device_cache(None)
    assert out["fsdp_views"] == 1 and calls == [("a", False)]
    del keep


def test_restore_resyncs_views_after_the_load(env, events):
    def after_restore(wg):
        events.append(("resync", tuple(wg.workers)))

    park = app_aware.TrainerPark(FakeWG(events), is_safe=lambda: True, after_restore=after_restore)
    park.snapshot("offload", [])
    park.restore([])
    assert events[-2:] == [("to", "cuda", True, True, True), ("resync", ("w0", "w1"))]


def test_restore_survives_a_failing_resync(env, events):
    def after_restore(_wg):
        raise RuntimeError("worker gone")

    park = app_aware.TrainerPark(FakeWG(events), is_safe=lambda: True, after_restore=after_restore)
    park.snapshot("offload", [])
    park.restore([])  # the next forward re-points the views anyway
    assert not park.parked


def test_park_refused_outside_safe_point(env, events):
    park = app_aware.TrainerPark(FakeWG(events), is_safe=lambda: False)
    with pytest.raises(app_aware.NotAtSafePointError):
        park.snapshot("offload", [])
    assert events == [] and not park.parked


def test_park_refuses_discard(env, events):
    park = app_aware.TrainerPark(FakeWG(events), is_safe=lambda: True)
    with pytest.raises(ValueError):
        park.snapshot("discard", [])
    assert events == []


def test_ensure_resident_self_restores(env, events):
    park = app_aware.TrainerPark(FakeWG(events), is_safe=lambda: True)
    assert park.ensure_resident() is False
    park.snapshot("offload", [])
    assert park.ensure_resident() is True
    assert park.self_restores == 1 and not park.parked


def test_park_and_restore_serialize(env, events):
    """A restore arriving during a park waits for the park to finish."""
    entered, gate = threading.Event(), threading.Event()

    class SlowWG(FakeWG):
        def to(self, device, **kw):
            if device == "cpu" and kw.get("model"):
                entered.set()
                gate.wait(2)
            super().to(device, **kw)

    park = app_aware.TrainerPark(SlowWG(events), is_safe=lambda: True)
    t = threading.Thread(target=park.snapshot, args=("offload", []))
    t.start()
    assert entered.wait(2)
    r = threading.Thread(target=park.restore, args=([],))
    r.start()
    gate.set()
    t.join(2)
    r.join(2)
    assert [e[1] for e in events] == ["cpu", "cpu", "cuda"]
    assert not park.parked


# -------------------------------------------------------------- addresses


def test_agent_addresses_dedup_port_and_ipv6(env, monkeypatch):
    assert app_aware.agent_addresses(["10.0.0.1", "10.0.0.1", "10.0.0.2"]) == ["10.0.0.1:9001", "10.0.0.2:9001"]
    monkeypatch.setenv(app_aware.ENV_AGENT_PORT, "9100")
    assert app_aware.agent_addresses(["fd00::1"]) == ["[fd00::1]:9100"]
    monkeypatch.setenv(app_aware.ENV_AGENT_ADDRS, "a:1, b:2")
    assert app_aware.agent_addresses(["10.0.0.1"]) == ["a:1", "b:2"]


def test_node_ip_prefers_the_host_ip(monkeypatch):
    # In a KubeRay pod the Ray node IP is the pod IP; the agent listens on the host network,
    # so the worker reports the host IP the webhook injected (downward API status.hostIP).
    monkeypatch.setenv(app_aware.ENV_NODE_IP, "10.1.2.3")
    monkeypatch.setitem(sys.modules, "ray", None)  # must not need ray when the host IP is known
    assert app_aware._node_ip(None) == "10.1.2.3"


def test_start_registers_once_per_node(env):
    calls = []

    def register(**kw):
        calls.append(kw)
        return object()

    park, handles = app_aware.start(
        FakeWG([]), JOB, GROUP, is_safe=lambda: True, register=register, node_ips=lambda wg: ["n1", "n1", "n2"]
    )
    assert [c["agent"] for c in calls] == ["n1:9001", "n2:9001"]
    assert all(c["job_id"] == JOB and c["group"] == GROUP and c["workload"] is park for c in calls)
    assert all(c["supported_modes"] == ["offload"] for c in calls)
    assert len(handles) == 2


def test_enabled_gate(monkeypatch):
    monkeypatch.delenv(app_aware.ENV_DONOR_BACKEND, raising=False)
    assert not app_aware.enabled()
    monkeypatch.setenv(app_aware.ENV_DONOR_BACKEND, "app_channel")
    assert app_aware.enabled()
    monkeypatch.setenv(app_aware.ENV_DONOR_BACKEND, "cuda")
    assert not app_aware.enabled()


# ------------------------------------------------------------------- hooks


@pytest.fixture
def registered(monkeypatch):
    """Replace the agent registration and node discovery with fakes."""
    calls = []
    real_start = app_aware.start

    def fake_start(wg, job_id, group, is_safe):
        return real_start(
            wg, job_id, group, is_safe, register=lambda **kw: calls.append(kw), node_ips=lambda _wg: ["10.0.0.1"]
        )

    monkeypatch.setattr(hooks_mod.app_aware, "start", fake_start)
    return calls


def test_hooks_cuda_default_registers_nothing(env, events, registered):
    t = FakeTrainer(events)

    async def seq():
        await t.on_init_workers_begin()
        await t.on_init_workers_end()

    asyncio.run(seq())
    assert registered == [] and t._park is None


def test_hooks_app_channel_park_only_while_yielded(env, events, registered, monkeypatch):
    monkeypatch.setenv(app_aware.ENV_DONOR_BACKEND, "app_channel")
    t = FakeTrainer(events)

    async def init():
        await t.on_init_workers_begin()
        await t.on_init_workers_end()

    asyncio.run(init())
    assert len(registered) == 1 and registered[0]["agent"] == "10.0.0.1:9001"
    assert registered[0]["job_id"] == JOB and registered[0]["group"] == GROUP
    park = t._park
    # Yielded after init: the agent may park the trainer.
    park.snapshot("offload", [])
    assert park.parked

    # The agent loses the restore (e.g. restarts): the next granted acquire
    # restores the trainer before any GPU work.
    asyncio.run(t.on_sample_end())
    assert not park.parked and park.self_restores == 1
    assert events.index(("acquire", GROUP), 1) < events.index(("to", "cuda", True, True, True))

    # Holding the lock: a park is refused.
    with pytest.raises(app_aware.NotAtSafePointError):
        park.snapshot("offload", [])

    async def update():
        await t.on_update_weights_begin()
        await t.on_update_weights_end(synced=True)

    asyncio.run(update())
    park.snapshot("offload", [])  # yielded again
    assert park.parked


def test_hooks_app_channel_trim_before_each_weight_sync(env, events, registered, monkeypatch):
    monkeypatch.setenv(app_aware.ENV_DONOR_BACKEND, "app_channel")
    monkeypatch.setattr(app_aware, "trim_worker_torch_caches", lambda wg: events.append("trim"))
    t = FakeTrainer(events)

    async def seq():
        await t.on_init_workers_begin()
        await t.on_init_workers_end()
        t._park.snapshot("offload", [])  # lent while yielded
        await t.on_update_weights_begin()  # pre-fit initial sync: acquire, restore, then trim
        await t.on_update_weights_end(synced=True)
        await t.on_sample_end()
        await t.on_update_weights_begin()  # steady state: lock already held
        await t.on_update_weights_end(synced=True)

    asyncio.run(seq())
    assert events.count("trim") == 2
    first = events.index("trim")
    assert events.index(("acquire", GROUP), 1) < events.index(("to", "cuda", True, True, True)) < first
    assert events[first + 1] == ("release", GROUP)


def test_hooks_app_channel_park_while_yield_rpc_in_flight(env, events, registered, monkeypatch):
    """The orchestrator lends inside the Yield RPC: the agent's park can arrive
    before Yield returns. The lock already counts as released then, so the
    park is granted instead of refused (a refusal FAULTs the job)."""
    monkeypatch.setenv(app_aware.ENV_DONOR_BACKEND, "app_channel")
    outcome = []

    def park_now():
        try:
            t._park.snapshot("offload", [])
            outcome.append("parked")
        except app_aware.NotAtSafePointError as e:
            outcome.append(e)

    client = FakeClient(events, during_release=park_now)
    t = FakeTrainer(events)
    t._client_factory = lambda target, job_id, group_id: client

    async def seq():
        await t.on_init_workers_begin()
        await t.on_init_workers_end()
        outcome.clear()  # the init yield also parks; check the update's yield
        await t.on_sample_end()  # holds the lock again (restores first)
        await t.on_update_weights_begin()
        await t.on_update_weights_end(synced=True)

    asyncio.run(seq())
    assert outcome == ["parked"]
    assert t._park.parked and not t._locks.held


def test_hooks_app_channel_registration_failure_releases_lock(env, events, monkeypatch):
    monkeypatch.setenv(app_aware.ENV_DONOR_BACKEND, "app_channel")

    def broken_start(*a, **kw):
        raise RuntimeError("no agent")

    monkeypatch.setattr(hooks_mod.app_aware, "start", broken_start)
    t = FakeTrainer(events)

    async def init():
        await t.on_init_workers_begin()
        await t.on_init_workers_end()

    with pytest.raises(RuntimeError):
        asyncio.run(init())
    assert events[-1] == ("release", GROUP)
