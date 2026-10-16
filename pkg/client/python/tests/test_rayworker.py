"""Tests for the Ray relay (timeslice.snapshot_agent.rayworker).

Ray is faked in sys.modules with a synchronous stand-in: "actors" are plain
objects, ``.method.remote(...)`` returns a lazily resolved ref, ``ray.get``
resolves refs (honouring a per-ref "blocking" flag that raises
GetTimeoutError), and ``ray.remote``/``.options`` record how the proxy actor
class was created. The agent side is the real in-process gRPC FakeAgent, so a
command travels agent -> proxy -> RankFanout -> fake rank actors end to end.
"""

import importlib.machinery
import importlib.util
import os
import sys
import types
import unittest
from concurrent import futures
from unittest import mock

import grpc

from timeslice.snapshot_agent import snapshot_agent_pb2 as pb2
from timeslice.snapshot_agent import snapshot_agent_pb2_grpc as pb2_grpc
from timeslice.snapshot_agent import workload as workload_mod

try:  # pytest imports this file as a package module, unittest discover as a top-level one
    from ._fake_agent import FakeAgent
except ImportError:
    from _fake_agent import FakeAgent

NODE_A = "aaaaaaaa-node-a"
NODE_B = "bbbbbbbb-node-b"


# ---------------------------------------------------------------- fake ray


class FakeRay:
    """Builds the ``ray`` module tree used by rayworker, recording activity."""

    def __init__(self):
        self.current_node = None
        self.options_calls = []
        self.remote_decorations = []
        self.killed = []
        self.last_get_timeout = None
        fake = self

        class GetTimeoutError(Exception):
            pass

        class FakeRef:
            def __init__(self, fn, blocking=False):
                self.fn = fn
                self.blocking = blocking

            def resolve(self, timeout):
                if self.blocking:
                    raise GetTimeoutError(f"Get timed out after {timeout}s")
                return self.fn()

        class FakeMethod:
            def __init__(self, fn, on_call=None, blocking=False):
                self.fn = fn
                self.on_call = on_call
                self.blocking = blocking

            def remote(self, *args, **kwargs):
                if self.on_call:
                    self.on_call(*args, **kwargs)
                return FakeRef(lambda: self.fn(*args, **kwargs), blocking=self.blocking)

        class FakeActorHandle:
            def __init__(self, inst):
                self._inst = inst

            def __getattr__(self, name):
                if name.startswith("_"):
                    raise AttributeError(name)
                return FakeMethod(getattr(self._inst, name))

        class FakeActorClass:
            def __init__(self, cls, opts):
                self.cls = cls
                self.opts = opts

            def options(self, **opts):
                fake.options_calls.append(opts)
                return self

            def remote(self, *args, **kwargs):
                return FakeActorHandle(self.cls(*args, **kwargs))

        def remote(**opts):
            def decorate(cls):
                fake.remote_decorations.append(opts)
                return FakeActorClass(cls, opts)

            return decorate

        def get(refs, timeout=None):
            fake.last_get_timeout = timeout
            if isinstance(refs, list):
                return [r.resolve(timeout) for r in refs]
            return refs.resolve(timeout)

        def kill(handle, no_restart=False):
            fake.killed.append(handle)

        class RuntimeContext:
            def get_node_id(self):
                return fake.current_node

        class NodeAffinitySchedulingStrategy:
            def __init__(self, node_id, soft=False):
                self.node_id = node_id
                self.soft = soft

        ray = types.ModuleType("ray")
        ray.remote = remote
        ray.get = get
        ray.kill = kill
        ray.get_runtime_context = lambda: RuntimeContext()
        exceptions = types.ModuleType("ray.exceptions")
        exceptions.GetTimeoutError = GetTimeoutError
        util = types.ModuleType("ray.util")
        strategies = types.ModuleType("ray.util.scheduling_strategies")
        strategies.NodeAffinitySchedulingStrategy = NodeAffinitySchedulingStrategy
        util.scheduling_strategies = strategies
        ray.exceptions = exceptions
        ray.util = util
        self.modules = {
            "ray": ray,
            "ray.exceptions": exceptions,
            "ray.util": util,
            "ray.util.scheduling_strategies": strategies,
        }
        for name, mod in self.modules.items():
            # A real-looking spec keeps importlib.util.find_spec("ray") working
            # elsewhere while the fake sits in sys.modules.
            mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        self.FakeMethod = FakeMethod
        self.GetTimeoutError = GetTimeoutError

    def worker(self, node_id, rank, calls, fail=None, block=False):
        return FakeWorker(self, node_id, rank, calls, fail=fail, block=block)


# rayworker imports ray at module level; stand in for it when ray is not
# installed. Each test then rebinds rayworker's ray names to its own FakeRay.
if importlib.util.find_spec("ray") is None:
    sys.modules.update(FakeRay().modules)

from timeslice.snapshot_agent import rayworker  # noqa: E402
from timeslice.snapshot_agent.rayworker import (  # noqa: E402
    OffloadSpec,
    RankFanout,
    group_handles_by_node,
    spawn_offload_proxies,
)


class FakeWorker:
    """A rank actor handle: ``__ray_call__`` runs a function "inside" it, any
    other method records ``(rank, name, kwargs)`` and optionally fails/blocks."""

    def __init__(self, fake, node_id, rank, calls, fail=None, block=False):
        self._fake = fake
        self.node_id = node_id
        self.rank = rank
        self._calls = calls
        self._fail = fail
        self._block = block

        def ray_call(fn, *args, **kwargs):
            fake.current_node = node_id
            return fn(self, *args, **kwargs)

        self.__ray_call__ = fake.FakeMethod(ray_call)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def invoke(**kwargs):
            if self._fail:
                raise self._fail
            return None

        def record(**kwargs):
            self._calls.append((self.rank, name, dict(kwargs)))

        return self._fake.FakeMethod(invoke, on_call=record, blocking=self._block)


SPEC = OffloadSpec(
    method="actor_to",
    snapshot_kwargs={"device": "cpu", "model": True, "optimizer": True, "grad": True},
    restore_kwargs={
        "device": "device",
        "model": True,
        "optimizer": True,
        "grad": False,
    },
    call_timeout_sec=7.5,
)


class RayWorkerTestBase(unittest.TestCase):
    def setUp(self):
        self.fake = FakeRay()
        ray_mod = self.fake.modules["ray"]
        bindings = {
            "ray": ray_mod,
            "NodeAffinitySchedulingStrategy": ray_mod.util.scheduling_strategies.NodeAffinitySchedulingStrategy,
            "OffloadProxyActor": ray_mod.remote(num_cpus=0)(rayworker.OffloadProxy),
        }
        for name, value in bindings.items():
            patcher = mock.patch.object(rayworker, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.calls = []

    def start_agent(self):
        self.agent = FakeAgent()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        pb2_grpc.add_SnapshotAgentServiceServicer_to_server(self.agent, self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.addCleanup(self.server.stop, 0)
        return f"127.0.0.1:{port}"

    def workers(self, layout, **kw):
        """layout: list of node ids, index = rank."""
        return [
            self.fake.worker(node, rank, self.calls, **kw)
            for rank, node in enumerate(layout)
        ]

    def spawn(self, handles, spec=SPEC, **kwargs):
        kwargs.setdefault("job_id", "test-job")
        kwargs.setdefault("group", "trainers")
        kwargs.setdefault("ready_timeout", 10.0)
        proxies = spawn_offload_proxies(handles, spec, **kwargs)
        self.addCleanup(proxies.close)
        return proxies

    @staticmethod
    def snapshot_cmd(cid="c1", mode=pb2.SUSPEND_MODE_OFFLOAD, tags=()):
        return pb2.AgentCommand(
            command_id=cid, snapshot=pb2.SnapshotCommand(mode=mode, tags=list(tags))
        )

    @staticmethod
    def restore_cmd(cid="c2", tags=()):
        return pb2.AgentCommand(
            command_id=cid, restore=pb2.RestoreCommand(tags=list(tags))
        )

    def send(self, stream, cmd):
        stream.commands.put(cmd)
        return stream.results.get(timeout=10)


# ---------------------------------------------------------------- tests


class OffloadSpecTest(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            OffloadSpec(method="")
        with self.assertRaises(ValueError):
            OffloadSpec(method="to", supported_modes=("hibernate",))
        with self.assertRaises(ValueError):
            OffloadSpec(
                method="to", supported_modes=("offload",), default_mode="discard"
            )
        with self.assertRaises(ValueError):
            OffloadSpec(method="to", call_timeout_sec=0)
        spec = OffloadSpec(method="to", restore_method="back")
        self.assertEqual(
            (spec.supported_modes, spec.default_mode, spec.restore_method),
            (("offload",), "offload", "back"),
        )


class GroupingTest(RayWorkerTestBase):
    def test_groups_handles_by_node_in_rank_order(self):
        handles = self.workers([NODE_A, NODE_B, NODE_A, NODE_B])
        groups = group_handles_by_node(handles)
        self.assertEqual(list(groups), [NODE_A, NODE_B])
        self.assertEqual([h.rank for h in groups[NODE_A]], [0, 2])
        self.assertEqual([h.rank for h in groups[NODE_B]], [1, 3])


class RankFanoutTest(RayWorkerTestBase):
    def test_snapshot_and_restore_use_spec_kwargs(self):
        handles = self.workers([NODE_A, NODE_A])
        fan = RankFanout(handles, SPEC, job_id="j", node="n")
        fan.snapshot("offload", [])
        fan.restore(["weights"])
        self.assertEqual(
            self.calls,
            [
                (0, "actor_to", SPEC.snapshot_kwargs),
                (1, "actor_to", SPEC.snapshot_kwargs),
                (0, "actor_to", SPEC.restore_kwargs),
                (1, "actor_to", SPEC.restore_kwargs),
            ],
        )
        self.assertEqual(self.fake.last_get_timeout, SPEC.call_timeout_sec)

    def test_restore_method_override(self):
        spec = OffloadSpec(method="offload", restore_method="onload")
        fan = RankFanout(self.workers([NODE_A]), spec)
        fan.snapshot("offload", [])
        fan.restore([])
        self.assertEqual([c[1] for c in self.calls], ["offload", "onload"])

    def test_unsupported_mode_rejected_locally(self):
        fan = RankFanout(self.workers([NODE_A]), SPEC)
        with self.assertRaises(ValueError):
            fan.snapshot("discard", [])
        self.assertEqual(self.calls, [])

    def test_counters_advance_only_on_success(self):
        fan = RankFanout(self.workers([NODE_A]), SPEC)
        self.assertEqual((fan.snapshots, fan.restores), (0, 0))
        fan.snapshot("offload", [])
        fan.restore([])
        fan.snapshot("offload", [])
        self.assertEqual((fan.snapshots, fan.restores), (2, 1))
        failing = RankFanout(self.workers([NODE_A], fail=RuntimeError("no")), SPEC)
        with self.assertRaises(RuntimeError):
            failing.snapshot("offload", [])
        self.assertEqual(failing.snapshots, 0)


class SpawnTest(RayWorkerTestBase):
    def test_spawns_one_pinned_zero_cpu_proxy_per_node(self):
        address = self.start_agent()
        proxies = self.spawn(
            self.workers([NODE_A, NODE_B, NODE_A, NODE_B]), agent=address
        )
        self.assertEqual(proxies.nodes, [NODE_A, NODE_B])
        self.assertEqual(len(proxies.actors), 2)
        self.assertEqual(self.fake.remote_decorations, [{"num_cpus": 0}])
        self.assertEqual(len(self.fake.options_calls), 2)
        for opts, node in zip(self.fake.options_calls, [NODE_A, NODE_B]):
            self.assertEqual(
                opts["name"], f"timeslice-offload-proxy-test-job-{node[:8]}"
            )
            self.assertNotIn("get_if_exists", opts)  # never adopt a stale proxy
            self.assertEqual(opts["num_cpus"], 0)
            self.assertEqual(
                (opts["scheduling_strategy"].node_id, opts["scheduling_strategy"].soft),
                (node, False),
            )
        infos = proxies.info()
        self.assertEqual(sorted(i["ranks"] for i in infos), [2, 2])
        self.assertTrue(all(i["connected"] for i in infos))

    def test_registration_advertises_offload_only(self):
        address = self.start_agent()
        self.spawn(self.workers([NODE_A]), agent=address)
        streams = self.agent.wait_streams(1)
        reg = streams[0].register
        self.assertEqual((reg.job_id, reg.group), ("test-job", "trainers"))
        self.assertEqual(
            list(reg.capabilities.supported_modes), [pb2.SUSPEND_MODE_OFFLOAD]
        )
        self.assertEqual(reg.capabilities.default_mode, pb2.SUSPEND_MODE_OFFLOAD)

    def test_snapshot_fans_out_spec_kwargs_to_local_ranks_only(self):
        address = self.start_agent()
        self.spawn(self.workers([NODE_A, NODE_B, NODE_A, NODE_B]), agent=address)
        streams = self.agent.wait_streams(2)
        self.assertEqual(len(streams), 2)

        seen = []
        for i, stream in enumerate(streams):
            del self.calls[:]
            result = self.send(stream, self.snapshot_cmd(cid=f"s{i}"))
            self.assertTrue(result.ok, result.error)
            ranks = [c[0] for c in self.calls]
            self.assertIn(
                ranks,
                ([0, 2], [1, 3]),
                "each proxy must hit exactly its node's ranks, in rank order",
            )
            self.assertTrue(
                all(c[1:] == ("actor_to", SPEC.snapshot_kwargs) for c in self.calls)
            )
            seen.append(tuple(ranks))
        self.assertEqual(sorted(seen), [(0, 2), (1, 3)])

    def test_info_reports_park_and_restore_counts(self):
        address = self.start_agent()
        proxies = self.spawn(self.workers([NODE_A]), agent=address)
        stream = self.agent.wait_streams(1)[0]
        self.assertEqual(
            (proxies.info()[0]["snapshots"], proxies.info()[0]["restores"]), (0, 0)
        )
        self.assertTrue(self.send(stream, self.snapshot_cmd()).ok)
        info = proxies.info()[0]
        self.assertEqual((info["snapshots"], info["restores"]), (1, 0))
        self.assertTrue(self.send(stream, self.restore_cmd()).ok)
        info = proxies.info()[0]
        self.assertEqual((info["snapshots"], info["restores"]), (1, 1))

    def test_unsupported_mode_reported_through_channel(self):
        address = self.start_agent()
        self.spawn(self.workers([NODE_A]), agent=address)
        stream = self.agent.wait_streams(1)[0]
        result = self.send(stream, self.snapshot_cmd(mode=pb2.SUSPEND_MODE_DISCARD))
        self.assertFalse(result.ok)
        self.assertIn("unsupported suspend mode", result.error)
        self.assertEqual(self.calls, [])

    def test_rank_timeout_surfaces_as_command_error(self):
        address = self.start_agent()
        self.spawn(self.workers([NODE_A, NODE_A], block=True), agent=address)
        stream = self.agent.wait_streams(1)[0]
        result = self.send(stream, self.snapshot_cmd())
        self.assertFalse(result.ok)
        self.assertIn("GetTimeoutError", result.error)
        self.assertEqual(self.fake.last_get_timeout, SPEC.call_timeout_sec)

    def test_rank_exception_surfaces_as_command_error(self):
        address = self.start_agent()
        self.spawn(
            self.workers([NODE_A], fail=RuntimeError("CUDA OOM on offload")),
            agent=address,
        )
        stream = self.agent.wait_streams(1)[0]
        result = self.send(stream, self.snapshot_cmd())
        self.assertFalse(result.ok)
        self.assertIn("CUDA OOM on offload", result.error)

    def test_close_deregisters_and_kills(self):
        address = self.start_agent()
        proxies = spawn_offload_proxies(
            self.workers([NODE_A]), SPEC, job_id="test-job", agent=address
        )
        stream = self.agent.wait_streams(1)[0]
        proxies.close()
        self.assertTrue(stream.closed.wait(timeout=10))
        self.assertEqual(len(self.fake.killed), 1)

    def test_spawn_raises_when_agent_unreachable(self):
        original = workload_mod._CONNECT_TIMEOUT_SEC
        workload_mod._CONNECT_TIMEOUT_SEC = 0.2
        self.addCleanup(setattr, workload_mod, "_CONNECT_TIMEOUT_SEC", original)
        with self.assertRaises(TimeoutError) as ctx:
            spawn_offload_proxies(
                self.workers([NODE_A, NODE_B]),
                SPEC,
                job_id="test-job",
                agent="127.0.0.1:1",
                ready_timeout=0.5,
            )
        self.assertIn(NODE_A[:8], str(ctx.exception))
        self.assertEqual(
            len(self.fake.killed), 2, "every started proxy is torn down on failure"
        )

    def test_spawn_validates_inputs(self):
        with self.assertRaises(ValueError):
            spawn_offload_proxies([], SPEC, job_id="j")
        with self.assertRaises(ValueError):
            spawn_offload_proxies(self.workers([NODE_A]), SPEC, job_id="")


class ProxyEnvTest(RayWorkerTestBase):
    def test_agent_from_proxy_env(self):
        address = self.start_agent()
        with mock.patch.dict(os.environ, {rayworker.ENV_AGENT_ADDR: address}):
            proxy = rayworker.OffloadProxy("env-job", "g", self.workers([NODE_A]), SPEC)
        self.addCleanup(proxy.close)
        self.assertTrue(proxy.ready(timeout=10))
        self.assertEqual(proxy.info()["agent"], address)
        self.assertTrue(self.agent.registered.wait(timeout=10))

    def test_missing_agent_address_is_an_error(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(rayworker.ENV_AGENT_ADDR, None)
            with self.assertRaises(ValueError):
                rayworker.OffloadProxy("env-job", "g", self.workers([NODE_A]), SPEC)


if __name__ == "__main__":
    unittest.main()
