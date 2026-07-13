import queue
import threading
import unittest
from concurrent import futures

import grpc

from timeslice.snapshot_agent import register_workload
from timeslice.snapshot_agent import snapshot_agent_pb2 as pb2
from timeslice.snapshot_agent import snapshot_agent_pb2_grpc as pb2_grpc
from timeslice.snapshot_agent.adapters import resolve_adapter

_END = object()


class FakeAgent(pb2_grpc.SnapshotAgentServiceServicer):
    """In-process agent: records registrations, pushes queued commands,
    collects results. drop_streams ends that many streams right after
    registration to exercise reconnect."""

    def __init__(self, drop_streams=0):
        self.registrations = []
        self.registered = threading.Event()
        self.results = queue.Queue()
        self.commands = queue.Queue()
        self.drop_streams = drop_streams

    def WorkloadChannel(self, request_iterator, context):  # noqa: N802 - grpc method name
        first = next(request_iterator)
        self.registrations.append(first.register)
        self.registered.set()
        if self.drop_streams > 0:
            self.drop_streams -= 1
            return

        def read_results():
            for msg in request_iterator:
                self.results.put(msg.result)

        reader = threading.Thread(target=read_results, daemon=True)
        reader.start()
        # Poll so the worker thread is released when the test stops the
        # server (a blocking get would pin a non-daemon grpc thread forever).
        while context.is_active():
            try:
                cmd = self.commands.get(timeout=0.2)
            except queue.Empty:
                continue
            if cmd is _END:
                return
            yield cmd


class RecordingWorkload:
    """Snapshottable test double."""

    supported_modes = ["offload", "discard"]
    default_mode = "offload"

    def __init__(self, fail_with=None):
        self.calls = []
        self.fail_with = fail_with

    def snapshot(self, mode, tags):
        self.calls.append(("snapshot", mode, tags))
        if self.fail_with:
            raise self.fail_with

    def restore(self, tags):
        self.calls.append(("restore", tags))


class WorkloadChannelTest(unittest.TestCase):
    def start_agent(self, drop_streams=0):
        self.agent = FakeAgent(drop_streams=drop_streams)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        pb2_grpc.add_SnapshotAgentServiceServicer_to_server(self.agent, self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.addCleanup(self.server.stop, 0)
        return f"127.0.0.1:{port}"

    def register(self, address, workload, **kwargs):
        handle = register_workload(
            address, job_id="test-job", group="g1", workload=workload, **kwargs
        )
        self.addCleanup(handle.close)
        return handle

    def send_and_wait(self, cmd):
        self.agent.commands.put(cmd)
        return self.agent.results.get(timeout=10)

    def test_registration_carries_capabilities(self):
        address = self.start_agent()
        self.register(address, RecordingWorkload())
        self.assertTrue(self.agent.registered.wait(timeout=10))
        reg = self.agent.registrations[0]
        self.assertEqual(reg.job_id, "test-job")
        self.assertEqual(reg.group, "g1")
        self.assertEqual(
            list(reg.capabilities.supported_modes),
            [pb2.SUSPEND_MODE_OFFLOAD, pb2.SUSPEND_MODE_DISCARD],
        )
        self.assertEqual(reg.capabilities.default_mode, pb2.SUSPEND_MODE_OFFLOAD)

    def test_snapshot_restore_round_trip(self):
        address = self.start_agent()
        workload = RecordingWorkload()
        self.register(address, workload)
        self.assertTrue(self.agent.registered.wait(timeout=10))

        result = self.send_and_wait(
            pb2.AgentCommand(
                command_id="c1",
                snapshot=pb2.SnapshotCommand(
                    mode=pb2.SUSPEND_MODE_DISCARD, tags=["weights"]
                ),
            )
        )
        self.assertEqual(result.command_id, "c1")
        self.assertTrue(result.ok)

        result = self.send_and_wait(
            pb2.AgentCommand(
                command_id="c2", restore=pb2.RestoreCommand(tags=["weights"])
            )
        )
        self.assertEqual(result.command_id, "c2")
        self.assertTrue(result.ok)

        self.assertEqual(
            workload.calls,
            [("snapshot", "discard", ["weights"]), ("restore", ["weights"])],
        )

    def test_workload_failure_reported(self):
        address = self.start_agent()
        self.register(
            address, RecordingWorkload(fail_with=RuntimeError("engine exploded"))
        )
        self.assertTrue(self.agent.registered.wait(timeout=10))

        result = self.send_and_wait(
            pb2.AgentCommand(command_id="c1", snapshot=pb2.SnapshotCommand())
        )
        self.assertFalse(result.ok)
        self.assertIn("engine exploded", result.error)

    def test_reconnects_and_reregisters(self):
        address = self.start_agent(drop_streams=1)
        workload = RecordingWorkload()
        self.register(address, workload)

        # First stream is dropped by the agent right after registration; the
        # client must reconnect with backoff and register again.
        deadline = threading.Event()
        for _ in range(100):
            if len(self.agent.registrations) >= 2:
                break
            deadline.wait(0.1)
        self.assertGreaterEqual(len(self.agent.registrations), 2)

        result = self.send_and_wait(
            pb2.AgentCommand(command_id="c1", snapshot=pb2.SnapshotCommand())
        )
        self.assertTrue(result.ok)

    def test_capability_overrides(self):
        address = self.start_agent()
        self.register(
            address,
            RecordingWorkload(),
            supported_modes=["offload"],
            default_mode="offload",
            tags=["weights"],
        )
        self.assertTrue(self.agent.registered.wait(timeout=10))
        caps = self.agent.registrations[0].capabilities
        self.assertEqual(list(caps.supported_modes), [pb2.SUSPEND_MODE_OFFLOAD])
        self.assertEqual(list(caps.tags), ["weights"])


class FakeVLLMEngines:
    """Builds classes that look like vLLM engines (module + name) without vllm."""

    @staticmethod
    def make(name, async_api=False):
        calls = []
        if async_api:

            async def sleep(self, level):
                calls.append(("sleep", level))

            async def wake_up(self, tags=None):
                calls.append(("wake_up", tags))
        else:

            def sleep(self, level):
                calls.append(("sleep", level))

            def wake_up(self, tags=None):
                calls.append(("wake_up", tags))

        cls = type(name, (), {"sleep": sleep, "wake_up": wake_up})
        cls.__module__ = "vllm.engine.fake"
        return cls(), calls


class VLLMAdapterTest(unittest.TestCase):
    def test_mode_maps_to_sleep_level(self):
        engine, calls = FakeVLLMEngines.make("AsyncLLMEngine")
        adapter = resolve_adapter(engine, None, None)
        self.assertEqual(adapter.name, "vllm")

        adapter.snapshot(pb2.SUSPEND_MODE_OFFLOAD, [])
        adapter.snapshot(pb2.SUSPEND_MODE_DISCARD, [])
        adapter.snapshot(pb2.SUSPEND_MODE_UNSPECIFIED, [])
        adapter.restore([])
        adapter.restore(["weights"])
        self.assertEqual(
            calls,
            [
                ("sleep", 1),
                ("sleep", 2),
                ("sleep", 1),
                ("wake_up", None),
                ("wake_up", ["weights"]),
            ],
        )

    def test_async_engine_through_channel(self):
        engine, calls = FakeVLLMEngines.make("AsyncLLM", async_api=True)
        agent = FakeAgent()
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        pb2_grpc.add_SnapshotAgentServiceServicer_to_server(agent, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        self.addCleanup(server.stop, 0)

        handle = register_workload(
            f"127.0.0.1:{port}", job_id="async-job", workload=engine
        )
        self.addCleanup(handle.close)
        self.assertTrue(agent.registered.wait(timeout=10))

        agent.commands.put(
            pb2.AgentCommand(command_id="c1", snapshot=pb2.SnapshotCommand())
        )
        result = agent.results.get(timeout=10)
        self.assertTrue(result.ok)
        self.assertEqual(calls, [("sleep", 1)])


class ResolveAdapterTest(unittest.TestCase):
    def test_requires_workload_or_callbacks(self):
        with self.assertRaises(ValueError):
            resolve_adapter(None, None, None)

    def test_rejects_workload_plus_callbacks(self):
        with self.assertRaises(ValueError):
            resolve_adapter(RecordingWorkload(), lambda m, t: None, lambda t: None)

    def test_rejects_unknown_workload(self):
        with self.assertRaises(TypeError):
            resolve_adapter(object(), None, None)

    def test_callbacks(self):
        calls = []
        adapter = resolve_adapter(
            None,
            lambda mode, tags: calls.append(("snap", mode, tags)),
            lambda tags: calls.append(("rest", tags)),
        )
        adapter.snapshot(pb2.SUSPEND_MODE_DISCARD, ["weights"])
        adapter.restore([])
        self.assertEqual(calls, [("snap", "discard", ["weights"]), ("rest", [])])


if __name__ == "__main__":
    unittest.main()
