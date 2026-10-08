"""In-process snapshot-agent double for workload-channel tests.

FakeAgent serves SnapshotAgentService.WorkloadChannel: it records every
registration, pushes queued AgentCommands to the workload, and collects the
CommandResults. Commands can be addressed to all streams (``agent.commands``)
or to one specific stream (``agent.streams[i].commands``); results land both
on the stream and on the aggregate ``agent.results`` queue.
"""

import queue
import threading

import grpc

from timeslice.snapshot_agent import snapshot_agent_pb2_grpc as pb2_grpc

_END = object()


class StreamState:
    """One open WorkloadChannel stream, in registration order."""

    def __init__(self, register):
        self.register = register
        self.commands = queue.Queue()
        self.results = queue.Queue()
        self.closed = threading.Event()


class FakeAgent(pb2_grpc.SnapshotAgentServiceServicer):
    """In-process agent: records registrations, pushes queued commands,
    collects results. drop_streams ends that many streams right after
    registration to exercise reconnect."""

    def __init__(self, drop_streams=0):
        self.registrations = []
        self.streams = []
        self.registered = threading.Event()
        self.results = queue.Queue()
        self.commands = queue.Queue()
        self.drop_streams = drop_streams
        self._lock = threading.Lock()

    def wait_streams(self, n, timeout=10.0):
        """Blocks until n streams have registered (and are still open)."""
        deadline = threading.Event()
        for _ in range(int(timeout / 0.05)):
            with self._lock:
                open_streams = [s for s in self.streams if not s.closed.is_set()]
            if len(open_streams) >= n:
                return open_streams
            deadline.wait(0.05)
        with self._lock:
            return [s for s in self.streams if not s.closed.is_set()]

    def WorkloadChannel(self, request_iterator, context):  # noqa: N802 - grpc method name
        first = next(request_iterator)
        stream = StreamState(first.register)
        with self._lock:
            self.registrations.append(first.register)
            self.streams.append(stream)
        self.registered.set()
        if self.drop_streams > 0:
            self.drop_streams -= 1
            stream.closed.set()
            return

        def read_results():
            try:
                for msg in request_iterator:
                    stream.results.put(msg.result)
                    self.results.put(msg.result)
            except grpc.RpcError:
                return  # the client cancelled the stream (close()): expected

        reader = threading.Thread(target=read_results, daemon=True)
        reader.start()
        try:
            # Poll so the worker thread is released when the test stops the
            # server (a blocking get would pin a non-daemon grpc thread forever).
            while context.is_active():
                cmd = None
                for q in (stream.commands, self.commands):
                    try:
                        cmd = q.get(timeout=0.1)
                        break
                    except queue.Empty:
                        continue
                if cmd is None:
                    continue
                if cmd is _END:
                    return
                yield cmd
        finally:
            stream.closed.set()
