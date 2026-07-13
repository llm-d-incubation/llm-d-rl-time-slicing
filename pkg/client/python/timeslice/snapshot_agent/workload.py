"""Workload-side of the snapshot-agent workload channel.

register_workload() opens a long-lived WorkloadChannel stream to the
(node-local) snapshot-agent, registers the job, and services SNAPSHOT/RESTORE
commands by invoking the resolved adapter in this process. The library owns
the stream lifecycle: a background daemon thread, command dispatch and acks,
and reconnect with exponential backoff (re-registering on every reconnect;
the agent replaces the previous registration).
"""

import asyncio
import inspect
import logging
import queue
import random
import threading
import time
from typing import Any, Callable, List, Optional

import grpc

from . import snapshot_agent_pb2, snapshot_agent_pb2_grpc
from .adapters import mode_from_name, resolve_adapter

logger = logging.getLogger(__name__)

_CLOSE = object()

_INITIAL_BACKOFF_SEC = 0.5
_MAX_BACKOFF_SEC = 30.0
# A connection that survived this long resets the backoff.
_STABLE_CONNECTION_SEC = 5.0


class WorkloadHandle:
    """Handle for a registered workload. close() deregisters by dropping the
    stream and stops the background thread."""

    def __init__(
        self,
        agent: str,
        register_msg: "snapshot_agent_pb2.RegisterWorkload",
        adapter: Any,
        loop: Optional[asyncio.AbstractEventLoop],
    ):
        self._agent = agent
        self._register_msg = register_msg
        self._adapter = adapter
        self._loop = loop
        self._stop = threading.Event()
        self._call: Optional[Any] = None
        self._call_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name=f"workload-channel-{register_msg.job_id}",
            daemon=True,
        )
        self._thread.start()

    def close(self, timeout: float = 5.0) -> None:
        """Stops servicing commands and closes the channel."""
        self._stop.set()
        with self._call_lock:
            if self._call is not None:
                self._call.cancel()
        self._thread.join(timeout=timeout)

    # -- background thread --

    def _run(self) -> None:
        backoff = _INITIAL_BACKOFF_SEC
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._connect_and_serve()
            except grpc.RpcError as err:
                if not self._stop.is_set():
                    logger.warning(
                        "workload channel for job %s disconnected: %s",
                        self._register_msg.job_id,
                        err,
                    )
            except Exception:
                logger.exception(
                    "workload channel for job %s failed", self._register_msg.job_id
                )
            if self._stop.is_set():
                return
            if time.monotonic() - started > _STABLE_CONNECTION_SEC:
                backoff = _INITIAL_BACKOFF_SEC
            sleep_for = backoff * (0.5 + random.random())  # noqa: S311 - jitter, not crypto
            backoff = min(backoff * 2, _MAX_BACKOFF_SEC)
            logger.info(
                "reconnecting workload channel for job %s in %.1fs",
                self._register_msg.job_id,
                sleep_for,
            )
            self._stop.wait(sleep_for)

    def _connect_and_serve(self) -> None:
        sendq: "queue.SimpleQueue" = queue.SimpleQueue()

        def requests():
            yield snapshot_agent_pb2.WorkloadMessage(register=self._register_msg)
            while True:
                item = sendq.get()
                if item is _CLOSE:
                    return
                yield item

        with grpc.insecure_channel(self._agent) as channel:
            stub = snapshot_agent_pb2_grpc.SnapshotAgentServiceStub(channel)
            call = stub.WorkloadChannel(requests())
            with self._call_lock:
                self._call = call
            logger.info(
                "registering workload for job %s with agent %s",
                self._register_msg.job_id,
                self._agent,
            )
            try:
                for command in call:
                    result = self._execute(command)
                    sendq.put(snapshot_agent_pb2.WorkloadMessage(result=result))
            finally:
                with self._call_lock:
                    self._call = None
                # Unblock the request generator so grpc's consumer thread exits.
                sendq.put(_CLOSE)

    # -- command execution --

    def _execute(
        self, command: "snapshot_agent_pb2.AgentCommand"
    ) -> "snapshot_agent_pb2.CommandResult":
        kind = command.WhichOneof("command")
        logger.info("executing %s command %s", kind, command.command_id)
        try:
            if kind == "snapshot":
                outcome = self._adapter.snapshot(
                    command.snapshot.mode, list(command.snapshot.tags)
                )
            elif kind == "restore":
                outcome = self._adapter.restore(list(command.restore.tags))
            else:
                raise ValueError(f"unknown command type {kind!r}")
            self._await_if_needed(outcome)
            return snapshot_agent_pb2.CommandResult(
                command_id=command.command_id, ok=True
            )
        except Exception as err:  # noqa: BLE001 - error is reported to the agent
            logger.exception("%s command %s failed", kind, command.command_id)
            return snapshot_agent_pb2.CommandResult(
                command_id=command.command_id,
                ok=False,
                error=f"{type(err).__name__}: {err}",
            )

    def _await_if_needed(self, outcome: Any) -> None:
        """Resolves awaitable adapter results. Async engines (e.g. vLLM
        AsyncLLMEngine) are driven by the app's event loop, captured at
        registration; their coroutines are scheduled onto it from this
        thread. Loop-free awaitables run on a private loop."""
        if not inspect.isawaitable(outcome):
            return
        if not inspect.iscoroutine(outcome):
            coro = _wrap_awaitable(outcome)
        else:
            coro = outcome
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(coro, self._loop).result()
        else:
            asyncio.run(coro)


async def _wrap_awaitable(awaitable: Any) -> Any:
    return await awaitable


def register_workload(
    agent: str,
    job_id: str,
    group: str = "",
    workload: Optional[Any] = None,
    on_snapshot: Optional[Callable[..., Any]] = None,
    on_restore: Optional[Callable[..., Any]] = None,
    supported_modes: Optional[List[str]] = None,
    default_mode: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> WorkloadHandle:
    """Registers this process's workload with the node's snapshot-agent and
    services suspend/resume commands for it.

    Args:
        agent: snapshot-agent address, e.g. "127.0.0.1:9001".
        job_id: job identity; must match the job_id used in Snapshot/Restore
            requests (in k8s, the timeslice.io/job-id pod label).
        group: optional group identity.
        workload: the engine or workload object to suspend. Recognized engines
            (vLLM LLM/AsyncLLMEngine/AsyncLLM) need nothing else; any object
            with snapshot(mode, tags)/restore(tags) methods also works.
        on_snapshot / on_restore: callback escape hatch instead of workload.
        supported_modes / default_mode: capability overrides ("offload",
            "discard"); defaults come from the adapter.
        tags: region tags the workload understands, e.g. ["weights", "kv_cache"].

    Returns:
        WorkloadHandle; call close() on clean shutdown.
    """
    if not job_id:
        raise ValueError("job_id is required")
    adapter = resolve_adapter(workload, on_snapshot, on_restore)

    modes = (
        [mode_from_name(m) for m in supported_modes]
        if supported_modes
        else adapter.supported_modes
    )
    default = mode_from_name(default_mode) if default_mode else adapter.default_mode
    capabilities = snapshot_agent_pb2.WorkloadCapabilities(
        supported_modes=modes,
        default_mode=default,
        tags=tags if tags is not None else adapter.tags,
    )
    register_msg = snapshot_agent_pb2.RegisterWorkload(
        job_id=job_id, group=group, capabilities=capabilities
    )

    try:
        loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    return WorkloadHandle(agent, register_msg, adapter, loop)
