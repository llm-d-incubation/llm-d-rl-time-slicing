"""Ray relay for the workload channel: per-node offload proxies for Ray actors.

Some workloads keep their accelerator state in Ray actors that cannot run the
workload channel themselves: an RL trainer, for example, is one actor per rank,
each executing long blocking tasks, and the snapshot-agent accepts exactly one
channel registration per job per node. This module bridges the two.

    spawn_offload_proxies(handles, spec, job_id=...)

groups the rank actors by the Ray node they live on and starts one zero-CPU
``OffloadProxy`` actor per node, pinned to that node. Each proxy registers the
job with its *node-local* snapshot-agent through ``register_workload`` and,
when the agent sends a snapshot or restore command, calls the method named by
the ``OffloadSpec`` on every rank of its node (as ordinary Ray RPCs, so each
rank runs the offload on its own thread, between its tasks) and acknowledges
once all of them returned.

The relay is framework-agnostic. A framework "binding" is only an
``OffloadSpec`` (which remote method to call, with which keyword arguments for
snapshot and for restore) plus the two lifecycle points where it spawns and
closes the proxies: after the rank actors exist, and before the job first
yields the accelerator. The verl binding lives in ``timeslice_verl.offload``.

Requirements and caveats:

* ``ray`` is imported lazily inside the functions, so importing ``timeslice``
  does not require Ray. ``timeslice`` itself must be importable on every Ray
  node that hosts rank actors (the proxy class and the node-id probe are
  pickled by reference).
* The agent address is resolved **inside the proxy's own process** (explicit
  ``agent`` argument, else ``TIMESLICE_AGENT_ADDR``), because it is node
  relative (``$(NODE_IP):9001`` with the agent on host networking) and the
  caller may run on a different node than the proxy.
* The agent gives a command two minutes; the fan-out waits
  ``OffloadSpec.call_timeout_sec`` (default 100 s) so a slow rank fails the
  command with an attributable error instead of an agent-side timeout.
* Proxies are non-detached actors owned by the caller, so they disappear with
  it; the dropped stream deregisters them at the agent.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .workload import register_workload

logger = logging.getLogger(__name__)

ENV_AGENT_ADDR = "TIMESLICE_AGENT_ADDR"
# Below the agent's fixed two-minute command deadline, leaving room for the
# acknowledgement round trip.
DEFAULT_CALL_TIMEOUT_SEC = 100.0
DEFAULT_READY_TIMEOUT_SEC = 30.0
DEFAULT_NAME_PREFIX = "timeslice-offload-proxy"

_MODE_NAMES = ("offload", "discard")

LogFn = Callable[[str], None]


def _default_log(msg: str) -> None:
    logger.info("%s", msg)


@dataclass(frozen=True)
class OffloadSpec:
    """What to call on each rank actor to offload and reload its state.

    ``method`` is the name of a remote method present on every handle; it is
    invoked as ``handle.<method>.remote(**snapshot_kwargs)`` on snapshot and
    ``handle.<restore_method or method>.remote(**restore_kwargs)`` on restore.
    Trainers advertise ``supported_modes=("offload",)`` (they cannot rebuild
    discarded state); the agent rejects other modes before any command is sent.
    Request tags are advertised but not interpreted by the fan-out.
    """

    method: str
    snapshot_kwargs: Mapping[str, Any] = field(default_factory=dict)
    restore_kwargs: Mapping[str, Any] = field(default_factory=dict)
    restore_method: Optional[str] = None
    supported_modes: Tuple[str, ...] = ("offload",)
    default_mode: str = "offload"
    tags: Tuple[str, ...] = ()
    call_timeout_sec: float = DEFAULT_CALL_TIMEOUT_SEC

    def __post_init__(self) -> None:
        if not self.method:
            raise ValueError("OffloadSpec.method is required")
        unknown = [m for m in self.supported_modes if m not in _MODE_NAMES]
        if unknown or not self.supported_modes:
            raise ValueError(
                f"supported_modes must be a non-empty subset of {_MODE_NAMES}, got {self.supported_modes!r}"
            )
        if self.default_mode not in self.supported_modes:
            raise ValueError(
                f"default_mode {self.default_mode!r} is not in supported_modes {self.supported_modes!r}"
            )
        if self.call_timeout_sec <= 0:
            raise ValueError("call_timeout_sec must be positive")


class RankFanout:
    """Snapshottable workload over the rank actors of one node.

    Implements the duck-typed protocol ``register_workload`` accepts
    (``snapshot(mode, tags)`` / ``restore(tags)``): each call fans the spec's
    method out to every handle and blocks until all ranks returned. Errors and
    timeouts propagate so the channel reports the command as failed.
    """

    def __init__(
        self,
        handles: Sequence[Any],
        spec: OffloadSpec,
        *,
        job_id: str = "",
        node: str = "",
        log: Optional[LogFn] = None,
    ):
        if not handles:
            raise ValueError("RankFanout needs at least one handle")
        self._handles = list(handles)
        self._spec = spec
        self._job_id = job_id
        self._node = node
        self._log = log or _default_log
        self.supported_modes = list(spec.supported_modes)
        self.default_mode = spec.default_mode
        self.tags = list(spec.tags)

    @property
    def ranks(self) -> int:
        return len(self._handles)

    def snapshot(self, mode: str, tags: List[str]) -> None:
        if mode not in self._spec.supported_modes:
            raise ValueError(
                f"unsupported suspend mode {mode!r}; this workload supports {self._spec.supported_modes}"
            )
        self._fanout(self._spec.method, self._spec.snapshot_kwargs, "OFFLOAD")

    def restore(self, tags: List[str]) -> None:
        self._fanout(
            self._spec.restore_method or self._spec.method,
            self._spec.restore_kwargs,
            "RELOAD",
        )

    def _fanout(self, method: str, kwargs: Mapping[str, Any], verb: str) -> None:
        import ray  # noqa: PLC0415 - ray is only needed where handles exist

        prefix = f"job={self._job_id} {verb} node={self._node} ranks={len(self._handles)} method={method}"
        started = time.monotonic()
        try:
            refs = [getattr(h, method).remote(**dict(kwargs)) for h in self._handles]
            ray.get(refs, timeout=self._spec.call_timeout_sec)
        except Exception as err:
            self._log(
                f"{prefix} FAILED after {time.monotonic() - started:.1f}s err={type(err).__name__}: {err}"
            )
            raise
        self._log(f"{prefix} took={time.monotonic() - started:.1f}s")


class OffloadProxy:
    """One per node: holds the workload channel and relays to local ranks.

    Instantiated as a Ray actor by ``spawn_offload_proxies`` (plain class
    here; ``ray.remote`` is applied lazily). The agent address is resolved in
    this process so that a node-relative ``TIMESLICE_AGENT_ADDR`` is correct.
    """

    def __init__(
        self,
        job_id: str,
        group: str,
        handles: Sequence[Any],
        spec: OffloadSpec,
        agent: Optional[str] = None,
        node: str = "",
        log: Optional[LogFn] = None,
    ):
        agent = agent or os.environ.get(ENV_AGENT_ADDR, "")
        if not agent:
            raise ValueError(
                f"no snapshot-agent address: pass agent= or set {ENV_AGENT_ADDR} on the node's pods"
            )
        self._job_id = job_id
        self._agent = agent
        self._node = node
        self._spec = spec
        self._fanout = RankFanout(handles, spec, job_id=job_id, node=node, log=log)
        self._handle = register_workload(
            agent,
            job_id=job_id,
            group=group,
            workload=self._fanout,
            supported_modes=list(spec.supported_modes),
            default_mode=spec.default_mode,
            tags=list(spec.tags),
        )

    def ready(self, timeout: float = DEFAULT_READY_TIMEOUT_SEC) -> bool:
        """Blocks until the channel to the agent is open (or timeout)."""
        return self._handle.wait_connected(timeout)

    def info(self) -> Dict[str, Any]:
        return {
            "job_id": self._job_id,
            "node": self._node,
            "agent": self._agent,
            "ranks": self._fanout.ranks,
            "method": self._spec.method,
            "connected": self._handle.connected,
        }

    def close(self, timeout: float = 5.0) -> None:
        self._handle.close(timeout=timeout)


_proxy_actor_cls: Any = None


def _proxy_cls() -> Any:
    """``OffloadProxy`` wrapped as a zero-CPU Ray actor class (cached)."""
    global _proxy_actor_cls  # noqa: PLW0603 - one-time lazy wrap
    if _proxy_actor_cls is None:
        import ray  # noqa: PLC0415 - lazy: keep the package importable without ray

        _proxy_actor_cls = ray.remote(num_cpus=0)(OffloadProxy)
    return _proxy_actor_cls


def _node_id(actor_self: Any) -> str:
    """Runs inside a rank actor (via ``__ray_call__``): its Ray node id."""
    import ray  # noqa: PLC0415 - executes in the rank's process

    return ray.get_runtime_context().get_node_id()


def group_handles_by_node(
    handles: Sequence[Any], *, timeout: float = 60.0
) -> Dict[str, List[Any]]:
    """Maps Ray node id -> the handles living on that node, preserving the
    input (rank) order within each node."""
    import ray  # noqa: PLC0415 - lazy: keep the package importable without ray

    node_ids = ray.get(
        [h.__ray_call__.remote(_node_id) for h in handles], timeout=timeout
    )
    groups: Dict[str, List[Any]] = {}
    for handle, node_id in zip(handles, node_ids):
        groups.setdefault(node_id, []).append(handle)
    return groups


@dataclass
class OffloadProxySet:
    """The proxies spawned for one job: one actor per node, in node order."""

    job_id: str
    actors: List[Any]
    nodes: List[str]
    log: LogFn = _default_log

    def info(self, timeout: float = 10.0) -> List[Dict[str, Any]]:
        import ray  # noqa: PLC0415

        return ray.get([a.info.remote() for a in self.actors], timeout=timeout)

    def close(self, timeout: float = 5.0) -> None:
        """Deregisters every proxy and kills the actors. Never raises."""
        import ray  # noqa: PLC0415

        try:
            ray.get([a.close.remote(timeout) for a in self.actors], timeout=timeout + 5)
        except Exception as err:  # noqa: BLE001 - best effort; the kill below still deregisters
            self.log(
                f"job={self.job_id} offload proxies: close reported {type(err).__name__}: {err}"
            )
        for actor in self.actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:  # noqa: BLE001, S110 - already gone is fine
                pass


def spawn_offload_proxies(
    handles: Sequence[Any],
    spec: OffloadSpec,
    *,
    job_id: str,
    group: str = "",
    agent: Optional[str] = None,
    ready_timeout: float = DEFAULT_READY_TIMEOUT_SEC,
    name_prefix: str = DEFAULT_NAME_PREFIX,
    log: Optional[LogFn] = None,
) -> OffloadProxySet:
    """Starts one pinned ``OffloadProxy`` actor per node hosting ``handles``
    and waits until every proxy has its channel to the agent open.

    Must be called after the rank actors exist and before the job first yields
    the accelerator. ``agent=None`` lets each proxy resolve
    ``TIMESLICE_AGENT_ADDR`` on its own node. Raises ``TimeoutError`` (after
    closing what it started) if any proxy cannot reach its agent within
    ``ready_timeout`` seconds.
    """
    import ray  # noqa: PLC0415 - lazy: keep the package importable without ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy  # noqa: PLC0415

    if not job_id:
        raise ValueError("job_id is required")
    if not handles:
        raise ValueError("no handles to proxy")
    log = log or _default_log

    actors: List[Any] = []
    nodes: List[str] = []
    for node_id, local in group_handles_by_node(handles).items():
        short = node_id[:8]
        actor = (
            _proxy_cls()
            .options(
                name=f"{name_prefix}-{job_id}-{short}",
                get_if_exists=True,
                num_cpus=0,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                ),
            )
            .remote(
                job_id=job_id,
                group=group,
                handles=local,
                spec=spec,
                agent=agent,
                node=short,
                log=log,
            )
        )
        actors.append(actor)
        nodes.append(node_id)
    proxies = OffloadProxySet(job_id=job_id, actors=actors, nodes=nodes, log=log)

    try:
        ready = ray.get(
            [a.ready.remote(ready_timeout) for a in actors], timeout=ready_timeout + 15
        )
    except Exception as err:
        proxies.close()
        raise RuntimeError(
            f"offload proxies for job {job_id} failed to start: {type(err).__name__}: {err}"
        ) from err
    not_ready = [node[:8] for node, ok in zip(nodes, ready) if not ok]
    if not_ready:
        proxies.close()
        raise TimeoutError(
            f"offload proxies for job {job_id} on nodes {not_ready} did not connect to the snapshot-agent "
            f"within {ready_timeout:.0f}s (agent={agent or '$' + ENV_AGENT_ADDR})"
        )
    return proxies
