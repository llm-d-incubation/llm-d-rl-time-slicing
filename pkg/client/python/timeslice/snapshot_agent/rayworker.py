"""Ray relay for the workload channel: one offload proxy actor per node.

Trainer ranks are usually one Ray actor per GPU, busy in long blocking tasks,
and the snapshot-agent accepts one channel registration per job per node.
``spawn_offload_proxies(handles, spec, job_id=...)`` therefore starts one
zero-CPU ``OffloadProxy`` actor per node hosting the handles, pinned there. Each
proxy registers the job with its node-local agent and relays every snapshot /
restore command to that node's ranks as Ray RPCs named by the ``OffloadSpec``.

A framework binding is an ``OffloadSpec`` plus the two lifecycle points where it
spawns (after the rank actors exist, before the first lock yield) and closes
the proxies; see ``timeslice_verl.offload`` and
guides/rl-frameworks/app-aware-offload-ray.md. This module needs ``ray``, which
is not a dependency of the ``timeslice`` client (hence not re-exported from
``timeslice.snapshot_agent``); ``timeslice`` must be importable on every node
that hosts rank actors.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from .workload import register_workload

ENV_AGENT_ADDR = "TIMESLICE_AGENT_ADDR"
# Below the agent's fixed two-minute command deadline, leaving room for the
# acknowledgement round trip.
DEFAULT_CALL_TIMEOUT_SEC = 100.0
DEFAULT_READY_TIMEOUT_SEC = 30.0
DEFAULT_NAME_PREFIX = "timeslice-offload-proxy"

_MODE_NAMES = ("offload", "discard")


def _log(msg: str) -> None:
    # print+flush: visible in job logs whatever the logging config, and Ray
    # forwards actor stdout to the driver.
    print(f"[timeslice] {msg}", flush=True)


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
    ):
        if not handles:
            raise ValueError("RankFanout needs at least one handle")
        self._handles = list(handles)
        self._spec = spec
        self._job_id = job_id
        self._node = node
        self.supported_modes = list(spec.supported_modes)
        self.default_mode = spec.default_mode
        self.tags = list(spec.tags)
        # Completed commands, exposed through OffloadProxy.info() so a binding
        # can tell that the job has been parked (e.g. before letting a
        # finishing driver exit).
        self.snapshots = 0
        self.restores = 0

    @property
    def ranks(self) -> int:
        return len(self._handles)

    def snapshot(self, mode: str, tags: List[str]) -> None:
        if mode not in self._spec.supported_modes:
            raise ValueError(
                f"unsupported suspend mode {mode!r}; this workload supports {self._spec.supported_modes}"
            )
        self._fanout(self._spec.method, self._spec.snapshot_kwargs, "OFFLOAD")
        self.snapshots += 1

    def restore(self, tags: List[str]) -> None:
        self._fanout(
            self._spec.restore_method or self._spec.method,
            self._spec.restore_kwargs,
            "RELOAD",
        )
        self.restores += 1

    def _fanout(self, method: str, kwargs: Mapping[str, Any], verb: str) -> None:
        prefix = f"job={self._job_id} {verb} node={self._node} ranks={len(self._handles)} method={method}"
        started = time.monotonic()
        try:
            refs = [getattr(h, method).remote(**dict(kwargs)) for h in self._handles]
            ray.get(refs, timeout=self._spec.call_timeout_sec)
        except Exception as err:
            _log(
                f"{prefix} FAILED after {time.monotonic() - started:.1f}s err={type(err).__name__}: {err}"
            )
            raise
        _log(f"{prefix} took={time.monotonic() - started:.1f}s")


class OffloadProxy:
    """One per node: holds the workload channel and relays to local ranks.

    ``OffloadProxyActor`` is this class as a zero-CPU Ray actor, started by
    ``spawn_offload_proxies``. The agent address is resolved in
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
    ):
        agent = agent or os.environ.get(ENV_AGENT_ADDR, "")
        if not agent:
            raise ValueError(
                f"no snapshot-agent address: pass agent= or set {ENV_AGENT_ADDR} on the node's pods"
            )
        self._job_id = job_id
        self._agent = agent
        self._node = node
        self._fanout = RankFanout(handles, spec, job_id=job_id, node=node)
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
        """Identity, connection state and completed-command counters of this
        proxy (``snapshots`` > 0 means the job's ranks on this node have been
        parked at least once)."""
        return {
            "job_id": self._job_id,
            "node": self._node,
            "agent": self._agent,
            "ranks": self._fanout.ranks,
            "connected": self._handle.connected,
            "snapshots": self._fanout.snapshots,
            "restores": self._fanout.restores,
        }

    def close(self, timeout: float = 5.0) -> None:
        self._handle.close(timeout=timeout)


OffloadProxyActor = ray.remote(num_cpus=0)(OffloadProxy)


def _node_id(actor_self: Any) -> str:
    """Runs inside a rank actor (via ``__ray_call__``): its Ray node id."""
    return ray.get_runtime_context().get_node_id()


def group_handles_by_node(
    handles: Sequence[Any], *, timeout: float = 60.0
) -> Dict[str, List[Any]]:
    """Maps Ray node id -> the handles living on that node, preserving the
    input (rank) order within each node."""
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

    def info(self, timeout: float = 10.0) -> List[Dict[str, Any]]:
        return ray.get([a.info.remote() for a in self.actors], timeout=timeout)

    def close(self, timeout: float = 5.0) -> None:
        """Deregisters every proxy and kills the actors. Never raises."""
        try:
            ray.get([a.close.remote(timeout) for a in self.actors], timeout=timeout + 5)
        except Exception as err:  # noqa: BLE001 - best effort; the kill below still deregisters
            _log(
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
) -> OffloadProxySet:
    """Starts one pinned ``OffloadProxy`` actor per node hosting ``handles``
    and waits until every proxy has its channel to the agent open.

    Must be called after the rank actors exist and before the job first yields
    the accelerator. ``agent=None`` lets each proxy resolve
    ``TIMESLICE_AGENT_ADDR`` on its own node. Raises ``TimeoutError`` (after
    closing what it started) if any proxy cannot reach its agent within
    ``ready_timeout`` seconds.
    """
    if not job_id:
        raise ValueError("job_id is required")
    if not handles:
        raise ValueError("no handles to proxy")

    actors: List[Any] = []
    nodes: List[str] = []
    for node_id, local in group_handles_by_node(handles, timeout=ready_timeout).items():
        short = node_id[:8]
        actor = OffloadProxyActor.options(
            name=f"{name_prefix}-{job_id}-{short}",
            num_cpus=0,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False
            ),
        ).remote(
            job_id=job_id,
            group=group,
            handles=local,
            spec=spec,
            agent=agent,
            node=short,
        )
        actors.append(actor)
        nodes.append(node_id)
    proxies = OffloadProxySet(job_id=job_id, actors=actors, nodes=nodes)

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
