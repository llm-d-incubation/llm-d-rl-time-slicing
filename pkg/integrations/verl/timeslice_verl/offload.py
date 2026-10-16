"""verl binding for the timeslice Ray relay: app_channel offload of the trainer.

Supplies *what* the relay calls on each trainer rank; ``TimesliceHooksMixin``
supplies *when* (spawn at the end of ``init_workers`` before the first yield,
close on crash). The method is verl's manual offload RPC
``to(device, model, optimizer, grad)``, implemented by every engine backend and
exposed on raw colocated rank handles as ``f"{train_role}_to"`` (``actor_to``,
or ``actor_rollout_to`` for a hybrid engine). Snapshot moves model + optimizer +
grads to host; restore brings model + optimizer back with ``grad=False``, the
state verl leaves after a weight sync. Trainer worker group only: verl's rollout
engines and their sleep/wake handling are untouched.

Environment (read by the hooks mixin): ``TIMESLICE_APP_OFFLOAD=1`` opts in;
``TIMESLICE_AGENT_ADDR`` is the node-local agent, e.g. ``$(NODE_IP):9001``,
resolved inside each proxy on its own node, so set it on every pod.
"""

import os
from collections.abc import Sequence
from typing import Any

ENV_APP_OFFLOAD = "TIMESLICE_APP_OFFLOAD"
ENV_AGENT_ADDR = "TIMESLICE_AGENT_ADDR"

# Sent as `backend_config` on every release once the proxies are up.
YIELD_BACKEND_CONFIG = "app_channel"

# How long init_workers waits for every per-node proxy to reach its agent
# before giving up (and failing the job while the lock is still held).
OFFLOAD_READY_TIMEOUT_SEC = 60.0

SNAPSHOT_KWARGS = {"device": "cpu", "model": True, "optimizer": True, "grad": True}
RESTORE_KWARGS = {"device": "device", "model": True, "optimizer": True, "grad": False}


def offload_spec_for_role(train_role: Any):
    """Relay OffloadSpec for a verl trainer role: ``str(Role.Actor)`` -> "actor" -> "actor_to"."""
    role = str(train_role)
    if not role:
        raise ValueError("train_role is required")
    from timeslice.snapshot_agent.rayworker import OffloadSpec  # noqa: PLC0415 - lazy: keep import-safe without client

    return OffloadSpec(
        method=f"{role}_to",
        snapshot_kwargs=dict(SNAPSHOT_KWARGS),
        restore_kwargs=dict(RESTORE_KWARGS),
    )


def default_proxy_spawner(handles: Sequence[Any], spec: Any, *, job_id: str, group: str):
    """Spawns the per-node proxies; each resolves TIMESLICE_AGENT_ADDR on its own node."""
    from timeslice.snapshot_agent.rayworker import spawn_offload_proxies  # noqa: PLC0415 - lazy, see above

    return spawn_offload_proxies(
        handles,
        spec,
        job_id=job_id,
        group=group,
        agent=None,
        ready_timeout=OFFLOAD_READY_TIMEOUT_SEC,
    )


def app_offload_requested() -> bool:
    return os.environ.get(ENV_APP_OFFLOAD, "").strip().lower() in ("1", "true", "yes")
