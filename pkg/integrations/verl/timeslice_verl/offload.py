"""verl binding for the timeslice Ray relay: app_channel offload of the trainer.

The generic relay (``timeslice.snapshot_agent.rayworker``) needs two things
from a framework: *what* to call on each rank actor to move its state, and
*when* to spawn/close the per-node proxies. This module supplies the first for
verl; :class:`timeslice_verl.hooks.TimesliceHooksMixin` supplies the second
(spawn at the end of ``init_workers``, before the first lock yield; close on
crash).

What is called: verl's engine workers expose a manual offload RPC,
``to(device, model=True, optimizer=True, grad=True)``, implemented by every
engine backend (FSDP, Megatron, ...) "irrespective of offload config". On the
raw rank handles of a colocated worker group the method carries the role
prefix, so the name is ``f"{train_role}_to"``: ``actor_to`` for a standalone
trainer (``hybrid_engine=False``) and ``actor_rollout_to`` for a hybrid one.

Snapshot moves model + optimizer (+ grads) to host; restore brings model +
optimizer back with ``grad=False`` — the state verl itself leaves after a
weight sync (FSDP ignores the flag; Megatron re-allocates grads lazily on the
next backward).

Environment contract (read by the hooks mixin, documented here):

  TIMESLICE_APP_OFFLOAD=1      opt in to app-channel offload; inert otherwise
  TIMESLICE_AGENT_ADDR         node-local snapshot-agent, e.g. "$(NODE_IP):9001"
                               with NODE_IP from the downward API
                               (status.hostIP). Resolved inside each proxy on
                               its own node, so set it on every pod.
"""

import os
from collections.abc import Sequence
from typing import Any

from timeslice_verl.locks import _log

ENV_APP_OFFLOAD = "TIMESLICE_APP_OFFLOAD"
ENV_AGENT_ADDR = "TIMESLICE_AGENT_ADDR"

# How long init_workers waits for every per-node proxy to reach its agent
# before giving up (and failing the job while the lock is still held).
OFFLOAD_READY_TIMEOUT_SEC = 60.0

SNAPSHOT_KWARGS = {"device": "cpu", "model": True, "optimizer": True, "grad": True}
RESTORE_KWARGS = {"device": "device", "model": True, "optimizer": True, "grad": False}


def offload_method_for_role(train_role: Any) -> str:
    """Name of the manual offload RPC on a raw rank handle for a verl role
    (``str(Role.Actor)`` -> "actor" -> "actor_to")."""
    role = str(train_role)
    if not role:
        raise ValueError("train_role is required")
    return f"{role}_to"


def offload_spec_for_role(train_role: Any):
    """The relay OffloadSpec for a verl trainer worker group of the given role."""
    from timeslice.snapshot_agent.rayworker import OffloadSpec  # noqa: PLC0415 - lazy: keep import-safe without client

    return OffloadSpec(
        method=offload_method_for_role(train_role),
        snapshot_kwargs=dict(SNAPSHOT_KWARGS),
        restore_kwargs=dict(RESTORE_KWARGS),
    )


def default_proxy_spawner(handles: Sequence[Any], spec: Any, *, job_id: str, group: str):
    """Spawns the per-node proxies with the relay's defaults. ``agent=None``
    makes each proxy resolve TIMESLICE_AGENT_ADDR on its own node; proxy log
    lines go through ``_log`` so they show up as ``[timeslice] ...`` in the
    job log."""
    from timeslice.snapshot_agent.rayworker import spawn_offload_proxies  # noqa: PLC0415 - lazy, see above

    return spawn_offload_proxies(
        handles,
        spec,
        job_id=job_id,
        group=group,
        agent=None,
        ready_timeout=OFFLOAD_READY_TIMEOUT_SEC,
        log=_log,
    )


def app_offload_requested() -> bool:
    return os.environ.get(ENV_APP_OFFLOAD, "").strip().lower() in ("1", "true", "yes")
