"""Adapters translating agent commands into workload-specific suspend/resume calls.

Resolution order (see resolve_adapter):
  1. Explicit callbacks (on_snapshot / on_restore).
  2. Known engine types recognized by inspection (vLLM).
  3. The Snapshottable duck-type protocol: any object with snapshot(mode, tags)
     and restore(tags) methods.

Adapters run inside the workload's process; the agent only decides and
transports. Engine mechanics (what "suspend" means) stay with the engine.
"""

import logging
from typing import Any, Callable, List, Optional

from . import snapshot_agent_pb2

logger = logging.getLogger(__name__)

# Public mode names accepted from Snapshottable attributes and register_workload kwargs.
_MODE_NAMES = {
    "offload": snapshot_agent_pb2.SUSPEND_MODE_OFFLOAD,
    "discard": snapshot_agent_pb2.SUSPEND_MODE_DISCARD,
}


def mode_from_name(name: str) -> int:
    """Maps a mode name ("offload"/"discard") to the SuspendMode enum."""
    try:
        return _MODE_NAMES[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown suspend mode {name!r}; expected one of {sorted(_MODE_NAMES)}"
        ) from None


def mode_name(mode: int) -> str:
    """Maps a SuspendMode enum value to its public name."""
    for name, value in _MODE_NAMES.items():
        if value == mode:
            return name
    return str(mode)


class WorkloadAdapter:
    """Base adapter. Subclasses implement snapshot/restore; either may return
    an awaitable, which the channel runner resolves."""

    name = "base"
    supported_modes: List[int] = []
    default_mode: int = snapshot_agent_pb2.SUSPEND_MODE_UNSPECIFIED
    tags: List[str] = []

    def snapshot(self, mode: int, tags: List[str]) -> Any:
        raise NotImplementedError

    def restore(self, tags: List[str]) -> Any:
        raise NotImplementedError


class CallbackAdapter(WorkloadAdapter):
    """Escape hatch: user-provided callables."""

    name = "callbacks"

    def __init__(self, on_snapshot: Callable[..., Any], on_restore: Callable[..., Any]):
        self._on_snapshot = on_snapshot
        self._on_restore = on_restore

    def snapshot(self, mode: int, tags: List[str]) -> Any:
        return self._on_snapshot(mode_name(mode), tags)

    def restore(self, tags: List[str]) -> Any:
        return self._on_restore(tags)


class SnapshottableAdapter(WorkloadAdapter):
    """Wraps any object implementing the Snapshottable protocol:

    class MyWorkload:
        supported_modes = ["offload"]        # optional
        default_mode = "offload"             # optional
        def snapshot(self, mode, tags): ...  # mode is "offload"/"discard"
        def restore(self, tags): ...
    """

    name = "snapshottable"

    def __init__(self, obj: Any):
        self._obj = obj
        declared = getattr(obj, "supported_modes", None)
        if declared:
            self.supported_modes = [mode_from_name(m) for m in declared]
        default = getattr(obj, "default_mode", None)
        if default:
            self.default_mode = mode_from_name(default)

    def snapshot(self, mode: int, tags: List[str]) -> Any:
        return self._obj.snapshot(mode_name(mode), tags)

    def restore(self, tags: List[str]) -> Any:
        return self._obj.restore(tags)


def _is_vllm_engine(obj: Any) -> bool:
    """Recognizes vLLM engines without importing vllm (it may not be installed
    here, and if obj is a vLLM engine the app has already imported it)."""
    cls = type(obj)
    module = cls.__module__ or ""
    return module.split(".")[0] == "vllm" and cls.__name__ in (
        "LLM",
        "AsyncLLMEngine",
        "AsyncLLM",
    )


class VLLMAdapter(WorkloadAdapter):
    """Drives a vLLM engine object through its Python sleep API.

    SuspendMode maps to the sleep level: OFFLOAD -> level 1 (weights backed up
    in host memory), DISCARD -> level 2 (weights dropped; the app re-provisions
    them, e.g. RL weight sync). Requires the engine to be constructed with
    enable_sleep_mode=True.
    """

    name = "vllm"
    supported_modes = [
        snapshot_agent_pb2.SUSPEND_MODE_OFFLOAD,
        snapshot_agent_pb2.SUSPEND_MODE_DISCARD,
    ]

    def __init__(self, engine: Any):
        self._engine = engine

    def snapshot(self, mode: int, tags: List[str]) -> Any:
        level = 2 if mode == snapshot_agent_pb2.SUSPEND_MODE_DISCARD else 1
        return self._engine.sleep(level=level)

    def restore(self, tags: List[str]) -> Any:
        if tags:
            return self._engine.wake_up(tags=list(tags))
        return self._engine.wake_up()


def resolve_adapter(
    workload: Optional[Any],
    on_snapshot: Optional[Callable[..., Any]],
    on_restore: Optional[Callable[..., Any]],
) -> WorkloadAdapter:
    """Picks the adapter for a workload; see module docstring for the order."""
    if on_snapshot is not None or on_restore is not None:
        if on_snapshot is None or on_restore is None:
            raise ValueError("on_snapshot and on_restore must be provided together")
        if workload is not None:
            raise ValueError("pass either workload= or callbacks, not both")
        return CallbackAdapter(on_snapshot, on_restore)
    if workload is None:
        raise ValueError(
            "a workload object or on_snapshot/on_restore callbacks are required"
        )
    if _is_vllm_engine(workload):
        return VLLMAdapter(workload)
    if callable(getattr(workload, "snapshot", None)) and callable(
        getattr(workload, "restore", None)
    ):
        return SnapshottableAdapter(workload)
    raise TypeError(
        f"don't know how to suspend {type(workload).__name__}: not a recognized engine and "
        "not Snapshottable (snapshot/restore methods); pass on_snapshot=/on_restore= callbacks"
    )
