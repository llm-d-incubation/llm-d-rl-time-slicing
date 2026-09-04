"""Helpers for building BackendConfig protos."""

from typing import Optional, Sequence, Tuple, Union

from . import snapshot_agent_pb2
from .types import MemoryRegion

RegionLike = Union[MemoryRegion, Tuple[int, int, int], str]


def _process_target(pids: Sequence[int]) -> snapshot_agent_pb2.ProcessTarget:
    """Validates pids and builds a ProcessTarget proto."""
    if not pids:
        raise ValueError("at least one PID is required")
    validated = []
    for pid in pids:
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise ValueError(f"PID must be a positive integer, got {pid!r}")
        validated.append(pid)
    return snapshot_agent_pb2.ProcessTarget(pids=validated)


_APPS = {
    "vllm": snapshot_agent_pb2.APP_VLLM,
    "sglang": snapshot_agent_pb2.APP_SGLANG,
}

_SUSPEND_MODES = {
    "offload": snapshot_agent_pb2.SUSPEND_MODE_OFFLOAD,
    "discard": snapshot_agent_pb2.SUSPEND_MODE_DISCARD,
}


def _suspend_mode(mode: Union[str, int, None]) -> int:
    """None -> UNSPECIFIED (workload/application default); accepts friendly
    strings ("offload"/"discard") or the raw enum value."""
    if mode is None:
        return snapshot_agent_pb2.SUSPEND_MODE_UNSPECIFIED
    if isinstance(mode, str):
        try:
            return _SUSPEND_MODES[mode.lower()]
        except KeyError:
            raise ValueError(
                f"unknown suspend mode {mode!r}; expected one of {sorted(_SUSPEND_MODES)}"
            )
    snapshot_agent_pb2.SuspendMode.Name(mode)  # raises ValueError if invalid
    return mode


def _tags(tags: Optional[Sequence[str]]) -> list:
    if isinstance(tags, str):
        raise ValueError("tags must be a sequence of non-empty strings")
    tags = list(tags or [])
    if any(not isinstance(t, str) or not t for t in tags):
        raise ValueError(f"tags must be non-empty strings, got {tags!r}")
    return tags


def cuda_config(
    pids: Optional[Sequence[int]] = None,
) -> snapshot_agent_pb2.BackendConfig:
    """Builds a BackendConfig selecting the cuda (cuda-checkpoint) backend.

    In k8s mode the agent discovers PIDs itself from the
    ``timeslice.io/job-id`` pod label, so ``pids`` may be omitted. Pass
    explicit PIDs for standalone mode or to override discovery.
    """
    cuda = snapshot_agent_pb2.CudaBackendConfig()
    if pids is not None:
        cuda.explicit_target.CopyFrom(_process_target(pids))
    return snapshot_agent_pb2.BackendConfig(cuda=cuda)


def direct_memory_config(
    pids: Optional[Sequence[int]] = None,
) -> snapshot_agent_pb2.BackendConfig:
    """Builds a BackendConfig selecting the direct_memory (GPU-CR
    full-process) backend.

    Experimental: the agent rejects this config with FAILED_PRECONDITION
    unless it runs with --feature-gates=DirectMemoryBackend=true (or the
    FEATURE_GATES env var). The target workload must run under the GPU-CR
    vGPU preloader.

    In k8s mode the agent discovers PIDs itself from the
    ``timeslice.io/job-id`` pod label, so ``pids`` may be omitted. Pass
    explicit PIDs for standalone mode or to override discovery.
    """
    direct_memory = snapshot_agent_pb2.DirectMemoryBackendConfig()
    if pids is not None:
        direct_memory.explicit_target.CopyFrom(_process_target(pids))
    return snapshot_agent_pb2.BackendConfig(direct_memory=direct_memory)


def app_endpoint_config(
    app: Union[str, int],
    endpoints: Sequence[str],
    mode: Union[str, int, None] = None,
    tags: Optional[Sequence[str]] = None,
) -> snapshot_agent_pb2.BackendConfig:
    """Builds a BackendConfig selecting the app_endpoint backend, which
    suspends/resumes an application-aware workload through its HTTP API.

    ``app`` selects the HTTP dialect: "vllm", "sglang", or an App enum
    value. ``endpoints`` targets the server(s), e.g.
    ``["http://localhost:8000"]``. ``mode`` states what happens to durable
    state while suspended: "offload", "discard", a SuspendMode value, or
    None for the application's default (for SGLang it is advisory: the
    effective mode is fixed by the server's launch flags). ``tags`` selects
    regions, e.g. ``["weights", "kv_cache"]``; if None or empty, the
    application's full default region set is used.
    """
    if isinstance(app, str):
        try:
            app = _APPS[app.lower()]
        except KeyError:
            raise ValueError(f"unknown app {app!r}; expected one of {sorted(_APPS)}")
    else:
        snapshot_agent_pb2.App.Name(app)  # raises ValueError if invalid
    if app == snapshot_agent_pb2.APP_UNSPECIFIED:
        raise ValueError(f"app must be specified; expected one of {sorted(_APPS)}")
    if isinstance(endpoints, str):
        raise ValueError("endpoints must be a sequence of non-empty endpoint URLs")
    endpoints = list(endpoints)
    if not endpoints or any(not isinstance(e, str) or not e for e in endpoints):
        raise ValueError("at least one non-empty endpoint URL is required")
    return snapshot_agent_pb2.BackendConfig(
        app_endpoint=snapshot_agent_pb2.AppEndpointConfig(
            app=app, endpoints=endpoints, mode=_suspend_mode(mode), tags=_tags(tags)
        )
    )


def vllm_config(
    endpoints: Sequence[str],
    mode: Union[str, int, None] = None,
    tags: Optional[Sequence[str]] = None,
) -> snapshot_agent_pb2.BackendConfig:
    """app_endpoint_config preset for vLLM's sleep/wake API."""
    return app_endpoint_config("vllm", endpoints, mode=mode, tags=tags)


def sglang_config(
    endpoints: Sequence[str],
    mode: Union[str, int, None] = None,
    tags: Optional[Sequence[str]] = None,
) -> snapshot_agent_pb2.BackendConfig:
    """app_endpoint_config preset for SGLang's memory-occupation API."""
    return app_endpoint_config("sglang", endpoints, mode=mode, tags=tags)


def app_channel_config(
    mode: Union[str, int, None] = None,
    tags: Optional[Sequence[str]] = None,
) -> snapshot_agent_pb2.BackendConfig:
    """Builds a BackendConfig selecting the app_channel backend, which
    suspends/resumes an application-aware workload through its registered
    workload channel (see ``register_workload``).

    The workload is addressed by the request's job_id, so no endpoints are
    needed. ``mode`` is "offload", "discard", a SuspendMode value, or None
    for the workload's registered default (OFFLOAD if the workload did not
    declare one). ``tags`` selects regions; see ``app_endpoint_config``.
    """
    return snapshot_agent_pb2.BackendConfig(
        app_channel=snapshot_agent_pb2.AppChannelConfig(
            mode=_suspend_mode(mode), tags=_tags(tags)
        )
    )


def memory_regions_config(
    regions: Sequence[RegionLike],
    snapshot_name: str = "",
) -> snapshot_agent_pb2.BackendConfig:
    """Builds a BackendConfig selecting the memory-regions backend.

    Accepts MemoryRegion dataclasses, (pid, address, size_bytes) tuples, or
    'pid:0xADDR:size' spec strings. Hex or decimal addresses are
    accepted; validation mirrors the agent's (non-empty regions, positive
    pid and size) plus the wire-format bounds (int32 pid, uint64 address
    and size).

    snapshot_name names the agent-side snapshot slot (defaults to the
    request's job_id server-side when empty). Use it — not the request's
    `group` — for slot naming: group identifies a set of related jobs for
    the orchestrator and does not name agent-side storage.
    """
    if not regions:
        raise ValueError("at least one memory region is required")

    pb_regions = []
    for region in regions:
        if isinstance(region, MemoryRegion):
            r = region
        elif isinstance(region, str):
            r = MemoryRegion.from_spec(region)
        elif isinstance(region, tuple):
            if len(region) != 3:
                raise ValueError(
                    f"memory region tuple must be (pid, address, size_bytes), got {region!r}"
                )
            r = MemoryRegion(pid=region[0], address=region[1], size_bytes=region[2])
        else:
            raise TypeError(
                "memory region must be a MemoryRegion, (pid, address, size_bytes) "
                f"tuple, or 'pid:0xADDR:size' string, got {type(region).__name__}"
            )
        pb_regions.append(
            snapshot_agent_pb2.MemoryRegion(
                pid=r.pid, address=r.address, size_bytes=r.size_bytes
            )
        )

    return snapshot_agent_pb2.BackendConfig(
        memory_regions=snapshot_agent_pb2.MemoryRegionsBackendConfig(
            regions=pb_regions,
            snapshot_name=snapshot_name,
        )
    )
