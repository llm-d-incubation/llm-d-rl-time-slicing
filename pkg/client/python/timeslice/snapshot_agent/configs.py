"""Helpers for building BackendConfig protos."""

from typing import Sequence

from . import snapshot_agent_pb2


def _process_target(pids: Sequence[int]) -> snapshot_agent_pb2.ProcessTarget:
    if not pids:
        raise ValueError("at least one PID is required")
    validated = []
    for pid in pids:
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise ValueError(f"PID must be a positive integer, got {pid!r}")
        validated.append(pid)
    return snapshot_agent_pb2.ProcessTarget(pids=validated)


def cuda_config(pids: Sequence[int]) -> snapshot_agent_pb2.BackendConfig:
    """Builds a BackendConfig selecting the cuda (cuda-checkpoint) backend
    with an explicit process target.

    In k8s mode the agent can discover PIDs itself from the
    ``timeslice.io/job-id`` pod label; pass explicit PIDs for standalone
    mode or to override discovery.
    """
    return snapshot_agent_pb2.BackendConfig(
        cuda=snapshot_agent_pb2.CudaBackendConfig(explicit_target=_process_target(pids))
    )


def direct_memory_config(pids: Sequence[int]) -> snapshot_agent_pb2.BackendConfig:
    """Builds a BackendConfig selecting the direct_memory (GPU-CR
    full-process) backend with an explicit process target.

    Experimental: the agent rejects this config with FAILED_PRECONDITION
    unless it runs with --feature-gates=DirectMemoryBackend=true (or the
    FEATURE_GATES env var). The target workload must run under the GPU-CR
    vGPU preloader.

    In k8s mode the agent can discover PIDs itself from the
    ``timeslice.io/job-id`` pod label; pass explicit PIDs for standalone
    mode or to override discovery.
    """
    return snapshot_agent_pb2.BackendConfig(
        direct_memory=snapshot_agent_pb2.DirectMemoryBackendConfig(
            explicit_target=_process_target(pids)
        )
    )
