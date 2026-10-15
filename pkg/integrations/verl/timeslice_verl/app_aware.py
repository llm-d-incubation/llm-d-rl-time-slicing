"""Application-aware parking of the trainer (opt-in; cuda-checkpoint is the default).

By default, while the trainer GPU is lent, the snapshot agent checkpoints the
trainer's whole CUDA state with cuda-checkpoint. With
``TIMESLICE_DONOR_BACKEND=app_channel`` (set by the admission webhook when the
trainer worker group is annotated ``timeslice.io/backend: app_channel``) the
trainer parks itself instead:

  park     the actor worker group moves its optimizer state, then its model
           (and gradients) to host memory and frees its cached GPU memory
           (verl's ``to("cpu", ...)``, the manual offload control every verl
           engine implements). Each worker then re-points FSDP1's unsharded
           parameter views at the moved flat parameters (see
           ``_resync_fsdp_views``), empties torch's allocator cache and, when
           cupy is already loaded in the worker (verl's NCCL checkpoint
           engine allocates its weight-sync buckets with cupy), cupy's memory
           pools, which torch's ``empty_cache`` does not touch. The CUDA
           context and a small allocator residue stay, which the agent bounds
           at verify.
  restore  the worker group moves everything back (``to("device", ...)``;
           verl's worker-level ``to`` names the accelerator "device"), then
           re-points the views again.

The agent drives both through a workload channel: the trainer driver registers
the park/restore workload with the snapshot agent of every node that runs an
actor worker (the agent is on the host network, so its address is the worker's
node IP and ``TIMESLICE_AGENT_PORT``). Each worker reports ``TIMESLICE_NODE_IP``
(its pod's host IP, set by the webhook) and falls back to Ray's node IP, which
is the pod IP in a KubeRay pod and so only right on the host network. The
agent selects the channel from the annotation on the trainer pods.

Safety:
  * a park is only accepted at a safe point: when the trainer does not hold the
    group lock (after a yield, before the next acquire). The orchestrator only
    parks a trainer that yielded, so a refusal means something is wrong, and
    the agent fails closed (the GPU is not lent).
  * after every granted acquire the trainer makes sure it is resident again
    (``ensure_resident``), so a restore the agent lost (an agent restart while
    the trainer was parked) is redone by the trainer itself.

Environment contract (on top of hooks.py's):

  TIMESLICE_DONOR_BACKEND=app_channel   activation gate of this module
  TIMESLICE_AGENT_PORT                  snapshot agent port (default 9001)
  TIMESLICE_NODE_IP                     the worker's node IP (downward API
                                        status.hostIP, set by the webhook)
  TIMESLICE_AGENT_ADDRS                 optional comma-separated host:port list
                                        that replaces the node discovery
  TIMESLICE_DEVICE                      device name of the restore (default
                                        "device", verl's accelerator alias)

Nothing here imports verl, ray or grpc at module level, so the logic is
testable without them.
"""

import gc
import os
import sys
import threading
from collections.abc import Callable, Iterable
from time import monotonic as _now

from timeslice_verl.locks import _log

ENV_DONOR_BACKEND = "TIMESLICE_DONOR_BACKEND"
ENV_AGENT_PORT = "TIMESLICE_AGENT_PORT"
ENV_AGENT_ADDRS = "TIMESLICE_AGENT_ADDRS"
ENV_DEVICE = "TIMESLICE_DEVICE"
ENV_NODE_IP = "TIMESLICE_NODE_IP"
BACKEND_APP_CHANNEL = "app_channel"
DEFAULT_AGENT_PORT = 9001
MODE_OFFLOAD = "offload"
# Above this much torch memory still allocated after a park, the park log
# lists the largest CUDA tensors left.
_LEFTOVER_REPORT_BYTES = 1 << 30


def enabled() -> bool:
    return os.environ.get(ENV_DONOR_BACKEND, "").strip().lower() == BACKEND_APP_CHANNEL


class NotAtSafePointError(RuntimeError):
    """The trainer holds the GPU lock: parking it now would pull its state
    from under a running step."""


def _device_name() -> str:
    # verl's worker-level to() accepts "cpu" and "device" (its alias for the
    # accelerator), and asserts on anything else, "cuda" included.
    return os.environ.get(ENV_DEVICE, "").strip() or "device"


def _resync_fsdp_views() -> int:
    """Re-point FSDP1's unsharded parameter views at their flat parameters.

    FSDP1 with one rank runs NO_SHARD, and with ``use_orig_params=False`` the
    module keeps the original parameters as Tensor views of the flat
    parameter outside forward. Moving the flat parameter (verl's offload
    does ``flat_param.data = flat_param.to(...)``) does not refresh them, so
    the views keep the old storage alive: after a park the whole model stays
    on the GPU. This does what FSDP's next forward would do. Returns the
    number of handles re-pointed (0 when FSDP1 is not loaded)."""
    mod = sys.modules.get("torch.distributed.fsdp._flat_param")
    handle_cls = getattr(mod, "FlatParamHandle", None) if mod is not None else None
    if handle_cls is None:
        return 0
    count = 0
    for obj in gc.get_objects():
        if not isinstance(obj, handle_cls):
            continue
        if obj._use_orig_params or obj.uses_sharded_strategy:
            continue  # FSDP refreshes these views itself, or frees the unsharded storage
        obj._use_unsharded_views(as_params=False)
        count += 1
    return count


def _largest_cuda_tensors(limit: int = 4) -> list[str]:
    """The largest CUDA storages still referenced from Python, for the park
    log when something kept more than expected."""
    seen: dict[int, tuple[int, str]] = {}
    for obj in gc.get_objects():
        try:
            if not getattr(obj, "is_cuda", False) or not hasattr(obj, "untyped_storage"):
                continue
            st = obj.untyped_storage()
            seen.setdefault(st.data_ptr(), (st.nbytes(), f"{type(obj).__name__}{tuple(obj.shape)} {obj.dtype}"))
        except Exception:  # noqa: BLE001, S112 - best effort over arbitrary objects
            continue
    top = sorted(seen.values(), reverse=True)[:limit]
    return [f"{desc}={size}" for size, desc in top]


def _free_device_cache(_worker) -> dict:
    """Return the worker's cached device memory to the driver; runs inside
    each actor worker after the offload. Returns what it saw (bytes).

    torch's allocator cache is emptied, then cupy's default and pinned memory
    pools when cupy is already imported in the worker: verl's NCCL checkpoint
    engine allocates its weight-sync buckets with cupy, and dropping them
    returns the blocks to cupy's pool, not to the driver. cupy is never
    imported here (importing it would create the pools it then frees)."""
    stats: dict = {}
    try:
        stats["fsdp_views"] = _resync_fsdp_views()
    except Exception as e:  # noqa: BLE001 - best effort: the agent's verify decides
        stats["fsdp_views_error"] = repr(e)
    gc.collect()
    try:
        import torch  # noqa: PLC0415 - only inside the workers

        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            stats["torch_alloc"] = int(torch.cuda.memory_allocated())
            stats["torch_reserved"] = int(torch.cuda.memory_reserved())
            if stats["torch_alloc"] > _LEFTOVER_REPORT_BYTES:
                stats["largest"] = _largest_cuda_tensors()
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001 - best effort: the agent's verify decides
        stats["torch_error"] = repr(e)
    cupy = sys.modules.get("cupy")
    if cupy is not None:
        try:
            pool = cupy.get_default_memory_pool()
            stats["cupy_before"] = int(pool.total_bytes())
            pool.free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
            stats["cupy_after"] = int(pool.total_bytes())
        except Exception as e:  # noqa: BLE001 - best effort: the agent's verify decides
            stats["cupy_error"] = repr(e)
    return stats


def _trim_torch_cache(_worker) -> dict:
    """Return torch's unused cached blocks to the driver before the weight
    sync; runs inside each actor worker. The park frees cupy's pools, so the
    NCCL checkpoint engine's next prepare() allocates its cupy buckets with a
    fresh cudaMalloc after the training step has grown torch's cache to the
    whole GPU; without this trim that allocation can fail although torch holds
    gigabytes it does not use. Never raises."""
    try:
        import torch  # noqa: PLC0415 - only inside the workers

        if not torch.cuda.is_initialized():
            return {}
        before = int(torch.cuda.memory_reserved())
        torch.cuda.empty_cache()
        return {"reserved_before": before, "reserved_after": int(torch.cuda.memory_reserved())}
    except Exception as e:  # noqa: BLE001 - best effort
        return {"error": repr(e)}


def _resync_views(_worker) -> int:
    return _resync_fsdp_views()


def _on_workers(worker_group, fn) -> list:
    import ray  # noqa: PLC0415 - only in the Ray driver

    workers = list(getattr(worker_group, "workers", []) or [])
    return list(ray.get([w.__ray_call__.remote(fn) for w in workers]))


def free_worker_caches(worker_group) -> list[dict]:
    """Run _free_device_cache in every actor worker of the group."""
    return _on_workers(worker_group, _free_device_cache)


def trim_worker_torch_caches(worker_group) -> list[dict]:
    """Run _trim_torch_cache in every actor worker of the group."""
    return _on_workers(worker_group, _trim_torch_cache)


def resync_worker_views(worker_group) -> list[int]:
    """Re-point the FSDP1 views in every actor worker after the restore."""
    return _on_workers(worker_group, _resync_views)


class TrainerPark:
    """The park/restore workload of the trainer's actor worker group.

    ``worker_group`` needs verl's manual offload control:
    ``to(device, model=..., optimizer=..., grad=...)``, dispatched to every
    rank. ``is_safe()`` tells whether the trainer is at a safe point (does not
    hold the group lock). Thread-safe: the agent's commands arrive on the
    workload-channel threads, ``ensure_resident`` on the driver's.
    """

    def __init__(
        self,
        worker_group,
        is_safe: Callable[[], bool],
        device: str | None = None,
        clock=_now,
        free_cache: Callable[[object], object] | None = None,
        after_restore: Callable[[object], object] | None = None,
        trim_cache: Callable[[object], object] | None = None,
    ):
        self._wg = worker_group
        self._free_cache = free_cache
        self._trim_cache = trim_cache
        self._after_restore = after_restore
        self._is_safe = is_safe
        self._device = device
        self._clock = clock
        self._lock = threading.Lock()
        self._parked = False
        self.parks = 0
        self.restores = 0
        self.self_restores = 0
        self.last_park_s: float | None = None
        self.last_restore_s: float | None = None

    @property
    def parked(self) -> bool:
        return self._parked

    # The SnapshottableAdapter interface of the timeslice client.
    def snapshot(self, mode: str, tags: list[str]) -> None:
        if mode != MODE_OFFLOAD:
            raise ValueError(f"the trainer only parks in mode {MODE_OFFLOAD!r} (got {mode!r}): it must restore")
        with self._lock:
            if self._parked:
                _log("app-park: already parked (repeated park, nothing to do)")
                return
            if not self._is_safe():
                raise NotAtSafePointError("park refused: the trainer holds the GPU lock (not at a safe point)")
            t0 = self._clock()
            # Optimizer first, then the model: the model offload empties the
            # allocator cache, which then also returns the optimizer's blocks.
            self._wg.to("cpu", model=False, optimizer=True, grad=False)
            self._wg.to("cpu", model=True, optimizer=False, grad=True)
            self._parked = True
            caches = ""
            if self._free_cache is not None:
                try:
                    caches = f" caches={self._free_cache(self._wg)}"
                except Exception as e:  # noqa: BLE001 - the park is done; the agent's verify decides
                    caches = f" caches=error({e!r})"
            self.parks += 1
            self.last_park_s = self._clock() - t0
            _log(f"app-park took={self.last_park_s:.3f}s parks={self.parks}{caches}")

    def trim_before_sync(self) -> None:
        """Empty torch's cache in the workers before the weight sync (see
        _trim_torch_cache). Only while resident; never raises."""
        with self._lock:
            if self._parked or self._trim_cache is None:
                return
            try:
                _log(f"pre-sync torch cache trim: {self._trim_cache(self._wg)}")
            except Exception as e:  # noqa: BLE001 - the sync itself decides
                _log(f"pre-sync torch cache trim failed (ignored): {e!r}")

    def restore(self, tags: list[str]) -> None:
        self._restore(source="agent")

    def ensure_resident(self) -> bool:
        """Restore if still parked (the agent's restore was lost). Returns
        True when it had to restore."""
        return self._restore(source="self")

    def _restore(self, source: str) -> bool:
        with self._lock:
            if not self._parked:
                return False
            t0 = self._clock()
            self._wg.to(self._device or _device_name(), model=True, optimizer=True, grad=True)
            self._parked = False
            if self._after_restore is not None:
                try:
                    self._after_restore(self._wg)
                except Exception as e:  # noqa: BLE001 - FSDP's next forward re-points the views anyway
                    _log(f"app-restore: re-pointing the parameter views failed: {e!r}")
            self.restores += 1
            if source == "self":
                self.self_restores += 1
            self.last_restore_s = self._clock() - t0
            note = ""
            if source == "self":
                note = " (WARNING: the agent did not restore the trainer; restored by the trainer)"
            _log(f"app-restore source={source} took={self.last_restore_s:.3f}s restores={self.restores}{note}")
            return True


def _host_port(host: str, port: int) -> str:
    host = host.strip().strip("[]")
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def agent_addresses(node_ips: Iterable[str], port: int | None = None) -> list[str]:
    """The agent address of every distinct node (in first-seen order), or the
    explicit TIMESLICE_AGENT_ADDRS list."""
    explicit = [a.strip() for a in os.environ.get(ENV_AGENT_ADDRS, "").split(",") if a.strip()]
    if explicit:
        return explicit
    if port is None:
        raw = os.environ.get(ENV_AGENT_PORT, "").strip()
        port = int(raw) if raw else DEFAULT_AGENT_PORT
    seen: list[str] = []
    for ip in node_ips:
        if ip and ip not in seen:
            seen.append(ip)
    return [_host_port(ip, port) for ip in seen]


def _node_ip(_worker) -> str:
    host_ip = os.environ.get(ENV_NODE_IP, "").strip()
    if host_ip:
        return host_ip
    import ray  # noqa: PLC0415 - only inside Ray workers

    return ray.util.get_node_ip_address()


def worker_node_ips(worker_group) -> list[str]:
    """Node IPs of the worker group's Ray actors (``__ray_call__`` runs a
    function inside each actor)."""
    import ray  # noqa: PLC0415 - only in the Ray driver

    workers = list(getattr(worker_group, "workers", []) or [])
    return list(ray.get([w.__ray_call__.remote(_node_ip) for w in workers]))


def start(
    worker_group,
    job_id: str,
    group: str,
    is_safe: Callable[[], bool],
    register=None,
    node_ips: Callable[[object], list[str]] = worker_node_ips,
):
    """Register the trainer's park/restore workload with the snapshot agent of
    every node of the worker group. Returns (TrainerPark, handles)."""
    if register is None:
        from timeslice.snapshot_agent.workload import register_workload as register  # noqa: PLC0415, N813

    park = TrainerPark(
        worker_group,
        is_safe,
        free_cache=free_worker_caches,
        after_restore=resync_worker_views,
        trim_cache=trim_worker_torch_caches,
    )
    addrs = agent_addresses(node_ips(worker_group))
    if not addrs:
        raise RuntimeError("app_channel donor: no node found for the actor workers; set TIMESLICE_AGENT_ADDRS")
    handles = [
        register(agent=a, job_id=job_id, group=group, workload=park, supported_modes=[MODE_OFFLOAD]) for a in addrs
    ]
    _log(f"app-park: workload of job={job_id} group={group} registered with agents {addrs}")
    return park, handles
