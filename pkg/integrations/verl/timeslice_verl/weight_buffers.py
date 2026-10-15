"""Keep the rollout side's weight-sync buffers allocated across syncs.

At every weight sync, verl's rollout-side CheckpointEngineWorker allocates
three bucket-sized CUDA buffers on the rollout GPU and frees them again when
the sync is done:

  - NCCLCheckpointEngine.prepare() (receiver, ``is_master=False``):
    send_buf and recv_buf, the double buffer the NCCL broadcast lands in;
    finalize() drops them and empties the torch cache;
  - BucketedWeightSender._init_buffer(): the CUDA IPC buffer that hands the
    received weights to the vLLM worker process; _cleanup() frees it.

Between syncs the vLLM process on the same GPU grows its own cache into the
memory these buffers left free (it reaches the whole card during validation,
whatever gpu_memory_utilization is), so the next sync's fresh allocation can
fail with CUDA OOM in the CheckpointEngineWorker, which kills the job. vLLM's
allocator gives back its cache and retries when it runs short; the worker's
fresh allocation in another process cannot.

With TIMESLICE_PERSIST_WEIGHT_BUFFERS=1 this module allocates those buffers
once per process and reuses them at every sync, so the rollout GPU holds
them for the whole run (3 x update_weights_bucket_megabytes) and the sync no
longer needs fresh device memory. The trainer side (``is_master=True``, cupy
buffers) and the shared-memory fallback are left as they are.

The gate is separate from TIMESLICE_FULLY_ASYNC so a run without time-slicing
can apply the same change. Inert without the gate; a no-op on verl layouts
without these classes.
"""

import functools
import importlib
import os

from timeslice_verl.locks import _log

ENV_PERSIST = "TIMESLICE_PERSIST_WEIGHT_BUFFERS"

_MARK = "_timeslice_persist_buffers"


def enabled() -> bool:
    return os.environ.get(ENV_PERSIST, "").strip().lower() in ("1", "true", "yes")


def apply() -> dict:
    """Install the patches when the gate is on. Returns what was patched."""
    done = {"nccl_receiver": False, "ipc_sender": False}
    if not enabled():
        return done
    engine_cls = _find("verl.checkpoint_engine.nccl_checkpoint_engine", "NCCLCheckpointEngine")
    if engine_cls is not None:
        done["nccl_receiver"] = patch_nccl_receiver(engine_cls)
    sender_cls = _find("verl.workers.rollout.vllm_rollout.bucketed_weight_transfer", "BucketedWeightSender")
    if sender_cls is not None:
        done["ipc_sender"] = patch_ipc_sender(sender_cls)
    if any(done.values()):
        _log(f"weight-sync buffers persist across syncs: {done}")
    return done


def _find(module: str, name: str):
    """The class ``name`` of verl's ``module``, or None on a verl layout without it."""
    try:
        return getattr(importlib.import_module(module), name)
    except Exception:  # noqa: BLE001 - any import failure means: nothing to patch
        return None


def patch_nccl_receiver(cls) -> bool:
    """Reuse the receiver's send/recv buffers across prepare()/finalize()."""
    prepare = getattr(cls, "prepare", None)
    finalize = getattr(cls, "finalize", None)
    if prepare is None or finalize is None:
        return False
    if getattr(prepare, _MARK, False):
        return True

    @functools.wraps(prepare)
    def patched_prepare(self, *args, **kwargs):
        if getattr(self, "is_master", False):
            return prepare(self, *args, **kwargs)
        kept = getattr(self, "_ts_kept_bufs", None)
        if kept is not None and kept[0] == self.bucket_size:
            self.send_buf, self.recv_buf = kept[1], kept[2]
            return None  # what prepare() returns on a non-master rank
        out = prepare(self, *args, **kwargs)
        self._ts_kept_bufs = (self.bucket_size, self.send_buf, self.recv_buf)
        _log(f"nccl receiver buffers allocated once: 2 x {self.bucket_size >> 20} MiB")
        return out

    # finalize() still drops self.send_buf/recv_buf and empties the cache; the
    # reference in _ts_kept_bufs keeps the memory allocated for the next sync.
    setattr(patched_prepare, _MARK, True)
    cls.prepare = patched_prepare
    return True


_SENDER_BUFFERS: dict = {}


def patch_ipc_sender(cls) -> bool:
    """Reuse one CUDA IPC buffer per (device, size) across BucketedWeightSenders."""
    init_buffer = getattr(cls, "_init_buffer", None)
    if init_buffer is None:
        return False
    if getattr(init_buffer, _MARK, False):
        return True

    @functools.wraps(init_buffer)
    def patched_init_buffer(self, *args, **kwargs):
        if getattr(self, "use_shm", False):
            return init_buffer(self, *args, **kwargs)
        import torch
        from torch.multiprocessing.reductions import reduce_tensor

        device = torch.device("cuda", torch.cuda.current_device())
        key = (str(device), self.bucket_size)
        buffer = _SENDER_BUFFERS.get(key)
        if buffer is None:
            buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device=device)
            _SENDER_BUFFERS[key] = buffer
            _log(f"ipc sender buffer allocated once: {self.bucket_size >> 20} MiB on {device}")
        self.socket.send_pyobj(reduce_tensor(buffer))
        self.socket.recv()
        self.buffer = buffer
        self.shm = None
        return None

    # _cleanup() still drops self.buffer and empties the cache; _SENDER_BUFFERS
    # keeps the memory allocated for the next sender.
    setattr(patched_init_buffer, _MARK, True)
    cls._init_buffer = patched_init_buffer
    return True
