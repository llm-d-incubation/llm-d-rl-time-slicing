"""Unit tests for timeslice_verl.weight_buffers (pure python, no verl/torch)."""

import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from timeslice_verl import weight_buffers  # noqa: E402


class _Engine:
    """Mimics NCCLCheckpointEngine.prepare/finalize."""

    allocs = 0

    def __init__(self, bucket_size, is_master=False):
        self.bucket_size = bucket_size
        self.is_master = is_master

    def prepare(self):
        type(self).allocs += 2
        self.send_buf = object()
        self.recv_buf = object()
        return "meta" if self.is_master else None

    def finalize(self):
        self.send_buf = None
        self.recv_buf = None


def _fresh_engine():
    return type("Engine", (_Engine,), {"allocs": 0})


def test_receiver_buffers_allocated_once_and_reused():
    cls = _fresh_engine()
    assert weight_buffers.patch_nccl_receiver(cls)
    eng = cls(bucket_size=8)
    assert eng.prepare() is None
    first = (eng.send_buf, eng.recv_buf)
    eng.finalize()
    assert eng.send_buf is None and eng.recv_buf is None
    for _ in range(3):
        assert eng.prepare() is None
        assert (eng.send_buf, eng.recv_buf) == first
        eng.finalize()
    assert cls.allocs == 2


def test_receiver_reallocates_when_bucket_size_changes():
    cls = _fresh_engine()
    weight_buffers.patch_nccl_receiver(cls)
    eng = cls(bucket_size=8)
    eng.prepare()
    eng.finalize()
    eng.bucket_size = 16
    eng.prepare()
    assert cls.allocs == 4


def test_master_unchanged():
    cls = _fresh_engine()
    weight_buffers.patch_nccl_receiver(cls)
    eng = cls(bucket_size=8, is_master=True)
    for _ in range(3):
        assert eng.prepare() == "meta"
        eng.finalize()
    assert cls.allocs == 6
    assert not hasattr(eng, "_ts_kept_bufs")


def test_receiver_patch_idempotent_and_needs_methods():
    cls = _fresh_engine()
    weight_buffers.patch_nccl_receiver(cls)
    first = cls.prepare
    assert weight_buffers.patch_nccl_receiver(cls)
    assert cls.prepare is first
    assert weight_buffers.patch_nccl_receiver(type("X", (), {})) is False


class _Sock:
    def __init__(self):
        self.sent = []

    def send_pyobj(self, obj):
        self.sent.append(obj)

    def recv(self):
        return b""


def _fake_torch(monkeypatch):
    allocs = []
    torch = types.ModuleType("torch")
    torch.uint8 = "uint8"
    torch.device = lambda kind, idx: f"{kind}:{idx}"
    torch.cuda = types.SimpleNamespace(current_device=lambda: 0)

    def empty(size, dtype=None, device=None):
        buf = ("buf", size, device, len(allocs))
        allocs.append(buf)
        return buf

    torch.empty = empty
    mp = types.ModuleType("torch.multiprocessing")
    red = types.ModuleType("torch.multiprocessing.reductions")
    red.reduce_tensor = lambda t: ("handle", t)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.multiprocessing", mp)
    monkeypatch.setitem(sys.modules, "torch.multiprocessing.reductions", red)
    return allocs


class _Sender:
    def __init__(self, bucket_size, use_shm=False):
        self.bucket_size = bucket_size
        self.use_shm = use_shm
        self.socket = _Sock()
        self.buffer = None
        self.shm = None

    def _init_buffer(self):
        self.buffer = "original"


def test_ipc_sender_buffer_shared_across_senders(monkeypatch):
    allocs = _fake_torch(monkeypatch)
    monkeypatch.setattr(weight_buffers, "_SENDER_BUFFERS", {})
    cls = type("Sender", (_Sender,), {"_init_buffer": _Sender._init_buffer})
    assert weight_buffers.patch_ipc_sender(cls)
    bufs = []
    for _ in range(3):
        s = cls(bucket_size=32)
        s._init_buffer()
        assert s.socket.sent == [("handle", s.buffer)]
        bufs.append(s.buffer)
    assert len(allocs) == 1 and bufs[0] is bufs[1] is bufs[2]
    s = cls(bucket_size=64)
    s._init_buffer()
    assert len(allocs) == 2


def test_ipc_sender_shm_unchanged(monkeypatch):
    _fake_torch(monkeypatch)
    cls = type("Sender", (_Sender,), {"_init_buffer": _Sender._init_buffer})
    weight_buffers.patch_ipc_sender(cls)
    s = cls(bucket_size=32, use_shm=True)
    s._init_buffer()
    assert s.buffer == "original"


def test_apply_inert_without_env(monkeypatch):
    monkeypatch.delenv("TIMESLICE_PERSIST_WEIGHT_BUFFERS", raising=False)
    monkeypatch.setenv("TIMESLICE_FULLY_ASYNC", "1")
    assert weight_buffers.apply() == {"nccl_receiver": False, "ipc_sender": False}


def test_apply_with_env_patches_both(monkeypatch):
    monkeypatch.setenv("TIMESLICE_PERSIST_WEIGHT_BUFFERS", "1")
    monkeypatch.delenv("TIMESLICE_FULLY_ASYNC", raising=False)
    eng = _fresh_engine()
    snd = type("Sender", (_Sender,), {"_init_buffer": _Sender._init_buffer})
    m1 = types.ModuleType("verl.checkpoint_engine.nccl_checkpoint_engine")
    m1.NCCLCheckpointEngine = eng
    m2 = types.ModuleType("verl.workers.rollout.vllm_rollout.bucketed_weight_transfer")
    m2.BucketedWeightSender = snd
    for name in (
        "verl",
        "verl.checkpoint_engine",
        "verl.workers",
        "verl.workers.rollout",
        "verl.workers.rollout.vllm_rollout",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, m1.__name__, m1)
    monkeypatch.setitem(sys.modules, m2.__name__, m2)
    assert weight_buffers.apply() == {"nccl_receiver": True, "ipc_sender": True}
    assert getattr(eng.prepare, "_timeslice_persist_buffers", False)
    assert getattr(snd._init_buffer, "_timeslice_persist_buffers", False)


def test_apply_without_verl_is_noop(monkeypatch):
    monkeypatch.setenv("TIMESLICE_PERSIST_WEIGHT_BUFFERS", "1")
    monkeypatch.setitem(sys.modules, "verl", None)
    assert weight_buffers.apply() == {"nccl_receiver": False, "ipc_sender": False}
