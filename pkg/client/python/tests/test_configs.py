"""Tests for the BackendConfig builder helpers."""

import unittest

from timeslice.snapshot_agent import (
    MemoryRegion,
    app_channel_config,
    app_endpoint_config,
    cuda_config,
    direct_memory_config,
    memory_regions_config,
    sglang_config,
    vllm_config,
)
from timeslice.snapshot_agent import snapshot_agent_pb2


class TestBackendConfigBuilders(unittest.TestCase):
    def test_cuda_config_builds_explicit_target(self):
        """Explicit PIDs land in cuda.explicit_target."""
        cfg = cuda_config([101, 102])
        self.assertEqual(cfg.WhichOneof("backend"), "cuda")
        self.assertEqual(list(cfg.cuda.explicit_target.pids), [101, 102])

    def test_direct_memory_config_builds_explicit_target(self):
        """Explicit PIDs land in direct_memory.explicit_target."""
        cfg = direct_memory_config([101, 102])
        self.assertEqual(cfg.WhichOneof("backend"), "direct_memory")
        self.assertEqual(list(cfg.direct_memory.explicit_target.pids), [101, 102])

    def test_cuda_config_omits_target_when_pids_not_given(self):
        """Omitting pids leaves cuda.explicit_target unset (k8s discovery)."""
        cfg = cuda_config()
        self.assertEqual(cfg.WhichOneof("backend"), "cuda")
        self.assertFalse(cfg.cuda.HasField("explicit_target"))

    def test_direct_memory_config_omits_target_when_pids_not_given(self):
        """Omitting pids leaves direct_memory.explicit_target unset (k8s discovery)."""
        cfg = direct_memory_config()
        self.assertEqual(cfg.WhichOneof("backend"), "direct_memory")
        self.assertFalse(cfg.direct_memory.HasField("explicit_target"))

    def test_rejects_empty_pids(self):
        """An explicitly empty PID list is an error, not discovery."""
        with self.assertRaises(ValueError):
            cuda_config([])
        with self.assertRaises(ValueError):
            direct_memory_config([])

    def test_rejects_non_positive_pids(self):
        """Zero and negative PIDs are rejected."""
        for pids in ([0], [-5]):
            with self.assertRaises(ValueError):
                cuda_config(pids)
            with self.assertRaises(ValueError):
                direct_memory_config(pids)

    def test_rejects_non_integer_pids(self):
        """Non-integer PID values are rejected."""
        for pids in ([1.5], ["123"], [True]):
            with self.assertRaises((ValueError, TypeError)):
                cuda_config(pids)
            with self.assertRaises((ValueError, TypeError)):
                direct_memory_config(pids)

    def test_memory_regions_config_builds_regions(self):
        """Regions and snapshot_name land in the memory_regions oneof."""
        cfg = memory_regions_config(
            [MemoryRegion(pid=123, address=0x7F00, size_bytes=1024)],
            snapshot_name="slot-a",
        )
        self.assertEqual(cfg.WhichOneof("backend"), "memory_regions")
        self.assertEqual(cfg.memory_regions.snapshot_name, "slot-a")
        region = cfg.memory_regions.regions[0]
        self.assertEqual(region.pid, 123)
        self.assertEqual(region.address, 0x7F00)
        self.assertEqual(region.size_bytes, 1024)

    def test_app_endpoint_config_builds_full_config(self):
        cfg = app_endpoint_config(
            snapshot_agent_pb2.APP_VLLM,
            ["http://localhost:8000", "http://localhost:8001"],
            mode=snapshot_agent_pb2.SUSPEND_MODE_DISCARD,
            tags=["weights", "kv_cache"],
        )
        self.assertEqual(cfg.WhichOneof("backend"), "app_endpoint")
        self.assertEqual(cfg.app_endpoint.app, snapshot_agent_pb2.APP_VLLM)
        self.assertEqual(
            list(cfg.app_endpoint.endpoints),
            ["http://localhost:8000", "http://localhost:8001"],
        )
        self.assertEqual(cfg.app_endpoint.mode, snapshot_agent_pb2.SUSPEND_MODE_DISCARD)
        self.assertEqual(list(cfg.app_endpoint.tags), ["weights", "kv_cache"])

    def test_app_endpoint_config_accepts_friendly_strings(self):
        cfg = app_endpoint_config("vllm", ["http://localhost:8000"], mode="offload")
        self.assertEqual(cfg.app_endpoint.app, snapshot_agent_pb2.APP_VLLM)
        self.assertEqual(cfg.app_endpoint.mode, snapshot_agent_pb2.SUSPEND_MODE_OFFLOAD)
        self.assertEqual(
            cfg,
            app_endpoint_config(
                snapshot_agent_pb2.APP_VLLM,
                ["http://localhost:8000"],
                mode=snapshot_agent_pb2.SUSPEND_MODE_OFFLOAD,
            ),
        )

    def test_app_endpoint_config_strings_are_case_insensitive(self):
        cfg = app_endpoint_config("SGLang", ["http://localhost:8000"], mode="Discard")
        self.assertEqual(cfg.app_endpoint.app, snapshot_agent_pb2.APP_SGLANG)
        self.assertEqual(cfg.app_endpoint.mode, snapshot_agent_pb2.SUSPEND_MODE_DISCARD)

    def test_app_endpoint_config_defaults(self):
        cfg = app_endpoint_config("sglang", ["http://localhost:8000"])
        self.assertEqual(
            cfg.app_endpoint.mode, snapshot_agent_pb2.SUSPEND_MODE_UNSPECIFIED
        )
        self.assertEqual(list(cfg.app_endpoint.tags), [])

    def test_vllm_and_sglang_presets(self):
        cfg = vllm_config(["http://localhost:8000"], mode="discard", tags=["weights"])
        self.assertEqual(
            cfg,
            app_endpoint_config(
                "vllm", ["http://localhost:8000"], mode="discard", tags=["weights"]
            ),
        )
        cfg = sglang_config(["http://localhost:8000"])
        self.assertEqual(cfg.app_endpoint.app, snapshot_agent_pb2.APP_SGLANG)

    def test_app_endpoint_config_rejects_bad_app(self):
        for app in (snapshot_agent_pb2.APP_UNSPECIFIED, 999, "triton", ""):
            with self.assertRaises(ValueError):
                app_endpoint_config(app, ["http://localhost:8000"])

    def test_app_endpoint_config_rejects_bad_endpoints(self):
        for endpoints in ([], [""], [b"http://localhost:8000"], [123]):
            with self.assertRaises(ValueError):
                app_endpoint_config("vllm", endpoints)

    def test_app_channel_config_builds_full_config(self):
        cfg = app_channel_config(mode="offload", tags=["weights"])
        self.assertEqual(cfg.WhichOneof("backend"), "app_channel")
        self.assertEqual(cfg.app_channel.mode, snapshot_agent_pb2.SUSPEND_MODE_OFFLOAD)
        self.assertEqual(list(cfg.app_channel.tags), ["weights"])

    def test_app_channel_config_defaults(self):
        cfg = app_channel_config()
        self.assertEqual(cfg.WhichOneof("backend"), "app_channel")
        self.assertEqual(
            cfg.app_channel.mode, snapshot_agent_pb2.SUSPEND_MODE_UNSPECIFIED
        )
        self.assertEqual(list(cfg.app_channel.tags), [])

    def test_rejects_bad_suspend_mode(self):
        for mode in (999, "sleep", ""):
            with self.assertRaises(ValueError):
                app_endpoint_config("vllm", ["http://localhost:8000"], mode=mode)
            with self.assertRaises(ValueError):
                app_channel_config(mode=mode)

    def test_rejects_bare_string_endpoints_and_tags(self):
        with self.assertRaises(ValueError):
            app_endpoint_config("vllm", "http://localhost:8000")
        with self.assertRaises(ValueError):
            app_endpoint_config("vllm", ["http://localhost:8000"], tags="weights")
        with self.assertRaises(ValueError):
            app_channel_config(tags="weights")

    def test_rejects_bad_tags(self):
        for tags in ([""], [123], [None]):
            with self.assertRaises(ValueError):
                app_endpoint_config("vllm", ["http://localhost:8000"], tags=tags)
            with self.assertRaises(ValueError):
                app_channel_config(tags=tags)


if __name__ == "__main__":
    unittest.main()
