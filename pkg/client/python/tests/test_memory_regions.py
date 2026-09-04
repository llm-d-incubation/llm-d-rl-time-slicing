"""Tests for the MemoryRegion type and the memory-regions config builder."""

import unittest

from timeslice.snapshot_agent import MemoryRegion, memory_regions_config
from timeslice.snapshot_agent import snapshot_agent_pb2


class TestMemoryRegionFromSpec(unittest.TestCase):
    def test_parses_hex_address(self):
        self.assertEqual(
            MemoryRegion.from_spec("123:0x7f00:1024"),
            MemoryRegion(pid=123, address=0x7F00, size_bytes=1024),
        )

    def test_parses_decimal_address(self):
        self.assertEqual(
            MemoryRegion.from_spec("123:139637976727552:1073741824"),
            MemoryRegion(pid=123, address=139637976727552, size_bytes=1073741824),
        )

    def test_parses_zero_address(self):
        self.assertEqual(
            MemoryRegion.from_spec("1:0x0:1"),
            MemoryRegion(pid=1, address=0, size_bytes=1),
        )

    def test_parses_hex_size(self):
        """Size accepts hex literals, not just the address."""
        self.assertEqual(
            MemoryRegion.from_spec("42:0xABCD:0x100"),
            MemoryRegion(pid=42, address=0xABCD, size_bytes=256),
        )

    def test_rejects_malformed_specs(self):
        """Wrong field count or non-numeric fields are a ValueError."""
        for spec in (
            "invalid-format",
            "123:0x7f00",  # missing size
            "123:0x7f00:1024:extra",
            "abc:0x7f00:1024",  # non-numeric pid
            "123:zz:1024",  # bad address
            "123:0x7f00:big",  # bad size
            "",
        ):
            with self.assertRaises(ValueError):
                MemoryRegion.from_spec(spec)


class TestMemoryRegionValidation(unittest.TestCase):
    def test_rejects_non_positive_pid(self):
        for pid in (0, -1):
            with self.assertRaises(ValueError):
                MemoryRegion(pid=pid, address=1, size_bytes=1)

    def test_rejects_negative_address(self):
        with self.assertRaises(ValueError):
            MemoryRegion(pid=1, address=-1, size_bytes=1)

    def test_rejects_non_positive_size(self):
        for size_bytes in (0, -5):
            with self.assertRaises(ValueError):
                MemoryRegion(pid=1, address=1, size_bytes=size_bytes)

    def test_accepts_wire_format_maxima(self):
        """The largest representable values (int32 pid, uint64 fields) pass."""
        region = MemoryRegion(pid=2**31 - 1, address=2**64 - 1, size_bytes=2**64 - 1)
        self.assertEqual(region.pid, 2**31 - 1)
        self.assertEqual(region.address, 2**64 - 1)
        self.assertEqual(region.size_bytes, 2**64 - 1)

    def test_rejects_pid_above_int32(self):
        with self.assertRaises(ValueError):
            MemoryRegion(pid=2**31, address=1, size_bytes=1)

    def test_rejects_address_above_uint64(self):
        with self.assertRaises(ValueError):
            MemoryRegion(pid=1, address=2**64, size_bytes=1)

    def test_rejects_size_above_uint64(self):
        with self.assertRaises(ValueError):
            MemoryRegion(pid=1, address=1, size_bytes=2**64)

    def test_rejects_non_integer_fields(self):
        """Non-int values (including bool, an int subclass) are a TypeError."""
        for kwargs in (
            {"pid": 1.5, "address": 1, "size_bytes": 1},
            {"pid": "1", "address": 1, "size_bytes": 1},
            {"pid": True, "address": 1, "size_bytes": 1},
            {"pid": 1, "address": 1.0, "size_bytes": 1},
            {"pid": 1, "address": True, "size_bytes": 1},
            {"pid": 1, "address": 1, "size_bytes": "1024"},
            {"pid": 1, "address": 1, "size_bytes": True},
        ):
            with self.assertRaises(TypeError):
                MemoryRegion(**kwargs)


class TestMemoryRegionsConfig(unittest.TestCase):
    def test_builds_config_from_dataclasses(self):
        cfg = memory_regions_config(
            [MemoryRegion(pid=123, address=0x7F00, size_bytes=1024)],
            snapshot_name="slot-a",
        )
        self.assertEqual(cfg.WhichOneof("backend"), "memory_regions")
        self.assertEqual(cfg.memory_regions.snapshot_name, "slot-a")
        self.assertEqual(len(cfg.memory_regions.regions), 1)
        region = cfg.memory_regions.regions[0]
        self.assertEqual(region.pid, 123)
        self.assertEqual(region.address, 0x7F00)
        self.assertEqual(region.size_bytes, 1024)

    def test_builds_config_from_tuples(self):
        cfg = memory_regions_config([(123, 0x7F00, 1024)])
        region = cfg.memory_regions.regions[0]
        self.assertEqual(region.pid, 123)
        self.assertEqual(region.address, 0x7F00)
        self.assertEqual(region.size_bytes, 1024)

    def test_builds_config_from_spec_strings(self):
        cfg = memory_regions_config(["123:0x7f00:1024"])
        region = cfg.memory_regions.regions[0]
        self.assertEqual(region.pid, 123)
        self.assertEqual(region.address, 0x7F00)
        self.assertEqual(region.size_bytes, 1024)

    def test_builds_config_from_mixed_region_kinds(self):
        """Dataclasses, tuples, and spec strings can be mixed in one call."""
        cfg = memory_regions_config(
            [
                MemoryRegion(pid=123, address=0x7F00, size_bytes=1024),
                (123, 0x8F00, 2048),
                "456:0x9f00:4096",
            ]
        )
        self.assertEqual(len(cfg.memory_regions.regions), 3)
        self.assertEqual(cfg.memory_regions.regions[1].address, 0x8F00)
        self.assertEqual(cfg.memory_regions.regions[2].pid, 456)

    def test_snapshot_name_defaults_to_empty(self):
        """Empty snapshot_name defers slot naming to the agent (job_id)."""
        cfg = memory_regions_config([(1, 0x10, 16)])
        self.assertEqual(cfg.memory_regions.snapshot_name, "")

    def test_large_address_round_trips(self):
        """Addresses above 2**32 survive serialization (uint64 on the wire)."""
        addr = 139637976727552
        cfg = memory_regions_config([(123, addr, 2**33)])
        self.assertEqual(cfg.memory_regions.regions[0].address, addr)
        self.assertEqual(cfg.memory_regions.regions[0].size_bytes, 2**33)
        parsed = snapshot_agent_pb2.BackendConfig.FromString(cfg.SerializeToString())
        self.assertEqual(parsed.memory_regions.regions[0].address, addr)

    def test_rejects_empty_regions(self):
        for regions in ([], None):
            with self.assertRaisesRegex(ValueError, "at least one memory region"):
                memory_regions_config(regions)

    def test_rejects_invalid_regions(self):
        for region in (
            "bad-spec",
            (1, 2),  # wrong arity
            (0, 2, 3),  # zero pid
            (1, 2, 0),  # zero size
        ):
            with self.assertRaises(ValueError):
                memory_regions_config([region])

    def test_rejects_unsupported_region_type(self):
        with self.assertRaises(TypeError):
            memory_regions_config([12345])


if __name__ == "__main__":
    unittest.main()
