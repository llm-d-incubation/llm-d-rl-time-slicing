"""Cross-language driver for the memory-regions backend (hermetic tier;
invoked by TestCrossLanguageMemoryRegions in memory_regions_test.go).

Runs against a live snapshot agent (standalone mode) whose cr_client is
stub_cr_client.sh. Exercises the real Python client and generated stubs
against the real Go server — this is the test that catches Go<->Python
codegen drift (e.g. the prototype's field-4 collision).

Env:
    AGENT_ENDPOINT  host:port of the agent
    CTL_DIR         the agent's EXPORT_FILE_PATH (shared with the stub)
    WORKLOAD_PID    pid of the live stand-in workload process (the backend
                    reads its /proc starttime to name the slot's owner dir)
"""

import os
import sys

from timeslice.snapshot_agent import (
    MemoryRegion,
    SnapshotAgentClient,
    memory_regions_config,
)

ADDRESS = 0x7F0000000000
SIZE = 64

CONTENT_A = b"\xa1" * SIZE
CONTENT_B = b"\xb2" * SIZE


def write_device(ctl_dir: str, content: bytes) -> None:
    with open(os.path.join(ctl_dir, "device"), "wb") as f:
        f.write(content)


def read_device(ctl_dir: str) -> bytes:
    with open(os.path.join(ctl_dir, "device"), "rb") as f:
        return f.read()


def expect_complete(result, what: str) -> None:
    if result.status != "OPERATION_STATUS_COMPLETE":
        raise SystemExit(f"{what} failed: {result.status} {result.error!r}")


def main() -> None:
    endpoint = os.environ["AGENT_ENDPOINT"]
    ctl_dir = os.environ["CTL_DIR"]
    pid = int(os.environ["WORKLOAD_PID"])
    job_id = "cross-lang-job"

    regions = [MemoryRegion(pid=pid, address=ADDRESS, size_bytes=SIZE)]

    with SnapshotAgentClient(endpoint=endpoint) as client:
        health = client.check_health("memory-regions")
        if health.status != "SERVING":
            raise SystemExit(f"memory-regions backend not SERVING: {health.status}")

        # Snapshot device state A into slot-a, restore to return to RUNNING.
        write_device(ctl_dir, CONTENT_A)
        expect_complete(
            client.snapshot_and_wait(
                job_id=job_id,
                backend_config=memory_regions_config(regions, snapshot_name="slot-a"),
            ),
            "snapshot slot-a",
        )
        expect_complete(
            client.restore_and_wait(
                job_id=job_id,
                backend_config=memory_regions_config(regions, snapshot_name="slot-a"),
            ),
            "restore slot-a",
        )

        # Snapshot device state B into slot-b.
        write_device(ctl_dir, CONTENT_B)
        expect_complete(
            client.snapshot_and_wait(
                job_id=job_id,
                backend_config=memory_regions_config(regions, snapshot_name="slot-b"),
            ),
            "snapshot slot-b",
        )

        # Alternate slots (live swap while RUNNING) and verify bitwise.
        for slot, expected in [
            ("slot-a", CONTENT_A),
            ("slot-b", CONTENT_B),
            ("slot-a", CONTENT_A),
            ("slot-b", CONTENT_B),
        ]:
            expect_complete(
                client.restore_and_wait(
                    job_id=job_id,
                    backend_config=memory_regions_config(regions, snapshot_name=slot),
                ),
                f"restore {slot}",
            )
            got = read_device(ctl_dir)
            if got != expected:
                raise SystemExit(
                    f"device bytes after restoring {slot} do not match: "
                    f"got {got[:8].hex()}..., want {expected[:8].hex()}..."
                )

    print("CROSS-LANG-OK")


if __name__ == "__main__":
    sys.exit(main())
