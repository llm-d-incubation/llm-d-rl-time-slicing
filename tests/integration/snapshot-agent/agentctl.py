#!/usr/bin/env python3
"""CLI used by the integration tests to drive the snapshot-agent.

All snapshot/restore calls in the test suite go through this script so that
the tests exercise the real production path: the Python client
(timeslice.snapshot_agent) talking gRPC to the Go agent. Configs are
constructed here, in Python, the same way a real workload would build them.

Usage:
  agentctl.py --agent HOST:PORT snapshot|restore --job-id ID
              --backend cuda|tpu|app|channel|memory-regions
              [--pids 1,2,3]                       (cuda, tpu)
              [--app vllm|sglang] [--endpoints URL,URL]
              [--mode offload|discard] [--tags a,b] (app, channel)
              [--regions pid:0xADDR:size,...]      (memory-regions)
              [--snapshot-name SLOT]               (memory-regions)

Exits 0 when the operation completes, 1 otherwise.
"""

import argparse
import sys

from timeslice.snapshot_agent import (
    SnapshotAgentClient,
    memory_regions_config,
    snapshot_agent_pb2,
)

APPS = {
    "vllm": snapshot_agent_pb2.APP_VLLM,
    "sglang": snapshot_agent_pb2.APP_SGLANG,
}

MODES = {
    "": snapshot_agent_pb2.SUSPEND_MODE_UNSPECIFIED,
    "offload": snapshot_agent_pb2.SUSPEND_MODE_OFFLOAD,
    "discard": snapshot_agent_pb2.SUSPEND_MODE_DISCARD,
}


def build_config(args: argparse.Namespace) -> snapshot_agent_pb2.BackendConfig:
    if args.backend == "cuda":
        cuda = snapshot_agent_pb2.CudaBackendConfig()
        if args.pids:
            cuda.explicit_target.pids.extend(int(p) for p in args.pids.split(","))
        return snapshot_agent_pb2.BackendConfig(cuda=cuda)

    if args.backend == "tpu":
        tpu = snapshot_agent_pb2.TpuBackendConfig()
        if args.pids:
            tpu.explicit_target.pids.extend(int(p) for p in args.pids.split(","))
        return snapshot_agent_pb2.BackendConfig(tpu=tpu)

    if args.backend == "app":
        if args.app not in APPS:
            raise ValueError(f"--app is required for --backend app (one of {sorted(APPS)})")
        app_endpoint = snapshot_agent_pb2.AppEndpointConfig(
            app=APPS[args.app],
            endpoints=args.endpoints.split(",") if args.endpoints else [],
            mode=MODES[args.mode],
        )
        if args.tags:
            app_endpoint.tags.extend(args.tags.split(","))
        return snapshot_agent_pb2.BackendConfig(app_endpoint=app_endpoint)

    if args.backend == "channel":
        app_channel = snapshot_agent_pb2.AppChannelConfig(mode=MODES[args.mode])
        if args.tags:
            app_channel.tags.extend(args.tags.split(","))
        return snapshot_agent_pb2.BackendConfig(app_channel=app_channel)

    if args.backend == "memory-regions":
        if not args.regions:
            raise ValueError("--regions is required for --backend memory-regions")
        return memory_regions_config(args.regions.split(","), snapshot_name=args.snapshot_name)

    raise ValueError(f"unknown backend {args.backend!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["snapshot", "restore"])
    parser.add_argument("--agent", required=True, help="agent endpoint HOST:PORT")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--group", default="test")
    parser.add_argument("--backend", required=True, choices=["cuda", "tpu", "app", "channel", "memory-regions"])
    parser.add_argument("--pids", default="", help="comma-separated PIDs (cuda, tpu)")
    parser.add_argument("--app", default="", choices=["", "vllm", "sglang"], help="application (app backend)")
    parser.add_argument("--endpoints", default="", help="comma-separated application URLs")
    parser.add_argument("--mode", default="", choices=["", "offload", "discard"], help="suspend mode")
    parser.add_argument("--tags", default="", help="comma-separated region tags")
    parser.add_argument("--regions", default="", help="comma-separated pid:0xADDR:size specs (memory-regions)")
    parser.add_argument("--snapshot-name", default="", help="snapshot slot name (memory-regions; empty = job id)")
    args = parser.parse_args()

    config = build_config(args)

    with SnapshotAgentClient(args.agent) as client:
        if args.action == "snapshot":
            result = client.snapshot_and_wait(args.job_id, args.group, backend_config=config)
        else:
            result = client.restore_and_wait(args.job_id, args.group, backend_config=config)

    if result.status != "OPERATION_STATUS_COMPLETE":
        print(
            f"{args.action} {args.job_id} finished with {result.status}: {result.error}",
            file=sys.stderr,
        )
        return 1

    print(f"{args.action} {args.job_id} complete in {result.elapsed_ms}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
