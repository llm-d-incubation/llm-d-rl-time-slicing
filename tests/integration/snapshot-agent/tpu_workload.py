#!/usr/bin/env python3
"""Multi-process TPU workload for the TPU integration test (see tpu.go).

Spawns TPU_NPROC single-chip JAX worker processes (the standard
chip-splitting env recipe), each running small TPU matmuls in a loop.
Every worker keeps a status file in TPU_STATE_DIR current:

    status-<i>:  "run <step>" | "parked <step>"

and obeys the quiesce flag file:

  - while TPU_STATE_DIR/quiesce exists the worker polls on pure CPU and
    issues NO TPU ops — the state the libtpu checkpoint contract requires;
  - once removed, the worker computes again, advancing its step counter —
    the test's proof that restore brought the chip back.

The parent touches TPU_STATE_DIR/ready (the pod's readiness probe) once
every worker has completed a step on its chip.
"""

import multiprocessing
import os
import sys
import time

NPROC = int(os.environ.get("TPU_NPROC", "8"))
STATE_DIR = os.environ.get("TPU_STATE_DIR", "/workload-state")
QUIESCE = os.path.join(STATE_DIR, "quiesce")


def write_status(i, state, step):
    # Single small write; readers tolerate a torn line by retrying.
    with open(os.path.join(STATE_DIR, f"status-{i}"), "w") as f:
        f.write(f"{state} {step}")


def worker(i):
    # Give each worker its own single-chip "slice": libtpu carves the host
    # per-process instead of forming one mesh.
    os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
    os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
    os.environ["TPU_PROCESS_ADDRESSES"] = f"localhost:{8476 + i}"
    os.environ["TPU_PROCESS_PORT"] = str(8476 + i)
    os.environ["CLOUD_TPU_TASK_ID"] = "0"
    os.environ["TPU_VISIBLE_CHIPS"] = str(i)
    os.environ["TPU_VISIBLE_DEVICES"] = str(i)
    # GKE-injected slice topology would fight the per-process env above.
    for gke_var in ("TPU_WORKER_ID", "TPU_WORKER_HOSTNAMES", "TPU_TOPOLOGY", "TPU_ACCELERATOR_TYPE"):
        os.environ.pop(gke_var, None)

    import jax
    import jax.numpy as jnp

    print(f"worker {i} pid {os.getpid()} devices={jax.devices()}", flush=True)

    x = jnp.ones((512, 512))
    step = 0
    parked = False
    while True:
        if os.path.exists(QUIESCE):
            if not parked:
                parked = True
                write_status(i, "parked", step)
                print(f"worker {i} pid {os.getpid()} parked at step {step}", flush=True)
            time.sleep(0.5)
            continue
        if parked:
            parked = False
            print(f"worker {i} pid {os.getpid()} resuming at step {step}", flush=True)
        x = x @ x
        x = x / (jnp.linalg.norm(x) + 1e-6)
        x.block_until_ready()
        step += 1
        write_status(i, "run", step)
        time.sleep(0.1)


def main():
    os.makedirs(STATE_DIR, exist_ok=True)
    multiprocessing.set_start_method("spawn")
    procs = [multiprocessing.Process(target=worker, args=(i,), daemon=True) for i in range(NPROC)]
    for p in procs:
        p.start()

    # Ready once every worker has computed at least one step.
    while True:
        done = 0
        for i in range(NPROC):
            try:
                with open(os.path.join(STATE_DIR, f"status-{i}")) as f:
                    if f.read().split()[1] != "0":
                        done += 1
            except (OSError, IndexError):
                pass
        if done == NPROC:
            break
        for p in procs:
            if not p.is_alive():
                print(f"worker exited during startup (exitcode {p.exitcode})", file=sys.stderr, flush=True)
                return 1
        time.sleep(1)
    with open(os.path.join(STATE_DIR, "ready"), "w"):
        pass
    print("all workers computing; ready", flush=True)

    for p in procs:
        p.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
