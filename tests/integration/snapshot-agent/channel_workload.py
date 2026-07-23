#!/usr/bin/env python3
"""In-pod test workload for the app-channel backend.

Embeds vLLM through its Python API (no HTTP server) and registers with the
node's snapshot-agent over the workload channel — the integration path for
Python-API workloads such as RL samplers. The harness drives deterministic
generations through a file protocol:

  touch STATE/trigger  ->  completion text written to STATE/result

STATE/ready is created once the model is loaded and the workload is
registered; the pod's readiness probe watches it.
"""

import os
import time
from pathlib import Path

from timeslice.snapshot_agent import register_workload
from vllm import LLM, SamplingParams

STATE = Path("/workload-state")
PROMPT = "The capital of France is"


def main() -> None:
    STATE.mkdir(exist_ok=True)
    llm = LLM(
        model=os.environ["MODEL"],
        enable_sleep_mode=True,
        gpu_memory_utilization=0.5,
    )
    params = SamplingParams(temperature=0, max_tokens=15)

    job_id = os.environ["TIME_SLICE_JOB_ID"]
    handle = register_workload(
        os.environ["SNAPSHOT_AGENT_ADDR"],
        job_id=job_id,
        group=os.environ.get("TIME_SLICE_GROUP", "test"),
        workload=llm,
    )
    print(f"registered workload for job {job_id}", flush=True)

    (STATE / "ready").write_text("ok")
    trigger = STATE / "trigger"
    result = STATE / "result"
    try:
        while True:
            if trigger.exists():
                outputs = llm.generate([PROMPT], params)
                result.write_text(outputs[0].outputs[0].text)
                trigger.unlink()
            time.sleep(0.5)
    finally:
        handle.close()


if __name__ == "__main__":
    main()
