"""TimesliceCallback: PhaseCallback for cooperative GPU time-slicing.

Maps Slime driver phase transitions to orchestrator lock acquire/release
so multiple RL jobs can share accelerator hardware.

Lock protocol per phase:

  begin("init", "both")         → acquire TRAINER, acquire SAMPLER
  end("init", "both")           → release SAMPLER, release TRAINER

  begin("generate", "sampler")  → acquire SAMPLER
  end("generate", "sampler")    → release SAMPLER

  begin("train", "trainer")     → acquire TRAINER
  end("train", "trainer")       → release TRAINER

  begin("weight_sync", "both")  → acquire TRAINER, acquire SAMPLER
  end("weight_sync", "both")    → release SAMPLER, release TRAINER

  begin("offload", "sampler")   → acquire SAMPLER
  end("offload", "sampler")     → release SAMPLER

  begin("onload", "sampler")    → acquire SAMPLER
  end("onload", "sampler")      → release SAMPLER

  begin("eval", "sampler")      → acquire SAMPLER
  end("eval", "sampler")        → release SAMPLER

  begin("create", "trainer")    → acquire TRAINER
  end("create", "trainer")      → release TRAINER

Env-gated: no-op when TIMESLICE_JOB_ID is not set. Same image works
with and without the time-slicing platform.

Environment variables:
  TIMESLICE_JOB_ID          unique job identifier (e.g. "job-a")
  TIMESLICE_ORCH_ADDR       orchestrator gRPC address (e.g. "localhost:50051")
  TIMESLICE_TRAINER_GROUP   trainer pool group id (e.g. "trainers")
  TIMESLICE_SAMPLER_GROUP   sampler pool group id (e.g. "samplers")
"""

from slime.utils.phase_callback import PhaseCallback

from timeslice_slime.locks import SAMPLER, TRAINER, RoleLocks


class TimesliceCallback(PhaseCallback):
    def __init__(self):
        self.locks = RoleLocks.from_env()

    def on_phase_begin(self, phase, role, context=None):
        if role in ("trainer", "both"):
            self.locks.acquire(TRAINER)
        if role in ("sampler", "both"):
            self.locks.acquire(SAMPLER)

    def on_phase_end(self, phase, role, context=None):
        if role in ("sampler", "both"):
            self.locks.release(SAMPLER)
        if role in ("trainer", "both"):
            self.locks.release(TRAINER)

    def close(self):
        self.locks.close()
