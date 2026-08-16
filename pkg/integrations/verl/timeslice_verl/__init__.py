"""timeslice_verl: verl <-> llm-d-rl-time-slicing integration (fully-async).

TimesliceFullyAsyncTrainer (``timeslice_verl.trainer``) subclasses verl's
FullyAsyncTrainer, overriding its empty ``on_*`` lifecycle hooks to map the
trainer's phase transitions onto orchestrator group-lock acquire/release, so
two RL jobs can time-slice one shared trainer GPU. It registers under the
trainer name ``timeslice`` via verl's fully-async trainer registry
(``register_trainer``); the ``verl.plugins`` entry point declared in this
package's pyproject.toml makes ``import verl`` load the registering module
automatically. Select it with the hydra override
``async_training.trainer_name=timeslice`` — no verl source changes and no
monkey-patching.

This module deliberately imports only the verl-free parts (the lock protocol
lives in :class:`TimesliceHooksMixin`), so the package is importable and
testable without verl/ray/grpc/GPU.

Env-gated: the hooks are fully inert unless TIMESLICE_FULLY_ASYNC=1 (see
hooks.py for the complete environment contract).
"""

from timeslice_verl.hooks import TimesliceHooksMixin
from timeslice_verl.locks import PhaseLocks

__all__ = ["TimesliceHooksMixin", "PhaseLocks"]
