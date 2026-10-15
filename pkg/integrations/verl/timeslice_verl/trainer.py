"""TimesliceFullyAsyncTrainer: verl FullyAsyncTrainer subclass for time-slicing.

Combines :class:`timeslice_verl.hooks.TimesliceHooksMixin` (the lock
protocol; pure python) with verl's FullyAsyncTrainer, and registers the
result under the trainer name ``timeslice``. Importing this module is all
that is needed for registration; the package's ``verl.plugins`` entry point
(see pyproject.toml) makes ``import verl`` do that automatically in every
process where the package is installed.

Select the trainer with the hydra override:

    async_training.trainer_name=timeslice

The hooks stay env-gated (inert unless TIMESLICE_FULLY_ASYNC=1), so with the
gate off this trainer behaves exactly like the built-in FullyAsyncTrainer.
"""

from verl.experimental.fully_async_policy.fully_async_trainer import (
    FullyAsyncTrainer,
    register_trainer,
)

from timeslice_verl import defaults, weight_buffers
from timeslice_verl.hooks import TimesliceHooksMixin

# With the env gate on, default the settings time-slicing needs (see defaults.py).
defaults.apply()
# With its own gate on (TIMESLICE_PERSIST_WEIGHT_BUFFERS=1), keep the rollout
# side's weight-sync buffers allocated across syncs (see weight_buffers.py).
weight_buffers.apply()


@register_trainer("timeslice")
class TimesliceFullyAsyncTrainer(TimesliceHooksMixin, FullyAsyncTrainer):
    """FullyAsyncTrainer whose lifecycle hooks drive the timeslice group lock."""

    def __init__(self, **kwargs):
        TimesliceHooksMixin.__init__(self)
        FullyAsyncTrainer.__init__(self, **kwargs)

    async def _fit_update_weights(self) -> dict | None:
        """Crash-release coverage for the pre-fit initial param sync.

        fully_async_main invokes _fit_update_weights directly (outside
        fit(), whose catch-all fires on_exception), right after
        init_workers — where the hooks yielded the group lock and
        on_update_weights_begin re-acquires it. A crash in that window must
        still release the lock. _dispatch_on_exception fires on_exception at
        most once per escaping exception, so this never double-fires with
        fit()'s handler during training.
        """
        try:
            return await super()._fit_update_weights()
        except BaseException as e:
            await self._dispatch_on_exception("update_weights", e)
            raise

    def _fit_save_checkpoint(self, force=False):
        """A due save outside a locked window takes the group lock first
        (verl's final save after fit's loop); saves inside the window are
        already done by on_update_weights_end."""
        return self._save_with_lock(super()._fit_save_checkpoint, force=force)

    async def load_checkpoint(self):
        """Take the group lock before a resume loads the checkpoint into the
        (possibly parked) worker; crash-release like _fit_update_weights."""
        try:
            await self._lock_for_load()
            return await super().load_checkpoint()
        except BaseException as e:
            await self._dispatch_on_exception("load_checkpoint", e)
            raise
