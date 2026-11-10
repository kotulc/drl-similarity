"""Algorithm-neutral SB3 callback that triggers checkpoint + evaluation at fixed steps."""

from __future__ import annotations

from collections.abc import Callable

from stable_baselines3.common.callbacks import BaseCallback

from drl_similarity.logging import get_logger, log_extra


class CheckpointEvaluationCallback(BaseCallback):
    """Call ``on_checkpoint(step)`` every ``interval_steps`` and once at the end of training."""

    def __init__(
        self,
        *,
        interval_steps: int,
        log_interval_steps: int,
        on_checkpoint: Callable[[int], None],
        config_hash: str,
    ) -> None:
        super().__init__()
        self.interval_steps = interval_steps
        self.log_interval_steps = log_interval_steps
        self.on_checkpoint = on_checkpoint
        self._last_checkpoint = -1
        self._log = get_logger("train")
        self._extra = log_extra(stage="train", config_hash=config_hash)

    def _checkpoint(self) -> None:
        if self.num_timesteps != self._last_checkpoint:
            self.on_checkpoint(self.num_timesteps)
            self._last_checkpoint = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_interval_steps == 0:
            self._log.info("step %d", self.num_timesteps, extra=self._extra)
        if self.num_timesteps % self.interval_steps == 0:
            self._checkpoint()

        return True

    def _on_training_end(self) -> None:
        # The final step is always checkpointed, even off the interval grid
        self._checkpoint()
