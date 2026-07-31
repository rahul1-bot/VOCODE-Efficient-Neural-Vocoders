# This module:
# 1. Records the optimizer's learning rates into the experiment logger at the
#    end of every training epoch
#
# Design decisions:
# - Rates are read directly from the optimizer's parameter groups, which is
#   the authoritative value after any scheduler stepping, rather than from
#   scheduler state
# - A single parameter group logs under the plain lr key, while several groups
#   log under indexed keys so per-group schedules remain distinguishable
# - Emission is restricted to rank zero and degrades to a no-op without an
#   optimizer or logger, so the callback is safe to include unconditionally
#
# Author: Rahul Sawhney

from typing import TYPE_CHECKING, override

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.utilities.distributed import is_rank_zero

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["LearningRateMonitor"]


class LearningRateMonitor(Callback):
    # Epoch-cadence learning-rate telemetry. At each training epoch end the
    # current rate of every optimizer parameter group is emitted to the
    # experiment logger against the global step.
    def __init__(self, logging_interval: str = "epoch") -> None:
        # Binds the logging interval label and clears the last-logged marker.
        # Epoch-end emission is the implemented cadence.
        super().__init__()
        self.logging_interval: str = logging_interval
        self._last_logged_step: int = -1

    @override
    def on_train_epoch_end(
        self, trainer: Trainer, module: Module
    ) -> None:
        # Reads the learning rate of every parameter group from the primary
        # optimizer and emits the values from rank zero, using the plain lr
        # key for a single group and indexed keys for several.
        if not is_rank_zero():
            return
        if trainer.optimizer is None or trainer.logger is None:
            return

        current_step: int = module.global_step
        lr_metrics: dict[str, float] = {}
        for idx, group in enumerate(trainer.optimizer.param_groups):
            lr: float = group.get("lr", 0.0)
            if len(trainer.optimizer.param_groups) == 1:
                lr_metrics["lr"] = lr
            else:
                lr_metrics[f"lr_group_{idx}"] = lr

        if lr_metrics:
            trainer.logger.log_metrics(lr_metrics, step=current_step)
            self._last_logged_step: int = current_step
