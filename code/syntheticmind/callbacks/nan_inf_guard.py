# This module:
# 1. Aborts training immediately when the step loss becomes non-finite, and
#    optionally when any parameter gradient contains non-finite values
#
# Design decisions:
# - The guard raises rather than logging, because continuing to optimize on a
#   non-finite loss corrupts every subsequent step; the error names the batch,
#   epoch, and global step so the failure point is identifiable from the
#   message alone
# - A missing loss entry is treated as zero and therefore finite, so the guard
#   remains usable with manual-optimization steps that return no loss
# - Gradient checking is opt-in because scanning every gradient tensor on
#   every batch has measurable cost on large models
# - The check runs at batch end, after backward and optimizer handling, so a
#   non-finite gradient is caught in the same batch that produced it
#
# Author: Rahul Sawhney

from typing import TYPE_CHECKING, override

import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.utilities.types import Batch, StepOutput

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["NanInfGuard"]


class NanInfGuard(Callback):
    # Numerical-stability tripwire over the training loop. Every batch's loss
    # is checked for NaN and infinity, and with parameter checking enabled the
    # gradients are scanned as well; any violation terminates the run with a
    # positioned error.
    def __init__(self, check_parameters: bool = False) -> None:
        # Binds the optional gradient-scanning switch; loss checking is always
        # active.
        super().__init__()
        self.check_parameters: bool = check_parameters

    @override
    def on_train_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int
    ) -> None:
        # Validates the batch outcome: the loss value is extracted (tensor or
        # scalar) and tested for finiteness, and when enabled every named
        # parameter's gradient is tested in full. Violations raise with the
        # batch index, epoch, and step embedded in the message.
        loss_value: torch.Tensor | float = outputs.get("loss", 0.0)
        if isinstance(loss_value, torch.Tensor):
            loss_value: torch.Tensor | float = loss_value.item()
        if not torch.isfinite(torch.tensor(loss_value)):
            raise ValueError(
                f"Non-finite loss detected at batch {batch_idx}, "
                f"epoch {module.current_epoch}, step {module.global_step}: {loss_value}"
            )
        if self.check_parameters:
            for name, param in module.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise ValueError(
                        f"Non-finite gradient in parameter '{name}' at batch {batch_idx}"
                    )
