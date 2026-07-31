# This module:
# 1. Renders terminal progress bars for training epochs, validation passes,
#    and test passes, with live metric postfixes on the training and
#    evaluation bars
#
# Design decisions:
# - Bars are created only on rank zero, so distributed runs render exactly one
#   set of bars; on other ranks every handler observes an absent bar and
#   returns immediately
# - The training bar's postfix shows only metrics whose logging configuration
#   requested progress-bar visibility, read live from the module's logging
#   buffer before the loop drains it, which works because the loops invoke
#   callback batch-end hooks before metric dispatch
# - Bar totals come from the trainer's cached batch counts and may be None
#   for iterable datasets, which tqdm renders as an open-ended counter
# - The training and test bars persist on screen as an epoch record while the
#   validation bar clears itself, keeping epoch summaries readable
# - Evaluation bars stamp their final reduced metrics as the closing postfix,
#   leaving the last rendered state meaningful
#
# Author: Rahul Sawhney

from typing import TYPE_CHECKING, override

import torch
from tqdm import tqdm

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.utilities.distributed import is_rank_zero
from syntheticmind.utilities.types import Batch, StepOutput

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["ProgressBarCallback"]


class ProgressBarCallback(Callback):
    # Terminal progress rendering over the training, validation, and test
    # loops. One bar exists per active stage on rank zero; batch-end hooks
    # advance it and epoch-end hooks close it.
    def __init__(self) -> None:
        # Prepares the three bar slots; each is created at its stage's epoch
        # start and cleared again when the stage ends.
        super().__init__()
        self._train_bar: tqdm | None = None  # type: ignore[type-arg]
        self._val_bar: tqdm | None = None  # type: ignore[type-arg]
        self._test_bar: tqdm | None = None  # type: ignore[type-arg]

    @override
    def on_train_epoch_start(self, trainer: Trainer, module: Module) -> None:
        # Opens the persistent training bar for the epoch, labeled with the
        # epoch position and sized from the trainer's cached batch total.
        if not is_rank_zero():
            return
        total: int | None = trainer._total_train_batches
        epoch: int = module.current_epoch
        max_epochs: int = trainer.max_epochs
        self._train_bar: tqdm | None = tqdm(
            total=total,
            desc=f"Train [{epoch}/{max_epochs - 1}]",
            leave=True,
            dynamic_ncols=True
        )

    @override
    def on_train_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int
    ) -> None:
        # Advances the training bar and refreshes its postfix with the
        # progress-bar-visible metrics currently buffered on the module,
        # which the loop has not yet drained at this hook.
        if self._train_bar is None:
            return
        self._train_bar.update(1)
        postfix: dict[str, str] = {}
        for name, metric in module.logged_metrics.items():
            if metric.progress_bar:
                value: torch.Tensor | float = metric.value
                if isinstance(value, torch.Tensor):
                    value: torch.Tensor | float = value.item()
                postfix[name] = f"{value:.4f}"
        if postfix:
            self._train_bar.set_postfix(postfix)

    @override
    def on_train_epoch_end(self, trainer: Trainer, module: Module) -> None:
        # Closes the training bar at the epoch boundary, leaving its final
        # state on screen as the epoch record.
        if self._train_bar is not None:
            self._train_bar.close()
            self._train_bar: tqdm | None = None

    @override
    def on_validation_epoch_start(self, trainer: Trainer, module: Module) -> None:
        # Opens the transient validation bar sized from the cached validation
        # batch total; it clears itself from the terminal when closed.
        if not is_rank_zero():
            return
        total: int | None = trainer._total_val_batches
        self._val_bar: tqdm | None = tqdm(
            total=total,
            desc="Validation",
            leave=False,
            dynamic_ncols=True
        )

    @override
    def on_validation_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Advances the validation bar by one batch.
        if self._val_bar is not None:
            self._val_bar.update(1)

    @override
    def on_validation_epoch_end(
        self, trainer: Trainer, module: Module, metrics: dict[str, float]
    ) -> None:
        # Stamps the reduced validation metrics as the closing postfix and
        # closes the bar.
        if self._val_bar is not None:
            if metrics:
                self._val_bar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})
            self._val_bar.close()
            self._val_bar: tqdm | None = None

    @override
    def on_test_epoch_start(self, trainer: Trainer, module: Module) -> None:
        # Opens the persistent test bar sized from the cached test batch
        # total.
        if not is_rank_zero():
            return
        total: int | None = trainer._total_test_batches
        self._test_bar: tqdm | None = tqdm(
            total=total,
            desc="Test",
            leave=True,
            dynamic_ncols=True
        )

    @override
    def on_test_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Advances the test bar by one batch.
        if self._test_bar is not None:
            self._test_bar.update(1)

    @override
    def on_test_epoch_end(
        self, trainer: Trainer, module: Module, metrics: dict[str, float]
    ) -> None:
        # Stamps the reduced test metrics as the closing postfix and closes
        # the bar.
        if self._test_bar is not None:
            if metrics:
                self._test_bar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})
            self._test_bar.close()
            self._test_bar: tqdm | None = None
