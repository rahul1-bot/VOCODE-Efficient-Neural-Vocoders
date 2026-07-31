# This module:
# 1. Measures wall-clock durations for the whole fit, each training epoch, and
#    each validation pass, and reports them through the log at rank zero
#
# Design decisions:
# - Durations are measured with the monotonic performance counter, so system
#   clock adjustments cannot corrupt them
# - One epoch-start timestamp slot serves both training epochs and validation
#   passes, which is sound because the harness never interleaves the start of
#   one with the end of the other on a single process
# - Timestamps are recorded on every rank so accumulated totals stay
#   meaningful everywhere, while log emission is restricted to rank zero to
#   keep distributed output uncluttered
#
# Author: Rahul Sawhney

import time
from typing import TYPE_CHECKING, override

from loguru import logger as log

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.utilities.distributed import is_rank_zero

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["TimerCallback"]


class TimerCallback(Callback):
    # Wall-clock timing over the fit lifecycle: per-epoch training durations,
    # per-pass validation durations, and the final summary splitting total
    # time into its training and validation components.
    def __init__(self) -> None:
        # Zeroes the fit-start and epoch-start timestamps and the accumulated
        # training and validation totals.
        super().__init__()
        self._fit_start: float = 0.0
        self._epoch_start: float = 0.0
        self._total_train_time: float = 0.0
        self._total_val_time: float = 0.0

    @override
    def on_fit_start(self, trainer: Trainer, module: Module) -> None:
        # Marks the beginning of the fit for the end-of-run total.
        self._fit_start: float = time.perf_counter()

    @override
    def on_fit_end(self, trainer: Trainer, module: Module) -> None:
        # Reports the total fit duration together with the accumulated
        # training and validation components, from rank zero only.
        if not is_rank_zero():
            return
        total: float = time.perf_counter() - self._fit_start
        log.info(
            f"Training complete: {total:.1f}s total, "
            f"{self._total_train_time:.1f}s training, "
            f"{self._total_val_time:.1f}s validation"
        )

    @override
    def on_train_epoch_start(self, trainer: Trainer, module: Module) -> None:
        # Marks the beginning of the training epoch.
        self._epoch_start: float = time.perf_counter()

    @override
    def on_train_epoch_end(self, trainer: Trainer, module: Module) -> None:
        # Accumulates the epoch duration into the training total and reports
        # it from rank zero.
        elapsed: float = time.perf_counter() - self._epoch_start
        self._total_train_time += elapsed
        if is_rank_zero():
            log.info(f"Epoch {module.current_epoch} training: {elapsed:.1f}s")

    @override
    def on_validation_epoch_start(self, trainer: Trainer, module: Module) -> None:
        # Marks the beginning of the validation pass.
        self._epoch_start: float = time.perf_counter()

    @override
    def on_validation_epoch_end(
        self, trainer: Trainer, module: Module, metrics: dict[str, float]
    ) -> None:
        # Accumulates the pass duration into the validation total and reports
        # it from rank zero.
        elapsed: float = time.perf_counter() - self._epoch_start
        self._total_val_time += elapsed
        if is_rank_zero():
            log.info(f"Validation: {elapsed:.1f}s")
