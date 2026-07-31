# This module:
# 1. Detects stalls between consecutive training batches by measuring the gap
#    between heartbeat timestamps and reporting an error when the gap exceeds
#    the configured timeout
#
# Design decisions:
# - Heartbeats are recorded at both batch start and batch end, so the measured
#   gap covers whichever segment stalled: the step itself, the optimizer
#   handling, or the data pipeline between batches
# - The watchdog only reports; it does not terminate the run, because a long
#   gap can be legitimate (first-batch compilation, checkpoint writes) and the
#   error log is the actionable signal for the operator
# - The first batch is exempt because no previous heartbeat exists to measure
#   against
# - The monotonic performance counter is used so system clock adjustments
#   cannot fabricate or hide a stall
#
# Author: Rahul Sawhney

import time
from typing import TYPE_CHECKING, override

from loguru import logger as log

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.utilities.types import Batch, StepOutput

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["WatchdogCallback"]


class WatchdogCallback(Callback):
    # Stall detector over the training batch cadence. An error is logged
    # whenever the time since the previous heartbeat exceeds the timeout,
    # identifying hangs in the data pipeline or the step without killing the
    # run.
    def __init__(self, timeout_seconds: float = 600.0) -> None:
        # Binds the stall threshold and clears the heartbeat timestamp; a
        # zero timestamp marks that no batch has been observed yet.
        #
        # Args:
        #     timeout_seconds: Maximum gap between consecutive batch
        #         heartbeats before the stall report is logged. The
        #         watchdog only reports; it never terminates the run.
        #         Default: ``600.0``.
        super().__init__()
        self.timeout_seconds: float = timeout_seconds
        self._last_heartbeat: float = 0.0

    @override
    def on_train_batch_start(
        self, trainer: Trainer, module: Module, batch: Batch, batch_idx: int
    ) -> None:
        # Measures the gap since the previous heartbeat, reports a stall when
        # it exceeds the timeout, and stamps the new heartbeat. The gap
        # measured here spans the previous batch end through this batch
        # start, which is where data-pipeline stalls appear.
        current_time: float = time.perf_counter()
        if self._last_heartbeat > 0:
            elapsed: float = current_time - self._last_heartbeat
            if elapsed > self.timeout_seconds:
                log.error(
                    f"Watchdog: {elapsed:.1f}s since last heartbeat "
                    f"(timeout={self.timeout_seconds:.1f}s) at batch {batch_idx}"
                )
        self._last_heartbeat: float = current_time

    @override
    def on_train_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int
    ) -> None:
        # Stamps the heartbeat after the step and optimizer handling, so the
        # next batch-start measurement isolates the inter-batch segment.
        self._last_heartbeat: float = time.perf_counter()
