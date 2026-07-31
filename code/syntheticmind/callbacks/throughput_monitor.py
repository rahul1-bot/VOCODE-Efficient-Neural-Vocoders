# This module:
# 1. Measures training throughput at two granularities: per-batch samples per
#    second and batch duration emitted to the experiment logger on a step
#    cadence, and a whole-epoch samples-per-second summary written to the log
#
# Design decisions:
# - The batch size is inferred structurally from the leading dimension of the
#   first tensor found in the batch (direct tensor, first mapping value, or
#   first sequence element), and defaults to one when no tensor is found, so
#   the monitor works without a batch-format contract
# - Per-batch emission is throttled to the configured step cadence to keep
#   logger volume proportional to run length, while the epoch summary is
#   always produced
# - Measurement uses the monotonic performance counter, and emission is
#   restricted to rank zero
#
# Author: Rahul Sawhney

import time
from typing import TYPE_CHECKING, override

import torch
from loguru import logger as log

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.utilities.distributed import is_rank_zero
from syntheticmind.utilities.types import Batch, StepOutput

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["ThroughputMonitor"]


class ThroughputMonitor(Callback):
    # Training throughput telemetry. Batch boundaries provide the per-batch
    # duration and inferred sample count; the accumulated counts produce the
    # epoch-level summary at epoch end.
    def __init__(self, log_every_n_steps: int = 10) -> None:
        # Binds the per-batch emission cadence and zeroes the timing and
        # sample accumulators.
        super().__init__()
        self.log_every_n_steps: int = log_every_n_steps
        self._batch_start_time: float = 0.0
        self._epoch_start_time: float = 0.0
        self._epoch_samples: int = 0

    @override
    def on_train_epoch_start(self, trainer: Trainer, module: Module) -> None:
        # Marks the epoch start and resets the epoch sample counter.
        self._epoch_start_time: float = time.perf_counter()
        self._epoch_samples: int = 0

    @override
    def on_train_batch_start(
        self, trainer: Trainer, module: Module, batch: Batch, batch_idx: int
    ) -> None:
        # Marks the batch start for the per-batch duration measurement.
        self._batch_start_time: float = time.perf_counter()

    @override
    def on_train_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int
    ) -> None:
        # Computes the batch duration and inferred sample count, accumulates
        # the epoch total, and on the configured step cadence emits samples
        # per second and batch time to the experiment logger from rank zero.
        if not is_rank_zero():
            return
        batch_time: float = time.perf_counter() - self._batch_start_time
        batch_size: int = self._infer_batch_size(batch)
        self._epoch_samples += batch_size

        if module.global_step % self.log_every_n_steps == 0 and trainer.logger is not None:
            samples_per_sec: float = batch_size / batch_time if batch_time > 0 else 0.0
            trainer.logger.log_metrics(
                {"throughput/samples_per_sec": samples_per_sec, "throughput/batch_time_ms": batch_time * 1000},
                step=module.global_step
            )

    @override
    def on_train_epoch_end(self, trainer: Trainer, module: Module) -> None:
        # Writes the whole-epoch samples-per-second summary to the log from
        # rank zero.
        if not is_rank_zero():
            return
        epoch_time: float = time.perf_counter() - self._epoch_start_time
        if epoch_time > 0:
            log.info(f"Epoch throughput: {self._epoch_samples / epoch_time:.1f} samples/sec")

    @staticmethod
    def _infer_batch_size(batch: Batch) -> int:
        # Infers the sample count from the leading dimension of the first
        # tensor reachable in the batch structure: the batch itself, the first
        # mapping value, or the first sequence element. Structures without a
        # reachable tensor count as one sample.
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]
        if isinstance(batch, dict):
            first: torch.Tensor | None = next(iter(batch.values()), None)
            if isinstance(first, torch.Tensor):
                return first.shape[0]
        if isinstance(batch, (tuple, list)) and len(batch) > 0:
            first_item: torch.Tensor = batch[0]
            if isinstance(first_item, torch.Tensor):
                return first_item.shape[0]
        return 1
