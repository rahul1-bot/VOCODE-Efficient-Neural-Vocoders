# This module:
# 1. Records CUDA memory statistics (allocated and reserved megabytes for the
#    current device) into the experiment logger on a step cadence
#
# Design decisions:
# - Allocated and reserved memory are reported side by side because their gap
#   exposes allocator fragmentation, which allocated alone cannot show
# - The monitor degrades to a no-op without CUDA or without a logger, so it is
#   safe to include unconditionally in device-agnostic configurations
# - Emission is throttled to the configured step cadence and restricted to
#   rank zero
#
# Author: Rahul Sawhney

from typing import TYPE_CHECKING, override

import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.utilities.distributed import is_rank_zero
from syntheticmind.utilities.types import Batch, StepOutput

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["DeviceStatsMonitor"]


class DeviceStatsMonitor(Callback):
    # CUDA memory telemetry on the training batch cadence. Each emission
    # reports the allocated and reserved bytes of the current device in
    # megabytes against the global step.
    def __init__(self, log_every_n_steps: int = 50) -> None:
        # Binds the emission cadence in optimizer steps.
        super().__init__()
        self.log_every_n_steps: int = log_every_n_steps

    @override
    def on_train_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int
    ) -> None:
        # Reads the current device's allocated and reserved memory and emits
        # both in megabytes, gated on rank zero, the step cadence, CUDA
        # availability, and logger presence.
        if not is_rank_zero():
            return
        if module.global_step % self.log_every_n_steps != 0:
            return
        if not torch.cuda.is_available():
            return
        device_idx: int = torch.cuda.current_device()
        allocated_mb: float = torch.cuda.memory_allocated(device_idx) / (1024 * 1024)
        reserved_mb: float = torch.cuda.memory_reserved(device_idx) / (1024 * 1024)
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {"device/memory_allocated_mb": allocated_mb, "device/memory_reserved_mb": reserved_mb},
                step=module.global_step
            )
