# This module:
# 1. Implements the accelerator contract for NVIDIA CUDA execution, including
#    process-wide current-device selection and multi-device counting
#
# Design decisions:
# - setup_device also calls torch.cuda.set_device so subsequent context-free
#   CUDA allocations land on the bound device instead of device zero
# - get_device falls back to cuda:0 before any binding, so early queries during
#   trainer construction return a usable default instead of failing
# - The device slot is Optional, distinguishing "not yet bound" from an actual
#   binding; the other backends can bind eagerly because they have one device
# - auto_device_count reports torch.cuda.device_count, backing multi-GPU
#   configuration resolution
#
# Author: Rahul Sawhney

from typing import override

import torch

from syntheticmind.accelerators.accelerator import Accelerator

__all__: list[str] = ["CUDAAccelerator"]


class CUDAAccelerator(Accelerator):
    # Accelerator for CUDA devices. Binding sets the process-wide current
    # device; queries before binding degrade to the first CUDA device.
    def __init__(self) -> None:
        # Starts unbound; the concrete device index arrives via setup_device
        # once the strategy has resolved placement.
        self._device: torch.device | None = None

    @override
    def setup_device(self, device: torch.device) -> None:
        # Binds the device and makes it the process-wide current CUDA device so
        # context-free allocations follow the binding.
        self._device: torch.device | None = device
        torch.cuda.set_device(device)

    @override
    def get_device(self) -> torch.device:
        # Returns the bound device, or cuda:0 when queried before binding.
        if self._device is None:
            return torch.device("cuda", 0)
        return self._device

    @override
    def auto_device_count(self) -> int:
        # Number of visible CUDA devices on this host.
        return torch.cuda.device_count()

    @staticmethod
    @override
    def is_available() -> bool:
        # Whether a usable CUDA runtime and at least one device are present.
        return torch.cuda.is_available()

    @staticmethod
    @override
    def name() -> str:
        # Canonical backend name used in configuration and log output.
        return "cuda"
