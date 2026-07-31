# This module:
# 1. Implements the accelerator contract for host-CPU execution
#
# Design decisions:
# - The CPU backend is always available and reports exactly one device, making
#   it the guaranteed fallback when no hardware backend is usable
# - setup_device only records the device because CPU placement requires no
#   backend-side preparation
#
# Author: Rahul Sawhney

from typing import override

import torch

from syntheticmind.accelerators.accelerator import Accelerator

__all__: list[str] = ["CPUAccelerator"]


class CPUAccelerator(Accelerator):
    # Accelerator bound to the host CPU. Serves as the universal fallback
    # backend; every query is trivial because there is only one CPU device.
    def __init__(self) -> None:
        # Binds the CPU device immediately; no later setup is required.
        self._device: torch.device = torch.device("cpu")

    @override
    def setup_device(self, device: torch.device) -> None:
        # Records the requested device; the CPU backend needs no preparation.
        self._device: torch.device = device

    @override
    def get_device(self) -> torch.device:
        # Returns the bound CPU device.
        return self._device

    @override
    def auto_device_count(self) -> int:
        # The host exposes exactly one CPU device to the harness.
        return 1

    @staticmethod
    @override
    def is_available() -> bool:
        # CPU execution is always available.
        return True

    @staticmethod
    @override
    def name() -> str:
        # Canonical backend name used in configuration and log output.
        return "cpu"
