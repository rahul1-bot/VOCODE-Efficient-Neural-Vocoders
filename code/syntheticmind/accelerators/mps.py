# This module:
# 1. Implements the accelerator contract for Apple Silicon execution through
#    the Metal Performance Shaders backend
#
# Design decisions:
# - The MPS device binds eagerly at construction because Apple Silicon exposes
#   exactly one such device per host
# - setup_device only records the device; the MPS backend performs no
#   process-wide selection equivalent to CUDA's set_device
# - Availability delegates to torch.backends.mps, which covers both hardware
#   presence and build support in one check
#
# Author: Rahul Sawhney

from typing import override

import torch

from syntheticmind.accelerators.accelerator import Accelerator

__all__: list[str] = ["MPSAccelerator"]


class MPSAccelerator(Accelerator):
    # Accelerator for the Apple Silicon GPU. Single-device by construction;
    # binding and queries are correspondingly simple.
    def __init__(self) -> None:
        # Binds the MPS device immediately; there is only one per host.
        self._device: torch.device = torch.device("mps")

    @override
    def setup_device(self, device: torch.device) -> None:
        # Records the requested device; MPS needs no further preparation.
        self._device: torch.device = device

    @override
    def get_device(self) -> torch.device:
        # Returns the bound MPS device.
        return self._device

    @override
    def auto_device_count(self) -> int:
        # Apple Silicon exposes exactly one MPS device.
        return 1

    @staticmethod
    @override
    def is_available() -> bool:
        # Whether the torch build and host expose a usable MPS backend.
        return torch.backends.mps.is_available()

    @staticmethod
    @override
    def name() -> str:
        # Canonical backend name used in configuration and log output.
        return "mps"
