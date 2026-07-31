# This module:
# 1. Defines the abstract accelerator contract: binding a concrete device,
#    reporting the bound device, counting available devices, and answering
#    availability and name queries per backend
#
# Design decisions:
# - Device binding (setup_device) is separated from device queries (get_device)
#   so the trainer can resolve placement once and read it many times
# - is_available and name are static because they describe the backend itself,
#   not a bound instance; availability is queried before any instance exists
# - auto_device_count backs the "use all devices" configuration path, letting
#   each backend report its own notion of how many devices are usable
#
# Author: Rahul Sawhney

from abc import ABC, abstractmethod

import torch

__all__: list[str] = ["Accelerator"]


class Accelerator(ABC):
    # Abstract base for device backends. A concrete accelerator owns exactly one
    # bound torch.device and answers backend-level questions; all placement
    # decisions above it live in the strategy and trainer layers.
    @abstractmethod
    def setup_device(self, device: torch.device) -> None:
        # Binds the concrete device this accelerator will report and prepares
        # any backend-side state for it.
        raise NotImplementedError

    @abstractmethod
    def get_device(self) -> torch.device:
        # Returns the bound device, or the backend's default when none is bound.
        raise NotImplementedError

    @abstractmethod
    def auto_device_count(self) -> int:
        # Number of devices this backend can drive on the current host.
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        # Whether the backend is usable on the current host; consulted before
        # any accelerator instance is constructed.
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def name() -> str:
        # Canonical backend name used in configuration and log output.
        raise NotImplementedError

    def __repr__(self) -> str:
        # Compact identity line for logs and interactive debugging.
        return f"{type(self).__name__}()"
