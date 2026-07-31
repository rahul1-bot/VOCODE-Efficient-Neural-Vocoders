# This module:
# 1. Implements the strategy contract for one process driving one device, with
#    direct delegation of every step to the module's own step methods
#
# Design decisions:
# - setup moves the model to the device and returns it unwrapped; there is no
#   container layer between the loops and the module in single-device execution
# - Every step delegation first type-guards that the object is a harness Module,
#   converting a mis-wired plain nn.Module into an immediate TypeError instead
#   of a later AttributeError deep inside a loop
# - backward is a plain loss.backward(); no gradient synchronization exists in
#   single-process execution
# - The inherited collective no-ops from Strategy are already correct for this
#   strategy, so nothing else is overridden
#
# Author: Rahul Sawhney

from typing import override

import torch
from torch import nn

from syntheticmind.core.module import Module
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.utilities.types import Batch, ModelOutput, StepOutput

__all__: list[str] = ["SingleDeviceStrategy"]


class SingleDeviceStrategy(Strategy):
    # Strategy for one process and one device. The model runs unwrapped, steps
    # delegate straight to the module, and no collective coordination exists.
    @override
    def setup_environment(self) -> None:
        # Single-process execution needs no environment preparation.
        pass

    @override
    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:
        # Binds the root device and moves the model onto it; the model is
        # returned unwrapped because no strategy container is needed.
        self._device: torch.device = device
        self._model: nn.Module | None = model.to(device)
        return self._model

    @override
    def training_step(
        self, model: nn.Module, batch: Batch, batch_idx: int
    ) -> StepOutput:
        # Guards the module type, then delegates to the model's training_step.
        if not isinstance(model, Module):
            raise TypeError(f"Expected Module, got {type(model).__name__}")
        output: StepOutput = model.training_step(batch, batch_idx)
        return output

    @override
    def validation_step(
        self, model: nn.Module, batch: Batch, batch_idx: int
    ) -> StepOutput:
        # Guards the module type, then delegates to the model's validation_step.
        if not isinstance(model, Module):
            raise TypeError(f"Expected Module, got {type(model).__name__}")
        output: StepOutput = model.validation_step(batch, batch_idx)
        return output

    @override
    def test_step(
        self, model: nn.Module, batch: Batch, batch_idx: int
    ) -> StepOutput:
        # Guards the module type, then delegates to the model's test_step.
        if not isinstance(model, Module):
            raise TypeError(f"Expected Module, got {type(model).__name__}")
        output: StepOutput = model.test_step(batch, batch_idx)
        return output

    @override
    def predict_step(
        self, model: nn.Module, batch: Batch, batch_idx: int
    ) -> ModelOutput:
        # Guards the module type, then delegates to the model's predict_step.
        if not isinstance(model, Module):
            raise TypeError(f"Expected Module, got {type(model).__name__}")
        output: ModelOutput = model.predict_step(batch, batch_idx)
        return output

    @override
    def backward(self, loss: torch.Tensor) -> None:
        # Plain autograd backward; nothing to synchronize in single-process
        # execution.
        loss.backward()
