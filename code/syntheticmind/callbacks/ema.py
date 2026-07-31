# This module:
# 1. Maintains an exponential moving average of the module parameters in a
#    gradient-free shadow copy updated after every training batch
# 2. Evaluates under the averaged weights by swapping them into the module for
#    validation, test, and prediction passes and restoring the live training
#    weights afterwards
# 3. Persists the shadow state into checkpoints and protects checkpoint
#    payloads written during a swapped evaluation
#
# Design decisions:
# - The shadow is a full deep copy created at fit setup with gradients
#   disabled, so averaging never interacts with autograd and the shadow
#   architecture always matches the module exactly
# - The per-batch update applies shadow = decay * shadow + (1 - decay) * live
#   under no_grad, with strict parameter pairing so a structural mismatch
#   fails loudly instead of averaging misaligned tensors
# - The evaluation swap snapshots the complete live state dictionary (clones
#   included) before copying shadow parameters in, and the restore loads that
#   snapshot back; only parameters are averaged and swapped, while buffers
#   remain those of the live module throughout
# - Checkpoints written during an evaluation swap would otherwise persist the
#   averaged weights as the training weights; the save hook overwrites the
#   payload's model state with the cloned pre-swap snapshot so resumption
#   always continues from the true training weights, and stores the shadow
#   state under its own dedicated key
# - The load hook restores the shadow from its dedicated key only when the
#   shadow exists, which requires setup to have run for the fit stage first
#
# Author: Rahul Sawhney

import copy
from typing import TYPE_CHECKING, override

import torch
from torch import nn

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.utilities.types import Batch, CheckpointDict, CheckpointValue, StepOutput, TrainerStage

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["EMACallback"]


class EMACallback(Callback):
    # Exponential-moving-average weight tracking with automatic evaluation
    # swapping. Training always steps the live weights; every evaluation
    # entry point observes the averaged weights; checkpoints preserve both,
    # each under its correct key.
    def __init__(self, decay: float = 0.999) -> None:
        # Binds the averaging decay and prepares the empty shadow and
        # snapshot slots; the shadow is created at fit setup, and the
        # snapshot exists only while an evaluation swap is active.
        #
        # Args:
        #     decay: Exponential moving-average coefficient. Each update
        #         keeps this fraction of the shadow value and takes the
        #         remainder from the live parameter, so values closer to
        #         one average over a longer training horizon.
        #         Default: ``0.999``.
        super().__init__()
        self.decay: float = decay
        self._shadow_model: nn.Module | None = None
        self._original_state: dict[str, torch.Tensor] | None = None

    @override
    def setup(self, trainer: Trainer, module: Module, stage: TrainerStage) -> None:
        # Creates the gradient-free shadow copy once per fit. Other stages
        # leave the shadow absent, in which case every hook in this callback
        # degrades to a no-op.
        if stage == "fit":
            self._shadow_model: nn.Module | None = copy.deepcopy(module)
            for param in self._shadow_model.parameters():
                param.requires_grad_(False)

    @override
    def on_train_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int
    ) -> None:
        # Advances the average after each training batch under no_grad, with
        # strict pairing between shadow and live parameters.
        if self._shadow_model is None:
            return
        with torch.no_grad():
            for shadow_param, model_param in zip(
                self._shadow_model.parameters(), module.parameters(), strict=True
            ):
                shadow_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)

    def _swap_to_ema(self, module: Module) -> None:
        # Snapshots the complete live state with cloned tensors, then copies
        # the shadow parameters into the module so the subsequent evaluation
        # runs under the averaged weights.
        if self._shadow_model is None:
            return
        self._original_state: dict[str, torch.Tensor] | None = {k: v.clone() for k, v in module.state_dict().items()}
        for shadow_param, model_param in zip(
            self._shadow_model.parameters(), module.parameters(), strict=True
        ):
            model_param.data.copy_(shadow_param.data)

    def _restore_from_ema(self, module: Module) -> None:
        # Loads the pre-swap snapshot back into the module and clears it, so
        # training resumes from the exact live weights that existed before
        # the evaluation swap.
        if self._original_state is None:
            return
        module.load_state_dict(self._original_state)
        self._original_state: dict[str, torch.Tensor] | None = None

    @override
    def on_validation_start(self, trainer: Trainer, module: Module) -> None:
        # Swaps the averaged weights in before the validation pass begins.
        self._swap_to_ema(module)

    @override
    def on_validation_end(self, trainer: Trainer, module: Module) -> None:
        # Restores the live training weights after the validation pass.
        self._restore_from_ema(module)

    @override
    def on_test_start(self, trainer: Trainer, module: Module) -> None:
        # Swaps the averaged weights in before the test pass begins.
        self._swap_to_ema(module)

    @override
    def on_test_end(self, trainer: Trainer, module: Module) -> None:
        # Restores the live training weights after the test pass.
        self._restore_from_ema(module)

    @override
    def on_predict_start(self, trainer: Trainer, module: Module) -> None:
        # Swaps the averaged weights in before the prediction pass begins.
        self._swap_to_ema(module)

    @override
    def on_predict_end(self, trainer: Trainer, module: Module) -> None:
        # Restores the live training weights after the prediction pass.
        self._restore_from_ema(module)

    @override
    def on_save_checkpoint(
        self, trainer: Trainer, module: Module, checkpoint: CheckpointDict
    ) -> None:
        # Stores the shadow state under its dedicated key, and when a swap is
        # active replaces the payload's model state with the cloned pre-swap
        # snapshot, so a checkpoint written during evaluation persists the
        # true training weights rather than the temporarily swapped average.
        if self._shadow_model is not None:
            checkpoint["ema_state_dict"] = self._shadow_model.state_dict()
        if self._original_state is not None:
            checkpoint["model_state_dict"] = {
                k: v.clone() for k, v in self._original_state.items()
            }

    @override
    def on_load_checkpoint(
        self, trainer: Trainer, module: Module, checkpoint: CheckpointDict
    ) -> None:
        # Restores the shadow from its dedicated key when both the key and
        # the shadow exist; payloads from runs without this callback restore
        # nothing.
        ema_state: CheckpointValue | None = checkpoint.get("ema_state_dict")
        if ema_state is not None and self._shadow_model is not None:
            self._shadow_model.load_state_dict(ema_state)  # type: ignore[arg-type]

    @override
    def __repr__(self) -> str:
        # Compact configuration line for logs and interactive debugging.
        return f"EMACallback(decay={self.decay})"
