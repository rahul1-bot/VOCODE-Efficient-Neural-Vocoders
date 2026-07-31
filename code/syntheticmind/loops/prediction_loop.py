# This module:
# 1. Executes one complete prediction pass under torch.inference_mode: the
#    prediction lifecycle hooks, the batch-transfer hook chain, per-batch
#    predict-step execution, and optional collection of the step outputs
#
# Design decisions:
# - Prediction produces no metrics and touches no optimization state; the loop
#   therefore has no accumulator machinery and only clears the module's logging
#   buffer at the end as a defensive measure, since logging is blocked inside
#   prediction hooks by the metric-logging contract
# - The entire batch loop runs inside torch.inference_mode because prediction
#   must never build autograd graphs
# - return_predictions controls whether step outputs are retained. Disabling
#   retention keeps memory flat for prediction passes whose outputs are written
#   to disk by the module or a callback rather than returned
# - The batch-transfer hook chain applies the same datamodule-over-module
#   precedence rule as the other loops, decided per hook by comparing bound
#   implementations against the DataModule base attributes
#
# Author: Rahul Sawhney

import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.module import Module
from syntheticmind.loops.loop import Loop
from syntheticmind.state.trainer_state import TrainerState
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.utilities.types import Batch, ModelOutput

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["PredictionLoop"]


class PredictionLoop(Loop):
    # Loop that executes one inference pass over the prediction dataloader each
    # time run() is invoked, returning the collected predict-step outputs in
    # batch order when retention is enabled.
    def __init__(self, limit_batches: int | float | None = None, return_predictions: bool = True) -> None:
        # Binds the optional batch ceiling and the output-retention policy.
        super().__init__()
        self.limit_batches: int | float | None = limit_batches
        self.return_predictions: bool = return_predictions

    def run(self) -> list[ModelOutput]:
        # Executes one full prediction pass. The sequence is: switch the module
        # to evaluation mode through its prediction hook, fire the stage-start
        # and epoch-start hook pairs, iterate batches under inference mode with
        # the batch-start hooks, the batch-transfer chain, the predict step,
        # and the batch-end hooks, then fire the epoch-end and stage-end hook
        # pairs and return the collected outputs.
        trainer: Trainer = self.trainer
        module: Module | None = trainer.module
        assert module is not None
        strategy: Strategy = trainer.strategy
        state: TrainerState = trainer.state
        callbacks: list[Callback] = trainer.callbacks
        predict_dataloader: DataLoader | None = trainer.predict_dataloader  # type: ignore[type-arg]
        assert predict_dataloader is not None
        device: torch.device = strategy.root_device
        model: nn.Module | None = strategy.model
        assert model is not None

        state.set_running_stage("predicting")
        module._running_stage: str | None = "predicting"
        module.on_predict_model_eval()

        for callback in callbacks:
            callback.on_predict_start(trainer, module)
        module.on_predict_start()
        for callback in callbacks:
            callback.on_predict_epoch_start(trainer, module)
        module.on_predict_epoch_start()

        total_batches: int | None = self._safe_len(predict_dataloader)
        max_batches: int | None = self._resolve_limit_batches(total_batches)
        all_predictions: list[ModelOutput] = []

        with torch.inference_mode():
            dataloader_iterator_start_time: float = time.perf_counter()
            dataloader_iterator: Iterator[Batch] = iter(predict_dataloader)
            trainer._record_dataloader_iterator_creation_time(
                "predicting",
                time.perf_counter() - dataloader_iterator_start_time
            )
            batch_idx: int = -1
            while True:
                dataloader_fetch_start_time: float = time.perf_counter()
                try:
                    batch: Batch = next(dataloader_iterator)
                except StopIteration:
                    break
                trainer._record_dataloader_fetch_time(
                    "predicting",
                    time.perf_counter() - dataloader_fetch_start_time
                )
                batch_idx += 1
                if max_batches is not None and batch_idx >= max_batches:
                    break

                for callback in callbacks:
                    callback.on_predict_batch_start(trainer, module, batch, batch_idx, 0)
                module.on_predict_batch_start(batch, batch_idx, 0)

                dm: DataModule | None = trainer.datamodule
                if dm is not None and type(dm).on_before_batch_transfer is not DataModule.on_before_batch_transfer:
                    batch: Batch = dm.on_before_batch_transfer(batch, dataloader_idx=0)
                else:
                    batch: Batch = module.on_before_batch_transfer(batch, dataloader_idx=0)
                if dm is not None and type(dm).transfer_batch_to_device is not DataModule.transfer_batch_to_device:
                    batch: Batch = dm.transfer_batch_to_device(batch, device, dataloader_idx=0)
                else:
                    batch: Batch = module.transfer_batch_to_device(batch, device, dataloader_idx=0)
                if dm is not None and type(dm).on_after_batch_transfer is not DataModule.on_after_batch_transfer:
                    batch: Batch = dm.on_after_batch_transfer(batch, dataloader_idx=0)
                else:
                    batch: Batch = module.on_after_batch_transfer(batch, dataloader_idx=0)

                outputs: ModelOutput = strategy.predict_step(model, batch, batch_idx)
                if self.return_predictions:
                    all_predictions.append(outputs)

                for callback in callbacks:
                    callback.on_predict_batch_end(trainer, module, outputs, batch, batch_idx, 0)
                module.on_predict_batch_end(outputs, batch, batch_idx, 0)

        module.on_predict_epoch_end()
        for callback in callbacks:
            callback.on_predict_epoch_end(trainer, module)
        module.on_predict_end()
        for callback in callbacks:
            callback.on_predict_end(trainer, module)

        module._reset_logged_metrics()
        return all_predictions

    @staticmethod
    def _safe_len(dataloader: DataLoader) -> int | None:  # type: ignore[type-arg]
        # Returns the dataloader length, or None for iterable datasets that do
        # not define one, so limit resolution can distinguish the two cases.
        try:
            return len(dataloader)
        except TypeError:
            return None

    def _resolve_limit_batches(self, total: int | None) -> int | None:
        # Resolves the effective batch ceiling for this pass. With a known
        # total, a float limit selects that fraction of the pass (at least one
        # batch) and an integer limit is capped at the total. Without a known
        # total, an integer limit is used directly, a fractional float limit is
        # rejected as a configuration error, and no limit means unbounded.
        if total is None:
            if isinstance(self.limit_batches, int):
                return self.limit_batches
            if isinstance(self.limit_batches, float) and self.limit_batches != 1.0:
                from syntheticmind.utilities.exceptions import MisconfigurationError
                raise MisconfigurationError(
                    f"limit_batches={self.limit_batches} (float) is not supported "
                    f"with iterable datasets that have no length. Use an integer limit or set to 1.0."
                )
            return None
        if self.limit_batches is None:
            return total
        if isinstance(self.limit_batches, float):
            return max(1, int(total * self.limit_batches))
        return min(int(self.limit_batches), total)
