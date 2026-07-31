# This module:
# 1. Detects train-validation and train-test split contamination at runtime by
#    collecting per-sample identifiers from batches and testing the evaluation
#    sets against the training set
#
# Design decisions:
# - Detection is identifier-based: batches must carry a sample-identifier
#   entry under the configured key, and batches without it pass through
#   unchecked, so the guard imposes no batch-format requirement on pipelines
#   that do not use it
# - Overlap is tested on every evaluation batch rather than at epoch end, so
#   contamination aborts the run at the first leaked sample instead of after
#   an entire contaminated evaluation
# - A violation raises with the overlap size and example identifiers, because
#   evaluation results produced from a contaminated split are scientifically
#   void and must not be produced silently
# - Identifiers are normalized to strings from tensor, list, or tuple form so
#   numeric and string identifier schemes compare consistently
# - The collected sets reset at stage setup, keeping observations scoped to
#   one run
#
# Author: Rahul Sawhney

from typing import TYPE_CHECKING, override

import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.utilities.types import Batch, StepOutput, TrainerStage

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["DataLeakageGuard"]


class DataLeakageGuard(Callback):
    # Runtime split-contamination detector. Sample identifiers observed in
    # training batches accumulate in one set; each validation and test batch
    # contributes to its own set and is immediately tested for intersection
    # with the training set, aborting the run on the first overlap.
    def __init__(self, id_key: str = "sample_id") -> None:
        # Binds the batch key under which sample identifiers travel and
        # prepares the three per-split identifier sets.
        super().__init__()
        self.id_key: str = id_key
        self._train_ids: set[str] = set()
        self._val_ids: set[str] = set()
        self._test_ids: set[str] = set()

    @override
    def setup(self, trainer: Trainer, module: Module, stage: TrainerStage) -> None:
        # Clears the identifier sets at stage setup so observations from a
        # previous run cannot produce false overlaps in this one.
        self._train_ids.clear()
        self._val_ids.clear()
        self._test_ids.clear()

    @override
    def on_train_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int
    ) -> None:
        # Accumulates the batch's sample identifiers into the training set.
        self._collect_ids(batch, self._train_ids)

    @override
    def on_validation_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Accumulates the batch's identifiers into the validation set and
        # aborts on any intersection with the training set, reporting the
        # overlap size and example identifiers.
        self._collect_ids(batch, self._val_ids)
        overlap: set[str] = self._train_ids & self._val_ids
        if overlap:
            raise RuntimeError(
                f"Data leakage detected: {len(overlap)} sample IDs found in both "
                f"train and val sets. Example: {list(overlap)[:3]}"
            )

    @override
    def on_test_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Accumulates the batch's identifiers into the test set and aborts on
        # any intersection with the training set.
        self._collect_ids(batch, self._test_ids)
        overlap: set[str] = self._train_ids & self._test_ids
        if overlap:
            raise RuntimeError(
                f"Data leakage detected: {len(overlap)} sample IDs found in both "
                f"train and test sets. Example: {list(overlap)[:3]}"
            )

    def _collect_ids(self, batch: Batch, target: set[str]) -> None:
        # Extracts identifiers from mapping-style batches carrying the
        # configured key, normalizing tensor, list, and tuple identifier
        # containers to strings; other batch shapes contribute nothing.
        if isinstance(batch, dict) and self.id_key in batch:
            ids: torch.Tensor | list[str | int] | tuple[str | int, ...] = batch[self.id_key]
            if isinstance(ids, torch.Tensor):
                target.update(str(i) for i in ids.tolist())
            elif isinstance(ids, (list, tuple)):
                target.update(str(i) for i in ids)
