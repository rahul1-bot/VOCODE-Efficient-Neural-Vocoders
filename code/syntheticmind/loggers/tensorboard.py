# This module:
# 1. Streams metric scalars into TensorBoard event files, one add_scalar call
#    per metric key with the trainer step as the global step
# 2. Records hyperparameters through the TensorBoard hparams plugin with values
#    coerced to strings
# 3. Defers the SummaryWriter import and construction until the first write
#
# Design decisions:
# - The concrete SummaryWriter is imported lazily inside _get_writer so
#   constructing this logger never forces the tensorboard dependency onto runs
#   that use a different backend
# - A minimal structural protocol stands in for the writer type, keeping the
#   module importable and type-checkable without tensorboard installed
# - Hyperparameter values are stringified before add_hparams because the plugin
#   accepts only primitive value types, while run configurations can carry
#   paths, tuples, and nested structures
# - finalize flushes before closing and drops the writer reference so a later
#   write would transparently create a fresh writer
#
# Author: Rahul Sawhney

from pathlib import Path
from typing import Protocol, override

from syntheticmind.loggers.logger import Logger
from syntheticmind.utilities.types import HyperparameterDict

__all__: list[str] = ["TensorBoardLogger"]


class _TensorBoardWriter(Protocol):
    # Structural stand-in for torch.utils.tensorboard.SummaryWriter covering
    # exactly the four members this logger touches, so the real class satisfies
    # it implicitly and no tensorboard import is needed at annotation time.
    def add_scalar(self, tag: str, scalar_value: float, global_step: int | None = None) -> None:
        # Records one scalar sample under the tag at the given global step.
        ...

    def add_hparams(
        self,
        hparam_dict: dict[str, str],
        metric_dict: dict[str, float]
    ) -> None:
        # Records the hyperparameter set, optionally paired with summary metrics.
        ...

    def flush(self) -> None:
        # Forces buffered events to disk.
        ...

    def close(self) -> None:
        # Flushes and releases the event-file resources.
        ...


class TensorBoardLogger(Logger):
    # Concrete logger backed by a lazily created SummaryWriter writing into
    # log_dir. Construction is cheap; the event file appears on first write.
    def __init__(self, save_dir: Path, name: str = "default", version: str | None = None) -> None:
        # Binds the logger identity; the writer slot stays empty until the
        # first metric or hyperparameter write.
        super().__init__(save_dir=save_dir, name=name, version=version)
        self._writer: _TensorBoardWriter | None = None

    @override
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        # Emits one scalar per metric key against the trainer step, creating
        # the writer on first use.
        writer: _TensorBoardWriter = self._get_writer()
        for key, value in metrics.items():
            writer.add_scalar(key, value, global_step=step)

    @override
    def log_hyperparams(self, params: HyperparameterDict) -> None:
        # Records the hyperparameter mapping with every value stringified for
        # the hparams plugin; no summary metrics are attached.
        writer: _TensorBoardWriter = self._get_writer()
        writer.add_hparams(
            {k: str(v) for k, v in params.items()},
            {}
        )

    @override
    def finalize(self) -> None:
        # Flushes and closes the writer if one exists, then clears the slot so
        # any later write would recreate it; safe to call repeatedly.
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer: _TensorBoardWriter | None = None

    def _get_writer(self) -> _TensorBoardWriter:
        # Returns the live writer, importing tensorboard and constructing the
        # SummaryWriter over log_dir on the first call only.
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self._writer: _TensorBoardWriter | None = SummaryWriter(log_dir=str(self.log_dir))
        return self._writer
