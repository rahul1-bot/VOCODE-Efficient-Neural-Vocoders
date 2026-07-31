# This module:
# 1. Defines the abstract logger contract: batched float metrics with an explicit
#    step index, one-time hyperparameter records, and a finalize hook
# 2. Resolves the on-disk logging directory from save_dir, run name, and an
#    optional version segment
#
# Design decisions:
# - Implementations receive already-reduced float metrics with the step index
#   attached; reduction happens upstream in the trainer so every backend records
#   identical values for the same run
# - log_dir composes save_dir/name, appends the version only when one is set, and
#   creates the directory on first access so implementations can open files
#   without pre-flight checks
# - finalize is part of the contract because buffered backends must flush and
#   close on teardown, including teardown after a failure
# - Identity (save_dir, name, version) is fixed at construction so a logger's
#   destination cannot drift while a run is writing to it
#
# Author: Rahul Sawhney

from abc import ABC, abstractmethod
from pathlib import Path

from syntheticmind.utilities.types import HyperparameterDict

__all__: list[str] = ["Logger"]


class Logger(ABC):
    # Abstract base for experiment loggers. Subclasses implement the three
    # abstract members; directory resolution and identity handling live here so
    # every backend lays out its files the same way.
    def __init__(self, save_dir: Path, name: str = "default", version: str | None = None) -> None:
        # Binds the immutable logger identity: the root directory, the run name
        # used as a subdirectory, and an optional version subdirectory.
        #
        # Args:
        #     save_dir: Root directory under which the logger's output
        #         directory is composed.
        #     name: Run name used as the first subdirectory level.
        #         Default: ``"default"``.
        #     version: Optional version label used as the second
        #         subdirectory level; ``None`` omits the segment entirely.
        #         Default: ``None``.
        self._save_dir: Path = save_dir
        self._name: str = name
        self._version: str | None = version

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        # Records one batch of reduced metric values against the given step.
        raise NotImplementedError

    @abstractmethod
    def log_hyperparams(self, params: HyperparameterDict) -> None:
        # Records the run's hyperparameters once, near the start of execution.
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> None:
        # Flushes buffers and releases file handles at the end of the run.
        raise NotImplementedError

    @property
    def save_dir(self) -> Path:
        # Root directory under which this logger writes.
        return self._save_dir

    @property
    def name(self) -> str:
        # Run name used as the first subdirectory level.
        return self._name

    @property
    def version(self) -> str | None:
        # Optional version label used as the second subdirectory level.
        return self._version

    @property
    def log_dir(self) -> Path:
        # Fully resolved output directory (save_dir/name[/version]), created on
        # access so callers can immediately open files inside it.
        directory: Path = self._save_dir / self._name
        if self._version is not None:
            directory: Path = directory / self._version
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def __repr__(self) -> str:
        # Compact identity line for logs and interactive debugging.
        return (
            f"{type(self).__name__}("
            f"save_dir={str(self._save_dir)!r}, "
            f"name={self._name!r}, "
            f"version={self._version!r})"
        )
