# This module:
# 1. Saves training checkpoints on the configured cadence: a rolling last.ckpt,
#    monitored top-k checkpoints named from a filename template, and an
#    optional exception checkpoint when a run fails
# 2. Builds full-resume payloads through CheckpointState, including mid-epoch
#    resume markers, and records save duration and file size as runtime
#    metrics
# 3. Persists its own selection state (best score, best path, retained top-k
#    set) so checkpoint retention continues correctly across resumption
#
# Design decisions:
# - All filesystem activity is restricted to rank zero, so distributed runs
#   produce exactly one checkpoint directory without write races
# - The saved epoch number is resume-oriented: a mid-epoch save records the
#   current epoch together with the completed-batch count and the epoch-start
#   random state, so resumption replays the same epoch from the interruption
#   point, while an epoch-boundary save records the next epoch to begin
# - Checkpoints are written to a same-directory temporary file and promoted by
#   atomic rename, so a crash during writing can never leave a truncated file
#   under a checkpoint name; the temporary file is removed in a finally block
# - The filename template resolves against epoch, step, and the metric values;
#   a collision with a retained checkpoint appends a version suffix instead of
#   overwriting the earlier file
# - Top-k retention sorts by score in the monitored direction and evicts beyond
#   k, but never deletes the file recorded as the best path
# - save_top_k accepts -1 to retain every eligible checkpoint and 0 to disable
#   monitored saves while save_last may continue independently
# - The exception path saves a full payload when the optimizer already exists
#   and falls back to a weights-only partial payload marked in its metadata,
#   so an early-construction failure still preserves the model weights
# - Restored top-k paths that resolve outside the current checkpoint directory
#   clear the retention state instead of adopting it, so a changed dirpath can
#   never cause deletions in a previous run's directory
#
# Author: Rahul Sawhney

import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, override

import torch
from loguru import logger as log

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.utilities.distributed import is_rank_zero
from syntheticmind.utilities.types import CheckpointDict, StateDict, StateValue, TrainerStage

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer
    from syntheticmind.loops.training_epoch_loop import TrainingEpochLoop

__all__: list[str] = ["ModelCheckpoint"]


class ModelCheckpoint(Callback):
    # Checkpointing callback driven by validation passes during fit. On each
    # eligible validation end it refreshes last.ckpt, evaluates the monitored
    # metric against the best score, saves and retains monitored checkpoints
    # under the top-k policy, and reports save telemetry through the
    # experiment logger.
    def __init__(
        self,
        dirpath: str | Path = "checkpoints",
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        save_top_k: int = 1,
        filename: str = "checkpoint-epoch{epoch:02d}-step{step:05d}",
        save_last: bool = True,
        every_n_epochs: int = 1,
        every_n_train_steps: int | None = None,
        save_on_exception: bool = False
    ) -> None:
        # Validates the cadence options and binds the checkpoint directory,
        # monitored metric and direction, retention count, filename template,
        # and exception-save policy. The best score starts at the identity
        # element for the chosen direction so the first monitored value always
        # qualifies.
        #
        # Args:
        #     dirpath: Directory receiving every checkpoint file written by
        #         this callback. Default: ``"checkpoints"``.
        #     monitor: Metric key compared for ranked retention, read from
        #         the trainer's callback metrics at each eligible
        #         validation boundary. Default: ``"val_loss"``.
        #     mode: Improvement direction for the monitored metric,
        #         ``"min"`` or ``"max"``. Default: ``"min"``.
        #     save_top_k: Number of best-ranked checkpoints retained;
        #         displaced entries are deleted from disk. Default: ``1``.
        #     filename: Filename template formatted with the current epoch
        #         and global step. Default:
        #         ``"checkpoint-epoch{epoch:02d}-step{step:05d}"``.
        #     save_last: Whether a last.ckpt copy is refreshed at every
        #         eligible boundary independently of the monitored ranking.
        #         Default: ``True``.
        #     every_n_epochs: Epoch cadence gating the validation-boundary
        #         saves; zero disables monitored epoch-cadence saving.
        #         Default: ``1``.
        #     every_n_train_steps: Optional optimizer-step cadence for
        #         mid-epoch saves; ``None`` disables step-cadence saving.
        #         Must be at least one when provided. Default: ``None``.
        #     save_on_exception: Whether a rescue checkpoint is written
        #         when training aborts on an exception. Default: ``False``.
        #
        # Raises:
        #     MisconfigurationError: If every_n_epochs is negative or
        #         every_n_train_steps is provided below one.
        super().__init__()
        if every_n_epochs < 0:
            from syntheticmind.utilities.exceptions import MisconfigurationError
            raise MisconfigurationError("every_n_epochs must be >= 0")
        if every_n_train_steps is not None and every_n_train_steps < 1:
            from syntheticmind.utilities.exceptions import MisconfigurationError
            raise MisconfigurationError("every_n_train_steps must be >= 1 when provided")
        self.dirpath: Path = Path(dirpath)
        self.monitor: str = monitor
        self.mode: Literal["min", "max"] = mode
        self.save_top_k: int = save_top_k
        self.filename: str = filename
        self.save_last: bool = save_last
        self.every_n_epochs: int = every_n_epochs
        self.every_n_train_steps: int | None = every_n_train_steps
        self.save_on_exception: bool = save_on_exception

        self.best_score: float = float("inf") if mode == "min" else float("-inf")
        self.best_path: Path | None = None
        self._top_k_scores: list[tuple[float, Path]] = []
        self._save_count: int = 0

    @override
    def setup(self, trainer: Trainer, module: Module, stage: TrainerStage) -> None:
        # Creates the checkpoint directory on rank zero before the stage
        # begins, so later saves never race on directory creation.
        if is_rank_zero():
            self.dirpath.mkdir(parents=True, exist_ok=True)

    @override
    def on_validation_epoch_end(
        self, trainer: Trainer, module: Module, metrics: dict[str, float]
    ) -> None:
        # Checkpoint decision point, executed on rank zero after each real
        # validation pass inside fit. The rolling last.ckpt is refreshed
        # first when enabled; the monitored path then requires the monitor
        # key to be present, saves on improvement (or unconditionally with
        # save_top_k=-1), and updates the retention set.
        if trainer.state.stage != "fit":
            return
        if trainer._sanity_checking:
            return
        if not is_rank_zero():
            return
        if not self._should_save_on_validation_end(trainer, module):
            return

        if self.save_last:
            last_path: Path = self.dirpath / "last.ckpt"
            self._save_checkpoint(trainer, module, last_path)

        if self.save_top_k == 0:
            return

        current_score: float | None = metrics.get(self.monitor)
        if current_score is None:
            from syntheticmind.utilities.exceptions import MisconfigurationError
            available: list[str] = list(metrics.keys())
            raise MisconfigurationError(
                f"ModelCheckpoint monitor '{self.monitor}' not found in validation metrics. "
                f"Available: {available}"
            )

        improved: bool = self._is_improvement(current_score)
        if improved or self.save_top_k == -1:
            current_epoch: int = module.current_epoch
            filepath: Path = self._format_filepath(current_epoch, trainer.state.global_step, metrics)

            self._save_checkpoint(trainer, module, filepath)

            if improved:
                self.best_score: float = current_score
                self.best_path: Path | None = filepath
            self._update_top_k(current_score, filepath)

    def _should_save_on_validation_end(self, trainer: Trainer, module: Module) -> bool:
        # Applies the configured cadence: a step interval, when set, gates on
        # the global step and takes precedence; otherwise the epoch interval
        # gates on the completed epoch count, and zero disables saving.
        if self.every_n_train_steps is not None:
            step: int = trainer.state.global_step
            return step > 0 and step % self.every_n_train_steps == 0
        if self.every_n_epochs <= 0:
            return False
        current_epoch: int = module.current_epoch
        return (current_epoch + 1) % self.every_n_epochs == 0

    def _is_improvement(self, current: float) -> bool:
        # Applies the strict improvement test in the monitored direction.
        if self.mode == "min":
            return current < self.best_score
        return current > self.best_score

    def _format_filepath(self, epoch: int, step: int, metrics: dict[str, float]) -> Path:
        # Resolves the filename template against epoch, step, and the metric
        # values, appends the checkpoint suffix when absent, and disambiguates
        # collisions with retained checkpoints through a version suffix so an
        # earlier retained file is never overwritten.
        name: str = self.filename.format(epoch=epoch, step=step, **metrics)
        if not name.endswith(".ckpt"):
            name: str = name + ".ckpt"
        candidate: Path = self.dirpath / name
        existing_paths: set[Path] = {p for _, p in self._top_k_scores}
        if candidate in existing_paths:
            stem: str = name.removesuffix(".ckpt")
            self._save_count += 1
            name: str = f"{stem}-v{self._save_count}.ckpt"
            candidate: Path = self.dirpath / name
        return candidate

    def _save_checkpoint(self, trainer: Trainer, module: Module, filepath: Path) -> None:
        # Assembles and writes one full-resume checkpoint. Mid-epoch state is
        # read from the training epoch loop: when the batch loop is active,
        # the completed-batch count and the epoch-start random state enter the
        # payload and the current epoch is recorded for replay; at an epoch
        # boundary the next epoch is recorded instead. The module and callback
        # save hooks run on the assembled payload, the write is atomic through
        # a same-directory temporary file, and save duration and size are
        # logged as runtime metrics.
        from syntheticmind.state.checkpoint_state import CheckpointState

        checkpoint_start_time: float = time.perf_counter()
        optimizer: torch.optim.Optimizer | None = trainer.optimizer
        assert optimizer is not None

        epoch_loop: TrainingEpochLoop | None = trainer.fit_loop.epoch_loop
        completed_batches: int = 0
        epoch_rng_state: CheckpointDict | None = None
        has_entered_batch_loop: bool = False
        if epoch_loop is not None:
            completed_batches: int = epoch_loop._completed_batches
            has_entered_batch_loop: bool = epoch_loop._has_entered_batch_loop
            if completed_batches > 0 or has_entered_batch_loop:
                epoch_rng_state: CheckpointDict | None = epoch_loop._epoch_start_rng_state

        is_mid_epoch: bool = completed_batches > 0 or has_entered_batch_loop
        epoch_to_save: int = (
            trainer.state.current_epoch if is_mid_epoch
            else trainer.state.current_epoch + 1
        )
        checkpoint: CheckpointDict = CheckpointState.build(
            model=module,
            optimizers=trainer.optimizers,
            scheduler=trainer.scheduler,
            schedulers=trainer.schedulers,
            epoch=epoch_to_save,
            global_step=trainer.state.global_step,
            callbacks=trainer.callbacks,
            datamodule=trainer.datamodule,
            completed_batches=completed_batches,
            epoch_rng_state=epoch_rng_state,
            strategy_name=type(trainer.strategy).__name__,
            world_size=getattr(trainer.strategy, "world_size", 1)
        )

        module.on_save_checkpoint(checkpoint)

        for cb in trainer.callbacks:
            cb.on_save_checkpoint(trainer, module, checkpoint)

        # mkstemp pre-creates the temporary file; the descriptor is closed
        # immediately because torch.save reopens the path itself.
        file_descriptor: int
        tmp_path_str: str
        file_descriptor, tmp_path_str = tempfile.mkstemp(dir=str(self.dirpath), suffix=".tmp")
        os.close(file_descriptor)
        tmp_path: Path = Path(tmp_path_str)
        try:
            torch.save(checkpoint, tmp_path)
            tmp_path.replace(filepath)
            checkpoint_duration_ms: float = (time.perf_counter() - checkpoint_start_time) * 1000.0
            checkpoint_size_mb: float = filepath.stat().st_size / 1048576.0
            if trainer.logger is not None:
                trainer.logger.log_metrics(
                    {
                        "runtime/checkpoint_save_time_ms": checkpoint_duration_ms,
                        "runtime/checkpoint_size_mb": checkpoint_size_mb
                    },
                    step=trainer.state.global_step
                )
            log.info(
                f"Checkpoint saved: {filepath} "
                f"duration_ms={checkpoint_duration_ms:.2f} size_mb={checkpoint_size_mb:.2f}"
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _save_partial_checkpoint(self, module: Module, filepath: Path) -> None:
        # Writes a weights-only payload marked partial in its metadata, used
        # by the exception path when the run failed before the optimizer
        # existed. The write follows the same atomic temporary-file protocol
        # as full checkpoints.
        checkpoint_start_time: float = time.perf_counter()
        checkpoint: CheckpointDict = {
            "model_state_dict": module.state_dict(),
            "metadata": {
                "torch_version": torch.__version__,
                "partial": True
            }
        }
        module.on_save_checkpoint(checkpoint)

        # mkstemp pre-creates the temporary file; the descriptor is closed
        # immediately because torch.save reopens the path itself.
        file_descriptor: int
        tmp_path_str: str
        file_descriptor, tmp_path_str = tempfile.mkstemp(dir=str(self.dirpath), suffix=".tmp")
        os.close(file_descriptor)
        tmp_path: Path = Path(tmp_path_str)
        try:
            torch.save(checkpoint, tmp_path)
            tmp_path.replace(filepath)
            checkpoint_duration_ms: float = (time.perf_counter() - checkpoint_start_time) * 1000.0
            checkpoint_size_mb: float = filepath.stat().st_size / 1048576.0
            log.info(
                f"Partial checkpoint saved: {filepath} "
                f"duration_ms={checkpoint_duration_ms:.2f} size_mb={checkpoint_size_mb:.2f}"
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _update_top_k(self, score: float, filepath: Path) -> None:
        # Inserts the new checkpoint into the retention set, sorts by score in
        # the monitored direction, and evicts entries beyond the retention
        # count, deleting their files unless a file is the recorded best path.
        # Unbounded retention (save_top_k=-1) appends without eviction.
        self._top_k_scores.append((score, filepath))
        if self.save_top_k > 0:
            if self.mode == "min":
                self._top_k_scores.sort(key=lambda x: x[0])
            else:
                self._top_k_scores.sort(key=lambda x: x[0], reverse=True)

            while len(self._top_k_scores) > self.save_top_k:
                old_path: Path
                _, old_path = self._top_k_scores.pop()
                if old_path.exists() and old_path != self.best_path:
                    old_path.unlink()

    @override
    def on_exception(self, trainer: Trainer, module: Module, exception: BaseException) -> None:
        # Optional last-resort save on run failure, restricted to rank zero
        # during fit. A full checkpoint is attempted when the optimizer
        # exists; otherwise the partial weights-only form is written. A
        # failure while saving is logged and suppressed so the original
        # exception continues to propagate.
        if not self.save_on_exception:
            return
        if not is_rank_zero():
            return
        if trainer.state.stage != "fit":
            return
        exception_path: Path = self.dirpath / "exception.ckpt"
        try:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            if trainer.optimizer is not None:
                self._save_checkpoint(trainer, module, exception_path)
            else:
                self._save_partial_checkpoint(module, exception_path)
            log.info(f"Exception checkpoint saved: {exception_path}")
        except Exception:
            log.warning("Failed to save exception checkpoint")

    @property
    def state_key(self) -> str:
        # Qualifies the checkpoint identity with monitor, direction, and both
        # cadence options so differently configured instances persist their
        # retention state independently.
        return self._generate_state_key(
            monitor=self.monitor,
            mode=self.mode,
            every_n_epochs=self.every_n_epochs,
            every_n_train_steps=self.every_n_train_steps
        )

    @override
    def state_dict(self) -> StateDict:
        # Persists the selection state: best score and path, the retained
        # top-k set with stringified paths, the collision counter, and the
        # step cadence for inspection.
        state_dict: StateDict = {
            "best_score": self.best_score,
            "best_path": str(self.best_path) if self.best_path else None,
            "top_k_scores": [(score, str(path)) for score, path in self._top_k_scores],
            "save_count": self._save_count,
            "every_n_train_steps": self.every_n_train_steps
        }
        return state_dict

    @override
    def load_state_dict(self, state_dict: StateDict) -> None:
        # Restores the selection state, then applies the cross-directory
        # protection: when any restored top-k path resolves outside the
        # current checkpoint directory, the retention state is cleared so
        # eviction can never delete files that belong to another directory.
        best_score: int | str | float | bool | None = state_dict.get("best_score", self.best_score)
        self.best_score: float = float(best_score if best_score is not None else self.best_score)
        path: StateValue | None = state_dict.get("best_path")
        self.best_path: Path | None = Path(str(path)) if path is not None else None
        top_k_raw: StateValue | None = state_dict.get("top_k_scores")
        if isinstance(top_k_raw, list):
            self._top_k_scores: list[tuple[float, Path]] = [(float(score), Path(str(path_value))) for score, path_value in top_k_raw]
        save_count: int | str | float | bool | None = state_dict.get("save_count", 0)
        self._save_count: int = int(save_count if save_count is not None else 0)

        if self._top_k_scores:
            resolved_dirpath: Path = self.dirpath.resolve()
            has_foreign_paths: bool = any(
                p.resolve().parent != resolved_dirpath for _, p in self._top_k_scores
            )
            if has_foreign_paths:
                log.warning(
                    f"ModelCheckpoint dirpath changed (now {self.dirpath}). "
                    f"Clearing top-k state to prevent cross-directory deletion."
                )
                self._top_k_scores.clear()
                self.best_path: Path | None = None

    @override
    def __repr__(self) -> str:
        # Compact configuration line for logs and interactive debugging.
        return (
            f"ModelCheckpoint("
            f"monitor={self.monitor!r}, "
            f"mode={self.mode!r}, "
            f"save_top_k={self.save_top_k}, "
            f"every_n_train_steps={self.every_n_train_steps}, "
            f"dirpath={str(self.dirpath)!r})"
        )
