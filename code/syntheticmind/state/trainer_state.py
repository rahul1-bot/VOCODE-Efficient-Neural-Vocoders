# This module:
# 1. Tracks the trainer's execution status, entry-point stage, fine-grained running
#    stage, epoch and global-step counters, and the cooperative stop flag
# 2. Serializes the resumable subset of that state into checkpoints and restores
#    it on load
#
# Design decisions:
# - Status follows a four-value lifecycle (initializing, running, finished,
#   interrupted) and finish() refuses to overwrite an interrupted status, so
#   teardown after an interruption cannot relabel the run as finished
# - should_stop is cooperative: callbacks and strategies set it, loops poll it at
#   safe boundaries, and reset_on_run clears it so a stop request from one run
#   cannot leak into the next
# - load_state_dict restores only the progress counters; the persisted status and
#   stage are informational for checkpoint inspection, and a resuming run derives
#   its stage from its own entry point
# - Counter restoration coerces through int() with None guards because persisted
#   mapping values arrive typed as the checkpoint scalar union, not as int
#
# Author: Rahul Sawhney

from typing import Literal

from syntheticmind.utilities.types import RunningStage, TrainerStage, TrainerStateDict

__all__: list[str] = ["TrainerState"]

type TrainerStatus = Literal["initializing", "running", "finished", "interrupted"]


class TrainerState:
    # Mutable execution-state container owned by the trainer. Stage identifies the
    # entry point (fit, validate, test, predict); running stage identifies the
    # fine-grained activity inside it, including sanity checking; the counters
    # advance monotonically within a run and survive checkpoint round trips.
    def __init__(self) -> None:
        # Starts in the initializing status with no stage bound; counters begin at
        # zero and the stop flag begins lowered.
        self._status: TrainerStatus = "initializing"
        self._stage: TrainerStage | None = None
        self._running_stage: RunningStage | None = None
        self._current_epoch: int = 0
        self._global_step: int = 0
        self._should_stop: bool = False

    @property
    def status(self) -> TrainerStatus:
        # Current lifecycle status of the trainer.
        return self._status

    @property
    def stage(self) -> TrainerStage | None:
        # Entry-point stage bound by set_stage, or None before any run starts.
        return self._stage

    @property
    def running_stage(self) -> RunningStage | None:
        # Fine-grained activity marker, or None outside an active loop.
        return self._running_stage

    @property
    def current_epoch(self) -> int:
        # Zero-based epoch counter advanced by the fit loop.
        return self._current_epoch

    @property
    def global_step(self) -> int:
        # Count of optimizer steps executed across the whole run.
        return self._global_step

    @property
    def should_stop(self) -> bool:
        # Cooperative stop flag polled by the loops at safe boundaries.
        return self._should_stop

    @should_stop.setter
    def should_stop(self, value: bool) -> None:
        # Raised by callbacks or strategies to request a graceful stop.
        self._should_stop: bool = value

    def set_stage(self, stage: TrainerStage) -> None:
        # Binds the entry-point stage and promotes the status to running.
        self._stage: TrainerStage | None = stage
        self._status: TrainerStatus = "running"

    def set_running_stage(self, running_stage: RunningStage) -> None:
        # Marks the fine-grained activity currently executing.
        self._running_stage: RunningStage | None = running_stage

    def increment_epoch(self) -> None:
        # Advances the epoch counter at the epoch boundary.
        self._current_epoch += 1

    def increment_global_step(self) -> None:
        # Advances the optimizer-step counter after each executed step.
        self._global_step += 1

    def finish(self) -> None:
        # Promotes the status to finished unless the run was interrupted; an
        # interruption outranks completion so it stays visible after teardown.
        if self._status != "interrupted":
            self._status: TrainerStatus = "finished"

    def interrupt(self) -> None:
        # Records a user or system interruption while leaving counters intact
        # for resumption.
        self._status: TrainerStatus = "interrupted"

    def reset_on_run(self) -> None:
        # Clears run-local progress at the start of a new run so a previous
        # run's stop request or counters cannot bleed into this one.
        self._should_stop: bool = False
        self._current_epoch: int = 0
        self._global_step: int = 0

    def state_dict(self) -> TrainerStateDict:
        # Serializes status, stage, and the progress counters. Status and stage
        # travel for checkpoint inspection; only the counters are restored.
        state_dict: TrainerStateDict = {
            "status": self._status,
            "stage": self._stage,
            "current_epoch": self._current_epoch,
            "global_step": self._global_step
        }
        return state_dict

    def load_state_dict(self, state_dict: TrainerStateDict) -> None:
        # Restores the progress counters with None-guarded int coercion; missing
        # keys resume from zero rather than failing the load.
        current_epoch: int | str | float | bool | None = state_dict.get("current_epoch", 0)
        global_step: int | str | float | bool | None = state_dict.get("global_step", 0)
        self._current_epoch: int = int(current_epoch if current_epoch is not None else 0)
        self._global_step: int = int(global_step if global_step is not None else 0)

    def __repr__(self) -> str:
        # Compact status line for logs and interactive debugging.
        return (
            f"TrainerState("
            f"status={self._status!r}, "
            f"stage={self._stage!r}, "
            f"current_epoch={self._current_epoch}, "
            f"global_step={self._global_step})"
        )
