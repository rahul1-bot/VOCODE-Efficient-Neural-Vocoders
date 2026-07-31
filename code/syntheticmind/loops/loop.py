# This module:
# 1. Defines the base class shared by the fit, training-epoch, evaluation, and
#    prediction loops: the trainer attachment point and the per-loop stop flag
#
# Design decisions:
# - Loops are constructed unattached and receive their trainer through the
#   property setter during trainer initialization. The property getter raises
#   immediately when a loop runs unattached, converting a wiring mistake into a
#   direct error instead of an AttributeError deep inside an epoch
# - The trainer reference is intentionally annotated lazily so this module does
#   not import the trainer and create an import cycle; the trainer imports the
#   loops, not the reverse
# - The per-loop stop flag is separate from the trainer-level stop state so an
#   individual loop can be reset without touching global run control
#
# Author: Rahul Sawhney

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["Loop"]


class Loop:
    # Base class for every execution loop. Concrete loops implement run() with
    # their stage-specific behavior; this class contributes only the trainer
    # attachment contract and the resettable stop flag, keeping the base free
    # of stage assumptions.
    def __init__(self) -> None:
        # Starts unattached with the stop flag lowered. The owning trainer
        # attaches itself through the trainer property during its own
        # construction.
        self._trainer: Trainer | None = None
        self._should_stop: bool = False

    @property
    def trainer(self) -> Trainer:
        # Returns the attached trainer. Raises when the loop was never
        # attached, because every concrete loop requires trainer collaborators
        # (strategy, state, callbacks) to execute.
        if self._trainer is None:
            raise RuntimeError("Loop is not connected to a Trainer")
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: Trainer) -> None:
        # Attaches the owning trainer; called once during trainer construction.
        self._trainer: Trainer | None = trainer

    @property
    def should_stop(self) -> bool:
        # Loop-local stop flag, independent of the trainer-level stop state.
        return self._should_stop

    @should_stop.setter
    def should_stop(self, value: bool) -> None:
        # Raises or lowers the loop-local stop flag.
        self._should_stop: bool = value

    def reset(self) -> None:
        # Lowers the stop flag so the loop instance can be reused for another
        # run without carrying over a stale stop request.
        self._should_stop: bool = False

    def __repr__(self) -> str:
        # Compact identity line for logs and interactive debugging.
        return f"{type(self).__name__}()"
