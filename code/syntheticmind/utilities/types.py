# This module:
# 1. Declares the closed Literal vocabularies the harness accepts for trainer stages,
#    running stages, precision policies, gradient-clip algorithms, scheduler stepping
#    intervals, and metric reductions
# 2. Declares the recursive structural aliases for batches, step outputs, model outputs,
#    hyperparameters, mutable component state, and checkpoint payloads
# 3. Defines the DeviceTransferable protocol used to duck-type device movement for
#    tensor-like containers that are not raw torch.Tensor objects
#
# Design decisions:
# - PEP 695 `type` statements are used for every alias because they evaluate lazily,
#   which lets Batch, ModelOutput, and the other recursive unions reference themselves
#   without string forward references
# - The stage and policy vocabularies live here, in one importable location, so invalid
#   values fail at configuration boundaries instead of deep inside a running loop
# - DeviceTransferable is runtime_checkable so batch transfer can test hasattr-style
#   conformance at runtime without importing any concrete container class
# - Each domain (batch, model, hyperparameter, state, checkpoint) keeps its own scalar
#   and value family even where the unions currently coincide, so one contract can
#   tighten later without silently rewriting the others
# - numpy arrays are admitted only in checkpoint payloads, where serialized RNG state
#   legitimately contains them; runtime batch and state contracts stay tensor-only
#
# Author: Rahul Sawhney

from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch

__all__: list[str] = [
    "TrainerStage",
    "RunningStage",
    "Precision",
    "ClipAlgorithm",
    "SchedulerInterval",
    "Reduction",
    "BatchScalar",
    "DeviceTransferable",
    "Batch",
    "StepOutput",
    "ModelScalar",
    "ModelOutput",
    "HyperparameterScalar",
    "HyperparameterValue",
    "HyperparameterDict",
    "StateScalar",
    "StateValue",
    "StateDict",
    "TrainerStateDict",
    "CheckpointScalar",
    "CheckpointValue",
    "CheckpointDict"
]

type TrainerStage = Literal["fit", "validate", "test", "predict"]

type RunningStage = Literal["training", "validating", "testing", "predicting", "sanity_checking"]

type Precision = Literal["32-true", "16-mixed", "bf16-mixed"]

type ClipAlgorithm = Literal["norm", "value"]

type SchedulerInterval = Literal["step", "epoch"]

type Reduction = Literal["mean", "sum", "min", "max"]

type BatchScalar = bool | int | float | str | None


@runtime_checkable
class DeviceTransferable(Protocol):
    # Structural protocol for objects that can move themselves between devices.
    # Batch transfer checks against this protocol as a fallback after the concrete
    # tensor, dict, list, and tuple branches, which lets custom tensor-like containers
    # participate in device placement without the harness importing their classes.
    def to(self, device: torch.device, non_blocking: bool = False) -> DeviceTransferable:
        # Returns this object placed on the requested device. non_blocking mirrors the
        # torch.Tensor.to keyword so asynchronous host-to-device copies stay possible.
        ...


type Batch = (
    BatchScalar
    | torch.Tensor
    | DeviceTransferable
    | dict[str, Batch]
    | list[Batch]
    | tuple[Batch, ...]
)

type StepOutput = dict[str, torch.Tensor | float]

type ModelScalar = bool | int | float | str | None

type ModelOutput = (
    ModelScalar
    | torch.Tensor
    | list[ModelOutput]
    | tuple[ModelOutput, ...]
    | dict[str, ModelOutput]
)

type HyperparameterScalar = bool | int | float | str | None

type HyperparameterValue = (
    HyperparameterScalar
    | list[HyperparameterValue]
    | tuple[HyperparameterValue, ...]
    | dict[str, HyperparameterValue]
)

type HyperparameterDict = dict[str, HyperparameterValue]

type StateScalar = bool | int | float | str | None

type StateValue = StateScalar | torch.Tensor | list[StateValue] | tuple[StateValue, ...] | dict[str, StateValue]

type StateDict = dict[str, StateValue]

type TrainerStateDict = dict[str, StateScalar]

type CheckpointScalar = StateScalar

type CheckpointValue = (
    CheckpointScalar
    | torch.Tensor
    | np.ndarray
    | list[CheckpointValue]
    | tuple[CheckpointValue, ...]
    | dict[str, CheckpointValue]
)

type CheckpointDict = dict[str, CheckpointValue]
