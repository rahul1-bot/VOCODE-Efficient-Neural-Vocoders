# This module:
# 1. Defines the validated optimizer and scheduler configuration models consumed
#    by the optimizer and scheduler builders
# 2. Defines the definition unions that additionally admit pre-constructed
#    torch optimizer and scheduler objects
# 3. Defines OptimizationConfiguration, the single return contract of
#    Module.configure_optimizers, with cross-field validation of the
#    optimizer-to-scheduler structure
#
# Design decisions:
# - Both configuration models are frozen value objects, so a configuration
#   cannot be mutated after validation and every consumer observes the same
#   validated values
# - Each configuration carries the superset of fields across its families
#   (betas for the Adam family, momentum and nesterov for SGD, and the
#   per-schedule fields for the eight schedule families); the builders project
#   only the fields relevant to the selected name, which keeps one model
#   sufficient for every family
# - The plateau schedule is validated at construction to require an epoch
#   interval and a monitor metric, because it can only advance on a monitored
#   validation value produced at epoch boundaries
# - The definition unions accept already-constructed torch objects so advanced
#   callers can supply instances the builders cannot express, at the cost of
#   bypassing configuration validation for those instances
# - OptimizationConfiguration validates structural pairing at construction: a
#   scheduler list requires an optimizer list of matching length, and a single
#   scheduler configuration cannot be paired with multiple optimizers, so
#   mismatches surface before training starts rather than mid-run
#
# Author: Rahul Sawhney

from typing import ClassVar, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from syntheticmind.utilities.types import SchedulerInterval

__all__: list[str] = [
    "OptimizerConfig",
    "SchedulerConfig",
    "OptimizerDefinition",
    "SchedulerDefinition",
    "OptimizationConfiguration"
]


class OptimizerConfig(BaseModel):
    # Validated optimizer selection. The name selects the torch.optim family;
    # lr is mandatory and must be positive, while the remaining fields carry
    # family-specific hyperparameters that the builder projects per family.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    name: Literal["adamw", "sgd", "adam"] = "adamw"
    lr: float = Field(gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = Field(default=0.0, ge=0.0)
    nesterov: bool = False


class SchedulerConfig(BaseModel):
    # Validated scheduler selection. The name selects the schedule family, the
    # interval declares whether the schedule advances per optimizer step or
    # per epoch, and monitor names the metric consumed by the plateau family.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    name: Literal[
        "poly",
        "cosine",
        "cosine_warm_restarts",
        "step",
        "multistep",
        "exponential",
        "reduce_on_plateau",
        "one_cycle",
    ] = "poly"
    interval: SchedulerInterval = "step"
    monitor: str | None = None

    warmup_epochs: int = Field(default=0, ge=0)
    power: float = Field(default=0.9, gt=0.0)
    t_max: int | None = None
    eta_min: float = Field(default=0.0, ge=0.0)
    step_size: int = Field(default=10, gt=0)
    milestones: list[int] | None = None
    gamma: float = Field(default=0.1, gt=0.0)
    patience: int = Field(default=10, gt=0)
    factor: float = Field(default=0.1, gt=0.0)
    max_lr: float | None = None

    @model_validator(mode="after")
    def _validate_reduce_on_plateau(self) -> SchedulerConfig:
        # Enforces the plateau family's structural requirements at
        # construction: the schedule advances on a monitored validation metric,
        # which exists only at epoch boundaries, so both the epoch interval and
        # a monitor name are mandatory.
        if self.name == "reduce_on_plateau":
            if self.interval != "epoch":
                raise ValueError(
                    f"ReduceLROnPlateau requires interval='epoch', got '{self.interval}'"
                )
            if self.monitor is None:
                raise ValueError(
                    "ReduceLROnPlateau requires a monitor metric. Set monitor in SchedulerConfig."
                )
        return self


type OptimizerDefinition = OptimizerConfig | torch.optim.Optimizer

type SchedulerDefinition = SchedulerConfig | torch.optim.lr_scheduler.LRScheduler


class OptimizationConfiguration(BaseModel):
    # Return contract of Module.configure_optimizers. Carries one optimizer or
    # an ordered optimizer list, with an optional scheduler or scheduler list
    # paired by position. Arbitrary types are allowed because the definition
    # unions admit live torch objects alongside validated configurations.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    optimizer: OptimizerDefinition | list[OptimizerDefinition]
    scheduler: SchedulerDefinition | list[SchedulerDefinition] | None = None

    @model_validator(mode="after")
    def validate_optimizer_scheduler_structure(self) -> OptimizationConfiguration:
        # Validates the pairing structure between optimizers and schedulers: an
        # empty optimizer list is rejected, a scheduler list demands an
        # optimizer list of exactly matching length, and a single scheduler
        # configuration cannot serve multiple optimizers because the pairing
        # would be ambiguous.
        if isinstance(self.optimizer, list):
            if not self.optimizer:
                raise ValueError("Expected at least one optimizer configuration")
            optimizer_count: int = len(self.optimizer)
        else:
            optimizer_count: int = 1

        if isinstance(self.scheduler, list):
            if optimizer_count == 1:
                raise ValueError(
                    "A scheduler list requires a matching optimizer list"
                )
            if len(self.scheduler) != optimizer_count:
                raise ValueError(
                    "The number of scheduler configurations must match the number of optimizer configurations"
                )
        elif isinstance(self.scheduler, SchedulerConfig) and optimizer_count > 1:
            raise ValueError(
                "When multiple optimizers are configured, scheduler must be None or a list of scheduler configurations"
            )

        return self
