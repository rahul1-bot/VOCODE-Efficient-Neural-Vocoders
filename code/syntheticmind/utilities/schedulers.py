# This module:
# 1. Maps a validated scheduler configuration onto the concrete
#    torch.optim.lr_scheduler implementation across the polynomial, cosine,
#    warm-restart, step, multistep, exponential, plateau, and one-cycle families
#
# Design decisions:
# - Construction dispatches over the closed scheduler-name vocabulary with
#   match/case, and the default arm raises so an unknown name cannot fall
#   through to a silently wrong schedule
# - Horizon-dependent schedules receive total_steps from the caller because only
#   the trainer knows the true optimization horizon after batch limits and
#   gradient accumulation are applied
# - cosine and cosine_warm_restarts fall back to total_steps when t_max is
#   unset, so an unconfigured period still spans the actual run instead of a
#   torch default
# - multistep normalizes an absent milestone list to an empty list rather than
#   forwarding None into the torch constructor
# - one_cycle resolves its peak rate from the optimizer's own default learning
#   rate when max_lr is unset, keeping the two objects consistent by default
# - reduce_on_plateau is fixed to min mode, matching the loss-monitoring
#   convention used across the harness
#
# Author: Rahul Sawhney

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LRScheduler,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    StepLR,
)

from syntheticmind.core.optimizer import SchedulerConfig

__all__: list[str] = ["build_scheduler"]


def build_scheduler(
    config: SchedulerConfig,
    optimizer: torch.optim.Optimizer,
    total_steps: int
) -> LRScheduler:
    # Constructs the scheduler named by the configuration, projecting only the
    # fields the selected schedule understands and resolving horizon defaults
    # from total_steps where the configuration leaves them unset.
    match config.name:
        case "poly":
            return PolynomialLR(
                optimizer,
                total_iters=total_steps,
                power=config.power
            )
        case "cosine":
            t_max: int = config.t_max if config.t_max is not None else total_steps
            return CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=config.eta_min
            )
        case "cosine_warm_restarts":
            t_0: int = config.t_max if config.t_max is not None else total_steps
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=t_0,
                eta_min=config.eta_min
            )
        case "step":
            return StepLR(
                optimizer,
                step_size=config.step_size,
                gamma=config.gamma
            )
        case "multistep":
            milestones: list[int] = config.milestones if config.milestones is not None else []
            return MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=config.gamma
            )
        case "exponential":
            return ExponentialLR(
                optimizer,
                gamma=config.gamma
            )
        case "reduce_on_plateau":
            return ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=config.factor,
                patience=config.patience
            )
        case "one_cycle":
            max_lr: float = config.max_lr if config.max_lr is not None else optimizer.defaults.get("lr", 0.01)
            return OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps
            )
        case _:
            raise ValueError(f"Unknown scheduler: {config.name}")
