# This module:
# 1. Maps a validated optimizer configuration onto the concrete torch.optim
#    implementation for the AdamW, Adam, and SGD families
#
# Design decisions:
# - Construction dispatches over the closed optimizer-name vocabulary with
#   match/case, and the default arm raises so an unknown name can never fall
#   through to a silently wrong optimizer
# - Each arm passes only the hyperparameters its optimizer understands: betas
#   for the Adam family, momentum and nesterov for SGD; the configuration
#   object carries the superset and this function performs the projection
# - The parameter iterator is taken once from the module and consumed by
#   exactly one constructor, so no arm ever receives an exhausted iterator
#
# Author: Rahul Sawhney

from collections.abc import Iterator

import torch
from torch import nn

from syntheticmind.core.optimizer import OptimizerConfig

__all__: list[str] = ["build_optimizer"]


def build_optimizer(config: OptimizerConfig, model: nn.Module) -> torch.optim.Optimizer:
    # Constructs the optimizer named by the configuration over the module's
    # parameters. The configuration is validated upstream, so this function
    # only projects fields onto the selected torch.optim constructor.
    parameters: Iterator[nn.Parameter] = model.parameters()

    match config.name:
        case "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=config.betas
            )
        case "adam":
            return torch.optim.Adam(
                parameters,
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=config.betas
            )
        case "sgd":
            return torch.optim.SGD(
                parameters,
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                nesterov=config.nesterov
            )
        case _:
            raise ValueError(f"Unknown optimizer: {config.name}")
