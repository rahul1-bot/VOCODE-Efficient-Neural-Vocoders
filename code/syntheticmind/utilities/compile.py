# This module:
# 1. Applies torch.compile graph capture to a module behind an explicit opt-in flag
# 2. Applies the channels_last memory format to a module behind the same opt-in pattern
#
# Design decisions:
# - Both helpers take enabled=False defaults and return the input unchanged when
#   disabled, so call sites can wire the flags through configuration without
#   conditional plumbing of their own
# - torch.compile is applied with default settings here; mode selection and backend
#   tuning stay with the caller that owns the experiment configuration
# - channels_last is opt-in because it only benefits convolution-heavy networks on
#   hardware with fused NHWC kernels and can regress other workloads
#
# Author: Rahul Sawhney

import torch
from torch import nn

__all__: list[str] = ["apply_compile", "apply_channels_last"]


def apply_compile(model: nn.Module, enabled: bool = False) -> nn.Module:
    # Returns the module wrapped by torch.compile when enabled, otherwise the
    # original module untouched. Compilation happens lazily on first forward.
    if not enabled:
        return model
    return torch.compile(model)  # type: ignore[return-value]


def apply_channels_last(model: nn.Module, enabled: bool = False) -> nn.Module:
    # Converts parameters and buffers to the channels_last memory layout when
    # enabled, otherwise returns the module untouched.
    if not enabled:
        return model
    return model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
