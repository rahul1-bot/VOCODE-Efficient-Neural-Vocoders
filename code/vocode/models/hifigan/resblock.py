# This module:
# 1. Implements the two HiFi-GAN residual block variants: ResBlock1 with
#    paired dilated-then-plain convolutions per dilation (V1, V2), and
#    the lighter single-convolution ResBlock2 (V3)
#
# Design decisions:
# - Padding is computed per kernel and dilation to keep the time axis
#   length-invariant, so residual additions never require cropping
# - All convolutions carry weight normalization, matching the reference
#   training dynamics and the published checkpoint layout
#
# Author: Rahul Sawhney

from typing import override

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

__all__: list[str] = ["ResBlock1", "ResBlock2"]


class ResBlock1(nn.Module):
    # One fusion branch of the V1 and V2 recipes. Each of the three
    # dilations owns a dilated convolution paired with a plain refinement
    # convolution at dilation one, and the three pairs are applied in
    # series, every pair adding its output back into the running residual.
    # Channel width and time length are invariant end to end, which is
    # what lets the generator average several branches of differing kernel
    # widths at each upsampling stage.
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, int, int] = (1, 3, 5),
        leaky_relu_slope: float = 0.1
    ) -> None:
        # Builds the two parallel module lists, one dilated convolution and
        # one refinement convolution per dilation, each weight-normalized
        # and padded to preserve the time axis. Holding the two roles in
        # separate lists is what the published-checkpoint key adaptation
        # maps the author's two convolution groups onto.
        #
        # Args:
        #     channels: Width of the feature map, unchanged by the block.
        #     kernel_size: Kernel width shared by both convolution roles.
        #         Default: ``3``.
        #     dilations: The three dilations of the branch; only the
        #         dilated convolutions use them, the refinement
        #         convolutions are always at dilation one.
        #         Default: ``(1, 3, 5)``.
        #     leaky_relu_slope: Negative slope of the activations
        #         preceding each convolution. Default: ``0.1``.
        super().__init__()
        self._leaky_relu_slope: float = leaky_relu_slope
        self._dilated_convolutions: nn.ModuleList = nn.ModuleList([
            weight_norm(nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=self._compute_padding(kernel_size, dilation)
            ))
            for dilation in dilations
        ])
        self._refinement_convolutions: nn.ModuleList = nn.ModuleList([
            weight_norm(nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=1,
                padding=self._compute_padding(kernel_size, 1)
            ))
            for _ in dilations
        ])

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Applies the three activation, dilated-convolution, activation,
        # refinement-convolution pairs in series, adding each pair's output
        # back into the residual before the next pair reads it. The
        # activations precede the convolutions rather than following them,
        # which is the pre-activation ordering of the reference block.
        #
        # Args:
        #     inputs: Feature map shaped [batch, channels, frames].
        #
        # Returns:
        #     A feature map of exactly the input shape.
        residual: torch.Tensor = inputs
        for dilated_conv, refine_conv in zip(self._dilated_convolutions, self._refinement_convolutions):
            activated: torch.Tensor = torch.nn.functional.leaky_relu(residual, self._leaky_relu_slope)
            transformed: torch.Tensor = dilated_conv(activated)
            transformed: torch.Tensor = torch.nn.functional.leaky_relu(transformed, self._leaky_relu_slope)
            transformed: torch.Tensor = refine_conv(transformed)
            residual: torch.Tensor = transformed + residual
        return residual

    def _compute_padding(self, kernel_size: int, dilation: int) -> int:
        # Computes convolution padding that preserves the intended temporal alignment.
        # The expression is the dilation times one less than the kernel
        # width, halved: the symmetric padding that leaves the time axis
        # length unchanged for the odd kernel widths every reference
        # recipe uses. An even kernel width would lose the final sample to
        # the integer division, which is why the recipes never declare
        # one.
        return (kernel_size * dilation - dilation) // 2


class ResBlock2(nn.Module):
    # One fusion branch of the V3 recipe: the lighter variant that drops
    # the refinement convolution, so each of its two dilations contributes
    # a single activation-then-convolution stage to the running residual.
    # Like the heavier variant it preserves channel width and time length,
    # so the two are interchangeable from the generator's perspective and
    # differ only in capacity.
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, int] = (1, 3),
        leaky_relu_slope: float = 0.1
    ) -> None:
        # Builds one weight-normalized dilated convolution per dilation in
        # a single module list, each padded to preserve the time axis.
        #
        # Args:
        #     channels: Width of the feature map, unchanged by the block.
        #     kernel_size: Kernel width shared by both convolutions.
        #         Default: ``3``.
        #     dilations: The two dilations of the branch. Default:
        #         ``(1, 3)``.
        #     leaky_relu_slope: Negative slope of the activation preceding
        #         each convolution. Default: ``0.1``.
        super().__init__()
        self._leaky_relu_slope: float = leaky_relu_slope
        self._convolutions: nn.ModuleList = nn.ModuleList([
            weight_norm(nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=self._compute_padding(kernel_size, dilation)
            ))
            for dilation in dilations
        ])

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Applies the two activation-then-convolution stages in series,
        # adding each stage's output back into the residual before the
        # next stage reads it.
        #
        # Args:
        #     inputs: Feature map shaped [batch, channels, frames].
        #
        # Returns:
        #     A feature map of exactly the input shape.
        residual: torch.Tensor = inputs
        for convolution in self._convolutions:
            activated: torch.Tensor = torch.nn.functional.leaky_relu(residual, self._leaky_relu_slope)
            transformed: torch.Tensor = convolution(activated)
            residual: torch.Tensor = transformed + residual
        return residual

    def _compute_padding(self, kernel_size: int, dilation: int) -> int:
        # Computes convolution padding that preserves the intended temporal alignment.
        # Identical to the heavier variant's rule: dilation times one less
        # than the kernel width, halved, which is length-preserving for
        # the odd kernel widths the V3 recipe declares.
        return (kernel_size * dilation - dilation) // 2
