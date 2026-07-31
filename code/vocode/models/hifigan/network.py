# This module:
# 1. Implements the HiFi-GAN generator network: a pre-convolution over the
#    conditioning mel, a stack of transposed-convolution upsampling stages
#    each followed by a multi-receptive-field fusion of residual blocks,
#    and a tanh-bounded post-convolution producing the waveform
#
# Design decisions:
# - Each upsampling stage averages the outputs of its parallel residual
#   blocks (kernel and dilation variants), the multi-receptive-field
#   fusion of the reference architecture
# - The resblock kind selects between the reference ResBlock1 (V1, V2)
#   and the lighter ResBlock2 (V3) exactly as in the published recipe
# - The channel multiplier scales stage widths uniformly, which is how
#   the project half-width control halves capacity without touching the
#   topology
# - Convolution weights are normally initialized at the reference scale
#   and wrapped in weight normalization, matching the author training
#   dynamics
#
# Author: Rahul Sawhney

from typing import Literal, override

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from vocode.models.hifigan.resblock import ResBlock1, ResBlock2

__all__: list[str] = ["HifiganNetwork"]


class HifiganNetwork(nn.Module):
    # The HiFi-GAN generator: mel input to tanh-bounded waveform through
    # upsampling stages with multi-receptive-field residual fusion. This
    # is the only module of the family that carries inference cost, so it
    # is what the complexity profiler measures and what the published
    # author weights are loaded onto.
    #
    # Integration: the residual blocks are held in one flat module list
    # rather than nested per stage, indexed as
    # ``stage_index * kernel_count + kernel_index``. That flat layout is
    # what lets the published jik876 checkpoint map onto this network by
    # key rename alone, so it is load-bearing for the author-weight lane
    # and must not be reorganized into nested lists.
    def __init__(
        self,
        input_mel_channels: int,
        upsample_initial_channels: int,
        upsample_rates: tuple[int, ...],
        upsample_kernel_sizes: tuple[int, ...],
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilation_sizes: tuple[tuple[int, ...], ...],
        resblock_kind: Literal["1", "2"],
        leaky_relu_slope: float = 0.1,
        channel_multiplier: float = 1.0
    ) -> None:
        # Validates that the paired topology tuples agree in length, then
        # assembles the pre-convolution, the upsampling stages, the flat
        # residual-block list, and the post-convolution. Stage widths are
        # derived by halving the scaled initial width at every stage, and
        # every convolution is wrapped in weight normalization.
        #
        # Args:
        #     input_mel_channels: Mel band count the pre-convolution
        #         consumes.
        #     upsample_initial_channels: Width entering the first stage
        #         before the channel multiplier is applied.
        #     upsample_rates: Transposed-convolution stride of each stage.
        #     upsample_kernel_sizes: Kernel width of each stage, paired
        #         positionally with the rates. Stage padding is derived
        #         from the kernel and stride so the time axis expands by
        #         exactly the stride.
        #     resblock_kernel_sizes: Kernel widths of the parallel fusion
        #         branches attached to every stage.
        #     resblock_dilation_sizes: Dilation sets paired positionally
        #         with the fusion kernel widths.
        #     resblock_kind: ``"1"`` for the paired-convolution block,
        #         ``"2"`` for the single-convolution block.
        #     leaky_relu_slope: Negative slope of the per-stage
        #         activations, also forwarded into every residual block.
        #         Default: ``0.1``.
        #     channel_multiplier: Uniform scale on the initial width, from
        #         which every stage width follows. The scaled width is
        #         floored at one channel. Default: ``1.0``.
        #
        # Raises:
        #     ValueError: If the rate and kernel tuples differ in length,
        #         if the fusion kernel and dilation tuples differ in
        #         length, if a dilation set does not match the arity the
        #         selected block variant requires, or if the block variant
        #         is outside the closed two-value vocabulary.
        super().__init__()
        if len(upsample_rates) != len(upsample_kernel_sizes):
            raise ValueError(
                f"upsample_rates length ({len(upsample_rates)}) must equal "
                f"upsample_kernel_sizes length ({len(upsample_kernel_sizes)})"
            )
        if len(resblock_kernel_sizes) != len(resblock_dilation_sizes):
            raise ValueError(
                f"resblock_kernel_sizes length ({len(resblock_kernel_sizes)}) must equal "
                f"resblock_dilation_sizes length ({len(resblock_dilation_sizes)})"
            )
        self._leaky_relu_slope: float = leaky_relu_slope
        self._num_kernels: int = len(resblock_kernel_sizes)
        self._num_upsamples: int = len(upsample_rates)
        scaled_initial: int = max(1, int(upsample_initial_channels * channel_multiplier))
        self._scaled_initial_channels: int = scaled_initial
        self._pre_convolution: nn.Module = weight_norm(nn.Conv1d(
            in_channels=input_mel_channels,
            out_channels=scaled_initial,
            kernel_size=7,
            stride=1,
            padding=3
        ))
        self._upsample_layers: nn.ModuleList = nn.ModuleList()
        for layer_index, (upsample_stride, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_channels: int = scaled_initial // (2 ** layer_index)
            out_channels: int = scaled_initial // (2 ** (layer_index + 1))
            self._upsample_layers.append(weight_norm(nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=upsample_stride,
                padding=(kernel_size - upsample_stride) // 2
            )))
        self._residual_blocks: nn.ModuleList = nn.ModuleList()
        for layer_index in range(self._num_upsamples):
            block_channels: int = scaled_initial // (2 ** (layer_index + 1))
            for kernel_size, dilation_set in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self._residual_blocks.append(
                    self._build_resblock(resblock_kind, block_channels, kernel_size, dilation_set, leaky_relu_slope)
                )
        final_channels: int = scaled_initial // (2 ** self._num_upsamples)
        self._post_convolution: nn.Module = weight_norm(nn.Conv1d(
            in_channels=final_channels,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding=3
        ))
        self._initialize_weights()

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Runs the generator chain: pre-convolution, then per-stage leaky
        # ReLU, upsampling, and the averaged residual-block fusion, then the
        # final activation, post-convolution, and tanh bound.
        #
        # Args:
        #     mel: Conditioning mel batch shaped [batch, bands, frames],
        #         whose band count must equal the configured mel channel
        #         count.
        #
        # Returns:
        #     A waveform batch shaped [batch, 1, frames * rate_product],
        #     bounded to the closed interval from minus one to one by the
        #     output tanh.
        #
        # Note:
        #     The final activation before the post-convolution is applied
        #     at the framework default negative slope rather than the
        #     configured one, matching the reference generator.
        features: torch.Tensor = self._pre_convolution(mel)
        for upsample_index in range(self._num_upsamples):
            features: torch.Tensor = torch.nn.functional.leaky_relu(features, self._leaky_relu_slope)
            features: torch.Tensor = self._upsample_layers[upsample_index](features)
            residual_accumulator: torch.Tensor | None = None
            for kernel_index in range(self._num_kernels):
                block_output: torch.Tensor = self._residual_blocks[
                    upsample_index * self._num_kernels + kernel_index
                ](features)
                residual_accumulator: torch.Tensor | None = (
                    block_output if residual_accumulator is None else residual_accumulator + block_output
                )
            assert residual_accumulator is not None
            features: torch.Tensor = residual_accumulator / float(self._num_kernels)
        features: torch.Tensor = torch.nn.functional.leaky_relu(features)
        features: torch.Tensor = self._post_convolution(features)
        return torch.tanh(features)

    def _build_resblock(
        self,
        resblock_kind: Literal["1", "2"],
        channels: int,
        kernel_size: int,
        dilation_set: tuple[int, ...],
        leaky_relu_slope: float
    ) -> nn.Module:
        # Builds the residual block variant the configuration selects.
        # The dilation arity is checked here rather than inside the blocks
        # because the two variants consume different arities, and the
        # check is what turns a V3 dilation pair handed to the V1 variant
        # into a construction-time failure instead of a shape error deep
        # in the first forward.
        #
        # Args:
        #     resblock_kind: The variant selector from the configuration.
        #     channels: Width of the stage this block is attached to.
        #     kernel_size: Kernel width of this fusion branch.
        #     dilation_set: The branch's dilations; three entries for
        #         variant ``"1"``, two for variant ``"2"``.
        #     leaky_relu_slope: Negative slope used inside the block.
        #
        # Returns:
        #     The constructed residual block for this fusion branch.
        #
        # Raises:
        #     ValueError: If the dilation count does not match the
        #         selected variant, or the variant is outside the closed
        #         vocabulary.
        match resblock_kind:
            case "1":
                if len(dilation_set) != 3:
                    raise ValueError(f"ResBlock1 expects 3 dilations, got {len(dilation_set)}")
                return ResBlock1(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilations=(dilation_set[0], dilation_set[1], dilation_set[2]),
                    leaky_relu_slope=leaky_relu_slope
                )
            case "2":
                if len(dilation_set) != 2:
                    raise ValueError(f"ResBlock2 expects 2 dilations, got {len(dilation_set)}")
                return ResBlock2(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilations=(dilation_set[0], dilation_set[1]),
                    leaky_relu_slope=leaky_relu_slope
                )
            case _:
                raise ValueError(f"Unsupported resblock_kind: {resblock_kind}")

    def _initialize_weights(self) -> None:
        # Initializes model parameters according to the architecture reference behavior.
        # The reference generator draws convolution weights from a normal
        # distribution at this scale after wrapping them in weight
        # normalization, and this method reproduces that call site
        # faithfully. Because every convolution here is weight-normalized,
        # the assignment writes into the tensor the parametrization
        # recomputes on each access rather than into the stored magnitude
        # and direction parameters, which therefore keep their framework
        # initialization. The network test module records the same fact
        # and deliberately asserts neither value.
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                module.weight.data.normal_(mean=0.0, std=0.01)
