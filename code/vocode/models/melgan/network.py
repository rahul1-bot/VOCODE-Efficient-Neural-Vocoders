# This module:
# 1. Implements the MelGAN generator network: an input convolution over the
#    conditioning mel, transposed-convolution upsampling stages each
#    followed by residual dilated stacks, and a tanh-bounded output
#    convolution
#
# Design decisions:
# - Residual stacks use exponentially growing dilations so the receptive
#   field covers long waveform context at low cost, per the reference
# - All convolutions carry weight normalization, matching the reference
#   training dynamics and released checkpoint layout
#
# Author: Rahul Sawhney

from typing import override

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

__all__: list[str] = ["MelganNetwork", "ResStack"]


class ResStack(nn.Module):
    # The receptive-field expander attached after every upsampling stage.
    # Its three blocks carry dilations one, three, and nine, so the stack
    # reaches thirteen frames of context on each side at the cost of three
    # kernel-three convolutions. Each block's output is added to its own
    # learned shortcut and the sum replaces the running features, so the
    # three blocks compose in series rather than being averaged. Channel
    # width and time length are invariant end to end.
    """Residual stack matching seungwon/melgan's structure: three dilated blocks combined
    additively with three learned 1x1 shortcut convolutions. This mirrors the upstream
    pretrained checkpoint layout so weights can be restored without architectural drift.
    """

    def __init__(self, channels: int, kernel_size: int = 3, leaky_relu_slope: float = 0.2) -> None:
        # Builds the three dilated blocks and their three learned
        # shortcuts as parallel module lists. Each block is an activation,
        # a reflection pad matched to its dilation, the dilated
        # convolution, a second activation, and a pointwise convolution;
        # the reflection pad is what keeps the time axis invariant without
        # zero-padding the signal boundary.
        #
        # Args:
        #     channels: Width of the feature map, unchanged by the stack.
        #     kernel_size: Kernel width of the dilated convolutions. The
        #         pointwise convolutions and the shortcuts are fixed at
        #         width one regardless. Default: ``3``.
        #     leaky_relu_slope: Negative slope of the activations inside
        #         each block. Default: ``0.2``.
        super().__init__()
        self._leaky_relu_slope: float = leaky_relu_slope
        self._blocks: nn.ModuleList = nn.ModuleList()
        self._shortcuts: nn.ModuleList = nn.ModuleList()
        for dilation_exponent in range(3):
            dilation: int = 3 ** dilation_exponent
            self._blocks.append(nn.Sequential(
                nn.LeakyReLU(leaky_relu_slope),
                nn.ReflectionPad1d(dilation),
                weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=dilation)),
                nn.LeakyReLU(leaky_relu_slope),
                weight_norm(nn.Conv1d(channels, channels, kernel_size=1))
            ))
            self._shortcuts.append(weight_norm(nn.Conv1d(channels, channels, kernel_size=1)))

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Runs the three dilation stages in series, each stage summing its
        # block output with its learned shortcut and replacing the running
        # features. Strict pairing asserts the block and shortcut lists
        # never fall out of step.
        #
        # Args:
        #     inputs: Feature map shaped [batch, channels, frames].
        #
        # Returns:
        #     A feature map of exactly the input shape.
        features: torch.Tensor = inputs
        for block, shortcut in zip(self._blocks, self._shortcuts, strict=True):
            features: torch.Tensor = shortcut(features) + block(features)
        return features


class MelganNetwork(nn.Module):
    # The MelGAN generator: mel input to tanh-bounded waveform through
    # transposed-convolution stages, each followed by its residual stack.
    # The entire chain lives in one flat sequential container, so every
    # parameter is keyed by its position in that container rather than by
    # a role name. That layout is load-bearing for the author-weight lane
    # and must not be split into named submodules.
    """Generator matching seungwon/melgan's release v0.3-alpha topology.

    Sequential indices align positionally with the upstream `generator` Sequential so a
    name-only state-dict remap suffices to load the pretrained nvidia_tacotron2_LJ11
    checkpoint. Forward applies the upstream `(mel + 5.0) / 5.0` log-mel normalization
    before the convolutional stack, which is required for the released weights to
    produce intelligible audio.
    """

    _INPUT_MEL_SHIFT: float = 5.0
    _INPUT_MEL_SCALE: float = 5.0

    def __init__(
        self,
        input_mel_channels: int = 80,
        ngf: int = 32,
        upsample_factors: tuple[int, ...] = (8, 8, 2, 2),
        leaky_relu_slope: float = 0.2
    ) -> None:
        # Assembles the sequential container in upstream order: reflection
        # pad and input convolution, then one activation, transposed
        # convolution, and residual stack per upsample factor, then the
        # closing activation, pad, output convolution, and tanh. The
        # working width starts at the growth factor doubled once per
        # factor and halves at every stage, so it returns to the growth
        # factor at the output convolution. Stage padding and output
        # padding are derived from the factor so each stage expands the
        # time axis by exactly that factor, including for odd factors.
        #
        # Args:
        #     input_mel_channels: Mel band count the input convolution
        #         consumes. Default: ``80``.
        #     ngf: Base generator width. Default: ``32``.
        #     upsample_factors: Transposed-convolution stride of each
        #         stage; their product is the total upsampling factor.
        #         Default: ``(8, 8, 2, 2)``.
        #     leaky_relu_slope: Negative slope of every activation,
        #         forwarded into each residual stack as well.
        #         Default: ``0.2``.
        super().__init__()
        self._leaky_relu_slope: float = leaky_relu_slope
        layers: list[nn.Module] = [
            nn.ReflectionPad1d(3),
            weight_norm(nn.Conv1d(input_mel_channels, ngf * (2 ** len(upsample_factors)), kernel_size=7))
        ]
        current_channels: int = ngf * (2 ** len(upsample_factors))
        for upsample_factor in upsample_factors:
            layers.append(nn.LeakyReLU(leaky_relu_slope))
            layers.append(weight_norm(nn.ConvTranspose1d(
                current_channels,
                current_channels // 2,
                kernel_size=upsample_factor * 2,
                stride=upsample_factor,
                padding=upsample_factor // 2 + upsample_factor % 2,
                output_padding=upsample_factor % 2
            )))
            current_channels: int = current_channels // 2
            layers.append(ResStack(current_channels, leaky_relu_slope=leaky_relu_slope))
        layers.extend([
            nn.LeakyReLU(leaky_relu_slope),
            nn.ReflectionPad1d(3),
            weight_norm(nn.Conv1d(current_channels, 1, kernel_size=7)),
            nn.Tanh()
        ])
        self._network: nn.Sequential = nn.Sequential(*layers)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Applies the upstream log-mel normalization and runs the
        # sequential container. The normalization is fused into the
        # forward rather than left to callers, so a mel extracted under
        # the family protocol is fed in unmodified; the released weights
        # were trained on the normalized scale and produce noise without
        # it.
        #
        # Args:
        #     mel: Conditioning mel batch shaped [batch, bands, frames],
        #         on the raw log-mel scale of the family protocol.
        #
        # Returns:
        #     A waveform batch shaped
        #     [batch, 1, frames * factor_product], bounded to the closed
        #     interval from minus one to one by the output tanh.
        shifted_mel: torch.Tensor = (mel + self._INPUT_MEL_SHIFT) / self._INPUT_MEL_SCALE
        return self._network(shifted_mel)
