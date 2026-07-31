# This module:
# 1. Implements the MelGAN multi-scale discriminator ensemble: identical
#    window-based discriminators judging the waveform at progressively
#    average-pooled scales
#
# Design decisions:
# - Each discriminator returns its intermediate feature maps alongside the
#   final logits, because the generator objective includes feature
#   matching
# - Grouped convolutions keep the per-scale discriminators lightweight,
#   following the reference
#
# Author: Rahul Sawhney

from typing import override

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

__all__: list[str] = ["MelganDiscriminator", "MelganMultiScaleDiscriminator"]


class MelganDiscriminator(nn.Module):
    # One member of the multi-scale ensemble: a window-based judge reading
    # the waveform directly in the time domain through wide, heavily
    # grouped, strided convolutions. Unlike the HiFi-GAN scale member it
    # returns its logits as a map rather than a flattened vector, and it
    # separates the intermediate maps from the final one at the boundary
    # rather than including the final map in both.
    def __init__(self, leaky_relu_slope: float = 0.2) -> None:
        # Builds the seven-layer stack. Each of the first six layers pairs
        # a weight-normalized convolution with its activation inside one
        # sequential block, while the seventh is the bare single-channel
        # output convolution with no activation, which is what makes the
        # last recorded entry the logit map.
        #
        # Args:
        #     leaky_relu_slope: Negative slope of the activation in each
        #         of the first six layers. Default: ``0.2``.
        super().__init__()
        self._layers: nn.ModuleList = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                weight_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1)),
                nn.LeakyReLU(leaky_relu_slope, inplace=True)
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4)),
                nn.LeakyReLU(leaky_relu_slope, inplace=True)
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16)),
                nn.LeakyReLU(leaky_relu_slope, inplace=True)
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64)),
                nn.LeakyReLU(leaky_relu_slope, inplace=True)
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256)),
                nn.LeakyReLU(leaky_relu_slope, inplace=True)
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
                nn.LeakyReLU(leaky_relu_slope, inplace=True)
            ),
            weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))
        ])

    @override
    def forward(self, waveform: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        # Runs the stack, recording every layer output, then splits the
        # record at its last entry.
        #
        # Args:
        #     waveform: Waveform batch in the [batch, 1, samples] layout,
        #         already pooled to this member's scale by the ensemble.
        #
        # Returns:
        #     The six intermediate feature maps paired with the final
        #     single-channel logit map, in that order. The logit map is
        #     deliberately excluded from the feature list, so the
        #     feature-matching term never double-counts the decision
        #     layer.
        features: list[torch.Tensor] = []
        current: torch.Tensor = waveform
        for layer in self._layers:
            current: torch.Tensor = layer(current)
            features.append(current)
        return features[:-1], features[-1]


class MelganMultiScaleDiscriminator(nn.Module):
    # The scale ensemble: three identically configured window
    # discriminators judging the waveform at the original rate and at two
    # successively halved rates. Unlike the HiFi-GAN ensemble every member
    # here uses weight normalization, and the ensemble judges a single
    # waveform per call rather than a real and fake pair.
    def __init__(self, leaky_relu_slope: float = 0.2) -> None:
        # Builds the three discriminators alongside three poolers, the
        # first of which is an identity. Carrying an identity rather than
        # special-casing the first scale keeps the forward a single
        # uniform loop over paired poolers and discriminators.
        #
        # Args:
        #     leaky_relu_slope: Negative slope forwarded into every
        #         member. Default: ``0.2``.
        super().__init__()
        self._discriminators: nn.ModuleList = nn.ModuleList([
            MelganDiscriminator(leaky_relu_slope=leaky_relu_slope),
            MelganDiscriminator(leaky_relu_slope=leaky_relu_slope),
            MelganDiscriminator(leaky_relu_slope=leaky_relu_slope)
        ])
        self._poolers: nn.ModuleList = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False)
        ])

    @override
    def forward(self, waveform: torch.Tensor) -> list[tuple[list[torch.Tensor], torch.Tensor]]:
        # Walks the scales in order, pooling cumulatively: each pooler
        # reads the already-pooled tensor rather than the original, so the
        # three views sit at the full rate, one half, and one quarter.
        # Strict pairing asserts the pooler and discriminator lists never
        # fall out of step.
        #
        # Args:
        #     waveform: Waveform batch in the [batch, 1, samples] layout.
        #         The caller invokes this once for the reference and once
        #         for the synthesis, rather than passing both together.
        #
        # Returns:
        #     One feature-maps-and-logits pair per scale, in scale order
        #     from the unpooled view downward. Logit width shrinks from
        #     scale to scale, which is the observable signature of the
        #     pooling.
        outputs: list[tuple[list[torch.Tensor], torch.Tensor]] = []
        current: torch.Tensor = waveform
        for pooler, discriminator in zip(self._poolers, self._discriminators, strict=True):
            current: torch.Tensor = pooler(current)
            outputs.append(discriminator(current))
        return outputs
