# This module:
# 1. Implements the HiFi-GAN discriminator ensembles: the multi-period
#    discriminator viewing the waveform folded at prime periods, and the
#    multi-scale discriminator viewing progressively average-pooled
#    scales
#
# Design decisions:
# - Each sub-discriminator returns both logits and intermediate feature
#   maps, because the generator objective includes feature matching
#   alongside the adversarial term
# - The first scale discriminator uses spectral normalization and the
#   remainder weight normalization, following the reference recipe
# - Period folding pads the waveform to a period multiple and reshapes
#   it into a two-dimensional view, letting strided 2D convolutions
#   examine periodic structure directly
#
# Author: Rahul Sawhney

from collections.abc import Callable
from typing import override

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

__all__: list[str] = [
    "DiscriminatorP",
    "DiscriminatorS",
    "MultiPeriodDiscriminator",
    "MultiScaleDiscriminator"
]


class DiscriminatorP(nn.Module):
    # One member of the multi-period ensemble. It reshapes the waveform
    # into a two-dimensional view whose second axis is the period, so
    # strided two-dimensional convolutions striding only along the first
    # axis examine samples that are one period apart. A generator that
    # reproduces the spectrum but not the periodic structure of voiced
    # speech is what this view is built to expose.
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
        leaky_relu_slope: float = 0.1
    ) -> None:
        # Builds the five widening convolutions and the single-channel
        # post-convolution. All kernels and strides are one along the
        # period axis, so folding is never mixed across periods, and the
        # last convolution drops to stride one before the head.
        #
        # Args:
        #     period: Sample distance the waveform is folded at; the
        #         ensemble uses five prime periods so their folds share no
        #         common structure.
        #     kernel_size: Kernel extent along the time axis of every
        #         convolution. Default: ``5``.
        #     stride: Time-axis stride of the first four convolutions.
        #         Default: ``3``.
        #     use_spectral_norm: Selects spectral normalization instead of
        #         weight normalization for every convolution.
        #         Default: ``False``.
        #     leaky_relu_slope: Negative slope of the activation after
        #         each convolution. Default: ``0.1``.
        super().__init__()
        self._period: int = period
        self._leaky_relu_slope: float = leaky_relu_slope
        norm_function: Callable[[nn.Module], nn.Module] = spectral_norm if use_spectral_norm else weight_norm
        self._convolutions: nn.ModuleList = nn.ModuleList([
            norm_function(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_function(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_function(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_function(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_function(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)))
        ])
        self._post_convolution: nn.Module = norm_function(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    @override
    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Reflect-pads the waveform up to a whole multiple of the period,
        # folds it into the two-dimensional period view, and runs the
        # convolution stack, recording every activation as a feature map.
        #
        # Args:
        #     waveform: Waveform batch in the [batch, 1, samples] layout;
        #         a two-dimensional input is read as a single channel.
        #
        # Returns:
        #     The flattened logits shaped [batch, values] paired with six
        #     feature maps, one per convolution including the
        #     post-convolution. The feature-matching term sums over all
        #     six, so the count is part of the objective rather than an
        #     implementation detail.
        #
        # Note:
        #     Padding is applied before folding, so a waveform whose
        #     length is not a period multiple yields the same logit width
        #     as the next multiple up rather than failing.
        feature_maps: list[torch.Tensor] = []
        batch_size: int = waveform.shape[0]
        channels: int = waveform.shape[1] if waveform.ndim > 2 else 1
        time_length: int = waveform.shape[-1]
        padding_amount: int = (self._period - (time_length % self._period)) % self._period
        if padding_amount > 0:
            waveform: torch.Tensor = torch.nn.functional.pad(waveform, (0, padding_amount), mode="reflect")
            time_length: int = time_length + padding_amount
        reshaped: torch.Tensor = waveform.view(batch_size, channels, time_length // self._period, self._period)
        features: torch.Tensor = reshaped
        for convolution in self._convolutions:
            features: torch.Tensor = convolution(features)
            features: torch.Tensor = torch.nn.functional.leaky_relu(features, self._leaky_relu_slope)
            feature_maps.append(features)
        features: torch.Tensor = self._post_convolution(features)
        feature_maps.append(features)
        return torch.flatten(features, start_dim=1, end_dim=-1), feature_maps


class MultiPeriodDiscriminator(nn.Module):
    # The period ensemble: one sub-discriminator per prime period, each
    # judging the same waveform through its own fold. Prime periods are
    # chosen so no two folds share a common divisor and the views overlap
    # as little as possible.
    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)) -> None:
        # Builds one sub-discriminator per period at their default kernel,
        # stride, and normalization settings.
        #
        # Args:
        #     periods: The fold periods of the ensemble; their count is
        #         the ensemble size. Default: ``(2, 3, 5, 7, 11)``.
        super().__init__()
        self._sub_discriminators: nn.ModuleList = nn.ModuleList([
            DiscriminatorP(period=period) for period in periods
        ])

    @override
    def forward(
        self,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        # Runs both waveforms through every sub-discriminator, keeping the
        # real and fake results in positional correspondence so the losses
        # can pair them per period without carrying an index.
        #
        # Args:
        #     real_waveform: Reference waveform in discriminator layout.
        #     fake_waveform: Synthesized waveform in the same layout. The
        #         caller decides whether it arrives detached, which is
        #         what separates the discriminator update from the
        #         generator update.
        #
        # Returns:
        #     Four lists of one entry per period: real logits, fake
        #     logits, real feature-map stacks, and fake feature-map
        #     stacks.
        real_outputs: list[torch.Tensor] = []
        fake_outputs: list[torch.Tensor] = []
        real_feature_maps: list[list[torch.Tensor]] = []
        fake_feature_maps: list[list[torch.Tensor]] = []
        for sub_discriminator in self._sub_discriminators:
            real_logit, real_features = sub_discriminator(real_waveform)
            fake_logit, fake_features = sub_discriminator(fake_waveform)
            real_outputs.append(real_logit)
            fake_outputs.append(fake_logit)
            real_feature_maps.append(real_features)
            fake_feature_maps.append(fake_features)
        return real_outputs, fake_outputs, real_feature_maps, fake_feature_maps


class DiscriminatorS(nn.Module):
    # One member of the multi-scale ensemble. It reads the waveform
    # directly in the time domain through a stack of wide, heavily grouped
    # strided convolutions, so it judges long-range temporal structure
    # rather than the periodic structure the period ensemble folds for.
    def __init__(self, use_spectral_norm: bool = False, leaky_relu_slope: float = 0.1) -> None:
        # Builds the seven-convolution stack and the single-channel post-
        # convolution. Grouping rises with width so the wide middle layers
        # stay affordable, and the final two convolutions return to stride
        # one before the head.
        #
        # Args:
        #     use_spectral_norm: Selects spectral normalization instead of
        #         weight normalization for every convolution. The ensemble
        #         sets this only on the member that sees the unpooled
        #         waveform, following the reference recipe.
        #         Default: ``False``.
        #     leaky_relu_slope: Negative slope of the activation after
        #         each convolution. Default: ``0.1``.
        super().__init__()
        self._leaky_relu_slope: float = leaky_relu_slope
        norm_function: Callable[[nn.Module], nn.Module] = spectral_norm if use_spectral_norm else weight_norm
        self._convolutions: nn.ModuleList = nn.ModuleList([
            norm_function(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_function(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_function(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_function(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_function(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_function(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_function(nn.Conv1d(1024, 1024, 5, 1, padding=2))
        ])
        self._post_convolution: nn.Module = norm_function(nn.Conv1d(1024, 1, 3, 1, padding=1))

    @override
    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Runs the waveform straight through the convolution stack with no
        # reshaping, recording every activation as a feature map.
        #
        # Args:
        #     waveform: Waveform batch in the [batch, 1, samples] layout,
        #         already pooled to this member's scale by the ensemble.
        #
        # Returns:
        #     The flattened logits shaped [batch, values] paired with
        #     eight feature maps, one per convolution including the
        #     post-convolution.
        feature_maps: list[torch.Tensor] = []
        features: torch.Tensor = waveform
        for convolution in self._convolutions:
            features: torch.Tensor = convolution(features)
            features: torch.Tensor = torch.nn.functional.leaky_relu(features, self._leaky_relu_slope)
            feature_maps.append(features)
        features: torch.Tensor = self._post_convolution(features)
        feature_maps.append(features)
        return torch.flatten(features, start_dim=1, end_dim=-1), feature_maps


class MultiScaleDiscriminator(nn.Module):
    # The scale ensemble: three identically shaped sub-discriminators
    # judging the waveform at the original rate and at two successively
    # halved rates. The ensemble takes no configuration because the
    # reference recipe fixes its three scales and their normalizations.
    def __init__(self) -> None:
        # Builds the three sub-discriminators and the two average poolers
        # applied between consecutive scales. Only the first
        # sub-discriminator uses spectral normalization, which is the
        # reference asymmetry: it is the one reading the unpooled
        # waveform, where the sharpest gradients arise.
        super().__init__()
        self._sub_discriminators: nn.ModuleList = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS()
        ])
        self._mean_poolers: nn.ModuleList = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2)
        ])

    @override
    def forward(
        self,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        # Walks the scales in order, pooling cumulatively: each scale
        # after the first pools the already-pooled tensor rather than the
        # original, so the three views sit at the full rate, one half, and
        # one quarter. Real and fake are pooled through the same operators
        # and stay in positional correspondence.
        #
        # Args:
        #     real_waveform: Reference waveform in discriminator layout.
        #     fake_waveform: Synthesized waveform in the same layout,
        #         detached or attached according to which update is
        #         running.
        #
        # Returns:
        #     Four lists of three entries: real logits, fake logits, real
        #     feature-map stacks, and fake feature-map stacks. Logit width
        #     shrinks from scale to scale, which is the observable
        #     signature of the pooling.
        real_outputs: list[torch.Tensor] = []
        fake_outputs: list[torch.Tensor] = []
        real_feature_maps: list[list[torch.Tensor]] = []
        fake_feature_maps: list[list[torch.Tensor]] = []
        scaled_real: torch.Tensor = real_waveform
        scaled_fake: torch.Tensor = fake_waveform
        for scale_index, sub_discriminator in enumerate(self._sub_discriminators):
            if scale_index > 0:
                scaled_real: torch.Tensor = self._mean_poolers[scale_index - 1](scaled_real)
                scaled_fake: torch.Tensor = self._mean_poolers[scale_index - 1](scaled_fake)
            real_logit, real_features = sub_discriminator(scaled_real)
            fake_logit, fake_features = sub_discriminator(scaled_fake)
            real_outputs.append(real_logit)
            fake_outputs.append(fake_logit)
            real_feature_maps.append(real_features)
            fake_feature_maps.append(fake_features)
        return real_outputs, fake_outputs, real_feature_maps, fake_feature_maps
