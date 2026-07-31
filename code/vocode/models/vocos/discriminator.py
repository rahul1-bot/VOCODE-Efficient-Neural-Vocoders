# This module:
# 1. Implements the Vocos discriminator ensembles: the multi-period
#    discriminator over period-folded waveform views and the
#    multi-resolution discriminator over STFT magnitudes at three
#    analysis grids
#
# Design decisions:
# - Both ensembles return logits and intermediate features for the
#   adversarial and feature-matching objectives
# - The resolution discriminators judge spectral structure directly,
#   complementing the period ensemble's time-domain view, per the
#   reference recipe
#
# Author: Rahul Sawhney

from typing import override

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

__all__: list[str] = ["VocosMultiPeriodDiscriminator", "VocosMultiResolutionDiscriminator"]


class _VocosPeriodDiscriminator(nn.Module):
    # One member of the period ensemble. It reshapes the waveform into a
    # two-dimensional view whose second axis is the period, so strided
    # two-dimensional convolutions striding only along the first axis
    # examine samples that are one period apart. The construction follows
    # the HiFi-GAN period discriminator, with two differences: padding is
    # derived from the kernel width rather than fixed, and the first
    # convolution's activation is excluded from the feature maps, so each
    # member contributes five maps rather than six.
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3) -> None:
        # Builds the five widening convolutions and the single-channel
        # post-convolution, all weight-normalized. The activation slope is
        # fixed here rather than accepted as an argument, because the
        # reference recipe holds it constant across the ensemble.
        #
        # Args:
        #     period: Sample distance the waveform is folded at.
        #     kernel_size: Kernel extent along the time axis; its half
        #         also becomes the time-axis padding. Default: ``5``.
        #     stride: Time-axis stride of the first four convolutions.
        #         Default: ``3``.
        super().__init__()
        self._period: int = period
        self._leaky_relu_slope: float = 0.1
        self._convolutions: nn.ModuleList = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), (1, 1), padding=(kernel_size // 2, 0)))
        ])
        self._post_convolution: nn.Module = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    @override
    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Reflect-pads the waveform up to a whole multiple of the period,
        # folds it into the two-dimensional period view, and runs the
        # convolution stack.
        #
        # Args:
        #     waveform: Waveform batch shaped [batch, samples]; the
        #         channel axis the convolutions need is inserted here
        #         rather than expected from the caller.
        #
        # Returns:
        #     The flattened logits shaped [batch, values] paired with five
        #     feature maps. The first convolution's activation is
        #     deliberately omitted from the maps, so the feature-matching
        #     term starts one layer deeper than in the HiFi-GAN ensemble.
        #
        # Note:
        #     Padding is applied before folding, so a waveform whose
        #     length is not a period multiple yields the same logit width
        #     as the next multiple up rather than failing.
        feature_maps: list[torch.Tensor] = []
        batch_size: int = waveform.shape[0]
        time_length: int = waveform.shape[-1]
        padding_amount: int = (self._period - (time_length % self._period)) % self._period
        if padding_amount > 0:
            waveform: torch.Tensor = torch.nn.functional.pad(waveform, (0, padding_amount), mode="reflect")
            time_length: int = time_length + padding_amount
        features: torch.Tensor = waveform.unsqueeze(1).view(batch_size, 1, time_length // self._period, self._period)
        for index, convolution in enumerate(self._convolutions):
            features: torch.Tensor = convolution(features)
            features: torch.Tensor = torch.nn.functional.leaky_relu(features, self._leaky_relu_slope)
            if index > 0:
                feature_maps.append(features)
        features: torch.Tensor = self._post_convolution(features)
        feature_maps.append(features)
        return torch.flatten(features, start_dim=1, end_dim=-1), feature_maps


class VocosMultiPeriodDiscriminator(nn.Module):
    # The period ensemble: one sub-discriminator per prime period, each
    # judging the same waveform through its own fold. Prime periods are
    # chosen so no two folds share a common divisor and the views overlap
    # as little as possible. This is the time-domain half of the family's
    # adversarial signal; the resolution ensemble supplies the spectral
    # half.
    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)) -> None:
        # Builds one sub-discriminator per period at their default kernel
        # and stride settings.
        #
        # Args:
        #     periods: The fold periods of the ensemble; their count is
        #         the ensemble size. Default: ``(2, 3, 5, 7, 11)``.
        super().__init__()
        self._sub_discriminators: nn.ModuleList = nn.ModuleList([
            _VocosPeriodDiscriminator(period=period) for period in periods
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
        #     real_waveform: Reference waveform shaped [batch, samples].
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


class _VocosResolutionDiscriminator(nn.Module):
    # One member of the resolution ensemble. It judges the waveform in the
    # spectral domain: the signal is analyzed at one window length, split
    # into frequency bands, and each band is examined by its own
    # convolution stack before the band outputs are concatenated for a
    # single decision. Judging bands separately is what lets the ensemble
    # penalize a generator that reproduces overall spectral energy while
    # getting one region wrong.
    def __init__(
        self,
        window_length: int,
        channels: int = 32,
        hop_factor: float = 0.25,
        bands: tuple[tuple[float, float], ...] = (
            (0.0, 0.1),
            (0.1, 0.25),
            (0.25, 0.5),
            (0.5, 0.75),
            (0.75, 1.0)
        )
    ) -> None:
        # Converts the fractional band edges into bin indices against this
        # window's bin count, registers the analysis window, and builds
        # one five-layer convolution stack per band plus a shared
        # post-convolution over the concatenated band outputs.
        #
        # Args:
        #     window_length: Analysis window and transform size, which
        #         fixes the frequency resolution of this member.
        #     channels: Width of every band stack. Default: ``32``.
        #     hop_factor: Analysis hop as a fraction of the window
        #         length. Default: ``0.25``.
        #     bands: Band edges as fractions of the spectrum, resolved to
        #         bin indices at construction. The reference bands are
        #         narrow at low frequency and wide at high frequency,
        #         which allocates capacity where speech energy and
        #         perceptual sensitivity concentrate.
        #
        # Note:
        #     The first convolution of each band stack takes two input
        #     channels, because the analysis keeps the real and imaginary
        #     parts of the spectrum as channels rather than reducing them
        #     to a magnitude.
        super().__init__()
        self._window_length: int = window_length
        self._hop_length: int = int(window_length * hop_factor)
        bin_count: int = window_length // 2 + 1
        self._bands: tuple[tuple[int, int], ...] = tuple(
            (int(start * bin_count), int(stop * bin_count)) for start, stop in bands
        )
        self.register_buffer("_window", torch.hann_window(window_length), persistent=False)
        self._band_convolutions: nn.ModuleList = nn.ModuleList([
            nn.ModuleList([
                weight_norm(nn.Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1)))
            ])
            for _ in self._bands
        ])
        self._post_convolution: nn.Module = weight_norm(nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))

    @override
    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Runs each frequency band through its own stack, concatenates the
        # band outputs along the frequency axis, and reduces the result to
        # a single channel. Strict pairing asserts the band views and the
        # band stacks never fall out of step.
        #
        # Args:
        #     waveform: Waveform batch shaped [batch, samples].
        #
        # Returns:
        #     The merged single-channel logit map, which keeps its
        #     spectral layout rather than being flattened, paired with
        #     twenty-one feature maps: four from each of the five bands,
        #     since the first layer of every stack is omitted, plus the
        #     merged post-convolution map.
        feature_maps: list[torch.Tensor] = []
        band_outputs: list[torch.Tensor] = []
        for band, convolution_stack in zip(self._spectrogram_bands(waveform), self._band_convolutions, strict=True):
            features: torch.Tensor = band
            for index, layer in enumerate(convolution_stack):
                features: torch.Tensor = layer(features)
                features: torch.Tensor = torch.nn.functional.leaky_relu(features, 0.1)
                if index > 0:
                    feature_maps.append(features)
            band_outputs.append(features)
        merged: torch.Tensor = torch.cat(band_outputs, dim=-1)
        merged: torch.Tensor = self._post_convolution(merged)
        feature_maps.append(merged)
        return merged, feature_maps

    def _spectrogram_bands(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        # Computes frequency-band spectrogram views for the Vocos resolution discriminator.
        # The waveform is first centered and peak-normalized to a fixed
        # amplitude, so the discriminator judges spectral shape rather
        # than loudness and cannot separate real from fake on gain alone;
        # the epsilon in the denominator keeps a silent segment finite.
        # The complex spectrum is then kept as real and imaginary channels
        # and permuted so frequency is the last axis, which is what makes
        # the band split a contiguous slice.
        #
        # Args:
        #     waveform: Waveform batch shaped [batch, samples].
        #
        # Returns:
        #     One contiguous view per configured band, each shaped
        #     [batch, 2, frames, band_bins].
        normalized: torch.Tensor = waveform - waveform.mean(dim=-1, keepdim=True)
        normalized: torch.Tensor = 0.8 * normalized / (normalized.abs().amax(dim=-1, keepdim=True) + 1e-9)
        stft: torch.Tensor = torch.stft(
            normalized,
            n_fft=self._window_length,
            hop_length=self._hop_length,
            win_length=self._window_length,
            window=self._window.to(normalized.device),
            return_complex=True
        )
        real_view: torch.Tensor = torch.view_as_real(stft).permute(0, 3, 2, 1)
        return [real_view[..., start:stop].contiguous() for start, stop in self._bands]


class VocosMultiResolutionDiscriminator(nn.Module):
    # The resolution ensemble: one sub-discriminator per analysis window
    # length, judging spectral structure at three time-frequency
    # trade-offs at once. A long window resolves frequency finely and time
    # coarsely, a short window the reverse, so no single artefact scale
    # can hide from all three. This complements the period ensemble's
    # time-domain view.
    def __init__(self, fft_sizes: tuple[int, ...] = (2048, 1024, 512)) -> None:
        # Builds one sub-discriminator per window length at the shared
        # default channel width, hop factor, and band split.
        #
        # Args:
        #     fft_sizes: Analysis window lengths of the ensemble; their
        #         count is the ensemble size. Default:
        #         ``(2048, 1024, 512)``.
        super().__init__()
        self._sub_discriminators: nn.ModuleList = nn.ModuleList([
            _VocosResolutionDiscriminator(window_length=window_length) for window_length in fft_sizes
        ])

    @override
    def forward(
        self,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        # Runs both waveforms through every analysis grid, keeping the
        # real and fake results in positional correspondence so the losses
        # can pair them per resolution without carrying an index.
        #
        # Args:
        #     real_waveform: Reference waveform shaped [batch, samples].
        #     fake_waveform: Synthesized waveform in the same layout,
        #         detached or attached according to which update is
        #         running.
        #
        # Returns:
        #     Four lists of one entry per analysis grid: real logits, fake
        #     logits, real feature-map stacks, and fake feature-map
        #     stacks. The logits keep their spectral layout rather than
        #     being flattened, unlike the period ensemble's.
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
