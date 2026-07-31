# This module:
# 1. Implements the RNDVoC discriminator ensembles used by the adversarial
#    objective over the synthesized waveform: a multi-period ensemble
#    judging the waveform in the time domain and a multi-resolution ensemble
#    judging its magnitude spectrogram
#
# Design decisions:
# - The ensembles return logits and intermediate features for the
#   adversarial and feature-matching objectives, per the reference recipe
# - The period ensemble folds the waveform at a set of mutually prime
#   periods, so no member's view is a subsampling of another's
# - The spectral ensemble strides along both the time and the frequency
#   axis, unlike the time-preserving spectral critic of the HiFTNet family,
#   so its members produce a coarse two-dimensional verdict map rather than
#   a time-resolved one
# - Spectral analysis is forced to float32 under a locally disabled autocast
#   context, so the critic's view of a waveform does not change with the
#   trainer's precision setting
# - Both ensembles are owned by the module rather than the network, so they
#   are discarded after training and never reach the synthesis surface
#
# Author: Rahul Sawhney

from typing import cast, override

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

__all__: list[str] = ["RndvocMultiPeriodDiscriminator", "RndvocMultiResolutionDiscriminator"]


class _RndvocPeriodDiscriminator(nn.Module):
    # One member of the multi-period ensemble. It reshapes the waveform so
    # that samples separated by its period become adjacent along one axis,
    # turning periodic structure at that period into local structure a
    # two-dimensional convolution can see. Every kernel is one column wide, so
    # the member only ever compares samples sharing a phase within its period.
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3) -> None:
        # Builds the widening convolution stack and the projection to a single
        # logit channel, all weight-normalized as the reference recipe
        # specifies.
        super().__init__()
        self._period: int = period
        self._leaky_slope: float = 0.1
        padding: int = (kernel_size - 1) // 2
        self._convolutions: nn.ModuleList = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(padding, 0))),
                weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(padding, 0))),
                weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(padding, 0))),
                weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(padding, 0))),
                weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)))
            ]
        )
        self._output_convolution: nn.Module = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    @override
    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Folds the waveform to this member's period and scores it. A channel
        # axis is added when absent, so the member accepts this family's
        # channel-free synthesis output directly.
        #
        # Args:
        #     waveform: Waveform of shape ``[batch, samples]`` or
        #         ``[batch, 1, samples]``.
        #
        # Returns:
        #     A pair of the flattened logits and the feature maps collected
        #     after each activation and after the output projection, so the
        #     feature-matching term compares every depth of the critic rather
        #     than only its verdict.
        feature_maps: list[torch.Tensor] = []
        if waveform.ndim == 2:
            waveform: torch.Tensor = waveform.unsqueeze(1)
        batch_size: int = waveform.shape[0]
        channel_count: int = waveform.shape[1]
        sample_count: int = waveform.shape[2]
        # The sample count is padded up to a whole multiple of the period so
        # the fold is exact. Reflection is used rather than zeros, because
        # appending silence would present a discontinuity the generator never
        # produced.
        if sample_count % self._period != 0:
            padding_amount: int = self._period - (sample_count % self._period)
            waveform: torch.Tensor = F.pad(waveform, (0, padding_amount), "reflect")
            sample_count: int = sample_count + padding_amount
        folded: torch.Tensor = waveform.view(batch_size, channel_count, sample_count // self._period, self._period)
        for convolution in self._convolutions:
            convolution_module: nn.Module = cast(nn.Module, convolution)
            folded: torch.Tensor = F.leaky_relu(convolution_module(folded), self._leaky_slope)
            feature_maps.append(folded)
        logits: torch.Tensor = self._output_convolution(folded)
        feature_maps.append(logits)
        return torch.flatten(logits, 1, -1), feature_maps


class RndvocMultiPeriodDiscriminator(nn.Module):
    # Time-domain critic ensemble. Each member folds the waveform at its own
    # period, and the default periods are prime, so no member's view is a
    # subsampling of another's and their receptive fields overlap as little as
    # possible.
    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)) -> None:
        # Builds one folded critic per period. Default: ``(2, 3, 5, 7, 11)``,
        # the reference set.
        super().__init__()
        self._discriminators: nn.ModuleList = nn.ModuleList(
            _RndvocPeriodDiscriminator(period) for period in periods
        )

    @override
    def forward(
        self,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        # Scores both waveforms through every member in one call.
        #
        # Args:
        #     real_waveform: Reference waveform.
        #     fake_waveform: Synthesized waveform, truncated by the caller to
        #         a common length and detached when the critics are the ones
        #         being updated.
        #
        # Returns:
        #     Four lists indexed alike by member: the reference logits, the
        #     synthesis logits, and both feature-map stacks. Taking both
        #     passes here rather than at two call sites is what guarantees the
        #     four stay aligned, which the objective relies on when it zips
        #     them strictly.
        real_logits: list[torch.Tensor] = []
        fake_logits: list[torch.Tensor] = []
        real_features: list[list[torch.Tensor]] = []
        fake_features: list[list[torch.Tensor]] = []
        for discriminator in self._discriminators:
            period_discriminator: _RndvocPeriodDiscriminator = cast(_RndvocPeriodDiscriminator, discriminator)
            real_logit: torch.Tensor
            real_feature_maps: list[torch.Tensor]
            fake_logit: torch.Tensor
            fake_feature_maps: list[torch.Tensor]
            real_logit, real_feature_maps = period_discriminator(real_waveform)
            fake_logit, fake_feature_maps = period_discriminator(fake_waveform)
            real_logits.append(real_logit)
            fake_logits.append(fake_logit)
            real_features.append(real_feature_maps)
            fake_features.append(fake_feature_maps)
        return real_logits, fake_logits, real_features, fake_features


class _RndvocResolutionDiscriminator(nn.Module):
    # One member of the multi-resolution spectral ensemble. It analyzes the
    # waveform at its own transform resolution and judges the resulting
    # magnitude spectrogram as a two-dimensional image, so it responds to
    # time-frequency artifacts a time-domain critic cannot localize.
    #
    # Unlike the corresponding HiFTNet member, this one strides along both
    # axes, reducing time as well as frequency. Its verdict is therefore a
    # coarse map over time-frequency regions rather than a per-frame score,
    # which trades temporal precision for a wider receptive field per unit of
    # depth.
    def __init__(self, resolution: tuple[int, int, int], channel_count: int = 64) -> None:
        # Records the transform geometry and builds the convolution stack at a
        # constant channel width.
        #
        # Args:
        #     resolution: The transform size, hop length, and window length of
        #         this member's analysis, in that order.
        #     channel_count: Constant width of the convolution stack.
        #         Default: ``64``.
        super().__init__()
        self._resolution: tuple[int, int, int] = resolution
        self._leaky_slope: float = 0.1
        self._convolutions: nn.ModuleList = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, channel_count, kernel_size=(7, 5), stride=(2, 2), padding=(3, 2))),
                weight_norm(nn.Conv2d(channel_count, channel_count, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
                weight_norm(nn.Conv2d(channel_count, channel_count, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1))),
                weight_norm(nn.Conv2d(channel_count, channel_count, kernel_size=3, stride=(2, 1), padding=1)),
                weight_norm(nn.Conv2d(channel_count, channel_count, kernel_size=3, stride=(2, 2), padding=1))
            ]
        )
        self._output_convolution: nn.Module = weight_norm(
            nn.Conv2d(channel_count, 1, (3, 3), padding=(1, 1))
        )

    @override
    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Analyzes the waveform at this member's resolution and scores the
        # magnitude spectrogram as a single-channel image, collecting a
        # feature map after each activation and after the output projection.
        # A singleton channel axis is dropped when present, so the member
        # accepts either waveform layout.
        feature_maps: list[torch.Tensor] = []
        if waveform.ndim == 3:
            waveform: torch.Tensor = waveform.squeeze(1)
        magnitude: torch.Tensor = self._magnitude_spectrogram(waveform).unsqueeze(1)
        for convolution in self._convolutions:
            convolution_module: nn.Module = cast(nn.Module, convolution)
            magnitude: torch.Tensor = F.leaky_relu(convolution_module(magnitude), self._leaky_slope)
            feature_maps.append(magnitude)
        logits: torch.Tensor = self._output_convolution(magnitude)
        feature_maps.append(logits)
        return torch.flatten(logits, 1, -1), feature_maps

    def _magnitude_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        # Computes the magnitude spectrogram at this member's transform
        # resolution. Analysis is forced to float32 under a locally disabled
        # autocast context, so what the critic sees does not depend on the
        # trainer's precision setting; a critic whose input changed with
        # precision would make adversarial pressure itself precision-dependent.
        # Only magnitude is retained, since phase is left to the time-domain
        # ensemble, which observes it directly.
        n_fft: int = self._resolution[0]
        hop_length: int = self._resolution[1]
        win_length: int = self._resolution[2]
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            return torch.stft(
                waveform.float(),
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=torch.hann_window(win_length, device=waveform.device),
                center=True,
                return_complex=True
            ).abs()


class RndvocMultiResolutionDiscriminator(nn.Module):
    # Spectral critic ensemble. Its three members analyze at coarse, finer,
    # and finest time resolution, so each sits at a different point on the
    # time-frequency resolution trade-off and an artifact one member's window
    # smears out remains visible to another. Each resolution's window length
    # equals its transform size, so the analyses use unpadded full-length
    # windows.
    def __init__(
        self,
        resolutions: tuple[tuple[int, int, int], ...] = ((1024, 256, 1024), (2048, 512, 2048), (512, 128, 512))
    ) -> None:
        # Builds one spectral critic per resolution. Each entry carries its
        # own transform size, hop, and window length as a single triple, so
        # the three settings of a member cannot become mismatched.
        super().__init__()
        self._discriminators: nn.ModuleList = nn.ModuleList(
            _RndvocResolutionDiscriminator(resolution) for resolution in resolutions
        )

    @override
    def forward(
        self,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        # Scores both waveforms at every resolution in one call, returning the
        # same four aligned lists as the period ensemble so the objective
        # treats the two ensembles identically.
        #
        # Args:
        #     real_waveform: Reference waveform.
        #     fake_waveform: Synthesized waveform, detached by the caller when
        #         the critics are being updated.
        #
        # Returns:
        #     The reference logits, the synthesis logits, and both
        #     feature-map stacks, each indexed by member.
        real_logits: list[torch.Tensor] = []
        fake_logits: list[torch.Tensor] = []
        real_features: list[list[torch.Tensor]] = []
        fake_features: list[list[torch.Tensor]] = []
        for discriminator in self._discriminators:
            resolution_discriminator: _RndvocResolutionDiscriminator = cast(
                _RndvocResolutionDiscriminator,
                discriminator
            )
            real_logit: torch.Tensor
            real_feature_maps: list[torch.Tensor]
            fake_logit: torch.Tensor
            fake_feature_maps: list[torch.Tensor]
            real_logit, real_feature_maps = resolution_discriminator(real_waveform)
            fake_logit, fake_feature_maps = resolution_discriminator(fake_waveform)
            real_logits.append(real_logit)
            fake_logits.append(fake_logit)
            real_features.append(real_feature_maps)
            fake_features.append(fake_feature_maps)
        return real_logits, fake_logits, real_features, fake_features
