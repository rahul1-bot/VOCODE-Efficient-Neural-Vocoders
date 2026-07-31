# This module:
# 1. Implements the BigVGAN discriminator ensembles: the multi-period
#    discriminator over period-folded views and the multi-resolution
#    discriminator over STFT magnitudes
# 2. Defines the per-period and per-resolution sub-discriminators the two
#    ensembles are composed of, together with the padding and normalization
#    helpers they share
#
# Design decisions:
# - Both ensembles return logits and intermediate features for the
#   adversarial and feature-matching objectives, per the reference recipe
# - Every convolution width is scaled by a single channel multiplier, so the
#   critics' capacity is one configured number rather than a hand-edited
#   channel schedule
# - The weight parametrization is configurable across both ensembles, since
#   the published family selects between spectral and weight normalization
#   per variant
# - The resolution ensemble reflect-pads and analyzes with uncentered frames,
#   which is deliberately not the conditioning mel protocol: the critic is
#   meant to see the raw spectral content
#
# Author: Rahul Sawhney

from typing import override

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

__all__: list[str] = [
    "BigvganMultiPeriodDiscriminator",
    "BigvganMultiResolutionDiscriminator"
]


class _ConvolutionPadding:
    # Padding rule for the period sub-discriminator's strided convolutions,
    # applied along the folded time axis only so the period axis is never
    # padded.
    def compute(self, kernel_size: int, dilation: int = 1) -> int:
        # Returns half the dilated kernel extent, the symmetric padding a
        # stride-one convolution of this geometry needs to preserve length.
        #
        # Args:
        #     kernel_size: Kernel extent along the padded axis.
        #     dilation: Spacing between kernel taps. Default: ``1``.
        return int((kernel_size * dilation - dilation) / 2)


class _NormalizationFactory:
    # Selects the weight parametrization applied to a discriminator
    # convolution, so the configured choice is honored identically at every
    # construction site in both ensembles.
    def create(self, module: nn.Module, use_spectral_norm: bool) -> nn.Module:
        # Wraps the module in spectral or weight normalization as selected.
        #
        # Args:
        #     module: Convolution to parametrize; it is returned wrapped, not
        #         copied, so the caller must use the return value.
        #     use_spectral_norm: Selects spectral normalization when true and
        #         weight normalization otherwise.
        if use_spectral_norm:
            return spectral_norm(module)
        return weight_norm(module)


class _BigvganPeriodDiscriminator(nn.Module):
    # One period view of the waveform. The signal is folded into a
    # two-dimensional ``[batch, channels, time // period, period]`` layout and
    # judged by convolutions whose kernels are ``(k, 1)``, so each kernel spans
    # only samples exactly one period apart. Periodicity that divides this
    # period therefore becomes structure along the folded axis, which is how
    # the ensemble sees artifacts a purely sequential critic would miss.
    def __init__(
        self,
        period: int,
        channel_multiplier: float,
        use_spectral_norm: bool,
        kernel_size: int = 5,
        stride: int = 3
    ) -> None:
        # Builds the five strided convolutions and the decision convolution.
        # The reference channel schedule doubles or quadruples at each stage
        # and is scaled throughout by the multiplier, so a fractional
        # multiplier yields a proportionally narrower critic at the same depth.
        #
        # Args:
        #     period: Sample stride the waveform is folded at.
        #     channel_multiplier: Scale applied to every convolution width.
        #     use_spectral_norm: Weight parametrization selector passed to the
        #         normalization factory.
        #     kernel_size: Time-axis extent of every strided convolution.
        #         Default: ``5``.
        #     stride: Downsampling factor along the folded time axis.
        #         Default: ``3``.
        super().__init__()
        self._period: int = period
        self._leaky_relu_slope: float = 0.1
        padding: _ConvolutionPadding = _ConvolutionPadding()
        normalization_factory: _NormalizationFactory = _NormalizationFactory()
        self._convolutions: nn.ModuleList = nn.ModuleList([
            normalization_factory.create(
                nn.Conv2d(
                    1,
                    int(32 * channel_multiplier),
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(padding.compute(kernel_size), 0)
                ),
                use_spectral_norm
            ),
            normalization_factory.create(
                nn.Conv2d(
                    int(32 * channel_multiplier),
                    int(128 * channel_multiplier),
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(padding.compute(kernel_size), 0)
                ),
                use_spectral_norm
            ),
            normalization_factory.create(
                nn.Conv2d(
                    int(128 * channel_multiplier),
                    int(512 * channel_multiplier),
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(padding.compute(kernel_size), 0)
                ),
                use_spectral_norm
            ),
            normalization_factory.create(
                nn.Conv2d(
                    int(512 * channel_multiplier),
                    int(1024 * channel_multiplier),
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(padding.compute(kernel_size), 0)
                ),
                use_spectral_norm
            ),
            normalization_factory.create(
                nn.Conv2d(
                    int(1024 * channel_multiplier),
                    int(1024 * channel_multiplier),
                    (kernel_size, 1),
                    1,
                    padding=(2, 0)
                ),
                use_spectral_norm
            )
        ])
        self._post_convolution: nn.Module = normalization_factory.create(
            nn.Conv2d(int(1024 * channel_multiplier), 1, (3, 1), 1, padding=(1, 0)),
            use_spectral_norm
        )

    @override
    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Folds the waveform to the period grid, runs the convolution stack,
        # and returns the flattened decision together with every intermediate
        # map.
        #
        # A length that is not a multiple of the period cannot be folded, so it
        # is reflect-padded up to the next multiple first; reflection is chosen
        # over zeros because a zero tail would itself be an artifact the critic
        # could learn to key on.
        #
        # Args:
        #     waveform: Signal shaped ``[batch, channels, time]``; a
        #         two-dimensional input is treated as single-channel.
        #
        # Returns:
        #     The decision flattened to ``[batch, elements]`` and the list of
        #     six feature maps, one per strided convolution plus the decision
        #     convolution's own map, which the feature-matching term consumes.
        feature_maps: list[torch.Tensor] = []
        batch_size: int = waveform.shape[0]
        channels: int = waveform.shape[1] if waveform.ndim > 2 else 1
        time_length: int = waveform.shape[-1]
        padding_amount: int = (self._period - (time_length % self._period)) % self._period
        if padding_amount > 0:
            waveform: torch.Tensor = torch.nn.functional.pad(waveform, (0, padding_amount), mode="reflect")
            time_length: int = time_length + padding_amount
        features: torch.Tensor = waveform.view(batch_size, channels, time_length // self._period, self._period)
        for convolution in self._convolutions:
            features: torch.Tensor = convolution(features)
            features: torch.Tensor = torch.nn.functional.leaky_relu(features, self._leaky_relu_slope)
            feature_maps.append(features)
        features: torch.Tensor = self._post_convolution(features)
        feature_maps.append(features)
        return torch.flatten(features, start_dim=1, end_dim=-1), feature_maps


class BigvganMultiPeriodDiscriminator(nn.Module):
    # Ensemble of period views. Each member folds the waveform at a different
    # period and judges it independently; the published periods are pairwise
    # coprime so their folded views share as little structure as possible and
    # the ensemble covers a wide band of periodicities with few members.
    def __init__(
        self,
        periods: tuple[int, ...],
        channel_multiplier: float,
        use_spectral_norm: bool
    ) -> None:
        # Instantiates one sub-discriminator per period, all at the same
        # capacity and parametrization.
        #
        # Args:
        #     periods: Sample strides judged by the ensemble.
        #     channel_multiplier: Capacity scale shared by every member.
        #     use_spectral_norm: Weight parametrization selector shared by
        #         every member.
        super().__init__()
        self._sub_discriminators: nn.ModuleList = nn.ModuleList([
            _BigvganPeriodDiscriminator(
                period=period,
                channel_multiplier=channel_multiplier,
                use_spectral_norm=use_spectral_norm
            )
            for period in periods
        ])

    @override
    def forward(
        self,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        # Judges both signals through every period view. Each member is applied
        # to the reference and then to the candidate, so the two feature stacks
        # at any index were produced by identical weights and the
        # feature-matching term compares like with like.
        #
        # Args:
        #     real_waveform: Reference signal shaped ``[batch, 1, time]``.
        #     fake_waveform: Candidate signal in the same layout; the caller
        #         detaches it for the discriminator update and leaves it
        #         attached for the generator update.
        #
        # Returns:
        #     Four lists in period order: the real logits, the candidate
        #     logits, the real feature stacks, and the candidate feature
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


class _BigvganResolutionDiscriminator(nn.Module):
    # One spectral view of the waveform. The signal is converted to an STFT
    # magnitude at the bound analysis resolution and judged as a
    # single-channel image whose axes are frequency and time, so this member
    # sees frequency-domain artifacts that survive the period-folded views
    # unchanged.
    def __init__(
        self,
        resolution: tuple[int, int, int],
        channel_multiplier: float,
        use_spectral_norm: bool
    ) -> None:
        # Builds a constant-width convolution stack and the decision
        # convolution, and records the analysis triple the forward pass
        # transforms under. The wide ``(3, 9)`` kernels striding only along
        # time keep the frequency axis intact through the stack, so a narrow
        # spectral artifact is not averaged away before the decision.
        #
        # Args:
        #     resolution: The ``(n_fft, hop_length, win_length)`` triple this
        #         member analyzes at.
        #     channel_multiplier: Capacity scale applied to the shared width.
        #     use_spectral_norm: Weight parametrization selector passed to the
        #         normalization factory.
        #
        # Raises:
        #     ValueError: If the resolution is not exactly three entries, since
        #         a shorter or longer tuple cannot describe an STFT grid.
        super().__init__()
        if len(resolution) != 3:
            raise ValueError(f"Expected BigVGAN MRD resolution of length 3, got {resolution}")
        self._resolution: tuple[int, int, int] = resolution
        self._leaky_relu_slope: float = 0.1
        normalization_factory: _NormalizationFactory = _NormalizationFactory()
        channels: int = int(32 * channel_multiplier)
        self._convolutions: nn.ModuleList = nn.ModuleList([
            normalization_factory.create(nn.Conv2d(1, channels, (3, 9), padding=(1, 4)), use_spectral_norm),
            normalization_factory.create(
                nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4)),
                use_spectral_norm
            ),
            normalization_factory.create(
                nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4)),
                use_spectral_norm
            ),
            normalization_factory.create(
                nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4)),
                use_spectral_norm
            ),
            normalization_factory.create(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1)), use_spectral_norm)
        ])
        self._post_convolution: nn.Module = normalization_factory.create(
            nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)),
            use_spectral_norm
        )

    @override
    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Analyzes the waveform into a magnitude spectrogram, adds the channel
        # axis the two-dimensional stack expects, and runs the convolutions.
        #
        # Args:
        #     waveform: Signal shaped ``[batch, 1, time]``; the channel axis is
        #         removed by the analysis and reintroduced as the image
        #         channel.
        #
        # Returns:
        #     The decision flattened to ``[batch, elements]`` and the six
        #     feature maps consumed by the feature-matching term.
        feature_maps: list[torch.Tensor] = []
        features: torch.Tensor = self._spectrogram(waveform).unsqueeze(1)
        for convolution in self._convolutions:
            features: torch.Tensor = convolution(features)
            features: torch.Tensor = torch.nn.functional.leaky_relu(features, self._leaky_relu_slope)
            feature_maps.append(features)
        features: torch.Tensor = self._post_convolution(features)
        feature_maps.append(features)
        return torch.flatten(features, start_dim=1, end_dim=-1), feature_maps

    def _spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        # Computes the magnitude spectrogram this member judges.
        #
        # The signal is reflect-padded by half the difference between the
        # transform size and the hop and then analyzed with framing disabled,
        # which reproduces centered framing while keeping the edge extension
        # reflective rather than zero-filled. The window is rectangular, so the
        # critic observes the raw spectral content instead of a tapered,
        # perceptually shaped view.
        n_fft, hop_length, win_length = self._resolution
        padding_amount: int = int((n_fft - hop_length) / 2)
        padded: torch.Tensor = torch.nn.functional.pad(waveform, (padding_amount, padding_amount), mode="reflect")
        squeezed: torch.Tensor = padded.squeeze(1)
        window: torch.Tensor = torch.ones(win_length, device=squeezed.device)
        stft: torch.Tensor = torch.stft(
            squeezed,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            return_complex=True
        )
        return torch.abs(stft)


class BigvganMultiResolutionDiscriminator(nn.Module):
    # Ensemble of spectral views. Each member analyzes at a different transform
    # size, so an artifact hidden by one time-frequency trade-off is still
    # visible to a member resolving the other way.
    def __init__(
        self,
        resolutions: tuple[tuple[int, int, int], ...],
        channel_multiplier: float,
        use_spectral_norm: bool
    ) -> None:
        # Instantiates one sub-discriminator per analysis triple.
        #
        # Args:
        #     resolutions: Exactly three ``(n_fft, hop_length, win_length)``
        #         triples the ensemble judges at.
        #     channel_multiplier: Capacity scale shared by every member.
        #     use_spectral_norm: Weight parametrization selector shared by
        #         every member.
        #
        # Raises:
        #     ValueError: If the count is not three. The published recipe fixes
        #         the ensemble at three resolutions, and the loss weighting
        #         assumes that count, so any other size is a configuration
        #         error rather than a supported variation.
        super().__init__()
        if len(resolutions) != 3:
            raise ValueError(f"BigVGAN MRD expects three resolutions, got {resolutions}")
        self._sub_discriminators: nn.ModuleList = nn.ModuleList([
            _BigvganResolutionDiscriminator(
                resolution=resolution,
                channel_multiplier=channel_multiplier,
                use_spectral_norm=use_spectral_norm
            )
            for resolution in resolutions
        ])

    @override
    def forward(
        self,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        # Judges both signals at every resolution, applying each member to the
        # reference and then to the candidate so the paired feature stacks come
        # from identical weights.
        #
        # Args:
        #     real_waveform: Reference signal shaped ``[batch, 1, time]``.
        #     fake_waveform: Candidate signal in the same layout.
        #
        # Returns:
        #     Four lists in resolution order: the real logits, the candidate
        #     logits, the real feature stacks, and the candidate feature
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
