# This module:
# 1. Implements the APNet2 discriminator ensembles used by the adversarial
#    objective: multi-period and multi-resolution views of the synthesized
#    waveform
# 2. Defines the per-period and per-resolution sub-discriminators the two
#    ensembles are composed of, together with the padding and normalization
#    helpers they share
#
# Design decisions:
# - Both ensembles return logits and intermediate features for the
#   adversarial and feature-matching objectives, per the reference recipe
# - Each ensemble applies every sub-discriminator to the real and the
#   candidate waveform in turn, so the feature-matching term always compares
#   maps produced by identical weights on the two signals
# - Every convolution in both ensembles is weight-normalized; APNet2 exposes
#   no spectral-norm option, unlike the BigVGAN ensembles
# - The resolution ensemble analyzes with a rectangular window and centered
#   frames, which is the reference's discriminator analysis and deliberately
#   not the conditioning mel protocol
#
# Author: Rahul Sawhney

from typing import override

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

__all__: list[str] = ["Apnet2MultiPeriodDiscriminator", "Apnet2MultiResolutionDiscriminator"]


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
    # convolution, keeping the choice in one place rather than repeated at
    # every construction site.
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


class _Apnet2PeriodDiscriminator(nn.Module):
    # One period view of the waveform. The signal is folded into a
    # two-dimensional ``[batch, channels, time // period, period]`` layout and
    # judged by convolutions whose kernels are ``(k, 1)``, so each kernel sees
    # only samples that are exactly one period apart. A periodicity that
    # divides this period therefore appears as structure along the folded
    # axis, which is how the ensemble detects the periodic artifacts a
    # waveform-domain critic misses.
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3) -> None:
        # Builds the five strided convolutions and the decision convolution at
        # the reference channel schedule. Every layer is weight-normalized:
        # the normalization factory is invoked with spectral normalization
        # disabled, because the APNet2 reference uses weight normalization
        # throughout this ensemble.
        #
        # Args:
        #     period: Sample stride the waveform is folded at; the ensemble
        #         instantiates one sub-discriminator per period.
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
                nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(padding.compute(kernel_size), 0)),
                False
            ),
            normalization_factory.create(
                nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(padding.compute(kernel_size), 0)),
                False
            ),
            normalization_factory.create(
                nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(padding.compute(kernel_size), 0)),
                False
            ),
            normalization_factory.create(
                nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(padding.compute(kernel_size), 0)),
                False
            ),
            normalization_factory.create(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)), False)
        ])
        self._post_convolution: nn.Module = normalization_factory.create(
            nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)),
            False
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


class Apnet2MultiPeriodDiscriminator(nn.Module):
    # Ensemble of period views. Each sub-discriminator folds the waveform at a
    # different period and judges it independently; the periods are pairwise
    # coprime so their folded views overlap as little as possible and the
    # ensemble covers a wide band of periodicities with few members.
    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)) -> None:
        # Instantiates one sub-discriminator per period.
        #
        # Args:
        #     periods: Sample strides judged by the ensemble; the default is
        #         the reference's five smallest primes.
        #         Default: ``(2, 3, 5, 7, 11)``.
        super().__init__()
        self._sub_discriminators: nn.ModuleList = nn.ModuleList([
            _Apnet2PeriodDiscriminator(period=period) for period in periods
        ])

    @override
    def forward(
        self,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        # Judges both signals through every period view. Each sub-discriminator
        # is applied to the reference and then to the candidate, so the two
        # feature stacks at any index were produced by identical weights and
        # the feature-matching term compares like with like.
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


class _Apnet2ResolutionDiscriminator(nn.Module):
    # One spectral view of the waveform. The signal is converted to an STFT
    # magnitude at the bound analysis resolution and judged as a
    # single-channel image, so this member sees frequency-domain artifacts
    # that survive the period-folded views unchanged.
    def __init__(self, resolution: tuple[int, int, int], channels: int = 64) -> None:
        # Builds the five strided convolutions and the decision convolution at
        # a constant width, and records the analysis triple the forward pass
        # transforms under.
        #
        # Args:
        #     resolution: The ``(n_fft, hop_length, win_length)`` triple this
        #         member analyzes at.
        #     channels: Constant width of the convolution stack.
        #         Default: ``64``.
        #
        # Raises:
        #     ValueError: If the resolution is not exactly three entries, since
        #         a shorter or longer tuple cannot describe an STFT grid.
        super().__init__()
        if len(resolution) != 3:
            raise ValueError(f"Expected APNet2 MRD resolution of length 3, got {resolution}")
        self._resolution: tuple[int, int, int] = resolution
        self._leaky_relu_slope: float = 0.1
        self._convolutions: nn.ModuleList = nn.ModuleList([
            weight_norm(nn.Conv2d(1, channels, kernel_size=(7, 5), stride=(2, 2), padding=(3, 2))),
            weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
            weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1))),
            weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 1), padding=1)),
            weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 2), padding=1))
        ])
        self._post_convolution: nn.Module = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

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
        # Computes the magnitude spectrogram this member judges. The window is
        # rectangular and the frames are centered, which is the reference's
        # discriminator analysis and intentionally differs from the tapered
        # conditioning protocol: the critic is meant to see the raw spectral
        # content, not a perceptually shaped view of it.
        n_fft, hop_length, win_length = self._resolution
        squeezed: torch.Tensor = waveform.squeeze(1)
        window: torch.Tensor = torch.ones(win_length, device=squeezed.device)
        return torch.stft(
            squeezed,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True
        ).abs()


class Apnet2MultiResolutionDiscriminator(nn.Module):
    # Ensemble of spectral views. Each member analyzes at a different
    # transform size, so an artifact that a coarse time-frequency trade-off
    # hides is still visible to a member resolving the other way.
    def __init__(
        self,
        resolutions: tuple[tuple[int, int, int], ...] = ((1024, 256, 1024), (2048, 512, 2048), (512, 128, 512))
    ) -> None:
        # Instantiates one sub-discriminator per analysis triple.
        #
        # Args:
        #     resolutions: The ``(n_fft, hop_length, win_length)`` triples the
        #         ensemble judges at; the default is the reference's three
        #         resolutions. Any count is accepted here, while each triple
        #         is validated by the sub-discriminator it configures.
        super().__init__()
        self._sub_discriminators: nn.ModuleList = nn.ModuleList([
            _Apnet2ResolutionDiscriminator(resolution=resolution) for resolution in resolutions
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
