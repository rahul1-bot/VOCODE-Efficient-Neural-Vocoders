# This module:
# 1. Implements the HiFTNet discriminator ensembles used by the adversarial
#    objective over the synthesized waveform: a multi-period ensemble
#    judging the waveform in the time domain and a multi-resolution
#    ensemble judging it in the spectral domain
#
# Design decisions:
# - The ensembles return logits and intermediate features for the
#   adversarial and feature-matching objectives, per the reference recipe
# - The period ensemble folds the waveform into a two-dimensional layout at
#   a set of mutually prime periods, so each member sees a different
#   periodic structure and their receptive fields do not coincide
# - The spectral ensemble runs at three transform resolutions, so artifacts
#   that are invisible at one time-frequency trade-off are still exposed
# - Both ensembles are discarded after training: they are owned by the
#   module rather than the network, so the synthesis surface never carries
#   critic parameters
#
# Author: Rahul Sawhney

from typing import override

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

__all__: list[str] = ["HiftnetMultiPeriodDiscriminator", "HiftnetMultiResolutionSpectrogramDiscriminator"]


class _ConvolutionPadding:
    # Stateless helper computing the symmetric padding that keeps a dilated
    # convolution length-preserving, stated once so every convolution in the
    # ensembles is visibly built from the same rule.
    def compute(self, kernel_size: int, dilation: int = 1) -> int:
        # Computes the padding that keeps the time axis length-invariant,
        # exact for the odd kernel sizes these ensembles use.
        return int((kernel_size * dilation - dilation) / 2)


class _NormalizationFactory:
    # Stateless selector between the two weight-reparameterization schemes the
    # reference recipe uses for critics. Routing the choice through one object
    # keeps the decision explicit at every construction site rather than
    # buried in a conditional import or a default argument.
    def create(self, module: nn.Module, use_spectral_norm: bool) -> nn.Module:
        # Wraps the module in spectral or weight normalization as selected.
        # Spectral normalization bounds the layer's Lipschitz constant and so
        # constrains how sharply the critic can respond, whereas weight
        # normalization only reparameterizes magnitude and direction and
        # imposes no such bound.
        if use_spectral_norm:
            return spectral_norm(module)
        return weight_norm(module)


class _HiftnetPeriodDiscriminator(nn.Module):
    # One member of the multi-period ensemble. It reshapes the waveform so
    # that samples separated by its period become adjacent along one axis,
    # turning periodic structure at that period into local structure a
    # two-dimensional convolution can see. Every convolution is restricted to
    # a single column, so the member only ever compares samples that share a
    # phase within its period.
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3) -> None:
        # Builds the convolution stack that widens from one channel to a
        # thousand twenty-four while striding along the folded time axis, plus
        # the projection to a single logit channel. Every layer is
        # weight-normalized: the spectral option is available through the
        # factory but is not selected for this member.
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
        # Folds the waveform to this member's period and scores it.
        #
        # Args:
        #     waveform: Waveform of shape ``[batch, 1, samples]``.
        #
        # Returns:
        #     A pair of the flattened logits and the list of intermediate
        #     feature maps. The features are collected after each activation
        #     and again after the output projection, so the feature-matching
        #     term compares every depth of the critic rather than only its
        #     verdict.
        feature_maps: list[torch.Tensor] = []
        batch_size: int = waveform.shape[0]
        channels: int = waveform.shape[1] if waveform.ndim > 2 else 1
        time_length: int = waveform.shape[-1]
        # The sample count is padded up to a whole multiple of the period so
        # the fold is exact. Reflection is used rather than zeros because
        # appending silence would present the critic with a discontinuity the
        # generator never produced.
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


class HiftnetMultiPeriodDiscriminator(nn.Module):
    # Time-domain critic ensemble. Each member folds the waveform at its own
    # period, and the default periods are prime, so no member's view is a
    # subsampling of another's and their receptive fields overlap as little as
    # possible. Together they cover a range of periodic structures that a
    # single flat critic would blur.
    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)) -> None:
        # Builds one folded critic per period. Default: ``(2, 3, 5, 7, 11)``,
        # the reference set.
        super().__init__()
        self._sub_discriminators: nn.ModuleList = nn.ModuleList([
            _HiftnetPeriodDiscriminator(period=period) for period in periods
        ])

    @override
    def forward(
        self,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        # Scores both waveforms through every member in one call.
        #
        # Args:
        #     real_waveform: Reference waveform of shape
        #         ``[batch, 1, samples]``.
        #     fake_waveform: Synthesized waveform of the same shape. The
        #         caller is responsible for truncating the pair to a common
        #         length and for detaching this argument when the critics are
        #         the ones being updated.
        #
        # Returns:
        #     Four lists indexed alike by member: the reference logits, the
        #     synthesis logits, the reference feature maps, and the synthesis
        #     feature maps. Taking both passes here rather than at two call
        #     sites is what guarantees the four lists stay aligned, which the
        #     objective relies on when it zips them strictly.
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


class _HiftnetSpectrogramDiscriminator(nn.Module):
    # One member of the multi-resolution spectral ensemble. It analyzes the
    # waveform at its own transform resolution and judges the resulting
    # magnitude spectrogram as a two-dimensional image, so it responds to
    # time-frequency artifacts that a time-domain critic cannot localize.
    def __init__(
        self,
        fft_size: int,
        hop_size: int,
        win_length: int,
        use_spectral_norm: bool = False
    ) -> None:
        # Records the transform geometry, registers the Hann window as a
        # non-persistent buffer, and builds the convolution stack. The stack
        # keeps a constant thirty-two channels and strides only along the
        # frequency axis, so the frame count survives to the output and the
        # logits stay time-resolved.
        #
        # Args:
        #     fft_size: Transform size of this member's analysis.
        #     hop_size: Hop length of this member's analysis.
        #     win_length: Window length of this member's analysis.
        #     use_spectral_norm: Selects spectral rather than weight
        #         normalization for every layer. Default: ``False``.
        super().__init__()
        self._fft_size: int = fft_size
        self._hop_size: int = hop_size
        self._win_length: int = win_length
        self._leaky_relu_slope: float = 0.1
        self.register_buffer("_window", torch.hann_window(win_length), persistent=False)
        normalization_factory: _NormalizationFactory = _NormalizationFactory()
        self._convolutions: nn.ModuleList = nn.ModuleList([
            normalization_factory.create(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4)), use_spectral_norm),
            normalization_factory.create(
                nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
                use_spectral_norm
            ),
            normalization_factory.create(
                nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
                use_spectral_norm
            ),
            normalization_factory.create(
                nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
                use_spectral_norm
            ),
            normalization_factory.create(nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)), use_spectral_norm)
        ])
        self._post_convolution: nn.Module = normalization_factory.create(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            use_spectral_norm
        )

    @override
    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Analyzes the waveform at this member's resolution and scores the
        # magnitude spectrogram, collecting a feature map after each
        # activation and after the output projection for the
        # feature-matching term.
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
        # Computes this member's magnitude spectrogram. The channel axis is
        # dropped for the transform and the result is transposed so frames
        # precede frequency bins, putting time on the axis the convolution
        # stack leaves unstrided. Only magnitude is retained: phase is left to
        # the time-domain ensemble, which sees it directly.
        squeezed: torch.Tensor = waveform.squeeze(1)
        stft: torch.Tensor = torch.stft(
            squeezed,
            n_fft=self._fft_size,
            hop_length=self._hop_size,
            win_length=self._win_length,
            window=self._window.to(squeezed.device),
            return_complex=True
        )
        return torch.abs(stft).transpose(2, 1)


class HiftnetMultiResolutionSpectrogramDiscriminator(nn.Module):
    # Spectral critic ensemble. Its three members analyze at coarse, finer,
    # and finest time resolution respectively, so each occupies a different
    # point on the time-frequency resolution trade-off and an artifact that
    # one member's window smears out remains visible to another.
    def __init__(
        self,
        fft_sizes: tuple[int, int, int] = (1024, 2048, 512),
        hop_sizes: tuple[int, int, int] = (120, 240, 50),
        win_lengths: tuple[int, int, int] = (600, 1200, 240)
    ) -> None:
        # Builds one spectral critic per resolution. The three tuples are
        # consumed positionally under a strict zip, so a member's transform
        # size, hop, and window length always come from the same column.
        #
        # Args:
        #     fft_sizes: Per-member transform sizes.
        #         Default: ``(1024, 2048, 512)``.
        #     hop_sizes: Per-member hop lengths.
        #         Default: ``(120, 240, 50)``.
        #     win_lengths: Per-member window lengths.
        #         Default: ``(600, 1200, 240)``.
        super().__init__()
        self._sub_discriminators: nn.ModuleList = nn.ModuleList([
            _HiftnetSpectrogramDiscriminator(fft_size=fft_size, hop_size=hop_size, win_length=win_length)
            for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths, strict=True)
        ])

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
        #     real_waveform: Reference waveform of shape
        #         ``[batch, 1, samples]``.
        #     fake_waveform: Synthesized waveform of the same shape, detached
        #         by the caller when the critics are being updated.
        #
        # Returns:
        #     The reference logits, the synthesis logits, the reference
        #     feature maps, and the synthesis feature maps, each indexed by
        #     member.
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
