# This module:
# 1. Implements the BigVGAN-base generator network: HiFi-GAN-style
#    transposed-convolution upsampling with anti-aliased multi-periodicity
#    (AMP) residual blocks built on snake activations with low-pass
#    filtered up-down sampling
# 2. Implements the anti-aliasing apparatus those blocks depend on: the
#    Kaiser windowed-sinc filter design, the filtered upsampler and
#    downsampler, and the wrapper that sandwiches an activation between them
# 3. Implements the two periodic activations (snake and snakebeta) and the
#    two residual block variants the published recipes select between
#
# Design decisions:
# - Snake activations inject learnable periodicity while the surrounding
#   anti-aliasing filters suppress the aliasing such nonlinearities
#   otherwise create, the architecture's defining idea
# - The snakebeta variant with log-scale amplitude parameters follows the
#   published base configuration
# - Every resampling filter is a unit-sum windowed sinc, so the anti-aliasing
#   apparatus is transparent at DC and cannot rescale the signal it protects
# - The upsampling stages are nested one level deep inside nn.ModuleList, so
#   the state-dictionary keys carry the author's index layout and the
#   published release loads strictly
# - Filters are registered as buffers, so they move with the module and
#   travel in the state dictionary rather than being rebuilt per call
#
# Author: Rahul Sawhney

import math
from typing import Literal, override

import torch
import torch.nn.functional as functional
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

__all__: list[str] = ["BigvganNetwork"]


class _ConvolutionPadding:
    # Padding rule for the dilated residual convolutions. The residual stacks
    # are shape-preserving refinements, so each convolution must return the
    # time axis it was given whatever its dilation.
    def compute(self, kernel_size: int, dilation: int = 1) -> int:
        # Returns half the dilated kernel extent, the symmetric padding that
        # leaves a stride-one convolution length-invariant.
        #
        # Args:
        #     kernel_size: Kernel extent along the time axis.
        #     dilation: Spacing between kernel taps. Default: ``1``.
        return int((kernel_size * dilation - dilation) / 2)


class _NormalWeightInitializer:
    # Applies the reference initialization to a generator convolution, and
    # exists as a separate object so the initialize-then-wrap ordering is
    # expressed once and cannot be inverted at an individual call site.
    # Generator convolutions start from normal(0, 0.01) before weight normalization wraps them,
    # because the parametrized weight decomposes the initialized tensor at wrap time.
    def apply[ConvolutionType: (nn.Conv1d, nn.ConvTranspose1d)](self, convolution: ConvolutionType) -> ConvolutionType:
        # Overwrites the convolution's weight in place with the reference
        # normal initialization and returns the same object, so the call
        # composes directly inside a weight_norm wrap. Ordering matters: the
        # wrap decomposes whatever weight it finds into its magnitude and
        # direction components, so initializing afterwards would not reach the
        # parametrized tensors.
        #
        # Args:
        #     convolution: Convolution or transposed convolution to
        #         initialize; returned unwrapped for the caller to wrap.
        torch.nn.init.normal_(convolution.weight, mean=0.0, std=0.01)
        return convolution


class _KaiserSincFilterFactory:
    # Designs the low-pass kernels the anti-aliasing apparatus resamples with.
    # A sinc is the ideal brick-wall low-pass but is infinitely long, so it is
    # truncated to the kernel size and tapered by a Kaiser window, which trades
    # transition width against stopband attenuation in a controlled way.
    def create(self, cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
        # Builds the windowed-sinc low-pass kernel for anti-aliased resampling.
        #
        # The sinc is sampled on a time axis centered on the kernel, scaled by
        # twice the cutoff so its zero crossings land at the intended band
        # edge, and normalized to unit sum, which fixes the DC gain at one so
        # resampling neither amplifies nor attenuates the signal. A zero cutoff
        # degenerates to an all-zero kernel and is returned directly, since the
        # normalization would otherwise divide by zero.
        #
        # Args:
        #     cutoff: Normalized cutoff frequency, in cycles per sample.
        #     half_width: Normalized half-width of the transition band; a
        #         narrower transition demands a higher window attenuation and
        #         therefore a larger Kaiser beta.
        #     kernel_size: Tap count of the returned kernel.
        #
        # Returns:
        #     The kernel shaped ``[1, 1, kernel_size]``, ready to be expanded
        #     across channels for a grouped convolution.
        even: bool = kernel_size % 2 == 0
        half_size: int = kernel_size // 2
        window: torch.Tensor = self._create_window(
            kernel_size=kernel_size,
            half_size=half_size,
            half_width=half_width
        )
        time: torch.Tensor = self._create_time_axis(even=even, half_size=half_size, kernel_size=kernel_size)
        if cutoff == 0:
            filtered: torch.Tensor = torch.zeros_like(time)
        else:
            filtered: torch.Tensor = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
            filtered /= filtered.sum()
        return filtered.view(1, 1, kernel_size)

    def _create_window(self, kernel_size: int, half_size: int, half_width: float) -> torch.Tensor:
        # Builds the Kaiser taper. The attenuation the window must deliver is
        # derived from the requested transition width by the standard Kaiser
        # design rule, and beta follows from that attenuation.
        #
        # Args:
        #     kernel_size: Tap count of the window.
        #     half_size: Half the tap count, the design rule's length term.
        #     half_width: Normalized half-width of the transition band.
        delta_frequency: float = 4 * half_width
        attenuation: float = 2.285 * (half_size - 1) * math.pi * delta_frequency + 7.95
        beta: float = self._compute_kaiser_beta(attenuation)
        return torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    def _compute_kaiser_beta(self, attenuation: float) -> float:
        # Computes the Kaiser-window beta parameter from the requested attenuation.
        # The three branches are the standard piecewise formula: above 50 dB
        # beta grows linearly, between 21 and 50 dB it follows the fractional
        # correction, and below 21 dB the rectangular window already meets the
        # requirement so beta is zero.
        if attenuation > 50.0:
            return 0.1102 * (attenuation - 8.7)
        if attenuation >= 21.0:
            return 0.5842 * (attenuation - 21.0) ** 0.4 + 0.07886 * (attenuation - 21.0)
        return 0.0

    def _create_time_axis(self, even: bool, half_size: int, kernel_size: int) -> torch.Tensor:
        # Returns the sample positions the sinc is evaluated at, centered on
        # the kernel. An even tap count has no center sample, so the axis is
        # offset by half a sample to keep the kernel symmetric about zero.
        if even:
            return torch.arange(-half_size, half_size) + 0.5
        return torch.arange(kernel_size) - half_size


class _LowPassFilter1d(nn.Module):
    # Band-limits and optionally decimates a signal. Convolving with the
    # unit-sum kernel removes the content above the cutoff, and the stride then
    # discards samples that no longer carry information, which is what makes
    # the decimation alias-free.
    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        kernel_size: int = 12
    ) -> None:
        # Designs the kernel and precomputes the asymmetric padding that keeps
        # the filtered output aligned with its input. The split is uneven for
        # an even tap count, because such a kernel's center falls between two
        # samples.
        #
        # Args:
        #     cutoff: Normalized cutoff, at most one half, the Nyquist limit
        #         of the input rate. Default: ``0.5``.
        #     half_width: Normalized transition half-width. Default: ``0.6``.
        #     stride: Decimation factor applied after filtering.
        #         Default: ``1``.
        #     kernel_size: Tap count of the low-pass kernel. Default: ``12``.
        #
        # Raises:
        #     ValueError: If the cutoff is negative or exceeds one half, since
        #         neither describes a realizable low-pass band.
        super().__init__()
        if cutoff < 0.0:
            raise ValueError("cutoff must be non-negative")
        if cutoff > 0.5:
            raise ValueError("cutoff must not exceed 0.5")
        even: bool = kernel_size % 2 == 0
        self._pad_left: int = kernel_size // 2 - int(even)
        self._pad_right: int = kernel_size // 2
        self._stride: int = stride
        filter_factory: _KaiserSincFilterFactory = _KaiserSincFilterFactory()
        self.register_buffer("filter", filter_factory.create(cutoff, half_width, kernel_size))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Filters every channel independently. The kernel is expanded across
        # channels and applied with one group per channel, so a single designed
        # filter serves an arbitrary width without being stored per channel.
        # Edges are extended by replication rather than zeros, because a zero
        # tail is itself a discontinuity the filter would ring on.
        channel_count: int = x.shape[1]
        padded: torch.Tensor = functional.pad(x, (self._pad_left, self._pad_right), mode="replicate")
        return functional.conv1d(
            padded,
            self.filter.expand(channel_count, -1, -1),
            stride=self._stride,
            groups=channel_count
        )


class _UpSample1d(nn.Module):
    # Interpolates a signal to a higher rate. The transposed convolution
    # inserts zeros between samples and convolves with the designed kernel in
    # one operation, so the images the zero-stuffing creates are removed by the
    # same pass that performs the interpolation.
    def __init__(self, ratio: int = 2, kernel_size: int = 12) -> None:
        # Designs a kernel whose cutoff and transition are both scaled down by
        # the ratio, which is the band the output rate must be limited to, and
        # precomputes the padding and the trim that keep the result aligned
        # with the input.
        #
        # Args:
        #     ratio: Interpolation factor and the transposed convolution's
        #         stride. Default: ``2``.
        #     kernel_size: Tap count of the interpolation kernel.
        #         Default: ``12``.
        super().__init__()
        self._ratio: int = ratio
        self._stride: int = ratio
        pad: int = kernel_size // ratio - 1
        self._pad: int = pad
        self._pad_left: int = pad * self._stride + (kernel_size - self._stride) // 2
        self._pad_right: int = pad * self._stride + (kernel_size - self._stride + 1) // 2
        filter_factory: _KaiserSincFilterFactory = _KaiserSincFilterFactory()
        self.register_buffer(
            "filter",
            filter_factory.create(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=kernel_size)
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Interpolates every channel independently and trims back to the
        # aligned span. The ratio multiplier compensates the amplitude loss of
        # zero-stuffing: a unit-sum kernel spread over the inserted zeros would
        # otherwise scale the signal down by exactly the ratio.
        channel_count: int = x.shape[1]
        padded: torch.Tensor = functional.pad(x, (self._pad, self._pad), mode="replicate")
        upsampled: torch.Tensor = self._ratio * functional.conv_transpose1d(
            padded,
            self.filter.expand(channel_count, -1, -1),
            stride=self._stride,
            groups=channel_count
        )
        return upsampled[..., self._pad_left:-self._pad_right]


class _DownSample1d(nn.Module):
    # Decimates a signal back to its original rate. It is exactly a low-pass
    # whose stride equals the ratio, so band-limiting and discarding samples
    # happen in the same operation and no aliased content is ever produced.
    def __init__(self, ratio: int = 2, kernel_size: int = 12) -> None:
        # Builds the decimating low-pass at the same cutoff and transition the
        # matching upsampler interpolates with, so the pair is symmetric.
        #
        # Args:
        #     ratio: Decimation factor. Default: ``2``.
        #     kernel_size: Tap count of the low-pass kernel. Default: ``12``.
        super().__init__()
        self.lowpass: _LowPassFilter1d = _LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=kernel_size
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Delegates to the low-pass, whose stride performs the decimation.
        return self.lowpass(x)


class _SnakeBeta(nn.Module):
    # Periodic activation ``x + (1 / beta) * sin(alpha * x) ** 2`` with alpha
    # and beta learned per channel. The sine term gives the network an explicit
    # mechanism for generating periodic structure, which a monotone activation
    # can only approximate; alpha sets the frequency of that structure and beta
    # its amplitude, and decoupling them is what distinguishes this variant.
    def __init__(self, channels: int, alpha_logscale: bool) -> None:
        # Allocates the per-channel parameter pair. Under log scale the stored
        # values are exponentiated at use, so they are initialized at zero to
        # start from a factor of one and can never become negative during
        # training; under linear scale they are initialized at one directly.
        #
        # Args:
        #     channels: Channel width the activation is applied to; each
        #         channel learns its own alpha and beta.
        #     alpha_logscale: Whether alpha and beta are stored as logarithms.
        super().__init__()
        self.alpha_logscale: bool = alpha_logscale
        if alpha_logscale:
            self.alpha: nn.Parameter = nn.Parameter(torch.zeros(channels))
            self.beta: nn.Parameter = nn.Parameter(torch.zeros(channels))
        else:
            self.alpha: nn.Parameter = nn.Parameter(torch.ones(channels))
            self.beta: nn.Parameter = nn.Parameter(torch.ones(channels))
        self._epsilon: float = 1e-9

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Applies the activation to a ``[batch, channels, frames]`` input. The
        # unsqueezes broadcast the per-channel parameters across batch and
        # time, and the epsilon guards the reciprocal against a beta that has
        # been driven to zero.
        alpha: torch.Tensor = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta: torch.Tensor = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha: torch.Tensor = torch.exp(alpha)
            beta: torch.Tensor = torch.exp(beta)
        return x + (1.0 / (beta + self._epsilon)) * torch.pow(torch.sin(x * alpha), 2)


class _Snake(nn.Module):
    # Periodic activation ``x + (1 / alpha) * sin(alpha * x) ** 2``, the
    # single-parameter variant. One learned alpha sets both the frequency and,
    # through its reciprocal, the amplitude of the periodic term, so this
    # variant owns half the parameters of snakebeta and no beta at all.
    def __init__(self, channels: int, alpha_logscale: bool) -> None:
        # Allocates the single per-channel parameter, at zero under log scale
        # so the exponential starts it at one, and at one otherwise.
        #
        # Args:
        #     channels: Channel width the activation is applied to.
        #     alpha_logscale: Whether alpha is stored as a logarithm.
        super().__init__()
        self.alpha_logscale: bool = alpha_logscale
        self.alpha: nn.Parameter = nn.Parameter(torch.zeros(channels) if alpha_logscale else torch.ones(channels))
        self._epsilon: float = 1e-9

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Applies the activation with alpha broadcast across batch and time.
        alpha: torch.Tensor = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha: torch.Tensor = torch.exp(alpha)
        return x + (1.0 / (alpha + self._epsilon)) * torch.pow(torch.sin(x * alpha), 2)


class _Activation1d(nn.Module):
    # The anti-aliased activation, and the reason this architecture is more
    # than a time-domain adversarial upsampler carrying a substituted
    # nonlinearity. A periodic activation applied at the signal's own rate
    # generates harmonics above Nyquist, which fold back into the audible band
    # as aliasing and are heard as a persistent metallic artifact. Applying the
    # same activation at twice the rate gives
    # those harmonics somewhere to go, and the decimating low-pass then removes
    # them before the signal returns to its original rate. The sandwich is
    # therefore always upsample, activate, downsample, and the pair is
    # length-preserving so callers can treat it as a drop-in activation.
    def __init__(self, activation: nn.Module) -> None:
        # Wraps the activation between a matched interpolation and decimation
        # pair at ratio two, the oversampling factor the published recipes use.
        #
        # Args:
        #     activation: The periodic activation to protect; it is stored
        #         under the author's attribute name, which the strict load
        #         matches on.
        super().__init__()
        self.act: nn.Module = activation
        self.upsample: _UpSample1d = _UpSample1d(ratio=2, kernel_size=12)
        self.downsample: _DownSample1d = _DownSample1d(ratio=2, kernel_size=12)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Runs the oversample, activate, decimate sandwich and returns a tensor
        # of the input's shape.
        return self.downsample(self.act(self.upsample(x)))


class _BigvganActivationFactory:
    # Resolves the configured activation name to its module, keeping the
    # closed vocabulary in one place so the residual blocks and the generator
    # head all construct the same variant.
    def create(
        self,
        activation: Literal["snake", "snakebeta"],
        channels: int,
        snake_logscale: bool
    ) -> nn.Module:
        # Builds the configured periodic activation (snake or snakebeta).
        #
        # Args:
        #     activation: Closed literal selecting the variant.
        #     channels: Channel width the activation is applied to.
        #     snake_logscale: Whether the variant stores its parameters as
        #         logarithms.
        match activation:
            case "snake":
                return _Snake(channels, alpha_logscale=snake_logscale)
            case "snakebeta":
                return _SnakeBeta(channels, alpha_logscale=snake_logscale)


class _AmpBlock1(nn.Module):
    # The wide residual block: for each configured dilation it runs a dilated
    # convolution followed by an undilated refinement, each preceded by its own
    # anti-aliased activation, and adds the result back to its input. The
    # dilated convolution widens the receptive field without extra parameters
    # while the refinement re-mixes locally, so one block sees several time
    # scales at once. Every activation is separately parametrized, which is
    # what makes the periodicity learnable per position in the stack rather
    # than shared across it.
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: tuple[int, ...],
        activation: Literal["snake", "snakebeta"],
        snake_logscale: bool
    ) -> None:
        # Builds one dilated convolution per dilation rate, one undilated
        # refinement each, and one activation per convolution across both
        # lists. Each convolution is normally initialized before weight
        # normalization wraps it, and padded so the block is shape-preserving.
        #
        # Args:
        #     channels: Width the block consumes and returns.
        #     kernel_size: Time extent of every convolution in the block.
        #     dilation: Dilation rates of the first convolution list; its
        #         length sets how many convolution pairs the block holds.
        #     activation: Periodic activation variant used throughout.
        #     snake_logscale: Whether that variant stores its parameters as
        #         logarithms.
        super().__init__()
        padding: _ConvolutionPadding = _ConvolutionPadding()
        activation_factory: _BigvganActivationFactory = _BigvganActivationFactory()
        initializer: _NormalWeightInitializer = _NormalWeightInitializer()
        self.convs1: nn.ModuleList = nn.ModuleList(
            [
                weight_norm(
                    initializer.apply(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            stride=1,
                            dilation=dilation_rate,
                            padding=padding.compute(kernel_size, dilation_rate)
                        )
                    )
                )
                for dilation_rate in dilation
            ]
        )
        self.convs2: nn.ModuleList = nn.ModuleList(
            [
                weight_norm(
                    initializer.apply(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            stride=1,
                            dilation=1,
                            padding=padding.compute(kernel_size, 1)
                        )
                    )
                )
                for _ in dilation
            ]
        )
        layer_count: int = len(self.convs1) + len(self.convs2)
        self.activations: nn.ModuleList = nn.ModuleList(
            [_Activation1d(activation_factory.create(activation, channels, snake_logscale)) for _ in range(layer_count)]
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Runs every convolution pair as a residual refinement. The activation
        # list is a single flat sequence covering both convolution lists, so it
        # is strided in two to recover the activation belonging to each list;
        # the strict zip then guarantees the four sequences stay in step and a
        # mismatched construction fails here rather than silently pairing the
        # wrong activation with a convolution.
        first_activations: nn.ModuleList = self.activations[::2]
        second_activations: nn.ModuleList = self.activations[1::2]
        for first_conv, second_conv, first_activation, second_activation in zip(
            self.convs1,
            self.convs2,
            first_activations,
            second_activations,
            strict=True
        ):
            transformed: torch.Tensor = first_activation(x)
            transformed: torch.Tensor = first_conv(transformed)
            transformed: torch.Tensor = second_activation(transformed)
            transformed: torch.Tensor = second_conv(transformed)
            x: torch.Tensor = transformed + x
        return x


class _AmpBlock2(nn.Module):
    # The narrow residual block: one dilated convolution per rate with no
    # refinement stage, so it carries half the convolutions and half the
    # activations of the wide variant at the same dilation coverage. The
    # published base recipe selects the wide variant; this one exists for the
    # lighter configurations of the family.
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: tuple[int, ...],
        activation: Literal["snake", "snakebeta"],
        snake_logscale: bool
    ) -> None:
        # Builds one dilated convolution and one anti-aliased activation per
        # dilation rate, under the same initialize-then-wrap ordering and the
        # same length-preserving padding as the wide variant.
        #
        # Args:
        #     channels: Width the block consumes and returns.
        #     kernel_size: Time extent of every convolution in the block.
        #     dilation: Dilation rates, one convolution each.
        #     activation: Periodic activation variant used throughout.
        #     snake_logscale: Whether that variant stores its parameters as
        #         logarithms.
        super().__init__()
        padding: _ConvolutionPadding = _ConvolutionPadding()
        activation_factory: _BigvganActivationFactory = _BigvganActivationFactory()
        initializer: _NormalWeightInitializer = _NormalWeightInitializer()
        self.convs: nn.ModuleList = nn.ModuleList(
            [
                weight_norm(
                    initializer.apply(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            stride=1,
                            dilation=dilation_rate,
                            padding=padding.compute(kernel_size, dilation_rate)
                        )
                    )
                )
                for dilation_rate in dilation
            ]
        )
        self.activations: nn.ModuleList = nn.ModuleList(
            [_Activation1d(activation_factory.create(activation, channels, snake_logscale)) for _ in self.convs]
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Runs each activate-then-convolve stage as a residual refinement. The
        # two lists are built one-for-one, and the strict zip enforces that.
        for convolution, activation in zip(self.convs, self.activations, strict=True):
            transformed: torch.Tensor = activation(x)
            transformed: torch.Tensor = convolution(transformed)
            x: torch.Tensor = transformed + x
        return x


class BigvganNetwork(nn.Module):
    # The BigVGAN generator. A conditioning convolution lifts the mel to the
    # initial width, then each stage transposed-convolves the time axis up by
    # its rate while halving the channel width, and a bank of residual blocks
    # refines the result. The blocks at one stage differ only in kernel size,
    # and their outputs are averaged rather than chained, so the stage observes
    # several receptive fields in parallel and no single field dominates. A
    # final anti-aliased activation and projection reduce the last width to the
    # single waveform channel.
    #
    # What separates this generator from its HiFi-GAN ancestor is entirely
    # inside the blocks: every activation is a learnable periodic function
    # wrapped in the oversample-activate-decimate sandwich, so periodicity can
    # be generated explicitly while the aliasing such a nonlinearity would
    # otherwise fold back into the audible band is filtered away.
    #
    # In the report's generation-mechanism taxonomy this places BigVGAN-base
    # among the time-domain adversarial upsamplers, conditioned by the
    # published base recipe on 100 normalized mel bands at 24 kHz.
    #
    # Integration: the module wrapper (vocode.models.bigvgan.bigvgan.Bigvgan)
    # owns an instance of this class under the attribute the published NVIDIA
    # release is loaded onto by vocode.models.bigvgan.weights.BigvganWeights.
    # The attribute names declared here are consequently part of the release
    # contract, up to the weight-normalization key renaming that adapter
    # performs.
    def __init__(
        self,
        num_mels: int,
        upsample_initial_channel: int,
        upsample_rates: tuple[int, ...],
        upsample_kernel_sizes: tuple[int, ...],
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilation_sizes: tuple[tuple[int, ...], ...],
        resblock: Literal["1", "2"],
        activation: Literal["snake", "snakebeta"],
        snake_logscale: bool,
        use_bias_at_final: bool = True,
        use_tanh_at_final: bool = True
    ) -> None:
        # Builds the conditioning convolution, the upsampling stages, the
        # residual bank, and the output head.
        #
        # Two counts govern the forward pass and are cached here: the number of
        # upsampling stages and the number of residual blocks per stage. The
        # residual bank is one flat list indexed arithmetically at run time
        # rather than a nested structure, which is the author's layout and what
        # the published state dictionary is keyed by. Every convolution is
        # normally initialized before weight normalization wraps it, except the
        # conditioning convolution, which is deliberately left at the torch
        # default.
        #
        # Args:
        #     num_mels: Band count of the conditioning mel.
        #     upsample_initial_channel: Width the conditioning convolution
        #         lifts to; each stage halves it.
        #     upsample_rates: Time expansion of each stage; their product is
        #         the samples per conditioning frame.
        #     upsample_kernel_sizes: Kernel of each stage's transposed
        #         convolution, paired with the rates by position.
        #     resblock_kernel_sizes: Kernel of each residual block within a
        #         stage; its length is the blocks per stage.
        #     resblock_dilation_sizes: Dilation tuple for each of those
        #         blocks, paired with the kernel sizes by position.
        #     resblock: Closed literal selecting the wide or narrow residual
        #         variant.
        #     activation: Closed literal selecting the periodic activation.
        #     snake_logscale: Whether the activation stores its parameters as
        #         logarithms.
        #     use_bias_at_final: Whether the output projection carries a bias.
        #         Default: ``True``.
        #     use_tanh_at_final: Selects the bounding of the output between a
        #         hyperbolic tangent and a hard clamp. Default: ``True``.
        super().__init__()
        self._num_kernels: int = len(resblock_kernel_sizes)
        self._num_upsamples: int = len(upsample_rates)
        self._use_tanh_at_final: bool = use_tanh_at_final
        self.conv_pre: nn.Conv1d = weight_norm(
            nn.Conv1d(num_mels, upsample_initial_channel, kernel_size=7, stride=1, padding=3)
        )
        resblock_class: type[_AmpBlock1] | type[_AmpBlock2]
        match resblock:
            case "1":
                resblock_class: type[_AmpBlock1] | type[_AmpBlock2] = _AmpBlock1
            case "2":
                resblock_class: type[_AmpBlock1] | type[_AmpBlock2] = _AmpBlock2
        initializer: _NormalWeightInitializer = _NormalWeightInitializer()
        self.ups: nn.ModuleList = nn.ModuleList()
        # Each stage is wrapped in its own single-element module list, which
        # produces the author's two-level ``ups.<stage>.0`` key layout that the
        # strict load matches on.
        for index, (upsample_rate, upsample_kernel_size) in enumerate(zip(
            upsample_rates,
            upsample_kernel_sizes,
            strict=True
        )):
            input_channels: int = upsample_initial_channel // (2 ** index)
            output_channels: int = upsample_initial_channel // (2 ** (index + 1))
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            initializer.apply(
                                nn.ConvTranspose1d(
                                    input_channels,
                                    output_channels,
                                    upsample_kernel_size,
                                    upsample_rate,
                                    padding=(upsample_kernel_size - upsample_rate) // 2
                                )
                            )
                        )
                    ]
                )
            )
        # The residual bank is flat and stage-major: the blocks of stage i
        # occupy the slice starting at i * num_kernels, which is the arithmetic
        # the forward pass indexes with. Each block is built at the width its
        # own stage emits.
        self.resblocks: nn.ModuleList = nn.ModuleList()
        current_channels: int = upsample_initial_channel
        activation_factory: _BigvganActivationFactory = _BigvganActivationFactory()
        for upsample_index in range(len(self.ups)):
            current_channels: int = upsample_initial_channel // (2 ** (upsample_index + 1))
            for kernel_size, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(
                    resblock_class(
                        current_channels,
                        kernel_size,
                        dilation,
                        activation=activation,
                        snake_logscale=snake_logscale
                    )
                )
        # The head operates at the final stage's width, which the loop leaves
        # in current_channels.
        self.activation_post: _Activation1d = _Activation1d(
            activation_factory.create(activation, current_channels, snake_logscale)
        )
        self.conv_post: nn.Conv1d = weight_norm(
            initializer.apply(
                nn.Conv1d(current_channels, 1, kernel_size=7, stride=1, padding=3, bias=use_bias_at_final)
            )
        )

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Synthesizes a waveform from the conditioning mel.
        #
        # Each stage upsamples and then averages its residual bank: every block
        # of the stage is applied to the same input and their outputs are
        # summed and divided by the block count, so the stage's receptive
        # fields are fused rather than composed. The head then activates,
        # projects to one channel, and bounds the result, either smoothly
        # through a hyperbolic tangent or by a hard clamp.
        #
        # Args:
        #     mel: Conditioning mel shaped ``[batch, bands, frames]``; the
        #         batch axis must be explicit.
        #
        # Raises:
        #     ValueError: If the mel does not carry exactly three dimensions.
        #     RuntimeError: If a stage's residual bank is empty, which means
        #         the network was configured with no residual kernel sizes and
        #         the averaging has nothing to divide.
        #
        # Returns:
        #     The waveform shaped ``[batch, 1, frames * product(rates)]``, with
        #     every sample inside the unit interval.
        if mel.ndim != 3:
            raise ValueError(f"Expected mel shape [batch, channels, frames], got {tuple(mel.shape)}")
        x: torch.Tensor = self.conv_pre(mel)
        for upsample_index in range(self._num_upsamples):
            for upsample_layer in self.ups[upsample_index]:
                x: torch.Tensor = upsample_layer(x)
            residual_sum: torch.Tensor | None = None
            for kernel_index in range(self._num_kernels):
                residual: torch.Tensor = self.resblocks[
                    upsample_index * self._num_kernels + kernel_index
                ](x)
                residual_sum: torch.Tensor | None = residual if residual_sum is None else residual_sum + residual
            if residual_sum is None:
                raise RuntimeError("BigVGAN resblock configuration produced no residual blocks")
            x: torch.Tensor = residual_sum / self._num_kernels
        x: torch.Tensor = self.activation_post(x)
        x: torch.Tensor = self.conv_post(x)
        if self._use_tanh_at_final:
            return torch.tanh(x)
        return torch.clamp(x, min=-1.0, max=1.0)
