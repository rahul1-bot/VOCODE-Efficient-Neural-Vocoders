# This module:
# 1. Implements the RFWave backbone network: a ConvNeXt-style stack with
#    Fourier time-embedding and band-index conditioning, predicting
#    per-band complex spectral velocities for the rectified-flow objective
#
# Conditioning structure:
# - The backbone is conditioned on three quantities at once and each enters
#   by a different route. The mel is concatenated with the state at the
#   input, because it varies along the frame axis and must align with it.
#   The flow time is embedded into a vector added to the state before every
#   block, because it is one scalar per row that every position needs. The
#   band index selects per-band scale and shift parameters inside the
#   normalizations, because it is discrete and small
# - The band route is what allows one shared backbone to serve every band:
#   the normalizations are the only band-dependent parameters, so widening
#   the band count costs two embedding rows rather than a whole network
#
# Design decisions:
# - All bands share one backbone distinguished by the band embedding, so
#   parameters scale with model width rather than band count
# - The time embedding enters in the network's compute dtype so
#   half-precision storage variants execute without dtype conflicts
# - The noisy state is presented both directly and through high-frequency
#   Fourier features, so the network can resolve fine amplitude differences
#   that a linear input alone would leave below its effective resolution
# - The time embedding is re-injected before every block rather than only at
#   the input, so deep blocks retain the path position that an input-only
#   injection would let the residual stream dilute
#
# Author: Rahul Sawhney

import math
from typing import ClassVar, cast, override

import numpy
import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from scipy import signal
from torch import nn

__all__: list[str] = ["RfwaveBackbone", "RfwaveNetworkConfig", "RfwavePqmfEqualizer", "RfwaveSpectralTransform"]


class RfwaveNetworkConfig(BaseModel):
    # Frozen RFWave backbone topology and band settings. Field defaults follow
    # the reference vocoder configuration.
    #
    # Fields:
    #     mel_channels: Mel band count of the conditioning input.
    #         Default: ``100``.
    #     output_channels: Width the backbone emits per row, which must equal
    #         twice the slab width, since one row carries a band's real and
    #         imaginary slabs. It is therefore determined by the transform
    #         size, band count, and overlaps rather than free.
    #         Default: ``160``.
    #     hidden_dimension: Residual stream width. Default: ``512``.
    #     intermediate_dimension: Expanded width inside each block's
    #         bottleneck. Default: ``1536``.
    #     layer_count: Number of residual blocks. Default: ``8``.
    #     band_count: Number of frequency subbands, and the number of rows in
    #         each band-conditioned embedding table. Default: ``8``.
    #     n_fft: Transform size of the spectral representation.
    #         Default: ``1024``.
    #     hop_length: Hop of that transform. Default: ``256``.
    #     left_overlap: Bins of context each band takes below its own range.
    #         Default: ``8``.
    #     right_overlap: Bins of context each band takes above its own range.
    #         Default: ``8``.
    #     fourier_start_exponent: First base-two exponent of the state's
    #         Fourier features. Default: ``6``.
    #     fourier_stop_exponent: Exclusive upper exponent, so the reference
    #         setting yields two frequencies. Default: ``8``.
    #     time_embedding_scale: Multiplier applied to the flow time before
    #         embedding, which spreads the unit interval across a range the
    #         sinusoidal embedding resolves usefully. Default: ``1000.0``.
    #     pqmf_taps: Filter length of the subband equalizer.
    #         Default: ``124``.
    #     pqmf_cutoff_ratio: Cutoff of the equalizer prototype filter, as a
    #         fraction of the band edge. Default: ``0.071``.
    #     pqmf_kaiser_beta: Window shape parameter of that filter, trading
    #         transition sharpness against stopband suppression.
    #         Default: ``9.0``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    mel_channels: PositiveInt = 100
    output_channels: PositiveInt = 160
    hidden_dimension: PositiveInt = 512
    intermediate_dimension: PositiveInt = 1536
    layer_count: PositiveInt = 8
    band_count: PositiveInt = 8
    n_fft: PositiveInt = 1024
    hop_length: PositiveInt = 256
    left_overlap: PositiveInt = 8
    right_overlap: PositiveInt = 8
    fourier_start_exponent: PositiveInt = 6
    fourier_stop_exponent: PositiveInt = 8
    time_embedding_scale: PositiveFloat = 1000.0
    pqmf_taps: PositiveInt = 124
    pqmf_cutoff_ratio: PositiveFloat = 0.071
    pqmf_kaiser_beta: PositiveFloat = 9.0


class _SinusoidalTimeEmbedding(nn.Module):
    # Embeds the scalar flow time as a bank of sinusoids at geometrically
    # spaced frequencies. A raw scalar would give the network almost nothing
    # to condition on; spreading it across many frequencies makes nearby times
    # similar and distant times distinguishable, at every resolution the
    # frequency range covers. The module holds no parameters, so its behavior
    # is fixed by the dimension and the caller's scale.
    def __init__(self, dimension: int) -> None:
        # Records the embedding width, which must be even because the output
        # is a sine half concatenated with a cosine half.
        super().__init__()
        self._dimension: int = dimension

    @override
    def forward(self, time_values: torch.Tensor, scale: float) -> torch.Tensor:
        # Builds the embedding for one time per row.
        #
        # Args:
        #     time_values: Flow times of shape ``[rows]``.
        #     scale: Multiplier applied before embedding, which stretches the
        #         unit interval the flow time lives in across a range the
        #         frequency bank actually resolves.
        #
        # Returns:
        #     The embedding of shape ``[rows, dimension]``, the sine and
        #     cosine halves concatenated.
        half_dimension: int = self._dimension // 2
        exponent: torch.Tensor = torch.exp(
            torch.arange(
                half_dimension,
                device=time_values.device,
                dtype=time_values.dtype
            )
            * (-math.log(10000.0) / (half_dimension - 1))
        )
        arguments: torch.Tensor = scale * time_values.unsqueeze(1) * exponent.unsqueeze(0)
        return torch.cat((arguments.sin(), arguments.cos()), dim=-1)


class _BaseTwoFourierFeatures(nn.Module):
    # Expands the noisy state into high-frequency sinusoidal features at
    # power-of-two frequencies.
    #
    # The purpose is resolution rather than periodicity. Spectral coefficients
    # differ by amounts far smaller than their range, and a network reading
    # them linearly resolves such differences poorly. Passing them through
    # rapidly oscillating functions turns a small change in the input into a
    # large change in the feature, so fine amplitude structure becomes visible
    # to the first convolution. The module is parameter-free; the exponent
    # range is the only control.
    def __init__(self, start_exponent: int, stop_exponent: int) -> None:
        # Records the half-open exponent range, so the number of frequencies
        # is the difference between the two bounds.
        super().__init__()
        self._start_exponent: int = start_exponent
        self._stop_exponent: int = stop_exponent

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Produces a sine and a cosine feature per input channel per
        # frequency, interleaved so each channel's features stay adjacent.
        #
        # Returns:
        #     Features whose channel count is the input's times twice the
        #     number of frequencies, with the frame axis unchanged.
        frequency_exponents: list[int] = list(range(self._start_exponent, self._stop_exponent))
        angular: torch.Tensor = (
            (2.0 ** torch.tensor(frequency_exponents, dtype=inputs.dtype, device=inputs.device))
            * 2.0 * torch.pi
        )
        angular: torch.Tensor = torch.tile(angular[None, :, None], (1, inputs.shape[1], 1))
        repeated: torch.Tensor = torch.repeat_interleave(inputs, len(frequency_exponents), dim=1)
        modulated: torch.Tensor = angular * repeated
        stacked: torch.Tensor = torch.stack([torch.sin(modulated), torch.cos(modulated)], dim=2)
        return stacked.reshape(stacked.size(0), -1, stacked.size(3))


class _AdaptiveLayerNorm(nn.Module):
    # Layer normalization whose scale and shift are looked up by band index
    # rather than being single shared parameters. This is the mechanism that
    # lets one backbone serve every band: the shared weights learn what is
    # common across bands, while these per-band terms absorb the differences
    # in scale and offset that distinguish a low-frequency band from a high
    # one. The cost is two embedding rows per band rather than a network per
    # band.
    def __init__(self, embedding_count: int, dimension: int) -> None:
        # Builds the two lookup tables at neutral values, so every band starts
        # from plain normalization and any per-band adaptation is learned
        # rather than imposed at initialization.
        super().__init__()
        self._dimension: int = dimension
        self._scale: nn.Embedding = nn.Embedding(embedding_count, dimension)
        self._shift: nn.Embedding = nn.Embedding(embedding_count, dimension)
        nn.init.ones_(self._scale.weight)
        nn.init.zeros_(self._shift.weight)

    @override
    def forward(self, x: torch.Tensor, band_index: torch.Tensor) -> torch.Tensor:
        # Normalizes over the feature axis, then applies the scale and shift
        # belonging to each row's band. The looked-up terms gain a frame axis
        # by broadcasting, so one band's calibration applies uniformly across
        # its frames.
        #
        # Args:
        #     x: Features of shape ``[rows, frames, dimension]``.
        #     band_index: Band identifier per row.
        scale: torch.Tensor = self._scale(band_index)
        shift: torch.Tensor = self._shift(band_index)
        x: torch.Tensor = nn.functional.layer_norm(x, (self._dimension,), eps=1e-6)
        return x * scale.unsqueeze(1) + shift.unsqueeze(1)


class _GlobalResponseNormalization(nn.Module):
    # Rescales each channel by how its energy compares with the average across
    # channels, amplifying unusually active channels and attenuating quiet
    # ones. This counteracts the tendency of wide layers to collapse onto
    # redundant responses. Both parameters start at zero, so the block begins
    # as an exact identity and any deviation is learned.
    def __init__(self, dimension: int) -> None:
        # Builds the per-channel scale and shift at zero, making the block
        # initially a pure residual pass-through.
        super().__init__()
        self._gamma: nn.Parameter = nn.Parameter(torch.zeros(1, 1, dimension))
        self._beta: nn.Parameter = nn.Parameter(torch.zeros(1, 1, dimension))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Measures each channel's energy over the frame axis, divides by the
        # mean energy across channels, and applies the result as a modulation
        # on top of an identity path. The division is floored, so an all-quiet
        # layer returns its input rather than producing non-finite values.
        global_norm: torch.Tensor = torch.norm(x, p=2, dim=1, keepdim=True)
        normalized: torch.Tensor = global_norm / (global_norm.mean(dim=-1, keepdim=True) + 1e-6)
        return self._gamma * (x * normalized) + self._beta + x


class _ConvNeXtV2AdaptiveBlock(nn.Module):
    # One residual block of the backbone, in inverted-bottleneck form: a
    # depthwise convolution gathers temporal context per channel, then a
    # pointwise pair expands and contracts the width around a nonlinearity,
    # with the response normalization placed inside the expansion where
    # channel redundancy would otherwise accumulate. Normalization is
    # band-conditioned, which is what specializes the shared block to each
    # band.
    def __init__(self, dimension: int, intermediate_dimension: int, band_count: int) -> None:
        # Builds the depthwise convolution, the band-conditioned
        # normalization, and the expanding pointwise pair. The convolution is
        # depthwise and length-preserving, so it mixes across time without
        # mixing channels and without changing the frame count.
        super().__init__()
        self._depthwise_convolution: nn.Conv1d = nn.Conv1d(
            dimension,
            dimension,
            kernel_size=7,
            padding=3,
            groups=dimension
        )
        self._normalization: _AdaptiveLayerNorm = _AdaptiveLayerNorm(band_count, dimension)
        self._pointwise_first: nn.Linear = nn.Linear(dimension, intermediate_dimension)
        self._activation: nn.GELU = nn.GELU()
        self._response_normalization: _GlobalResponseNormalization = _GlobalResponseNormalization(intermediate_dimension)
        self._pointwise_second: nn.Linear = nn.Linear(intermediate_dimension, dimension)

    @override
    def forward(self, x: torch.Tensor, band_index: torch.Tensor) -> torch.Tensor:
        # Applies the block and adds its result to the input. The tensor is
        # transposed once into the feature-last layout the normalization and
        # linear layers require, and transposed back only at the addition, so
        # the block performs exactly two transpositions rather than one per
        # stage.
        residual: torch.Tensor = x
        x: torch.Tensor = self._depthwise_convolution(x)
        x: torch.Tensor = x.transpose(1, 2)
        x: torch.Tensor = self._normalization(x, band_index)
        x: torch.Tensor = self._pointwise_first(x)
        x: torch.Tensor = self._activation(x)
        x: torch.Tensor = self._response_normalization(x)
        x: torch.Tensor = self._pointwise_second(x)
        return residual + x.transpose(1, 2)


class RfwaveBackbone(nn.Module):
    # The velocity predictor of the rectified flow. It reads a state on the
    # path, the time of that state, the conditioning mel, and a band index,
    # and emits the velocity to follow.
    #
    # One instance serves every band. The only band-dependent parameters are
    # the scale and shift tables inside the normalizations, so the band count
    # affects parameter count negligibly while the shared weights see every
    # band's data. That sharing is what makes a multi-band decomposition
    # affordable, and it is the reason the band index must be supplied at
    # every call rather than baked into a per-band instance.
    def __init__(self, configuration: RfwaveNetworkConfig) -> None:
        # Builds the input embedding, the residual stack, the time-embedding
        # projection, and the velocity head.
        #
        # The embedding's input width is the sum of three conditioning
        # sources: the state itself, the mel, and the state's Fourier
        # expansion. That last term dominates, since the expansion produces
        # two features per channel per frequency, which is why the width is
        # computed here rather than configured.
        super().__init__()
        self._configuration: RfwaveNetworkConfig = configuration
        fourier_count: int = configuration.fourier_stop_exponent - configuration.fourier_start_exponent
        fourier_dimension: int = configuration.output_channels * 2 * fourier_count
        embedding_inputs: int = configuration.mel_channels + configuration.output_channels + fourier_dimension
        self._fourier_features: _BaseTwoFourierFeatures = _BaseTwoFourierFeatures(
            start_exponent=configuration.fourier_start_exponent,
            stop_exponent=configuration.fourier_stop_exponent
        )
        self._embedding: nn.Conv1d = nn.Conv1d(
            embedding_inputs,
            configuration.hidden_dimension,
            kernel_size=7,
            padding=3
        )
        self._input_normalization: _AdaptiveLayerNorm = _AdaptiveLayerNorm(
            configuration.band_count,
            configuration.hidden_dimension
        )
        self._blocks: nn.ModuleList = nn.ModuleList([
            _ConvNeXtV2AdaptiveBlock(
                dimension=configuration.hidden_dimension,
                intermediate_dimension=configuration.intermediate_dimension,
                band_count=configuration.band_count
            )
            for _ in range(configuration.layer_count)
        ])
        self._final_normalization: nn.LayerNorm = nn.LayerNorm(configuration.hidden_dimension, eps=1e-6)
        self._time_embedding: _SinusoidalTimeEmbedding = _SinusoidalTimeEmbedding(configuration.hidden_dimension)
        self._time_projection: nn.Sequential = nn.Sequential(
            nn.Linear(configuration.hidden_dimension, configuration.hidden_dimension * 4),
            nn.GELU(),
            nn.Linear(configuration.hidden_dimension * 4, configuration.hidden_dimension)
        )
        self._output_projection: nn.Linear = nn.Linear(
            configuration.hidden_dimension,
            configuration.output_channels
        )
        self.apply(self._initialize_weights)

    @override
    def forward(
        self,
        noisy_state: torch.Tensor,
        time_values: torch.Tensor,
        mel: torch.Tensor,
        band_index: torch.Tensor
    ) -> torch.Tensor:
        # Predicts the velocity for every row of a joint-parallel batch.
        #
        # Args:
        #     noisy_state: The state on the path, of shape
        #         ``[rows, output_channels, frames]``.
        #     time_values: The path position per row.
        #     mel: Conditioning mel per row, already expanded over bands.
        #     band_index: Band identifier per row.
        #
        # Returns:
        #     The predicted velocity, matching the state's shape so it can be
        #     added to the state directly by the integrator.
        fourier: torch.Tensor = self._fourier_features(noisy_state)
        # The three frame-aligned inputs are concatenated along channels and
        # embedded together, so the first convolution sees the state, its
        # high-resolution expansion, and the conditioning at once.
        x: torch.Tensor = self._embedding(torch.cat([noisy_state, mel, fourier], dim=1))
        # The time embedding is projected once and reused by every block, and
        # gains a frame axis by broadcasting because one time governs the
        # whole row.
        time_projection: torch.Tensor = self._time_projection(
            self._time_embedding(time_values, scale=self._configuration.time_embedding_scale)
        ).unsqueeze(2)
        x: torch.Tensor = self._input_normalization(x.transpose(1, 2), band_index).transpose(1, 2)
        for block_module in self._blocks:
            block: _ConvNeXtV2AdaptiveBlock = cast(_ConvNeXtV2AdaptiveBlock, block_module)
            # The time embedding is added before every block rather than only
            # at the input, so deep blocks still receive the path position
            # instead of relying on the residual stream to have preserved it.
            x: torch.Tensor = block(x + time_projection, band_index)
        x: torch.Tensor = self._final_normalization(x.transpose(1, 2))
        return self._output_projection(x).transpose(1, 2)

    def _initialize_weights(self, module: nn.Module) -> None:
        # Applies the reference truncated-normal initialization to every
        # convolution and linear layer, with zeroed biases. Truncation keeps
        # the initial weights away from the tails a plain normal draw would
        # produce, which matters across a stack this deep because a single
        # outlying weight early on propagates through every later block. The
        # method is applied through nn.Module.apply, so it must ignore types
        # it does not recognize.
        if isinstance(module, nn.Conv1d | nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @property
    def configuration(self) -> RfwaveNetworkConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration


class RfwaveSpectralTransform(nn.Module):
    # Short-time Fourier analysis and synthesis pair, with the real and
    # imaginary parts carried as concatenated channels rather than as a
    # complex tensor. That real-valued layout is what lets the flow operate on
    # spectra with ordinary convolutions, since the two parts are then
    # channels like any other.
    #
    # Both directions apply the reciprocal square-root scaling of the
    # transform size. The purpose is that the flow's noise endpoint, drawn as
    # unit-variance noise in the waveform domain, arrives in the spectral
    # domain at a comparable scale; without it the two path endpoints would
    # differ in magnitude by a factor that grows with the transform size.
    def __init__(self, n_fft: int, hop_length: int) -> None:
        # Records the transform geometry and registers the window as a
        # non-persistent buffer, so it moves with the module across devices
        # but never appears in the state dictionary.
        super().__init__()
        self._n_fft: int = n_fft
        self._hop_length: int = hop_length
        self.register_buffer("_window", torch.hann_window(n_fft), persistent=False)

    def stft(self, waveform: torch.Tensor) -> torch.Tensor:
        # Computes the scaled spectrum as stacked real and imaginary channels.
        # Analysis runs in float32 regardless of the caller's dtype and the
        # result is cast back, so the transform's accuracy does not vary with
        # the trainer's precision setting while the surrounding graph stays at
        # whatever precision is configured.
        window: torch.Tensor = self.get_buffer("_window")
        spectrum: torch.Tensor = torch.stft(
            waveform.float(),
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            win_length=self._n_fft,
            window=window.to(waveform.device),
            center=True,
            return_complex=True
        ) / math.sqrt(float(self._n_fft))
        return torch.cat([spectrum.real, spectrum.imag], dim=1).type_as(waveform)

    def istft(self, stacked_spectrum: torch.Tensor) -> torch.Tensor:
        # Reconstructs the waveform from stacked real and imaginary channels,
        # undoing the analysis scaling first so the two directions compose to
        # the identity on any consistent spectrum. Splitting the channel axis
        # in half is the exact inverse of the concatenation analysis performs.
        window: torch.Tensor = self.get_buffer("_window")
        scaled: torch.Tensor = stacked_spectrum.float() * math.sqrt(float(self._n_fft))
        real: torch.Tensor
        imaginary: torch.Tensor
        real, imaginary = torch.chunk(scaled, 2, dim=1)
        return torch.istft(
            torch.complex(real, imaginary),
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            win_length=self._n_fft,
            window=window.to(stacked_spectrum.device),
            center=True
        ).type_as(stacked_spectrum)


class RfwavePqmfEqualizer(nn.Module):
    # Equalizes the waveform so that its frequency subbands carry comparable
    # energy, and restores the original balance afterwards.
    #
    # The motivation is that speech energy falls steeply with frequency, often
    # by orders of magnitude across the band range. A flow whose destination
    # varied that widely between bands would be dominated by the loudest, and
    # the quiet high bands would contribute almost nothing to the objective.
    # Normalizing each band by its running statistics puts them on a common
    # scale for the flow to work in; the inverse is applied after synthesis,
    # so the output recovers the natural spectral balance.
    #
    # The statistics are running estimates held as buffers, not parameters.
    # They are updated during training and frozen afterwards, which makes them
    # part of the checkpointed state and means a checkpoint restored without
    # them would de-equalize incorrectly. The update is additionally capped
    # after a fixed number of batches, so a long run's statistics settle
    # rather than continuing to drift.
    #
    # The filterbank analyzes without downsampling, which is unusual for this
    # filter family and deliberate: the flow operates on full-rate spectra, so
    # decimating the subbands would force a resampling step on both sides.
    def __init__(
        self,
        band_count: int,
        taps: int,
        cutoff_ratio: float,
        kaiser_beta: float,
        statistics_momentum: float = 0.01,
        statistics_update_limit: int = 100000
    ) -> None:
        # Designs the prototype filter once and modulates it into the analysis
        # and synthesis banks, then registers both banks and the running
        # statistics as buffers.
        #
        # The alternating sign in the phase term is what gives this filter
        # family its cancellation property: adjacent bands are offset in
        # opposite directions, so the aliasing each introduces cancels against
        # its neighbor's on reconstruction.
        #
        # Args:
        #     band_count: Number of subbands, which must match the flow's.
        #     taps: Prototype filter length; longer filters separate bands
        #         more sharply at proportionally higher cost.
        #     cutoff_ratio: Prototype cutoff as a fraction of the band edge.
        #     kaiser_beta: Window shape parameter, trading transition
        #         sharpness against stopband suppression.
        #     statistics_momentum: Weight of each batch in the running
        #         estimates. Default: ``0.01``.
        #     statistics_update_limit: Batch count after which the estimates
        #         stop updating, so they settle rather than drifting for the
        #         whole of a long run. Default: ``100000``.
        super().__init__()
        self._band_count: int = band_count
        self._taps: int = taps
        self._statistics_momentum: float = statistics_momentum
        self._statistics_update_limit: int = statistics_update_limit
        prototype: numpy.ndarray = self._design_prototype_filter(taps, cutoff_ratio, kaiser_beta)
        analysis_bank: numpy.ndarray = numpy.zeros((band_count, taps + 1))
        synthesis_bank: numpy.ndarray = numpy.zeros((band_count, taps + 1))
        for band in range(band_count):
            modulation: numpy.ndarray = (
                (2 * band + 1)
                * (numpy.pi / (2 * band_count))
                * (numpy.arange(taps + 1) - (taps / 2))
            )
            analysis_bank[band] = 2 * prototype * numpy.cos(modulation + ((-1) ** band) * numpy.pi / 4)
            synthesis_bank[band] = 2 * prototype * numpy.cos(modulation - ((-1) ** band) * numpy.pi / 4)
        self.register_buffer("_analysis_filter", torch.from_numpy(analysis_bank).float().unsqueeze(1))
        self.register_buffer("_synthesis_filter", torch.from_numpy(synthesis_bank).float().unsqueeze(0))
        self.register_buffer("_mean_statistics", torch.zeros(band_count))
        self.register_buffer("_variance_statistics", torch.ones(band_count))
        self.register_buffer("_observed_batches", torch.zeros(()))

    def _design_prototype_filter(self, taps: int, cutoff_ratio: float, kaiser_beta: float) -> numpy.ndarray:
        # Designs the low-pass prototype every band's filter is a modulated
        # copy of: an ideal low-pass impulse response truncated to the tap
        # count and tapered by a Kaiser window. The window is what makes the
        # truncation usable, since an abruptly truncated ideal response has
        # severe stopband ripple.
        #
        # The centre sample is assigned directly because the ideal expression
        # is indeterminate there; the invalid-value guard suppresses the
        # warning that division produces before the assignment overwrites it.
        angular_cutoff: float = numpy.pi * cutoff_ratio
        with numpy.errstate(invalid="ignore"):
            impulse: numpy.ndarray = numpy.sin(angular_cutoff * (numpy.arange(taps + 1) - 0.5 * taps)) / (
                numpy.pi * (numpy.arange(taps + 1) - 0.5 * taps)
            )
        impulse[taps // 2] = cutoff_ratio
        window: numpy.ndarray = signal.windows.kaiser(taps + 1, kaiser_beta)
        return impulse * window

    def _analyze(self, waveform: torch.Tensor) -> torch.Tensor:
        # Splits the waveform into one signal per subband, each still at the
        # full sample rate. Symmetric padding by half the filter length makes
        # the convolution length-preserving, so the subbands align sample for
        # sample with the input and with each other.
        analysis_filter: torch.Tensor = self.get_buffer("_analysis_filter")
        padded: torch.Tensor = nn.functional.pad(waveform, (self._taps // 2, self._taps // 2))
        return nn.functional.conv1d(padded, analysis_filter)

    def _synthesize(self, subbands: torch.Tensor) -> torch.Tensor:
        # Recombines the subband signals into one waveform. The synthesis bank
        # is the analysis bank with its phase offset reversed, which is what
        # makes each band's aliasing cancel against its neighbor's on
        # summation; the filter is shaped so that a single convolution both
        # filters and sums.
        synthesis_filter: torch.Tensor = self.get_buffer("_synthesis_filter")
        padded: torch.Tensor = nn.functional.pad(subbands, (self._taps // 2, self._taps // 2))
        return nn.functional.conv1d(padded, synthesis_filter)

    def project(self, waveform: torch.Tensor) -> torch.Tensor:
        # Equalizes the waveform: it is split into subbands, each is
        # standardized by its running statistics, and the bands are recombined
        # into a full-rate waveform whose spectral tilt has been flattened.
        #
        # The statistics update only while training and only until the batch
        # cap is reached, so evaluation never perturbs them and a long run's
        # estimates settle. Batch statistics are detached before the update,
        # so the running estimates never carry gradient into the flow.
        #
        # Args:
        #     waveform: Waveform of shape ``[batch, samples]``. A channel axis
        #         is added for the filterbank and removed afterwards.
        #
        # Returns:
        #     The equalized waveform, in the same shape as the input.
        subbands: torch.Tensor = self._analyze(waveform.unsqueeze(1))
        mean_statistics: torch.Tensor = self.get_buffer("_mean_statistics")
        variance_statistics: torch.Tensor = self.get_buffer("_variance_statistics")
        observed_batches: torch.Tensor = self.get_buffer("_observed_batches")
        if self.training and bool(observed_batches < self._statistics_update_limit):
            batch_mean: torch.Tensor = subbands.float().mean(dim=(0, 2)).detach()
            batch_variance: torch.Tensor = subbands.float().var(dim=(0, 2)).detach()
            mean_statistics.lerp_(batch_mean, self._statistics_momentum)
            variance_statistics.lerp_(batch_variance, self._statistics_momentum)
            observed_batches += 1
        normalized: torch.Tensor = (
            (subbands - mean_statistics.view(1, -1, 1))
            / torch.sqrt(variance_statistics.view(1, -1, 1) + 1e-6)
        )
        return self._synthesize(normalized).squeeze(1)

    def restore(self, waveform: torch.Tensor) -> torch.Tensor:
        # Inverts the equalization, rescaling each subband by the statistics
        # that standardized it and recombining. This never updates the
        # statistics, because it is applied to synthesized audio whose subband
        # distribution is a property of the model rather than of the data the
        # estimates describe.
        #
        # The inversion is exact only to the extent the filterbank round trip
        # is, since analysis and synthesis reconstruct near-perfectly rather
        # than perfectly; the residual is the filter family's aliasing
        # cancellation error and is negligible at the reference filter length.
        subbands: torch.Tensor = self._analyze(waveform.unsqueeze(1))
        mean_statistics: torch.Tensor = self.get_buffer("_mean_statistics")
        variance_statistics: torch.Tensor = self.get_buffer("_variance_statistics")
        restored: torch.Tensor = (
            subbands * torch.sqrt(variance_statistics.view(1, -1, 1) + 1e-6)
            + mean_statistics.view(1, -1, 1)
        )
        return self._synthesize(restored).squeeze(1)
