# This module:
# 1. Implements the Vocos generator network: an embedding convolution over
#    the conditioning mel, a ConvNeXt block stack operating at frame rate,
#    and the inverse-STFT head predicting magnitude and phase for one-pass
#    waveform reconstruction
#
# Design decisions:
# - The network never upsamples in time: all computation happens at frame
#    rate and the ISTFT performs the sample-rate expansion, which is the
#    architecture's defining efficiency property
# - Magnitude predictions are exponentiated and clipped, and phase is
#    produced as unit-norm real and imaginary parts, following the
#    reference head
#
# Author: Rahul Sawhney

from typing import ClassVar, Literal, override

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from torch import nn

__all__: list[str] = ["VocosNetwork", "VocosNetworkConfig"]


class VocosNetworkConfig(BaseModel):
    # Frozen Vocos network topology and ISTFT grid settings.
    # Explicit validated fields prevent experiment settings from drifting between runs.
    #
    # Fields:
    #     input_channels: Mel band count the embedding convolution
    #         consumes. Default: ``100``.
    #     hidden_dimension: Working width of the ConvNeXt stack and input
    #         width of the head's projection. Default: ``512``.
    #     intermediate_dimension: Expanded width inside each block's
    #         pointwise pair. Default: ``1536``.
    #     layer_count: Number of ConvNeXt blocks. Default: ``8``.
    #     n_fft: Transform size of the inverse-STFT head, which projects
    #         to two more than this many channels and splits them evenly
    #         into magnitude and phase. Default: ``1024``.
    #     hop_length: Hop of the inverse-STFT head and, since the backbone
    #         never upsamples, the entire sample-rate expansion of the
    #         architecture. Default: ``256``.
    #     padding: Inverse-STFT grid. ``"center"`` uses the centered
    #         transform and yields one hop less than the frame count
    #         implies; ``"same"`` uses the explicit overlap-add path and
    #         yields exactly the frame count times the hop.
    #         Default: ``"center"``.
    #     layer_scale_initial_value: Initial value of each block's
    #         per-channel output scale. ``None`` selects the reciprocal of
    #         the layer count, which is the reference rule and keeps the
    #         summed residual contribution stable as depth changes.
    #         Default: ``None``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    input_channels: PositiveInt = 100
    hidden_dimension: PositiveInt = 512
    intermediate_dimension: PositiveInt = 1536
    layer_count: PositiveInt = 8
    n_fft: PositiveInt = 1024
    hop_length: PositiveInt = 256
    padding: Literal["center", "same"] = "center"
    layer_scale_initial_value: PositiveFloat | None = None


class _ConvNeXtBlock(nn.Module):
    # One block of the frame-rate backbone: a wide depthwise convolution
    # mixing across time, followed by a pointwise expansion and
    # contraction mixing across channels, scaled per channel and added
    # back to the residual. Time resolution is untouched, so the block can
    # be stacked to any depth without changing the frame grid the
    # inverse-STFT head expects.
    def __init__(self, dimension: int, intermediate_dimension: int, layer_scale_initial_value: float) -> None:
        # Builds the depthwise convolution, the normalization, the
        # pointwise pair with its activation, and the per-channel scale.
        # Grouping the depthwise convolution by the full width is what
        # makes the wide kernel affordable; the pointwise pair is
        # expressed as linear layers because the channel mixing runs in
        # channels-last layout.
        #
        # Args:
        #     dimension: Working width, unchanged by the block.
        #     intermediate_dimension: Expanded width between the two
        #         pointwise layers.
        #     layer_scale_initial_value: Initial value of every entry of
        #         the per-channel output scale, which starts the block
        #         near a no-op and lets depth grow into the residual
        #         stream gradually.
        super().__init__()
        self._depthwise_convolution: nn.Conv1d = nn.Conv1d(
            dimension,
            dimension,
            kernel_size=7,
            padding=3,
            groups=dimension
        )
        self._normalization: nn.LayerNorm = nn.LayerNorm(dimension, eps=1e-6)
        self._pointwise_first: nn.Linear = nn.Linear(dimension, intermediate_dimension)
        self._activation: nn.GELU = nn.GELU()
        self._pointwise_second: nn.Linear = nn.Linear(intermediate_dimension, dimension)
        self._gamma: nn.Parameter = nn.Parameter(layer_scale_initial_value * torch.ones(dimension))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mixes across time in channels-first layout, transposes to
        # channels-last for the normalization and the pointwise pair, then
        # transposes back before the residual addition, so the block's
        # input and output layouts agree.
        #
        # Args:
        #     x: Feature map shaped [batch, channels, frames].
        #
        # Returns:
        #     A feature map of exactly the input shape.
        residual: torch.Tensor = x
        x: torch.Tensor = self._depthwise_convolution(x)
        x: torch.Tensor = self._normalization(x.transpose(1, 2))
        x: torch.Tensor = self._pointwise_first(x)
        x: torch.Tensor = self._activation(x)
        x: torch.Tensor = self._pointwise_second(x)
        x: torch.Tensor = self._gamma * x
        return residual + x.transpose(1, 2)


class _VocosBackbone(nn.Module):
    # The frame-rate trunk: an embedding convolution lifting the mel into
    # the working width, a normalization, the ConvNeXt stack, and a
    # closing normalization. No stage changes the frame count, which is
    # the architecture's defining efficiency property: all computation
    # happens at frame rate and only the head expands to sample rate.
    def __init__(
        self,
        input_channels: int,
        dimension: int,
        intermediate_dimension: int,
        layer_count: int,
        layer_scale_initial_value: float | None
    ) -> None:
        # Builds the embedding convolution, the two normalizations, and
        # the block stack, then applies the reference initialization to
        # every convolution and linear layer.
        #
        # Args:
        #     input_channels: Mel band count the embedding consumes.
        #     dimension: Working width of the trunk.
        #     intermediate_dimension: Expanded width inside each block.
        #     layer_count: Number of blocks.
        #     layer_scale_initial_value: Initial per-channel output scale
        #         of every block; ``None`` resolves to the reciprocal of
        #         the layer count.
        super().__init__()
        self._embedding: nn.Conv1d = nn.Conv1d(input_channels, dimension, kernel_size=7, padding=3)
        self._normalization: nn.LayerNorm = nn.LayerNorm(dimension, eps=1e-6)
        scale_value: float = layer_scale_initial_value if layer_scale_initial_value is not None else 1 / layer_count
        self._blocks: nn.ModuleList = nn.ModuleList([
            _ConvNeXtBlock(
                dimension=dimension,
                intermediate_dimension=intermediate_dimension,
                layer_scale_initial_value=scale_value
            )
            for _ in range(layer_count)
        ])
        self._final_normalization: nn.LayerNorm = nn.LayerNorm(dimension, eps=1e-6)
        self.apply(self._initialize_weights)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Embeds the mel, normalizes in channels-last layout and returns
        # to channels-first for the block stack, then applies the closing
        # normalization and leaves the result in channels-last layout.
        #
        # Args:
        #     mel: Conditioning mel shaped [batch, bands, frames].
        #
        # Returns:
        #     Features shaped [batch, frames, channels]. The trailing
        #     transpose is deliberately not undone, because the head's
        #     projection is a linear layer that consumes the channel axis
        #     last.
        x: torch.Tensor = self._embedding(mel)
        x: torch.Tensor = self._normalization(x.transpose(1, 2)).transpose(1, 2)
        for block in self._blocks:
            x: torch.Tensor = block(x)
        return self._final_normalization(x.transpose(1, 2))

    def _initialize_weights(self, module: nn.Module) -> None:
        # Initializes model parameters according to the architecture reference behavior.
        # Every convolution and linear layer of the trunk is drawn from a
        # truncated normal at the reference scale with a zeroed bias.
        # None of them carry a weight parametrization, so unlike the
        # convolutional families this assignment writes the stored
        # parameters directly and takes effect.
        if isinstance(module, nn.Conv1d | nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class _VocosIstft(nn.Module):
    # The inverse short-time Fourier transform that turns the predicted
    # spectrum into the waveform. This single operator performs the entire
    # sample-rate expansion of the architecture, which is why the backbone
    # needs no transposed convolutions at all.
    #
    # It offers the two grids the configuration admits. The centered grid
    # delegates to the framework transform and returns one hop less than
    # the frame count implies, because the first and last half-windows are
    # trimmed. The same-padding grid performs the overlap-add explicitly
    # and returns exactly the frame count times the hop, at the cost of
    # dividing out the window envelope by hand.
    def __init__(self, n_fft: int, hop_length: int, padding: str) -> None:
        # Records the transform grid and registers the analysis window.
        # The window is a non-persistent buffer, so it moves with the
        # module across devices but never enters a checkpoint, which keeps
        # a derived constant out of the saved state.
        #
        # Args:
        #     n_fft: Transform size, also used as the window length.
        #     hop_length: Advance between consecutive frames.
        #     padding: Grid selector, validated here rather than relied on
        #         from the configuration, because this module is
        #         constructed with a plain string.
        #
        # Raises:
        #     ValueError: If the padding mode is outside the two supported
        #         grids.
        super().__init__()
        if padding not in ("center", "same"):
            raise ValueError(f"Unsupported Vocos ISTFT padding: {padding}")
        self._n_fft: int = n_fft
        self._hop_length: int = hop_length
        self._win_length: int = n_fft
        self._padding: str = padding
        self.register_buffer("_window", torch.hann_window(n_fft), persistent=False)

    @override
    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        # Reconstructs the waveform on the configured grid. The
        # same-padding branch inverts each frame, applies the synthesis
        # window, overlap-adds the frames by folding them into one signal,
        # and divides by the folded squared window so overlapping frames
        # sum to unity; the divisor is floored before the division, which
        # is what keeps a silent boundary from producing a non-finite
        # sample.
        #
        # Args:
        #     spectrum: Complex spectrum shaped
        #         [batch, bins, frames], with one more bin than half the
        #         transform size.
        #
        # Returns:
        #     A waveform batch shaped [batch, samples]. On the centered
        #     grid the sample count is one less than the frame count times
        #     the hop; on the same-padding grid it is the frame count
        #     times the hop.
        if self._padding == "center":
            return torch.istft(
                spectrum,
                n_fft=self._n_fft,
                hop_length=self._hop_length,
                win_length=self._win_length,
                window=self._window.to(spectrum.device),
                center=True
            )
        pad: int = (self._win_length - self._hop_length) // 2
        inverse: torch.Tensor = torch.fft.irfft(spectrum, self._n_fft, dim=1, norm="backward")
        inverse: torch.Tensor = inverse * self._window[None, :, None].to(spectrum.device)
        output_size: int = (spectrum.shape[-1] - 1) * self._hop_length + self._win_length
        waveform: torch.Tensor = torch.nn.functional.fold(
            inverse,
            output_size=(1, output_size),
            kernel_size=(1, self._win_length),
            stride=(1, self._hop_length)
        )[:, 0, 0, pad:-pad]
        window_square: torch.Tensor = self._window.square().to(spectrum.device).expand(1, spectrum.shape[-1], -1).transpose(1, 2)
        envelope: torch.Tensor = torch.nn.functional.fold(
            window_square,
            output_size=(1, output_size),
            kernel_size=(1, self._win_length),
            stride=(1, self._hop_length)
        ).squeeze()[pad:-pad]
        return waveform / envelope.clamp_min(1e-11)


class _IstftHead(nn.Module):
    # The synthesis head: one linear projection from the backbone width to
    # the Fourier coefficients of every frame, followed by the inverse
    # transform. The projection emits two channels per frequency bin, read
    # as a log magnitude and a phase angle rather than as a real and
    # imaginary pair, which is what keeps the predicted magnitude
    # positive and the predicted phase unconstrained.
    def __init__(self, dimension: int, n_fft: int, hop_length: int, padding: str) -> None:
        # Builds the projection to two more than the transform size, which
        # is exactly two channels for each of the transform's bins, and
        # the inverse transform it feeds.
        #
        # Args:
        #     dimension: Backbone width the projection reads.
        #     n_fft: Transform size, fixing the projection width.
        #     hop_length: Advance between frames.
        #     padding: Inverse-transform grid.
        super().__init__()
        self._output_projection: nn.Linear = nn.Linear(dimension, n_fft + 2)
        self._istft: _VocosIstft = _VocosIstft(n_fft=n_fft, hop_length=hop_length, padding=padding)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Projects the backbone features onto the Fourier coefficients of
        # every frame and reconstructs the waveform from them. The complex
        # reconstruction runs in float32 because reduced-precision complex inverse STFT is unsupported.
        # The projection output is split in half: the first half is
        # exponentiated into a magnitude and the second is read as a phase
        # angle whose cosine and sine form the unit-norm complex
        # direction. The magnitude is clipped after exponentiation, which
        # bounds an unbounded activation before it can overflow the
        # transform.
        #
        # Args:
        #     x: Backbone features shaped [batch, frames, channels].
        #
        # Returns:
        #     A waveform batch shaped [batch, samples] on the head's
        #     configured grid.
        x: torch.Tensor = self._output_projection(x).transpose(1, 2).float()
        magnitude, phase = x.chunk(2, dim=1)
        magnitude: torch.Tensor = torch.exp(magnitude).clip(max=1e2)
        real: torch.Tensor = torch.cos(phase)
        imaginary: torch.Tensor = torch.sin(phase)
        spectrum: torch.Tensor = magnitude * (real + 1j * imaginary)
        return self._istft(spectrum)


class VocosNetwork(nn.Module):
    # The Vocos generator: the frame-rate backbone composed with the
    # inverse-STFT head. This is the only module of the family that
    # carries inference cost, so it is what the complexity profiler
    # measures and what the published charactr weights are loaded onto.
    #
    # Integration: the backbone and head are held as two named members
    # whose own submodules mirror the release's structure closely enough
    # that the published checkpoint maps onto this network by key rename
    # alone. That correspondence is load-bearing for the author-weight
    # lane. The network is also the single component VocosFormer replaces:
    # everything the surrounding module contributes stays shared, so this
    # class is the boundary of the controlled comparison.
    def __init__(self, configuration: VocosNetworkConfig) -> None:
        # Builds the backbone and the head from one frozen topology
        # record and retains it, so a constructed network can always be
        # asked what recipe it was built from.
        #
        # Args:
        #     configuration: The frozen topology and grid record.
        super().__init__()
        self._configuration: VocosNetworkConfig = configuration
        self._backbone: _VocosBackbone = _VocosBackbone(
            input_channels=configuration.input_channels,
            dimension=configuration.hidden_dimension,
            intermediate_dimension=configuration.intermediate_dimension,
            layer_count=configuration.layer_count,
            layer_scale_initial_value=configuration.layer_scale_initial_value
        )
        self._head: _IstftHead = _IstftHead(
            dimension=configuration.hidden_dimension,
            n_fft=configuration.n_fft,
            hop_length=configuration.hop_length,
            padding=configuration.padding
        )

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Runs the conditioning mel through the frame-rate backbone and
        # the inverse-STFT head in one pass, with no autoregression and no
        # intermediate waveform.
        #
        # Args:
        #     mel: Conditioning mel shaped [batch, bands, frames], whose
        #         band count must equal the configured input channel
        #         count.
        #
        # Returns:
        #     A waveform batch shaped [batch, samples], with no channel
        #     axis. The sample count follows the configured grid.
        return self._head(self._backbone(mel))

    @property
    def configuration(self) -> VocosNetworkConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration
