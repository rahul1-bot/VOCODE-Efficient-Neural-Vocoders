# This module:
# 1. Implements the VocosFormer generator network: the Vocos ConvNeXt
#    chassis with the WavTokenizer position network inserted ahead of the
#    ConvNeXt stack and the ISTFT head. That position network is a
#    residual grouped convolution followed by full self-attention over
#    frames
#
# Design decisions:
# - The position network is the single architectural delta against Vocos;
#   everything else matches the reproduced baseline so the attention
#   contribution is measurable in isolation
# - Trimmed ConvNeXt dimensions offset the position network's parameters,
#   holding total capacity at the Vocos baseline within two percent
#
# Author: Rahul Sawhney

from typing import ClassVar, Literal, override

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from torch import nn

__all__: list[str] = ["VocosformerNetwork", "VocosformerNetworkConfig"]


class VocosformerNetworkConfig(BaseModel):
    # Frozen VocosFormer topology: the Vocos grid plus the position-network
    # settings.
    # The trimmed ConvNeXt stack (4 blocks at intermediate dimension 1344) holds the total
    # parameter count at the Vocos baseline within two percent after the position network
    # is inserted, per the VocosFormer design specification.
    #
    # Fields:
    #     input_channels: Mel band count the embedding convolution
    #         consumes, unchanged from the Vocos baseline.
    #         Default: ``100``.
    #     hidden_dimension: Working width of the backbone, unchanged from
    #         the baseline and shared by the position network.
    #         Default: ``512``.
    #     intermediate_dimension: Expanded width inside each ConvNeXt
    #         block. Trimmed below the baseline to absorb the position
    #         network's parameters. Default: ``1344``.
    #     layer_count: Number of ConvNeXt blocks, halved against the
    #         baseline for the same reason. It also sets the default
    #         layer-scale initialization, which is its reciprocal.
    #         Default: ``4``.
    #     n_fft: Transform size of the inverse-STFT head, unchanged from
    #         the baseline. Default: ``1024``.
    #     hop_length: Hop of the inverse-STFT head and the entire
    #         sample-rate expansion of the architecture, unchanged from
    #         the baseline. Default: ``256``.
    #     padding: Inverse-STFT grid, unchanged from the baseline.
    #         Default: ``"center"``.
    #     layer_scale_initial_value: Initial per-channel output scale of
    #         every ConvNeXt block; ``None`` selects the reciprocal of the
    #         layer count. Default: ``None``.
    #     position_group_count: Group count of every normalization inside
    #         the position network, taken from the reference decoder.
    #         Default: ``32``.
    #     position_dropout: Dropout rate inside the position network's
    #         residual convolution blocks. It is the only stochastic
    #         element in the whole network, so evaluation-mode synthesis
    #         is deterministic while training-mode synthesis is not.
    #         Default: ``0.1``.
    #
    # Note:
    #     The two trimmed fields are the parameter-matching mechanism and
    #     are not free settings: raising either without re-deriving the
    #     match would let backbone capacity masquerade as the attention
    #     contribution, which is exactly what the controlled comparison
    #     exists to rule out.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    input_channels: PositiveInt = 100
    hidden_dimension: PositiveInt = 512
    intermediate_dimension: PositiveInt = 1344
    layer_count: PositiveInt = 4
    n_fft: PositiveInt = 1024
    hop_length: PositiveInt = 256
    padding: Literal["center", "same"] = "center"
    layer_scale_initial_value: PositiveFloat | None = None
    position_group_count: PositiveInt = 32
    position_dropout: float = 0.1


class _ResidualConvBlock(nn.Module):
    # WavTokenizer-style residual convolution block for the position network.
    # The local reimplementation follows the reference decoder's GroupNorm, swish,
    # kernel-3 convolution, dropout, and residual operation sequence.
    def __init__(self, dimension: int, group_count: int, dropout: float) -> None:
        # Builds the two normalization and convolution pairs and the
        # dropout between them. Width is unchanged throughout, so the
        # block composes with the rest of the position network without any
        # projection on the residual path.
        #
        # Args:
        #     dimension: Working width, unchanged by the block.
        #     group_count: Group count of both normalizations, taken from
        #         the reference decoder rather than derived here.
        #     dropout: Rate of the dropout applied between the two
        #         convolutions.
        super().__init__()
        self._first_normalization: nn.GroupNorm = nn.GroupNorm(group_count, dimension, eps=1e-6, affine=True)
        self._first_convolution: nn.Conv1d = nn.Conv1d(dimension, dimension, kernel_size=3, stride=1, padding=1)
        self._second_normalization: nn.GroupNorm = nn.GroupNorm(group_count, dimension, eps=1e-6, affine=True)
        self._dropout: nn.Dropout = nn.Dropout(dropout)
        self._second_convolution: nn.Conv1d = nn.Conv1d(dimension, dimension, kernel_size=3, stride=1, padding=1)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Runs normalization, swish, convolution, normalization, swish,
        # dropout, convolution, and adds the result to the input. The
        # swish is written out as the product of the input with its own
        # sigmoid rather than taken from a module, matching the reference
        # decoder's inline formulation.
        #
        # Args:
        #     x: Feature map shaped [batch, channels, frames].
        #
        # Returns:
        #     A feature map of exactly the input shape.
        hidden: torch.Tensor = self._first_normalization(x)
        hidden: torch.Tensor = hidden * torch.sigmoid(hidden)
        hidden: torch.Tensor = self._first_convolution(hidden)
        hidden: torch.Tensor = self._second_normalization(hidden)
        hidden: torch.Tensor = hidden * torch.sigmoid(hidden)
        hidden: torch.Tensor = self._dropout(hidden)
        hidden: torch.Tensor = self._second_convolution(hidden)
        return x + hidden


class _FrameAttentionBlock(nn.Module):
    # WavTokenizer-style vanilla single-head self-attention over the frame axis.
    # The local reimplementation follows the reference decoder's 1x1 query, key, value,
    # and output projections, scaled attention, and residual connection.
    def __init__(self, dimension: int, group_count: int) -> None:
        # Builds the normalization, the three width-one projections that
        # produce queries, keys, and values, and the output projection.
        # Expressing the projections as convolutions rather than linear
        # layers is what keeps the whole position network in channels-first
        # layout, so it inserts into the backbone without transposes.
        #
        # Args:
        #     dimension: Working width, unchanged by the block and used
        #         directly as the attention scale.
        #     group_count: Group count of the normalization, taken from
        #         the reference decoder.
        super().__init__()
        self._normalization: nn.GroupNorm = nn.GroupNorm(group_count, dimension, eps=1e-6, affine=True)
        self._query: nn.Conv1d = nn.Conv1d(dimension, dimension, kernel_size=1, stride=1, padding=0)
        self._key: nn.Conv1d = nn.Conv1d(dimension, dimension, kernel_size=1, stride=1, padding=0)
        self._value: nn.Conv1d = nn.Conv1d(dimension, dimension, kernel_size=1, stride=1, padding=0)
        self._output_projection: nn.Conv1d = nn.Conv1d(dimension, dimension, kernel_size=1, stride=1, padding=0)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computes single-head scaled dot-product attention across the
        # frame axis and adds the projected result to the input. The
        # attention is global and unmasked, so every frame attends to
        # every other in both directions, and its cost is quadratic in the
        # frame count rather than linear. This is the one place in the
        # network where information crosses arbitrary temporal distance in
        # a single step; every other operator is local.
        #
        # No attention mask is applied, so padded frames in a batched
        # analysis are attended to exactly like signal frames. The study
        # records this row as strongly padding-sensitive, measuring a
        # materially lower padded-batch quality than true-length quality
        # on the same checkpoint, and advances unmasked global attention
        # propagating padded-frame contamination as a bounded mechanism
        # hypothesis rather than an established cause; attributing the gap
        # would require masked-attention and local-attention controls that
        # this cohort does not contain.
        #
        # Args:
        #     x: Feature map shaped [batch, channels, frames].
        #
        # Returns:
        #     A feature map of exactly the input shape, so the block can
        #     be inserted anywhere in the chain without a projection.
        normalized: torch.Tensor = self._normalization(x)
        query: torch.Tensor = self._query(normalized)
        key: torch.Tensor = self._key(normalized)
        value: torch.Tensor = self._value(normalized)
        batch_size: int = query.shape[0]
        channel_count: int = query.shape[1]
        attention_weights: torch.Tensor = torch.bmm(query.permute(0, 2, 1), key)
        attention_weights: torch.Tensor = attention_weights * (int(channel_count) ** (-0.5))
        attention_weights: torch.Tensor = torch.nn.functional.softmax(attention_weights, dim=2)
        attended: torch.Tensor = torch.bmm(value, attention_weights.permute(0, 2, 1))
        attended: torch.Tensor = attended.reshape(batch_size, channel_count, -1)
        return x + self._output_projection(attended)


class _PositionNetwork(nn.Module):
    # WavTokenizer-style position network inserted between embedding and the ConvNeXt stack.
    # Two residual convolution blocks, one frame-attention block, two further residual
    # blocks, and a closing GroupNorm, matching the reference decoder composition exactly.
    def __init__(self, dimension: int, group_count: int, dropout: float) -> None:
        # Builds the five stages and the closing normalization as one
        # sequential chain. The attention block sits in the middle rather
        # than at either end, so local convolutional context is
        # established before frames attend to one another and refined
        # afterwards; this placement is the reference decoder's, not a
        # choice made here.
        #
        # Args:
        #     dimension: Working width, unchanged throughout the network
        #         and equal to the backbone width it is inserted into.
        #     group_count: Group count of every normalization in the
        #         chain, including the closing one.
        #     dropout: Rate applied inside each residual convolution
        #         block; the attention block carries none.
        super().__init__()
        self._stages: nn.Sequential = nn.Sequential(
            _ResidualConvBlock(dimension=dimension, group_count=group_count, dropout=dropout),
            _ResidualConvBlock(dimension=dimension, group_count=group_count, dropout=dropout),
            _FrameAttentionBlock(dimension=dimension, group_count=group_count),
            _ResidualConvBlock(dimension=dimension, group_count=group_count, dropout=dropout),
            _ResidualConvBlock(dimension=dimension, group_count=group_count, dropout=dropout),
            nn.GroupNorm(group_count, dimension, eps=1e-6, affine=True)
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Runs the five stages and the closing normalization in order.
        # The frame count is preserved end to end, which is what lets the
        # network be inserted into the Vocos chassis without disturbing
        # the length arithmetic of the inverse-STFT head.
        #
        # Args:
        #     x: Feature map shaped [batch, channels, frames].
        #
        # Returns:
        #     A feature map of exactly the input shape.
        return self._stages(x)


class _ConvNeXtBlock(nn.Module):
    # One block of the frame-rate backbone, reproducing the Vocos operator
    # definition unchanged: a wide depthwise convolution mixing across
    # time, then a pointwise expansion and contraction mixing across
    # channels, scaled per channel and added back to the residual.
    # Retains the reproduced Vocos ConvNeXt operator definition; the inserted position
    # network and parameter-matching depth/width choices define the controlled adaptation.
    def __init__(self, dimension: int, intermediate_dimension: int, layer_scale_initial_value: float) -> None:
        # Builds the depthwise convolution, the normalization, the
        # pointwise pair with its activation, and the per-channel scale.
        #
        # Args:
        #     dimension: Working width, unchanged by the block.
        #     intermediate_dimension: Expanded width between the two
        #         pointwise layers, trimmed against the baseline by the
        #         parameter-matching choice.
        #     layer_scale_initial_value: Initial value of every entry of
        #         the per-channel output scale.
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


class _VocosformerBackbone(nn.Module):
    # VocosFormer backbone: Vocos embedding, the attention position network, and the
    # trimmed ConvNeXt stack, keeping temporal resolution constant at every depth.
    def __init__(
        self,
        input_channels: int,
        dimension: int,
        intermediate_dimension: int,
        layer_count: int,
        layer_scale_initial_value: float | None,
        position_group_count: int,
        position_dropout: float
    ) -> None:
        # Builds the Vocos embedding, inserts the position network
        # directly after it, and then builds the trimmed ConvNeXt stack
        # and its normalizations exactly as the baseline does. The
        # reference initialization is applied to every convolution and
        # linear layer at the end, so the position network's projections
        # are initialized on the same rule as the rest of the trunk.
        #
        # Args:
        #     input_channels: Mel band count the embedding consumes.
        #     dimension: Working width shared by the embedding, the
        #         position network, and the ConvNeXt stack.
        #     intermediate_dimension: Expanded width inside each ConvNeXt
        #         block.
        #     layer_count: Number of ConvNeXt blocks.
        #     layer_scale_initial_value: Initial per-channel output scale
        #         of every block; ``None`` resolves to the reciprocal of
        #         the layer count.
        #     position_group_count: Group count of the position network's
        #         normalizations.
        #     position_dropout: Dropout rate inside the position network.
        #
        # Note:
        #     The insertion point is between the embedding and the first
        #     normalization, so the attention operates on the embedded mel
        #     before the ConvNeXt stack sees it. Placement is part of the
        #     controlled design and is not configurable.
        super().__init__()
        self._embedding: nn.Conv1d = nn.Conv1d(input_channels, dimension, kernel_size=7, padding=3)
        self._position_network: _PositionNetwork = _PositionNetwork(
            dimension=dimension,
            group_count=position_group_count,
            dropout=position_dropout
        )
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
        # Embeds the mel, applies the position network, normalizes in
        # channels-last layout and returns to channels-first for the block
        # stack, then applies the closing normalization and leaves the
        # result in channels-last layout.
        #
        # Args:
        #     mel: Conditioning mel shaped [batch, bands, frames].
        #
        # Returns:
        #     Features shaped [batch, frames, channels]. The trailing
        #     transpose is deliberately not undone, because the inherited
        #     head's projection consumes the channel axis last.
        x: torch.Tensor = self._embedding(mel)
        x: torch.Tensor = self._position_network(x)
        x: torch.Tensor = self._normalization(x.transpose(1, 2)).transpose(1, 2)
        for block in self._blocks:
            x: torch.Tensor = block(x)
        return self._final_normalization(x.transpose(1, 2))

    def _initialize_weights(self, module: nn.Module) -> None:
        # Initializes model parameters according to the architecture reference behavior.
        # This is the Vocos rule applied unchanged: every convolution and
        # linear layer is drawn from a truncated normal at the reference
        # scale with a zeroed bias. None of them carry a weight
        # parametrization, so the assignment writes the stored parameters
        # directly and takes effect. Reusing the baseline's rule rather
        # than introducing a separate one for the position network keeps
        # initialization out of the set of differences between the two
        # rows.
        if isinstance(module, nn.Conv1d | nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class _VocosIstft(nn.Module):
    # The inverse short-time Fourier transform that turns the predicted
    # spectrum into the waveform, performing the entire sample-rate
    # expansion of the architecture. The centered grid delegates to the
    # framework transform and returns one hop less than the frame count
    # implies; the same-padding grid performs the overlap-add explicitly
    # and returns exactly the frame count times the hop.
    # Retains the reproduced Vocos inverse-STFT synthesis procedure; this head is not an
    # experimental variable in the attention-versus-convolution comparison.
    def __init__(self, n_fft: int, hop_length: int, padding: str) -> None:
        # Records the transform grid and registers the analysis window as
        # a non-persistent buffer, so it moves with the module across
        # devices but never enters a checkpoint.
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
            raise ValueError(f"Unsupported VocosFormer ISTFT padding: {padding}")
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
        # keeps a silent boundary from producing a non-finite sample.
        #
        # Args:
        #     spectrum: Complex spectrum shaped [batch, bins, frames],
        #         with one more bin than half the transform size.
        #
        # Returns:
        #     A waveform batch shaped [batch, samples]. On the centered
        #     grid the sample count is one less than the frame count times
        #     the hop; on the same-padding grid it is the frame count
        #     times the hop.
        window: torch.Tensor = self.get_buffer("_window")
        if self._padding == "center":
            return torch.istft(
                spectrum,
                n_fft=self._n_fft,
                hop_length=self._hop_length,
                win_length=self._win_length,
                window=window.to(spectrum.device),
                center=True
            )
        pad: int = (self._win_length - self._hop_length) // 2
        inverse: torch.Tensor = torch.fft.irfft(spectrum, self._n_fft, dim=1, norm="backward")
        inverse: torch.Tensor = inverse * window[None, :, None].to(spectrum.device)
        output_size: int = (spectrum.shape[-1] - 1) * self._hop_length + self._win_length
        waveform: torch.Tensor = torch.nn.functional.fold(
            inverse,
            output_size=(1, output_size),
            kernel_size=(1, self._win_length),
            stride=(1, self._hop_length)
        )[:, 0, 0, pad:-pad]
        window_square: torch.Tensor = window.square().to(spectrum.device).expand(1, spectrum.shape[-1], -1).transpose(1, 2)
        envelope: torch.Tensor = torch.nn.functional.fold(
            window_square,
            output_size=(1, output_size),
            kernel_size=(1, self._win_length),
            stride=(1, self._hop_length)
        ).squeeze()[pad:-pad]
        return waveform / envelope.clamp_min(1e-11)


class _IstftHead(nn.Module):
    # The synthesis head: one linear projection from the backbone width to
    # the Fourier coefficients of every frame, read as a log magnitude and
    # a phase angle, followed by the inverse transform.
    # Retains the reproduced Vocos spectral projection and inverse-STFT head; this head is
    # not an experimental variable in the attention-versus-convolution comparison.
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
        magnitude: torch.Tensor
        phase: torch.Tensor
        magnitude, phase = x.chunk(2, dim=1)
        magnitude: torch.Tensor = torch.exp(magnitude).clip(max=1e2)
        real: torch.Tensor = torch.cos(phase)
        imaginary: torch.Tensor = torch.sin(phase)
        spectrum: torch.Tensor = magnitude * (real + 1j * imaginary)
        return self._istft(spectrum)


class VocosformerNetwork(nn.Module):
    # The VocosFormer generator: the attention-augmented frame-rate
    # backbone composed with the reproduced inverse-STFT head. It is the
    # single component the adaptation replaces in the Vocos module, which
    # is what makes the backbone the only experimental variable in the
    # comparison.
    #
    # Integration: the class deliberately mirrors the Vocos generator's
    # constructor signature and forward contract, taking one frozen
    # topology record and returning a [batch, samples] waveform on the
    # same grid. That correspondence is what lets the module inherit
    # every step, property, and optimization declaration unchanged.
    #
    # No published release exists for this row and none is expected: the
    # registry carries no author-weight adapter for it, so it is reached
    # only through the project-trained lane. The architecture itself is
    # literature-derived, pairing the Vocos chassis with the published
    # position-network concept, and carries no novelty claim.
    def __init__(self, configuration: VocosformerNetworkConfig) -> None:
        # Builds the attention-augmented backbone and the reproduced head
        # from one frozen topology record and retains it, so a constructed
        # network can always be asked what recipe it was built from.
        #
        # Args:
        #     configuration: The frozen topology and grid record,
        #         including the position-network settings and the trimmed
        #         ConvNeXt dimensions that hold the parameter match.
        super().__init__()
        self._configuration: VocosformerNetworkConfig = configuration
        self._backbone: _VocosformerBackbone = _VocosformerBackbone(
            input_channels=configuration.input_channels,
            dimension=configuration.hidden_dimension,
            intermediate_dimension=configuration.intermediate_dimension,
            layer_count=configuration.layer_count,
            layer_scale_initial_value=configuration.layer_scale_initial_value,
            position_group_count=configuration.position_group_count,
            position_dropout=configuration.position_dropout
        )
        self._head: _IstftHead = _IstftHead(
            dimension=configuration.hidden_dimension,
            n_fft=configuration.n_fft,
            hop_length=configuration.hop_length,
            padding=configuration.padding
        )

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Runs the conditioning mel through the attention-augmented
        # backbone and the inverse-STFT head in one pass, with no
        # autoregression and no intermediate waveform.
        #
        # Args:
        #     mel: Conditioning mel shaped [batch, bands, frames], whose
        #         band count must equal the configured input channel
        #         count. Any frame count is accepted, since the attention
        #         is length-agnostic.
        #
        # Returns:
        #     A waveform batch shaped [batch, samples], with no channel
        #     axis, on the same grid the baseline produces. In training
        #     mode the result varies between calls because the position
        #     network's dropout is active; in evaluation mode it is
        #     deterministic.
        return self._head(self._backbone(mel))

    @property
    def configuration(self) -> VocosformerNetworkConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration
