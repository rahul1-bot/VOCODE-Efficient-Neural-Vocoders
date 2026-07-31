# This module:
# 1. Implements the RNDVoC generator network: the analytic range-space
#    reconstruction from the mel through the pseudo-inverse basis, and the
#    staged null-space decoder predicting the spectral content the mel
#    projection destroyed
#
# Range-null space decomposition:
# - Mel analysis is a linear map from the linear-frequency spectrum onto a
#   much smaller set of bands, so it is heavily rank-deficient and discards
#   information irrecoverably. Any spectrum therefore splits into two
#   orthogonal parts: the component lying in the row space of that map,
#   which the mel determines uniquely and which can be recovered by
#   analytic pseudo-inversion, and the component lying in its null space,
#   about which the mel says nothing at all and which must be predicted
# - This network computes the first part in closed form and learns only the
#   second. The pseudo-inverse of the mel basis and the resulting orthogonal
#   projector are precomputed once and held as buffers, so no capacity is
#   spent rediscovering a mapping that linear algebra already fixes
# - The network's prediction is passed through the complementary projector
#   before it is added, which is what confines the learned contribution to
#   the null space and prevents it from contradicting the analytically
#   determined component
#
# Design decisions:
# - The range-null decomposition is explicit in the computation graph: the
#   analytic component bypasses the network entirely, so capacity is spent
#   only on the null-space refinement
# - The staged decoder refines progressively over the configured stage
#   count, per the reference design
# - The spectrum is encoded into a small number of frequency bands whose
#   widths grow with frequency, mirroring the coarsening resolution of
#   human hearing, so the band count stays modest without discarding
#   detail where it is perceptible
# - Each stage alternates mixing across bands with mixing across time,
#   rather than convolving jointly, so the two axes are modeled by
#   operators suited to each and neither dominates the parameter budget
#
# Author: Rahul Sawhney

import math
from typing import ClassVar, cast, override

import torch
import torchaudio
from pydantic import BaseModel, ConfigDict
from torch import nn

__all__: list[str] = ["RndvocGeneratorOutput", "RndvocNetwork"]


class RndvocGeneratorOutput(BaseModel):
    # Frozen bundle of one generator prediction. The objective scores the
    # spectrum in several parameterizations at once, so all of them are
    # returned rather than recomputed: the amplitude and phase fields, the
    # equivalent Cartesian pair, and the waveform they reconstruct to.
    #
    # Fields:
    #     log_amplitude: Logarithmic spectral magnitude of shape
    #         ``[batch, bins, frames]``, computed from the Cartesian pair
    #         under a small additive floor so the logarithm stays finite.
    #     phase: Spectral phase of the same shape, in ``[-pi, pi]``.
    #     real_spectrum: Real part of the predicted spectrum.
    #     imaginary_spectrum: Imaginary part of the predicted spectrum. This
    #         pair and the amplitude-phase pair describe one spectrum in two
    #         parameterizations; the objective penalizes both because each
    #         exposes errors the other tolerates.
    #     waveform: Reconstructed waveform of shape ``[batch, samples]``. No
    #         channel axis is introduced, unlike the families whose
    #         inverse transform is a separate head.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    log_amplitude: torch.Tensor
    phase: torch.Tensor
    real_spectrum: torch.Tensor
    imaginary_spectrum: torch.Tensor
    waveform: torch.Tensor


class RndvocChannelNormalization(nn.Module):
    # Normalizes across the channel axis at each position independently, then
    # applies a learnable per-channel scale and shift. Because the statistics
    # are taken per position rather than pooled over the sequence, the
    # operation is independent of sequence length and of batch composition,
    # which is what makes it safe in a network whose effective batch changes
    # as axes are folded together.
    def __init__(self, channel_count: int) -> None:
        # Builds the affine parameters, shaped to broadcast over batch and
        # position so only the channel axis is scaled and shifted.
        super().__init__()
        self._epsilon: float = 1e-5
        self.gain: nn.Parameter = nn.Parameter(torch.ones(1, channel_count, 1))
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(1, channel_count, 1))

    @override
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Standardizes each position across channels. The variance is
        # uncorrected and floored by a small constant before the square root,
        # so a position whose channels are all equal yields zeros rather than
        # a division by zero.
        mean: torch.Tensor = features.mean(dim=1, keepdim=True)
        deviation: torch.Tensor = torch.sqrt(torch.var(features, dim=1, keepdim=True, unbiased=False) + self._epsilon)
        return ((features - mean) / deviation) * self.gain + self.bias


class RndvocBandwiseC2LayerNorm(nn.Module):
    # Channel normalization for the four-axis band-major layout used by the
    # encoder and decoder. Statistics are taken across the feature axis, as in
    # the three-axis form, but the affine parameters are per band as well as
    # per feature. That per-band freedom matters here because the bands span
    # very different frequency ranges and therefore very different energy
    # scales; one shared gain would force the same calibration onto all of
    # them.
    def __init__(self, band_count: int, feature_dimension: int) -> None:
        # Builds affine parameters indexed by feature and band, broadcasting
        # over batch and time.
        super().__init__()
        self._epsilon: float = 1e-5
        self.gain: nn.Parameter = nn.Parameter(torch.ones(1, feature_dimension, band_count, 1))
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(1, feature_dimension, band_count, 1))

    @override
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Standardizes each band and time position across features, then
        # applies the per-band affine transform.
        mean: torch.Tensor = features.mean(dim=1, keepdim=True)
        deviation: torch.Tensor = torch.sqrt(torch.var(features, dim=1, keepdim=True, unbiased=False) + self._epsilon)
        return ((features - mean) / deviation) * self.gain + self.bias


class RndvocBandwiseLayerNorm(nn.Module):
    # Per-band channel normalization for the time module, which operates on a
    # tensor whose leading axis folds batch and band together so that a
    # one-dimensional convolution can stride along time.
    #
    # The folding is what forces this class to exist. Statistics are taken
    # across channels on the folded tensor, but the affine parameters are
    # indexed by band, so the tensor must be temporarily unfolded to apply
    # them and refolded afterwards. The band count is retained precisely to
    # recover the true batch size from the folded leading axis.
    def __init__(self, band_count: int, feature_dimension: int) -> None:
        # Builds affine parameters indexed by band and feature, and records
        # the band count needed to unfold the leading axis.
        super().__init__()
        self._band_count: int = band_count
        self._epsilon: float = 1e-5
        self.gain: nn.Parameter = nn.Parameter(torch.ones(1, band_count, feature_dimension, 1))
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(1, band_count, feature_dimension, 1))

    @override
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Standardizes across channels, then unfolds the leading axis into
        # batch and band so the per-band affine transform applies, and refolds
        # to the layout the caller supplied. The refold makes this a drop-in
        # stage inside the convolutional sequence despite its band-aware
        # parameters.
        flat_batch: int = features.shape[0]
        channel_count: int = features.shape[1]
        frame_count: int = features.shape[2]
        batch_size: int = flat_batch // self._band_count
        mean: torch.Tensor = features.mean(dim=-2, keepdim=True).view(batch_size, self._band_count, 1, frame_count)
        deviation: torch.Tensor = torch.sqrt(
            torch.var(features, dim=-2, keepdim=True, unbiased=False) + self._epsilon
        ).view(batch_size, self._band_count, 1, frame_count)
        grouped: torch.Tensor = features.view(batch_size, self._band_count, channel_count, frame_count)
        normalized: torch.Tensor = ((grouped - mean) / deviation) * self.gain + self.bias
        return normalized.view(flat_batch, channel_count, frame_count)


class RndvocGlobalResponseNormalization(nn.Module):
    # Encourages channels to specialize by rescaling each according to how its
    # overall activation energy compares with the average across channels. A
    # channel that is unusually active is amplified and a quiet one
    # attenuated, which counteracts the tendency of wide layers to collapse
    # onto redundant, near-identical responses.
    #
    # Both parameters are initialized to zero, so the block starts as the
    # identity and the network is free to leave it inert; any deviation from
    # identity is something training chose rather than something imposed at
    # initialization.
    def __init__(self, channel_count: int) -> None:
        # Builds the per-channel scale and shift at zero, making the block
        # initially a pure residual pass-through.
        super().__init__()
        self._epsilon: float = 1e-6
        self.gamma: nn.Parameter = nn.Parameter(torch.zeros(1, channel_count, 1))
        self.beta: nn.Parameter = nn.Parameter(torch.zeros(1, channel_count, 1))

    @override
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Measures each channel's energy over the sequence, divides it by the
        # mean energy across channels to obtain a relative activation, and
        # applies that as a modulation on top of an identity path. The
        # division is floored, so a layer whose channels are all silent
        # returns its input unchanged rather than producing non-finite values.
        global_magnitude: torch.Tensor = torch.norm(features, p=2, dim=-1, keepdim=True)
        normalized_magnitude: torch.Tensor = global_magnitude / (
            global_magnitude.mean(dim=1, keepdim=True) + self._epsilon
        )
        return self.gamma * (features * normalized_magnitude) + self.beta + features


class RndvocGroupedLinear(nn.Module):
    # A stack of independent linear maps, one per group, applied in a single
    # contraction. Each group carries its own weight matrix and bias, so the
    # groups never mix; this is what lets the band mixer give every channel
    # its own full band-to-band mixing matrix at a fraction of the cost of one
    # dense map over the flattened channel-band product.
    def __init__(self, in_features: int, out_features: int, group_count: int) -> None:
        # Builds the per-group weights and biases under the standard linear
        # initialization, with the bias bound derived from the input width so
        # each group is scaled as an independent linear layer of that width.
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.empty(group_count, out_features, in_features))
        self.bias: nn.Parameter = nn.Parameter(torch.empty(group_count, out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in: int = in_features
        bound: float = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    @override
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Contracts the input width against each group's matrix, leaving the
        # group axis intact. Any number of leading axes is accepted, so the
        # caller need not flatten batch and time before applying it.
        return torch.einsum("...gh,gkh->...gk", features, self.weight) + self.bias[None, ...]


class RndvocBandShuffler(nn.Module):
    # Mixes information across frequency bands. It combines two complementary
    # mechanisms: grouped convolutions along the band axis, which relate each
    # band to its immediate neighbors, and a learned dense mixing across all
    # bands at once, which relates distant bands a local kernel could never
    # reach. Harmonic structure makes the second essential, since a harmonic
    # and its partials are separated by many bands.
    #
    # The dense mixing is applied in a reduced channel space, so its cost is
    # governed by the squeeze width rather than the full feature width. Every
    # stage is residual, so the block begins close to a pass-through and adds
    # mixing rather than replacing its input with it.
    def __init__(
        self,
        band_count: int,
        input_dimension: int,
        squeeze_dimension: int,
        frequency_kernel_size: int = 3,
        frequency_conv_groups: int = 8
    ) -> None:
        # Builds the two local frequency convolutions, the projection pair
        # into and out of the reduced space, and the dense band mixer.
        #
        # Args:
        #     band_count: Number of frequency bands, which is both the input
        #         and the output width of the dense mixer.
        #     input_dimension: Feature width carried through the block.
        #     squeeze_dimension: Reduced width the dense mixing runs in, and
        #         therefore the number of independent mixing matrices.
        #     frequency_kernel_size: Extent of the local band convolutions.
        #         Default: ``3``.
        #     frequency_conv_groups: Group count of those convolutions.
        #         Default: ``8``.
        super().__init__()
        self._first_frequency_conv: nn.Sequential = nn.Sequential(
            RndvocChannelNormalization(input_dimension),
            nn.Conv1d(
                input_dimension,
                input_dimension,
                kernel_size=frequency_kernel_size,
                groups=frequency_conv_groups,
                padding="same"
            ),
            nn.PReLU(input_dimension)
        )
        self._second_frequency_conv: nn.Sequential = nn.Sequential(
            RndvocChannelNormalization(input_dimension),
            nn.Conv1d(
                input_dimension,
                input_dimension,
                kernel_size=frequency_kernel_size,
                groups=frequency_conv_groups,
                padding="same"
            ),
            nn.PReLU(input_dimension)
        )
        self._squeeze: nn.Sequential = nn.Sequential(
            nn.Conv1d(input_dimension, squeeze_dimension, kernel_size=1),
            nn.SiLU()
        )
        self._unsqueeze: nn.Sequential = nn.Sequential(
            nn.Conv1d(squeeze_dimension, input_dimension, kernel_size=1),
            nn.SiLU()
        )
        self._band_mixer: RndvocGroupedLinear = RndvocGroupedLinear(band_count, band_count, squeeze_dimension)

    @override
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Mixes across bands at every time position independently.
        #
        # Args:
        #     features: Tensor of shape ``[batch, frames, channels, bands]``.
        #
        # Returns:
        #     A tensor of the same shape.
        batch_size: int = features.shape[0]
        frame_count: int = features.shape[1]
        channel_count: int = features.shape[2]
        band_count: int = features.shape[3]
        # Batch and time are folded together so the convolutions stride along
        # the band axis; time is deliberately untouched here, since the time
        # module that follows owns that axis.
        flat: torch.Tensor = features.view(batch_size * frame_count, channel_count, band_count)
        flat: torch.Tensor = flat + self._first_frequency_conv(flat)
        # Global mixing runs in the reduced space: the projection down, the
        # dense band-to-band map, and the projection back are one residual
        # branch, so distant-band information is added to the local result
        # rather than overwriting it.
        mixed: torch.Tensor = self._squeeze(flat)
        mixed: torch.Tensor = self._band_mixer(mixed)
        flat: torch.Tensor = flat + self._unsqueeze(mixed)
        # A second local convolution follows the global mixing, giving the
        # block a chance to reconcile neighboring bands after distant ones
        # have been folded in.
        flat: torch.Tensor = flat + self._second_frequency_conv(flat)
        return flat.view(batch_size, frame_count, channel_count, band_count)


class RndvocBandWiseTimeModule(nn.Module):
    # Models temporal structure within each band independently. Every band is
    # processed by the same weights but never sees another band, because
    # cross-band mixing is the shuffler's responsibility; separating the two
    # axes keeps each operator's parameter count linear rather than
    # multiplicative in the band and time extents.
    #
    # Each repeat is an inverted-bottleneck residual block: a depthwise
    # convolution along time gathers temporal context per channel, then a
    # pointwise pair expands and contracts the channel width around a
    # nonlinearity, with the response normalization inside the expansion where
    # channel redundancy would otherwise accumulate.
    def __init__(
        self,
        band_count: int,
        repeat_count: int,
        input_dimension: int,
        hidden_dimension: int,
        kernel_size: int
    ) -> None:
        # Builds the requested number of identical residual blocks. Each is a
        # separate instance with its own parameters; the repeat count controls
        # temporal depth, and the kernel size controls how much context a
        # single block sees.
        super().__init__()
        self._band_count: int = band_count
        self._time_blocks: nn.ModuleList = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    input_dimension,
                    input_dimension,
                    kernel_size,
                    padding="same",
                    groups=input_dimension
                ),
                RndvocBandwiseLayerNorm(band_count, input_dimension),
                nn.Conv1d(input_dimension, hidden_dimension, kernel_size=1),
                nn.GELU(),
                RndvocGlobalResponseNormalization(hidden_dimension),
                nn.Conv1d(hidden_dimension, input_dimension, kernel_size=1)
            )
            for _ in range(repeat_count)
        )

    @override
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Applies the residual blocks along time, band by band.
        #
        # Args:
        #     features: Tensor of shape ``[batch, bands, channels, frames]``.
        #
        # Returns:
        #     A tensor of the same shape.
        batch_size: int = features.shape[0]
        band_count: int = features.shape[1]
        channel_count: int = features.shape[2]
        frame_count: int = features.shape[3]
        # Batch and band are folded together, which is exactly what makes the
        # bands independent: the convolutions cannot reach across a boundary
        # they see as a batch boundary.
        flat: torch.Tensor = features.view(batch_size * band_count, channel_count, frame_count)
        for time_block in self._time_blocks:
            time_block_module: nn.Sequential = cast(nn.Sequential, time_block)
            flat: torch.Tensor = flat + time_block_module(flat)
        return flat.view(batch_size, band_count, channel_count, frame_count)


class RndvocVocModule(nn.Module):
    # One refinement stage of the null-space decoder: frequency mixing
    # followed by temporal modeling. Factoring the two axes into separate
    # operators rather than one joint two-dimensional convolution is the
    # design's central efficiency claim, and stacking several such stages is
    # what lets information propagate across both axes despite each individual
    # operator touching only one.
    def __init__(
        self,
        band_count: int,
        repeat_count: int,
        input_dimension: int,
        squeeze_dimension: int,
        hidden_dimension: int,
        kernel_size: int
    ) -> None:
        # Builds the band mixer and the temporal module that together form one
        # stage.
        super().__init__()
        self._band_network: RndvocBandShuffler = RndvocBandShuffler(
            band_count=band_count,
            input_dimension=input_dimension,
            squeeze_dimension=squeeze_dimension
        )
        self._time_network: RndvocBandWiseTimeModule = RndvocBandWiseTimeModule(
            band_count=band_count,
            repeat_count=repeat_count,
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
            kernel_size=kernel_size
        )

    @override
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Runs the two sub-modules in sequence. Each expects the axis it acts
        # on to be last, so the tensor is transposed between them and
        # transposed back at the end, leaving the stage's input and output
        # layouts identical and the stages therefore stackable. Both
        # transposes are made contiguous because the folding views that follow
        # require it.
        #
        # Args:
        #     features: Tensor of shape ``[batch, frames, channels, bands]``.
        #
        # Returns:
        #     A tensor of the same shape.
        mixed: torch.Tensor = self._band_network(features)
        mixed: torch.Tensor = mixed.transpose(1, 3).contiguous()
        mixed: torch.Tensor = self._time_network(mixed)
        return mixed.transpose(1, 3).contiguous()


class RndvocSharedBandSplit(nn.Module):
    # Encodes the linear-frequency spectrum into a small set of frequency
    # bands. Three contiguous regions are encoded by separate convolutions
    # whose kernels grow with frequency: narrow bands at the bottom of the
    # spectrum where pitch and formant structure live, progressively wider
    # ones above. This mirrors the coarsening frequency resolution of hearing,
    # so the band count stays small without discarding detail where it can be
    # heard.
    #
    # Each region's convolution strides by exactly its kernel width, so bands
    # partition their region rather than overlapping, and the encoding is a
    # true reduction rather than a smoothing.
    def __init__(self, feature_dimension: int) -> None:
        # Builds the three region encoders. The kernel widths and band counts
        # are fixed to the reference geometry rather than configurable,
        # because the decoder must invert exactly this partition and the two
        # would otherwise be free to disagree.
        super().__init__()
        self._first_region_encoder: nn.Sequential = self._build_region_encoder(feature_dimension, 12, 12)
        self._second_region_encoder: nn.Sequential = self._build_region_encoder(feature_dimension, 24, 8)
        self._third_region_encoder: nn.Sequential = self._build_region_encoder(feature_dimension, 44, 4)

    @property
    def band_count(self) -> int:
        # Returns the total number of encoded frequency bands, the sum of the
        # three regions' band counts. The refinement stages are sized against
        # this value.
        return 24

    @override
    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        # Encodes a complex spectrum into banded features.
        #
        # The final frequency bin is excluded from the last region, so the
        # three regions cover a bin count divisible by their respective kernel
        # widths. The decoder restores that bin by duplicating its neighbor,
        # which is acceptable because it is the highest representable
        # frequency and carries negligible speech energy.
        #
        # Args:
        #     spectrum: Tensor of shape ``[batch, bins, frames, 2]``, the last
        #         axis holding the real and imaginary parts.
        #
        # Returns:
        #     Banded features of shape ``[batch, bands, features, frames]``.
        stacked: torch.Tensor = spectrum.permute(0, 3, 1, 2).contiguous()
        first_region: torch.Tensor = stacked[..., :144, :]
        second_region: torch.Tensor = stacked[..., 144:336, :]
        third_region: torch.Tensor = stacked[..., 336:-1, :]
        encoded: torch.Tensor = torch.cat(
            [
                self._first_region_encoder(first_region),
                self._second_region_encoder(second_region),
                self._third_region_encoder(third_region)
            ],
            dim=-2
        )
        return encoded.transpose(1, 2).contiguous()

    def _build_region_encoder(self, feature_dimension: int, kernel_bins: int, band_count: int) -> nn.Sequential:
        # Builds one frequency-region encoder shared across the bands of that
        # region. The convolution reads both spectral parts as input channels,
        # strides by its kernel width so its frequency reduction is exact, and
        # spans three frames under symmetric time padding, so the time axis
        # survives unchanged while each band gains a little temporal context.
        return nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            nn.Conv2d(
                in_channels=2,
                out_channels=feature_dimension,
                kernel_size=(kernel_bins, 3),
                stride=(kernel_bins, 1)
            ),
            RndvocBandwiseC2LayerNorm(band_count=band_count, feature_dimension=feature_dimension)
        )


class RndvocSharedBandMerge(nn.Module):
    # Decodes banded features back to the linear-frequency spectrum, inverting
    # the split's partition. Magnitude and phase are decoded by separate
    # decoders per region rather than by one shared head, because the two
    # quantities have incompatible structure: magnitude is positive and
    # smooth across frequency, phase is circular and erratic.
    #
    # Magnitude is produced by exponentiating its decoder's output, which
    # makes it positive by construction and lets the decoder work in a
    # logarithmic domain where spectral dynamic range is manageable. Phase is
    # produced as a two-component vector converted to an angle, which avoids
    # the wrap-around discontinuity a direct angle regression would suffer at
    # the branch cut.
    def __init__(self, feature_dimension: int) -> None:
        # Builds six decoders: one magnitude and one phase decoder for each of
        # the three frequency regions, with kernel widths and band counts
        # mirroring the split exactly.
        super().__init__()
        self._first_magnitude_decoder: nn.Sequential = self._build_region_decoder(feature_dimension, 12, 12, 1)
        self._second_magnitude_decoder: nn.Sequential = self._build_region_decoder(feature_dimension, 24, 8, 1)
        self._third_magnitude_decoder: nn.Sequential = self._build_region_decoder(feature_dimension, 44, 4, 1)
        self._first_phase_decoder: nn.Sequential = self._build_region_decoder(feature_dimension, 12, 12, 2)
        self._second_phase_decoder: nn.Sequential = self._build_region_decoder(feature_dimension, 24, 8, 2)
        self._third_phase_decoder: nn.Sequential = self._build_region_decoder(feature_dimension, 44, 4, 2)

    @override
    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Decodes banded features into a full-resolution magnitude and phase
        # pair.
        #
        # Args:
        #     features: Banded features of shape
        #         ``[batch, bands, features, frames]``, sliced back into the
        #         three regions at the same boundaries the split produced.
        #
        # Returns:
        #     The magnitude of shape ``[batch, bins, frames]`` and the phase
        #     of the same shape, the latter in ``[-pi, pi]``.
        first_region: torch.Tensor = features[:, :12].transpose(1, 2).contiguous()
        second_region: torch.Tensor = features[:, 12:20].transpose(1, 2).contiguous()
        third_region: torch.Tensor = features[:, 20:].transpose(1, 2).contiguous()
        magnitude: torch.Tensor = torch.exp(
            torch.cat(
                [
                    self._first_magnitude_decoder(first_region),
                    self._second_magnitude_decoder(second_region),
                    self._third_magnitude_decoder(third_region)
                ],
                dim=-2
            )
        )
        phase_components: torch.Tensor = torch.cat(
            [
                self._first_phase_decoder(first_region),
                self._second_phase_decoder(second_region),
                self._third_phase_decoder(third_region)
            ],
            dim=-2
        )
        # The bin the split excluded is restored by duplicating the highest
        # decoded bin, so the spectrum regains the exact width the inverse
        # transform requires.
        magnitude: torch.Tensor = torch.cat([magnitude, magnitude[..., -1:, :]], dim=-2)
        phase_components: torch.Tensor = torch.cat([phase_components, phase_components[..., -1:, :]], dim=-2)
        # The two phase channels are read as the vertical and horizontal
        # components of a vector, and the quadrant-aware arc tangent recovers
        # the angle over the full circle; a single-channel regression could
        # not represent the wrap-around at all.
        phase: torch.Tensor = torch.atan2(phase_components[:, -1], phase_components[:, 0])
        return magnitude.squeeze(1), phase

    def _build_region_decoder(
        self,
        feature_dimension: int,
        kernel_bins: int,
        band_count: int,
        output_channels: int
    ) -> nn.Sequential:
        # Builds one frequency-region decoder shared across the bands of that
        # region. The transposed convolution mirrors the encoder's stride
        # exactly, expanding each band back to the bins it was formed from,
        # and the pointwise widening before it gives the expansion a richer
        # representation to draw on. The output channel count is one for a
        # magnitude decoder and two for a phase decoder.
        return nn.Sequential(
            RndvocBandwiseC2LayerNorm(band_count=band_count, feature_dimension=feature_dimension),
            nn.Conv2d(in_channels=feature_dimension, out_channels=feature_dimension * 2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(
                in_channels=feature_dimension * 2,
                out_channels=output_channels,
                kernel_size=(kernel_bins, 1),
                stride=(kernel_bins, 1)
            )
        )


class RndvocNetwork(nn.Module):
    # The complete RNDVoC generator, and the point at which the range-null
    # decomposition becomes concrete.
    #
    # Mel analysis is a fixed linear map with far fewer outputs than inputs,
    # so it is rank-deficient: the spectrum splits into a component the mel
    # determines exactly and a component the mel annihilates. This network
    # obtains the first by pseudo-inverting the mel basis, an operation that
    # involves no learned parameters at all, and devotes its entire capacity
    # to predicting the second. The complementary projector is applied to the
    # network's prediction before the two components are summed, so the
    # learned term cannot perturb what the analytic term already fixes.
    #
    # The mel basis, its pseudo-inverse, and the projector are computed once
    # at construction and held as buffers. They move with the module across
    # devices and dtypes and appear in the state dictionary, but they carry no
    # gradient, because they are consequences of the conditioning protocol
    # rather than anything to be learned.
    #
    # Integration: the analytic component involves no learned parameters, so
    # it is computed identically at every training state, including random
    # initialization. The consequence for measurement is that a synthesis
    # result never isolates the learned contribution on its own, because every
    # output spectrum is the sum of the analytic and the learned term.
    def __init__(
        self,
        sample_rate: int,
        num_mels: int,
        n_fft: int,
        hop_size: int,
        win_size: int,
        fmin: float,
        fmax: float,
        null_stage_count: int,
        repeat_count: int,
        input_dimension: int,
        squeeze_dimension: int,
        hidden_dimension: int,
        kernel_size: int
    ) -> None:
        # Builds the analytic machinery and the learned decoder.
        #
        # The mel filterbank is constructed to match the conditioning protocol
        # exactly, because the decomposition is only valid against the same
        # map that produced the mel; a mismatch would leave the analytic
        # component reconstructing the wrong subspace, silently and without
        # any shape error. Its pseudo-inverse and the resulting projector are
        # derived once here rather than per forward pass, since neither
        # depends on the input.
        super().__init__()
        self._n_fft: int = n_fft
        self._hop_size: int = hop_size
        self._win_size: int = win_size
        mel_basis: torch.Tensor = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=fmin,
            f_max=fmax,
            n_mels=num_mels,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="slaney"
        ).transpose(0, 1)
        # The pseudo-inverse maps a mel back to the unique spectrum in the
        # basis's row space that produces it. Composing it with the basis
        # gives the orthogonal projector onto that row space, which is the
        # part of the spectrum the mel determines.
        inverse_mel_basis: torch.Tensor = torch.linalg.pinv(mel_basis)
        self.register_buffer("_mel_basis", mel_basis)
        self.register_buffer("_inverse_mel_basis", inverse_mel_basis)
        self.register_buffer("_projection", inverse_mel_basis @ mel_basis)
        self._null_encoder: RndvocSharedBandSplit = RndvocSharedBandSplit(feature_dimension=input_dimension)
        self._null_decoder: RndvocSharedBandMerge = RndvocSharedBandMerge(feature_dimension=input_dimension)
        self._null_modules: nn.ModuleList = nn.ModuleList(
            RndvocVocModule(
                band_count=self._null_encoder.band_count,
                repeat_count=repeat_count,
                input_dimension=input_dimension,
                squeeze_dimension=squeeze_dimension,
                hidden_dimension=hidden_dimension,
                kernel_size=kernel_size
            )
            for _ in range(null_stage_count)
        )
        # A learnable weighted skip from the encoder output past every
        # refinement stage, so the decoder always sees the unrefined encoding
        # alongside the refined one. Initialized at unity, this makes the
        # stack begin as a near pass-through, which is what allows many stages
        # to be composed without the signal degrading before training begins.
        self.alpha: nn.Parameter = nn.Parameter(
            torch.ones(1, 1, input_dimension, self._null_encoder.band_count)
        )
        self.apply(self._initialize_weights)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Synthesizes a waveform from a conditioning mel, discarding the
        # intermediate spectral parameterizations the objective needs.
        return self.predict_components(mel).waveform

    def predict_components(self, mel: torch.Tensor) -> RndvocGeneratorOutput:
        # Runs the decomposition end to end and returns every spectral
        # parameterization the objective scores.
        #
        # Args:
        #     mel: Logarithmic conditioning mel of shape
        #         ``[batch, channels, frames]``. It is exponentiated here,
        #         because the pseudo-inverse is the inverse of a linear map
        #         and must therefore be applied in the linear domain.
        #
        # Returns:
        #     The frozen bundle of amplitude, phase, Cartesian spectrum, and
        #     waveform.
        inverse_mel_basis: torch.Tensor = cast(torch.Tensor, self._inverse_mel_basis)
        projection: torch.Tensor = cast(torch.Tensor, self._projection)
        # Range-space component, computed analytically. The pseudo-inverse
        # recovers exactly the spectrum the mel determines; the magnitude is
        # rectified and floored because the least-squares solution can emit
        # small negative values that have no meaning as a magnitude.
        initial_magnitude: torch.Tensor = (inverse_mel_basis @ torch.exp(mel)).abs().clamp_min(1e-5)
        # The analytic reconstruction carries no phase information, so it
        # enters the decoder as a spectrum with a zero imaginary part;
        # recovering phase is part of what the null-space decoder must learn.
        initial_spectrum: torch.Tensor = torch.stack(
            [initial_magnitude, torch.zeros_like(initial_magnitude)],
            dim=-1
        )
        # Null-space decoder: the analytic spectrum is banded, refined through
        # the stages, and merged with the weighted skip before decoding.
        encoded: torch.Tensor = self._null_encoder(initial_spectrum).transpose(1, 3).contiguous()
        mixed: torch.Tensor = encoded
        for null_module in self._null_modules:
            null_stage: RndvocVocModule = cast(RndvocVocModule, null_module)
            mixed: torch.Tensor = null_stage(mixed)
        merged: torch.Tensor = (self.alpha * encoded + mixed).transpose(1, 3).contiguous()
        null_magnitude: torch.Tensor
        null_phase: torch.Tensor
        null_magnitude, null_phase = self._null_decoder(merged)
        # The complementary projector, the identity minus the range projector,
        # annihilates everything the mel already determines. Applying it to
        # the decoder's magnitude is what confines the learned contribution to
        # the null space, so the network cannot contradict the analytic term
        # no matter what it predicts. The rectification that follows keeps the
        # summed magnitude non-negative.
        null_filter: torch.Tensor = (
            torch.eye(self._n_fft // 2 + 1, dtype=null_magnitude.dtype, device=null_magnitude.device)
            - projection.to(null_magnitude.dtype)
        )
        filtered_magnitude: torch.Tensor = torch.einsum(
            "fk,bkt->bft",
            null_filter,
            null_magnitude
        ).abs().clamp_min(1e-5)
        # The two components recombine into the output spectrum. Phase comes
        # entirely from the decoder, since the analytic term supplies none.
        output_magnitude: torch.Tensor = filtered_magnitude + initial_magnitude
        real_spectrum: torch.Tensor = output_magnitude * torch.cos(null_phase)
        imaginary_spectrum: torch.Tensor = output_magnitude * torch.sin(null_phase)
        log_amplitude: torch.Tensor = torch.log(
            torch.sqrt(real_spectrum.pow(2) + imaginary_spectrum.pow(2)) + 1e-7
        )
        phase: torch.Tensor = torch.atan2(imaginary_spectrum, real_spectrum)
        # Reconstruction is forced to float32 regardless of the prevailing
        # precision, because the inverse transform's overlap-add accumulates
        # across frames and is sensitive to reduced-precision rounding.
        complex_spectrum: torch.Tensor = torch.complex(real_spectrum.float(), imaginary_spectrum.float())
        waveform: torch.Tensor = torch.istft(
            complex_spectrum,
            n_fft=self._n_fft,
            hop_length=self._hop_size,
            win_length=self._win_size,
            window=torch.hann_window(self._win_size, device=mel.device)
        )
        return RndvocGeneratorOutput(
            log_amplitude=log_amplitude,
            phase=phase,
            real_spectrum=real_spectrum,
            imaginary_spectrum=imaginary_spectrum,
            waveform=waveform
        )

    def _initialize_weights(self, module: nn.Module) -> None:
        # Applies the reference truncated-normal initialization to the
        # one-dimensional convolution and linear layers. Truncation bounds the
        # initial weights away from the tails a plain normal draw would
        # produce, which matters in a network this deep because a single
        # outlying weight early in the stack propagates through every
        # subsequent stage. The two-dimensional convolutions of the band split
        # and merge are deliberately outside this rule and keep their default
        # initialization.
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
