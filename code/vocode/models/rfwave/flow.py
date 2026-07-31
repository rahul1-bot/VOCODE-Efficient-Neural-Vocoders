# This module:
# 1. Implements the rectified-flow training machinery: the noised-state
#    construction pairing clean band spectra with Gaussian noise at
#    sampled times, and the velocity-target definition the network
#    regresses
#
# Flow-matching formulation:
# - Generation is posed as transport along a path from a Gaussian noise
#   spectrum to the data spectrum. The path chosen is the straight line
#   between the two endpoints, so the state at time t is their convex
#   combination and the velocity along that path is their difference
# - Because the path is straight, that velocity is constant in t. This is
#   the property the whole design rests on: the exact solution is a single
#   linear step, so integrating it numerically converges in very few steps
#   rather than requiring the long schedules a curved path would demand
# - Training therefore never simulates the path. It samples one time
#   uniformly, forms the corresponding state directly, and regresses the
#   network's predicted velocity onto the constant target, which makes each
#   training step cost one forward pass regardless of how many steps
#   synthesis will later use
#
# Design decisions:
# - The straight-path interpolation between noise and data defines the
#   velocity target, the rectified-flow formulation of the reference
# - Training tuples carry the sampled time and band index so the backbone
#   conditions on both
# - All bands are processed jointly by folding the band axis into the batch
#   axis, so one backbone serves every band and is distinguished only by the
#   band-index conditioning it receives
# - One time value is drawn per utterance and shared across its bands, not
#   drawn per band, so the bands of a sample are always at the same point on
#   the path and can be recombined into a coherent spectrum
# - Subband slicing pads circularly, so the highest band's upper context
#   wraps around to the lowest bins; the padding widths are asymmetric and
#   the final band retains one extra bin, which is how the odd bin count of
#   a real spectrum is partitioned into equal bands
#
# Author: Rahul Sawhney

import torch
from torch import nn

from vocode.models.rfwave.network import (
    RfwaveBackbone,
    RfwaveNetworkConfig,
    RfwavePqmfEqualizer,
    RfwaveSpectralTransform,
)

__all__: list[str] = ["RfwaveRectifiedFlow", "RfwaveTrainTuple"]


class RfwaveTrainTuple:
    # Value object carrying one joint-parallel rectified-flow training tuple.
    # Every member is already flattened over the band axis, so its leading
    # dimension is the batch size times the band count and the five members
    # are aligned row for row.
    #
    # Fields:
    #     expanded_mel: Conditioning mel repeated once per band, so each
    #         band's row carries its own utterance's conditioning.
    #     band_index: Integer band identifier per row, which is the only
    #         signal distinguishing one band's row from another's inside the
    #         shared backbone.
    #     noisy_state: The state on the straight path at the sampled time,
    #         formed directly rather than by simulating the path.
    #     time_values: The sampled path position per row, shared across the
    #         bands of one utterance.
    #     velocity_target: The path velocity, which is the difference of the
    #         two endpoints and therefore independent of the sampled time.
    def __init__(
        self,
        expanded_mel: torch.Tensor,
        band_index: torch.Tensor,
        noisy_state: torch.Tensor,
        time_values: torch.Tensor,
        velocity_target: torch.Tensor
    ) -> None:
        # Stores the five aligned members without copying or validating them;
        # this is a transport record between the flow and the objective rather
        # than a validated configuration.
        self.expanded_mel: torch.Tensor = expanded_mel
        self.band_index: torch.Tensor = band_index
        self.noisy_state: torch.Tensor = noisy_state
        self.time_values: torch.Tensor = time_values
        self.velocity_target: torch.Tensor = velocity_target


class RfwaveRectifiedFlow(nn.Module):
    # Joint-parallel multi-band rectified flow over equalized STFT slabs. This
    # class owns everything about the flow except the network that predicts
    # velocities and the integrator that follows them: the subband geometry
    # and its exact inverse, the construction of both path endpoints, the
    # training-tuple assembly, and the several placements the objective needs.
    #
    # The subband decomposition and its inverse are the delicate part. Slicing
    # pads circularly and overlaps adjacent bands, so a band sees context
    # beyond its own range and the backbone can model structure that straddles
    # a boundary; placement then trims exactly what slicing added, so the two
    # compose to the identity on the spectrum. Because the operations are
    # mutual inverses by construction rather than by approximation, any change
    # to one must be mirrored in the other.
    #
    # Integration: this class is a torch module rather than a plain object
    # because it owns the backbone and the equalizer, whose parameters and
    # running statistics must appear in the module's state. The sampler, by
    # contrast, holds no state and is passed this object at call time.
    def __init__(self, configuration: RfwaveNetworkConfig) -> None:
        # Builds the backbone, the spectral transform pair, and the equalizer,
        # then derives the subband geometry. The bins-per-band figure is a
        # floor division that deliberately excludes the final bin of a real
        # spectrum, which placement restores by giving the last band one extra
        # bin; that asymmetry is what lets an odd bin count be partitioned
        # into equal bands.
        super().__init__()
        self._configuration: RfwaveNetworkConfig = configuration
        self.backbone: RfwaveBackbone = RfwaveBackbone(configuration)
        self.spectral_transform: RfwaveSpectralTransform = RfwaveSpectralTransform(
            n_fft=configuration.n_fft,
            hop_length=configuration.hop_length
        )
        self.equalizer: RfwavePqmfEqualizer = RfwavePqmfEqualizer(
            band_count=configuration.band_count,
            taps=configuration.pqmf_taps,
            cutoff_ratio=configuration.pqmf_cutoff_ratio,
            kaiser_beta=configuration.pqmf_kaiser_beta
        )
        self._band_count: int = configuration.band_count
        self._bins_per_band: int = configuration.n_fft // 2 // configuration.band_count
        self._left_overlap: int = configuration.left_overlap
        self._right_overlap: int = configuration.right_overlap

    @property
    def band_count(self) -> int:
        # Returns the number of frequency subbands.
        return self._band_count

    @property
    def overlap(self) -> int:
        # Returns the total overlap width shared between adjacent bands.
        return self._left_overlap + self._right_overlap

    def gather_joint_subbands(self, stacked_spectrum: torch.Tensor) -> torch.Tensor:
        # Slices the stacked spectrum into overlapping subband slabs for all
        # bands jointly.
        #
        # Args:
        #     stacked_spectrum: Spectrum of shape
        #         ``[batch, 2 * bins, frames]``, the real and imaginary parts
        #         concatenated along the channel axis.
        #
        # Returns:
        #     Slabs of shape ``[batch, band_count * 2 * slab_width, frames]``
        #     ordered band-major, each band contributing its real slab
        #     followed by its imaginary slab.
        complex_stack: torch.Tensor = torch.stack(torch.chunk(stacked_spectrum, 2, dim=1), dim=-1)
        # Padding is applied to the frequency axis only, and circularly: the
        # top of the spectrum supplies the bottom band's lower context and
        # vice versa. The right pad is one short of the overlap width, which
        # together with the extra bin placement gives the final band is what
        # makes the sliding windows tile the odd bin count exactly.
        padded: torch.Tensor = torch.nn.functional.pad(
            complex_stack,
            (0, 0, 0, 0, self._left_overlap, self._right_overlap - 1),
            mode="circular"
        )
        # The stride is the band width while the window is wider by the total
        # overlap, so consecutive bands share bins at their boundary.
        unfolded: torch.Tensor = padded.unfold(1, self._bins_per_band + self.overlap, self._bins_per_band)
        unfolded: torch.Tensor = unfolded.permute(0, 1, 4, 2, 3)
        merged: torch.Tensor = torch.cat([unfolded[..., 0], unfolded[..., 1]], dim=2)
        return merged.reshape(merged.size(0), -1, merged.size(-1))

    def place_joint_subbands(self, joint_slabs: torch.Tensor) -> torch.Tensor:
        # Inverts the joint slicing by trimming overlaps and reassembling the
        # stacked spectrum. Overlapping bins are discarded rather than
        # averaged, so each output bin comes from exactly one band; the
        # overlap exists to give the backbone context, not to be blended back.
        #
        # Args:
        #     joint_slabs: Slabs in the band-major layout the slicing
        #         produces, with the band axis already folded out of the batch
        #         axis by the caller.
        #
        # Returns:
        #     The reassembled spectrum in the concatenated real and imaginary
        #     layout the transform consumes.
        slab_parts: tuple[torch.Tensor, ...] = torch.chunk(joint_slabs, self._band_count * 2, dim=1)
        real_parts: list[torch.Tensor] = []
        imaginary_parts: list[torch.Tensor] = []
        for band_index, (real_slab, imaginary_slab) in enumerate(zip(slab_parts[0::2], slab_parts[1::2])):
            real_parts.append(self._trim_slab(real_slab, band_index))
            imaginary_parts.append(self._trim_slab(imaginary_slab, band_index))
        return torch.cat([torch.cat(real_parts, dim=1), torch.cat(imaginary_parts, dim=1)], dim=1)

    def _trim_slab(self, slab: torch.Tensor, band_index: int) -> torch.Tensor:
        # Trims the overlap context from one subband slab, leaving exactly the
        # bins that band owns. The final band keeps one additional bin,
        # because a real spectrum has one more bin than the band count divides
        # evenly; that bin is the highest frequency, and giving it to the last
        # band is what makes slicing and placement exact inverses.
        if band_index == self._band_count - 1:
            return slab[:, self._left_overlap: slab.size(1) - self._right_overlap + 1]
        return slab[:, self._left_overlap: slab.size(1) - self._right_overlap]

    def split_band_lists(self, joint_slabs: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        # Splits flattened joint predictions into per-band real and imaginary
        # slab lists, retaining the overlap context rather than trimming it.
        # The overlap term needs precisely what placement discards: it
        # penalizes disagreement between what one band predicts for a shared
        # bin and what its neighbor predicts for the same bin, which is only
        # observable before the trim.
        #
        # Returns:
        #     The per-band real slabs and the per-band imaginary slabs, each
        #     list ordered by band.
        stacked: torch.Tensor = joint_slabs.reshape(
            joint_slabs.shape[0] // self._band_count,
            -1,
            joint_slabs.shape[2]
        )
        slab_parts: tuple[torch.Tensor, ...] = torch.chunk(stacked, self._band_count * 2, dim=1)
        return list(slab_parts[0::2]), list(slab_parts[1::2])

    def noise_endpoint(self, mel: torch.Tensor) -> torch.Tensor:
        # Builds the path's starting point as the transform of Gaussian noise.
        #
        # The noise is drawn in the waveform domain and then analyzed, rather
        # than drawn directly as a spectrum. This matters: an independently
        # drawn spectrum would generally not be the transform of any real
        # signal, so the path would begin off the manifold the flow is
        # supposed to travel within. Drawing in the waveform domain guarantees
        # the starting point is consistent by construction.
        #
        # Args:
        #     mel: Conditioning mel, used only for its frame count, device,
        #         and batch size. The noise length is chosen so that analyzing
        #         it yields exactly the mel's frame count.
        #
        # Returns:
        #     The starting state with the band axis folded into the batch
        #     axis, ready to be paired row for row with the other members of a
        #     training tuple.
        frame_count: int = mel.shape[2] - 1
        noise: torch.Tensor = torch.randn(
            [mel.shape[0], self._configuration.hop_length * frame_count],
            device=mel.device
        )
        noise_spectrum: torch.Tensor = self.spectral_transform.stft(noise)
        joint: torch.Tensor = self.gather_joint_subbands(noise_spectrum)
        return joint.reshape(joint.size(0) * self._band_count, joint.size(1) // self._band_count, joint.size(2))

    def target_endpoint(self, waveform: torch.Tensor) -> torch.Tensor:
        # Builds the path's destination as the transform of the equalized
        # target waveform. Equalization runs first because subbands of speech
        # differ enormously in energy, and a flow whose destination varied by
        # orders of magnitude across bands would be dominated by the loudest;
        # normalizing each band puts them on a comparable scale, and the
        # inverse is applied after synthesis.
        #
        # Returns:
        #     The destination state with the band axis folded into the batch
        #     axis, aligned row for row with the starting state.
        equalized: torch.Tensor = self.equalizer.project(waveform)
        spectrum: torch.Tensor = self.spectral_transform.stft(equalized)
        joint: torch.Tensor = self.gather_joint_subbands(spectrum)
        return joint.reshape(joint.size(0) * self._band_count, joint.size(1) // self._band_count, joint.size(2))

    def build_train_tuple(self, mel: torch.Tensor, waveform: torch.Tensor) -> RfwaveTrainTuple:
        # Assembles one training tuple. Both endpoints are constructed, a path
        # position is sampled, and the state at that position is formed
        # directly by convex combination; the path is never simulated, so a
        # training step costs one forward pass no matter how many steps
        # synthesis will use.
        #
        # Args:
        #     mel: Conditioning mel of shape ``[batch, channels, frames]``.
        #     waveform: Target waveform the destination endpoint is built
        #         from.
        #
        # Returns:
        #     The five aligned members of a training tuple, all flattened over
        #     the band axis.
        # Time is drawn once per utterance and repeated across its bands, so a
        # sample's bands sit at one common point on the path. The row ordering
        # this produces must match the band index and mel expansion below,
        # which is why all three are built from the same batch and band
        # counts.
        time_values: torch.Tensor = torch.rand(
            (mel.size(0),),
            device=mel.device
        ).repeat_interleave(self._band_count, 0)
        noise_state: torch.Tensor = self.noise_endpoint(mel)
        target_state: torch.Tensor = self.target_endpoint(waveform)
        # The band index cycles fastest while the mel repeats slowest, which
        # is the row ordering the folded band axis produces; the two
        # expansions are deliberately different functions for that reason.
        band_index: torch.Tensor = torch.tile(
            torch.arange(self._band_count, device=mel.device),
            (mel.size(0),)
        )
        expanded_mel: torch.Tensor = torch.repeat_interleave(mel, self._band_count, 0)
        time_shaped: torch.Tensor = time_values.view(-1, 1, 1)
        # The state interpolates between the endpoints, reaching the noise at
        # time zero and the data at time one. The velocity is their plain
        # difference and carries no dependence on the sampled time, which is
        # exactly what makes the learned field integrable in few steps.
        noisy_state: torch.Tensor = time_shaped * target_state + (1.0 - time_shaped) * noise_state
        velocity_target: torch.Tensor = target_state - noise_state
        return RfwaveTrainTuple(
            expanded_mel=expanded_mel,
            band_index=band_index,
            noisy_state=noisy_state,
            time_values=time_values,
            velocity_target=velocity_target
        )

    def predict_velocity(
        self,
        noisy_state: torch.Tensor,
        time_values: torch.Tensor,
        expanded_mel: torch.Tensor,
        band_index: torch.Tensor
    ) -> torch.Tensor:
        # Runs the backbone over the joint-parallel flow state, predicting the
        # velocity for every band of every sample in one pass. One shared
        # backbone serves all bands, distinguished only by the band index, so
        # parameter count scales with model width rather than band count.
        return self.backbone(noisy_state, time_values, expanded_mel, band_index)

    def waveform_from_joint(self, joint_slabs: torch.Tensor) -> torch.Tensor:
        # Reconstructs the waveform from flattened joint slabs, undoing every
        # transformation the endpoint construction applied: the band axis is
        # unfolded, the overlaps trimmed, the spectrum inverted, and the
        # equalization restored, in that order.
        placed: torch.Tensor = self.place_joint_subbands(
            joint_slabs.reshape(joint_slabs.shape[0] // self._band_count, -1, joint_slabs.shape[2])
        )
        waveform: torch.Tensor = self.spectral_transform.istft(placed)
        return self.equalizer.restore(waveform)

    def project_to_consistent_spectrum(self, joint_slabs: torch.Tensor) -> torch.Tensor:
        # Projects joint slabs onto the set of spectra that are genuinely the
        # transform of some real waveform.
        #
        # Not every array of the right shape is such a spectrum. Overlapping
        # analysis windows make neighboring frames redundant, so an arbitrary
        # array generally corresponds to no signal at all. Inverting the
        # transform and re-analyzing the result is the standard projection
        # onto that set: the inverse resolves the redundancy by overlap-add,
        # and re-analyzing yields the consistent spectrum nearest the input.
        #
        # The sampler applies this to each predicted velocity before stepping,
        # so accumulated inconsistency cannot compound across integration
        # steps into a state the final inverse would have to silently discard.
        placed: torch.Tensor = self.place_joint_subbands(
            joint_slabs.reshape(joint_slabs.shape[0] // self._band_count, -1, joint_slabs.shape[2])
        )
        consistent: torch.Tensor = self.spectral_transform.stft(self.spectral_transform.istft(placed))
        regathered: torch.Tensor = self.gather_joint_subbands(consistent)
        return regathered.reshape(
            regathered.size(0) * self._band_count,
            regathered.size(1) // self._band_count,
            regathered.size(2)
        )

    def velocity_error_waveform(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Places the velocity error back onto the full spectrum and returns
        # its inverse transform, so the flow-matching term is measured in the
        # waveform domain rather than on the slabs directly.
        #
        # Scoring the error after the inverse transform is what makes the
        # penalty reflect audible discrepancy: the transform is not
        # norm-preserving across the overlapping bands, so a spectral-domain
        # penalty would weight bins by an artifact of the decomposition rather
        # than by their contribution to the signal. Because the transform is
        # linear, the inverse of the difference equals the difference of the
        # inverses, and no second reconstruction is needed.
        difference: torch.Tensor = prediction - target
        placed: torch.Tensor = self.place_joint_subbands(
            difference.reshape(difference.shape[0] // self._band_count, -1, difference.shape[2])
        )
        return self.spectral_transform.istft(placed)

    def implied_endpoints(
        self,
        noisy_state: torch.Tensor,
        time_values: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Recovers the destination spectrum each velocity implies, so the
        # magnitude term can compare spectra rather than velocities.
        #
        # The recovery is exact algebra rather than an approximation.
        # Subtracting the time-scaled true velocity from the interpolated
        # state cancels every term involving the destination and leaves the
        # starting point exactly. Adding the true velocity back to that
        # recovers the true destination; adding the predicted velocity instead
        # gives the destination the network's prediction implies. Comparing
        # those two is what turns a velocity error into a spectral error the
        # magnitude term can weigh.
        #
        # Returns:
        #     The implied and the true destination spectra, both already
        #     placed back onto the full spectrum.
        time_shaped: torch.Tensor = time_values.view(-1, 1, 1)
        noise_state: torch.Tensor = noisy_state - time_shaped * target
        predicted_endpoint: torch.Tensor = noise_state + prediction
        target_endpoint: torch.Tensor = noise_state + target
        placed_prediction: torch.Tensor = self.place_joint_subbands(
            predicted_endpoint.reshape(predicted_endpoint.shape[0] // self._band_count, -1, predicted_endpoint.shape[2])
        )
        placed_target: torch.Tensor = self.place_joint_subbands(
            target_endpoint.reshape(target_endpoint.shape[0] // self._band_count, -1, target_endpoint.shape[2])
        )
        return placed_prediction, placed_target

    @property
    def configuration(self) -> RfwaveNetworkConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration
