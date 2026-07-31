# This module:
# 1. Implements the RNDVoC training objective: reconstruction supervision
#    over the decomposed spectral prediction together with the adversarial
#    and feature-matching terms, per the reference recipe
# 2. Implements the omnidirectional anti-wrapping phase loss this family
#    substitutes for the APNet2 three-view phase objective
#
# Published composition:
# - Reconstruction group, shared by training and validation:
#   ``45 * MSE(log-amplitude) + 100 * phase + 45 * consistency +
#   45 * (L1(real) + L1(imaginary)) + 45 * L1(mel)``
# - Generator: the reconstruction group plus
#   ``1 * (hinge_period + 1 * hinge_resolution) +
#   1 * (FM_period + 1 * FM_resolution)``
# - Discriminator: ``hinge_period + 1 * hinge_resolution``
# - Validation: the reconstruction group alone
#
# Update semantics:
# - Two-phase: the driving model computes the discriminator scalar against
#   the detached synthesis and steps the discriminator optimizer, then
#   computes the generator scalar through the same discriminators. This
#   class supplies all three scalars and owns no optimizer state
#
# Report alignment:
# - The report records the APNet2, FreeV, and RNDVoC Project-Trained
#   Configurations as adding log-amplitude, anti-wrapping phase, and
#   STFT-consistency terms with real and imaginary reconstruction to mel,
#   hinge adversarial, and feature-matching supervision. This family
#   realizes that description with an omnidirectional phase term in place of
#   the APNet2 axis-aligned pair, and with consistency and real-imaginary
#   error carried as separately weighted terms
#
# Design decisions:
# - The reconstruction group is computed by one private method that both
#   the training and validation paths call, so the two can never drift
#   apart in what they measure; only the adversarial and feature-matching
#   terms distinguish them
# - Phase is supervised omnidirectionally rather than through the APNet2
#   pair of axis-aligned derivative views: one convolution compares each
#   time-frequency position against all eight of its neighbours at once, so
#   diagonal structure in the phase surface is constrained as directly as
#   horizontal and vertical structure
# - Consistency and the real-imaginary error are separate weighted terms
#   here, whereas APNet2 folds them into one spectrum term; this family can
#   therefore trade spectral realizability against pointwise spectral
#   accuracy directly
# - The multi-resolution ensemble enters at full strength rather than the
#   one-tenth weight of the APNet2 and Vocos recipes, so both ensembles
#   contribute equally on both sides of the game under the defaults
#
# Author: Rahul Sawhney

from typing import ClassVar, cast

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat
from torch import nn
from torch.nn import functional as F

from vocode.losses.apnet2 import Apnet2Spectrum, Apnet2SpectrumAnalyzer
from vocode.losses.feature_matching import FeatureMatchingLoss
from vocode.losses.mel_reconstruction import MelReconstructionLoss

__all__: list[str] = ["RndvocLoss", "RndvocLossConfig", "RndvocOmniPhaseLoss"]


class RndvocLossConfig(BaseModel):
    # Frozen loss weights of the reference composition.
    #
    # Fields:
    #     amplitude_weight: Multiplier on the mean squared error between
    #         the reference and predicted log amplitude spectra.
    #         Default: ``45.0``.
    #     phase_weight: Multiplier on the omnidirectional anti-wrapping
    #         phase term. Default: ``100.0``.
    #     consistency_weight: Multiplier on the STFT consistency term,
    #         which measures whether the predicted complex spectrum
    #         corresponds to a realizable waveform. Default: ``45.0``.
    #     real_imaginary_weight: Multiplier on the summed L1 errors of the
    #         real and imaginary spectra. Unlike APNet2, where the
    #         equivalent weight sits inside a combined spectrum term, this
    #         one applies directly to the total. Default: ``45.0``.
    #     adversarial_weight: Multiplier on the combined hinge term of both
    #         ensembles. Default: ``1.0``.
    #     feature_matching_weight: Multiplier on the combined
    #         feature-matching term of both ensembles. Default: ``1.0``.
    #     mel_weight: Multiplier on the L1 log-mel reconstruction term,
    #         applied on both the training and validation paths.
    #         Default: ``45.0``.
    #     resolution_discriminator_weight: Multiplier on the
    #         multi-resolution ensemble wherever it appears: the
    #         discriminator separation term, the generator's adversarial
    #         term, and the generator's feature-matching term. At the
    #         default of one the two ensembles contribute equally.
    #         Default: ``1.0``.
    #
    # Every field is a PositiveFloat on a strict, extra-forbidding, frozen
    # model, so a weight may be attenuated but never zeroed, negated, or
    # misspelled into silent acceptance.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    amplitude_weight: PositiveFloat = 45.0
    phase_weight: PositiveFloat = 100.0
    consistency_weight: PositiveFloat = 45.0
    real_imaginary_weight: PositiveFloat = 45.0
    adversarial_weight: PositiveFloat = 1.0
    feature_matching_weight: PositiveFloat = 1.0
    mel_weight: PositiveFloat = 45.0
    resolution_discriminator_weight: PositiveFloat = 1.0


class RndvocOmniPhaseLoss(nn.Module):
    # Omnidirectional anti-wrapping phase objective. Where the APNet2 phase
    # loss supervises three views built from axis-aligned difference
    # matrices, this one supervises nine views at once through a single
    # convolution: the phase value itself and its difference against each
    # of the eight neighbours surrounding it in the time-frequency plane.
    # Diagonal structure in the phase surface is therefore constrained as
    # directly as horizontal and vertical structure.
    #
    # Integration: constructed by RndvocLoss and exported publicly so the
    # RNDVoC model and its tests can exercise the phase term in isolation.
    def __init__(self) -> None:
        # Builds the nine 3x3 difference kernels and the full-turn
        # constant. Each kernel carries a one at the centre tap; the eight
        # kernels whose offset is not the centre additionally carry a minus
        # one at that offset, so they compute centre-minus-neighbour, while
        # the centre kernel passes the value through unchanged and yields
        # the instantaneous error. The stack is registered as a persistent
        # buffer, so it follows the module across devices and dtypes and
        # appears in the state dict, unlike the APNet2 phase loss which
        # rebuilds its matrices on every call.
        super().__init__()
        self._two_pi: float = 6.283185307179586
        kernels: torch.Tensor = torch.zeros(9, 1, 3, 3)
        neighbor_offsets: list[tuple[int, int]] = [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2)
        ]
        kernel_index: int
        offset: tuple[int, int]
        for kernel_index, offset in enumerate(neighbor_offsets):
            row: int = offset[0]
            column: int = offset[1]
            kernels[kernel_index, 0, 1, 1] = 1.0
            if (row, column) != (1, 1):
                kernels[kernel_index, 0, row, column] = -1.0
        self.register_buffer("_kernels", kernels)

    def forward(self, reference_phase: torch.Tensor, candidate_phase: torch.Tensor) -> torch.Tensor:
        # Computes this criterion's value for the given inputs. Both phase
        # surfaces are cropped to their common extent, transposed so the
        # frame axis becomes the convolution's height and the frequency axis
        # its width, and passed through the nine-kernel stack with unit
        # padding. Subtracting the two responses gives, for every position
        # and every one of the nine directions, the error of that
        # directional phase relation; the anti-wrapping projection then
        # bounds each error by pi before the mean is taken.
        #
        # Args:
        #     reference_phase: Principal-value phase of the reference
        #         audio, laid out as ``[batch, bins, frames]``.
        #     candidate_phase: The predicted phase in the same layout.
        #
        # Returns:
        #     A scalar tensor holding three times the mean anti-wrapped
        #     directional error. The factor of three is a fixed scale of the
        #     reference recipe, applied before the configured phase weight,
        #     so the effective multiplier on this term is three hundred
        #     under the default configuration.
        #
        # Note:
        #     Unit padding means positions on the border of the
        #     time-frequency plane compare against zero-valued neighbours
        #     that lie outside the surface. Both the reference and the
        #     candidate are padded identically, so the border contributions
        #     cancel wherever the two surfaces already agree.
        reference_phase, candidate_phase = self._crop_pair(reference_phase, candidate_phase)
        reference_planes: torch.Tensor = reference_phase.transpose(-2, -1).unsqueeze(1)
        candidate_planes: torch.Tensor = candidate_phase.transpose(-2, -1).unsqueeze(1)
        kernels: torch.Tensor = cast(torch.Tensor, self._kernels).to(dtype=reference_planes.dtype)
        reference_differences: torch.Tensor = F.conv2d(reference_planes, kernels, bias=None, stride=1, padding=1)
        candidate_differences: torch.Tensor = F.conv2d(candidate_planes, kernels, bias=None, stride=1, padding=1)
        return 3.0 * torch.mean(self._anti_wrap(candidate_differences - reference_differences))

    def _anti_wrap(self, phase_difference: torch.Tensor) -> torch.Tensor:
        # Applies phase anti-wrapping so angular differences remain numerically stable.
        # Rounding the difference to the nearest whole turn and subtracting
        # that turn projects any angular error into the interval around
        # zero, and the absolute value then bounds it by pi. A difference of
        # very nearly two pi therefore costs almost nothing, which is
        # correct because it describes the same angle.
        return torch.abs(
            phase_difference - torch.round(phase_difference / self._two_pi) * self._two_pi
        )

    def _crop_pair(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Crops paired tensors to a shared shape before loss computation.
        # The two trailing axes are truncated to their common extent, so a
        # predicted spectrum carrying one boundary frame more or fewer than
        # the analyzed reference remains comparable. This is deliberately
        # more permissive than the mel reconstruction guard, which rejects
        # any disagreement: a spectral off-by-one is a framing boundary
        # effect, whereas a mel disagreement indicates the two spectrograms
        # were extracted under different protocols.
        minimum_bins: int = min(first.shape[-2], second.shape[-2])
        minimum_frames: int = min(first.shape[-1], second.shape[-1])
        return first[..., :minimum_bins, :minimum_frames], second[..., :minimum_bins, :minimum_frames]


class _RndvocHingeGanLoss(nn.Module):
    # Private hinge adversarial objective for the RNDVoC composition,
    # holding both halves of the game over ensemble logit lists. Like the
    # APNet2 hinge and unlike the Vocos one, the per-member terms are summed
    # rather than averaged, so ensemble balance is governed entirely by the
    # configured resolution weight. The class holds no parameters and no
    # mutable state.
    def compute_generator_loss(self, fake_logits: list[torch.Tensor]) -> torch.Tensor:
        # Computes the generator objective for one GAN training step by
        # accumulating ``mean(clamp(1 - logit, min=0))`` over the ensemble.
        # A member whose logits already reach one contributes nothing, so
        # the term saturates instead of rewarding ever-larger logits.
        #
        # Args:
        #     fake_logits: One logit tensor per sub-discriminator, evaluated
        #         on synthesized audio.
        #
        # Returns:
        #     A scalar tensor holding the summed hinge term, placed on the
        #     device of the first logit. An empty list returns a CPU zero
        #     scalar.
        loss: torch.Tensor = torch.zeros((), device=fake_logits[0].device) if fake_logits else torch.zeros(())
        fake_logit: torch.Tensor
        for fake_logit in fake_logits:
            loss: torch.Tensor = loss + torch.mean(torch.clamp(1.0 - fake_logit, min=0.0))
        return loss

    def compute_discriminator_loss(
        self,
        real_logits: list[torch.Tensor],
        fake_logits: list[torch.Tensor]
    ) -> torch.Tensor:
        # Computes the discriminator objective for one GAN training step.
        # Each aligned pair contributes ``mean(clamp(1 - real, min=0)) +
        # mean(clamp(1 + fake, min=0))``, so a member pays nothing once it
        # scores reference audio at or above one and synthesized audio at or
        # below minus one.
        #
        # Args:
        #     real_logits: One logit tensor per sub-discriminator, evaluated
        #         on reference audio; also fixes the accumulator's device.
        #     fake_logits: The matching logit tensors evaluated on the
        #         detached synthesis.
        #
        # Raises:
        #     ValueError: If the two lists disagree in length; the pairing
        #         uses strict zip, so a member-count mismatch fails closed.
        #
        # Returns:
        #     A scalar tensor holding the summed separation term. An empty
        #     real_logits list returns a CPU zero scalar.
        loss: torch.Tensor = torch.zeros((), device=real_logits[0].device) if real_logits else torch.zeros(())
        real_logit: torch.Tensor
        fake_logit: torch.Tensor
        for real_logit, fake_logit in zip(real_logits, fake_logits, strict=True):
            real_loss: torch.Tensor = torch.mean(torch.clamp(1.0 - real_logit, min=0.0))
            fake_loss: torch.Tensor = torch.mean(torch.clamp(1.0 + fake_logit, min=0.0))
            loss: torch.Tensor = loss + real_loss + fake_loss
        return loss


class RndvocLoss(nn.Module):
    # The RNDVoC composite objective over reconstruction and adversarial
    # terms. The five reconstruction terms are factored into one private
    # method that both the training and validation paths call, so the two
    # can never disagree about what reconstruction means; only the
    # adversarial and feature-matching terms separate them. The instance
    # owns the frozen weight record and four component losses, and registers
    # no parameters of its own, though the omnidirectional phase loss it
    # holds does contribute a persistent kernel buffer to the state dict.
    #
    # Integration: the driving model constructs one instance from its
    # configuration, calls compute_discriminator_loss and then
    # compute_generator_loss once each per training batch under its own
    # optimizer toggles, and calls compute_validation_loss on the
    # evaluation path. The model obtains its spectrum analyzer through the
    # spectrum_analyzer_type property rather than importing it directly.
    def __init__(self, configuration: RndvocLossConfig) -> None:
        # Binds the frozen weight record and constructs the omnidirectional
        # phase, hinge, feature-matching, and mel component losses. All but
        # the phase loss are stateless; the phase loss carries its kernel
        # stack as a buffer.
        #
        # Args:
        #     configuration: The frozen weight record, retained by
        #         reference and republished unchanged through the
        #         configuration property.
        super().__init__()
        self._configuration: RndvocLossConfig = configuration
        self._omni_phase_loss: RndvocOmniPhaseLoss = RndvocOmniPhaseLoss()
        self._hinge_loss: _RndvocHingeGanLoss = _RndvocHingeGanLoss()
        self._feature_matching_loss: FeatureMatchingLoss = FeatureMatchingLoss()
        self._mel_reconstruction_loss: MelReconstructionLoss = MelReconstructionLoss()

    def compute_discriminator_loss(
        self,
        real_period_logits: list[torch.Tensor],
        fake_period_logits: list[torch.Tensor],
        real_resolution_logits: list[torch.Tensor],
        fake_resolution_logits: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes the discriminator objective for one GAN training step as
        # the period ensemble's hinge separation plus the resolution
        # ensemble's, the latter scaled by the configured resolution weight.
        # At the default weight of one the two ensembles train at equal
        # strength, unlike the APNet2 and Vocos recipes that attenuate the
        # resolution ensemble to a tenth.
        #
        # Args:
        #     real_period_logits: Multi-period logits for the reference
        #         audio.
        #     fake_period_logits: Multi-period logits for the synthesis,
        #         which the caller detaches before the discriminator forward
        #         pass so this term cannot reach the generator.
        #     real_resolution_logits: Multi-resolution logits for the
        #         reference audio.
        #     fake_resolution_logits: Multi-resolution logits for the
        #         detached synthesis.
        #
        # Raises:
        #     ValueError: If either ensemble's real and fake lists disagree
        #         in length, raised by the strict pairing inside the hinge
        #         component.
        #
        # Returns:
        #     The weighted total as a graph-carrying scalar tensor, paired
        #     with a component panel of detached Python floats keyed
        #     discriminator_loss_total, discriminator_loss_period, and
        #     discriminator_loss_resolution; the two per-ensemble entries
        #     are reported before the resolution weight is applied.
        period_loss: torch.Tensor = self._hinge_loss.compute_discriminator_loss(real_period_logits, fake_period_logits)
        resolution_loss: torch.Tensor = self._hinge_loss.compute_discriminator_loss(
            real_resolution_logits,
            fake_resolution_logits
        )
        total_loss: torch.Tensor = period_loss + self._configuration.resolution_discriminator_weight * resolution_loss
        components: dict[str, float] = {
            "discriminator_loss_total": float(total_loss.detach().item()),
            "discriminator_loss_period": float(period_loss.detach().item()),
            "discriminator_loss_resolution": float(resolution_loss.detach().item())
        }
        return total_loss, components

    def compute_generator_loss(
        self,
        reference_spectrum: Apnet2Spectrum,
        candidate_log_amplitude: torch.Tensor,
        candidate_phase: torch.Tensor,
        candidate_real_spectrum: torch.Tensor,
        candidate_imaginary_spectrum: torch.Tensor,
        final_spectrum: Apnet2Spectrum,
        reference_mel: torch.Tensor,
        candidate_mel: torch.Tensor,
        fake_period_logits: list[torch.Tensor],
        fake_resolution_logits: list[torch.Tensor],
        real_period_features: list[list[torch.Tensor]],
        fake_period_features: list[list[torch.Tensor]],
        real_resolution_features: list[list[torch.Tensor]],
        fake_resolution_features: list[list[torch.Tensor]]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes the generator objective for one GAN training step. The
        # five weighted reconstruction terms are delegated to the shared
        # private method, so this method's own work is the adversarial half:
        # each ensemble contributes a hinge and a feature-matching term, the
        # resolution ensemble's share of each is scaled by the resolution
        # weight, and the two resulting groups are scaled by the adversarial
        # and feature-matching weights before being added to the
        # reconstruction total.
        #
        # Args:
        #     reference_spectrum: Analyzed STFT views of the reference
        #         audio, supplying the amplitude, phase, real, and
        #         imaginary regression targets.
        #     candidate_log_amplitude: The predicted log amplitude stream.
        #     candidate_phase: The predicted phase stream.
        #     candidate_real_spectrum: Real part of the complex spectrum
        #         recombined from the two predicted streams.
        #     candidate_imaginary_spectrum: Imaginary part of that same
        #         recombined spectrum.
        #     final_spectrum: Re-analysis of the waveform actually
        #         synthesized from the recombined spectrum; the consistency
        #         term compares against this, so the two must come from the
        #         same synthesis.
        #     reference_mel: Log-mel spectrogram of the reference audio.
        #     candidate_mel: Log-mel spectrogram of the synthesis; must
        #         match reference_mel in shape exactly.
        #     fake_period_logits: Multi-period logits for the synthesis,
        #         carrying gradient back to the generator.
        #     fake_resolution_logits: Multi-resolution logits for the
        #         synthesis.
        #     real_period_features: Multi-period intermediate activations
        #         for the reference audio; detached inside the matching
        #         term.
        #     fake_period_features: The corresponding activations for the
        #         synthesis.
        #     real_resolution_features: Multi-resolution intermediate
        #         activations for the reference audio; likewise detached.
        #     fake_resolution_features: The corresponding activations for
        #         the synthesis.
        #
        # Raises:
        #     ValueError: If reference_mel and candidate_mel disagree in
        #         shape. The spectral arguments carry no such requirement
        #         because they are cropped to their common extent.
        #
        # Returns:
        #     The weighted total as a graph-carrying scalar tensor, paired
        #     with an eight-entry component panel: the five reconstruction
        #     entries produced by the shared method, plus
        #     generator_loss_total, generator_loss_adversarial, and
        #     generator_loss_feature_matching. The two adversarial entries
        #     are reported after their per-ensemble resolution weighting but
        #     before the outer adversarial and feature-matching weights.
        reconstruction_total: torch.Tensor
        reconstruction_components: dict[str, float]
        reconstruction_total, reconstruction_components = self._compute_reconstruction_terms(
            reference_spectrum=reference_spectrum,
            candidate_log_amplitude=candidate_log_amplitude,
            candidate_phase=candidate_phase,
            candidate_real_spectrum=candidate_real_spectrum,
            candidate_imaginary_spectrum=candidate_imaginary_spectrum,
            final_spectrum=final_spectrum,
            reference_mel=reference_mel,
            candidate_mel=candidate_mel
        )
        adversarial_period: torch.Tensor = self._hinge_loss.compute_generator_loss(fake_period_logits)
        adversarial_resolution: torch.Tensor = self._hinge_loss.compute_generator_loss(fake_resolution_logits)
        feature_matching_period: torch.Tensor = self._feature_matching_loss(real_period_features, fake_period_features)
        feature_matching_resolution: torch.Tensor = self._feature_matching_loss(
            real_resolution_features,
            fake_resolution_features
        )
        adversarial_loss: torch.Tensor = (
            adversarial_period + self._configuration.resolution_discriminator_weight * adversarial_resolution
        )
        feature_matching_loss: torch.Tensor = (
            feature_matching_period + self._configuration.resolution_discriminator_weight * feature_matching_resolution
        )
        total_loss: torch.Tensor = (
            reconstruction_total
            + self._configuration.adversarial_weight * adversarial_loss
            + self._configuration.feature_matching_weight * feature_matching_loss
        )
        components: dict[str, float] = {
            **reconstruction_components,
            "generator_loss_total": float(total_loss.detach().item()),
            "generator_loss_adversarial": float(adversarial_loss.detach().item()),
            "generator_loss_feature_matching": float(feature_matching_loss.detach().item())
        }
        return total_loss, components

    def compute_validation_loss(
        self,
        reference_spectrum: Apnet2Spectrum,
        candidate_log_amplitude: torch.Tensor,
        candidate_phase: torch.Tensor,
        candidate_real_spectrum: torch.Tensor,
        candidate_imaginary_spectrum: torch.Tensor,
        final_spectrum: Apnet2Spectrum,
        reference_mel: torch.Tensor,
        candidate_mel: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes validation loss without mutating optimizer state. The
        # returned total is exactly the shared reconstruction group with no
        # adversarial participation at all, so the reported value is
        # comparable across checkpoints without depending on how well the
        # discriminators happen to be trained at that moment. Because the
        # same private method produces the training path's reconstruction
        # total, a validation figure is directly comparable against the
        # reconstruction share of the training figure.
        #
        # Args:
        #     reference_spectrum: Analyzed STFT views of the reference
        #         audio.
        #     candidate_log_amplitude: The predicted log amplitude stream.
        #     candidate_phase: The predicted phase stream.
        #     candidate_real_spectrum: Real part of the recombined complex
        #         spectrum.
        #     candidate_imaginary_spectrum: Imaginary part of that spectrum.
        #     final_spectrum: Re-analysis of the synthesized waveform,
        #         against which consistency is measured.
        #     reference_mel: Log-mel spectrogram of the reference audio.
        #     candidate_mel: Log-mel spectrogram of the synthesis; must
        #         match reference_mel in shape exactly.
        #
        # Raises:
        #     ValueError: If reference_mel and candidate_mel disagree in
        #         shape; the same guard applies on the evaluation path as in
        #         training.
        #
        # Returns:
        #     The reconstruction total as a scalar tensor, paired with a
        #     six-entry component panel keyed loss_total, loss_amplitude,
        #     loss_phase, loss_consistency, loss_real_imaginary, and
        #     loss_mel. The five per-term entries are the shared method's
        #     values republished under validation names, so the two paths
        #     never collide in the logged metric buffer.
        total_loss: torch.Tensor
        reconstruction_components: dict[str, float]
        total_loss, reconstruction_components = self._compute_reconstruction_terms(
            reference_spectrum=reference_spectrum,
            candidate_log_amplitude=candidate_log_amplitude,
            candidate_phase=candidate_phase,
            candidate_real_spectrum=candidate_real_spectrum,
            candidate_imaginary_spectrum=candidate_imaginary_spectrum,
            final_spectrum=final_spectrum,
            reference_mel=reference_mel,
            candidate_mel=candidate_mel
        )
        components: dict[str, float] = {
            "loss_total": float(total_loss.detach().item()),
            "loss_amplitude": reconstruction_components["generator_loss_amplitude"],
            "loss_phase": reconstruction_components["generator_loss_phase"],
            "loss_consistency": reconstruction_components["generator_loss_consistency"],
            "loss_real_imaginary": reconstruction_components["generator_loss_real_imaginary"],
            "loss_mel": reconstruction_components["generator_loss_mel"]
        }
        return total_loss, components

    def _compute_reconstruction_terms(
        self,
        reference_spectrum: Apnet2Spectrum,
        candidate_log_amplitude: torch.Tensor,
        candidate_phase: torch.Tensor,
        candidate_real_spectrum: torch.Tensor,
        candidate_imaginary_spectrum: torch.Tensor,
        final_spectrum: Apnet2Spectrum,
        reference_mel: torch.Tensor,
        candidate_mel: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes the weighted spectral reconstruction terms shared by training and validation.
        # Five terms are formed and summed under their configured weights:
        # the mean squared log-amplitude error, the omnidirectional phase
        # error, the STFT consistency between the predicted spectrum and the
        # re-analysis of the waveform synthesized from it, the combined L1
        # error of the real and imaginary components, and the L1 log-mel
        # error. Factoring these here is what guarantees the training and
        # validation paths measure reconstruction identically.
        #
        # Raises:
        #     ValueError: If reference_mel and candidate_mel disagree in
        #         shape, raised by the mel reconstruction guard. The mel
        #         term is evaluated last, so the four spectral terms are
        #         already computed when this fires.
        #
        # Returns:
        #     The weighted reconstruction total as a graph-carrying scalar
        #     tensor, paired with a five-entry panel of detached floats
        #     under generator-prefixed names. The validation path renames
        #     these entries; the training path merges them into its own
        #     panel unchanged.
        amplitude_loss: torch.Tensor = F.mse_loss(
            *self._crop_pair(reference_spectrum.log_amplitude, candidate_log_amplitude)
        )
        phase_loss: torch.Tensor = self._omni_phase_loss(reference_spectrum.phase, candidate_phase)
        consistency_loss: torch.Tensor = self._stft_consistency_loss(
            candidate_real_spectrum,
            final_spectrum.real_spectrum,
            candidate_imaginary_spectrum,
            final_spectrum.imaginary_spectrum
        )
        real_loss: torch.Tensor = F.l1_loss(
            *self._crop_pair(reference_spectrum.real_spectrum, candidate_real_spectrum)
        )
        imaginary_loss: torch.Tensor = F.l1_loss(
            *self._crop_pair(reference_spectrum.imaginary_spectrum, candidate_imaginary_spectrum)
        )
        real_imaginary_loss: torch.Tensor = real_loss + imaginary_loss
        mel_loss: torch.Tensor = self._mel_reconstruction_loss(reference_mel, candidate_mel)
        total_loss: torch.Tensor = (
            self._configuration.amplitude_weight * amplitude_loss
            + self._configuration.phase_weight * phase_loss
            + self._configuration.consistency_weight * consistency_loss
            + self._configuration.real_imaginary_weight * real_imaginary_loss
            + self._configuration.mel_weight * mel_loss
        )
        components: dict[str, float] = {
            "generator_loss_amplitude": float(amplitude_loss.detach().item()),
            "generator_loss_phase": float(phase_loss.detach().item()),
            "generator_loss_consistency": float(consistency_loss.detach().item()),
            "generator_loss_real_imaginary": float(real_imaginary_loss.detach().item()),
            "generator_loss_mel": float(mel_loss.detach().item())
        }
        return total_loss, components

    def _stft_consistency_loss(
        self,
        first_real: torch.Tensor,
        second_real: torch.Tensor,
        first_imaginary: torch.Tensor,
        second_imaginary: torch.Tensor
    ) -> torch.Tensor:
        # Computes the STFT consistency term as the squared complex distance
        # between the predicted spectrum and the re-analysis of the waveform
        # synthesized from it. An arbitrary array of complex numbers is not
        # in general the STFT of any real signal, because overlapping frames
        # must agree; this term measures exactly that discrepancy, so
        # minimizing it pushes the predicted spectrum toward the realizable
        # set and keeps the amplitude and phase streams mutually coherent.
        # The reduction averages within each batch element before averaging
        # across the batch, so every utterance carries equal weight.
        #
        # Args:
        #     first_real: Real part of the predicted spectrum.
        #     second_real: Real part of the re-analyzed synthesis.
        #     first_imaginary: Imaginary part of the predicted spectrum.
        #     second_imaginary: Imaginary part of the re-analyzed synthesis.
        #
        # Returns:
        #     A scalar tensor holding the mean squared complex distance,
        #     computed after both pairs are cropped to their common extent.
        first_real, second_real = self._crop_pair(first_real, second_real)
        first_imaginary, second_imaginary = self._crop_pair(first_imaginary, second_imaginary)
        return torch.mean(
            torch.mean((first_real - second_real).pow(2) + (first_imaginary - second_imaginary).pow(2), dim=(1, 2))
        )

    def _crop_pair(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Crops paired tensors to a shared shape before loss computation.
        # The two trailing axes are truncated to their common extent, so a
        # predicted spectrum carrying one boundary frame more or fewer than
        # the analyzed reference remains comparable. This is deliberately
        # more permissive than the mel reconstruction guard, which rejects
        # any disagreement: a spectral off-by-one is a framing boundary
        # effect, whereas a mel disagreement indicates the two spectrograms
        # were extracted under different protocols.
        minimum_bins: int = min(first.shape[-2], second.shape[-2])
        minimum_frames: int = min(first.shape[-1], second.shape[-1])
        return first[..., :minimum_bins, :minimum_frames], second[..., :minimum_bins, :minimum_frames]

    @property
    def configuration(self) -> RndvocLossConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    @property
    def spectrum_analyzer_type(self) -> type[Apnet2SpectrumAnalyzer]:
        # Returns the spectrum analyzer type shared with the APNet2 loss family.
        # Publishing the class rather than an instance lets the driving
        # model construct the analyzer with its own framing settings while
        # still guaranteeing that both families analyze audio through the
        # same transform.
        return Apnet2SpectrumAnalyzer
