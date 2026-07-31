# This module:
# 1. Implements the HiFTNet training objective: least-squares adversarial
#    and feature-matching terms plus the mel reconstruction term of the
#    reference source-filter recipe
# 2. Implements the truncated relative least-squares term that this family
#    adds on top of the plain least-squares game
#
# Report alignment:
# - This family is not one of the twelve Project-Trained Configurations. No
#   admissible project-trained checkpoint was produced for it, so Study 1
#   retains it as a documented exclusion rather than a thirteenth result,
#   and no measurement in the report or the study surfaces derives from this
#   objective. The implementation is retained because the exclusion is
#   documented rather than deleted
#
# Published composition:
# - Generator: ``LSGAN_period + LSGAN_spectrogram + TPRLS_period +
#   TPRLS_spectrogram + 2 * (FM_period + FM_spectrogram) + 45 * L1(mel)``
# - Discriminator: ``LSGAN_period + LSGAN_spectrogram + TPRLS_period +
#   TPRLS_spectrogram``, with no relative weighting between the ensembles
#   or between the two adversarial forms
# - Validation: ``45 * L1(mel)`` alone, with no adversarial participation
#
# Update semantics:
# - Two-phase: the driving model computes the discriminator scalar against
#   the detached synthesis and steps the discriminator optimizer, then
#   computes the generator scalar through the same discriminators. The
#   TPRLS term appears on both phases and consumes the real logits on both,
#   so the generator path receives real logits as well as fake ones, unlike
#   the plain least-squares generator term
#
# Design decisions:
# - The second ensemble is a spectrogram discriminator rather than the
#   HiFi-GAN multi-scale or BigVGAN multi-resolution ensemble, which is the
#   structural signature of this source-filter family
# - The truncated relative least-squares term concentrates the adversarial
#   pressure on the hardest-ranked positions: for each ensemble member it
#   shifts the fake logits by the median of the real-minus-fake difference,
#   keeps only the positions where the real logit still falls below that
#   shifted value, and penalizes the squared residual on that subset alone
# - Each member's contribution is capped, because ``tau - relu(tau - value)``
#   is exactly ``min(value, tau)``; no single sub-discriminator can dominate
#   the objective through one badly ranked logit map, and the ceiling is the
#   configured tprls_tau
# - A member whose selection mask is empty is skipped entirely rather than
#   contributing the cap, so an already well-ranked member drops out of the
#   term instead of adding a constant
#
# Author: Rahul Sawhney

from typing import ClassVar

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat
from torch import nn

from vocode.losses.adversarial import LeastSquaresGanLoss
from vocode.losses.feature_matching import FeatureMatchingLoss
from vocode.losses.mel_reconstruction import MelReconstructionLoss

__all__: list[str] = ["HiftnetLoss", "HiftnetLossConfig"]


class HiftnetLossConfig(BaseModel):
    # Frozen loss weights of the reference composition.
    #
    # Fields:
    #     mel_weight: Multiplier on the L1 log-mel reconstruction term,
    #         applied on both the training and validation paths.
    #         Default: ``45.0``.
    #     feature_matching_weight: Multiplier applied separately to each
    #         ensemble's feature-matching term. Default: ``2.0``.
    #     tprls_tau: Per-member ceiling on the truncated relative
    #         least-squares contribution. This is a saturation threshold
    #         rather than a scaling weight: a member's term is
    #         ``min(residual, tprls_tau)``, so raising it admits more
    #         pressure from badly ranked members and lowering it caps them
    #         sooner. Default: ``0.04``.
    #
    # Every field is a PositiveFloat on a strict, extra-forbidding, frozen
    # model, so the TPRLS ceiling can be tightened or relaxed but never set
    # to zero, which would silently remove the term.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    mel_weight: PositiveFloat = 45.0
    feature_matching_weight: PositiveFloat = 2.0
    tprls_tau: PositiveFloat = 0.04


class _TopKRelativeLeastSquaresLoss(nn.Module):
    # Private truncated relative least-squares term of the HiFTNet
    # composition. Where the plain least-squares objective penalizes every
    # position of every logit map against a fixed target, this term is
    # relative and selective: it measures each ensemble member against its
    # own median real-minus-fake offset and charges only the positions that
    # remain misranked after that offset is removed. The class holds the
    # saturation threshold and no other state, and registers no parameters.
    def __init__(self, tau: float) -> None:
        # Binds the per-member saturation threshold.
        #
        # Args:
        #     tau: Ceiling on each member's contribution; the accumulation
        #         adds ``min(residual, tau)`` for every member that has at
        #         least one misranked position.
        super().__init__()
        self._tau: float = tau

    def forward(self, real_logits: list[torch.Tensor], fake_logits: list[torch.Tensor]) -> torch.Tensor:
        # Computes the truncated relative term over the ensemble in three
        # stages per member. First the median of the real-minus-fake
        # difference is taken as that member's global offset. Second, a mask
        # selects the positions where the real logit still falls below the
        # fake logit shifted by that offset, which are the positions the
        # discriminator has ranked worst. Third, the mean squared residual
        # over the masked subset is added under the ``min(residual, tau)``
        # ceiling, expressed as ``tau - relu(tau - residual)``. A member
        # whose mask is empty is skipped and contributes nothing.
        #
        # Args:
        #     real_logits: One logit tensor per sub-discriminator, evaluated
        #         on reference audio; also fixes the accumulator's device.
        #     fake_logits: The matching logit tensors evaluated on
        #         synthesized audio.
        #
        # Raises:
        #     ValueError: If the two lists disagree in length; the pairing
        #         uses strict zip so a member-count mismatch fails closed.
        #
        # Returns:
        #     A scalar tensor holding the accumulated capped residuals,
        #     bounded above by ``tau`` times the member count. An empty
        #     real_logits list returns a CPU zero scalar.
        loss: torch.Tensor = torch.zeros((), device=real_logits[0].device) if real_logits else torch.zeros(())
        for real_logit, fake_logit in zip(real_logits, fake_logits, strict=True):
            median_difference: torch.Tensor = torch.median(real_logit - fake_logit)
            mask: torch.Tensor = real_logit < fake_logit + median_difference
            if not bool(mask.any()):
                continue
            relative_loss: torch.Tensor = torch.mean(
                ((real_logit - fake_logit - median_difference) ** 2)[mask]
            )
            loss: torch.Tensor = loss + self._tau - torch.nn.functional.relu(self._tau - relative_loss)
        return loss


class HiftnetLoss(nn.Module):
    # The HiFTNet composite objective: adversarial, feature-matching, and
    # mel terms for the generator; least-squares separation for the
    # discriminators. Both phases additionally carry the truncated relative
    # least-squares term over each ensemble. The instance owns the frozen
    # weight record and four stateless component losses, and registers no
    # parameters of its own, so despite subclassing nn.Module it contributes
    # nothing to the driving model's optimizer state.
    #
    # Integration: the driving model constructs one instance from its
    # configuration, calls compute_discriminator_loss and then
    # compute_generator_loss once each per training batch under its own
    # optimizer toggles, and calls compute_validation_loss on the
    # evaluation path. Unlike the sibling families this class publishes no
    # configuration property; the driving model retains its own reference to
    # the record it constructed.
    def __init__(self, configuration: HiftnetLossConfig) -> None:
        # Binds the frozen weight record and constructs the four component
        # losses. The truncated relative term is the only one parameterized
        # at construction, receiving the configured saturation threshold;
        # the other three are stateless and serve both ensembles.
        #
        # Args:
        #     configuration: The frozen weight record, retained by
        #         reference so the weights a run logs are exactly the
        #         weights it applies.
        super().__init__()
        self._configuration: HiftnetLossConfig = configuration
        self._least_squares_gan: LeastSquaresGanLoss = LeastSquaresGanLoss()
        self._feature_matching_loss: FeatureMatchingLoss = FeatureMatchingLoss()
        self._mel_reconstruction_loss: MelReconstructionLoss = MelReconstructionLoss()
        self._tprls_loss: _TopKRelativeLeastSquaresLoss = _TopKRelativeLeastSquaresLoss(configuration.tprls_tau)

    def compute_discriminator_loss(
        self,
        real_period_logits: list[torch.Tensor],
        fake_period_logits: list[torch.Tensor],
        real_spectrogram_logits: list[torch.Tensor],
        fake_spectrogram_logits: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes the discriminator objective for one GAN training step as
        # the unweighted sum of four terms: the plain least-squares
        # separation and the truncated relative term, each evaluated once
        # per ensemble. No configured weight applies on this path, so the
        # period and spectrogram ensembles and the two adversarial forms all
        # train at equal strength.
        #
        # Args:
        #     real_period_logits: Multi-period logits for the reference
        #         audio.
        #     fake_period_logits: Multi-period logits for the synthesis,
        #         which the caller detaches before the discriminator forward
        #         pass so this term cannot reach the generator.
        #     real_spectrogram_logits: Spectrogram-discriminator logits for
        #         the reference audio.
        #     fake_spectrogram_logits: Spectrogram-discriminator logits for
        #         the detached synthesis.
        #
        # Raises:
        #     ValueError: If either ensemble's real and fake lists disagree
        #         in length, raised by the strict pairing inside the
        #         truncated relative term.
        #
        # Returns:
        #     The summed total as a graph-carrying scalar tensor, paired
        #     with a five-entry component panel keyed
        #     discriminator_loss_total and one entry for each of the four
        #     constituent terms, so the two adversarial forms remain
        #     separable in the logs.
        period_lsgan: torch.Tensor = self._least_squares_gan.discriminator_loss(
            real_period_logits,
            fake_period_logits
        )
        spectrogram_lsgan: torch.Tensor = self._least_squares_gan.discriminator_loss(
            real_spectrogram_logits,
            fake_spectrogram_logits
        )
        period_tprls: torch.Tensor = self._tprls_loss(real_period_logits, fake_period_logits)
        spectrogram_tprls: torch.Tensor = self._tprls_loss(real_spectrogram_logits, fake_spectrogram_logits)
        total_loss: torch.Tensor = period_lsgan + spectrogram_lsgan + period_tprls + spectrogram_tprls
        components: dict[str, float] = {
            "discriminator_loss_total": float(total_loss.detach().item()),
            "discriminator_loss_period_lsgan": float(period_lsgan.detach().item()),
            "discriminator_loss_spectrogram_lsgan": float(spectrogram_lsgan.detach().item()),
            "discriminator_loss_period_tprls": float(period_tprls.detach().item()),
            "discriminator_loss_spectrogram_tprls": float(spectrogram_tprls.detach().item())
        }
        return total_loss, components

    def compute_generator_loss(
        self,
        reference_mel: torch.Tensor,
        candidate_mel: torch.Tensor,
        real_period_logits: list[torch.Tensor],
        fake_period_logits: list[torch.Tensor],
        real_spectrogram_logits: list[torch.Tensor],
        fake_spectrogram_logits: list[torch.Tensor],
        real_period_features: list[list[torch.Tensor]],
        fake_period_features: list[list[torch.Tensor]],
        real_spectrogram_features: list[list[torch.Tensor]],
        fake_spectrogram_features: list[list[torch.Tensor]]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes the generator objective for one GAN training step. The
        # plain least-squares term consumes only the fake logits, but the
        # truncated relative term consumes both sides, which is why this
        # signature takes the real logits of both ensembles as well. The two
        # adversarial forms and the mel term enter unweighted or at their
        # own weight, while each ensemble's feature-matching term is scaled
        # by the shared feature-matching weight.
        #
        # Args:
        #     reference_mel: Log-mel spectrogram of the reference audio.
        #     candidate_mel: Log-mel spectrogram of the synthesis; must
        #         match reference_mel in shape exactly.
        #     real_period_logits: Multi-period logits for the reference
        #         audio, consumed only by the truncated relative term.
        #     fake_period_logits: Multi-period logits for the synthesis,
        #         carrying gradient back to the generator.
        #     real_spectrogram_logits: Spectrogram logits for the reference
        #         audio, likewise consumed only by the truncated relative
        #         term.
        #     fake_spectrogram_logits: Spectrogram logits for the synthesis.
        #     real_period_features: Multi-period intermediate activations
        #         for the reference audio; detached inside the matching term.
        #     fake_period_features: The corresponding activations for the
        #         synthesis.
        #     real_spectrogram_features: Spectrogram-discriminator
        #         activations for the reference audio; likewise detached.
        #     fake_spectrogram_features: The corresponding activations for
        #         the synthesis.
        #
        # Raises:
        #     ValueError: If either ensemble's real and fake logit lists
        #         disagree in length, or if the two spectrograms disagree in
        #         shape. The logit pairing is checked first because the mel
        #         term is evaluated last.
        #
        # Returns:
        #     The weighted total as a graph-carrying scalar tensor, paired
        #     with an eight-entry component panel keyed
        #     generator_loss_total, the four adversarial entries, the two
        #     feature-matching entries, and generator_loss_mel. Every
        #     per-term entry is reported before its configured weight is
        #     applied.
        period_lsgan: torch.Tensor = self._least_squares_gan.generator_loss(fake_period_logits)
        spectrogram_lsgan: torch.Tensor = self._least_squares_gan.generator_loss(fake_spectrogram_logits)
        period_tprls: torch.Tensor = self._tprls_loss(real_period_logits, fake_period_logits)
        spectrogram_tprls: torch.Tensor = self._tprls_loss(real_spectrogram_logits, fake_spectrogram_logits)
        period_feature_matching: torch.Tensor = self._feature_matching_loss(real_period_features, fake_period_features)
        spectrogram_feature_matching: torch.Tensor = self._feature_matching_loss(
            real_spectrogram_features,
            fake_spectrogram_features
        )
        mel_loss: torch.Tensor = self._mel_reconstruction_loss(reference_mel, candidate_mel)
        total_loss: torch.Tensor = (
            period_lsgan
            + spectrogram_lsgan
            + period_tprls
            + spectrogram_tprls
            + self._configuration.feature_matching_weight * period_feature_matching
            + self._configuration.feature_matching_weight * spectrogram_feature_matching
            + self._configuration.mel_weight * mel_loss
        )
        components: dict[str, float] = {
            "generator_loss_total": float(total_loss.detach().item()),
            "generator_loss_period_lsgan": float(period_lsgan.detach().item()),
            "generator_loss_spectrogram_lsgan": float(spectrogram_lsgan.detach().item()),
            "generator_loss_period_tprls": float(period_tprls.detach().item()),
            "generator_loss_spectrogram_tprls": float(spectrogram_tprls.detach().item()),
            "generator_loss_feature_matching_period": float(period_feature_matching.detach().item()),
            "generator_loss_feature_matching_spectrogram": float(spectrogram_feature_matching.detach().item()),
            "generator_loss_mel": float(mel_loss.detach().item())
        }
        return total_loss, components

    def compute_validation_loss(
        self,
        reference_mel: torch.Tensor,
        candidate_mel: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes validation loss without mutating optimizer state. Only
        # the weighted mel reconstruction term participates: neither
        # adversarial form is evaluated, so the reported value is comparable
        # across checkpoints without depending on how well the
        # discriminators happen to be trained at that moment. This is the
        # quantity a checkpoint monitor can rank.
        #
        # Args:
        #     reference_mel: Log-mel spectrogram of the reference audio.
        #     candidate_mel: Log-mel spectrogram of the synthesis; must
        #         match reference_mel in shape exactly.
        #
        # Raises:
        #     ValueError: If the two spectrograms disagree in shape; the
        #         same guard applies on the evaluation path as in training.
        #
        # Returns:
        #     The weighted mel term as a scalar tensor, paired with a
        #     two-entry component panel keyed validation_loss_total and
        #     validation_loss_mel. These keys are deliberately distinct from
        #     the training panel's, so the two paths never collide in the
        #     logged metric buffer.
        mel_loss: torch.Tensor = self._mel_reconstruction_loss(reference_mel, candidate_mel)
        total_loss: torch.Tensor = self._configuration.mel_weight * mel_loss
        components: dict[str, float] = {
            "validation_loss_total": float(total_loss.detach().item()),
            "validation_loss_mel": float(mel_loss.detach().item())
        }
        return total_loss, components
