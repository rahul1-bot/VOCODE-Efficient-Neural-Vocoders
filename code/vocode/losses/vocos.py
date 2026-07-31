# This module:
# 1. Implements the Vocos training objective: hinge adversarial and
#    feature-matching terms over both ensembles plus the weighted mel
#    reconstruction term, per the reference recipe
#
# Published composition:
# - Generator: ``hinge_period + 0.1 * hinge_resolution + FM_period +
#   0.1 * FM_resolution + 45 * L1(mel)``, where every ensemble term is a
#   mean over sub-discriminators rather than a sum
# - Discriminator: ``hinge_period + 0.1 * hinge_resolution``, again
#   member-averaged, so the multi-resolution ensemble enters at a tenth of
#   the multi-period ensemble on both sides of the game
# - Validation: ``45 * L1(mel)`` alone, with no adversarial participation
#
# Update semantics:
# - Two-phase: the driving model computes the discriminator scalar against
#   the detached synthesis and steps the discriminator optimizer, then
#   computes the generator scalar through the same discriminators. This
#   class supplies all three scalars and owns no optimizer state
#
# Report alignment:
# - The report records Vocos and VocosFormer as applying the hinge form of
#   the HiFi-GAN composition, which is what the clamped hinge below
#   realizes. Both Project-Trained Configurations drive this one objective
#
# Design decisions:
# - The adversarial form is the clamped hinge, not the least-squares form
#   used by the HiFi-GAN and BigVGAN objectives: once a sub-discriminator is
#   fully fooled the term saturates at exactly zero instead of continuing to
#   reward ever-larger logits
# - Every ensemble term is divided by its member count, so widening an
#   ensemble changes what the discriminators see without changing the
#   magnitude of the generator's adversarial or matching signal; this is the
#   opposite convention from the summed HiFi-GAN family and is the reason
#   the shared FeatureMatchingLoss result is normalized here at the call
#   site rather than inside that component
# - The member-count divisors are guarded with ``max(1, ...)``, so an empty
#   ensemble yields zero rather than a division by zero
#
# Author: Rahul Sawhney

from typing import ClassVar

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat
from torch import nn

from vocode.losses.feature_matching import FeatureMatchingLoss
from vocode.losses.mel_reconstruction import MelReconstructionLoss

__all__: list[str] = ["VocosLoss", "VocosLossConfig"]


class VocosLossConfig(BaseModel):
    # Frozen loss weights of the reference composition.
    #
    # Fields:
    #     mel_weight: Multiplier on the L1 log-mel reconstruction term,
    #         applied on both the training and validation paths.
    #         Default: ``45.0``.
    #     multi_resolution_discriminator_weight: Multiplier on every
    #         multi-resolution ensemble term. The one weight governs the
    #         resolution ensemble's adversarial and feature-matching
    #         contributions to the generator and its separation term on the
    #         discriminator path. Default: ``0.1``.
    #
    # The record carries no separate feature-matching weight, because the
    # resolution weight already scales the matching term and the period
    # ensemble's matching term enters unscaled. Both fields are
    # PositiveFloat on a strict, extra-forbidding, frozen model, so a weight
    # may be attenuated but never zeroed, negated, or misspelled into
    # silent acceptance.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    mel_weight: PositiveFloat = 45.0
    multi_resolution_discriminator_weight: PositiveFloat = 0.1


class _VocosHingeGanLoss(nn.Module):
    # Private hinge adversarial objective for the Vocos composition, holding
    # both halves of the game over ensemble logit lists. It differs from the
    # package's shared LeastSquaresGanLoss in two ways that matter to the
    # recipe: the penalty is a clamped hinge rather than a squared error, and
    # the per-member terms are averaged rather than summed. The class holds
    # no parameters and no mutable state, so one instance serves both
    # ensembles.
    def compute_generator_loss(self, fake_logits: list[torch.Tensor]) -> torch.Tensor:
        # Computes the generator objective for one GAN training step. The clamped hinge form
        # saturates once a sub-discriminator is fully fooled, unlike a raw negative mean.
        # Each member contributes ``mean(clamp(1 - logit, min=0))`` and the
        # accumulated value is divided by the member count, so a logit at or
        # above one costs nothing and widening the ensemble does not inflate
        # the generator's signal.
        #
        # Args:
        #     fake_logits: One logit tensor per sub-discriminator, evaluated
        #         on synthesized audio.
        #
        # Returns:
        #     A scalar tensor holding the member-averaged hinge term, placed
        #     on the device of the first logit. An empty list returns a CPU
        #     zero scalar; the divisor is floored at one so no division by
        #     zero occurs.
        loss: torch.Tensor = torch.zeros((), device=fake_logits[0].device) if fake_logits else torch.zeros(())
        for fake_logit in fake_logits:
            loss: torch.Tensor = loss + torch.mean(torch.clamp(1.0 - fake_logit, min=0.0))
        return loss / max(1, len(fake_logits))

    def compute_discriminator_loss(
        self,
        real_logits: list[torch.Tensor],
        fake_logits: list[torch.Tensor]
    ) -> torch.Tensor:
        # Computes the discriminator objective for one GAN training step.
        # Each aligned pair contributes ``mean(clamp(1 - real, min=0)) +
        # mean(clamp(1 + fake, min=0))``, so a member pays nothing once it
        # scores reference audio at or above one and synthesized audio at or
        # below minus one. The accumulated value is divided by the member
        # count, matching the generator-side convention.
        #
        # Args:
        #     real_logits: One logit tensor per sub-discriminator, evaluated
        #         on reference audio; also fixes the accumulator's device.
        #     fake_logits: The matching logit tensors evaluated on the
        #         detached synthesis.
        #
        # Raises:
        #     ValueError: If the two lists disagree in length; the pairing
        #         uses strict zip, so a member-count mismatch fails closed
        #         rather than truncating to the shorter side.
        #
        # Returns:
        #     A scalar tensor holding the member-averaged separation term.
        #     An empty real_logits list returns a CPU zero scalar.
        loss: torch.Tensor = torch.zeros((), device=real_logits[0].device) if real_logits else torch.zeros(())
        for real_logit, fake_logit in zip(real_logits, fake_logits, strict=True):
            real_loss: torch.Tensor = torch.mean(torch.clamp(1.0 - real_logit, min=0.0))
            fake_loss: torch.Tensor = torch.mean(torch.clamp(1.0 + fake_logit, min=0.0))
            loss: torch.Tensor = loss + real_loss + fake_loss
        return loss / max(1, len(real_logits))


class VocosLoss(nn.Module):
    # The Vocos composite objective: hinge adversarial, feature-matching,
    # and weighted mel terms for the generator; hinge separation for both
    # discriminator ensembles. The instance owns the frozen weight record
    # and three stateless component losses, and registers no parameters of
    # its own, so despite subclassing nn.Module it contributes nothing to
    # the driving model's optimizer state.
    #
    # Integration: the driving model constructs one instance from its
    # configuration, calls compute_discriminator_loss and then
    # compute_generator_loss once each per training batch under its own
    # optimizer toggles, and calls compute_validation_loss on the
    # evaluation path where no discriminator participates.
    def __init__(self, configuration: VocosLossConfig) -> None:
        # Binds the frozen weight record and constructs the hinge,
        # feature-matching, and mel component losses. Each is stateless, so
        # one instance of each serves both ensembles.
        #
        # Args:
        #     configuration: The frozen weight record, retained by
        #         reference so the weights a run logs are exactly the
        #         weights it applies.
        super().__init__()
        self._configuration: VocosLossConfig = configuration
        self._hinge_loss: _VocosHingeGanLoss = _VocosHingeGanLoss()
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
        # ensemble's, the latter scaled by the configured multi-resolution
        # weight. Both ensembles are member-averaged inside the hinge
        # component, so the configured weight is the only asymmetry between
        # them.
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
        #     discriminator_loss_resolution; the two per-ensemble entries are
        #     reported before the resolution weight is applied.
        period_loss: torch.Tensor = self._hinge_loss.compute_discriminator_loss(real_period_logits, fake_period_logits)
        resolution_loss: torch.Tensor = self._hinge_loss.compute_discriminator_loss(
            real_resolution_logits,
            fake_resolution_logits
        )
        total_loss: torch.Tensor = (
            period_loss + self._configuration.multi_resolution_discriminator_weight * resolution_loss
        )
        components: dict[str, float] = {
            "discriminator_loss_total": float(total_loss.detach().item()),
            "discriminator_loss_period": float(period_loss.detach().item()),
            "discriminator_loss_resolution": float(resolution_loss.detach().item())
        }
        return total_loss, components

    def compute_generator_loss(
        self,
        reference_mel: torch.Tensor,
        candidate_mel: torch.Tensor,
        fake_period_logits: list[torch.Tensor],
        fake_resolution_logits: list[torch.Tensor],
        real_period_features: list[list[torch.Tensor]],
        fake_period_features: list[list[torch.Tensor]],
        real_resolution_features: list[list[torch.Tensor]],
        fake_resolution_features: list[list[torch.Tensor]]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes the generator objective for one GAN training step. Both
        # ensembles contribute a member-averaged hinge term and a
        # feature-matching term; the matching results are divided by the
        # ensemble's member count here rather than inside the shared
        # component, because that component sums by design for the HiFi-GAN
        # family. The resolution ensemble's adversarial and matching terms
        # both enter at the configured multi-resolution weight, the period
        # ensemble's enter unscaled, and the mel term enters at its own
        # weight.
        #
        # Args:
        #     reference_mel: Log-mel spectrogram of the reference audio.
        #     candidate_mel: Log-mel spectrogram of the synthesis; must
        #         match reference_mel in shape exactly.
        #     fake_period_logits: Multi-period logits for the synthesis,
        #         carrying gradient back to the generator.
        #     fake_resolution_logits: Multi-resolution logits for the
        #         synthesis.
        #     real_period_features: Multi-period intermediate activations
        #         for the reference audio; detached inside the matching
        #         term, and their count sets that term's divisor.
        #     fake_period_features: The corresponding activations for the
        #         synthesis.
        #     real_resolution_features: Multi-resolution intermediate
        #         activations for the reference audio; likewise detached and
        #         count-setting.
        #     fake_resolution_features: The corresponding activations for
        #         the synthesis.
        #
        # Raises:
        #     ValueError: If reference_mel and candidate_mel disagree in
        #         shape, raised by the mel reconstruction guard. Unlike the
        #         HiFi-GAN family this check runs after the adversarial and
        #         matching terms, because the mel term is evaluated last.
        #
        # Returns:
        #     The weighted total as a graph-carrying scalar tensor, paired
        #     with a six-entry component panel keyed generator_loss_total,
        #     the two per-ensemble adversarial entries, the two
        #     per-ensemble feature-matching entries, and generator_loss_mel.
        #     Every per-term entry is reported before its configured weight
        #     is applied.
        period_adversarial: torch.Tensor = self._hinge_loss.compute_generator_loss(fake_period_logits)
        resolution_adversarial: torch.Tensor = self._hinge_loss.compute_generator_loss(fake_resolution_logits)
        period_feature_matching: torch.Tensor = (
            self._feature_matching_loss(real_period_features, fake_period_features)
            / max(1, len(real_period_features))
        )
        resolution_feature_matching: torch.Tensor = (
            self._feature_matching_loss(real_resolution_features, fake_resolution_features)
            / max(1, len(real_resolution_features))
        )
        mel_loss: torch.Tensor = self._mel_reconstruction_loss(reference_mel, candidate_mel)
        total_loss: torch.Tensor = (
            period_adversarial
            + self._configuration.multi_resolution_discriminator_weight * resolution_adversarial
            + period_feature_matching
            + self._configuration.multi_resolution_discriminator_weight * resolution_feature_matching
            + self._configuration.mel_weight * mel_loss
        )
        components: dict[str, float] = {
            "generator_loss_total": float(total_loss.detach().item()),
            "generator_loss_period_adversarial": float(period_adversarial.detach().item()),
            "generator_loss_resolution_adversarial": float(resolution_adversarial.detach().item()),
            "generator_loss_feature_matching_period": float(period_feature_matching.detach().item()),
            "generator_loss_feature_matching_resolution": float(resolution_feature_matching.detach().item()),
            "generator_loss_mel": float(mel_loss.detach().item())
        }
        return total_loss, components

    def compute_validation_loss(
        self,
        reference_mel: torch.Tensor,
        candidate_mel: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes validation loss without mutating optimizer state. Only
        # the weighted mel reconstruction term participates: no
        # discriminator is evaluated, so the reported value is comparable
        # across checkpoints and across the run's own training history
        # without depending on how well the discriminators happen to be
        # trained at that moment. This is the quantity a checkpoint monitor
        # can rank.
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
