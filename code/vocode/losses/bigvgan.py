# This module:
# 1. Implements the BigVGAN training objective: least-squares adversarial
#    and feature-matching terms over both ensembles plus the weighted
#    mel reconstruction term, per the reference recipe
#
# Published composition:
# - Generator: ``45 * L1(mel) + 2 * (FM_period + FM_resolution) +
#   1 * (LSGAN_period + LSGAN_resolution)``, each ensemble term summed over
#   its sub-discriminators without normalization by member count
# - Discriminator: ``LSGAN_period + LSGAN_resolution`` with no relative
#   weighting between the two ensembles
#
# Update semantics:
# - Two-phase and identical in shape to the HiFi-GAN lineage: a
#   discriminator step over the detached synthesis, then a generator step
#   through the same discriminators. This class supplies both scalars and
#   owns no optimizer state
#
# Report alignment:
# - The report records BigVGAN-base alongside the HiFi-GAN configurations
#   as combining least-squares adversarial and feature-matching terms over
#   their discriminator ensembles with weighted mel reconstruction, which is
#   the composition below
#
# Design decisions:
# - The composition is structurally the HiFi-GAN one with the second
#   ensemble's identity changed from multi-scale to multi-resolution; the
#   three weights are carried over unchanged, so a comparison between the
#   two families isolates the architectural difference rather than a
#   difference in objective weighting
# - Both ensembles share one LeastSquaresGanLoss and one
#   FeatureMatchingLoss instance, and their scalars are summed here so each
#   ensemble keeps its own reported component
#
# Author: Rahul Sawhney

from typing import ClassVar

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat
from torch import nn

from vocode.losses.adversarial import LeastSquaresGanLoss
from vocode.losses.feature_matching import FeatureMatchingLoss
from vocode.losses.mel_reconstruction import MelReconstructionLoss

__all__: list[str] = ["BigvganLoss", "BigvganLossConfig"]


class BigvganLossConfig(BaseModel):
    # Frozen loss weights of the reference composition.
    #
    # Fields:
    #     mel_reconstruction_weight: Multiplier on the L1 log-mel
    #         reconstruction term. Default: ``45.0``.
    #     feature_matching_weight: Multiplier on the combined
    #         feature-matching term of the period and resolution ensembles.
    #         Default: ``2.0``.
    #     adversarial_weight: Multiplier on the combined least-squares
    #         generator term of both ensembles. Default: ``1.0``.
    #
    # The three defaults are the HiFi-GAN values carried over unchanged.
    # Every field is a PositiveFloat on a strict, extra-forbidding, frozen
    # model, so a weight may be attenuated but never zeroed, negated, or
    # misspelled into silent acceptance.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    mel_reconstruction_weight: PositiveFloat = 45.0
    feature_matching_weight: PositiveFloat = 2.0
    adversarial_weight: PositiveFloat = 1.0


class BigvganLoss(nn.Module):
    # The BigVGAN composite objective: adversarial, feature-matching, and
    # weighted mel terms for the generator; least-squares separation for
    # both discriminator ensembles. The instance owns the frozen weight
    # record and the three stateless component losses, and registers no
    # parameters of its own, so despite subclassing nn.Module it contributes
    # nothing to the driving model's optimizer state.
    #
    # Integration: the driving model constructs one instance from its
    # configuration and calls compute_discriminator_loss and then
    # compute_generator_loss once each per training batch under its own
    # optimizer toggles. The call surface is identical to HifiganLoss except
    # that the second ensemble is multi-resolution rather than multi-scale.
    def __init__(self, configuration: BigvganLossConfig) -> None:
        # Binds the frozen weight record and constructs the three component
        # losses this composition reduces through. Each component is
        # stateless, so a single instance of each serves both discriminator
        # ensembles.
        #
        # Args:
        #     configuration: The frozen weight record, retained by
        #         reference and republished unchanged through the
        #         configuration property.
        super().__init__()
        self._configuration: BigvganLossConfig = configuration
        self._mel_reconstruction_loss: MelReconstructionLoss = MelReconstructionLoss()
        self._adversarial_loss: LeastSquaresGanLoss = LeastSquaresGanLoss()
        self._feature_matching_loss: FeatureMatchingLoss = FeatureMatchingLoss()

    def compute_generator_loss(
        self,
        reference_mel: torch.Tensor,
        synthesized_mel: torch.Tensor,
        fake_period_logits: list[torch.Tensor],
        fake_resolution_logits: list[torch.Tensor],
        real_period_features: list[list[torch.Tensor]],
        fake_period_features: list[list[torch.Tensor]],
        real_resolution_features: list[list[torch.Tensor]],
        fake_resolution_features: list[list[torch.Tensor]]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes the generator objective for one GAN training step. The mel
        # term is measured first, so a spectrogram-shape disagreement fails
        # before any adversarial work is done. The period and resolution
        # ensembles each contribute an independent adversarial and
        # feature-matching term; the two are summed within each group and the
        # three groups are scaled by their configured weights to form the
        # total. Per-ensemble terms are reported unweighted so their relative
        # contribution stays legible in the logs.
        #
        # Args:
        #     reference_mel: Log-mel spectrogram of the reference audio,
        #         serving as the reconstruction target.
        #     synthesized_mel: Log-mel spectrogram of the synthesis; must
        #         match reference_mel in shape exactly.
        #     fake_period_logits: Multi-period discriminator logits for the
        #         synthesis, carrying gradient back to the generator.
        #     fake_resolution_logits: Multi-resolution discriminator logits
        #         for the synthesis.
        #     real_period_features: Multi-period intermediate activations
        #         for the reference audio; detached inside the matching term.
        #     fake_period_features: The corresponding multi-period
        #         activations for the synthesis.
        #     real_resolution_features: Multi-resolution intermediate
        #         activations for the reference audio; likewise detached.
        #     fake_resolution_features: The corresponding multi-resolution
        #         activations for the synthesis.
        #
        # Raises:
        #     ValueError: If reference_mel and synthesized_mel disagree in
        #         shape, raised by the mel reconstruction guard before the
        #         adversarial and feature-matching terms are evaluated.
        #
        # Returns:
        #     The weighted total as a graph-carrying scalar tensor, paired
        #     with a component panel of detached Python floats keyed
        #     generator_loss_total, generator_loss_mel, and the four
        #     per-ensemble adversarial and feature-matching entries.
        mel_loss: torch.Tensor = self._mel_reconstruction_loss(reference_mel, synthesized_mel)
        adversarial_period: torch.Tensor = self._adversarial_loss.generator_loss(fake_period_logits)
        adversarial_resolution: torch.Tensor = self._adversarial_loss.generator_loss(fake_resolution_logits)
        feature_matching_period: torch.Tensor = self._feature_matching_loss(real_period_features, fake_period_features)
        feature_matching_resolution: torch.Tensor = self._feature_matching_loss(
            real_resolution_features,
            fake_resolution_features
        )
        adversarial_total: torch.Tensor = adversarial_period + adversarial_resolution
        feature_matching_total: torch.Tensor = feature_matching_period + feature_matching_resolution
        total_loss: torch.Tensor = (
            self._configuration.mel_reconstruction_weight * mel_loss
            + self._configuration.feature_matching_weight * feature_matching_total
            + self._configuration.adversarial_weight * adversarial_total
        )
        components: dict[str, float] = {
            "generator_loss_total": float(total_loss.detach().item()),
            "generator_loss_mel": float(mel_loss.detach().item()),
            "generator_loss_adversarial_period": float(adversarial_period.detach().item()),
            "generator_loss_adversarial_resolution": float(adversarial_resolution.detach().item()),
            "generator_loss_feature_matching_period": float(feature_matching_period.detach().item()),
            "generator_loss_feature_matching_resolution": float(feature_matching_resolution.detach().item())
        }
        return total_loss, components

    def compute_discriminator_loss(
        self,
        real_period_logits: list[torch.Tensor],
        fake_period_logits: list[torch.Tensor],
        real_resolution_logits: list[torch.Tensor],
        fake_resolution_logits: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes the discriminator objective for one GAN training step as
        # the unweighted sum of the two ensembles' least-squares separation
        # terms. No configured weight applies on this path: the adversarial
        # weight scales only the generator's view of the game, so the period
        # and resolution discriminators always train at equal strength.
        #
        # Args:
        #     real_period_logits: Multi-period logits for the reference
        #         audio.
        #     fake_period_logits: Multi-period logits for the synthesis. The
        #         caller detaches the synthesis before the discriminator
        #         forward pass, so this term cannot reach the generator.
        #     real_resolution_logits: Multi-resolution logits for the
        #         reference audio.
        #     fake_resolution_logits: Multi-resolution logits for the
        #         detached synthesis.
        #
        # Returns:
        #     The summed separation term as a graph-carrying scalar tensor,
        #     paired with a component panel of detached Python floats keyed
        #     discriminator_loss_total, discriminator_loss_period, and
        #     discriminator_loss_resolution.
        period_loss: torch.Tensor = self._adversarial_loss.discriminator_loss(real_period_logits, fake_period_logits)
        resolution_loss: torch.Tensor = self._adversarial_loss.discriminator_loss(
            real_resolution_logits,
            fake_resolution_logits
        )
        total_loss: torch.Tensor = period_loss + resolution_loss
        components: dict[str, float] = {
            "discriminator_loss_total": float(total_loss.detach().item()),
            "discriminator_loss_period": float(period_loss.detach().item()),
            "discriminator_loss_resolution": float(resolution_loss.detach().item())
        }
        return total_loss, components

    @property
    def configuration(self) -> BigvganLossConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration
