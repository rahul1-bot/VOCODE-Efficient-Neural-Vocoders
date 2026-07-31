# This module:
# 1. Implements the HiFi-GAN training objective: least-squares adversarial
#    terms over both discriminator ensembles, feature matching, and the
#    weighted full-band mel reconstruction term, composed per the
#    reference recipe
#
# Published composition:
# - Generator: ``45 * L1(mel) + 2 * (FM_period + FM_scale) +
#   1 * (LSGAN_period + LSGAN_scale)``, where each ensemble term is the
#   unnormalized sum over that ensemble's sub-discriminators
# - Discriminator: ``LSGAN_period + LSGAN_scale`` with no relative
#   weighting, so the multi-period and multi-scale ensembles are trained at
#   equal strength
#
# Update semantics:
# - The objective is two-phase. The driving model first computes the
#   discriminator scalar against the detached synthesis and steps the
#   discriminator optimizer, then computes the generator scalar through the
#   same discriminators with gradients flowing back into the generator. This
#   class supplies both scalars but owns no optimizer state and enforces no
#   ordering; toggling and stepping belong to the model
#
# Report alignment:
# - The report records the HiFi-GAN V1, V2, and V3 and BigVGAN-base
#   Project-Trained Configurations as combining least-squares adversarial
#   and feature-matching terms over their discriminator ensembles with
#   weighted mel reconstruction, which is the composition below. The three
#   HiFi-GAN configurations differ in capacity and budget, not in objective
#
# Design decisions:
# - The mel term carries the reference weight of forty-five, which
#   dominates early training and anchors the adversarial game
# - Both ensembles share one LeastSquaresGanLoss and one
#   FeatureMatchingLoss instance because those components are stateless;
#   the per-ensemble scalars are summed here rather than by passing
#   concatenated lists, so each ensemble keeps its own reported component
#
# Author: Rahul Sawhney

from typing import ClassVar

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat
from torch import nn

from vocode.losses.adversarial import LeastSquaresGanLoss
from vocode.losses.feature_matching import FeatureMatchingLoss
from vocode.losses.mel_reconstruction import MelReconstructionLoss

__all__: list[str] = ["HifiganLoss", "HifiganLossConfig"]


class HifiganLossConfig(BaseModel):
    # Frozen loss weights of the reference composition.
    #
    # Fields:
    #     mel_reconstruction_weight: Multiplier on the L1 log-mel
    #         reconstruction term. Default: ``45.0``.
    #     feature_matching_weight: Multiplier on the combined
    #         feature-matching term of both ensembles. Default: ``2.0``.
    #     adversarial_weight: Multiplier on the combined least-squares
    #         generator term of both ensembles. Default: ``1.0``.
    #
    # Every field is a PositiveFloat on a strict, extra-forbidding, frozen
    # model, so a weight may be attenuated but never zeroed, negated, or
    # misspelled into silent acceptance, and the record cannot drift after
    # the run has logged it.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    mel_reconstruction_weight: PositiveFloat = 45.0
    feature_matching_weight: PositiveFloat = 2.0
    adversarial_weight: PositiveFloat = 1.0


class HifiganLoss(nn.Module):
    # The HiFi-GAN composite objective: adversarial, feature-matching, and
    # weighted mel terms for the generator; least-squares real/fake
    # separation for the discriminators. The instance owns the frozen weight
    # record and the three stateless component losses, and registers no
    # parameters of its own, so despite subclassing nn.Module it contributes
    # nothing to the driving model's optimizer state.
    #
    # Integration: the driving model constructs one instance from its
    # configuration and calls compute_discriminator_loss and then
    # compute_generator_loss once each per training batch under its own
    # optimizer toggles. This class never touches optimizers, gradients, or
    # module modes; it only reduces the tensors it is handed.
    def __init__(self, configuration: HifiganLossConfig) -> None:
        # Binds the frozen weight record and constructs the three component
        # losses this composition reduces through. Each component is
        # stateless, so a single instance of each serves both discriminator
        # ensembles.
        #
        # Args:
        #     configuration: The frozen weight record, retained by
        #         reference and republished unchanged through the
        #         configuration property, so the weights a run logs are
        #         exactly the weights it applies.
        super().__init__()
        self._configuration: HifiganLossConfig = configuration
        self._mel_reconstruction_loss: MelReconstructionLoss = MelReconstructionLoss()
        self._adversarial_loss: LeastSquaresGanLoss = LeastSquaresGanLoss()
        self._feature_matching_loss: FeatureMatchingLoss = FeatureMatchingLoss()

    def compute_generator_loss(
        self,
        reference_mel: torch.Tensor,
        synthesized_mel: torch.Tensor,
        fake_period_logits: list[torch.Tensor],
        fake_scale_logits: list[torch.Tensor],
        real_period_features: list[list[torch.Tensor]],
        fake_period_features: list[list[torch.Tensor]],
        real_scale_features: list[list[torch.Tensor]],
        fake_scale_features: list[list[torch.Tensor]]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes the generator objective for one GAN training step. The mel
        # term is measured first, so a spectrogram-shape disagreement fails
        # before any adversarial work is done. Each ensemble then contributes
        # an independent adversarial and feature-matching term; the two
        # ensembles are summed within each group, and the three groups are
        # scaled by their configured weights to form the total. The component
        # panel reports every per-ensemble term unweighted, so a reader can
        # recover the effect of any weight from the logged values alone.
        #
        # Args:
        #     reference_mel: Log-mel spectrogram of the reference audio,
        #         serving as the reconstruction target.
        #     synthesized_mel: Log-mel spectrogram of the synthesis; must
        #         match reference_mel in shape exactly.
        #     fake_period_logits: Multi-period discriminator logits for the
        #         synthesis, carrying gradient back to the generator.
        #     fake_scale_logits: Multi-scale discriminator logits for the
        #         synthesis.
        #     real_period_features: Multi-period intermediate activations
        #         for the reference audio; detached inside the matching term.
        #     fake_period_features: The corresponding multi-period
        #         activations for the synthesis.
        #     real_scale_features: Multi-scale intermediate activations for
        #         the reference audio; likewise detached.
        #     fake_scale_features: The corresponding multi-scale activations
        #         for the synthesis.
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
        adversarial_scale: torch.Tensor = self._adversarial_loss.generator_loss(fake_scale_logits)
        feature_matching_period: torch.Tensor = self._feature_matching_loss(real_period_features, fake_period_features)
        feature_matching_scale: torch.Tensor = self._feature_matching_loss(real_scale_features, fake_scale_features)
        adversarial_total: torch.Tensor = adversarial_period + adversarial_scale
        feature_matching_total: torch.Tensor = feature_matching_period + feature_matching_scale
        total_loss: torch.Tensor = (
            self._configuration.mel_reconstruction_weight * mel_loss
            + self._configuration.feature_matching_weight * feature_matching_total
            + self._configuration.adversarial_weight * adversarial_total
        )
        components: dict[str, float] = {
            "generator_loss_total": float(total_loss.detach().item()),
            "generator_loss_mel": float(mel_loss.detach().item()),
            "generator_loss_adversarial_period": float(adversarial_period.detach().item()),
            "generator_loss_adversarial_scale": float(adversarial_scale.detach().item()),
            "generator_loss_feature_matching_period": float(feature_matching_period.detach().item()),
            "generator_loss_feature_matching_scale": float(feature_matching_scale.detach().item())
        }
        return total_loss, components

    def compute_discriminator_loss(
        self,
        real_period_logits: list[torch.Tensor],
        fake_period_logits: list[torch.Tensor],
        real_scale_logits: list[torch.Tensor],
        fake_scale_logits: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes the discriminator objective for one GAN training step as
        # the unweighted sum of the two ensembles' least-squares separation
        # terms. No configured weight applies on this path: the adversarial
        # weight scales only the generator's view of the game, so the two
        # discriminators always train at equal strength.
        #
        # Args:
        #     real_period_logits: Multi-period logits for the reference
        #         audio.
        #     fake_period_logits: Multi-period logits for the synthesis. The
        #         caller detaches the synthesis before the discriminator
        #         forward pass, so this term cannot reach the generator.
        #     real_scale_logits: Multi-scale logits for the reference audio.
        #     fake_scale_logits: Multi-scale logits for the detached
        #         synthesis.
        #
        # Returns:
        #     The summed separation term as a graph-carrying scalar tensor,
        #     paired with a component panel of detached Python floats keyed
        #     discriminator_loss_total, discriminator_loss_period, and
        #     discriminator_loss_scale.
        period_loss: torch.Tensor = self._adversarial_loss.discriminator_loss(real_period_logits, fake_period_logits)
        scale_loss: torch.Tensor = self._adversarial_loss.discriminator_loss(real_scale_logits, fake_scale_logits)
        total_loss: torch.Tensor = period_loss + scale_loss
        components: dict[str, float] = {
            "discriminator_loss_total": float(total_loss.detach().item()),
            "discriminator_loss_period": float(period_loss.detach().item()),
            "discriminator_loss_scale": float(scale_loss.detach().item())
        }
        return total_loss, components

    @property
    def configuration(self) -> HifiganLossConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration
