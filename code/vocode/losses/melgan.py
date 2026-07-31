# This module:
# 1. Implements the MelGAN training objective: least-squares adversarial
#    terms over the multi-scale ensemble and weighted feature matching,
#    without a spectral reconstruction term
#
# Report alignment:
# - The report records this family as optimizing least-squares adversarial
#   and feature-matching terms alone, which is what the absence of any mel
#   or STFT term below realizes
#
# Published composition:
# - Generator: ``adversarial + 10 * FM``, where the adversarial term sums
#   the squared error of the fake score against the real target of one over
#   the channel and time axes and then averages over the batch, and the
#   matching term is the mean absolute difference between paired feature
#   maps with the reference side detached
# - Discriminator: the same least-squares reduction applied on both sides,
#   pulling real scores toward one and fake scores toward zero, summed over
#   ensemble members with no relative weighting
#
# Update semantics:
# - Two-phase: the driving model computes the discriminator scalar against
#   the detached synthesis and steps the discriminator optimizer, then
#   computes the generator scalar through the same discriminators. This
#   class supplies both scalars and owns no optimizer state
#
# Design decisions:
# - This family carries no mel or STFT reconstruction term at all; feature
#   matching at weight ten is the entire reconstruction signal, which is
#   what distinguishes the recipe from the HiFi-GAN family and is why the
#   configuration record has exactly one field
# - The adversarial reduction sums over the channel and time axes before
#   the batch mean rather than taking a plain mean over all elements, so
#   the term's magnitude scales with the score map's resolution
# - Members and layers are paired with strict zip, so a structural mismatch
#   in the discriminator wiring raises instead of silently truncating to
#   the shorter side
# - The reductions are inlined here rather than reusing the shared
#   LeastSquaresGanLoss and FeatureMatchingLoss components, because this
#   family's discriminator returns each member's features and score as one
#   tuple and its adversarial reduction sums before averaging
#
# Author: Rahul Sawhney

from typing import ClassVar

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat
from torch import nn

__all__: list[str] = ["MelganLoss", "MelganLossConfig"]


class MelganLossConfig(BaseModel):
    # Frozen loss weights of the reference composition.
    #
    # Fields:
    #     feature_matching_weight: Multiplier on the feature-matching term,
    #         which is this family's entire reconstruction signal.
    #         Default: ``10.0``.
    #
    # The record has exactly one field because the recipe carries no
    # spectral reconstruction term; naming a mel weight here is rejected
    # rather than ignored, since extra fields are forbidden. The field is a
    # PositiveFloat on a strict, frozen model, so feature matching may be
    # attenuated but never disabled through configuration.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    feature_matching_weight: PositiveFloat = 10.0


class MelganLoss(nn.Module):
    # The MelGAN composite objective: least-squares adversarial and weighted
    # feature-matching terms for the generator; least-squares separation for
    # the multi-scale discriminator ensemble. Unlike the HiFi-GAN family
    # this class composes no sub-loss objects, because both reductions are
    # inlined to consume the ensemble's combined feature-and-score tuples;
    # it therefore holds only the frozen weight record and registers no
    # parameters.
    #
    # Integration: the driving model constructs one instance from its
    # configuration and calls compute_discriminator_loss and then
    # compute_generator_loss once each per training batch under its own
    # optimizer toggles. Both methods take the same two ensemble-output
    # arguments, so the caller passes one pair of structures per phase.
    def __init__(self, configuration: MelganLossConfig) -> None:
        # Binds the frozen weight record. No component losses are
        # constructed because this family's reductions are inlined in the
        # two compute methods.
        #
        # Args:
        #     configuration: The frozen weight record, retained by
        #         reference and republished unchanged through the
        #         configuration property.
        super().__init__()
        self._configuration: MelganLossConfig = configuration

    def compute_generator_loss(
        self,
        fake_outputs: list[tuple[list[torch.Tensor], torch.Tensor]],
        real_outputs: list[tuple[list[torch.Tensor], torch.Tensor]]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes the generator objective for one GAN training step in a
        # single pass over the ensemble. For each member the fake score is
        # penalized by its squared distance from the real target of one,
        # summed over the channel and time axes and averaged over the batch;
        # every layer of that member then contributes the mean absolute
        # difference against the detached reference activation. The total is
        # the adversarial sum plus the weighted matching sum, so the returned
        # panel and the total agree by construction.
        #
        # Args:
        #     fake_outputs: Per-member ``(features, score)`` tuples produced
        #         from the synthesis, carrying gradient back to the
        #         generator. The first member's score also fixes the
        #         accumulators' device.
        #     real_outputs: The corresponding tuples produced from the
        #         reference audio. Only the features are consumed here; the
        #         real score has no role in the generator term and its
        #         features are detached before comparison.
        #
        # Raises:
        #     ValueError: If the two ensembles disagree in member count, or
        #         if any paired member disagrees in layer count; both
        #         pairings use strict zip, so a discriminator wiring defect
        #         fails closed instead of truncating.
        #     IndexError: If fake_outputs is empty, raised while the
        #         accumulators are seeded from the first member's score.
        #
        # Returns:
        #     The weighted total as a graph-carrying scalar tensor, paired
        #     with a three-entry component panel keyed generator_loss_total,
        #     generator_loss_adversarial, and generator_loss_feature_matching.
        #     The panel carries no mel entry, which is what distinguishes
        #     this recipe from the HiFi-GAN family.
        total_loss: torch.Tensor = torch.zeros((), device=fake_outputs[0][1].device)
        adversarial_loss: torch.Tensor = torch.zeros((), device=fake_outputs[0][1].device)
        feature_matching_loss: torch.Tensor = torch.zeros((), device=fake_outputs[0][1].device)
        for (fake_features, fake_score), (real_features, _) in zip(fake_outputs, real_outputs, strict=True):
            adversarial_loss: torch.Tensor = adversarial_loss + torch.mean(torch.sum((fake_score - 1.0) ** 2, dim=[1, 2]))
            for fake_feature, real_feature in zip(fake_features, real_features, strict=True):
                feature_matching_loss: torch.Tensor = feature_matching_loss + torch.mean(torch.abs(fake_feature - real_feature.detach()))
        total_loss: torch.Tensor = adversarial_loss + self._configuration.feature_matching_weight * feature_matching_loss
        return total_loss, {
            "generator_loss_total": float(total_loss.detach().item()),
            "generator_loss_adversarial": float(adversarial_loss.detach().item()),
            "generator_loss_feature_matching": float(feature_matching_loss.detach().item())
        }

    def compute_discriminator_loss(
        self,
        fake_outputs: list[tuple[list[torch.Tensor], torch.Tensor]],
        real_outputs: list[tuple[list[torch.Tensor], torch.Tensor]]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes the discriminator objective for one GAN training step by
        # accumulating both sides of the least-squares separation per member:
        # the real score is pulled toward one and the fake score toward zero,
        # each summed over the channel and time axes before the batch mean.
        # Only the scores are consumed; the feature lists carried in the
        # tuples have no role on this path.
        #
        # Args:
        #     fake_outputs: Per-member ``(features, score)`` tuples produced
        #         from the detached synthesis. The first member's score fixes
        #         the accumulator's device.
        #     real_outputs: The corresponding tuples produced from the
        #         reference audio.
        #
        # Raises:
        #     ValueError: If the two ensembles disagree in member count; the
        #         pairing uses strict zip so the mismatch fails closed.
        #     IndexError: If fake_outputs is empty, raised while the
        #         accumulator is seeded from the first member's score.
        #
        # Returns:
        #     The summed separation term as a graph-carrying scalar tensor,
        #     paired with a single-entry component panel keyed
        #     discriminator_loss_total; this family logs no per-ensemble
        #     breakdown because it drives only one ensemble.
        total_loss: torch.Tensor = torch.zeros((), device=fake_outputs[0][1].device)
        for (_, fake_score), (_, real_score) in zip(fake_outputs, real_outputs, strict=True):
            total_loss: torch.Tensor = total_loss + torch.mean(torch.sum((real_score - 1.0) ** 2, dim=[1, 2]))
            total_loss: torch.Tensor = total_loss + torch.mean(torch.sum(fake_score ** 2, dim=[1, 2]))
        return total_loss, {
            "discriminator_loss_total": float(total_loss.detach().item())
        }

    @property
    def configuration(self) -> MelganLossConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration
