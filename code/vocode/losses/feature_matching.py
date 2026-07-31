# This module:
# 1. Implements the feature-matching loss: L1 distance between the
#    discriminator's intermediate feature maps for real and synthesized
#    audio, summed over every ensemble member and layer
#
# Design decisions:
# - Real features are detached inside the reduction so the loss shapes only
#   the generator, never the discriminator's representation; detaching here
#   rather than at the call site means no caller can forget it
# - Feature matching stabilizes adversarial training by giving the
#   generator a dense signal even where logits saturate, because every
#   intermediate activation contributes a gradient regardless of how
#   confidently the sub-discriminator has classified the sample
# - The distances are accumulated without any division by the ensemble or
#   layer count, so the term's magnitude scales with discriminator depth;
#   families that require a normalized term (Vocos) divide the returned
#   scalar themselves rather than changing this shared reduction
#
# Author: Rahul Sawhney

import torch
from torch import nn

__all__: list[str] = ["FeatureMatchingLoss"]


class FeatureMatchingLoss(nn.Module):
    # L1 feature-map matching over nested ensemble feature lists. The nesting
    # is two levels deep: the outer list indexes sub-discriminators and each
    # inner list holds that member's intermediate activations in forward
    # order. The class holds no parameters and no mutable state, so one
    # instance serves every ensemble a composite objective drives.
    #
    # Integration: every adversarial family in this package except MelGAN
    # composes this class; MelGAN instead inlines an equivalent reduction
    # because its discriminator returns features and scores as one tuple.
    def forward(
        self,
        real_feature_maps: list[list[torch.Tensor]],
        fake_feature_maps: list[list[torch.Tensor]]
    ) -> torch.Tensor:
        # Sums the L1 distances between corresponding real (detached) and
        # fake feature maps across all ensemble members and layers. Both
        # levels of nesting are walked with positional, non-strict pairing,
        # so a member or layer present on only one side is silently skipped
        # rather than raising; the reference target of each distance is
        # detached at the point of comparison, which is what confines the
        # gradient to the generator.
        #
        # Args:
        #     real_feature_maps: Per-sub-discriminator lists of intermediate
        #         activations produced from reference audio. This argument
        #         also determines the accumulator's device and the
        #         empty-input behavior.
        #     fake_feature_maps: The matching activations produced from
        #         synthesized audio, aligned to real_feature_maps by
        #         position at both nesting levels.
        #
        # Returns:
        #     A scalar tensor holding the summed L1 distance, placed on the
        #     device of the first reference activation. An empty outer list
        #     returns a CPU zero scalar; a non-empty outer list whose first
        #     member carries no layers raises IndexError while the
        #     accumulator is seeded.
        loss: torch.Tensor = torch.zeros(
            (),
            device=real_feature_maps[0][0].device
        ) if real_feature_maps else torch.zeros(())
        for real_set, fake_set in zip(real_feature_maps, fake_feature_maps):
            for real_feature, fake_feature in zip(real_set, fake_set):
                loss: torch.Tensor = loss + torch.nn.functional.l1_loss(fake_feature, real_feature.detach())
        return loss
