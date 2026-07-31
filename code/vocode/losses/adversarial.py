# This module:
# 1. Implements the least-squares GAN objective shared by the adversarial
#    vocoder families: the generator term pulling fake logits toward one
#    and the discriminator term separating real from fake
#
# Design decisions:
# - Losses sum over every sub-discriminator's logits so ensemble members
#   contribute equally, per the reference recipes; no division by the
#   ensemble size is applied, so the effective adversarial magnitude scales
#   with the number of sub-discriminators a composite objective configures
# - The least-squares form replaces the saturating cross-entropy GAN
#   objective, matching the HiFi-GAN-family training dynamics: the quadratic
#   penalty keeps returning informative gradients after a sub-discriminator
#   has confidently separated reference from synthesized audio
# - An empty logit list yields a plain CPU zero scalar instead of raising, so
#   a composite objective that disables one ensemble remains additive at the
#   call site without a guarding branch
# - Each accumulator is seeded on the device of the first logit it consumes,
#   so the returned scalar already lives beside the tensors that produced it
#   and needs no placement by the caller
#
# Author: Rahul Sawhney

import torch
from torch import nn

__all__: list[str] = ["LeastSquaresGanLoss"]


class LeastSquaresGanLoss(nn.Module):
    # Least-squares adversarial objective over ensemble logit lists. Both
    # halves of the LSGAN game live here: the generator term that pulls
    # synthesized-audio logits toward the real target of one, and the
    # discriminator term that separates reference logits at one from
    # synthesized logits at zero. The class holds no parameters and no
    # mutable state, so a single instance is safely shared across every
    # discriminator ensemble a composite objective drives, and the two
    # methods may be called in either order within a step.
    #
    # Integration: the HiFi-GAN, BigVGAN, and HiFTNet composite objectives
    # each construct one instance and invoke it once per ensemble (period
    # and scale, period and resolution, or period and spectrogram
    # respectively), summing the per-ensemble scalars themselves rather than
    # passing concatenated lists, so every ensemble keeps its own weight.
    def generator_loss(self, fake_logits: list[torch.Tensor]) -> torch.Tensor:
        # Computes the generator-side least-squares term by accumulating
        # ``mean((1 - logit) ** 2)`` across every sub-discriminator output.
        # The generator is therefore rewarded for driving each ensemble
        # member's verdict on synthesized audio toward the real target of
        # one, and each sub-discriminator contributes through its own mean
        # regardless of its logit map's resolution.
        #
        # Args:
        #     fake_logits: One logit tensor per sub-discriminator, each
        #         evaluated on synthesized audio. Tensors of differing
        #         shapes are accepted because reduction is per tensor.
        #
        # Returns:
        #     A scalar tensor holding the summed least-squares term, placed
        #     on the device of the first logit and inheriting whatever
        #     autograd graph the inputs carry. An empty list returns a CPU
        #     zero scalar with no graph attached.
        loss: torch.Tensor = torch.zeros((), device=fake_logits[0].device) if fake_logits else torch.zeros(())
        for logit in fake_logits:
            loss: torch.Tensor = loss + torch.mean((1.0 - logit) ** 2)
        return loss

    def discriminator_loss(
        self,
        real_logits: list[torch.Tensor],
        fake_logits: list[torch.Tensor]
    ) -> torch.Tensor:
        # Computes the discriminator-side least-squares term by accumulating
        # ``mean((1 - real) ** 2) + mean(fake ** 2)`` for each positionally
        # aligned pair, training every ensemble member to score reference
        # audio at one and synthesized audio at zero. Pairing is positional
        # and non-strict, so accumulation stops at the shorter of the two
        # lists rather than raising on a length mismatch; callers that
        # require matched ensembles enforce that upstream.
        #
        # Args:
        #     real_logits: One logit tensor per sub-discriminator, each
        #         evaluated on reference audio. This list also determines
        #         the accumulator's device and the empty-input behavior.
        #     fake_logits: The corresponding logit tensors evaluated on
        #         synthesized audio, aligned to real_logits by position.
        #         Callers detach the synthesized audio before this call so
        #         the term shapes only the discriminators.
        #
        # Returns:
        #     A scalar tensor holding the summed real-versus-fake separation
        #     term, placed on the device of the first real logit. An empty
        #     real_logits list returns a CPU zero scalar.
        loss: torch.Tensor = torch.zeros((), device=real_logits[0].device) if real_logits else torch.zeros(())
        for real, fake in zip(real_logits, fake_logits):
            real_term: torch.Tensor = torch.mean((1.0 - real) ** 2)
            fake_term: torch.Tensor = torch.mean(fake ** 2)
            loss: torch.Tensor = loss + real_term + fake_term
        return loss
