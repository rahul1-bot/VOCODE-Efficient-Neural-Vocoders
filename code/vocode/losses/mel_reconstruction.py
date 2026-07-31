# This module:
# 1. Implements the mel reconstruction loss: L1 distance between reference
#    and synthesized log-mel spectrograms under a shape-agreement guard
#
# Design decisions:
# - Shape agreement is a hard requirement; silent broadcasting or cropping
#   would hide an extraction-protocol mismatch. A disagreement in the frame
#   axis usually means the reference and candidate mel spectrograms were
#   extracted under different hop or padding settings, which would make the
#   resulting loss value meaningless rather than merely imprecise
# - The guard is deliberately stricter than the spectral terms of the
#   APNet2 lineage, which crop paired tensors to their common extent: mel
#   supervision compares two views of the same utterance under one agreed
#   extraction protocol, so a mismatch here is a configuration defect, not
#   an off-by-one boundary effect
#
# Author: Rahul Sawhney

import torch
from torch import nn

__all__: list[str] = ["MelReconstructionLoss"]


class MelReconstructionLoss(nn.Module):
    # L1 log-mel reconstruction objective with strict shape agreement. This
    # is the perceptual anchor of every mel-supervised family in the
    # package: it is the one term that constrains the synthesized waveform
    # to the reference content rather than merely to the discriminators'
    # notion of realism. The class holds no parameters and no mutable
    # state, so one instance is shared freely across a composite objective.
    #
    # Integration: composed by the HiFi-GAN, BigVGAN, HiFTNet, Vocos,
    # APNet2, and RNDVoC objectives, each of which scales the returned
    # scalar by its own configured mel weight.
    def forward(self, reference_mel: torch.Tensor, candidate_mel: torch.Tensor) -> torch.Tensor:
        # Validates exact shape agreement between the two spectrograms and
        # then reduces their mean absolute difference. No broadcasting,
        # cropping, or padding is attempted, so the returned value always
        # compares frame to frame and band to band.
        #
        # Args:
        #     reference_mel: Log-mel spectrogram of the reference audio,
        #         serving as the regression target.
        #     candidate_mel: Log-mel spectrogram of the synthesized audio,
        #         extracted under the same protocol as the reference.
        #
        # Raises:
        #     ValueError: If the two spectrograms differ in shape at any
        #         axis; the message reports both shapes so the mismatched
        #         extraction setting is identifiable from the failure alone.
        #
        # Returns:
        #     A scalar tensor holding the mean absolute difference,
        #     unweighted; the composing objective applies its own mel
        #     weight.
        if reference_mel.shape != candidate_mel.shape:
            raise ValueError(
                f"Mel reconstruction tensors must have identical shape: "
                f"reference={tuple(reference_mel.shape)}, candidate={tuple(candidate_mel.shape)}"
            )
        return torch.nn.functional.l1_loss(candidate_mel, reference_mel)
