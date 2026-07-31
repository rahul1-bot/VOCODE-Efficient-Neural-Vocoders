# This module:
# 1. Computes the mel-error metric: the mean absolute difference between two
#    log-mel spectrograms extracted under the evaluated model's declared
#    metric mel protocol
#
# Design decisions:
# - The metric operates on already-extracted spectrograms rather than
#   waveforms, so the caller controls the extraction protocol and the same
#   metric serves every architecture family
# - Band-count and frame-count agreement are hard requirements; silently
#   cropping either axis would hide a protocol mismatch between reference
#   and candidate extraction
#
# Author: Rahul Sawhney

import torch

__all__: list[str] = ["MelError"]


class MelError:
    # Mean absolute log-mel distance between two spectrograms of identical
    # shape. The class is stateless and takes no configuration, because the
    # extraction protocol that decides what the spectrograms mean belongs
    # to the caller.
    #
    # The reported value is a diagnostic rather than a cross-model
    # endpoint. Each configuration is measured under its own declared mel
    # protocol, and those protocols differ across the cohort in sample
    # rate, band count, and band normalization, so two models' mel errors
    # do not lie on one instrument and must not be ranked against each
    # other.
    def __call__(self, reference_mel: torch.Tensor, candidate_mel: torch.Tensor) -> float:
        # Validates band and frame agreement, then reduces the absolute
        # difference over all elements.
        #
        # Args:
        #     reference_mel: Log-mel spectrogram of the true signal, with
        #         bands on the second-to-last axis and frames on the last.
        #     candidate_mel: Log-mel spectrogram of the synthesized signal,
        #         required to agree on both of those axes.
        #
        # Returns:
        #     The mean absolute difference over every element, on which
        #     lower is better and zero means the two spectrograms are
        #     identical. Any leading axis, including a batch axis, is
        #     reduced along with the rest, so the value is one global mean
        #     rather than a per-item sequence.
        #
        # Raises:
        #     ValueError: If the band counts disagree, or if the frame
        #         counts disagree. Both name the two extents, because
        #         either mismatch means the reference and candidate were
        #         extracted under different protocols and cropping the
        #         shorter axis would hide that.
        if reference_mel.shape[-2] != candidate_mel.shape[-2]:
            raise ValueError(
                f"Mel-band counts must match: reference={reference_mel.shape[-2]}, "
                f"candidate={candidate_mel.shape[-2]}"
            )
        if reference_mel.shape[-1] != candidate_mel.shape[-1]:
            raise ValueError(
                f"Mel-frame counts must match: reference={reference_mel.shape[-1]}, "
                f"candidate={candidate_mel.shape[-1]}"
            )
        return torch.nn.functional.l1_loss(candidate_mel, reference_mel).item()
