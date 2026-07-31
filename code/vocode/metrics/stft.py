# This module:
# 1. Computes the multi-resolution STFT error between reference and
#    candidate waveforms through the auraloss implementation, under the
#    conventional three-resolution analysis grid
#
# Design decisions:
# - The three FFT, hop, and window triples are the widely used
#   Parallel-WaveGAN evaluation grid, so reported values are comparable
#   with the vocoder literature
# - Waveforms are truncated to their common length and evaluated as
#   float32 on the CPU, keeping the value independent of the accelerator
#   lane
#
# Author: Rahul Sawhney

from typing import ClassVar

import torch
from auraloss.freq import MultiResolutionSTFTLoss
from pydantic import BaseModel, ConfigDict

__all__: list[str] = ["MultiResolutionStftError", "MultiResolutionStftErrorConfig"]


class MultiResolutionStftErrorConfig(BaseModel):
    # Frozen analysis grid: parallel FFT-size, hop, and window triples.
    #
    # The three tuples are parallel and are read by position: entry i of
    # each describes one of the three resolutions the criterion evaluates
    # in parallel and sums. The arity is fixed at three by the declared
    # types, so a grid of any other width is rejected at validation rather
    # than silently truncated to the shortest tuple.
    #
    # Fields:
    #     fft_sizes: Transform size of each resolution.
    #         Default: ``(1024, 2048, 512)``.
    #     hop_sizes: Samples advanced between frames at each resolution.
    #         Default: ``(120, 240, 50)``.
    #     win_lengths: Analysis window length at each resolution.
    #         Default: ``(600, 1200, 240)``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    fft_sizes: tuple[int, int, int] = (1024, 2048, 512)
    hop_sizes: tuple[int, int, int] = (120, 240, 50)
    win_lengths: tuple[int, int, int] = (600, 1200, 240)


class MultiResolutionStftError:
    # Multi-resolution spectral distance over the configured grid.
    def __init__(self, configuration: MultiResolutionStftErrorConfig) -> None:
        # Binds the grid and constructs the auraloss criterion once.
        self._configuration: MultiResolutionStftErrorConfig = configuration
        self._loss: MultiResolutionSTFTLoss = MultiResolutionSTFTLoss(
            fft_sizes=list(configuration.fft_sizes),
            hop_sizes=list(configuration.hop_sizes),
            win_lengths=list(configuration.win_lengths)
        )

    def __call__(self, reference: torch.Tensor, candidate: torch.Tensor) -> float:
        # Truncates to the common length, shapes both signals into the
        # criterion's [batch, channel, time] layout on the CPU, and
        # evaluates without gradient tracking.
        #
        # Args:
        #     reference: The true signal, in any shape; leading axes are
        #         flattened into the criterion's layout.
        #     candidate: The synthesized signal, truncated to the reference
        #         length before analysis.
        #
        # Returns:
        #     The summed spectral distance over the three resolutions, on
        #     which lower is better and zero means an exact match. Argument
        #     order matters: the spectral convergence term normalizes by
        #     the reference magnitude, so the measure is genuinely
        #     asymmetric and the true signal must be passed first.
        minimum_samples: int = min(reference.shape[-1], candidate.shape[-1])
        reference_batch: torch.Tensor = reference[..., :minimum_samples].reshape(1, 1, -1).float().cpu()
        candidate_batch: torch.Tensor = candidate[..., :minimum_samples].reshape(1, 1, -1).float().cpu()
        with torch.no_grad():
            error_value: torch.Tensor = self._loss(candidate_batch, reference_batch)
        return float(error_value.item())

    @property
    def configuration(self) -> MultiResolutionStftErrorConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration
