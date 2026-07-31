# This module:
# 1. Computes the short-time objective intelligibility (STOI) score between
#    reference and candidate waveforms through the pystoi implementation
#
# Design decisions:
# - Both signals are resampled to the measure's ten-kilohertz operating
#   rate, which the STOI definition fixes
# - The classic (non-extended) variant is the default, matching the values
#   conventionally reported for vocoded speech
# - Waveforms are truncated to their common length before resampling so the
#   compared material is time-aligned
#
# Author: Rahul Sawhney

from typing import ClassVar

import numpy as np
import torch
import torchaudio
from pydantic import BaseModel, ConfigDict, PositiveInt
from pystoi import stoi as stoi_score

__all__: list[str] = ["Stoi", "StoiConfig"]


class StoiConfig(BaseModel):
    # Frozen STOI settings: the operating sample rate and the extended
    # variant flag.
    #
    # Fields:
    #     target_sample_rate: Rate in hertz both signals are resampled to
    #         before scoring; the STOI definition fixes this rate, so it is
    #         a setting only in the sense that the record states it
    #         explicitly. Default: ``10000``.
    #     extended: Whether the extended variant is evaluated in place of
    #         the classic one. The classic variant is the value
    #         conventionally reported for vocoded speech and stays
    #         non-negative; the extended variant is an uncleaned
    #         correlation that may fall below zero on badly degraded
    #         material, so the two are bounded differently when reported.
    #         Default: ``False``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    target_sample_rate: PositiveInt = 10000
    extended: bool = False


class Stoi:
    # Short-time objective intelligibility score in the configured variant.
    def __init__(self, configuration: StoiConfig) -> None:
        # Binds the operating settings.
        self._configuration: StoiConfig = configuration

    def __call__(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
        source_sample_rate: int
    ) -> float:
        # Truncates to the common unpadded length, resamples both signals to
        # the operating rate when needed, and scores on CPU arrays.
        # Truncation precedes resampling, so padding can never be resampled
        # into the compared material.
        #
        # Args:
        #     reference: The true signal, as a one-dimensional waveform.
        #     candidate: The synthesized signal to score against it, also
        #         one-dimensional. Samples beyond the reference length are
        #         truncated away before scoring.
        #     source_sample_rate: Rate both signals are currently sampled
        #         at. Resampling is skipped when it already equals the
        #         configured operating rate.
        #
        # Returns:
        #     The intelligibility score, on which higher is better. The
        #     classic variant occupies the unit interval and reaches one on
        #     an exact reconstruction; the extended variant reaches one
        #     likewise but may fall below zero on badly degraded material.
        #
        # Raises:
        #     ValueError: If either signal carries a channel axis, since
        #         the pystoi implementation consumes one-dimensional
        #         buffers only.
        minimum_samples: int = min(reference.shape[-1], candidate.shape[-1])
        reference_cropped: torch.Tensor = reference[..., :minimum_samples]
        candidate_cropped: torch.Tensor = candidate[..., :minimum_samples]
        target_rate: int = self._configuration.target_sample_rate
        if source_sample_rate != target_rate:
            reference_resampled: torch.Tensor = torchaudio.functional.resample(
                reference_cropped, orig_freq=source_sample_rate, new_freq=target_rate
            )
            candidate_resampled: torch.Tensor = torchaudio.functional.resample(
                candidate_cropped, orig_freq=source_sample_rate, new_freq=target_rate
            )
        else:
            reference_resampled: torch.Tensor = reference_cropped
            candidate_resampled: torch.Tensor = candidate_cropped
        reference_numpy: np.ndarray = reference_resampled.detach().to("cpu").numpy()
        candidate_numpy: np.ndarray = candidate_resampled.detach().to("cpu").numpy()
        return float(
            stoi_score(reference_numpy, candidate_numpy, target_rate, extended=self._configuration.extended)
        )

    @property
    def configuration(self) -> StoiConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration
