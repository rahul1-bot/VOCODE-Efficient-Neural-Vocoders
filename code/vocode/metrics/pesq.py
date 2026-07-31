# This module:
# 1. Computes the PESQ perceptual quality score between reference and
#    candidate waveforms through the ITU-T P.862 reference implementation
#
# Design decisions:
# - Both signals are resampled to the PESQ operating rate (sixteen
#   kilohertz in wideband mode) because the standard defines the measure
#   only at its operating rates
# - Wideband mode is the default because the evaluated vocoders synthesize
#   full-band speech
# - Waveforms are truncated to their common length before resampling so
#   the compared material is time-aligned
#
# Author: Rahul Sawhney

from typing import ClassVar, Literal

import numpy as np
import torch
import torchaudio
from pesq import pesq as pesq_score
from pydantic import BaseModel, ConfigDict, PositiveInt

__all__: list[str] = ["Pesq", "PesqConfig"]


class PesqConfig(BaseModel):
    # Frozen PESQ settings: the operating sample rate and the wideband or
    # narrowband mode.
    #
    # The two fields are chosen together rather than independently: the
    # reference implementation accepts only eight and sixteen kilohertz and
    # refuses the wideband mode at the lower of the two, so a rate and mode
    # pairing outside that surface is rejected when scoring runs.
    #
    # Fields:
    #     target_sample_rate: Rate in hertz both signals are resampled to
    #         before scoring, because the standard defines the measure only
    #         at its operating rates. Default: ``16000``, the wideband
    #         operating rate.
    #     mode: Band variant of the standard, ``"wb"`` for wideband or
    #         ``"nb"`` for narrowband. Wideband is the default because the
    #         evaluated vocoders synthesize full-band speech.
    #         Default: ``"wb"``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    target_sample_rate: PositiveInt = 16000
    mode: Literal["wb", "nb"] = "wb"


class Pesq:
    # ITU-T P.862 perceptual quality score in the configured mode.
    def __init__(self, configuration: PesqConfig) -> None:
        # Binds the operating settings.
        self._configuration: PesqConfig = configuration

    def __call__(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
        source_sample_rate: int
    ) -> float:
        # Truncates to the common unpadded length, resamples both signals to
        # the operating rate when needed, and scores through the reference
        # implementation on CPU arrays. Truncation precedes resampling, so
        # padding can never be resampled into the compared material.
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
        #     The predicted mean-opinion score, on which higher is better.
        #     The wideband mode spans roughly one to four and two thirds,
        #     with the upper end reached by an exact reconstruction. The
        #     standard normalizes active speech level internally, so the
        #     score is invariant to a uniform gain on the candidate.
        #
        # Raises:
        #     ValueError: If either signal carries a channel axis, since
        #         the reference implementation consumes one-dimensional
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
        return float(pesq_score(target_rate, reference_numpy, candidate_numpy, self._configuration.mode))

    @property
    def configuration(self) -> PesqConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration
