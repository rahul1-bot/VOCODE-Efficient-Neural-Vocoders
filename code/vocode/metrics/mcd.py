# This module:
# 1. Computes the mel-cepstral distortion between reference and candidate
#    waveforms: mel-cepstral coefficients from a DCT over the log-mel
#    spectrum, per-frame Euclidean distances, and the conventional decibel
#    scaling constant
#
# Design decisions:
# - The zeroth cepstral coefficient is excluded because it encodes overall
#   frame energy, which mel-cepstral distortion conventionally ignores in
#   favor of spectral envelope shape
# - The scaling constant (ten times the square root of two over the natural
#   log of ten) converts the coefficient distance to the decibel-domain
#   value reported in the vocoder literature
# - The analysis grid binds to the sample rate at construction, which is
#   why the metric sequence caches one instance per observed rate
#
# Author: Rahul Sawhney

from typing import ClassVar

import torch
import torchaudio
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

__all__: list[str] = ["MelCepstralDistortion", "MelCepstralDistortionConfig"]


class MelCepstralDistortionConfig(BaseModel):
    # Frozen analysis settings: sample rate, STFT and mel grid, retained
    # coefficient count, and the amplitude floor before log compression.
    #
    # Fields:
    #     sample_rate: Rate in hertz the mel filterbank is constructed for.
    #         The filterbank is built once at construction, which is why
    #         the metric sequence caches one instance per observed rate
    #         rather than rebuilding per utterance. Default: ``22050``,
    #         the corpus rate.
    #     n_fft: Transform size of the underlying spectrogram.
    #         Default: ``1024``.
    #     hop_length: Samples advanced between consecutive frames.
    #         Default: ``256``.
    #     mel_bands: Mel filters the magnitude spectrum is projected onto,
    #         which is also the input width of the cepstral transform.
    #         Default: ``80``.
    #     fmin_hertz: Lower edge of the mel filterbank.
    #         Default: ``0.0``.
    #     fmax_hertz: Upper edge of the mel filterbank.
    #         Default: ``8000.0``.
    #     coefficient_count: Cepstral coefficients the distance is measured
    #         over. The transform produces one more than this and the
    #         energy-carrying zeroth coefficient is then discarded, so this
    #         is the count that actually reaches the comparison.
    #         Default: ``13``.
    #     amplitude_floor: Lower bound applied to the mel magnitudes before
    #         the logarithm, so a silent band maps to a finite value rather
    #         than diverging. Default: ``1.0e-5``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    sample_rate: PositiveInt = 22050
    n_fft: PositiveInt = 1024
    hop_length: PositiveInt = 256
    mel_bands: PositiveInt = 80
    fmin_hertz: float = 0.0
    fmax_hertz: PositiveFloat = 8000.0
    coefficient_count: PositiveInt = 13
    amplitude_floor: PositiveFloat = 1.0e-5


class MelCepstralDistortion:
    # Decibel-domain cepstral distance over the configured analysis grid.
    #
    # The reported value is a diagnostic rather than a cross-model
    # endpoint. Frames are compared at matching positions with no
    # alignment step, so a systematic group delay between reference and
    # candidate inflates the distance for reasons unrelated to spectral
    # fidelity. A configuration suspected of such a delay is therefore not
    # ranked on this measure.
    def __init__(self, configuration: MelCepstralDistortionConfig) -> None:
        # Precomputes the scaling constant, the rate-bound mel transform,
        # and the orthonormal DCT matrix.
        self._configuration: MelCepstralDistortionConfig = configuration
        self._constant: float = 10.0 * (2.0 ** 0.5) / torch.log(torch.tensor(10.0)).item()
        self._mel_transform: torchaudio.transforms.MelSpectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=configuration.sample_rate,
            n_fft=configuration.n_fft,
            hop_length=configuration.hop_length,
            f_min=configuration.fmin_hertz,
            f_max=configuration.fmax_hertz,
            n_mels=configuration.mel_bands,
            power=1.0,
            center=True
        )
        self._dct_matrix: torch.Tensor = torchaudio.functional.create_dct(
            n_mfcc=configuration.coefficient_count + 1,
            n_mels=configuration.mel_bands,
            norm="ortho"
        )

    def __call__(self, reference: torch.Tensor, candidate: torch.Tensor) -> float:
        # Truncates to the common length, extracts both cepstral sequences,
        # and reduces the per-frame Euclidean coefficient distances to their
        # scaled mean.
        #
        # Args:
        #     reference: The true signal, in any shape; leading axes are
        #         flattened before analysis.
        #     candidate: The synthesized signal, truncated to the reference
        #         length before analysis.
        #
        # Returns:
        #     The distortion in decibels, on which lower is better and zero
        #     means an exact match. Because the energy-carrying zeroth
        #     cepstral coefficient is excluded, a uniform gain on either
        #     signal barely moves the value: the measure judges spectral
        #     envelope shape rather than loudness.
        minimum_samples: int = min(reference.shape[-1], candidate.shape[-1])
        reference_cepstra: torch.Tensor = self._cepstra(reference[..., :minimum_samples])
        candidate_cepstra: torch.Tensor = self._cepstra(candidate[..., :minimum_samples])
        coefficient_difference: torch.Tensor = reference_cepstra - candidate_cepstra
        per_frame_distance: torch.Tensor = torch.linalg.vector_norm(coefficient_difference, dim=-1)
        return float(self._constant * per_frame_distance.mean().item())

    @property
    def configuration(self) -> MelCepstralDistortionConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    def _cepstra(self, waveform: torch.Tensor) -> torch.Tensor:
        # Extracts the cepstral sequence of one waveform: floored log-mel
        # spectrum, orthonormal DCT projection, and removal of the
        # energy-carrying zeroth coefficient.
        flat_waveform: torch.Tensor = waveform.reshape(-1).float().cpu()
        mel_magnitude: torch.Tensor = self._mel_transform(flat_waveform)
        log_mel: torch.Tensor = torch.log(torch.clamp(mel_magnitude, min=self._configuration.amplitude_floor))
        cepstra: torch.Tensor = log_mel.transpose(-1, -2) @ self._dct_matrix
        return cepstra[..., 1:]
