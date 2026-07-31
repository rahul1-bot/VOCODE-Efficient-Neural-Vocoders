# This module:
# 1. Computes the log-amplitude-spectrum root-mean-square error in decibels
#    between reference and candidate waveforms over a single centered STFT
#    grid
#
# Design decisions:
# - Amplitudes are floored before the logarithm so silent bins map to a
#   finite decibel floor instead of negative infinity
# - The decibel scale (twenty times the base-ten logarithm) makes the error
#   directly interpretable as spectral deviation in dB
# - Waveforms are truncated to their common length and analyzed as float32
#   on the CPU
#
# Author: Rahul Sawhney

from typing import ClassVar

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

__all__: list[str] = ["LogAmplitudeSpectrumRmse", "LogAmplitudeSpectrumRmseConfig"]


class LogAmplitudeSpectrumRmseConfig(BaseModel):
    # Frozen analysis settings: STFT grid and the amplitude floor applied
    # before log compression.
    #
    # Fields:
    #     n_fft: Transform size of the single analysis grid. It also fixes
    #         the window length, so the Hann window always spans the whole
    #         frame. Default: ``1024``.
    #     hop_length: Samples advanced between consecutive frames.
    #         Default: ``256``.
    #     amplitude_floor: Lower bound applied to the magnitude spectrum
    #         before the logarithm, so a silent bin maps to a finite
    #         decibel floor instead of negative infinity. Raising it
    #         narrows how far a silent bin can sit below an occupied one
    #         and therefore lowers the error a silent comparison reports.
    #         Default: ``1.0e-5``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    n_fft: PositiveInt = 1024
    hop_length: PositiveInt = 256
    amplitude_floor: PositiveFloat = 1.0e-5


class LogAmplitudeSpectrumRmse:
    # Decibel-domain spectral RMSE over one STFT grid.
    #
    # The reported value is a diagnostic rather than a cross-model
    # endpoint. Spectra are compared frame by frame with no alignment
    # step, so a systematic group delay between reference and candidate
    # inflates the error independently of spectral fidelity.
    def __init__(self, configuration: LogAmplitudeSpectrumRmseConfig) -> None:
        # Binds the settings and precomputes the Hann analysis window.
        self._configuration: LogAmplitudeSpectrumRmseConfig = configuration
        self._window: torch.Tensor = torch.hann_window(configuration.n_fft)

    def __call__(self, reference: torch.Tensor, candidate: torch.Tensor) -> float:
        # Truncates to the common length, converts both signals to decibel
        # spectra, and reduces the squared difference to its root mean.
        #
        # Args:
        #     reference: The true signal, in any shape; leading axes are
        #         flattened before analysis.
        #     candidate: The synthesized signal, truncated to the reference
        #         length before analysis.
        #
        # Returns:
        #     The spectral error in decibels, on which lower is better and
        #     zero means an exact match. Squaring the difference makes the
        #     value independent of which signal is passed first.
        minimum_samples: int = min(reference.shape[-1], candidate.shape[-1])
        reference_log: torch.Tensor = self._log_amplitude(reference[..., :minimum_samples])
        candidate_log: torch.Tensor = self._log_amplitude(candidate[..., :minimum_samples])
        squared_error: torch.Tensor = (reference_log - candidate_log) ** 2
        return float(torch.sqrt(squared_error.mean()).item())

    @property
    def configuration(self) -> LogAmplitudeSpectrumRmseConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    def _log_amplitude(self, waveform: torch.Tensor) -> torch.Tensor:
        # Produces the floored decibel amplitude spectrum of one waveform
        # on the configured centered STFT grid.
        flat_waveform: torch.Tensor = waveform.reshape(-1).float().cpu()
        spectrum: torch.Tensor = torch.stft(
            flat_waveform,
            n_fft=self._configuration.n_fft,
            hop_length=self._configuration.hop_length,
            win_length=self._configuration.n_fft,
            window=self._window,
            center=True,
            return_complex=True
        )
        amplitude: torch.Tensor = torch.clamp(
            spectrum.abs(),
            min=self._configuration.amplitude_floor
        )
        return 20.0 * torch.log10(amplitude)
