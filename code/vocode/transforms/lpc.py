# This module:
# 1. Computes LPCNet frame features from raw waveforms as a documented
#    PyTorch reimplementation of the reference C feature extractor:
#    triangular Bark-frequency band energies with DCT cepstra, autocorrelation
#    pitch search, and Levinson-Durbin LPC coefficients recovered from the
#    band-interpolated spectrum
# 2. Implements the reference 8-bit mu-law companding protocol used for
#    LPCNet's sample-domain prediction targets
#
# Design decisions:
# - Band edges, window sizes, the lag-window taper, and the numeric floors
#   follow the reference implementation so extracted features are
#   protocol-compatible with the published recipe
# - LPC coefficients are recovered from the band energies rather than the
#   raw spectrum, mirroring the reference pipeline in which the decoder
#   sees only band-resolution spectral information
# - Precomputed band weights, the DCT matrix, and the analysis window are
#   non-persistent buffers: they follow device movement but stay out of
#   checkpoints because they are derivable from the configuration
#
# Author: Rahul Sawhney

import math
from typing import ClassVar, override

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from torch import nn

__all__: list[str] = ["LpcnetFeatureConfig", "LpcnetFeatureExtractor", "LpcnetFeatures", "LpcnetMuLaw"]


class LpcnetFeatureConfig(BaseModel):
    # Frozen LPCNet analysis protocol; defaults follow the 16 kHz reference.
    #
    # Fields:
    #     sample_rate: Waveform sample rate, in hertz. Default: ``16000``.
    #     frame_size: Hop between analysis frames, in samples.
    #         Default: ``160``.
    #     window_size: Analysis window length, in samples. Default: ``320``.
    #     band_count: Number of Bark-frequency bands. Default: ``18``.
    #     lpc_order: Order of the linear-prediction filter. Default: ``16``.
    #     minimum_pitch_lag: Lower bound of the pitch search, in samples.
    #         A silent or aperiodic frame reports this floor, because the
    #         search always returns its best candidate. Default: ``32``.
    #     maximum_pitch_lag: Upper bound of the pitch search, in samples.
    #         It stays within the byte-wide index domain, because the
    #         detected lag addresses an embedding table in the decoder.
    #         Default: ``255``.
    #     preemphasis_coefficient: First-order preemphasis constant of the
    #         reference recipe, recorded here as part of the analysis
    #         protocol. The extractor in this module does not apply it: the
    #         band analysis runs on the unfiltered waveform, and the
    #         preemphasis filter itself is applied by the LPCNet model from
    #         its own configuration. Default: ``0.85``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    sample_rate: PositiveInt = 16000
    frame_size: PositiveInt = 160
    window_size: PositiveInt = 320
    band_count: PositiveInt = 18
    lpc_order: PositiveInt = 16
    minimum_pitch_lag: PositiveInt = 32
    maximum_pitch_lag: PositiveInt = 255
    preemphasis_coefficient: PositiveFloat = 0.85


class LpcnetFeatures(BaseModel):
    # Feature bundle produced by the LPCNet frame analysis for conditioning and prediction.
    # The fields mirror the reference feature file layout: cepstra, pitch, and LPC coefficients.
    # Every tensor is indexed by frame along its second axis, so one frame index selects the
    # same instant in all three.
    #
    # Fields:
    #     conditioning: Frame conditioning shaped
    #         ``[batch, frames, band_count + 2]``. The leading
    #         ``band_count`` columns are the DCT cepstra of the log band
    #         energies; the final two columns are the detected pitch lag
    #         rescaled by ``(lag - 100) / 50`` and the normalized pitch
    #         correlation in [-1, 1].
    #     pitch_index: Detected pitch lag in samples shaped
    #         ``[batch, frames]``, held as long integers clamped into the
    #         byte-wide domain the decoder's embedding table addresses.
    #     lpc_coefficients: Linear-prediction coefficients shaped
    #         ``[batch, frames, lpc_order]``, recovered per frame from the
    #         band-interpolated spectrum rather than from the raw one.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    conditioning: torch.Tensor
    pitch_index: torch.Tensor
    lpc_coefficients: torch.Tensor


class LpcnetMuLaw:
    # Transform component implementing the reference 8-bit mu-law companding protocol.
    # The scale constants match the reference implementation exactly.
    def __init__(self) -> None:
        # Precomputes the int16-to-index scale constants and the log-256
        # divisor of the reference companding formula.
        self._scale: float = 255.0 / 32768.0
        self._scale_inverse: float = 32768.0 / 255.0
        self._log_256: float = math.log(256.0)

    def linear_to_mulaw(self, linear: torch.Tensor) -> torch.Tensor:
        # Maps int16-scaled linear samples onto the continuous 0-255 mu-law index domain.
        # The logarithmic law spends index resolution where speech energy sits: quiet samples
        # are separated finely near the centre while loud ones share coarser steps.
        #
        # Args:
        #     linear: Samples on the int16 scale, that is in
        #         [-32768, 32768] rather than in [-1, 1].
        #
        # Returns:
        #     Continuous indices in [0.0, 255.0], not yet rounded to
        #     integers. Silence maps to the centre index 128, full scale
        #     maps to the two bounds, the mapping is symmetric about the
        #     centre and non-decreasing, and inputs beyond full scale are
        #     clamped into the domain rather than wrapped around it.
        sign: torch.Tensor = torch.sign(linear)
        magnitude: torch.Tensor = torch.abs(linear)
        companded: torch.Tensor = sign * (128.0 * torch.log1p(self._scale * magnitude) / self._log_256)
        return torch.clamp(128.0 + companded, 0.0, 255.0)

    def mulaw_to_linear(self, mulaw: torch.Tensor) -> torch.Tensor:
        # Maps 0-255 mu-law indices back onto int16-scaled linear samples.
        #
        # Args:
        #     mulaw: Indices in [0.0, 255.0] as produced by the companding
        #         direction, accepted continuously so a decoder may expand
        #         an interpolated index.
        #
        # Returns:
        #     Samples back on the int16 scale. This inverts the companding
        #     exactly for any sample that did not saturate, so a round trip
        #     recovers the original amplitude to within float epsilon, and
        #     the centre index expands back to silence.
        centered: torch.Tensor = mulaw - 128.0
        sign: torch.Tensor = torch.sign(centered)
        magnitude: torch.Tensor = torch.abs(centered)
        return sign * self._scale_inverse * (torch.exp(magnitude / 128.0 * self._log_256) - 1.0)


class LpcnetFeatureExtractor(nn.Module):
    # Transform component computing LPCNet frame features from raw waveforms.
    # This is a documented PyTorch reimplementation of the reference C feature extractor:
    # triangular Bark-frequency band energies with DCT cepstra, autocorrelation pitch search,
    # and Levinson-Durbin LPC coefficients recovered from the band-interpolated spectrum.
    def __init__(self, configuration: LpcnetFeatureConfig) -> None:
        # Precomputes the analysis machinery from the protocol: the
        # triangular band weights over the FFT grid, the orthonormal DCT
        # matrix, and the periodic Hann window, all registered as
        # non-persistent buffers.
        super().__init__()
        self._configuration: LpcnetFeatureConfig = configuration
        band_edges_hz: tuple[float, ...] = (
            0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1600.0,
            2000.0, 2400.0, 2800.0, 3200.0, 4000.0, 4800.0, 5600.0, 6800.0, 8000.0
        )
        if len(band_edges_hz) != configuration.band_count:
            raise ValueError(
                f"Band center table must have {configuration.band_count} entries, got {len(band_edges_hz)}"
            )
        bin_count: int = configuration.window_size // 2 + 1
        bin_frequencies: torch.Tensor = torch.linspace(
            0.0,
            configuration.sample_rate / 2.0,
            bin_count
        )
        band_weights: torch.Tensor = self._build_triangular_band_weights(
            band_centers_hz=band_edges_hz,
            bin_frequencies=bin_frequencies
        )
        dct_matrix: torch.Tensor = self._build_dct_matrix(configuration.band_count)
        analysis_window: torch.Tensor = torch.hann_window(configuration.window_size, periodic=True)
        self.register_buffer("_band_weights", band_weights, persistent=False)
        self.register_buffer("_dct_matrix", dct_matrix, persistent=False)
        self.register_buffer("_analysis_window", analysis_window, persistent=False)

    @property
    def configuration(self) -> LpcnetFeatureConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    @override
    def forward(self, waveform: torch.Tensor) -> LpcnetFeatures:
        # Module-call form of extract, so the extractor composes like any
        # other torch transform.
        return self.extract(waveform)

    def extract(self, waveform: torch.Tensor) -> LpcnetFeatures:
        # Computes the conditioning features, pitch indices, and LPC coefficients per frame.
        # The analysis runs in five stages: frame the waveform, project each windowed spectrum
        # onto the Bark-frequency bands, compress and decorrelate those energies into cepstra,
        # search the waveform for its pitch period, and recover the prediction filter from the
        # band energies. Every stage is a batched tensor operation, so utterances in one batch
        # are analysed independently and never influence each other.
        #
        # Args:
        #     waveform: Audio at the protocol's sample rate, shaped
        #         ``[time]``, ``[batch, time]``, or ``[batch, 1, time]``.
        #
        # Returns:
        #     An LpcnetFeatures bundle whose three tensors share one frame
        #     axis. Extraction is deterministic, so repeating it on the
        #     same waveform returns identical values.
        #
        # Raises:
        #     ValueError: If the waveform rank is not one the analysis
        #         accepts.
        prepared: torch.Tensor = self._prepare_waveform(waveform)
        frame_size: int = self._configuration.frame_size
        window_size: int = self._configuration.window_size
        # Front padding by the window's excess over one hop makes the frame count exactly
        # time // frame_size, so a feature row exists for every hop of the original signal
        # including the first, which has no preceding context of its own.
        padded: torch.Tensor = torch.nn.functional.pad(
            prepared,
            (window_size - frame_size, 0)
        )
        frames: torch.Tensor = padded.unfold(-1, window_size, frame_size)
        analysis_window: torch.Tensor = self._buffer("_analysis_window").to(dtype=frames.dtype)
        spectrum: torch.Tensor = torch.fft.rfft(frames * analysis_window, dim=-1)
        power: torch.Tensor = spectrum.real.pow(2) + spectrum.imag.pow(2)
        band_weights: torch.Tensor = self._buffer("_band_weights").to(dtype=power.dtype)
        # The energy floor bounds the logarithm below, so a silent frame yields log10 of the
        # floor in every band rather than negative infinity; the orthonormal DCT then turns
        # those band energies into decorrelated cepstra.
        band_energies: torch.Tensor = torch.einsum("bfk,nk->bfn", power, band_weights).clamp_min(1e-2)
        log_band_energies: torch.Tensor = torch.log10(band_energies)
        dct_matrix: torch.Tensor = self._buffer("_dct_matrix").to(dtype=power.dtype)
        cepstra: torch.Tensor = torch.einsum("bfn,cn->bfc", log_band_energies, dct_matrix)
        # Pitch is searched on the unframed waveform, so a period longer than one analysis
        # window is still visible to the correlation.
        pitch_lag, pitch_correlation = self._search_pitch(prepared, frame_count=frames.shape[1])
        # The lag is centred and scaled into a small numeric range before it joins the cepstra,
        # so the conditioning row carries no column with a disproportionate magnitude.
        pitch_feature: torch.Tensor = (pitch_lag.to(cepstra.dtype) - 100.0) / 50.0
        conditioning: torch.Tensor = torch.cat(
            [cepstra, pitch_feature.unsqueeze(-1), pitch_correlation.unsqueeze(-1)],
            dim=-1
        )
        lpc_coefficients: torch.Tensor = self._compute_lpc_from_bands(band_energies)
        return LpcnetFeatures(
            conditioning=conditioning,
            pitch_index=pitch_lag.clamp(0, 255).to(torch.long),
            lpc_coefficients=lpc_coefficients
        )

    def _prepare_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        # Normalizes accepted waveform shapes to the [batch, time] layout
        # the frame analysis operates on.
        if waveform.ndim == 1:
            return waveform.unsqueeze(0)
        if waveform.ndim == 2:
            return waveform
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            return waveform.squeeze(1)
        raise ValueError(f"Expected waveform shape [time], [batch, time], or [batch, 1, time], got {tuple(waveform.shape)}")

    def _buffer(self, name: str) -> torch.Tensor:
        # Returns a registered buffer with an explicit tensor type for the type checker.
        buffer: torch.Tensor | None = getattr(self, name)
        if not isinstance(buffer, torch.Tensor):
            raise RuntimeError(f"Buffer {name} is not initialized")
        return buffer

    def _build_triangular_band_weights(
        self,
        band_centers_hz: tuple[float, ...],
        bin_frequencies: torch.Tensor
    ) -> torch.Tensor:
        # Builds the triangular interpolation weights mapping FFT bins onto Bark-frequency bands.
        # Each band rises linearly from the preceding centre to its own and falls to the next,
        # so neighbouring triangles overlap and every interior bin contributes to exactly two
        # bands. The first and last bands have no neighbour on one side and borrow a synthetic
        # edge instead, narrow below the lowest centre and wide above the highest, which
        # matches the widening bandwidths of the reference band table. The direct-current bin
        # is pinned at unit weight in the lowest band.
        band_count: int = len(band_centers_hz)
        bin_count: int = bin_frequencies.shape[0]
        weights: torch.Tensor = torch.zeros(band_count, bin_count)
        for band_index in range(band_count):
            center: float = band_centers_hz[band_index]
            lower: float = band_centers_hz[band_index - 1] if band_index > 0 else center - 200.0
            upper: float = band_centers_hz[band_index + 1] if band_index < band_count - 1 else center + 1200.0
            for bin_index in range(bin_count):
                frequency: float = float(bin_frequencies[bin_index].item())
                if lower < frequency <= center:
                    weights[band_index, bin_index] = (frequency - lower) / (center - lower)
                elif center < frequency < upper:
                    weights[band_index, bin_index] = (upper - frequency) / (upper - center)
                elif frequency == center:
                    weights[band_index, bin_index] = 1.0
        weights[0, 0] = 1.0
        return weights

    def _build_dct_matrix(self, size: int) -> torch.Tensor:
        # Builds the orthonormal DCT-II matrix used for the cepstral projection.
        matrix: torch.Tensor = torch.zeros(size, size)
        for row in range(size):
            for column in range(size):
                matrix[row, column] = math.cos(math.pi * row * (column + 0.5) / size)
        matrix[0, :] = matrix[0, :] * math.sqrt(1.0 / size)
        matrix[1:, :] = matrix[1:, :] * math.sqrt(2.0 / size)
        return matrix

    def _search_pitch(self, waveform: torch.Tensor, frame_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Searches the normalized autocorrelation peak per frame for period and correlation.
        # Every candidate lag in the configured range is scored by correlating the frame with
        # the signal that far back, normalized by the geometric mean of the two energies so
        # the score measures waveform similarity rather than loudness.
        #
        # Args:
        #     waveform: The unframed ``[batch, time]`` signal, so lags
        #         longer than one analysis window remain reachable.
        #     frame_count: Number of frames to score, matching the frame
        #         axis the band analysis produced.
        #
        # Returns:
        #     The best lag in samples and its correlation, both shaped
        #     ``[batch, frames]``. The lag always lies inside the
        #     configured search range and the correlation inside [-1, 1].
        #     A normalized autocorrelation peaks equally at every integer
        #     multiple of the true period, so the reported lag may be an
        #     octave of it. Energy floors keep the score defined on silent
        #     frames, where every candidate scores zero and the search
        #     therefore reports its lower bound.
        frame_size: int = self._configuration.frame_size
        minimum_lag: int = self._configuration.minimum_pitch_lag
        maximum_lag: int = self._configuration.maximum_pitch_lag
        context: int = maximum_lag + frame_size
        padded: torch.Tensor = torch.nn.functional.pad(waveform, (context, 0))
        batch_size: int = waveform.shape[0]
        device: torch.device = waveform.device
        frame_starts: torch.Tensor = torch.arange(frame_count, device=device) * frame_size + context
        sample_offsets: torch.Tensor = torch.arange(frame_size, device=device)
        current_indices: torch.Tensor = frame_starts.unsqueeze(-1) + sample_offsets
        current_frames: torch.Tensor = padded[:, current_indices.reshape(-1)].reshape(
            batch_size,
            frame_count,
            frame_size
        )
        current_energy: torch.Tensor = current_frames.pow(2).sum(dim=-1).clamp_min(1e-6)
        lags: torch.Tensor = torch.arange(minimum_lag, maximum_lag + 1, device=device)
        correlations: list[torch.Tensor] = []
        for lag_value in lags.tolist():
            lag_indices: torch.Tensor = current_indices - int(lag_value)
            lagged_frames: torch.Tensor = padded[:, lag_indices.reshape(-1)].reshape(
                batch_size,
                frame_count,
                frame_size
            )
            lagged_energy: torch.Tensor = lagged_frames.pow(2).sum(dim=-1).clamp_min(1e-6)
            cross_correlation: torch.Tensor = (current_frames * lagged_frames).sum(dim=-1)
            correlations.append(cross_correlation / torch.sqrt(current_energy * lagged_energy))
        correlation_stack: torch.Tensor = torch.stack(correlations, dim=-1)
        best_correlation: torch.Tensor
        best_lag_offset: torch.Tensor
        best_correlation, best_lag_offset = correlation_stack.max(dim=-1)
        best_lag: torch.Tensor = best_lag_offset + minimum_lag
        return best_lag, best_correlation.clamp(-1.0, 1.0)

    def _compute_lpc_from_bands(self, band_energies: torch.Tensor) -> torch.Tensor:
        # Recovers LPC coefficients per frame from the band-interpolated power spectrum.
        # Deriving the filter from band energies rather than the raw spectrum is deliberate: it
        # mirrors the reference pipeline, in which the decoder only ever sees band-resolution
        # spectral information, so the filter the encoder solves for is one the decoder could
        # have solved for too.
        #
        # Args:
        #     band_energies: Clamped band energies shaped
        #         ``[batch, frames, band_count]``.
        #
        # Returns:
        #     Prediction coefficients shaped ``[batch, frames, lpc_order]``.
        band_weights: torch.Tensor = self._buffer("_band_weights").to(dtype=band_energies.dtype)
        # Spreading the band energies back over the FFT grid divides by the total weight each
        # bin received, so bins covered by two overlapping triangles are not counted twice.
        weight_totals: torch.Tensor = band_weights.sum(dim=0).clamp_min(1e-6)
        interpolated_power: torch.Tensor = torch.einsum(
            "bfn,nk->bfk",
            band_energies,
            band_weights
        ) / weight_totals
        # Inverse-transforming a power spectrum yields the autocorrelation sequence, of which
        # only the lags up to the filter order enter the normal equations.
        autocorrelation: torch.Tensor = torch.fft.irfft(
            interpolated_power,
            n=self._configuration.window_size,
            dim=-1
        )[..., : self._configuration.lpc_order + 1]
        lag_indices: torch.Tensor = torch.arange(
            self._configuration.lpc_order + 1,
            device=autocorrelation.device,
            dtype=autocorrelation.dtype
        )
        # The lag-window taper of the reference recipe attenuates higher lags, which broadens
        # the spectral peaks the filter models and keeps the resulting synthesis filter stable.
        lag_window: torch.Tensor = 1.0 - 0.003 * lag_indices.pow(2)
        conditioned: torch.Tensor = autocorrelation * lag_window
        # Lifting the zero lag slightly is a ridge on the normal equations: it guarantees a
        # positive-definite system, so the recursion below cannot divide by a vanished error
        # term on frames whose spectrum is nearly flat or nearly silent.
        conditioned: torch.Tensor = torch.cat(
            [conditioned[..., :1] * 1.0001 + 1e-4, conditioned[..., 1:]],
            dim=-1
        )
        return self._levinson_durbin(conditioned)

    def _levinson_durbin(self, autocorrelation: torch.Tensor) -> torch.Tensor:
        # Solves the Toeplitz normal equations per frame with the Levinson-Durbin recursion.
        # Each step derives one reflection coefficient from the residual error, extends the
        # filter by one order, and updates the earlier coefficients from their own reversal;
        # the prediction error decreases monotonically and is floored so a degenerate frame
        # cannot produce a division by zero. Every operation is batched over frames, so one
        # recursion solves all frames of all utterances at once.
        #
        # Args:
        #     autocorrelation: Conditioned autocorrelation lags shaped
        #         ``[batch, frames, lpc_order + 1]``, zero lag first.
        #
        # Returns:
        #     Prediction coefficients shaped ``[batch, frames, lpc_order]``.
        order: int = self._configuration.lpc_order
        error: torch.Tensor = autocorrelation[..., 0]
        coefficients: torch.Tensor = torch.zeros(
            *autocorrelation.shape[:-1],
            order,
            device=autocorrelation.device,
            dtype=autocorrelation.dtype
        )
        for step in range(order):
            accumulator: torch.Tensor = autocorrelation[..., step + 1].clone()
            for previous in range(step):
                accumulator: torch.Tensor = accumulator + coefficients[..., previous] * autocorrelation[..., step - previous]
            reflection: torch.Tensor = -accumulator / error.clamp_min(1e-9)
            updated: torch.Tensor = coefficients.clone()
            updated[..., step] = reflection
            for previous in range(step):
                updated[..., previous] = (
                    coefficients[..., previous]
                    + reflection * coefficients[..., step - 1 - previous]
                )
            coefficients: torch.Tensor = updated
            error: torch.Tensor = error * (1.0 - reflection.pow(2)).clamp_min(1e-9)
        return coefficients
