# This module:
# 1. Verifies the MelCepstralDistortionConfig validation surface: shipped
#    defaults, frozen semantics, forbidden extras, and the positive-grid and
#    positive-floor constraints
# 2. Verifies scoring behavior: identical waveforms give exactly zero, differing
#    waveforms give a strictly positive decibel distance, and the distance grows
#    monotonically with additive noise
# 3. Verifies the analysis grid: common-length truncation, shape flattening,
#    the effect of changing the mel band count, and the gain invariance that
#    follows from excluding the energy-carrying zeroth cepstral coefficient
#
# Design decisions:
# - Test material is a deterministic harmonic stack under a slow amplitude
#   envelope at 4096 samples, which fills seventeen analysis frames on the
#   default grid while staying inexpensive; a fixed torch seed precedes every
#   random tensor so distances never drift between runs
# - The identical-input case is asserted as exact zero because the computation
#   is deterministic pure math; every other value is bounded or ordered rather
#   than pinned
# - Gain invariance is the observable consequence of dropping the zeroth
#   coefficient: scaling a waveform shifts its log-mel spectrum uniformly,
#   which lands entirely in the discarded direct-current cepstral term, so it
#   is asserted with amplification where the amplitude floor cannot clip
# - The decibel scaling constant is not asserted directly: recomputing it in
#   the test would restate the implementation instead of checking it, so the
#   scale is exercised through zero, ordering, and invariance instead
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError

from vocode.metrics.mcd import MelCepstralDistortion, MelCepstralDistortionConfig


class HarmonicWaveformBuilder:
    # Builds deterministic speech-shaped waveforms and noise-degraded variants.
    def __init__(self, sample_rate: int, sample_count: int) -> None:
        # Binds the analysis grid and the fixed seed backing every draw.
        self._sample_rate: int = sample_rate
        self._sample_count: int = sample_count
        self._harmonic_frequencies: list[float] = [120.0, 240.0, 360.0, 480.0, 720.0]
        self._noise_seed: int = 20260730

    def clean(self) -> torch.Tensor:
        # Sums decaying harmonics under a three-hertz amplitude envelope.
        times: torch.Tensor = torch.arange(self._sample_count, dtype=torch.float32) / self._sample_rate
        harmonics: torch.Tensor = torch.zeros(self._sample_count)
        index: int
        frequency: float
        for index, frequency in enumerate(self._harmonic_frequencies):
            harmonics: torch.Tensor = harmonics + (0.6 ** index) * torch.sin(
                2.0 * torch.pi * frequency * times
            )
        envelope: torch.Tensor = 0.5 + 0.5 * torch.sin(2.0 * torch.pi * 3.0 * times)
        return 0.3 * harmonics * envelope

    def noise(self, noise_scale: float) -> torch.Tensor:
        # Draws a reproducible gaussian sequence at the requested amplitude.
        torch.manual_seed(self._noise_seed)
        return noise_scale * torch.randn(self._sample_count)

    def degraded(self, noise_scale: float) -> torch.Tensor:
        # Adds the reproducible noise sequence onto the clean waveform.
        return self.clean() + self.noise(noise_scale)


class MelCepstralDistortionConfigurationValidationTest(unittest.TestCase):
    # Verifies the frozen analysis-grid record and its constraints.
    def setUp(self) -> None:
        # Builds the default record shared by the validation checks.
        self._configuration: MelCepstralDistortionConfig = MelCepstralDistortionConfig()

    def test_default_configuration_matches_the_study_analysis_grid(self) -> None:
        # The shipped defaults are the grid the reported values were produced on.
        self.assertEqual(self._configuration.sample_rate, 22050)
        self.assertEqual(self._configuration.n_fft, 1024)
        self.assertEqual(self._configuration.hop_length, 256)
        self.assertEqual(self._configuration.mel_bands, 80)
        self.assertEqual(self._configuration.coefficient_count, 13)

    def test_configuration_rejects_unknown_field(self) -> None:
        # A forbidden extra turns a mistyped setting into an immediate failure.
        with self.assertRaises(ValidationError):
            MelCepstralDistortionConfig(mel_channels=80)

    def test_configuration_rejects_non_positive_grid_extents(self) -> None:
        # Every grid extent is a positive integer, so zero is refused outright.
        with self.assertRaises(ValidationError):
            MelCepstralDistortionConfig(n_fft=0)
        with self.assertRaises(ValidationError):
            MelCepstralDistortionConfig(hop_length=0)
        with self.assertRaises(ValidationError):
            MelCepstralDistortionConfig(mel_bands=0)
        with self.assertRaises(ValidationError):
            MelCepstralDistortionConfig(coefficient_count=0)

    def test_configuration_rejects_non_positive_amplitude_floor(self) -> None:
        # A non-positive floor would let the logarithm diverge.
        with self.assertRaises(ValidationError):
            MelCepstralDistortionConfig(amplitude_floor=0.0)
        with self.assertRaises(ValidationError):
            MelCepstralDistortionConfig(amplitude_floor=-1.0)

    def test_configuration_rejects_non_positive_upper_band_edge(self) -> None:
        # The upper mel edge must lie above direct current.
        with self.assertRaises(ValidationError):
            MelCepstralDistortionConfig(fmax_hertz=0.0)

    def test_configuration_rejects_non_integer_sample_rate(self) -> None:
        # Strict validation refuses a float where an integer rate is declared.
        with self.assertRaises(ValidationError):
            MelCepstralDistortionConfig(sample_rate=22050.0)

    def test_configuration_is_frozen(self) -> None:
        # A bound configuration cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.mel_bands = 40


class MelCepstralDistortionScoringTest(unittest.TestCase):
    # Verifies the decibel-domain cepstral distance over agreeing waveforms.
    def setUp(self) -> None:
        # Prepares the default metric and its reference waveform.
        self._builder: HarmonicWaveformBuilder = HarmonicWaveformBuilder(22050, 4096)
        self._metric: MelCepstralDistortion = MelCepstralDistortion(MelCepstralDistortionConfig())
        self._reference: torch.Tensor = self._builder.clean()

    def test_identical_waveforms_give_exactly_zero(self) -> None:
        # Matching cepstral sequences leave no per-frame distance to accumulate.
        distortion: float = self._metric(self._reference, self._reference.clone())
        self.assertEqual(
            distortion,
            0.0,
            msg=f"Identical waveforms gave {distortion} instead of zero distortion"
        )

    def test_differing_waveforms_give_a_positive_distortion(self) -> None:
        # Any spectral envelope disagreement registers as positive distortion.
        distortion: float = self._metric(self._reference, self._builder.degraded(0.05))
        self.assertGreater(distortion, 0.0, msg=f"Differing waveforms gave {distortion}")

    def test_distortion_grows_as_additive_noise_grows(self) -> None:
        # The distance orders degradation strength across a widening noise sweep.
        light_distortion: float = self._metric(self._reference, self._builder.degraded(0.01))
        moderate_distortion: float = self._metric(self._reference, self._builder.degraded(0.1))
        heavy_distortion: float = self._metric(self._reference, self._builder.degraded(0.5))
        self.assertLess(
            light_distortion,
            moderate_distortion,
            msg=f"Light noise gave {light_distortion}, not below moderate {moderate_distortion}"
        )
        self.assertLess(
            moderate_distortion,
            heavy_distortion,
            msg=f"Moderate noise gave {moderate_distortion}, not below heavy {heavy_distortion}"
        )

    def test_distortion_is_never_negative(self) -> None:
        # A scaled mean of euclidean norms cannot fall below zero.
        distortion: float = self._metric(self._reference, self._builder.noise(0.3))
        self.assertIsInstance(distortion, float)
        self.assertGreaterEqual(distortion, 0.0, msg=f"Distortion {distortion} went negative")

    def test_distortion_is_symmetric_in_its_arguments(self) -> None:
        # Euclidean coefficient distances do not depend on argument order.
        candidate: torch.Tensor = self._builder.degraded(0.05)
        forward_distortion: float = self._metric(self._reference, candidate)
        reversed_distortion: float = self._metric(candidate, self._reference)
        self.assertAlmostEqual(
            forward_distortion,
            reversed_distortion,
            places=4,
            msg=f"Swapping arguments changed {forward_distortion} into {reversed_distortion}"
        )


class MelCepstralDistortionAnalysisGridTest(unittest.TestCase):
    # Verifies truncation, flattening, grid selection, and gain invariance.
    def setUp(self) -> None:
        # Prepares the default metric and its reference waveform.
        self._builder: HarmonicWaveformBuilder = HarmonicWaveformBuilder(22050, 4096)
        self._metric: MelCepstralDistortion = MelCepstralDistortion(MelCepstralDistortionConfig())
        self._reference: torch.Tensor = self._builder.clean()

    def test_uniform_gain_leaves_the_distortion_unchanged(self) -> None:
        # Excluding the zeroth coefficient discards the frame-energy term.
        candidate: torch.Tensor = self._builder.degraded(0.05)
        base_distortion: float = self._metric(self._reference, candidate)
        amplified_distortion: float = self._metric(2.0 * self._reference, 2.0 * candidate)
        self.assertAlmostEqual(
            amplified_distortion,
            base_distortion,
            delta=0.01,
            msg=f"Amplifying both signals changed {base_distortion} into {amplified_distortion}"
        )

    def test_a_purely_amplified_candidate_is_near_zero(self) -> None:
        # Gain alone is not an envelope difference, so it must barely register.
        distortion: float = self._metric(self._reference, 2.0 * self._reference)
        self.assertLess(
            distortion,
            0.01,
            msg=f"A gain-only candidate gave {distortion}, expected a near-zero distortion"
        )

    def test_longer_candidate_is_truncated_to_the_common_length(self) -> None:
        # Candidate samples beyond the reference length never enter the grid.
        padded_candidate: torch.Tensor = torch.cat([self._reference, self._builder.noise(1.0)])
        distortion: float = self._metric(self._reference, padded_candidate)
        self.assertEqual(
            distortion,
            0.0,
            msg=f"Padded identical candidate gave {distortion} instead of zero"
        )

    def test_leading_axes_are_flattened_before_analysis(self) -> None:
        # A channel axis is reshaped away, so shape cannot change the value.
        candidate: torch.Tensor = self._builder.degraded(0.05)
        flat_distortion: float = self._metric(self._reference, candidate)
        channelled_distortion: float = self._metric(
            self._reference.unsqueeze(0), candidate.unsqueeze(0)
        )
        self.assertAlmostEqual(
            channelled_distortion,
            flat_distortion,
            places=6,
            msg=f"A channel axis changed {flat_distortion} into {channelled_distortion}"
        )

    def test_a_coarser_mel_grid_still_scores_consistently(self) -> None:
        # Reconfiguring the band count keeps zero at zero and stays positive.
        coarse_metric: MelCepstralDistortion = MelCepstralDistortion(
            MelCepstralDistortionConfig(mel_bands=40)
        )
        identical_distortion: float = coarse_metric(self._reference, self._reference.clone())
        degraded_distortion: float = coarse_metric(self._reference, self._builder.degraded(0.05))
        self.assertEqual(identical_distortion, 0.0, msg=f"Coarse grid gave {identical_distortion}")
        self.assertGreater(degraded_distortion, 0.0, msg=f"Coarse grid gave {degraded_distortion}")

    def test_retaining_more_coefficients_changes_the_reported_value(self) -> None:
        # The retained coefficient count is a real degree of freedom.
        candidate: torch.Tensor = self._builder.degraded(0.05)
        default_distortion: float = self._metric(self._reference, candidate)
        extended_metric: MelCepstralDistortion = MelCepstralDistortion(
            MelCepstralDistortionConfig(coefficient_count=24)
        )
        extended_distortion: float = extended_metric(self._reference, candidate)
        self.assertNotAlmostEqual(
            extended_distortion,
            default_distortion,
            places=3,
            msg=f"Retaining more coefficients left the value at {default_distortion}"
        )

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The metric exposes exactly the record it was constructed with.
        configuration: MelCepstralDistortionConfig = MelCepstralDistortionConfig(mel_bands=40)
        metric: MelCepstralDistortion = MelCepstralDistortion(configuration)
        self.assertIs(metric.configuration, configuration)


if __name__ == "__main__":
    unittest.main()
