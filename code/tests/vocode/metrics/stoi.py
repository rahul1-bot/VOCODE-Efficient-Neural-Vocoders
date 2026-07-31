# This module:
# 1. Verifies the StoiConfig validation surface: shipped defaults, frozen
#    semantics, forbidden extras, and the positive-rate and boolean-variant
#    constraints
# 2. Verifies Stoi scoring order: identical waveforms reach the intelligibility
#    ceiling of one while degraded and noise-substituted candidates score
#    strictly lower, declining monotonically as additive noise grows
# 3. Verifies the extended variant, common-length truncation, the native
#    ten-kilohertz path that skips resampling, and rejection of multi-channel
#    buffers
#
# Design decisions:
# - Test material is a deterministic harmonic stack under a slow amplitude
#   envelope; a fixed torch seed precedes every random tensor so scores never
#   drift between runs
# - Half a second at 22050 Hz (11025 samples) is the shortest material the
#   pystoi implementation accepts: below roughly 0.4 seconds it warns about
#   insufficient short-time frames and returns its sentinel value instead of
#   a score, so shorter tensors would test the sentinel rather than the metric
# - Scores are bounded rather than pinned: assertions check the ceiling, the
#   valid range, relative ordering, and monotonicity
# - The classic and extended variants are bounded separately because the
#   extended variant is an uncleaned correlation that may go negative on
#   badly degraded material, while the classic variant stays non-negative
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError

from vocode.metrics.stoi import Stoi, StoiConfig


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


class StoiConfigurationValidationTest(unittest.TestCase):
    # Verifies the frozen STOI configuration record and its constraints.
    def setUp(self) -> None:
        # Builds the default record shared by the validation checks.
        self._configuration: StoiConfig = StoiConfig()

    def test_default_configuration_selects_the_classic_variant_at_ten_kilohertz(self) -> None:
        # The shipped defaults fix the operating rate the measure is defined at.
        self.assertEqual(self._configuration.target_sample_rate, 10000)
        self.assertFalse(self._configuration.extended)

    def test_configuration_rejects_unknown_field(self) -> None:
        # A forbidden extra turns a mistyped setting into an immediate failure.
        with self.assertRaises(ValidationError):
            StoiConfig(sampling_rate=10000)

    def test_configuration_rejects_non_positive_sample_rate(self) -> None:
        # The positive-integer constraint excludes zero and negative rates.
        with self.assertRaises(ValidationError):
            StoiConfig(target_sample_rate=0)
        with self.assertRaises(ValidationError):
            StoiConfig(target_sample_rate=-10000)

    def test_configuration_rejects_non_boolean_variant_flag(self) -> None:
        # Strict validation refuses an integer where the variant flag is declared.
        with self.assertRaises(ValidationError):
            StoiConfig(extended=1)

    def test_configuration_is_frozen(self) -> None:
        # A bound configuration cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.extended = True


class StoiScoringTest(unittest.TestCase):
    # Verifies how STOI ranks identical, degraded, and corrupted candidates.
    def setUp(self) -> None:
        # Prepares the classic metric and its half-second reference signal.
        self._builder: HarmonicWaveformBuilder = HarmonicWaveformBuilder(22050, 11025)
        self._metric: Stoi = Stoi(StoiConfig())
        self._reference: torch.Tensor = self._builder.clean()
        self._source_sample_rate: int = 22050

    def test_identical_waveforms_score_at_the_intelligibility_ceiling(self) -> None:
        # A candidate equal to the reference correlates perfectly in every band.
        score: float = self._metric(self._reference, self._reference.clone(), self._source_sample_rate)
        self.assertAlmostEqual(
            score,
            1.0,
            places=4,
            msg=f"Identical waveforms scored {score}, expected the ceiling of one"
        )

    def test_degraded_candidate_scores_below_identical(self) -> None:
        # Additive noise must cost intelligibility against the same reference.
        identical_score: float = self._metric(
            self._reference, self._reference.clone(), self._source_sample_rate
        )
        degraded_score: float = self._metric(
            self._reference, self._builder.degraded(0.1), self._source_sample_rate
        )
        self.assertLess(
            degraded_score,
            identical_score,
            msg=f"Degraded candidate scored {degraded_score}, not below identical {identical_score}"
        )

    def test_noise_substituted_candidate_scores_below_identical(self) -> None:
        # Replacing the candidate with pure noise destroys the band correlation.
        identical_score: float = self._metric(
            self._reference, self._reference.clone(), self._source_sample_rate
        )
        noise_score: float = self._metric(
            self._reference, self._builder.noise(0.3), self._source_sample_rate
        )
        self.assertLess(
            noise_score,
            identical_score,
            msg=f"Noise candidate scored {noise_score}, not below identical {identical_score}"
        )

    def test_score_declines_as_additive_noise_grows(self) -> None:
        # The measure orders degradation strength across a widening noise sweep.
        light_score: float = self._metric(
            self._reference, self._builder.degraded(0.01), self._source_sample_rate
        )
        moderate_score: float = self._metric(
            self._reference, self._builder.degraded(0.1), self._source_sample_rate
        )
        heavy_score: float = self._metric(
            self._reference, self._builder.degraded(0.5), self._source_sample_rate
        )
        self.assertGreater(
            light_score,
            moderate_score,
            msg=f"Light noise scored {light_score}, not above moderate noise {moderate_score}"
        )
        self.assertGreater(
            moderate_score,
            heavy_score,
            msg=f"Moderate noise scored {moderate_score}, not above heavy noise {heavy_score}"
        )

    def test_scores_remain_inside_the_classic_range(self) -> None:
        # Every produced value is a finite correlation measure in [0.0, 1.0].
        identical_score: float = self._metric(
            self._reference, self._reference.clone(), self._source_sample_rate
        )
        noise_score: float = self._metric(
            self._reference, self._builder.noise(0.3), self._source_sample_rate
        )
        score: float
        for score in (identical_score, noise_score):
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0, msg=f"Score {score} fell below the classic floor")
            self.assertLessEqual(score, 1.0, msg=f"Score {score} exceeded the ceiling of one")


class StoiVariantAndPreparationTest(unittest.TestCase):
    # Verifies the extended variant, truncation, resampling, and input shapes.
    def setUp(self) -> None:
        # Prepares the classic metric and its half-second reference signal.
        self._builder: HarmonicWaveformBuilder = HarmonicWaveformBuilder(22050, 11025)
        self._metric: Stoi = Stoi(StoiConfig())
        self._reference: torch.Tensor = self._builder.clean()
        self._source_sample_rate: int = 22050

    def test_extended_variant_scores_identical_waveforms_at_the_ceiling(self) -> None:
        # The extended variant also saturates when the candidate matches exactly.
        extended_metric: Stoi = Stoi(StoiConfig(extended=True))
        score: float = extended_metric(
            self._reference, self._reference.clone(), self._source_sample_rate
        )
        self.assertAlmostEqual(
            score,
            1.0,
            places=4,
            msg=f"Extended variant scored {score} on identical waveforms"
        )

    def test_extended_variant_ranks_degradation_below_identical(self) -> None:
        # The uncleaned extended correlation may go negative but must still drop.
        extended_metric: Stoi = Stoi(StoiConfig(extended=True))
        identical_score: float = extended_metric(
            self._reference, self._reference.clone(), self._source_sample_rate
        )
        degraded_score: float = extended_metric(
            self._reference, self._builder.degraded(0.1), self._source_sample_rate
        )
        self.assertLess(
            degraded_score,
            identical_score,
            msg=f"Extended degraded scored {degraded_score}, not below identical {identical_score}"
        )
        self.assertGreaterEqual(
            degraded_score,
            -1.0,
            msg=f"Extended score {degraded_score} fell below the correlation floor"
        )

    def test_longer_candidate_is_truncated_to_the_common_length(self) -> None:
        # Candidate samples beyond the reference length never reach the scorer.
        padded_candidate: torch.Tensor = torch.cat([self._reference, self._builder.noise(1.0)])
        padded_score: float = self._metric(
            self._reference, padded_candidate, self._source_sample_rate
        )
        identical_score: float = self._metric(
            self._reference, self._reference.clone(), self._source_sample_rate
        )
        self.assertAlmostEqual(
            padded_score,
            identical_score,
            places=6,
            msg=f"Padded candidate scored {padded_score} instead of the truncated {identical_score}"
        )

    def test_native_operating_rate_skips_resampling(self) -> None:
        # Material already at ten kilohertz scores without a resample step.
        native_builder: HarmonicWaveformBuilder = HarmonicWaveformBuilder(10000, 5000)
        native_reference: torch.Tensor = native_builder.clean()
        score: float = self._metric(native_reference, native_reference.clone(), 10000)
        self.assertAlmostEqual(
            score,
            1.0,
            places=4,
            msg=f"Native-rate identical waveforms scored {score}, expected the ceiling"
        )

    def test_multichannel_input_is_rejected(self) -> None:
        # The pystoi implementation consumes one-dimensional buffers only.
        channelled_reference: torch.Tensor = self._reference.unsqueeze(0)
        with self.assertRaises(ValueError):
            self._metric(channelled_reference, channelled_reference.clone(), self._source_sample_rate)

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The metric exposes exactly the record it was constructed with.
        configuration: StoiConfig = StoiConfig(extended=True)
        metric: Stoi = Stoi(configuration)
        self.assertIs(metric.configuration, configuration)


if __name__ == "__main__":
    unittest.main()
