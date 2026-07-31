# This module:
# 1. Verifies the PesqConfig validation surface: shipped defaults, frozen
#    semantics, forbidden extras, positive-rate and literal-mode constraints
# 2. Verifies Pesq scoring order: identical waveforms reach the wideband
#    ceiling while degraded and noise-substituted candidates score strictly
#    lower, with a monotone decline as additive noise grows
# 3. Verifies signal preparation: common-length truncation, the native
#    operating-rate path that skips resampling, the narrowband variant, gain
#    invariance, and rejection of multi-channel buffers
#
# Design decisions:
# - Test material is a deterministic harmonic stack under a slow amplitude
#   envelope, which the ITU implementation accepts as speech-like; a fixed
#   torch seed precedes every random tensor so scores never drift between runs
# - Half a second at 22050 Hz (11025 samples) is used instead of the minimal
#   tensors preferred elsewhere, because the ITU implementation needs enough
#   frames to detect utterances; one call still costs about ten milliseconds
# - Scores are bounded rather than pinned: assertions check the wideband
#   range, the ceiling, relative ordering, and monotonicity
# - The monotonicity check uses small noise scales because PESQ saturates
#   near its floor once degradation is severe, which would make a wider
#   sweep non-monotone for reasons unrelated to the code under test
# - An all-silent candidate is excluded: the reference implementation raises
#   from its own level normalization there, which is library behavior rather
#   than a contract of this component
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError

from vocode.metrics.pesq import Pesq, PesqConfig


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


class PesqConfigurationValidationTest(unittest.TestCase):
    # Verifies the frozen PESQ configuration record and its constraints.
    def setUp(self) -> None:
        # Builds the default record shared by the validation checks.
        self._configuration: PesqConfig = PesqConfig()

    def test_default_configuration_selects_wideband_at_sixteen_kilohertz(self) -> None:
        # The shipped defaults are the wideband mode at the ITU operating rate.
        self.assertEqual(self._configuration.target_sample_rate, 16000)
        self.assertEqual(self._configuration.mode, "wb")

    def test_configuration_rejects_unknown_field(self) -> None:
        # A forbidden extra turns a mistyped setting into an immediate failure.
        with self.assertRaises(ValidationError):
            PesqConfig(sampling_rate=16000)

    def test_configuration_rejects_non_positive_sample_rate(self) -> None:
        # The positive-integer constraint excludes zero and negative rates.
        with self.assertRaises(ValidationError):
            PesqConfig(target_sample_rate=0)
        with self.assertRaises(ValidationError):
            PesqConfig(target_sample_rate=-16000)

    def test_configuration_rejects_non_integer_sample_rate(self) -> None:
        # Strict validation refuses a float where an integer rate is declared.
        with self.assertRaises(ValidationError):
            PesqConfig(target_sample_rate=16000.0)

    def test_configuration_rejects_unknown_mode(self) -> None:
        # The mode literal admits only the wideband and narrowband names.
        with self.assertRaises(ValidationError):
            PesqConfig(mode="fullband")

    def test_configuration_is_frozen(self) -> None:
        # A bound configuration cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.mode = "nb"


class PesqScoringTest(unittest.TestCase):
    # Verifies how PESQ ranks identical, mildly degraded, and corrupted candidates.
    def setUp(self) -> None:
        # Prepares the wideband metric and its half-second reference signal.
        self._builder: HarmonicWaveformBuilder = HarmonicWaveformBuilder(22050, 11025)
        self._metric: Pesq = Pesq(PesqConfig())
        self._reference: torch.Tensor = self._builder.clean()
        self._source_sample_rate: int = 22050

    def test_identical_waveforms_score_near_the_wideband_ceiling(self) -> None:
        # A candidate equal to the reference sits at the top of the scale.
        score: float = self._metric(self._reference, self._reference.clone(), self._source_sample_rate)
        self.assertGreaterEqual(
            score,
            4.5,
            msg=f"Identical waveforms scored {score}, expected the wideband ceiling near 4.64"
        )

    def test_degraded_candidate_scores_below_identical(self) -> None:
        # Additive noise must cost perceptual quality against the same reference.
        identical_score: float = self._metric(
            self._reference, self._reference.clone(), self._source_sample_rate
        )
        degraded_score: float = self._metric(
            self._reference, self._builder.degraded(0.01), self._source_sample_rate
        )
        self.assertLess(
            degraded_score,
            identical_score,
            msg=f"Degraded candidate scored {degraded_score}, not below identical {identical_score}"
        )

    def test_noise_substituted_candidate_scores_below_identical(self) -> None:
        # Replacing the candidate with pure noise destroys the perceptual match.
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
        # Within the graded region the measure orders degradation strength.
        light_score: float = self._metric(
            self._reference, self._builder.degraded(0.0001), self._source_sample_rate
        )
        moderate_score: float = self._metric(
            self._reference, self._builder.degraded(0.0005), self._source_sample_rate
        )
        heavy_score: float = self._metric(
            self._reference, self._builder.degraded(0.001), self._source_sample_rate
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

    def test_scores_remain_inside_the_wideband_range(self) -> None:
        # Every produced value is a finite mean-opinion score in [1.0, 4.65].
        identical_score: float = self._metric(
            self._reference, self._reference.clone(), self._source_sample_rate
        )
        noise_score: float = self._metric(
            self._reference, self._builder.noise(0.3), self._source_sample_rate
        )
        score: float
        for score in (identical_score, noise_score):
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 1.0, msg=f"Score {score} fell below the wideband floor")
            self.assertLessEqual(score, 4.65, msg=f"Score {score} exceeded the wideband ceiling")


class PesqSignalPreparationTest(unittest.TestCase):
    # Verifies truncation, resampling, mode selection, and input-shape handling.
    def setUp(self) -> None:
        # Prepares the wideband metric and its half-second reference signal.
        self._builder: HarmonicWaveformBuilder = HarmonicWaveformBuilder(22050, 11025)
        self._metric: Pesq = Pesq(PesqConfig())
        self._reference: torch.Tensor = self._builder.clean()
        self._source_sample_rate: int = 22050

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
            places=4,
            msg=f"Padded candidate scored {padded_score} instead of the truncated {identical_score}"
        )

    def test_native_operating_rate_skips_resampling(self) -> None:
        # Material already at the operating rate scores without a resample step.
        native_builder: HarmonicWaveformBuilder = HarmonicWaveformBuilder(16000, 8000)
        native_reference: torch.Tensor = native_builder.clean()
        score: float = self._metric(native_reference, native_reference.clone(), 16000)
        self.assertGreaterEqual(
            score,
            4.5,
            msg=f"Native-rate identical waveforms scored {score}, expected the wideband ceiling"
        )

    def test_narrowband_mode_scores_identical_waveforms_near_its_ceiling(self) -> None:
        # The narrowband variant runs at eight kilohertz with its own ceiling.
        narrowband_metric: Pesq = Pesq(PesqConfig(target_sample_rate=8000, mode="nb"))
        score: float = narrowband_metric(
            self._reference, self._reference.clone(), self._source_sample_rate
        )
        self.assertGreaterEqual(
            score,
            4.4,
            msg=f"Narrowband identical waveforms scored {score}, expected the ceiling near 4.55"
        )
        self.assertLessEqual(score, 4.6, msg=f"Narrowband score {score} exceeded its ceiling")

    def test_score_is_invariant_to_candidate_gain(self) -> None:
        # The standard normalizes active speech level before comparing.
        identical_score: float = self._metric(
            self._reference, self._reference.clone(), self._source_sample_rate
        )
        attenuated_score: float = self._metric(
            self._reference, 0.5 * self._reference, self._source_sample_rate
        )
        self.assertAlmostEqual(
            attenuated_score,
            identical_score,
            places=3,
            msg=f"Attenuated candidate scored {attenuated_score} against identical {identical_score}"
        )

    def test_multichannel_input_is_rejected(self) -> None:
        # The reference implementation consumes one-dimensional buffers only.
        channelled_reference: torch.Tensor = self._reference.unsqueeze(0)
        with self.assertRaises(ValueError):
            self._metric(channelled_reference, channelled_reference.clone(), self._source_sample_rate)

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The metric exposes exactly the record it was constructed with.
        configuration: PesqConfig = PesqConfig(target_sample_rate=8000, mode="nb")
        metric: Pesq = Pesq(configuration)
        self.assertIs(metric.configuration, configuration)


if __name__ == "__main__":
    unittest.main()
