# This module:
# 1. Verifies the LogAmplitudeSpectrumRmseConfig validation surface: shipped
#    defaults, frozen semantics, forbidden extras, and the positive-grid and
#    positive-floor constraints
# 2. Verifies scoring behavior: identical waveforms give exactly zero, differing
#    waveforms give a strictly positive decibel error, the error grows with
#    additive noise, and the measure is symmetric in its arguments
# 3. Verifies the amplitude floor: silence compares to silence at zero, a signal
#    against silence stays finite instead of diverging, and raising the floor
#    lowers that bounded error
#
# Design decisions:
# - Test material is a deterministic harmonic stack under a slow amplitude
#   envelope at 4096 samples, which fills seventeen analysis frames on the
#   default grid while staying inexpensive; a fixed torch seed precedes every
#   random tensor so errors never drift between runs
# - The identical-input case is asserted as exact zero because the computation
#   is deterministic pure math; every other value is bounded or ordered rather
#   than pinned
# - The floor is exercised through a fully silent signal, which is the case
#   the flooring exists for: without it the logarithm would produce negative
#   infinity and the root mean square would become undefined
#
# Author: Rahul Sawhney

import math
import unittest

import torch
from pydantic import ValidationError

from vocode.metrics.las import LogAmplitudeSpectrumRmse, LogAmplitudeSpectrumRmseConfig


class HarmonicWaveformBuilder:
    # Builds deterministic speech-shaped waveforms, silence, and noisy variants.
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

    def silence(self) -> torch.Tensor:
        # Produces an all-zero waveform on the bound length.
        return torch.zeros(self._sample_count)

    def noise(self, noise_scale: float) -> torch.Tensor:
        # Draws a reproducible gaussian sequence at the requested amplitude.
        torch.manual_seed(self._noise_seed)
        return noise_scale * torch.randn(self._sample_count)

    def degraded(self, noise_scale: float) -> torch.Tensor:
        # Adds the reproducible noise sequence onto the clean waveform.
        return self.clean() + self.noise(noise_scale)


class LogAmplitudeSpectrumRmseConfigurationValidationTest(unittest.TestCase):
    # Verifies the frozen spectral-grid record and its constraints.
    def setUp(self) -> None:
        # Builds the default record shared by the validation checks.
        self._configuration: LogAmplitudeSpectrumRmseConfig = LogAmplitudeSpectrumRmseConfig()

    def test_default_configuration_matches_the_study_spectral_grid(self) -> None:
        # The shipped defaults are the grid the reported values were produced on.
        self.assertEqual(self._configuration.n_fft, 1024)
        self.assertEqual(self._configuration.hop_length, 256)
        self.assertAlmostEqual(self._configuration.amplitude_floor, 1.0e-5, places=9)

    def test_configuration_rejects_unknown_field(self) -> None:
        # A forbidden extra turns a mistyped setting into an immediate failure.
        with self.assertRaises(ValidationError):
            LogAmplitudeSpectrumRmseConfig(window_length=1024)

    def test_configuration_rejects_non_positive_grid_extents(self) -> None:
        # Both grid extents are positive integers, so zero is refused outright.
        with self.assertRaises(ValidationError):
            LogAmplitudeSpectrumRmseConfig(n_fft=0)
        with self.assertRaises(ValidationError):
            LogAmplitudeSpectrumRmseConfig(hop_length=0)

    def test_configuration_rejects_non_positive_amplitude_floor(self) -> None:
        # A non-positive floor would let the logarithm diverge.
        with self.assertRaises(ValidationError):
            LogAmplitudeSpectrumRmseConfig(amplitude_floor=0.0)
        with self.assertRaises(ValidationError):
            LogAmplitudeSpectrumRmseConfig(amplitude_floor=-1.0e-5)

    def test_configuration_rejects_non_integer_transform_size(self) -> None:
        # Strict validation refuses a float where an integer extent is declared.
        with self.assertRaises(ValidationError):
            LogAmplitudeSpectrumRmseConfig(n_fft=1024.0)

    def test_configuration_is_frozen(self) -> None:
        # A bound configuration cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.n_fft = 512


class LogAmplitudeSpectrumRmseScoringTest(unittest.TestCase):
    # Verifies the decibel-domain spectral error over agreeing waveforms.
    def setUp(self) -> None:
        # Prepares the default metric and its reference waveform.
        self._builder: HarmonicWaveformBuilder = HarmonicWaveformBuilder(22050, 4096)
        self._metric: LogAmplitudeSpectrumRmse = LogAmplitudeSpectrumRmse(
            LogAmplitudeSpectrumRmseConfig()
        )
        self._reference: torch.Tensor = self._builder.clean()

    def test_identical_waveforms_give_exactly_zero(self) -> None:
        # Matching decibel spectra leave no squared error to accumulate.
        error: float = self._metric(self._reference, self._reference.clone())
        self.assertEqual(error, 0.0, msg=f"Identical waveforms gave {error} instead of zero error")

    def test_differing_waveforms_give_a_positive_error(self) -> None:
        # Any spectral disagreement registers as a positive decibel error.
        error: float = self._metric(self._reference, self._builder.degraded(0.05))
        self.assertGreater(error, 0.0, msg=f"Differing waveforms gave {error}")

    def test_error_grows_as_additive_noise_grows(self) -> None:
        # The error orders degradation strength across a widening noise sweep.
        light_error: float = self._metric(self._reference, self._builder.degraded(0.01))
        moderate_error: float = self._metric(self._reference, self._builder.degraded(0.1))
        heavy_error: float = self._metric(self._reference, self._builder.degraded(0.5))
        self.assertLess(
            light_error,
            moderate_error,
            msg=f"Light noise gave {light_error}, not below moderate {moderate_error}"
        )
        self.assertLess(
            moderate_error,
            heavy_error,
            msg=f"Moderate noise gave {moderate_error}, not below heavy {heavy_error}"
        )

    def test_error_is_never_negative(self) -> None:
        # A root mean of squared differences cannot fall below zero.
        error: float = self._metric(self._reference, self._builder.noise(0.3))
        self.assertIsInstance(error, float)
        self.assertGreaterEqual(error, 0.0, msg=f"Error {error} went negative")

    def test_error_is_symmetric_in_its_arguments(self) -> None:
        # Squared differences do not depend on which side is the reference.
        candidate: torch.Tensor = self._builder.degraded(0.05)
        forward_error: float = self._metric(self._reference, candidate)
        reversed_error: float = self._metric(candidate, self._reference)
        self.assertAlmostEqual(
            forward_error,
            reversed_error,
            places=4,
            msg=f"Swapping the arguments changed {forward_error} into {reversed_error}"
        )


class LogAmplitudeSpectrumRmseFlooringTest(unittest.TestCase):
    # Verifies that the amplitude floor keeps silent bins finite and bounded.
    def setUp(self) -> None:
        # Prepares the default metric, a reference waveform, and silence.
        self._builder: HarmonicWaveformBuilder = HarmonicWaveformBuilder(22050, 4096)
        self._metric: LogAmplitudeSpectrumRmse = LogAmplitudeSpectrumRmse(
            LogAmplitudeSpectrumRmseConfig()
        )
        self._reference: torch.Tensor = self._builder.clean()
        self._silence: torch.Tensor = self._builder.silence()

    def test_silence_compared_to_silence_gives_exactly_zero(self) -> None:
        # Two floored spectra agree everywhere, so the error collapses.
        error: float = self._metric(self._silence, self._silence.clone())
        self.assertEqual(error, 0.0, msg=f"Silence against silence gave {error}")

    def test_signal_compared_to_silence_stays_finite(self) -> None:
        # Without the floor the silent spectrum would drive the error to infinity.
        error: float = self._metric(self._reference, self._silence)
        self.assertTrue(math.isfinite(error), msg=f"Signal against silence gave {error}")
        self.assertGreater(error, 0.0, msg=f"Signal against silence gave {error}")

    def test_raising_the_floor_lowers_the_silent_comparison_error(self) -> None:
        # The floor sets how far a silent bin can sit below an occupied one.
        low_floor_metric: LogAmplitudeSpectrumRmse = LogAmplitudeSpectrumRmse(
            LogAmplitudeSpectrumRmseConfig(amplitude_floor=1.0e-5)
        )
        high_floor_metric: LogAmplitudeSpectrumRmse = LogAmplitudeSpectrumRmse(
            LogAmplitudeSpectrumRmseConfig(amplitude_floor=1.0e-2)
        )
        low_floor_error: float = low_floor_metric(self._reference, self._silence)
        high_floor_error: float = high_floor_metric(self._reference, self._silence)
        self.assertLess(
            high_floor_error,
            low_floor_error,
            msg=f"Raising the floor gave {high_floor_error}, not below {low_floor_error}"
        )


class LogAmplitudeSpectrumRmseGridTest(unittest.TestCase):
    # Verifies truncation, flattening, and grid reconfiguration.
    def setUp(self) -> None:
        # Prepares the default metric and its reference waveform.
        self._builder: HarmonicWaveformBuilder = HarmonicWaveformBuilder(22050, 4096)
        self._metric: LogAmplitudeSpectrumRmse = LogAmplitudeSpectrumRmse(
            LogAmplitudeSpectrumRmseConfig()
        )
        self._reference: torch.Tensor = self._builder.clean()

    def test_longer_candidate_is_truncated_to_the_common_length(self) -> None:
        # Candidate samples beyond the reference length never enter the transform.
        padded_candidate: torch.Tensor = torch.cat([self._reference, self._builder.noise(1.0)])
        error: float = self._metric(self._reference, padded_candidate)
        self.assertEqual(
            error,
            0.0,
            msg=f"Padded identical candidate gave {error} instead of zero"
        )

    def test_leading_axes_are_flattened_before_analysis(self) -> None:
        # A channel axis is reshaped away, so shape cannot change the value.
        candidate: torch.Tensor = self._builder.degraded(0.05)
        flat_error: float = self._metric(self._reference, candidate)
        channelled_error: float = self._metric(self._reference.unsqueeze(0), candidate.unsqueeze(0))
        self.assertAlmostEqual(
            channelled_error,
            flat_error,
            places=6,
            msg=f"A channel axis changed {flat_error} into {channelled_error}"
        )

    def test_a_finer_transform_grid_still_scores_consistently(self) -> None:
        # Reconfiguring the transform keeps zero at zero and stays positive.
        fine_metric: LogAmplitudeSpectrumRmse = LogAmplitudeSpectrumRmse(
            LogAmplitudeSpectrumRmseConfig(n_fft=256, hop_length=64)
        )
        identical_error: float = fine_metric(self._reference, self._reference.clone())
        degraded_error: float = fine_metric(self._reference, self._builder.degraded(0.05))
        self.assertEqual(identical_error, 0.0, msg=f"Fine grid gave {identical_error}")
        self.assertGreater(degraded_error, 0.0, msg=f"Fine grid gave {degraded_error}")

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The metric exposes exactly the record it was constructed with.
        configuration: LogAmplitudeSpectrumRmseConfig = LogAmplitudeSpectrumRmseConfig(n_fft=256)
        metric: LogAmplitudeSpectrumRmse = LogAmplitudeSpectrumRmse(configuration)
        self.assertIs(metric.configuration, configuration)


if __name__ == "__main__":
    unittest.main()
