# This module:
# 1. Verifies the MultiResolutionStftErrorConfig validation surface: the
#    published three-resolution defaults, frozen semantics, forbidden extras,
#    and the strict tuple arity and element typing of the analysis triples
# 2. Verifies scoring behavior: identical waveforms give exactly zero, differing
#    waveforms give a strictly positive spectral distance, and the distance
#    grows monotonically with additive noise
# 3. Verifies the evaluation path: common-length truncation, reshaping into the
#    criterion's batch-channel-time layout, gradient-free evaluation, and
#    reconfiguration onto a different resolution grid
#
# Design decisions:
# - Test material is a deterministic harmonic stack under a slow amplitude
#   envelope at 4096 samples, which exceeds the largest default transform size
#   of 2048 while staying inexpensive; a fixed torch seed precedes every random
#   tensor so distances never drift between runs
# - The identical-input case is asserted as exact zero because both the spectral
#   convergence and log-magnitude terms vanish there; every other value is
#   bounded or ordered rather than pinned
# - Argument symmetry is deliberately not asserted: the spectral convergence
#   term normalizes by the reference magnitude, so the criterion is genuinely
#   asymmetric and a symmetry assertion would encode a false expectation
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError

from vocode.metrics.stft import MultiResolutionStftError, MultiResolutionStftErrorConfig


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


class MultiResolutionStftErrorConfigurationValidationTest(unittest.TestCase):
    # Verifies the frozen resolution-grid record and its constraints.
    def setUp(self) -> None:
        # Builds the default record shared by the validation checks.
        self._configuration: MultiResolutionStftErrorConfig = MultiResolutionStftErrorConfig()

    def test_default_configuration_matches_the_published_resolution_grid(self) -> None:
        # The shipped defaults are the grid reported values are comparable with.
        self.assertEqual(self._configuration.fft_sizes, (1024, 2048, 512))
        self.assertEqual(self._configuration.hop_sizes, (120, 240, 50))
        self.assertEqual(self._configuration.win_lengths, (600, 1200, 240))

    def test_configuration_rejects_unknown_field(self) -> None:
        # A forbidden extra turns a mistyped setting into an immediate failure.
        with self.assertRaises(ValidationError):
            MultiResolutionStftErrorConfig(window_sizes=(600, 1200, 240))

    def test_configuration_rejects_a_grid_of_the_wrong_arity(self) -> None:
        # Each triple carries exactly three parallel resolutions.
        with self.assertRaises(ValidationError):
            MultiResolutionStftErrorConfig(fft_sizes=(1024, 2048))
        with self.assertRaises(ValidationError):
            MultiResolutionStftErrorConfig(hop_sizes=(120, 240, 50, 25))

    def test_configuration_rejects_a_list_where_a_tuple_is_declared(self) -> None:
        # Strict validation keeps the immutable tuple contract intact.
        with self.assertRaises(ValidationError):
            MultiResolutionStftErrorConfig(fft_sizes=[1024, 2048, 512])

    def test_configuration_rejects_non_integer_resolutions(self) -> None:
        # Strict validation refuses a float where integer extents are declared.
        with self.assertRaises(ValidationError):
            MultiResolutionStftErrorConfig(fft_sizes=(1024.0, 2048, 512))

    def test_configuration_is_frozen(self) -> None:
        # A bound configuration cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.fft_sizes = (256, 512, 128)


class MultiResolutionStftErrorScoringTest(unittest.TestCase):
    # Verifies the multi-resolution spectral distance over agreeing waveforms.
    def setUp(self) -> None:
        # Prepares the default metric and its reference waveform.
        self._builder: HarmonicWaveformBuilder = HarmonicWaveformBuilder(22050, 4096)
        self._metric: MultiResolutionStftError = MultiResolutionStftError(
            MultiResolutionStftErrorConfig()
        )
        self._reference: torch.Tensor = self._builder.clean()

    def test_identical_waveforms_give_exactly_zero(self) -> None:
        # Both the convergence and log-magnitude terms vanish on a perfect match.
        error: float = self._metric(self._reference, self._reference.clone())
        self.assertEqual(error, 0.0, msg=f"Identical waveforms gave {error} instead of zero error")

    def test_differing_waveforms_give_a_positive_error(self) -> None:
        # Any spectral disagreement registers across the resolution grid.
        error: float = self._metric(self._reference, self._builder.degraded(0.05))
        self.assertGreater(error, 0.0, msg=f"Differing waveforms gave {error}")

    def test_error_grows_as_additive_noise_grows(self) -> None:
        # The distance orders degradation strength across a widening noise sweep.
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
        # The summed spectral terms cannot fall below zero.
        error: float = self._metric(self._reference, self._builder.noise(0.3))
        self.assertIsInstance(error, float)
        self.assertGreaterEqual(error, 0.0, msg=f"Error {error} went negative")


class MultiResolutionStftErrorEvaluationTest(unittest.TestCase):
    # Verifies truncation, layout reshaping, gradient isolation, and grid choice.
    def setUp(self) -> None:
        # Prepares the default metric and its reference waveform.
        self._builder: HarmonicWaveformBuilder = HarmonicWaveformBuilder(22050, 4096)
        self._metric: MultiResolutionStftError = MultiResolutionStftError(
            MultiResolutionStftErrorConfig()
        )
        self._reference: torch.Tensor = self._builder.clean()

    def test_longer_candidate_is_truncated_to_the_common_length(self) -> None:
        # Candidate samples beyond the reference length never enter the criterion.
        padded_candidate: torch.Tensor = torch.cat([self._reference, self._builder.noise(1.0)])
        error: float = self._metric(self._reference, padded_candidate)
        self.assertEqual(
            error,
            0.0,
            msg=f"Padded identical candidate gave {error} instead of zero"
        )

    def test_leading_axes_are_reshaped_into_the_criterion_layout(self) -> None:
        # A channel axis is folded into the batch-channel-time layout unchanged.
        candidate: torch.Tensor = self._builder.degraded(0.05)
        flat_error: float = self._metric(self._reference, candidate)
        channelled_error: float = self._metric(self._reference.unsqueeze(0), candidate.unsqueeze(0))
        self.assertAlmostEqual(
            channelled_error,
            flat_error,
            places=6,
            msg=f"A channel axis changed {flat_error} into {channelled_error}"
        )

    def test_evaluation_leaves_no_gradient_on_a_tracked_input(self) -> None:
        # The criterion runs under no-grad and returns a detached scalar.
        tracked_reference: torch.Tensor = self._reference.clone().requires_grad_(True)
        error: float = self._metric(tracked_reference, self._builder.degraded(0.05))
        self.assertIsInstance(error, float)
        self.assertIsNone(
            tracked_reference.grad,
            msg="Evaluation attached a gradient to a tracked input waveform"
        )

    def test_a_compact_resolution_grid_still_scores_consistently(self) -> None:
        # Reconfiguring the grid keeps zero at zero and stays positive.
        compact_metric: MultiResolutionStftError = MultiResolutionStftError(
            MultiResolutionStftErrorConfig(
                fft_sizes=(256, 512, 128),
                hop_sizes=(32, 64, 16),
                win_lengths=(128, 256, 64)
            )
        )
        identical_error: float = compact_metric(self._reference, self._reference.clone())
        degraded_error: float = compact_metric(self._reference, self._builder.degraded(0.05))
        self.assertEqual(identical_error, 0.0, msg=f"Compact grid gave {identical_error}")
        self.assertGreater(degraded_error, 0.0, msg=f"Compact grid gave {degraded_error}")

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The metric exposes exactly the record it was constructed with.
        configuration: MultiResolutionStftErrorConfig = MultiResolutionStftErrorConfig(
            fft_sizes=(256, 512, 128),
            hop_sizes=(32, 64, 16),
            win_lengths=(128, 256, 64)
        )
        metric: MultiResolutionStftError = MultiResolutionStftError(configuration)
        self.assertIs(metric.configuration, configuration)


if __name__ == "__main__":
    unittest.main()
