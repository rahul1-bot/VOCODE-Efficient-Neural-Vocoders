# This module:
# 1. Verifies MelError reduction behavior: identical spectrograms give exactly
#    zero, a uniform offset reproduces itself exactly, differing spectrograms
#    give a positive distance, and the measure is symmetric in its arguments
# 2. Verifies that band-count and frame-count disagreement raise rather than
#    silently cropping either axis, with the mismatched extents reported
# 3. Verifies that leading batch axes are reduced along with everything else
#
# Design decisions:
# - The reduction is exact arithmetic (a mean absolute difference), so a
#   uniform offset of a known size is asserted exactly rather than bounded;
#   this pins the reduction itself, not a floating-point accident
# - Spectrograms are small fixed-seed tensors on the metric's own protocol
#   shape (eighty bands, thirty-two frames), because the component consumes
#   already-extracted spectrograms and never touches waveforms or files
# - The shape guards are checked through their message text, since the whole
#   point of raising is to name which axis disagreed and by how much
#
# Author: Rahul Sawhney

import unittest

import torch

from vocode.metrics.mel import MelError


class MelSpectrogramBuilder:
    # Builds deterministic log-mel spectrograms on a fixed protocol shape.
    def __init__(self, mel_bands: int, frame_count: int) -> None:
        # Binds the spectrogram extent and the fixed seed backing every draw.
        self._mel_bands: int = mel_bands
        self._frame_count: int = frame_count
        self._draw_seed: int = 20260730

    def spectrogram(self) -> torch.Tensor:
        # Draws a reproducible log-domain spectrogram on the bound shape.
        torch.manual_seed(self._draw_seed)
        return torch.randn(self._mel_bands, self._frame_count)

    def batched_spectrogram(self, batch_size: int) -> torch.Tensor:
        # Draws the same material carrying a leading batch axis.
        torch.manual_seed(self._draw_seed)
        return torch.randn(batch_size, self._mel_bands, self._frame_count)

    def resized(self, mel_bands: int, frame_count: int) -> torch.Tensor:
        # Draws a spectrogram on a deliberately disagreeing extent.
        torch.manual_seed(self._draw_seed)
        return torch.randn(mel_bands, frame_count)


class MelErrorScoringTest(unittest.TestCase):
    # Verifies the mean absolute log-mel reduction over agreeing spectrograms.
    def setUp(self) -> None:
        # Prepares the metric and one reference spectrogram.
        self._builder: MelSpectrogramBuilder = MelSpectrogramBuilder(80, 32)
        self._metric: MelError = MelError()
        self._reference: torch.Tensor = self._builder.spectrogram()

    def test_identical_spectrograms_give_exactly_zero(self) -> None:
        # An unchanged candidate has no absolute difference to accumulate.
        error: float = self._metric(self._reference, self._reference.clone())
        self.assertEqual(error, 0.0, msg=f"Identical spectrograms gave {error} instead of zero")

    def test_uniform_offset_is_reproduced_exactly(self) -> None:
        # A constant shift of every bin reduces to that same constant.
        offset: float = 2.5
        error: float = self._metric(self._reference, self._reference + offset)
        self.assertAlmostEqual(
            error,
            offset,
            places=5,
            msg=f"A uniform offset of {offset} reduced to {error}"
        )

    def test_differing_spectrograms_give_a_positive_distance(self) -> None:
        # Independent material must register a strictly positive distance.
        candidate: torch.Tensor = self._reference.flip(-1)
        error: float = self._metric(self._reference, candidate)
        self.assertGreater(error, 0.0, msg=f"Differing spectrograms gave {error}")

    def test_larger_offsets_give_larger_distances(self) -> None:
        # The reduction grows with the magnitude of the discrepancy.
        small_error: float = self._metric(self._reference, self._reference + 0.5)
        large_error: float = self._metric(self._reference, self._reference + 3.0)
        self.assertGreater(
            large_error,
            small_error,
            msg=f"A larger offset gave {large_error}, not above the smaller {small_error}"
        )

    def test_distance_is_symmetric_in_its_arguments(self) -> None:
        # Absolute differences do not depend on which side is the reference.
        candidate: torch.Tensor = self._reference.flip(-1)
        forward_error: float = self._metric(self._reference, candidate)
        reversed_error: float = self._metric(candidate, self._reference)
        self.assertAlmostEqual(
            forward_error,
            reversed_error,
            places=6,
            msg=f"Swapping the arguments changed {forward_error} into {reversed_error}"
        )

    def test_reduction_returns_a_plain_float(self) -> None:
        # The metric hands back a scalar, never a tensor holding graph state.
        error: float = self._metric(self._reference, self._reference + 1.0)
        self.assertIsInstance(error, float)

    def test_batched_spectrograms_reduce_over_every_axis(self) -> None:
        # A leading batch axis participates in the same global mean.
        batched_reference: torch.Tensor = self._builder.batched_spectrogram(2)
        identical_error: float = self._metric(batched_reference, batched_reference.clone())
        offset_error: float = self._metric(batched_reference, batched_reference + 1.5)
        self.assertEqual(identical_error, 0.0, msg=f"Batched identical input gave {identical_error}")
        self.assertAlmostEqual(
            offset_error,
            1.5,
            places=5,
            msg=f"Batched uniform offset reduced to {offset_error}"
        )


class MelErrorShapeValidationTest(unittest.TestCase):
    # Verifies that disagreeing spectrogram extents raise instead of cropping.
    def setUp(self) -> None:
        # Prepares the metric and one reference spectrogram.
        self._builder: MelSpectrogramBuilder = MelSpectrogramBuilder(80, 32)
        self._metric: MelError = MelError()
        self._reference: torch.Tensor = self._builder.spectrogram()

    def test_disagreeing_band_counts_are_rejected(self) -> None:
        # A band-count mismatch signals a protocol mismatch, never a crop.
        candidate: torch.Tensor = self._builder.resized(64, 32)
        with self.assertRaisesRegex(ValueError, "Mel-band counts must match"):
            self._metric(self._reference, candidate)

    def test_disagreeing_band_counts_report_both_extents(self) -> None:
        # The message names the reference and candidate band counts.
        candidate: torch.Tensor = self._builder.resized(64, 32)
        with self.assertRaisesRegex(ValueError, "reference=80, candidate=64"):
            self._metric(self._reference, candidate)

    def test_disagreeing_frame_counts_are_rejected(self) -> None:
        # A frame-count mismatch signals unaligned material, never a crop.
        candidate: torch.Tensor = self._builder.resized(80, 16)
        with self.assertRaisesRegex(ValueError, "Mel-frame counts must match"):
            self._metric(self._reference, candidate)

    def test_disagreeing_frame_counts_report_both_extents(self) -> None:
        # The message names the reference and candidate frame counts.
        candidate: torch.Tensor = self._builder.resized(80, 16)
        with self.assertRaisesRegex(ValueError, "reference=32, candidate=16"):
            self._metric(self._reference, candidate)

    def test_band_disagreement_is_reported_before_frame_disagreement(self) -> None:
        # When both axes disagree the band guard fires first.
        candidate: torch.Tensor = self._builder.resized(64, 16)
        with self.assertRaisesRegex(ValueError, "Mel-band counts must match"):
            self._metric(self._reference, candidate)


if __name__ == "__main__":
    unittest.main()
