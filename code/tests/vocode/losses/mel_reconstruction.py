# This module:
# 1. Verifies the mel reconstruction loss: the L1 distance between reference
#    and synthesized log-mel spectrograms, its optimum at identity, and its
#    gradient path into the synthesized spectrogram
# 2. Verifies the shape-agreement guard, which must reject any mismatch in the
#    mel-bin or frame dimension rather than broadcasting or cropping silently
#
# Design decisions:
# - Constant offsets make the expected L1 distance exactly derivable, so the
#   reduction is asserted against arithmetic rather than a recorded number
# - The guard is exercised on each dimension separately (bins, frames, batch)
#   because a mismatch in any one of them signals a different extraction bug
# - Spectrograms are minimal because the loss reduces over every axis and has no
#   shape-dependent behavior beyond the equality guard
#
# Author: Rahul Sawhney

import unittest

import torch

from vocode.losses.mel_reconstruction import MelReconstructionLoss


class MelSpectrogramBuilder:
    # Builds the minimal log-mel spectrograms the assertions compare.
    def __init__(self, batch_size: int, mel_bin_count: int, frame_count: int) -> None:
        # Binds the constructor inputs into this component's state.
        self._batch_size: int = batch_size
        self._mel_bin_count: int = mel_bin_count
        self._frame_count: int = frame_count

    def constant(self, value: float) -> torch.Tensor:
        # Builds a spectrogram whose every entry holds the given value.
        return torch.full((self._batch_size, self._mel_bin_count, self._frame_count), value)

    def seeded(self, seed: int) -> torch.Tensor:
        # Builds a seeded random spectrogram of the configured shape.
        torch.manual_seed(seed)
        return torch.randn(self._batch_size, self._mel_bin_count, self._frame_count)

    def seeded_requiring_gradient(self, seed: int) -> torch.Tensor:
        # Builds a seeded spectrogram that participates in autograd.
        return self.seeded(seed).requires_grad_(True)


class MelReconstructionDistanceTest(unittest.TestCase):
    # Verifies the L1 reduction, its optimum, and its ordering behavior.
    def setUp(self) -> None:
        # The three axis lengths are deliberately all different, so a
        # reduction that transposed or collapsed the wrong axis would change
        # the result rather than coincidentally agreeing on a square shape.
        self._loss: MelReconstructionLoss = MelReconstructionLoss()
        self._builder: MelSpectrogramBuilder = MelSpectrogramBuilder(
            batch_size=2,
            mel_bin_count=8,
            frame_count=12
        )

    def test_loss_is_zero_for_identical_spectrograms(self) -> None:
        # Perfect reconstruction is the analytic optimum of the term.
        reference_mel: torch.Tensor = self._builder.seeded(seed=13)
        value: torch.Tensor = self._loss(reference_mel, reference_mel)
        self.assertAlmostEqual(
            float(value.item()),
            0.0,
            places=6,
            msg="Identical spectrograms must carry no reconstruction cost"
        )

    def test_loss_equals_the_constant_offset_between_spectrograms(self) -> None:
        # A uniform offset reduces to its own absolute value.
        reference_mel: torch.Tensor = self._builder.constant(-4.0)
        candidate_mel: torch.Tensor = self._builder.constant(-3.25)
        value: torch.Tensor = self._loss(reference_mel, candidate_mel)
        self.assertAlmostEqual(float(value.item()), 0.75, places=5)

    def test_loss_is_symmetric_in_its_two_arguments(self) -> None:
        # The absolute distance does not depend on argument order.
        reference_mel: torch.Tensor = self._builder.seeded(seed=3)
        candidate_mel: torch.Tensor = self._builder.seeded(seed=4)
        forward: torch.Tensor = self._loss(reference_mel, candidate_mel)
        reversed_order: torch.Tensor = self._loss(candidate_mel, reference_mel)
        self.assertAlmostEqual(float(forward.item()), float(reversed_order.item()), places=6)

    def test_loss_grows_with_the_size_of_the_offset(self) -> None:
        # A larger uniform deviation is strictly more expensive.
        reference_mel: torch.Tensor = self._builder.constant(0.0)
        near: torch.Tensor = self._loss(reference_mel, self._builder.constant(0.5))
        far: torch.Tensor = self._loss(reference_mel, self._builder.constant(2.0))
        self.assertGreater(float(far.item()), float(near.item()))

    def test_loss_returns_a_finite_non_negative_scalar(self) -> None:
        # The reduction collapses the spectrogram pair to a zero-dimensional tensor.
        reference_mel: torch.Tensor = self._builder.seeded(seed=21)
        candidate_mel: torch.Tensor = self._builder.seeded(seed=22)
        value: torch.Tensor = self._loss(reference_mel, candidate_mel)
        self.assertEqual(value.shape, torch.Size([]), msg="The objective must reduce to a scalar")
        self.assertEqual(value.dtype, torch.float32)
        self.assertTrue(torch.isfinite(value).item())
        self.assertGreater(float(value.item()), 0.0)

    def test_loss_propagates_gradient_to_the_candidate_spectrogram(self) -> None:
        # The synthesized side carries the reconstruction signal.
        reference_mel: torch.Tensor = self._builder.seeded(seed=31)
        candidate_mel: torch.Tensor = self._builder.seeded_requiring_gradient(seed=32)
        value: torch.Tensor = self._loss(reference_mel, candidate_mel)
        value.backward()
        self.assertIsNotNone(candidate_mel.grad, msg="The candidate spectrogram must receive gradient")
        self.assertEqual(candidate_mel.grad.shape, candidate_mel.shape)
        self.assertTrue(torch.isfinite(candidate_mel.grad).all().item())


class MelReconstructionShapeGuardTest(unittest.TestCase):
    # Verifies that a shape disagreement fails closed instead of broadcasting.
    def setUp(self) -> None:
        # Fixes one reference spectrogram that every case pairs with a
        # deliberately mismatched candidate. Holding the reference constant
        # means each case varies exactly one axis, so a failure names the
        # axis whose guard broke.
        self._loss: MelReconstructionLoss = MelReconstructionLoss()
        self._reference_mel: torch.Tensor = torch.zeros(2, 8, 12)

    def test_frame_count_mismatch_is_rejected(self) -> None:
        # A differing frame count signals a hop or padding mismatch.
        candidate_mel: torch.Tensor = torch.zeros(2, 8, 11)
        with self.assertRaisesRegex(ValueError, "identical shape"):
            self._loss(self._reference_mel, candidate_mel)

    def test_mel_bin_count_mismatch_is_rejected(self) -> None:
        # A differing bin count signals an extraction-protocol mismatch.
        candidate_mel: torch.Tensor = torch.zeros(2, 7, 12)
        with self.assertRaisesRegex(ValueError, "identical shape"):
            self._loss(self._reference_mel, candidate_mel)

    def test_batch_size_mismatch_is_rejected(self) -> None:
        # A broadcastable batch dimension must still be refused.
        candidate_mel: torch.Tensor = torch.zeros(1, 8, 12)
        with self.assertRaisesRegex(ValueError, "identical shape"):
            self._loss(self._reference_mel, candidate_mel)

    def test_rejection_message_reports_both_offending_shapes(self) -> None:
        # The error must name the reference and candidate shapes for debugging.
        candidate_mel: torch.Tensor = torch.zeros(2, 8, 11)
        with self.assertRaises(ValueError) as failure:
            self._loss(self._reference_mel, candidate_mel)
        message: str = str(failure.exception)
        self.assertIn("(2, 8, 12)", message)
        self.assertIn("(2, 8, 11)", message)
