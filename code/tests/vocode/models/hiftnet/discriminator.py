# This module:
# 1. Verifies the HiFTNet multi-period discriminator: the per-period output
#    structure, the reflect padding of lengths that do not divide the
#    period, and the fixed reference channel widths
# 2. Verifies the multi-resolution spectrogram discriminator: the
#    per-resolution output structure and the time-major spectrogram view the
#    convolution stack consumes
#
# Design decisions:
# - Structural forwards run with reduced period and resolution sets on short
#   waveforms, because the ensembles are shape contracts here
# - The reference period and resolution defaults are exercised for their
#   judgement counts only, one forward each. The default resolution forward
#   is the single expensive assertion in this file: it runs the 2048-point
#   analysis, and its waveform is 4096 samples because that analysis needs
#   the length
# - Assertions bound structure (list lengths, feature-map counts, channel
#   widths) rather than logit values, which are meaningless at random
#   initialization
#
# Author: Rahul Sawhney

import unittest

import torch

from vocode.models.hiftnet.discriminator import (
    HiftnetMultiPeriodDiscriminator,
    HiftnetMultiResolutionSpectrogramDiscriminator,
)


class WaveformBatchBuilder:
    # Builds the discriminator-shaped waveform batches used across the
    # assertions. Every batch carries the explicit channel axis the ensembles
    # require, so a layout error in a fixture cannot be mistaken for a defect
    # in the component under test.
    def __init__(self, batch_size: int, sample_count: int) -> None:
        # Binds the batch and waveform geometry this builder emits.
        #
        # Args:
        #     batch_size: Number of rows in the emitted batch.
        #     sample_count: Waveform length in samples. This value is
        #         load-bearing for the spectral assertions, whose expected
        #         frame counts follow from it and the analysis hop.
        self._batch_size: int = batch_size
        self._sample_count: int = sample_count

    def build(self) -> torch.Tensor:
        # Returns one channel-carrying waveform batch drawn from the global
        # generator, so the seed set by a test's setUp governs its content.
        return torch.randn(self._batch_size, 1, self._sample_count)


class HiftnetMultiPeriodDiscriminatorTest(unittest.TestCase):
    # Verifies the per-period judgement structure and the period folding of
    # the time-domain ensemble. Logit values are meaningless at random
    # initialization, so the assertions establish structure instead: that one
    # judgement is produced per period, that every depth of each member is
    # exposed for feature matching, that the channel widths match the
    # reference, and that a length not divisible by the period is handled
    # without changing the output contract.
    def setUp(self) -> None:
        # Builds the reduced two-period ensemble and the waveform it judges.
        # Two periods suffice for every structural claim here, since members
        # are independent and the reference count is checked separately.
        torch.manual_seed(1234)
        self._discriminator: HiftnetMultiPeriodDiscriminator = HiftnetMultiPeriodDiscriminator(periods=(2, 3))
        self._waveform: torch.Tensor = WaveformBatchBuilder(batch_size=1, sample_count=512).build()

    def test_one_judgement_is_returned_per_configured_period(self) -> None:
        # The ensemble emits real logits, fake logits, and both feature-map stacks per period.
        with torch.no_grad():
            real_logits, fake_logits, real_features, fake_features = self._discriminator(
                self._waveform,
                self._waveform
            )
        self.assertEqual(len(real_logits), 2)
        self.assertEqual(len(fake_logits), 2)
        self.assertEqual(len(real_features), 2)
        self.assertEqual(len(fake_features), 2)

    def test_each_sub_discriminator_reports_six_feature_maps(self) -> None:
        # Five strided convolutions plus the output convolution feed the
        # feature-matching loss. The count is asserted because the objective
        # zips the reference and synthesis stacks strictly: a member that
        # silently stopped collecting one depth would weaken the term without
        # any error surfacing.
        with torch.no_grad():
            _, _, real_features, _ = self._discriminator(self._waveform, self._waveform)
        feature_stack: list[torch.Tensor]
        for feature_stack in real_features:
            self.assertEqual(len(feature_stack), 6)

    def test_reference_channel_widths_are_fixed(self) -> None:
        # The HiFTNet period discriminator carries the unscaled reference
        # widths, which are hard-coded in the implementation rather than
        # derived from a configuration field. Because nothing else constrains
        # them, this assertion is the only guard against a silent capacity
        # change to the critics, which would alter training dynamics without
        # touching any configuration record.
        with torch.no_grad():
            _, _, real_features, _ = self._discriminator(self._waveform, self._waveform)
        expected_widths: tuple[int, ...] = (32, 128, 512, 1024, 1024, 1)
        stage_index: int
        expected_width: int
        for stage_index, expected_width in enumerate(expected_widths):
            self.assertEqual(
                real_features[0][stage_index].shape[1],
                expected_width,
                msg=f"Unexpected channel width at period-discriminator stage {stage_index}"
            )

    def test_logits_are_flattened_per_batch_element(self) -> None:
        # Logits arrive as one flat row per batch element for the adversarial objective.
        with torch.no_grad():
            real_logits, _, _, _ = self._discriminator(self._waveform, self._waveform)
        self.assertEqual(real_logits[0].ndim, 2)
        self.assertEqual(tuple(real_logits[0].shape), (1, 8))

    def test_length_that_does_not_divide_the_period_is_reflect_padded(self) -> None:
        # An odd length is padded up to the next multiple of the period and
        # judged identically. This matters in practice rather than only in
        # principle: the generator's inverse transform produces a sample count
        # determined by frames and hop, which carries no guarantee of being
        # divisible by any of the ensemble's periods, so unpadded folding
        # would fail on ordinary synthesis lengths. Comparing logit shapes
        # against the divisible case is the check that padding restores the
        # contract rather than merely avoiding a crash.
        odd_waveform: torch.Tensor = torch.randn(1, 1, 511)
        with torch.no_grad():
            odd_logits, _, _, _ = self._discriminator(odd_waveform, odd_waveform)
            even_logits, _, _, _ = self._discriminator(self._waveform, self._waveform)
        self.assertEqual(tuple(odd_logits[0].shape), tuple(even_logits[0].shape))

    def test_batch_dimension_is_preserved(self) -> None:
        # Two-element batches produce two logit rows per period.
        batched: torch.Tensor = WaveformBatchBuilder(batch_size=2, sample_count=512).build()
        with torch.no_grad():
            real_logits, _, _, _ = self._discriminator(batched, batched)
        self.assertEqual(real_logits[0].shape[0], 2)

    def test_default_periods_are_the_reference_prime_set(self) -> None:
        # The reference recipe folds the waveform at five prime periods. Only
        # the member count is asserted, because that is what the objective
        # depends on structurally; the specific prime values are a property of
        # the constructor default and are visible at the definition site.
        default_discriminator: HiftnetMultiPeriodDiscriminator = HiftnetMultiPeriodDiscriminator()
        with torch.no_grad():
            real_logits, _, _, _ = default_discriminator(self._waveform, self._waveform)
        self.assertEqual(len(real_logits), 5)


class HiftnetMultiResolutionSpectrogramDiscriminatorTest(unittest.TestCase):
    # Verifies the per-resolution judgement structure and the time-major
    # spectrogram view. The axis order is the substantive claim: the
    # convolution stack strides only along its last axis, so frames must
    # precede frequency bins for the members to remain time-resolved, and a
    # transposition error would silently turn a time-resolved critic into a
    # frequency-resolved one while every shape still looked plausible.
    def setUp(self) -> None:
        # Builds the reduced three-resolution ensemble and the waveform it
        # judges. The transform sizes are scaled down by a common factor of
        # sixteen from the reference triple, preserving their relative
        # ordering while keeping the analyses cheap on a five-hundred-twelve
        # sample waveform.
        torch.manual_seed(1234)
        self._discriminator: HiftnetMultiResolutionSpectrogramDiscriminator = (
            HiftnetMultiResolutionSpectrogramDiscriminator(
                fft_sizes=(64, 128, 32),
                hop_sizes=(32, 64, 16),
                win_lengths=(64, 128, 32)
            )
        )
        self._waveform: torch.Tensor = WaveformBatchBuilder(batch_size=1, sample_count=512).build()

    def test_one_judgement_is_returned_per_resolution(self) -> None:
        # Each configured analysis resolution contributes one logit row and one feature stack.
        with torch.no_grad():
            real_logits, fake_logits, real_features, fake_features = self._discriminator(
                self._waveform,
                self._waveform
            )
        self.assertEqual(len(real_logits), 3)
        self.assertEqual(len(fake_logits), 3)
        self.assertEqual(len(real_features), 3)
        self.assertEqual(len(fake_features), 3)

    def test_each_resolution_reports_six_feature_maps(self) -> None:
        # Five convolutions plus the output convolution feed the feature-matching loss.
        with torch.no_grad():
            _, _, real_features, _ = self._discriminator(self._waveform, self._waveform)
        feature_stack: list[torch.Tensor]
        for feature_stack in real_features:
            self.assertEqual(len(feature_stack), 6)

    def test_spectrogram_view_is_time_major(self) -> None:
        # The magnitude view is transposed so frames index rows and bins index
        # columns. The three expected extents pin all of it at once: the
        # channel width the first convolution produces, the frame count the
        # sixty-four-point analysis at hop thirty-two yields from five hundred
        # twelve samples, and the bin count of that transform. Because the
        # frame and bin counts differ, this assertion would fail under a
        # transposed view rather than passing on coincidentally equal axes.
        with torch.no_grad():
            _, _, real_features, _ = self._discriminator(self._waveform, self._waveform)
        first_map: torch.Tensor = real_features[0][0]
        self.assertEqual(first_map.shape[1], 32)
        self.assertEqual(first_map.shape[2], 17)
        self.assertEqual(first_map.shape[3], 33)

    def test_logits_are_flattened_per_batch_element(self) -> None:
        # Logits arrive as one flat row per batch element for every resolution.
        with torch.no_grad():
            real_logits, _, _, _ = self._discriminator(self._waveform, self._waveform)
        logit_row: torch.Tensor
        for logit_row in real_logits:
            self.assertEqual(logit_row.ndim, 2)
            self.assertEqual(logit_row.shape[0], 1)

    def test_outputs_are_finite(self) -> None:
        # The spectrogram analysis and convolution stack produce no non-finite logits.
        with torch.no_grad():
            real_logits, _, _, _ = self._discriminator(self._waveform, self._waveform)
        self.assertTrue(bool(torch.isfinite(real_logits[0]).all()))

    def test_default_resolutions_are_the_reference_triple(self) -> None:
        # The reference recipe analyzes at 1024, 2048, and 512 points. The
        # waveform is lengthened to four thousand ninety-six specifically for
        # this assertion, because the widest of those analyses cannot run on
        # the short fixture the other tests share; this is the one deliberately
        # expensive forward pass in the file.
        default_discriminator: HiftnetMultiResolutionSpectrogramDiscriminator = (
            HiftnetMultiResolutionSpectrogramDiscriminator()
        )
        long_waveform: torch.Tensor = WaveformBatchBuilder(batch_size=1, sample_count=4096).build()
        with torch.no_grad():
            real_logits, _, _, _ = default_discriminator(long_waveform, long_waveform)
        self.assertEqual(len(real_logits), 3)
