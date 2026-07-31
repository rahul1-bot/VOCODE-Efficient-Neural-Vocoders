# This module:
# 1. Verifies the RNDVoC multi-period discriminator: the per-period output
#    structure, the reflect padding of lengths that do not divide the
#    period, the channel-carrying and flat waveform layouts it accepts, and
#    the reference channel widths
# 2. Verifies the multi-resolution discriminator: the per-resolution output
#    structure and the magnitude-spectrogram view its convolution stack
#    consumes
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

from vocode.models.rndvoc.discriminator import RndvocMultiPeriodDiscriminator, RndvocMultiResolutionDiscriminator


class WaveformBatchBuilder:
    # Builds the waveform batches the discriminator ensembles judge, in both
    # layouts. Two builders exist because this family's generator emits a
    # channel-free waveform while the critics are written for the
    # channel-carrying convention, so each ensemble adapts on entry and both
    # paths need covering.
    def __init__(self, batch_size: int, sample_count: int) -> None:
        # Binds the batch and waveform geometry both layouts share.
        #
        # Args:
        #     batch_size: Number of rows in the emitted batch.
        #     sample_count: Waveform length in samples, which the spectral
        #         assertions' expected frame counts follow from.
        self._batch_size: int = batch_size
        self._sample_count: int = sample_count

    def build_channel_first(self) -> torch.Tensor:
        # Returns the channel-carrying layout the period ensemble folds
        # directly, without any promotion.
        return torch.randn(self._batch_size, 1, self._sample_count)

    def build_flat(self) -> torch.Tensor:
        # Returns the channel-free layout, which is what this family's
        # generator actually produces and therefore the layout a real training
        # step hands the critics.
        return torch.randn(self._batch_size, self._sample_count)


class RndvocMultiPeriodDiscriminatorTest(unittest.TestCase):
    # Verifies the per-period judgement structure, the accepted layouts, and
    # the period folding. Logit values are meaningless at random
    # initialization, so the assertions establish structure instead: one
    # judgement per period, every depth exposed for feature matching, the
    # reference channel widths, and correct handling of a length that does not
    # divide the period.
    def setUp(self) -> None:
        # Builds the reduced two-period ensemble and the waveform it judges.
        # Two periods suffice for every structural claim, since members are
        # independent and the reference count is checked separately.
        torch.manual_seed(1234)
        self._discriminator: RndvocMultiPeriodDiscriminator = RndvocMultiPeriodDiscriminator(periods=(2, 3))
        self._builder: WaveformBatchBuilder = WaveformBatchBuilder(batch_size=1, sample_count=512)
        self._waveform: torch.Tensor = self._builder.build_channel_first()

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
        # Five strided convolutions plus the output convolution feed the feature-matching loss.
        with torch.no_grad():
            _, _, real_features, _ = self._discriminator(self._waveform, self._waveform)
        feature_stack: list[torch.Tensor]
        for feature_stack in real_features:
            self.assertEqual(len(feature_stack), 6)

    def test_reference_channel_widths_are_fixed(self) -> None:
        # The period discriminator carries the unscaled reference widths,
        # which are hard-coded in the implementation rather than derived from
        # any configuration field. Because nothing else constrains them, this
        # is the only guard against a silent capacity change to the critics,
        # which would alter training dynamics without touching a single
        # configuration record.
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

    def test_flat_waveform_layout_is_accepted(self) -> None:
        # A batch without the channel axis is promoted before folding, which
        # matters in practice rather than only in principle: this family's
        # generator emits exactly that layout, so a critic that required the
        # channel axis would fail on every real training step. Comparing logit
        # shapes against the channel-carrying case is what proves the
        # promotion is equivalent rather than merely non-fatal.
        flat_waveform: torch.Tensor = self._builder.build_flat()
        with torch.no_grad():
            flat_logits, _, _, _ = self._discriminator(flat_waveform, flat_waveform)
            channel_logits, _, _, _ = self._discriminator(self._waveform, self._waveform)
        self.assertEqual(tuple(flat_logits[0].shape), tuple(channel_logits[0].shape))

    def test_length_that_does_not_divide_the_period_is_reflect_padded(self) -> None:
        # An odd length is padded up to the next multiple of the period and
        # judged identically. This is not a corner case: the inverse transform
        # produces a sample count determined by frames and hop, which carries
        # no guarantee of divisibility by any of the ensemble's periods, so
        # unpadded folding would fail on ordinary synthesis lengths.
        odd_waveform: torch.Tensor = torch.randn(1, 1, 511)
        with torch.no_grad():
            odd_logits, _, _, _ = self._discriminator(odd_waveform, odd_waveform)
            even_logits, _, _, _ = self._discriminator(self._waveform, self._waveform)
        self.assertEqual(tuple(odd_logits[0].shape), tuple(even_logits[0].shape))

    def test_batch_dimension_is_preserved(self) -> None:
        # Two-element batches produce two logit rows per period.
        batched: torch.Tensor = WaveformBatchBuilder(batch_size=2, sample_count=512).build_channel_first()
        with torch.no_grad():
            real_logits, _, _, _ = self._discriminator(batched, batched)
        self.assertEqual(real_logits[0].shape[0], 2)

    def test_default_periods_are_the_reference_prime_set(self) -> None:
        # The reference recipe folds the waveform at five prime periods.
        default_discriminator: RndvocMultiPeriodDiscriminator = RndvocMultiPeriodDiscriminator()
        with torch.no_grad():
            real_logits, _, _, _ = default_discriminator(self._waveform, self._waveform)
        self.assertEqual(len(real_logits), 5)


class RndvocMultiResolutionDiscriminatorTest(unittest.TestCase):
    # Verifies the per-resolution judgement structure and the
    # magnitude-spectrogram view. Unlike the HiFTNet spectral ensemble, these
    # members stride along both axes, so the assertions bound the channel
    # width and the member and feature counts rather than any per-frame
    # extent, which the striding deliberately does not preserve.
    def setUp(self) -> None:
        # Builds the reduced three-resolution ensemble and the waveform it
        # judges. The transform sizes are scaled down by a common factor of
        # sixteen from the reference triple, preserving their relative ordering
        # while keeping the analyses affordable on a short waveform. The flat
        # layout is used here because it is what the generator emits.
        torch.manual_seed(1234)
        self._discriminator: RndvocMultiResolutionDiscriminator = RndvocMultiResolutionDiscriminator(
            resolutions=((64, 16, 64), (128, 32, 128), (32, 8, 32))
        )
        self._builder: WaveformBatchBuilder = WaveformBatchBuilder(batch_size=1, sample_count=512)
        self._waveform: torch.Tensor = self._builder.build_flat()

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

    def test_spectrogram_convolutions_use_the_reference_width(self) -> None:
        # The resolution discriminator runs at sixty-four channels throughout.
        with torch.no_grad():
            _, _, real_features, _ = self._discriminator(self._waveform, self._waveform)
        self.assertEqual(real_features[0][0].shape[1], 64)

    def test_channel_carrying_waveform_layout_is_accepted(self) -> None:
        # A batch with the channel axis is squeezed before the analysis, the
        # mirror image of the period ensemble's promotion. Both ensembles
        # therefore accept either layout, which is what allows the training
        # step to pass one pair of tensors to both without adapting between
        # them.
        channel_waveform: torch.Tensor = self._builder.build_channel_first()
        with torch.no_grad():
            channel_logits, _, _, _ = self._discriminator(channel_waveform, channel_waveform)
            flat_logits, _, _, _ = self._discriminator(self._waveform, self._waveform)
        self.assertEqual(tuple(channel_logits[0].shape), tuple(flat_logits[0].shape))

    def test_logits_are_flattened_per_batch_element(self) -> None:
        # Logits arrive as one flat row per batch element for every resolution.
        with torch.no_grad():
            real_logits, _, _, _ = self._discriminator(self._waveform, self._waveform)
        logit_row: torch.Tensor
        for logit_row in real_logits:
            self.assertEqual(logit_row.ndim, 2)
            self.assertEqual(logit_row.shape[0], 1)

    def test_outputs_are_finite(self) -> None:
        # The magnitude analysis and convolution stack produce no non-finite logits.
        with torch.no_grad():
            real_logits, _, _, _ = self._discriminator(self._waveform, self._waveform)
        self.assertTrue(bool(torch.isfinite(real_logits[0]).all()))

    def test_default_resolutions_are_the_reference_triple(self) -> None:
        # The reference recipe analyzes at 1024, 2048, and 512 points. The
        # waveform is lengthened to four thousand ninety-six specifically for
        # this assertion, because the widest of those analyses cannot run on
        # the short fixture the other tests share; this is the one deliberately
        # expensive forward pass in the file.
        default_discriminator: RndvocMultiResolutionDiscriminator = RndvocMultiResolutionDiscriminator()
        long_waveform: torch.Tensor = WaveformBatchBuilder(batch_size=1, sample_count=4096).build_flat()
        with torch.no_grad():
            real_logits, _, _, _ = default_discriminator(long_waveform, long_waveform)
        self.assertEqual(len(real_logits), 3)
