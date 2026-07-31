# This module:
# 1. Verifies the BigVGAN multi-period discriminator: the per-period output
#    structure, the reflect padding of lengths that do not divide the
#    period, and the channel scaling driven by the channel multiplier
# 2. Verifies the multi-resolution discriminator: its three-resolution
#    contract, its spectrogram-view output structure, and the equivalence
#    of the spectral-norm and weight-norm variants in output geometry
#
# Design decisions:
# - Both ensembles are exercised with reduced periods, reduced resolutions,
#   and a fractional channel multiplier so the file stays inside the runtime
#   budget; the published resolution triple is asserted only for its
#   structural contract, not by running the 2048-point analysis
# - Assertions bound structure (list lengths, feature-map counts, batch and
#   channel dimensions) rather than logit values, which are meaningless at
#   random initialization
#
# Author: Rahul Sawhney

import unittest

import torch

from vocode.models.bigvgan.discriminator import BigvganMultiPeriodDiscriminator, BigvganMultiResolutionDiscriminator


class WaveformBatchBuilder:
    # Builds the discriminator-shaped waveform batches used across the assertions.
    # The [batch, 1, time] layout is what the module normalizes to before
    # handing anything to an ensemble, so the fixtures exercise the ensembles
    # exactly as production calls them.
    def __init__(self, batch_size: int, sample_count: int) -> None:
        # Binds the batch and waveform geometry this builder emits.
        #
        # Args:
        #     batch_size: Rows per batch; the ensembles must return one logit
        #         row per element, which is what the batch assertions check.
        #     sample_count: Samples per waveform. It must exceed the widest
        #         analysis window of the reduced resolution ensemble, or the
        #         largest transform has too little signal to frame.
        self._batch_size: int = batch_size
        self._sample_count: int = sample_count

    def build(self) -> torch.Tensor:
        # Returns one channel-carrying waveform batch from the global generator.
        # No seed is set here: the fixture draws from the generator each test
        # class seeds once in setUp. That is sufficient because every assertion
        # in this file bounds structure rather than values, and logits at random
        # initialization carry no meaning worth pinning.
        #
        # Returns:
        #     A batch shaped ``[batch_size, 1, sample_count]``.
        return torch.randn(self._batch_size, 1, self._sample_count)


class BigvganMultiPeriodDiscriminatorTest(unittest.TestCase):
    # Verifies the per-period judgement structure and the period folding of the period ensemble.
    def setUp(self) -> None:
        # Builds the reduced two-period ensemble and the waveform it judges.
        torch.manual_seed(1234)
        self._discriminator: BigvganMultiPeriodDiscriminator = BigvganMultiPeriodDiscriminator(
            periods=(2, 3),
            channel_multiplier=0.25,
            use_spectral_norm=False
        )
        self._builder: WaveformBatchBuilder = WaveformBatchBuilder(batch_size=1, sample_count=512)
        self._waveform: torch.Tensor = self._builder.build()

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

    def test_logits_are_flattened_per_batch_element(self) -> None:
        # Logits arrive as one flat row per batch element for the adversarial objective.
        with torch.no_grad():
            real_logits, _, _, _ = self._discriminator(self._waveform, self._waveform)
        self.assertEqual(real_logits[0].ndim, 2)
        self.assertEqual(tuple(real_logits[0].shape), (1, 8))

    def test_channel_multiplier_scales_the_first_feature_width(self) -> None:
        # A multiplier of one quarter reduces the first convolution to eight channels.
        with torch.no_grad():
            _, _, real_features, _ = self._discriminator(self._waveform, self._waveform)
        self.assertEqual(real_features[0][0].shape[1], 8)

    def test_length_that_does_not_divide_the_period_is_reflect_padded(self) -> None:
        # An odd length is padded up to the next multiple of the period and judged identically.
        # Comparing the odd-length logit shape against the even-length one is
        # what shows the padding restored the folded grid: an unpadded fold
        # would either fail outright or yield a shorter decision. The case is
        # not hypothetical, since the synthesis length is frame-derived and has
        # no reason to be a multiple of any period.
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

    def test_spectral_normalization_preserves_the_output_geometry(self) -> None:
        # Choosing spectral normalization changes the parametrization, never the logit shape.
        # The two options constrain the weights differently but wrap the same
        # convolutions, so the adversarial and feature-matching terms consume
        # identically shaped outputs either way and the setting can be varied
        # per experiment without touching the loss composition.
        spectral_discriminator: BigvganMultiPeriodDiscriminator = BigvganMultiPeriodDiscriminator(
            periods=(2, 3),
            channel_multiplier=0.25,
            use_spectral_norm=True
        )
        with torch.no_grad():
            spectral_logits, _, _, _ = spectral_discriminator(self._waveform, self._waveform)
            weight_logits, _, _, _ = self._discriminator(self._waveform, self._waveform)
        self.assertEqual(tuple(spectral_logits[0].shape), tuple(weight_logits[0].shape))


class BigvganMultiResolutionDiscriminatorTest(unittest.TestCase):
    # Verifies the three-resolution contract and the spectrogram-view outputs of the resolution ensemble.
    def setUp(self) -> None:
        # Builds the reduced three-resolution ensemble and the waveform it judges.
        torch.manual_seed(1234)
        self._resolutions: tuple[tuple[int, int, int], ...] = ((32, 8, 32), (64, 16, 64), (16, 4, 16))
        self._discriminator: BigvganMultiResolutionDiscriminator = BigvganMultiResolutionDiscriminator(
            resolutions=self._resolutions,
            channel_multiplier=0.25,
            use_spectral_norm=False
        )
        self._waveform: torch.Tensor = torch.randn(1, 1, 512)

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

    def test_channel_multiplier_scales_the_spectrogram_convolution_width(self) -> None:
        # A multiplier of one quarter reduces the shared convolution width to eight channels.
        with torch.no_grad():
            _, _, real_features, _ = self._discriminator(self._waveform, self._waveform)
        self.assertEqual(real_features[0][0].shape[1], 8)

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

    def test_resolution_count_other_than_three_is_rejected(self) -> None:
        # The published recipe fixes three analysis resolutions, so any other count fails loudly.
        with self.assertRaisesRegex(ValueError, "expects three resolutions"):
            BigvganMultiResolutionDiscriminator(
                resolutions=((32, 8, 32), (64, 16, 64)),
                channel_multiplier=0.25,
                use_spectral_norm=False
            )

    def test_published_resolution_triple_is_accepted(self) -> None:
        # The published 1024, 2048, and 512 point analyses satisfy the three-resolution contract.
        published: BigvganMultiResolutionDiscriminator = BigvganMultiResolutionDiscriminator(
            resolutions=((1024, 120, 600), (2048, 240, 1200), (512, 50, 240)),
            channel_multiplier=0.25,
            use_spectral_norm=False
        )
        self.assertEqual(len(published.get_submodule("_sub_discriminators")), 3)
