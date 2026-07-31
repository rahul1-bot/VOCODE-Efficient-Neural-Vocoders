# This module:
# 1. Verifies the APNet2 multi-period discriminator ensemble: the per-period
#    sub-discriminator count, the logit and feature-map contract consumed by
#    the adversarial and feature-matching objectives, and the reflect-padding
#    branch for lengths that are not period multiples
# 2. Verifies the APNet2 multi-resolution discriminator ensemble: the
#    per-resolution sub-discriminator count, its feature-map contract, and the
#    rejection of a malformed resolution triple
#
# Design decisions:
# - Both ensembles consume the channel-carrying [batch, 1, time] layout the
#   module hands them, and are exercised on a single 4096-sample synthetic
#   waveform, long enough for the widest analysis window of the resolution
#   ensemble
# - Feature-map counts are asserted exactly, because the feature-matching
#   term sums over them and a silently dropped map would weaken the objective
#   without failing any shape check
# - The APNet2 period ensemble retains every convolution stage's feature map,
#   unlike the Vocos ensemble which discards the first; the counts asserted
#   here pin that difference
#
# Author: Rahul Sawhney

import unittest

import torch

from vocode.models.apnet2.discriminator import Apnet2MultiPeriodDiscriminator, Apnet2MultiResolutionDiscriminator


class WaveformPairBuilder:
    # Produces deterministic real and fake waveform pairs in the [batch, channel, time] layout.
    # This is the layout the module normalizes to before handing anything to
    # an ensemble, so the fixtures exercise the ensembles exactly as production
    # calls them.
    def __init__(self, sample_count: int) -> None:
        # Binds the sample count of every waveform this builder emits.
        #
        # Args:
        #     sample_count: Samples per waveform. It must exceed the widest
        #         analysis window of the resolution ensemble, or the largest
        #         transform has too little signal to frame.
        self._sample_count: int = sample_count

    def build_real(self, seed: int) -> torch.Tensor:
        # Seeds the generator and returns the reference waveform.
        #
        # Args:
        #     seed: Global generator seed applied before sampling.
        #
        # Returns:
        #     One channel-carrying waveform shaped ``[1, 1, sample_count]``.
        torch.manual_seed(seed)
        return torch.randn(1, 1, self._sample_count)

    def build_fake(self, seed: int) -> torch.Tensor:
        # Seeds the generator and returns a distinct candidate waveform.
        # The offset seed guarantees the candidate differs from the reference
        # built at the same seed, so an ensemble that ignored one of its two
        # arguments would produce identical outputs and be detectable.
        #
        # Args:
        #     seed: The reference's seed; the candidate is drawn at the seed
        #         immediately after it.
        #
        # Returns:
        #     One channel-carrying waveform shaped ``[1, 1, sample_count]``.
        torch.manual_seed(seed + 1)
        return torch.randn(1, 1, self._sample_count)


class Apnet2MultiPeriodDiscriminatorTest(unittest.TestCase):
    # Verifies the period ensemble's sub-discriminator count and its logit and feature-map contract.
    def setUp(self) -> None:
        # Builds the default period ensemble and the waveform pair builder.
        self._builder: WaveformPairBuilder = WaveformPairBuilder(sample_count=4096)
        torch.manual_seed(20260815)
        self._discriminator: Apnet2MultiPeriodDiscriminator = Apnet2MultiPeriodDiscriminator().eval()

    def test_default_ensemble_judges_through_five_periods(self) -> None:
        # The reference recipe folds the waveform at the five prime periods.
        real_waveform: torch.Tensor = self._builder.build_real(seed=81)
        fake_waveform: torch.Tensor = self._builder.build_fake(seed=81)
        with torch.no_grad():
            real_logits, fake_logits, real_features, fake_features = self._discriminator(
                real_waveform,
                fake_waveform
            )
        self.assertEqual(len(real_logits), 5)
        self.assertEqual(len(fake_logits), 5)
        self.assertEqual(len(real_features), 5)
        self.assertEqual(len(fake_features), 5)

    def test_every_sub_discriminator_returns_six_feature_maps(self) -> None:
        # All five convolution stages plus the post-convolution feed the feature-matching term.
        real_waveform: torch.Tensor = self._builder.build_real(seed=82)
        fake_waveform: torch.Tensor = self._builder.build_fake(seed=82)
        with torch.no_grad():
            _, _, real_features, fake_features = self._discriminator(real_waveform, fake_waveform)
        real_map_counts: list[int] = [len(feature_maps) for feature_maps in real_features]
        fake_map_counts: list[int] = [len(feature_maps) for feature_maps in fake_features]
        self.assertEqual(real_map_counts, [6, 6, 6, 6, 6])
        self.assertEqual(fake_map_counts, [6, 6, 6, 6, 6])

    def test_logits_are_flat_finite_and_shape_aligned(self) -> None:
        # Real and fake logits must be comparable per sub-discriminator for the adversarial term.
        real_waveform: torch.Tensor = self._builder.build_real(seed=83)
        fake_waveform: torch.Tensor = self._builder.build_fake(seed=83)
        with torch.no_grad():
            real_logits, fake_logits, _, _ = self._discriminator(real_waveform, fake_waveform)
        index: int
        real_logit: torch.Tensor
        fake_logit: torch.Tensor
        for index, (real_logit, fake_logit) in enumerate(zip(real_logits, fake_logits, strict=True)):
            self.assertEqual(real_logit.ndim, 2, msg=f"Sub-discriminator {index} logit is not flattened")
            self.assertEqual(real_logit.shape, fake_logit.shape)
            self.assertTrue(bool(torch.isfinite(real_logit).all()))
            self.assertTrue(bool(torch.isfinite(fake_logit).all()))

    def test_length_that_is_not_a_period_multiple_is_reflect_padded(self) -> None:
        # A prime-length waveform exercises the padding branch of every period view.
        # A prime length is divisible by none of the five periods, so every
        # sub-discriminator takes the padding branch in one call; any composite
        # length would leave at least one member on the unpadded path and the
        # test would not cover them all. This matters in production because the
        # synthesis length is frame-derived and has no reason to be a multiple
        # of any period.
        odd_builder: WaveformPairBuilder = WaveformPairBuilder(sample_count=4093)
        real_waveform: torch.Tensor = odd_builder.build_real(seed=84)
        fake_waveform: torch.Tensor = odd_builder.build_fake(seed=84)
        with torch.no_grad():
            real_logits, fake_logits, _, _ = self._discriminator(real_waveform, fake_waveform)
        self.assertEqual(len(real_logits), 5)
        self.assertTrue(all(bool(torch.isfinite(logit).all()) for logit in real_logits))
        self.assertTrue(all(bool(torch.isfinite(logit).all()) for logit in fake_logits))


class Apnet2MultiResolutionDiscriminatorTest(unittest.TestCase):
    # Verifies the resolution ensemble's sub-discriminator count, feature-map contract, and resolution guard.
    def setUp(self) -> None:
        # Builds the default resolution ensemble and the waveform pair builder.
        self._builder: WaveformPairBuilder = WaveformPairBuilder(sample_count=4096)
        torch.manual_seed(20260816)
        self._discriminator: Apnet2MultiResolutionDiscriminator = Apnet2MultiResolutionDiscriminator().eval()

    def test_default_ensemble_judges_through_three_analysis_grids(self) -> None:
        # The reference recipe judges magnitude spectra at three resolutions.
        real_waveform: torch.Tensor = self._builder.build_real(seed=91)
        fake_waveform: torch.Tensor = self._builder.build_fake(seed=91)
        with torch.no_grad():
            real_logits, fake_logits, real_features, fake_features = self._discriminator(
                real_waveform,
                fake_waveform
            )
        self.assertEqual(len(real_logits), 3)
        self.assertEqual(len(fake_logits), 3)
        self.assertEqual(len(real_features), 3)
        self.assertEqual(len(fake_features), 3)

    def test_every_sub_discriminator_returns_six_feature_maps(self) -> None:
        # All five convolution stages plus the post-convolution feed the feature-matching term.
        real_waveform: torch.Tensor = self._builder.build_real(seed=92)
        fake_waveform: torch.Tensor = self._builder.build_fake(seed=92)
        with torch.no_grad():
            _, _, real_features, fake_features = self._discriminator(real_waveform, fake_waveform)
        real_map_counts: list[int] = [len(feature_maps) for feature_maps in real_features]
        fake_map_counts: list[int] = [len(feature_maps) for feature_maps in fake_features]
        self.assertEqual(real_map_counts, [6, 6, 6])
        self.assertEqual(fake_map_counts, [6, 6, 6])

    def test_logits_are_flat_finite_and_shape_aligned(self) -> None:
        # The spectral head flattens its decision map for the adversarial term.
        real_waveform: torch.Tensor = self._builder.build_real(seed=93)
        fake_waveform: torch.Tensor = self._builder.build_fake(seed=93)
        with torch.no_grad():
            real_logits, fake_logits, _, _ = self._discriminator(real_waveform, fake_waveform)
        index: int
        real_logit: torch.Tensor
        fake_logit: torch.Tensor
        for index, (real_logit, fake_logit) in enumerate(zip(real_logits, fake_logits, strict=True)):
            self.assertEqual(real_logit.ndim, 2, msg=f"Resolution logit {index} is not flattened")
            self.assertEqual(real_logit.shape, fake_logit.shape)
            self.assertTrue(bool(torch.isfinite(real_logit).all()))
            self.assertTrue(bool(torch.isfinite(fake_logit).all()))

    def test_resolution_selection_controls_the_ensemble_size(self) -> None:
        # One sub-discriminator is built per declared analysis grid.
        torch.manual_seed(94)
        reduced_discriminator: Apnet2MultiResolutionDiscriminator = Apnet2MultiResolutionDiscriminator(
            resolutions=((512, 128, 512),)
        ).eval()
        real_waveform: torch.Tensor = self._builder.build_real(seed=94)
        fake_waveform: torch.Tensor = self._builder.build_fake(seed=94)
        with torch.no_grad():
            real_logits, _, real_features, _ = reduced_discriminator(real_waveform, fake_waveform)
        self.assertEqual(len(real_logits), 1)
        self.assertEqual(len(real_features), 1)

    def test_resolution_without_three_entries_is_rejected(self) -> None:
        # A grid is an STFT size, hop, and window triple; anything else is a configuration error.
        # The guard lives on the sub-discriminator rather than the ensemble, so
        # APNet2 validates each triple's shape while accepting any number of
        # them; this is where it differs from the BigVGAN ensemble, which
        # additionally fixes the count at three.
        with self.assertRaisesRegex(ValueError, "resolution of length 3"):
            Apnet2MultiResolutionDiscriminator(resolutions=((1024, 256),))
