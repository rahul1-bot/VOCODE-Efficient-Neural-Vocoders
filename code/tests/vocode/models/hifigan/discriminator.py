# This module:
# 1. Verifies the period sub-discriminator: the flattened logit layout,
#    the intermediate feature maps the feature-matching loss consumes,
#    and the reflect padding that folds a waveform whose length is not a
#    multiple of the period
# 2. Verifies the scale sub-discriminator under both normalizations, and
#    the two ensembles: five prime periods for the multi-period
#    discriminator and three progressively pooled scales for the
#    multi-scale discriminator
#
# Design decisions:
# - Logit widths are asserted as layout properties (batch dimension
#   preserved, two-dimensional, non-empty) rather than pinned integers,
#   because the exact width is a function of period folding and stride
#   arithmetic that carries no reference value
# - The multi-scale ensemble is checked by strictly decreasing logit width
#   across scales, which is the observable signature of the average
#   pooling between scales
# - Waveforms are one thousand twenty-four samples, the smallest length
#   that survives the full strided convolution stack of every
#   sub-discriminator
#
# Author: Rahul Sawhney

import unittest

import torch

from vocode.models.hifigan.discriminator import (
    DiscriminatorP,
    DiscriminatorS,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)


class DiscriminatorPTest(unittest.TestCase):
    # Verifies the period sub-discriminator folds the waveform at its
    # period and returns flattened logits alongside its feature maps.
    def setUp(self) -> None:
        # Builds one seeded period-three sub-discriminator and a waveform long
        # enough to survive its strided convolution stack.
        torch.manual_seed(0)
        self._discriminator: DiscriminatorP = DiscriminatorP(period=3)
        self._waveform: torch.Tensor = torch.randn(1, 1, 1024)

    def test_forward_returns_flattened_logits_and_feature_maps(self) -> None:
        # The adversarial term reads the logits, feature matching the maps.
        logits: torch.Tensor
        feature_maps: list[torch.Tensor]
        logits, feature_maps = self._discriminator(self._waveform)
        self.assertEqual(logits.ndim, 2)
        self.assertEqual(logits.shape[0], 1)
        self.assertGreater(logits.shape[1], 0)
        self.assertIsInstance(feature_maps, list)

    def test_forward_returns_one_feature_map_per_convolution(self) -> None:
        # Five strided convolutions plus the post-convolution give six maps.
        feature_maps: list[torch.Tensor]
        _, feature_maps = self._discriminator(self._waveform)
        self.assertEqual(len(feature_maps), 6)

    def test_forward_returns_finite_logits(self) -> None:
        # A non-finite logit would break the least-squares objective.
        logits: torch.Tensor
        logits, _ = self._discriminator(self._waveform)
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_forward_accepts_a_length_that_is_not_a_period_multiple(self) -> None:
        # Reflect padding completes the final period before folding.
        logits: torch.Tensor
        logits, _ = self._discriminator(torch.randn(1, 1, 1000))
        self.assertEqual(logits.ndim, 2)
        self.assertEqual(logits.shape[0], 1)

    def test_padding_leaves_the_logit_width_unchanged_within_one_period(self) -> None:
        # A length one sample short folds to the same number of rows.
        padded_logits: torch.Tensor
        exact_logits: torch.Tensor
        padded_logits, _ = self._discriminator(torch.randn(1, 1, 1023))
        exact_logits, _ = self._discriminator(torch.randn(1, 1, 1024))
        self.assertEqual(padded_logits.shape, exact_logits.shape)

    def test_forward_preserves_the_batch_dimension(self) -> None:
        # Each batch element receives its own logit row.
        logits: torch.Tensor
        logits, _ = self._discriminator(torch.randn(2, 1, 1024))
        self.assertEqual(logits.shape[0], 2)

    def test_every_reference_period_produces_logits(self) -> None:
        # The reference ensemble folds at five prime periods.
        period: int
        for period in (2, 3, 5, 7, 11):
            discriminator: DiscriminatorP = DiscriminatorP(period=period)
            logits: torch.Tensor
            logits, _ = discriminator(self._waveform)
            self.assertGreater(logits.shape[1], 0, msg=f"period={period} produced no logits")


class DiscriminatorSTest(unittest.TestCase):
    # Verifies the scale sub-discriminator under weight and spectral
    # normalization returns flattened logits and its feature maps.
    def setUp(self) -> None:
        # Builds one seeded weight-normalized scale sub-discriminator and its waveform.
        torch.manual_seed(0)
        self._discriminator: DiscriminatorS = DiscriminatorS()
        self._waveform: torch.Tensor = torch.randn(1, 1, 1024)

    def test_forward_returns_flattened_logits(self) -> None:
        # The adversarial term consumes one logit row per batch element.
        logits: torch.Tensor
        logits, _ = self._discriminator(self._waveform)
        self.assertEqual(logits.ndim, 2)
        self.assertEqual(logits.shape[0], 1)

    def test_forward_returns_one_feature_map_per_convolution(self) -> None:
        # Seven grouped convolutions plus the post-convolution give eight maps.
        feature_maps: list[torch.Tensor]
        _, feature_maps = self._discriminator(self._waveform)
        self.assertEqual(len(feature_maps), 8)

    def test_spectral_normalization_variant_produces_finite_logits(self) -> None:
        # The first ensemble member uses spectral normalization instead.
        spectral_discriminator: DiscriminatorS = DiscriminatorS(use_spectral_norm=True)
        logits: torch.Tensor
        logits, _ = spectral_discriminator(self._waveform)
        self.assertEqual(logits.ndim, 2)
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_forward_preserves_the_batch_dimension(self) -> None:
        # Each batch element receives its own logit row.
        logits: torch.Tensor
        logits, _ = self._discriminator(torch.randn(2, 1, 1024))
        self.assertEqual(logits.shape[0], 2)


class MultiPeriodDiscriminatorTest(unittest.TestCase):
    # Verifies the multi-period ensemble judges the real and synthesized
    # waveform through five sub-discriminators and returns aligned lists.
    def setUp(self) -> None:
        # Builds the seeded five-period ensemble and a distinct real and fake pair.
        torch.manual_seed(0)
        self._discriminator: MultiPeriodDiscriminator = MultiPeriodDiscriminator()
        self._real_waveform: torch.Tensor = torch.randn(1, 1, 1024)
        self._fake_waveform: torch.Tensor = torch.randn(1, 1, 1024)

    def test_forward_returns_four_lists_of_five_entries(self) -> None:
        # Real logits, fake logits, and both feature-map stacks stay aligned.
        real_logits: list[torch.Tensor]
        fake_logits: list[torch.Tensor]
        real_features: list[list[torch.Tensor]]
        fake_features: list[list[torch.Tensor]]
        real_logits, fake_logits, real_features, fake_features = self._discriminator(
            self._real_waveform,
            self._fake_waveform
        )
        self.assertEqual(len(real_logits), 5)
        self.assertEqual(len(fake_logits), 5)
        self.assertEqual(len(real_features), 5)
        self.assertEqual(len(fake_features), 5)

    def test_each_sub_discriminator_returns_six_feature_maps(self) -> None:
        # Feature matching pairs every real map with its fake counterpart.
        real_features: list[list[torch.Tensor]]
        fake_features: list[list[torch.Tensor]]
        _, _, real_features, fake_features = self._discriminator(
            self._real_waveform,
            self._fake_waveform
        )
        sub_index: int
        for sub_index in range(5):
            self.assertEqual(len(real_features[sub_index]), 6)
            self.assertEqual(len(fake_features[sub_index]), 6)

    def test_real_and_fake_logits_differ_for_different_waveforms(self) -> None:
        # The ensemble must respond to its input rather than a constant.
        real_logits: list[torch.Tensor]
        fake_logits: list[torch.Tensor]
        real_logits, fake_logits, _, _ = self._discriminator(
            self._real_waveform,
            self._fake_waveform
        )
        self.assertFalse(bool(torch.equal(real_logits[0], fake_logits[0])))

    def test_configured_periods_determine_the_ensemble_size(self) -> None:
        # The period tuple is the only degree of freedom of the ensemble.
        two_period_discriminator: MultiPeriodDiscriminator = MultiPeriodDiscriminator(periods=(2, 3))
        real_logits: list[torch.Tensor]
        real_logits, _, _, _ = two_period_discriminator(self._real_waveform, self._fake_waveform)
        self.assertEqual(len(real_logits), 2)

    def test_every_logit_is_finite(self) -> None:
        # A non-finite logit would break the least-squares objective.
        real_logits: list[torch.Tensor]
        real_logits, _, _, _ = self._discriminator(self._real_waveform, self._fake_waveform)
        logits: torch.Tensor
        for logits in real_logits:
            self.assertTrue(bool(torch.isfinite(logits).all()))


class MultiScaleDiscriminatorTest(unittest.TestCase):
    # Verifies the multi-scale ensemble judges three progressively
    # average-pooled views and returns aligned logits and feature maps.
    def setUp(self) -> None:
        # Builds the seeded three-scale ensemble and a distinct real and fake pair.
        torch.manual_seed(0)
        self._discriminator: MultiScaleDiscriminator = MultiScaleDiscriminator()
        self._real_waveform: torch.Tensor = torch.randn(1, 1, 1024)
        self._fake_waveform: torch.Tensor = torch.randn(1, 1, 1024)

    def test_forward_returns_four_lists_of_three_entries(self) -> None:
        # The ensemble holds one sub-discriminator per scale.
        real_logits: list[torch.Tensor]
        fake_logits: list[torch.Tensor]
        real_features: list[list[torch.Tensor]]
        fake_features: list[list[torch.Tensor]]
        real_logits, fake_logits, real_features, fake_features = self._discriminator(
            self._real_waveform,
            self._fake_waveform
        )
        self.assertEqual(len(real_logits), 3)
        self.assertEqual(len(fake_logits), 3)
        self.assertEqual(len(real_features), 3)
        self.assertEqual(len(fake_features), 3)

    def test_logit_width_shrinks_with_each_pooled_scale(self) -> None:
        # Average pooling between scales halves the time axis each step.
        real_logits: list[torch.Tensor]
        real_logits, _, _, _ = self._discriminator(self._real_waveform, self._fake_waveform)
        self.assertGreater(real_logits[0].shape[1], real_logits[1].shape[1])
        self.assertGreater(real_logits[1].shape[1], real_logits[2].shape[1])

    def test_each_scale_returns_eight_feature_maps(self) -> None:
        # Every scale runs the same convolution stack as the base scale.
        real_features: list[list[torch.Tensor]]
        _, _, real_features, _ = self._discriminator(self._real_waveform, self._fake_waveform)
        scale_features: list[torch.Tensor]
        for scale_features in real_features:
            self.assertEqual(len(scale_features), 8)

    def test_real_and_fake_logits_differ_for_different_waveforms(self) -> None:
        # The ensemble must respond to its input rather than a constant.
        real_logits: list[torch.Tensor]
        fake_logits: list[torch.Tensor]
        real_logits, fake_logits, _, _ = self._discriminator(
            self._real_waveform,
            self._fake_waveform
        )
        self.assertFalse(bool(torch.equal(real_logits[0], fake_logits[0])))

    def test_every_logit_is_finite(self) -> None:
        # A non-finite logit would break the least-squares objective.
        real_logits: list[torch.Tensor]
        real_logits, _, _, _ = self._discriminator(self._real_waveform, self._fake_waveform)
        logits: torch.Tensor
        for logits in real_logits:
            self.assertTrue(bool(torch.isfinite(logits).all()))


if __name__ == "__main__":
    unittest.main()
