# This module:
# 1. Verifies the Vocos multi-period discriminator ensemble: the per-period
#    sub-discriminator count, the logit and feature-map contract consumed by
#    the adversarial and feature-matching objectives, and the reflect-padding
#    branch that folds waveforms whose length is not a period multiple
# 2. Verifies the Vocos multi-resolution discriminator ensemble: the
#    per-resolution sub-discriminator count and the band-wise feature-map
#    contract of its spectral view
#
# Design decisions:
# - Both ensembles are exercised on a single 4096-sample synthetic waveform,
#   which is long enough for the widest analysis window of the resolution
#   ensemble while keeping the forward well under a second on CPU
# - Feature-map counts are asserted exactly, because the feature-matching
#   objective sums over them and a silently dropped map would weaken the
#   term without failing any shape check
# - Logit values are bounded for finiteness and real-versus-fake shape
#   agreement rather than pinned, because untrained discriminators carry no
#   meaningful decision values
#
# Author: Rahul Sawhney

import unittest

import torch

from vocode.models.vocos.discriminator import VocosMultiPeriodDiscriminator, VocosMultiResolutionDiscriminator


class WaveformPairBuilder:
    # Produces deterministic real and fake waveform pairs in the [batch, time] layout both ensembles consume.
    def __init__(self, sample_count: int) -> None:
        # Binds the sample count of every waveform this builder emits.
        self._sample_count: int = sample_count

    def build_real(self, seed: int) -> torch.Tensor:
        # Seeds the generator and returns the reference waveform.
        torch.manual_seed(seed)
        return torch.randn(1, self._sample_count)

    def build_fake(self, seed: int) -> torch.Tensor:
        # Seeds the generator and returns a distinct candidate waveform.
        # The seed is offset by one from the reference builder's, so a
        # pair requested under the same seed argument is guaranteed to
        # differ; without that offset both sides would be identical and
        # every real-against-fake comparison would pass vacuously.
        torch.manual_seed(seed + 1)
        return torch.randn(1, self._sample_count)


class VocosMultiPeriodDiscriminatorTest(unittest.TestCase):
    # Verifies the period ensemble's sub-discriminator count and its logit and feature-map contract.
    def setUp(self) -> None:
        # Builds the default period ensemble and the waveform pair builder.
        self._builder: WaveformPairBuilder = WaveformPairBuilder(sample_count=4096)
        torch.manual_seed(20260801)
        self._discriminator: VocosMultiPeriodDiscriminator = VocosMultiPeriodDiscriminator().eval()

    def test_default_ensemble_judges_through_five_periods(self) -> None:
        # The published recipe folds the waveform at the five prime periods.
        real_waveform: torch.Tensor = self._builder.build_real(seed=21)
        fake_waveform: torch.Tensor = self._builder.build_fake(seed=21)
        real_logits: list[torch.Tensor]
        fake_logits: list[torch.Tensor]
        real_features: list[list[torch.Tensor]]
        fake_features: list[list[torch.Tensor]]
        with torch.no_grad():
            real_logits, fake_logits, real_features, fake_features = self._discriminator(
                real_waveform,
                fake_waveform
            )
        self.assertEqual(len(real_logits), 5)
        self.assertEqual(len(fake_logits), 5)
        self.assertEqual(len(real_features), 5)
        self.assertEqual(len(fake_features), 5)

    def test_every_sub_discriminator_returns_five_feature_maps(self) -> None:
        # Four convolution stages plus the post-convolution feed the feature-matching term.
        real_waveform: torch.Tensor = self._builder.build_real(seed=22)
        fake_waveform: torch.Tensor = self._builder.build_fake(seed=22)
        real_features: list[list[torch.Tensor]]
        fake_features: list[list[torch.Tensor]]
        with torch.no_grad():
            _, _, real_features, fake_features = self._discriminator(real_waveform, fake_waveform)
        real_map_counts: list[int] = [len(feature_maps) for feature_maps in real_features]
        fake_map_counts: list[int] = [len(feature_maps) for feature_maps in fake_features]
        self.assertEqual(real_map_counts, [5, 5, 5, 5, 5])
        self.assertEqual(fake_map_counts, [5, 5, 5, 5, 5])

    def test_logits_are_flat_finite_and_shape_aligned(self) -> None:
        # Real and fake logits must be comparable per sub-discriminator for the hinge objective.
        real_waveform: torch.Tensor = self._builder.build_real(seed=23)
        fake_waveform: torch.Tensor = self._builder.build_fake(seed=23)
        real_logits: list[torch.Tensor]
        fake_logits: list[torch.Tensor]
        with torch.no_grad():
            real_logits, fake_logits, _, _ = self._discriminator(real_waveform, fake_waveform)
        index: int
        real_logit: torch.Tensor
        fake_logit: torch.Tensor
        # Strict pairing additionally asserts both ensembles judged the same sub-discriminator count.
        for index, (real_logit, fake_logit) in enumerate(zip(real_logits, fake_logits, strict=True)):
            self.assertEqual(real_logit.ndim, 2, msg=f"Sub-discriminator {index} logit is not flattened")
            self.assertEqual(real_logit.shape[0], 1)
            self.assertEqual(real_logit.shape, fake_logit.shape)
            self.assertTrue(bool(torch.isfinite(real_logit).all()))
            self.assertTrue(bool(torch.isfinite(fake_logit).all()))

    def test_length_that_is_not_a_period_multiple_is_reflect_padded(self) -> None:
        # A prime-length waveform exercises the padding branch of every period view.
        odd_builder: WaveformPairBuilder = WaveformPairBuilder(sample_count=4093)
        real_waveform: torch.Tensor = odd_builder.build_real(seed=24)
        fake_waveform: torch.Tensor = odd_builder.build_fake(seed=24)
        real_logits: list[torch.Tensor]
        fake_logits: list[torch.Tensor]
        with torch.no_grad():
            real_logits, fake_logits, _, _ = self._discriminator(real_waveform, fake_waveform)
        self.assertEqual(len(real_logits), 5)
        self.assertTrue(all(bool(torch.isfinite(logit).all()) for logit in real_logits))
        self.assertTrue(all(bool(torch.isfinite(logit).all()) for logit in fake_logits))
        logit: torch.Tensor
        for logit in real_logits:
            self.assertEqual(logit.ndim, 2)
            self.assertEqual(logit.shape[0], 1)

    def test_period_selection_controls_the_ensemble_size(self) -> None:
        # A reduced period tuple builds exactly one sub-discriminator per period.
        torch.manual_seed(25)
        reduced_discriminator: VocosMultiPeriodDiscriminator = VocosMultiPeriodDiscriminator(periods=(2, 3)).eval()
        real_waveform: torch.Tensor = self._builder.build_real(seed=25)
        fake_waveform: torch.Tensor = self._builder.build_fake(seed=25)
        real_logits: list[torch.Tensor]
        real_features: list[list[torch.Tensor]]
        with torch.no_grad():
            real_logits, _, real_features, _ = reduced_discriminator(real_waveform, fake_waveform)
        self.assertEqual(len(real_logits), 2)
        self.assertEqual(len(real_features), 2)


class VocosMultiResolutionDiscriminatorTest(unittest.TestCase):
    # Verifies the resolution ensemble's sub-discriminator count and its band-wise feature-map contract.
    def setUp(self) -> None:
        # Builds the default resolution ensemble and the waveform pair builder.
        self._builder: WaveformPairBuilder = WaveformPairBuilder(sample_count=4096)
        torch.manual_seed(20260802)
        self._discriminator: VocosMultiResolutionDiscriminator = VocosMultiResolutionDiscriminator().eval()

    def test_default_ensemble_judges_through_three_analysis_grids(self) -> None:
        # The published recipe judges spectral structure at three window lengths.
        real_waveform: torch.Tensor = self._builder.build_real(seed=31)
        fake_waveform: torch.Tensor = self._builder.build_fake(seed=31)
        real_logits: list[torch.Tensor]
        fake_logits: list[torch.Tensor]
        real_features: list[list[torch.Tensor]]
        fake_features: list[list[torch.Tensor]]
        with torch.no_grad():
            real_logits, fake_logits, real_features, fake_features = self._discriminator(
                real_waveform,
                fake_waveform
            )
        self.assertEqual(len(real_logits), 3)
        self.assertEqual(len(fake_logits), 3)
        self.assertEqual(len(real_features), 3)
        self.assertEqual(len(fake_features), 3)

    def test_every_sub_discriminator_returns_twenty_one_feature_maps(self) -> None:
        # Five frequency bands contribute four maps each, plus the merged post-convolution map.
        real_waveform: torch.Tensor = self._builder.build_real(seed=32)
        fake_waveform: torch.Tensor = self._builder.build_fake(seed=32)
        real_features: list[list[torch.Tensor]]
        fake_features: list[list[torch.Tensor]]
        with torch.no_grad():
            _, _, real_features, fake_features = self._discriminator(real_waveform, fake_waveform)
        real_map_counts: list[int] = [len(feature_maps) for feature_maps in real_features]
        fake_map_counts: list[int] = [len(feature_maps) for feature_maps in fake_features]
        self.assertEqual(real_map_counts, [21, 21, 21])
        self.assertEqual(fake_map_counts, [21, 21, 21])

    def test_logits_keep_the_spectral_layout_and_stay_finite(self) -> None:
        # The resolution head returns its merged spectral map rather than a flattened vector.
        real_waveform: torch.Tensor = self._builder.build_real(seed=33)
        fake_waveform: torch.Tensor = self._builder.build_fake(seed=33)
        real_logits: list[torch.Tensor]
        fake_logits: list[torch.Tensor]
        with torch.no_grad():
            real_logits, fake_logits, _, _ = self._discriminator(real_waveform, fake_waveform)
        index: int
        real_logit: torch.Tensor
        fake_logit: torch.Tensor
        # Strict pairing additionally asserts both ensembles judged the same resolution count.
        for index, (real_logit, fake_logit) in enumerate(zip(real_logits, fake_logits, strict=True)):
            self.assertEqual(real_logit.ndim, 4, msg=f"Resolution logit {index} lost its spectral layout")
            self.assertEqual(real_logit.shape[0], 1)
            self.assertEqual(real_logit.shape, fake_logit.shape)
            self.assertTrue(bool(torch.isfinite(real_logit).all()))
            self.assertTrue(bool(torch.isfinite(fake_logit).all()))

    def test_resolution_selection_controls_the_ensemble_size(self) -> None:
        # A reduced window tuple builds exactly one sub-discriminator per analysis grid.
        torch.manual_seed(34)
        reduced_discriminator: VocosMultiResolutionDiscriminator = VocosMultiResolutionDiscriminator(
            fft_sizes=(512,)
        ).eval()
        real_waveform: torch.Tensor = self._builder.build_real(seed=34)
        fake_waveform: torch.Tensor = self._builder.build_fake(seed=34)
        real_logits: list[torch.Tensor]
        real_features: list[list[torch.Tensor]]
        with torch.no_grad():
            real_logits, _, real_features, _ = reduced_discriminator(real_waveform, fake_waveform)
        self.assertEqual(len(real_logits), 1)
        self.assertEqual(len(real_features), 1)


if __name__ == "__main__":
    unittest.main()
