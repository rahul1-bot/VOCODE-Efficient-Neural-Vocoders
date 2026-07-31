# This module:
# 1. Verifies the MelGAN window-based discriminator: the intermediate
#    feature maps the feature-matching objective consumes, the final
#    logit map, and the channel layout of both
# 2. Verifies the multi-scale ensemble: one discriminator per scale, the
#    unpooled first scale, and the progressively pooled remainder
#
# Design decisions:
# - Logit widths are compared across scales rather than pinned, because
#   the exact width follows from the strided convolution arithmetic and
#   carries no reference value; the strictly decreasing relation is the
#   observable signature of the average pooling between scales
# - The unpooled first scale is verified by shape agreement with a
#   standalone discriminator on the same waveform, which is the observable
#   consequence of the identity pooler
# - Waveforms are two thousand forty-eight samples, the smallest length
#   that survives the strided stack at the most pooled scale
#
# Author: Rahul Sawhney

import unittest

import torch

from vocode.models.melgan.discriminator import MelganDiscriminator, MelganMultiScaleDiscriminator


class MelganDiscriminatorTest(unittest.TestCase):
    # Verifies the single-scale discriminator returns its intermediate
    # feature maps separately from the final logit map.
    def setUp(self) -> None:
        # Builds one seeded window discriminator and a waveform long enough to
        # survive the strided stack at the most pooled scale.
        torch.manual_seed(0)
        self._discriminator: MelganDiscriminator = MelganDiscriminator()
        self._waveform: torch.Tensor = torch.randn(1, 1, 2048)

    def test_forward_returns_feature_maps_and_logits(self) -> None:
        # The generator objective reads both halves of this pair.
        feature_maps: list[torch.Tensor]
        logits: torch.Tensor
        feature_maps, logits = self._discriminator(self._waveform)
        self.assertIsInstance(feature_maps, list)
        self.assertIsInstance(logits, torch.Tensor)

    def test_forward_excludes_the_logit_map_from_the_features(self) -> None:
        # Six intermediate layers feed feature matching; the seventh is the logit.
        feature_maps: list[torch.Tensor]
        logits: torch.Tensor
        feature_maps, logits = self._discriminator(self._waveform)
        self.assertEqual(len(feature_maps), 6)
        self.assertEqual(logits.shape[1], 1)
        self.assertNotEqual(feature_maps[-1].shape[1], 1)

    def test_logits_keep_the_batch_and_channel_layout(self) -> None:
        # The hinge objective consumes a single-channel logit map.
        logits: torch.Tensor
        _, logits = self._discriminator(self._waveform)
        self.assertEqual(logits.ndim, 3)
        self.assertEqual(logits.shape[0], 1)
        self.assertGreater(logits.shape[2], 0)

    def test_logits_are_finite(self) -> None:
        # A non-finite logit would break the hinge objective.
        logits: torch.Tensor
        _, logits = self._discriminator(self._waveform)
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_forward_preserves_the_batch_dimension(self) -> None:
        # Each batch element receives its own logit map.
        logits: torch.Tensor
        _, logits = self._discriminator(torch.randn(2, 1, 2048))
        self.assertEqual(logits.shape[0], 2)

    def test_forward_responds_to_its_input(self) -> None:
        # The discriminator must judge the waveform rather than a constant.
        first_logits: torch.Tensor
        second_logits: torch.Tensor
        _, first_logits = self._discriminator(self._waveform)
        _, second_logits = self._discriminator(torch.randn(1, 1, 2048))
        self.assertFalse(bool(torch.equal(first_logits, second_logits)))


class MelganMultiScaleDiscriminatorTest(unittest.TestCase):
    # Verifies the ensemble judges three scales, leaving the first
    # unpooled and average-pooling each subsequent scale.
    def setUp(self) -> None:
        # Builds the seeded three-scale ensemble on the same waveform length.
        torch.manual_seed(0)
        self._discriminator: MelganMultiScaleDiscriminator = MelganMultiScaleDiscriminator()
        self._waveform: torch.Tensor = torch.randn(1, 1, 2048)

    def test_ensemble_returns_one_output_per_scale(self) -> None:
        # The reference ensemble judges three progressively pooled views.
        outputs: list[tuple[list[torch.Tensor], torch.Tensor]] = self._discriminator(self._waveform)
        self.assertEqual(len(outputs), 3)

    def test_every_scale_returns_six_feature_maps(self) -> None:
        # Feature matching pairs the same six layers at every scale.
        outputs: list[tuple[list[torch.Tensor], torch.Tensor]] = self._discriminator(self._waveform)
        scale_features: list[torch.Tensor]
        scale_logits: torch.Tensor
        for scale_features, scale_logits in outputs:
            self.assertEqual(len(scale_features), 6)
            self.assertEqual(scale_logits.shape[1], 1)

    def test_logit_width_shrinks_with_each_pooled_scale(self) -> None:
        # Average pooling halves the time axis between consecutive scales.
        outputs: list[tuple[list[torch.Tensor], torch.Tensor]] = self._discriminator(self._waveform)
        self.assertGreater(outputs[0][1].shape[2], outputs[1][1].shape[2])
        self.assertGreater(outputs[1][1].shape[2], outputs[2][1].shape[2])

    def test_the_first_scale_sees_the_unpooled_waveform(self) -> None:
        # The first pooler is an identity, so the base scale keeps full rate.
        standalone_discriminator: MelganDiscriminator = MelganDiscriminator()
        standalone_logits: torch.Tensor
        _, standalone_logits = standalone_discriminator(self._waveform)
        outputs: list[tuple[list[torch.Tensor], torch.Tensor]] = self._discriminator(self._waveform)
        self.assertEqual(outputs[0][1].shape, standalone_logits.shape)

    def test_every_scale_returns_finite_logits(self) -> None:
        # A non-finite logit would break the hinge objective.
        outputs: list[tuple[list[torch.Tensor], torch.Tensor]] = self._discriminator(self._waveform)
        scale_logits: torch.Tensor
        for _, scale_logits in outputs:
            self.assertTrue(bool(torch.isfinite(scale_logits).all()))

    def test_ensemble_preserves_the_batch_dimension(self) -> None:
        # Each batch element receives its own logit map at every scale.
        outputs: list[tuple[list[torch.Tensor], torch.Tensor]] = self._discriminator(
            torch.randn(2, 1, 2048)
        )
        scale_logits: torch.Tensor
        for _, scale_logits in outputs:
            self.assertEqual(scale_logits.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
