# This module:
# 1. Verifies the feature-matching loss: the L1 distance between the
#    discriminator's real and synthesized feature maps, summed over every
#    ensemble member and every layer
# 2. Verifies the detach contract, where the real feature maps are excluded
#    from the gradient path while the fake maps carry the exact gradient of
#    the per-map mean reduction
# 3. Verifies the empty-ensemble boundary and the finite scalar contract
#
# Design decisions:
# - Constant offsets between the real and fake maps make the expected L1
#   distance exactly derivable, so the summation over members and layers is
#   asserted against arithmetic rather than a recorded number
# - The same constant offsets make the gradient exactly derivable, so the
#   backward assertion pins the per-map mean rather than only proving that
#   some gradient arrived
# - The detach contract is asserted through autograd itself: the real maps
#   require gradient and must still have none after backward, which fails if
#   the detach call is ever dropped
# - Feature maps are minimal three-dimensional tensors because the loss reduces
#   over every axis and carries no shape-dependent behavior
#
# Author: Rahul Sawhney

import math
import unittest

import torch

from vocode.losses.feature_matching import FeatureMatchingLoss


class FeatureMapEnsembleBuilder:
    # Builds the nested member-by-layer feature-map ensembles under test.
    def __init__(self, member_count: int, layer_count: int) -> None:
        # Binds the constructor inputs into this component's state.
        self._member_count: int = member_count
        self._layer_count: int = layer_count
        self._map_shape: tuple[int, int, int] = (1, 2, 3)

    def constant(self, value: float) -> list[list[torch.Tensor]]:
        # Builds a constant feature-map stack for every ensemble member.
        ensemble: list[list[torch.Tensor]] = [
            [torch.full(self._map_shape, value) for layer_index in range(self._layer_count)]
            for member_index in range(self._member_count)
        ]
        return ensemble

    def constant_requiring_gradient(self, value: float) -> list[list[torch.Tensor]]:
        # Builds a constant ensemble whose maps participate in autograd.
        ensemble: list[list[torch.Tensor]] = [
            [
                torch.full(self._map_shape, value, requires_grad=True)
                for layer_index in range(self._layer_count)
            ]
            for member_index in range(self._member_count)
        ]
        return ensemble

    def seeded(self, seed: int) -> list[list[torch.Tensor]]:
        # Builds a seeded random feature-map stack for every ensemble member.
        torch.manual_seed(seed)
        ensemble: list[list[torch.Tensor]] = [
            [torch.randn(self._map_shape) for layer_index in range(self._layer_count)]
            for member_index in range(self._member_count)
        ]
        return ensemble

    @property
    def entry_count(self) -> int:
        # Returns the number of member-layer pairs the loss accumulates over.
        return self._member_count * self._layer_count

    @property
    def map_element_count(self) -> int:
        # Returns the number of entries in one feature map, which is the
        # denominator of that map's mean reduction.
        return math.prod(self._map_shape)


class FeatureMatchingDistanceTest(unittest.TestCase):
    # Verifies the L1 accumulation over ensemble members and layers.
    def setUp(self) -> None:
        # Two members and three layers make the two nesting levels
        # distinguishable: a reduction that walked only the outer list, or
        # that collapsed the inner one, would produce a different total than
        # the six per-layer distances this shape implies.
        self._loss: FeatureMatchingLoss = FeatureMatchingLoss()
        self._builder: FeatureMapEnsembleBuilder = FeatureMapEnsembleBuilder(member_count=2, layer_count=3)

    def test_loss_is_zero_for_identical_feature_maps(self) -> None:
        # Identical real and fake features are the analytic optimum.
        real_maps: list[list[torch.Tensor]] = self._builder.seeded(seed=11)
        fake_maps: list[list[torch.Tensor]] = self._builder.seeded(seed=11)
        value: torch.Tensor = self._loss(real_maps, fake_maps)
        self.assertAlmostEqual(
            float(value.item()),
            0.0,
            places=6,
            msg="Matching discriminator features must cost nothing"
        )

    def test_loss_equals_the_constant_offset_times_the_entry_count(self) -> None:
        # Every member-layer pair contributes the mean absolute offset.
        real_maps: list[list[torch.Tensor]] = self._builder.constant(1.0)
        fake_maps: list[list[torch.Tensor]] = self._builder.constant(1.5)
        value: torch.Tensor = self._loss(real_maps, fake_maps)
        expected: float = 0.5 * float(self._builder.entry_count)
        self.assertAlmostEqual(float(value.item()), expected, places=5)

    def test_loss_is_symmetric_in_the_sign_of_the_offset(self) -> None:
        # The absolute distance ignores which side of the offset is larger.
        positive: torch.Tensor = self._loss(self._builder.constant(1.0), self._builder.constant(1.5))
        negative: torch.Tensor = self._loss(self._builder.constant(1.5), self._builder.constant(1.0))
        self.assertAlmostEqual(float(positive.item()), float(negative.item()), places=6)

    def test_loss_grows_with_the_number_of_ensemble_members(self) -> None:
        # A wider ensemble accumulates strictly more distance at equal offset.
        narrow_builder: FeatureMapEnsembleBuilder = FeatureMapEnsembleBuilder(member_count=1, layer_count=3)
        narrow: torch.Tensor = self._loss(narrow_builder.constant(1.0), narrow_builder.constant(1.5))
        wide: torch.Tensor = self._loss(self._builder.constant(1.0), self._builder.constant(1.5))
        self.assertAlmostEqual(float(wide.item()), 2.0 * float(narrow.item()), places=5)

    def test_loss_returns_a_finite_non_negative_scalar(self) -> None:
        # The nested reduction collapses to a zero-dimensional tensor.
        real_maps: list[list[torch.Tensor]] = self._builder.seeded(seed=5)
        fake_maps: list[list[torch.Tensor]] = self._builder.seeded(seed=7)
        value: torch.Tensor = self._loss(real_maps, fake_maps)
        self.assertEqual(value.shape, torch.Size([]), msg="The objective must reduce to a scalar")
        self.assertTrue(torch.isfinite(value).item())
        self.assertGreater(float(value.item()), 0.0)

    def test_loss_is_zero_for_an_empty_ensemble(self) -> None:
        # The empty-ensemble guard returns a zero scalar instead of indexing.
        empty_real: list[list[torch.Tensor]] = []
        empty_fake: list[list[torch.Tensor]] = []
        value: torch.Tensor = self._loss(empty_real, empty_fake)
        self.assertEqual(value.shape, torch.Size([]))
        self.assertAlmostEqual(float(value.item()), 0.0, places=6)


class FeatureMatchingGradientPathTest(unittest.TestCase):
    # Verifies that the loss shapes the generator only, never the
    # discriminator's own representation.
    def setUp(self) -> None:
        # A smaller two-by-two ensemble suffices here, because these cases
        # assert on which tensors received gradient rather than on the
        # accumulated magnitude, and gradient presence is a property of every
        # participating tensor regardless of how many there are.
        self._loss: FeatureMatchingLoss = FeatureMatchingLoss()
        self._builder: FeatureMapEnsembleBuilder = FeatureMapEnsembleBuilder(member_count=2, layer_count=2)

    def test_backward_populates_gradient_on_the_fake_feature_maps(self) -> None:
        # The synthesized side carries the training signal. With the fake side
        # uniformly below the real side, every element gradient is exactly minus
        # one over the map's element count, which pins the per-map mean.
        real_maps: list[list[torch.Tensor]] = self._builder.constant(1.0)
        fake_maps: list[list[torch.Tensor]] = self._builder.constant_requiring_gradient(0.0)
        value: torch.Tensor = self._loss(real_maps, fake_maps)
        value.backward()
        expected_gradient: float = -1.0 / float(self._builder.map_element_count)
        member: list[torch.Tensor]
        for member in fake_maps:
            layer: torch.Tensor
            for layer in member:
                self.assertIsNotNone(layer.grad, msg="Fake feature maps must receive gradient")
                self.assertEqual(layer.grad.shape, layer.shape)
                self.assertTrue(
                    torch.allclose(layer.grad, torch.full_like(layer, expected_gradient)),
                    msg="Each map contributes its own mean absolute difference"
                )

    def test_backward_leaves_the_real_feature_maps_without_gradient(self) -> None:
        # Real features are detached, so the objective cannot reshape them.
        real_maps: list[list[torch.Tensor]] = self._builder.constant_requiring_gradient(1.0)
        fake_maps: list[list[torch.Tensor]] = self._builder.constant_requiring_gradient(0.0)
        value: torch.Tensor = self._loss(real_maps, fake_maps)
        value.backward()
        member: list[torch.Tensor]
        for member in real_maps:
            layer: torch.Tensor
            for layer in member:
                self.assertIsNone(
                    layer.grad,
                    msg="Real feature maps are detached and must stay outside the gradient path"
                )
