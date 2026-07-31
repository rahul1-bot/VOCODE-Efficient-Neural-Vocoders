# This module:
# 1. Verifies the MelGAN loss configuration: the reference feature-matching
#    weight of ten, the frozen record, and the rejection of unknown or
#    non-positive weights
# 2. Verifies the generator composition over the multi-scale ensemble: the
#    analytic zero at unit scores with matched features, the closed-form
#    adversarial value, the absence of any spectral reconstruction term, and
#    the gradient path
# 3. Verifies the discriminator composition and the strict pairing of the fake
#    and real ensemble structures
#
# Design decisions:
# - The adversarial term sums the squared score error across the channel and
#   time axes before averaging over the batch, so score tensors are given a
#   known element count and the expected value is derived from it
# - Strict zip pairing is exercised at both nesting levels, because a member or
#   layer count mismatch is a structural bug in the discriminator wiring rather
#   than a value that should be silently truncated
# - The absence of a mel term is asserted directly through the component panel,
#   which is what distinguishes this recipe from the HiFi-GAN family
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError

from vocode.losses.melgan import MelganLoss, MelganLossConfig


class ScaleEnsembleBuilder:
    # Builds the feature-and-score output tuples of the multi-scale ensemble.
    def __init__(self, member_count: int, layer_count: int) -> None:
        # Binds the constructor inputs into this component's state.
        self._member_count: int = member_count
        self._layer_count: int = layer_count
        self._score_shape: tuple[int, int, int] = (1, 1, 4)
        self._feature_shape: tuple[int, int, int] = (1, 2, 3)

    def outputs(self, score_value: float, feature_value: float) -> list[tuple[list[torch.Tensor], torch.Tensor]]:
        # Builds one constant feature-and-score output per ensemble member.
        ensemble: list[tuple[list[torch.Tensor], torch.Tensor]] = [
            (
                [torch.full(self._feature_shape, feature_value) for layer_index in range(self._layer_count)],
                torch.full(self._score_shape, score_value)
            )
            for member_index in range(self._member_count)
        ]
        return ensemble

    def outputs_requiring_gradient(
        self,
        score_value: float,
        feature_value: float
    ) -> list[tuple[list[torch.Tensor], torch.Tensor]]:
        # Builds an ensemble whose scores and features participate in autograd.
        ensemble: list[tuple[list[torch.Tensor], torch.Tensor]] = [
            (
                [
                    torch.full(self._feature_shape, feature_value, requires_grad=True)
                    for layer_index in range(self._layer_count)
                ],
                torch.full(self._score_shape, score_value, requires_grad=True)
            )
            for member_index in range(self._member_count)
        ]
        return ensemble

    @property
    def score_element_count(self) -> int:
        # Returns the number of score entries summed inside one member.
        return self._score_shape[1] * self._score_shape[2]

    @property
    def member_count(self) -> int:
        # Returns the number of sub-discriminators in the ensemble.
        return self._member_count


class MelganLossConfigurationTest(unittest.TestCase):
    # Verifies the frozen weight record behind the MelGAN composition.
    def setUp(self) -> None:
        # Constructs the record from its defaults with no arguments, so the
        # cases below assert the reference weight the shipped configuration
        # actually applies. This record has only one field, which is itself
        # part of what the rejection cases prove.
        self._configuration: MelganLossConfig = MelganLossConfig()

    def test_default_feature_matching_weight_matches_the_reference_recipe(self) -> None:
        # MelGAN leans on feature matching at weight ten instead of a mel term.
        self.assertEqual(self._configuration.feature_matching_weight, 10.0)

    def test_configuration_rejects_mutation_after_construction(self) -> None:
        # The record is frozen so a run cannot drift from its logged weights.
        with self.assertRaises(ValidationError):
            self._configuration.feature_matching_weight: float = 1.0

    def test_configuration_rejects_an_unknown_weight(self) -> None:
        # MelGAN carries no mel weight; naming one must fail loudly.
        with self.assertRaises(ValidationError):
            MelganLossConfig(mel_reconstruction_weight=45.0)

    def test_configuration_rejects_a_non_positive_weight(self) -> None:
        # The feature-matching weight is strictly positive.
        with self.assertRaises(ValidationError):
            MelganLossConfig(feature_matching_weight=0.0)

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The composite exposes the exact record it was constructed with.
        configuration: MelganLossConfig = MelganLossConfig(feature_matching_weight=4.0)
        loss: MelganLoss = MelganLoss(configuration)
        self.assertIs(loss.configuration, configuration)


class MelganGeneratorObjectiveTest(unittest.TestCase):
    # Verifies the generator composition, its optimum, and its weighting.
    def setUp(self) -> None:
        # Two members and two layers is the smallest ensemble that keeps the
        # per-member summation observable and still lets the strict pairing be
        # violated in either direction, by narrowing the member count or the
        # layer count against this fixture. The loss is built from the default
        # weight so the composition arithmetic reflects the shipped recipe.
        self._builder: ScaleEnsembleBuilder = ScaleEnsembleBuilder(member_count=2, layer_count=2)
        self._loss: MelganLoss = MelganLoss(MelganLossConfig())

    def test_generator_loss_is_zero_at_unit_scores_with_matched_features(self) -> None:
        # A fully fooled ensemble with identical features is the analytic optimum.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_generator_loss(
            fake_outputs=self._builder.outputs(score_value=1.0, feature_value=0.5),
            real_outputs=self._builder.outputs(score_value=1.0, feature_value=0.5)
        )
        self.assertAlmostEqual(
            float(total.item()),
            0.0,
            places=6,
            msg="A perfect generator must pay nothing under the MelGAN composition"
        )
        self.assertAlmostEqual(components["generator_loss_adversarial"], 0.0, places=6)
        self.assertAlmostEqual(components["generator_loss_feature_matching"], 0.0, places=6)

    def test_adversarial_term_sums_squared_score_error_over_channel_and_time(self) -> None:
        # Zero scores cost one per summed score entry, per ensemble member.
        components: dict[str, float]
        _, components = self._loss.compute_generator_loss(
            fake_outputs=self._builder.outputs(score_value=0.0, feature_value=0.5),
            real_outputs=self._builder.outputs(score_value=1.0, feature_value=0.5)
        )
        expected: float = float(self._builder.score_element_count * self._builder.member_count)
        self.assertAlmostEqual(components["generator_loss_adversarial"], expected, places=5)

    def test_generator_total_applies_the_feature_matching_weight(self) -> None:
        # The total is the adversarial term plus the weighted matching term.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_generator_loss(
            fake_outputs=self._builder.outputs(score_value=0.5, feature_value=0.0),
            real_outputs=self._builder.outputs(score_value=1.0, feature_value=1.0)
        )
        expected: float = (
            components["generator_loss_adversarial"]
            + self._loss.configuration.feature_matching_weight * components["generator_loss_feature_matching"]
        )
        self.assertAlmostEqual(float(total.item()), expected, places=4)
        self.assertAlmostEqual(components["generator_loss_total"], float(total.item()), places=5)

    def test_generator_components_carry_no_spectral_reconstruction_term(self) -> None:
        # MelGAN trains without a mel term, unlike the HiFi-GAN family.
        components: dict[str, float]
        _, components = self._loss.compute_generator_loss(
            fake_outputs=self._builder.outputs(score_value=0.0, feature_value=0.0),
            real_outputs=self._builder.outputs(score_value=1.0, feature_value=1.0)
        )
        expected_keys: set[str] = {
            "generator_loss_total",
            "generator_loss_adversarial",
            "generator_loss_feature_matching"
        }
        self.assertEqual(set(components), expected_keys)

    def test_generator_loss_propagates_gradient_to_the_fake_outputs(self) -> None:
        # Both the score and the feature path must reach the generator.
        fake_outputs: list[tuple[list[torch.Tensor], torch.Tensor]] = (
            self._builder.outputs_requiring_gradient(score_value=0.0, feature_value=0.0)
        )
        total: torch.Tensor
        total, _ = self._loss.compute_generator_loss(
            fake_outputs=fake_outputs,
            real_outputs=self._builder.outputs(score_value=1.0, feature_value=1.0)
        )
        total.backward()
        member: tuple[list[torch.Tensor], torch.Tensor]
        for member in fake_outputs:
            self.assertIsNotNone(member[1].grad, msg="Fake scores must receive gradient")
            self.assertIsNotNone(member[0][0].grad, msg="Fake features must receive gradient")
            self.assertTrue(torch.isfinite(member[1].grad).all().item())

    def test_generator_loss_leaves_the_real_features_without_gradient(self) -> None:
        # Real features are detached so the term shapes only the generator.
        real_outputs: list[tuple[list[torch.Tensor], torch.Tensor]] = (
            self._builder.outputs_requiring_gradient(score_value=1.0, feature_value=1.0)
        )
        total: torch.Tensor
        total, _ = self._loss.compute_generator_loss(
            fake_outputs=self._builder.outputs_requiring_gradient(score_value=0.0, feature_value=0.0),
            real_outputs=real_outputs
        )
        total.backward()
        member: tuple[list[torch.Tensor], torch.Tensor]
        for member in real_outputs:
            self.assertIsNone(
                member[0][0].grad,
                msg="Real features are detached and must stay outside the gradient path"
            )

    def test_generator_loss_rejects_a_member_count_mismatch(self) -> None:
        # A structural mismatch between the ensembles must fail closed.
        narrow_builder: ScaleEnsembleBuilder = ScaleEnsembleBuilder(member_count=1, layer_count=2)
        with self.assertRaisesRegex(ValueError, "zip"):
            self._loss.compute_generator_loss(
                fake_outputs=narrow_builder.outputs(score_value=0.0, feature_value=0.0),
                real_outputs=self._builder.outputs(score_value=1.0, feature_value=1.0)
            )

    def test_generator_loss_rejects_a_layer_count_mismatch(self) -> None:
        # Unequal feature depth inside a member is a wiring bug, not a truncation.
        shallow_builder: ScaleEnsembleBuilder = ScaleEnsembleBuilder(member_count=2, layer_count=1)
        with self.assertRaisesRegex(ValueError, "zip"):
            self._loss.compute_generator_loss(
                fake_outputs=shallow_builder.outputs(score_value=0.0, feature_value=0.0),
                real_outputs=self._builder.outputs(score_value=1.0, feature_value=1.0)
            )


class MelganDiscriminatorObjectiveTest(unittest.TestCase):
    # Verifies the discriminator separation term over the multi-scale ensemble.
    def setUp(self) -> None:
        # Two members and two layers is the smallest ensemble that keeps the
        # per-member summation observable and still lets the strict pairing be
        # violated in either direction, by narrowing the member count or the
        # layer count against this fixture. The loss is built from the default
        # weight so the composition arithmetic reflects the shipped recipe.
        self._builder: ScaleEnsembleBuilder = ScaleEnsembleBuilder(member_count=2, layer_count=2)
        self._loss: MelganLoss = MelganLoss(MelganLossConfig())

    def test_discriminator_loss_is_zero_for_a_perfect_separation(self) -> None:
        # Real scores at one and fake scores at zero are the analytic optimum.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_discriminator_loss(
            fake_outputs=self._builder.outputs(score_value=0.0, feature_value=0.0),
            real_outputs=self._builder.outputs(score_value=1.0, feature_value=1.0)
        )
        self.assertAlmostEqual(float(total.item()), 0.0, places=6)
        self.assertAlmostEqual(components["discriminator_loss_total"], 0.0, places=6)

    def test_discriminator_loss_matches_the_closed_form_for_inverted_scores(self) -> None:
        # A fully inverted member costs one per score entry on each side.
        total: torch.Tensor
        total, _ = self._loss.compute_discriminator_loss(
            fake_outputs=self._builder.outputs(score_value=1.0, feature_value=0.0),
            real_outputs=self._builder.outputs(score_value=0.0, feature_value=0.0)
        )
        expected: float = float(2 * self._builder.score_element_count * self._builder.member_count)
        self.assertAlmostEqual(float(total.item()), expected, places=5)

    def test_discriminator_components_expose_only_the_total(self) -> None:
        # The reference discriminator step logs a single scalar term.
        components: dict[str, float]
        _, components = self._loss.compute_discriminator_loss(
            fake_outputs=self._builder.outputs(score_value=0.5, feature_value=0.0),
            real_outputs=self._builder.outputs(score_value=0.5, feature_value=0.0)
        )
        self.assertEqual(set(components), {"discriminator_loss_total"})

    def test_discriminator_loss_propagates_gradient_to_the_fake_scores(self) -> None:
        # The separation term must reach the scores under optimization.
        fake_outputs: list[tuple[list[torch.Tensor], torch.Tensor]] = (
            self._builder.outputs_requiring_gradient(score_value=0.5, feature_value=0.0)
        )
        total: torch.Tensor
        total, _ = self._loss.compute_discriminator_loss(
            fake_outputs=fake_outputs,
            real_outputs=self._builder.outputs(score_value=1.0, feature_value=0.0)
        )
        total.backward()
        member: tuple[list[torch.Tensor], torch.Tensor]
        for member in fake_outputs:
            self.assertIsNotNone(member[1].grad, msg="Fake scores must receive gradient")
            self.assertTrue(torch.isfinite(member[1].grad).all().item())

    def test_discriminator_loss_rejects_a_member_count_mismatch(self) -> None:
        # A structural mismatch between the ensembles must fail closed.
        narrow_builder: ScaleEnsembleBuilder = ScaleEnsembleBuilder(member_count=1, layer_count=2)
        with self.assertRaisesRegex(ValueError, "zip"):
            self._loss.compute_discriminator_loss(
                fake_outputs=narrow_builder.outputs(score_value=0.0, feature_value=0.0),
                real_outputs=self._builder.outputs(score_value=1.0, feature_value=0.0)
            )
