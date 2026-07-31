# This module:
# 1. Verifies the BigVGAN loss configuration: the reference weights, the
#    frozen record, and the rejection of unknown or non-positive weights
# 2. Verifies the generator composition over the period and multi-resolution
#    ensembles: the analytic zero at perfect reconstruction with full fooling,
#    the weighted sum of the reported components, and the gradient path
# 3. Verifies the discriminator composition and its analytic zero at perfect
#    separation
#
# Design decisions:
# - The total is asserted against the weighted sum of the components the call
#   itself reports, so the test pins the composition arithmetic rather than a
#   recorded floating-point total
# - The two ensembles are given deliberately different logit values so that a
#   swapped period and resolution term would surface as unequal components
# - Fixtures are constant minimal tensors: the composite consumes discriminator
#   logits and feature maps directly, so no discriminator network is built here
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError

from vocode.losses.bigvgan import BigvganLoss, BigvganLossConfig


class GanFixtureBuilder:
    # Builds the logit, feature-map, and mel fixtures the composite consumes.
    def __init__(self, member_count: int, layer_count: int) -> None:
        # Binds the constructor inputs into this component's state.
        self._member_count: int = member_count
        self._layer_count: int = layer_count
        self._logit_shape: tuple[int, int] = (1, 4)
        self._feature_shape: tuple[int, int, int] = (1, 2, 3)
        self._mel_shape: tuple[int, int, int] = (1, 8, 12)

    def logits(self, value: float) -> list[torch.Tensor]:
        # Builds one constant logit tensor per ensemble member.
        ensemble: list[torch.Tensor] = [
            torch.full(self._logit_shape, value) for member_index in range(self._member_count)
        ]
        return ensemble

    def logits_requiring_gradient(self, value: float) -> list[torch.Tensor]:
        # Builds a constant logit ensemble that participates in autograd.
        ensemble: list[torch.Tensor] = [
            torch.full(self._logit_shape, value, requires_grad=True)
            for member_index in range(self._member_count)
        ]
        return ensemble

    def feature_maps(self, value: float) -> list[list[torch.Tensor]]:
        # Builds a constant feature-map stack for every ensemble member.
        ensemble: list[list[torch.Tensor]] = [
            [torch.full(self._feature_shape, value) for layer_index in range(self._layer_count)]
            for member_index in range(self._member_count)
        ]
        return ensemble

    def mel(self, value: float) -> torch.Tensor:
        # Builds a log-mel spectrogram whose every entry holds the given value.
        return torch.full(self._mel_shape, value)

    def seeded_mel(self, seed: int) -> torch.Tensor:
        # Builds a seeded random log-mel spectrogram.
        torch.manual_seed(seed)
        return torch.randn(self._mel_shape)


class BigvganLossConfigurationTest(unittest.TestCase):
    # Verifies the frozen weight record behind the BigVGAN composition.
    def setUp(self) -> None:
        # Constructs the record from its defaults with no arguments, so the
        # cases below assert the reference weights the shipped configuration
        # actually applies rather than values restated at the call site.
        self._configuration: BigvganLossConfig = BigvganLossConfig()

    def test_default_weights_match_the_reference_recipe(self) -> None:
        # BigVGAN inherits the HiFi-GAN weighting of the reference recipe.
        self.assertEqual(self._configuration.mel_reconstruction_weight, 45.0)
        self.assertEqual(self._configuration.feature_matching_weight, 2.0)
        self.assertEqual(self._configuration.adversarial_weight, 1.0)

    def test_configuration_rejects_mutation_after_construction(self) -> None:
        # The record is frozen so a run cannot drift from its logged weights.
        with self.assertRaises(ValidationError):
            self._configuration.adversarial_weight: float = 5.0

    def test_configuration_rejects_an_unknown_weight(self) -> None:
        # An unknown field is a typo, never a silently ignored setting.
        with self.assertRaises(ValidationError):
            BigvganLossConfig(resolution_weight=0.1)

    def test_configuration_rejects_a_negative_weight(self) -> None:
        # Every weight is strictly positive in the reference composition.
        with self.assertRaises(ValidationError):
            BigvganLossConfig(mel_reconstruction_weight=-45.0)

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The composite exposes the exact record it was constructed with.
        configuration: BigvganLossConfig = BigvganLossConfig(feature_matching_weight=3.0)
        loss: BigvganLoss = BigvganLoss(configuration)
        self.assertIs(loss.configuration, configuration)


class BigvganGeneratorObjectiveTest(unittest.TestCase):
    # Verifies the generator composition, its optimum, and its gradient path.
    def setUp(self) -> None:
        # Two members and two layers is the smallest ensemble that still
        # distinguishes a sum over sub-discriminators from a mean, which is
        # what this family's unnormalized accumulation turns on. The loss is
        # built from the default weights, so the composition arithmetic is
        # checked against the shipped recipe rather than a test-only one.
        self._builder: GanFixtureBuilder = GanFixtureBuilder(member_count=2, layer_count=2)
        self._loss: BigvganLoss = BigvganLoss(BigvganLossConfig())

    def test_generator_loss_is_zero_at_perfect_reconstruction_and_full_fooling(self) -> None:
        # Identical mels, matched features, and unit logits are the analytic optimum.
        reference_mel: torch.Tensor = self._builder.seeded_mel(seed=61)
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_generator_loss(
            reference_mel=reference_mel,
            synthesized_mel=reference_mel,
            fake_period_logits=self._builder.logits(1.0),
            fake_resolution_logits=self._builder.logits(1.0),
            real_period_features=self._builder.feature_maps(0.5),
            fake_period_features=self._builder.feature_maps(0.5),
            real_resolution_features=self._builder.feature_maps(0.5),
            fake_resolution_features=self._builder.feature_maps(0.5)
        )
        self.assertAlmostEqual(
            float(total.item()),
            0.0,
            places=6,
            msg="A perfect generator must pay nothing under the BigVGAN composition"
        )
        self.assertAlmostEqual(components["generator_loss_mel"], 0.0, places=6)
        self.assertAlmostEqual(components["generator_loss_feature_matching_resolution"], 0.0, places=6)

    def test_generator_total_equals_the_weighted_sum_of_its_components(self) -> None:
        # The composition arithmetic must match the reported per-term values.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_generator_loss(
            reference_mel=self._builder.seeded_mel(seed=3),
            synthesized_mel=self._builder.seeded_mel(seed=4),
            fake_period_logits=self._builder.logits(0.25),
            fake_resolution_logits=self._builder.logits(-0.5),
            real_period_features=self._builder.feature_maps(1.0),
            fake_period_features=self._builder.feature_maps(0.25),
            real_resolution_features=self._builder.feature_maps(1.0),
            fake_resolution_features=self._builder.feature_maps(-0.5)
        )
        configuration: BigvganLossConfig = self._loss.configuration
        expected: float = (
            configuration.mel_reconstruction_weight * components["generator_loss_mel"]
            + configuration.feature_matching_weight * (
                components["generator_loss_feature_matching_period"]
                + components["generator_loss_feature_matching_resolution"]
            )
            + configuration.adversarial_weight * (
                components["generator_loss_adversarial_period"]
                + components["generator_loss_adversarial_resolution"]
            )
        )
        self.assertAlmostEqual(float(total.item()), expected, places=4)
        self.assertAlmostEqual(components["generator_loss_total"], float(total.item()), places=5)

    def test_generator_keeps_the_period_and_resolution_terms_separate(self) -> None:
        # Different logits per ensemble must produce different reported terms.
        components: dict[str, float]
        _, components = self._loss.compute_generator_loss(
            reference_mel=self._builder.mel(0.0),
            synthesized_mel=self._builder.mel(0.0),
            fake_period_logits=self._builder.logits(0.0),
            fake_resolution_logits=self._builder.logits(0.5),
            real_period_features=self._builder.feature_maps(1.0),
            fake_period_features=self._builder.feature_maps(0.0),
            real_resolution_features=self._builder.feature_maps(1.0),
            fake_resolution_features=self._builder.feature_maps(0.5)
        )
        self.assertGreater(
            components["generator_loss_adversarial_period"],
            components["generator_loss_adversarial_resolution"],
            msg="Logits further from one must cost more, keeping the ensembles distinguishable"
        )
        self.assertGreater(
            components["generator_loss_feature_matching_period"],
            components["generator_loss_feature_matching_resolution"]
        )

    def test_generator_components_expose_the_reference_keys(self) -> None:
        # The logged component panel is part of the training contract.
        components: dict[str, float]
        _, components = self._loss.compute_generator_loss(
            reference_mel=self._builder.mel(0.0),
            synthesized_mel=self._builder.mel(1.0),
            fake_period_logits=self._builder.logits(0.0),
            fake_resolution_logits=self._builder.logits(0.0),
            real_period_features=self._builder.feature_maps(1.0),
            fake_period_features=self._builder.feature_maps(0.0),
            real_resolution_features=self._builder.feature_maps(1.0),
            fake_resolution_features=self._builder.feature_maps(0.0)
        )
        expected_keys: set[str] = {
            "generator_loss_total",
            "generator_loss_mel",
            "generator_loss_adversarial_period",
            "generator_loss_adversarial_resolution",
            "generator_loss_feature_matching_period",
            "generator_loss_feature_matching_resolution"
        }
        self.assertEqual(set(components), expected_keys)

    def test_generator_loss_propagates_gradient_to_the_synthesized_mel(self) -> None:
        # The reconstruction term must reach the generator's spectrogram.
        synthesized_mel: torch.Tensor = self._builder.seeded_mel(seed=71).requires_grad_(True)
        total: torch.Tensor
        total, _ = self._loss.compute_generator_loss(
            reference_mel=self._builder.seeded_mel(seed=72),
            synthesized_mel=synthesized_mel,
            fake_period_logits=self._builder.logits(0.0),
            fake_resolution_logits=self._builder.logits(0.0),
            real_period_features=self._builder.feature_maps(1.0),
            fake_period_features=self._builder.feature_maps(0.0),
            real_resolution_features=self._builder.feature_maps(1.0),
            fake_resolution_features=self._builder.feature_maps(0.0)
        )
        total.backward()
        self.assertIsNotNone(synthesized_mel.grad, msg="The synthesized mel must receive gradient")
        self.assertTrue(torch.isfinite(synthesized_mel.grad).all().item())

    def test_generator_loss_rejects_a_mel_shape_mismatch(self) -> None:
        # The mel guard fails closed rather than cropping inside the composite.
        with self.assertRaisesRegex(ValueError, "identical shape"):
            self._loss.compute_generator_loss(
                reference_mel=torch.zeros(1, 8, 12),
                synthesized_mel=torch.zeros(1, 6, 12),
                fake_period_logits=self._builder.logits(0.0),
                fake_resolution_logits=self._builder.logits(0.0),
                real_period_features=self._builder.feature_maps(0.0),
                fake_period_features=self._builder.feature_maps(0.0),
                real_resolution_features=self._builder.feature_maps(0.0),
                fake_resolution_features=self._builder.feature_maps(0.0)
            )


class BigvganDiscriminatorObjectiveTest(unittest.TestCase):
    # Verifies the discriminator composition over both ensembles.
    def setUp(self) -> None:
        # Two members and two layers is the smallest ensemble that still
        # distinguishes a sum over sub-discriminators from a mean, which is
        # what this family's unnormalized accumulation turns on. The loss is
        # built from the default weights, so the composition arithmetic is
        # checked against the shipped recipe rather than a test-only one.
        self._builder: GanFixtureBuilder = GanFixtureBuilder(member_count=2, layer_count=2)
        self._loss: BigvganLoss = BigvganLoss(BigvganLossConfig())

    def test_discriminator_loss_is_zero_for_a_perfect_separation(self) -> None:
        # Real logits at one and fake logits at zero are the analytic optimum.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_discriminator_loss(
            real_period_logits=self._builder.logits(1.0),
            fake_period_logits=self._builder.logits(0.0),
            real_resolution_logits=self._builder.logits(1.0),
            fake_resolution_logits=self._builder.logits(0.0)
        )
        self.assertAlmostEqual(float(total.item()), 0.0, places=6)
        self.assertAlmostEqual(components["discriminator_loss_period"], 0.0, places=6)
        self.assertAlmostEqual(components["discriminator_loss_resolution"], 0.0, places=6)

    def test_discriminator_total_is_the_unweighted_sum_of_both_ensembles(self) -> None:
        # Neither ensemble carries a weight in the reference discriminator step.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_discriminator_loss(
            real_period_logits=self._builder.logits(0.25),
            fake_period_logits=self._builder.logits(0.75),
            real_resolution_logits=self._builder.logits(-0.5),
            fake_resolution_logits=self._builder.logits(0.5)
        )
        expected: float = (
            components["discriminator_loss_period"] + components["discriminator_loss_resolution"]
        )
        self.assertAlmostEqual(float(total.item()), expected, places=5)
        self.assertGreater(float(total.item()), 0.0)

    def test_discriminator_components_expose_the_reference_keys(self) -> None:
        # The logged component panel is part of the training contract.
        components: dict[str, float]
        _, components = self._loss.compute_discriminator_loss(
            real_period_logits=self._builder.logits(1.0),
            fake_period_logits=self._builder.logits(0.5),
            real_resolution_logits=self._builder.logits(1.0),
            fake_resolution_logits=self._builder.logits(0.5)
        )
        expected_keys: set[str] = {
            "discriminator_loss_total",
            "discriminator_loss_period",
            "discriminator_loss_resolution"
        }
        self.assertEqual(set(components), expected_keys)

    def test_discriminator_loss_propagates_gradient_to_the_fake_logits(self) -> None:
        # The separation term must reach the ensemble logits under optimization.
        fake_resolution_logits: list[torch.Tensor] = self._builder.logits_requiring_gradient(0.5)
        total: torch.Tensor
        total, _ = self._loss.compute_discriminator_loss(
            real_period_logits=self._builder.logits(1.0),
            fake_period_logits=self._builder.logits(0.5),
            real_resolution_logits=self._builder.logits(1.0),
            fake_resolution_logits=fake_resolution_logits
        )
        total.backward()
        member: torch.Tensor
        for member in fake_resolution_logits:
            self.assertIsNotNone(member.grad, msg="Fake resolution logits must receive gradient")
            self.assertTrue(torch.isfinite(member.grad).all().item())
