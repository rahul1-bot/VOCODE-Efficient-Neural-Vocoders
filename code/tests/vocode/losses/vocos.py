# This module:
# 1. Verifies the Vocos loss configuration: the reference mel weight and the
#    multi-resolution discriminator weight, the frozen record, and the
#    rejection of unknown or non-positive weights
# 2. Verifies the hinge adversarial behavior reached through the composite:
#    saturation once a sub-discriminator is fooled, the analytic zero at a
#    perfect separation, and the averaging over ensemble members
# 3. Verifies the generator composition, the validation composition, the
#    reported component panels, and the gradient path into the candidate
#    spectrogram
#
# Design decisions:
# - The hinge terms are exercised through the public composite rather than the
#   private hinge component, so the tests bind to the published surface while
#   still driving the clamp and the member averaging directly through logits
# - Member averaging is asserted by comparing a one-member ensemble against a
#   repeated-member ensemble, which must agree exactly under a mean
# - Saturation is asserted at a logit beyond the hinge point, where the clamp
#   must return exactly zero rather than a negative reward
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError

from vocode.losses.vocos import VocosLoss, VocosLossConfig


class HingeFixtureBuilder:
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


class VocosLossConfigurationTest(unittest.TestCase):
    # Verifies the frozen weight record behind the Vocos composition.
    def setUp(self) -> None:
        # Constructs the record from its defaults with no arguments, so the
        # cases below assert the reference weights the shipped configuration
        # actually applies rather than values restated at the call site.
        self._configuration: VocosLossConfig = VocosLossConfig()

    def test_default_weights_match_the_reference_recipe(self) -> None:
        # The multi-resolution ensemble enters at a tenth of the period ensemble.
        self.assertEqual(self._configuration.mel_weight, 45.0)
        self.assertEqual(self._configuration.multi_resolution_discriminator_weight, 0.1)

    def test_configuration_rejects_mutation_after_construction(self) -> None:
        # The record is frozen so a run cannot drift from its logged weights.
        with self.assertRaises(ValidationError):
            self._configuration.mel_weight: float = 1.0

    def test_configuration_rejects_an_unknown_weight(self) -> None:
        # An unknown field is a typo, never a silently ignored setting.
        with self.assertRaises(ValidationError):
            VocosLossConfig(feature_matching_weight=2.0)

    def test_configuration_rejects_a_non_positive_weight(self) -> None:
        # Every weight is strictly positive in the reference composition.
        with self.assertRaises(ValidationError):
            VocosLossConfig(multi_resolution_discriminator_weight=0.0)


class VocosGeneratorObjectiveTest(unittest.TestCase):
    # Verifies the generator composition, the hinge saturation, and the
    # averaging of the adversarial and feature-matching terms.
    def setUp(self) -> None:
        # Two members is what makes this family's member averaging observable
        # at all: the averaging case compares this ensemble against a
        # one-member one and requires the two to agree exactly, which a
        # summing reduction would fail. The configuration is retained as an
        # attribute rather than only handed to the loss, because the
        # composition cases read the resolution weight back to rebuild the
        # expected total from the reported components.
        self._builder: HingeFixtureBuilder = HingeFixtureBuilder(member_count=2, layer_count=2)
        self._configuration: VocosLossConfig = VocosLossConfig()
        self._loss: VocosLoss = VocosLoss(self._configuration)

    def test_generator_loss_is_zero_when_the_hinge_is_saturated_and_the_mel_matches(self) -> None:
        # Logits beyond the hinge point with matched features are the optimum.
        reference_mel: torch.Tensor = self._builder.seeded_mel(seed=81)
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_generator_loss(
            reference_mel,
            reference_mel,
            self._builder.logits(3.0),
            self._builder.logits(3.0),
            self._builder.feature_maps(0.5),
            self._builder.feature_maps(0.5),
            self._builder.feature_maps(0.5),
            self._builder.feature_maps(0.5)
        )
        self.assertAlmostEqual(
            float(total.item()),
            0.0,
            places=6,
            msg="The clamped hinge must saturate at zero once the discriminator is fooled"
        )
        self.assertAlmostEqual(components["generator_loss_period_adversarial"], 0.0, places=6)
        self.assertAlmostEqual(components["generator_loss_resolution_adversarial"], 0.0, places=6)

    def test_adversarial_term_matches_the_hinge_closed_form_at_zero_logits(self) -> None:
        # A zero logit costs the full hinge margin of one after averaging.
        components: dict[str, float]
        _, components = self._loss.compute_generator_loss(
            self._builder.mel(0.0),
            self._builder.mel(0.0),
            self._builder.logits(0.0),
            self._builder.logits(0.0),
            self._builder.feature_maps(0.0),
            self._builder.feature_maps(0.0),
            self._builder.feature_maps(0.0),
            self._builder.feature_maps(0.0)
        )
        self.assertAlmostEqual(components["generator_loss_period_adversarial"], 1.0, places=5)

    def test_adversarial_term_averages_over_ensemble_members(self) -> None:
        # A wider ensemble of identical members must not inflate the term.
        narrow_builder: HingeFixtureBuilder = HingeFixtureBuilder(member_count=1, layer_count=2)
        narrow_components: dict[str, float]
        _, narrow_components = self._loss.compute_generator_loss(
            narrow_builder.mel(0.0),
            narrow_builder.mel(0.0),
            narrow_builder.logits(0.25),
            narrow_builder.logits(0.25),
            narrow_builder.feature_maps(0.0),
            narrow_builder.feature_maps(1.0),
            narrow_builder.feature_maps(0.0),
            narrow_builder.feature_maps(1.0)
        )
        wide_components: dict[str, float]
        _, wide_components = self._loss.compute_generator_loss(
            self._builder.mel(0.0),
            self._builder.mel(0.0),
            self._builder.logits(0.25),
            self._builder.logits(0.25),
            self._builder.feature_maps(0.0),
            self._builder.feature_maps(1.0),
            self._builder.feature_maps(0.0),
            self._builder.feature_maps(1.0)
        )
        self.assertAlmostEqual(
            wide_components["generator_loss_period_adversarial"],
            narrow_components["generator_loss_period_adversarial"],
            places=5,
            msg="The hinge term is a mean over sub-discriminators, not a sum"
        )
        self.assertAlmostEqual(
            wide_components["generator_loss_feature_matching_period"],
            narrow_components["generator_loss_feature_matching_period"],
            places=5,
            msg="Feature matching is normalized by the ensemble size"
        )

    def test_generator_total_equals_the_weighted_sum_of_its_components(self) -> None:
        # The composition arithmetic must match the reported per-term values.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_generator_loss(
            self._builder.seeded_mel(seed=5),
            self._builder.seeded_mel(seed=6),
            self._builder.logits(0.25),
            self._builder.logits(-0.5),
            self._builder.feature_maps(1.0),
            self._builder.feature_maps(0.25),
            self._builder.feature_maps(1.0),
            self._builder.feature_maps(-0.5)
        )
        resolution_weight: float = self._configuration.multi_resolution_discriminator_weight
        expected: float = (
            components["generator_loss_period_adversarial"]
            + resolution_weight * components["generator_loss_resolution_adversarial"]
            + components["generator_loss_feature_matching_period"]
            + resolution_weight * components["generator_loss_feature_matching_resolution"]
            + self._configuration.mel_weight * components["generator_loss_mel"]
        )
        self.assertAlmostEqual(float(total.item()), expected, places=4)
        self.assertAlmostEqual(components["generator_loss_total"], float(total.item()), places=5)

    def test_generator_components_expose_the_reference_keys(self) -> None:
        # The logged component panel is part of the training contract.
        components: dict[str, float]
        _, components = self._loss.compute_generator_loss(
            self._builder.mel(0.0),
            self._builder.mel(1.0),
            self._builder.logits(0.0),
            self._builder.logits(0.0),
            self._builder.feature_maps(1.0),
            self._builder.feature_maps(0.0),
            self._builder.feature_maps(1.0),
            self._builder.feature_maps(0.0)
        )
        expected_keys: set[str] = {
            "generator_loss_total",
            "generator_loss_period_adversarial",
            "generator_loss_resolution_adversarial",
            "generator_loss_feature_matching_period",
            "generator_loss_feature_matching_resolution",
            "generator_loss_mel"
        }
        self.assertEqual(set(components), expected_keys)

    def test_generator_loss_propagates_gradient_to_the_candidate_mel(self) -> None:
        # The reconstruction term must reach the generator's spectrogram.
        candidate_mel: torch.Tensor = self._builder.seeded_mel(seed=91).requires_grad_(True)
        total: torch.Tensor
        total, _ = self._loss.compute_generator_loss(
            self._builder.seeded_mel(seed=92),
            candidate_mel,
            self._builder.logits(0.0),
            self._builder.logits(0.0),
            self._builder.feature_maps(1.0),
            self._builder.feature_maps(0.0),
            self._builder.feature_maps(1.0),
            self._builder.feature_maps(0.0)
        )
        total.backward()
        self.assertIsNotNone(candidate_mel.grad, msg="The candidate mel must receive gradient")
        self.assertTrue(torch.isfinite(candidate_mel.grad).all().item())


class VocosDiscriminatorObjectiveTest(unittest.TestCase):
    # Verifies the hinge separation term and the multi-resolution weighting.
    def setUp(self) -> None:
        # Two members is what makes this family's member averaging observable
        # at all: the averaging case compares this ensemble against a
        # one-member one and requires the two to agree exactly, which a
        # summing reduction would fail. The configuration is retained as an
        # attribute rather than only handed to the loss, because the
        # composition cases read the resolution weight back to rebuild the
        # expected total from the reported components.
        self._builder: HingeFixtureBuilder = HingeFixtureBuilder(member_count=2, layer_count=2)
        self._configuration: VocosLossConfig = VocosLossConfig()
        self._loss: VocosLoss = VocosLoss(self._configuration)

    def test_discriminator_loss_is_zero_beyond_both_hinge_margins(self) -> None:
        # Real logits above one and fake logits below minus one are the optimum.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_discriminator_loss(
            real_period_logits=self._builder.logits(2.0),
            fake_period_logits=self._builder.logits(-2.0),
            real_resolution_logits=self._builder.logits(2.0),
            fake_resolution_logits=self._builder.logits(-2.0)
        )
        self.assertAlmostEqual(float(total.item()), 0.0, places=6)
        self.assertAlmostEqual(components["discriminator_loss_period"], 0.0, places=6)
        self.assertAlmostEqual(components["discriminator_loss_resolution"], 0.0, places=6)

    def test_discriminator_total_weights_the_resolution_ensemble(self) -> None:
        # The multi-resolution ensemble enters at its configured weight.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_discriminator_loss(
            real_period_logits=self._builder.logits(0.0),
            fake_period_logits=self._builder.logits(0.0),
            real_resolution_logits=self._builder.logits(0.5),
            fake_resolution_logits=self._builder.logits(0.5)
        )
        expected: float = (
            components["discriminator_loss_period"]
            + self._configuration.multi_resolution_discriminator_weight
            * components["discriminator_loss_resolution"]
        )
        self.assertAlmostEqual(float(total.item()), expected, places=5)
        self.assertGreater(float(total.item()), 0.0)

    def test_discriminator_components_expose_the_reference_keys(self) -> None:
        # The logged component panel is part of the training contract.
        components: dict[str, float]
        _, components = self._loss.compute_discriminator_loss(
            real_period_logits=self._builder.logits(1.0),
            fake_period_logits=self._builder.logits(-1.0),
            real_resolution_logits=self._builder.logits(1.0),
            fake_resolution_logits=self._builder.logits(-1.0)
        )
        expected_keys: set[str] = {
            "discriminator_loss_total",
            "discriminator_loss_period",
            "discriminator_loss_resolution"
        }
        self.assertEqual(set(components), expected_keys)

    def test_discriminator_loss_rejects_a_member_count_mismatch(self) -> None:
        # The real and fake ensembles must be paired one to one.
        narrow_builder: HingeFixtureBuilder = HingeFixtureBuilder(member_count=1, layer_count=2)
        with self.assertRaisesRegex(ValueError, "zip"):
            self._loss.compute_discriminator_loss(
                real_period_logits=self._builder.logits(1.0),
                fake_period_logits=narrow_builder.logits(-1.0),
                real_resolution_logits=self._builder.logits(1.0),
                fake_resolution_logits=self._builder.logits(-1.0)
            )

    def test_discriminator_loss_propagates_gradient_to_the_fake_logits(self) -> None:
        # The separation term must reach the ensemble logits under optimization.
        fake_period_logits: list[torch.Tensor] = self._builder.logits_requiring_gradient(0.0)
        total: torch.Tensor
        total, _ = self._loss.compute_discriminator_loss(
            real_period_logits=self._builder.logits(0.0),
            fake_period_logits=fake_period_logits,
            real_resolution_logits=self._builder.logits(0.0),
            fake_resolution_logits=self._builder.logits(0.0)
        )
        total.backward()
        member: torch.Tensor
        for member in fake_period_logits:
            self.assertIsNotNone(member.grad, msg="Fake period logits must receive gradient")
            self.assertTrue(torch.isfinite(member.grad).all().item())


class VocosValidationObjectiveTest(unittest.TestCase):
    # Verifies the adversarial-free validation composition.
    def setUp(self) -> None:
        # Two members is what makes this family's member averaging observable
        # at all: the averaging case compares this ensemble against a
        # one-member one and requires the two to agree exactly, which a
        # summing reduction would fail. The configuration is retained as an
        # attribute rather than only handed to the loss, because the
        # composition cases read the resolution weight back to rebuild the
        # expected total from the reported components.
        self._builder: HingeFixtureBuilder = HingeFixtureBuilder(member_count=2, layer_count=2)
        self._configuration: VocosLossConfig = VocosLossConfig()
        self._loss: VocosLoss = VocosLoss(self._configuration)

    def test_validation_loss_is_the_weighted_mel_distance(self) -> None:
        # A unit mel offset costs exactly the configured mel weight.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_validation_loss(
            self._builder.mel(0.0),
            self._builder.mel(1.0)
        )
        self.assertAlmostEqual(components["validation_loss_mel"], 1.0, places=5)
        self.assertAlmostEqual(float(total.item()), self._configuration.mel_weight, places=4)

    def test_validation_loss_is_zero_for_identical_spectrograms(self) -> None:
        # Perfect reconstruction is the analytic optimum of the validation term.
        reference_mel: torch.Tensor = self._builder.seeded_mel(seed=101)
        total: torch.Tensor
        total, _ = self._loss.compute_validation_loss(reference_mel, reference_mel)
        self.assertAlmostEqual(float(total.item()), 0.0, places=6)

    def test_validation_components_expose_the_reference_keys(self) -> None:
        # The validation panel is distinct from the training panel.
        components: dict[str, float]
        _, components = self._loss.compute_validation_loss(
            self._builder.mel(0.0),
            self._builder.mel(0.5)
        )
        self.assertEqual(set(components), {"validation_loss_total", "validation_loss_mel"})

    def test_validation_loss_rejects_a_mel_shape_mismatch(self) -> None:
        # The mel guard fails closed on the validation path as well.
        with self.assertRaisesRegex(ValueError, "identical shape"):
            self._loss.compute_validation_loss(torch.zeros(1, 8, 12), torch.zeros(1, 8, 10))
