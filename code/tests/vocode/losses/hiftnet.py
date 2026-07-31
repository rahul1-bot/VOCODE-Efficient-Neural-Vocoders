# This module:
# 1. Verifies the HiFTNet loss configuration: the reference mel and
#    feature-matching weights, the truncation threshold, the frozen record,
#    and the rejection of unknown or non-positive weights
# 2. Verifies the top-k relative least-squares term reached through both
#    composites: it vanishes when no real logit falls below the shifted fake
#    logit, and it saturates at the configured threshold per ensemble member
# 3. Verifies the generator and discriminator compositions, their analytic
#    zeros, the weighted sums of their reported components, and the gradient path
# 4. Verifies the adversarial-free validation composition
#
# Design decisions:
# - The relative term is exercised through the public composites rather than
#   the private criterion, so the tests bind to the published surface while
#   still driving the mask, the median shift, and the threshold directly
# - Saturation is asserted against the threshold times the ensemble size, which
#   is the exact ceiling the truncation imposes and fails if the clamp is lost
# - Constant logits make the median shift analytically predictable, so the mask
#   branch under test is selected deliberately rather than by chance
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError

from vocode.losses.hiftnet import HiftnetLoss, HiftnetLossConfig


class LogitAndFeatureBuilder:
    # Builds the logit, feature-map, and mel fixtures the composites consume.
    def __init__(self, member_count: int, layer_count: int) -> None:
        # Binds the constructor inputs into this component's state.
        self._member_count: int = member_count
        self._layer_count: int = layer_count
        self._logit_shape: tuple[int, int] = (1, 5)
        self._feature_shape: tuple[int, int, int] = (1, 2, 3)
        self._mel_shape: tuple[int, int, int] = (1, 6, 8)

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

    def ramp_logits(self, low: float, high: float) -> list[torch.Tensor]:
        # Builds a spread logit tensor per member so the truncation mask fires.
        ensemble: list[torch.Tensor] = [
            torch.linspace(low, high, self._logit_shape[1]).reshape(self._logit_shape)
            for member_index in range(self._member_count)
        ]
        return ensemble

    def seeded_logits(self, seed: int) -> list[torch.Tensor]:
        # Builds a seeded random logit tensor per ensemble member.
        torch.manual_seed(seed)
        ensemble: list[torch.Tensor] = [
            torch.randn(self._logit_shape) for member_index in range(self._member_count)
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

    @property
    def member_count(self) -> int:
        # Returns the number of sub-discriminators in the ensemble.
        return self._member_count


class HiftnetLossConfigurationTest(unittest.TestCase):
    # Verifies the frozen weight record behind the HiFTNet composition.
    def setUp(self) -> None:
        # Constructs the record from its defaults with no arguments, so the
        # cases below assert the reference weights and the truncation
        # threshold the shipped configuration actually applies.
        self._configuration: HiftnetLossConfig = HiftnetLossConfig()

    def test_default_weights_match_the_reference_recipe(self) -> None:
        # The source-filter recipe keeps the HiFi-GAN mel and matching weights.
        self.assertEqual(self._configuration.mel_weight, 45.0)
        self.assertEqual(self._configuration.feature_matching_weight, 2.0)
        self.assertEqual(self._configuration.tprls_tau, 0.04)

    def test_configuration_rejects_mutation_after_construction(self) -> None:
        # The record is frozen so a run cannot drift from its logged weights.
        with self.assertRaises(ValidationError):
            self._configuration.tprls_tau: float = 1.0

    def test_configuration_rejects_an_unknown_weight(self) -> None:
        # An unknown field is a typo, never a silently ignored setting.
        with self.assertRaises(ValidationError):
            HiftnetLossConfig(adversarial_weight=1.0)

    def test_configuration_rejects_a_non_positive_threshold(self) -> None:
        # The truncation threshold is strictly positive.
        with self.assertRaises(ValidationError):
            HiftnetLossConfig(tprls_tau=0.0)


class HiftnetRelativeLeastSquaresTest(unittest.TestCase):
    # Verifies the truncated relative term: its empty-mask boundary, its
    # threshold ceiling, and its dependence on the configured threshold.
    def setUp(self) -> None:
        # The configuration is retained because these cases compare the term
        # against the configured threshold directly: the ceiling is what a
        # saturated member pays, and a member with an empty selection mask
        # must pay nothing at all, so both boundaries are stated in terms of
        # that one value rather than a literal.
        self._builder: LogitAndFeatureBuilder = LogitAndFeatureBuilder(member_count=2, layer_count=2)
        self._configuration: HiftnetLossConfig = HiftnetLossConfig()
        self._loss: HiftnetLoss = HiftnetLoss(self._configuration)

    def test_relative_term_vanishes_for_identical_logit_sets(self) -> None:
        # With a zero median shift no real logit falls below its fake partner.
        identical: list[torch.Tensor] = self._builder.seeded_logits(seed=601)
        components: dict[str, float]
        _, components = self._loss.compute_discriminator_loss(
            real_period_logits=identical,
            fake_period_logits=identical,
            real_spectrogram_logits=identical,
            fake_spectrogram_logits=identical
        )
        self.assertAlmostEqual(components["discriminator_loss_period_tprls"], 0.0, places=6)
        self.assertAlmostEqual(components["discriminator_loss_spectrogram_tprls"], 0.0, places=6)

    def test_relative_term_saturates_at_the_threshold_for_each_member(self) -> None:
        # A widely separated pair drives every member to the truncation ceiling.
        real_logits: list[torch.Tensor] = self._builder.logits(40.0)
        fake_logits: list[torch.Tensor] = self._builder.ramp_logits(-1.0, 1.0)
        components: dict[str, float]
        _, components = self._loss.compute_discriminator_loss(
            real_period_logits=real_logits,
            fake_period_logits=fake_logits,
            real_spectrogram_logits=real_logits,
            fake_spectrogram_logits=fake_logits
        )
        expected: float = self._configuration.tprls_tau * float(self._builder.member_count)
        self.assertAlmostEqual(components["discriminator_loss_period_tprls"], expected, places=5)

    def test_relative_term_never_exceeds_its_ceiling_on_random_logits(self) -> None:
        # The truncation bounds the term regardless of the logit distribution.
        components: dict[str, float]
        _, components = self._loss.compute_discriminator_loss(
            real_period_logits=self._builder.seeded_logits(seed=611),
            fake_period_logits=self._builder.seeded_logits(seed=612),
            real_spectrogram_logits=self._builder.seeded_logits(seed=613),
            fake_spectrogram_logits=self._builder.seeded_logits(seed=614)
        )
        ceiling: float = self._configuration.tprls_tau * float(self._builder.member_count)
        self.assertLessEqual(components["discriminator_loss_period_tprls"], ceiling + 1e-6)
        self.assertLessEqual(components["discriminator_loss_spectrogram_tprls"], ceiling + 1e-6)

    def test_a_larger_threshold_raises_the_saturated_term(self) -> None:
        # The ceiling is exactly the configured threshold, not a fixed constant.
        real_logits: list[torch.Tensor] = self._builder.logits(40.0)
        fake_logits: list[torch.Tensor] = self._builder.ramp_logits(-1.0, 1.0)
        wide_loss: HiftnetLoss = HiftnetLoss(HiftnetLossConfig(tprls_tau=0.2))
        components: dict[str, float]
        _, components = wide_loss.compute_discriminator_loss(
            real_period_logits=real_logits,
            fake_period_logits=fake_logits,
            real_spectrogram_logits=real_logits,
            fake_spectrogram_logits=fake_logits
        )
        expected: float = 0.2 * float(self._builder.member_count)
        self.assertAlmostEqual(components["discriminator_loss_period_tprls"], expected, places=5)

    def test_relative_term_rejects_a_member_count_mismatch(self) -> None:
        # The real and fake ensembles must be paired one to one.
        narrow_builder: LogitAndFeatureBuilder = LogitAndFeatureBuilder(member_count=1, layer_count=2)
        with self.assertRaisesRegex(ValueError, "zip"):
            self._loss.compute_discriminator_loss(
                real_period_logits=self._builder.logits(1.0),
                fake_period_logits=narrow_builder.logits(0.0),
                real_spectrogram_logits=self._builder.logits(1.0),
                fake_spectrogram_logits=self._builder.logits(0.0)
            )


class HiftnetDiscriminatorObjectiveTest(unittest.TestCase):
    # Verifies the discriminator composition over both ensembles.
    def setUp(self) -> None:
        # Keeps no reference to the configuration, because the discriminator
        # path applies no configured weight at all: both ensembles and both
        # adversarial forms enter unweighted, which is itself part of what
        # these cases establish.
        self._builder: LogitAndFeatureBuilder = LogitAndFeatureBuilder(member_count=2, layer_count=2)
        self._loss: HiftnetLoss = HiftnetLoss(HiftnetLossConfig())

    def test_discriminator_loss_is_zero_for_a_perfect_separation(self) -> None:
        # Real logits at one and fake logits at zero zero both term families.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_discriminator_loss(
            real_period_logits=self._builder.logits(1.0),
            fake_period_logits=self._builder.logits(0.0),
            real_spectrogram_logits=self._builder.logits(1.0),
            fake_spectrogram_logits=self._builder.logits(0.0)
        )
        self.assertAlmostEqual(float(total.item()), 0.0, places=6)
        self.assertAlmostEqual(components["discriminator_loss_period_lsgan"], 0.0, places=6)
        self.assertAlmostEqual(components["discriminator_loss_period_tprls"], 0.0, places=6)

    def test_discriminator_total_is_the_sum_of_its_four_terms(self) -> None:
        # Both ensembles contribute a least-squares and a relative term.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_discriminator_loss(
            real_period_logits=self._builder.logits(40.0),
            fake_period_logits=self._builder.ramp_logits(-1.0, 1.0),
            real_spectrogram_logits=self._builder.seeded_logits(seed=621),
            fake_spectrogram_logits=self._builder.seeded_logits(seed=622)
        )
        expected: float = (
            components["discriminator_loss_period_lsgan"]
            + components["discriminator_loss_spectrogram_lsgan"]
            + components["discriminator_loss_period_tprls"]
            + components["discriminator_loss_spectrogram_tprls"]
        )
        self.assertAlmostEqual(float(total.item()), expected, places=3)

    def test_discriminator_components_expose_the_reference_keys(self) -> None:
        # The logged component panel is part of the training contract.
        components: dict[str, float]
        _, components = self._loss.compute_discriminator_loss(
            real_period_logits=self._builder.logits(1.0),
            fake_period_logits=self._builder.logits(0.5),
            real_spectrogram_logits=self._builder.logits(1.0),
            fake_spectrogram_logits=self._builder.logits(0.5)
        )
        expected_keys: set[str] = {
            "discriminator_loss_total",
            "discriminator_loss_period_lsgan",
            "discriminator_loss_spectrogram_lsgan",
            "discriminator_loss_period_tprls",
            "discriminator_loss_spectrogram_tprls"
        }
        self.assertEqual(set(components), expected_keys)

    def test_discriminator_loss_propagates_gradient_to_the_fake_logits(self) -> None:
        # The separation term must reach the ensemble logits under optimization.
        fake_period_logits: list[torch.Tensor] = self._builder.logits_requiring_gradient(0.5)
        total: torch.Tensor
        total, _ = self._loss.compute_discriminator_loss(
            real_period_logits=self._builder.logits(1.0),
            fake_period_logits=fake_period_logits,
            real_spectrogram_logits=self._builder.logits(1.0),
            fake_spectrogram_logits=self._builder.logits(0.5)
        )
        total.backward()
        member: torch.Tensor
        for member in fake_period_logits:
            self.assertIsNotNone(member.grad, msg="Fake period logits must receive gradient")
            self.assertTrue(torch.isfinite(member.grad).all().item())


class HiftnetGeneratorObjectiveTest(unittest.TestCase):
    # Verifies the generator composition, its optimum, and its weighting.
    def setUp(self) -> None:
        # The configuration is retained because the composition cases read
        # the feature-matching and mel weights back to rebuild the expected
        # total from the reported components, rather than restating the
        # recipe's numbers alongside the code that applies them.
        self._builder: LogitAndFeatureBuilder = LogitAndFeatureBuilder(member_count=2, layer_count=2)
        self._configuration: HiftnetLossConfig = HiftnetLossConfig()
        self._loss: HiftnetLoss = HiftnetLoss(self._configuration)

    def test_generator_loss_is_zero_at_perfect_reconstruction_and_full_fooling(self) -> None:
        # Unit logits on both sides with a matched mel is the analytic optimum.
        reference_mel: torch.Tensor = self._builder.seeded_mel(seed=631)
        unit_logits: list[torch.Tensor] = self._builder.logits(1.0)
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_generator_loss(
            reference_mel=reference_mel,
            candidate_mel=reference_mel,
            real_period_logits=unit_logits,
            fake_period_logits=unit_logits,
            real_spectrogram_logits=unit_logits,
            fake_spectrogram_logits=unit_logits,
            real_period_features=self._builder.feature_maps(0.5),
            fake_period_features=self._builder.feature_maps(0.5),
            real_spectrogram_features=self._builder.feature_maps(0.5),
            fake_spectrogram_features=self._builder.feature_maps(0.5)
        )
        self.assertAlmostEqual(
            float(total.item()),
            0.0,
            places=6,
            msg="A perfect generator must pay nothing under the HiFTNet composition"
        )
        self.assertAlmostEqual(components["generator_loss_mel"], 0.0, places=6)
        self.assertAlmostEqual(components["generator_loss_period_tprls"], 0.0, places=6)

    def test_generator_total_equals_the_weighted_sum_of_its_components(self) -> None:
        # The composition arithmetic must match the reported per-term values.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_generator_loss(
            reference_mel=self._builder.seeded_mel(seed=641),
            candidate_mel=self._builder.seeded_mel(seed=642),
            real_period_logits=self._builder.logits(40.0),
            fake_period_logits=self._builder.ramp_logits(-1.0, 1.0),
            real_spectrogram_logits=self._builder.seeded_logits(seed=643),
            fake_spectrogram_logits=self._builder.seeded_logits(seed=644),
            real_period_features=self._builder.feature_maps(1.0),
            fake_period_features=self._builder.feature_maps(0.0),
            real_spectrogram_features=self._builder.feature_maps(1.0),
            fake_spectrogram_features=self._builder.feature_maps(0.25)
        )
        expected: float = (
            components["generator_loss_period_lsgan"]
            + components["generator_loss_spectrogram_lsgan"]
            + components["generator_loss_period_tprls"]
            + components["generator_loss_spectrogram_tprls"]
            + self._configuration.feature_matching_weight * (
                components["generator_loss_feature_matching_period"]
                + components["generator_loss_feature_matching_spectrogram"]
            )
            + self._configuration.mel_weight * components["generator_loss_mel"]
        )
        self.assertAlmostEqual(float(total.item()), expected, places=3)
        self.assertAlmostEqual(components["generator_loss_total"], float(total.item()), places=4)

    def test_generator_components_expose_the_reference_keys(self) -> None:
        # The logged component panel is part of the training contract.
        components: dict[str, float]
        _, components = self._loss.compute_generator_loss(
            reference_mel=self._builder.mel(0.0),
            candidate_mel=self._builder.mel(1.0),
            real_period_logits=self._builder.logits(1.0),
            fake_period_logits=self._builder.logits(0.0),
            real_spectrogram_logits=self._builder.logits(1.0),
            fake_spectrogram_logits=self._builder.logits(0.0),
            real_period_features=self._builder.feature_maps(1.0),
            fake_period_features=self._builder.feature_maps(0.0),
            real_spectrogram_features=self._builder.feature_maps(1.0),
            fake_spectrogram_features=self._builder.feature_maps(0.0)
        )
        expected_keys: set[str] = {
            "generator_loss_total",
            "generator_loss_period_lsgan",
            "generator_loss_spectrogram_lsgan",
            "generator_loss_period_tprls",
            "generator_loss_spectrogram_tprls",
            "generator_loss_feature_matching_period",
            "generator_loss_feature_matching_spectrogram",
            "generator_loss_mel"
        }
        self.assertEqual(set(components), expected_keys)

    def test_generator_loss_propagates_gradient_to_the_candidate_mel(self) -> None:
        # The reconstruction term must reach the generator's spectrogram.
        candidate_mel: torch.Tensor = self._builder.seeded_mel(seed=651).requires_grad_(True)
        total: torch.Tensor
        total, _ = self._loss.compute_generator_loss(
            reference_mel=self._builder.seeded_mel(seed=652),
            candidate_mel=candidate_mel,
            real_period_logits=self._builder.logits(1.0),
            fake_period_logits=self._builder.logits(0.0),
            real_spectrogram_logits=self._builder.logits(1.0),
            fake_spectrogram_logits=self._builder.logits(0.0),
            real_period_features=self._builder.feature_maps(1.0),
            fake_period_features=self._builder.feature_maps(0.0),
            real_spectrogram_features=self._builder.feature_maps(1.0),
            fake_spectrogram_features=self._builder.feature_maps(0.0)
        )
        total.backward()
        self.assertIsNotNone(candidate_mel.grad, msg="The candidate mel must receive gradient")
        self.assertTrue(torch.isfinite(candidate_mel.grad).all().item())

    def test_generator_loss_rejects_a_mel_shape_mismatch(self) -> None:
        # The mel guard fails closed rather than cropping inside the composite.
        with self.assertRaisesRegex(ValueError, "identical shape"):
            self._loss.compute_generator_loss(
                reference_mel=torch.zeros(1, 6, 8),
                candidate_mel=torch.zeros(1, 6, 7),
                real_period_logits=self._builder.logits(1.0),
                fake_period_logits=self._builder.logits(0.0),
                real_spectrogram_logits=self._builder.logits(1.0),
                fake_spectrogram_logits=self._builder.logits(0.0),
                real_period_features=self._builder.feature_maps(0.0),
                fake_period_features=self._builder.feature_maps(0.0),
                real_spectrogram_features=self._builder.feature_maps(0.0),
                fake_spectrogram_features=self._builder.feature_maps(0.0)
            )


class HiftnetValidationObjectiveTest(unittest.TestCase):
    # Verifies the adversarial-free validation composition.
    def setUp(self) -> None:
        # A single-member ensemble is enough because the validation path
        # evaluates no discriminator; the builder is retained only for its
        # mel fixtures, and the configuration for the mel weight the
        # validation total is expected to equal.
        self._builder: LogitAndFeatureBuilder = LogitAndFeatureBuilder(member_count=1, layer_count=1)
        self._configuration: HiftnetLossConfig = HiftnetLossConfig()
        self._loss: HiftnetLoss = HiftnetLoss(self._configuration)

    def test_validation_loss_is_the_weighted_mel_distance(self) -> None:
        # A unit mel offset costs exactly the configured mel weight.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_validation_loss(
            reference_mel=self._builder.mel(0.0),
            candidate_mel=self._builder.mel(1.0)
        )
        self.assertAlmostEqual(components["validation_loss_mel"], 1.0, places=5)
        self.assertAlmostEqual(float(total.item()), self._configuration.mel_weight, places=4)

    def test_validation_loss_is_zero_for_identical_spectrograms(self) -> None:
        # Perfect reconstruction is the analytic optimum of the validation term.
        reference_mel: torch.Tensor = self._builder.seeded_mel(seed=661)
        total: torch.Tensor
        total, _ = self._loss.compute_validation_loss(
            reference_mel=reference_mel,
            candidate_mel=reference_mel
        )
        self.assertAlmostEqual(float(total.item()), 0.0, places=6)

    def test_validation_components_expose_the_reference_keys(self) -> None:
        # The validation panel is distinct from the training panel.
        components: dict[str, float]
        _, components = self._loss.compute_validation_loss(
            reference_mel=self._builder.mel(0.0),
            candidate_mel=self._builder.mel(0.5)
        )
        self.assertEqual(set(components), {"validation_loss_total", "validation_loss_mel"})
