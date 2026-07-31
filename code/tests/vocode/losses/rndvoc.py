# This module:
# 1. Verifies the RNDVoC loss configuration: the reference weights, the frozen
#    record, the rejection of unknown or non-positive weights, and the
#    spectrum analyzer type shared with the APNet2 family
# 2. Verifies the omni-directional phase loss: its zero at a matching phase,
#    its invariance to a full-turn offset across the nine-neighbor stencil, its
#    penalty on a partial turn, and its gradient path
# 3. Verifies the generator composition: the analytic zero at perfect
#    reconstruction with full fooling, the weighted sum of the reported
#    components, and the gradient path into the candidate views
# 4. Verifies the hinge discriminator composition and the validation
#    composition, including the key remapping between the two panels
#
# Design decisions:
# - The phase invariance is asserted on synthetic phase planes so the equality
#   is exact, and it is asserted on the loss rather than on the stencil, which
#   is where a lost anti-wrap would actually cost training
# - The validation total is additionally cross-checked against the generator
#   total minus its adversarial and feature-matching contributions, which pins
#   the shared reconstruction path both calls delegate to
# - Fixtures are synthetic spectra rather than analyzer output, because the
#   composition is independent of how the spectra were produced
#
# Author: Rahul Sawhney

import math
import unittest

import torch
from pydantic import ValidationError

from vocode.losses.apnet2 import Apnet2Spectrum, Apnet2SpectrumAnalyzer
from vocode.losses.rndvoc import RndvocLoss, RndvocLossConfig, RndvocOmniPhaseLoss


class RndvocFixtureBuilder:
    # Builds the spectra, logits, feature maps, and mels the composite consumes.
    def __init__(self, member_count: int, layer_count: int) -> None:
        # Binds the constructor inputs into this component's state.
        self._member_count: int = member_count
        self._layer_count: int = layer_count
        self._bin_count: int = 9
        self._frame_count: int = 7
        self._logit_shape: tuple[int, int] = (1, 4)
        self._feature_shape: tuple[int, int, int] = (1, 2, 3)
        self._mel_shape: tuple[int, int, int] = (1, 6, 7)

    def spectrum(self, seed: int) -> Apnet2Spectrum:
        # Builds a seeded synthetic spectrum record of the configured shape.
        torch.manual_seed(seed)
        return Apnet2Spectrum(
            log_amplitude=torch.randn(1, self._bin_count, self._frame_count),
            phase=torch.randn(1, self._bin_count, self._frame_count),
            real_spectrum=torch.randn(1, self._bin_count, self._frame_count),
            imaginary_spectrum=torch.randn(1, self._bin_count, self._frame_count)
        )

    def spectral_plane(self, seed: int) -> torch.Tensor:
        # Builds one seeded spectral plane matching the record shape.
        torch.manual_seed(seed)
        return torch.randn(1, self._bin_count, self._frame_count)

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


class RndvocLossConfigurationTest(unittest.TestCase):
    # Verifies the frozen weight record behind the RNDVoC composition.
    def setUp(self) -> None:
        # Constructs the record from its defaults with no arguments, so the
        # cases below assert the eight reference weights the shipped
        # configuration actually applies rather than values restated at the
        # call site.
        self._configuration: RndvocLossConfig = RndvocLossConfig()

    def test_default_weights_match_the_reference_recipe(self) -> None:
        # The phase term dominates, and both ensembles enter unweighted.
        self.assertEqual(self._configuration.amplitude_weight, 45.0)
        self.assertEqual(self._configuration.phase_weight, 100.0)
        self.assertEqual(self._configuration.consistency_weight, 45.0)
        self.assertEqual(self._configuration.real_imaginary_weight, 45.0)
        self.assertEqual(self._configuration.adversarial_weight, 1.0)
        self.assertEqual(self._configuration.feature_matching_weight, 1.0)
        self.assertEqual(self._configuration.mel_weight, 45.0)
        self.assertEqual(self._configuration.resolution_discriminator_weight, 1.0)

    def test_configuration_rejects_mutation_after_construction(self) -> None:
        # The record is frozen so a run cannot drift from its logged weights.
        with self.assertRaises(ValidationError):
            self._configuration.phase_weight: float = 1.0

    def test_configuration_rejects_an_unknown_weight(self) -> None:
        # An unknown field is a typo, never a silently ignored setting.
        with self.assertRaises(ValidationError):
            RndvocLossConfig(spectrum_weight=20.0)

    def test_configuration_rejects_a_non_positive_weight(self) -> None:
        # Every weight is strictly positive in the reference composition.
        with self.assertRaises(ValidationError):
            RndvocLossConfig(consistency_weight=0.0)

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The composite exposes the exact record it was constructed with.
        configuration: RndvocLossConfig = RndvocLossConfig(mel_weight=1.0)
        loss: RndvocLoss = RndvocLoss(configuration)
        self.assertIs(loss.configuration, configuration)

    def test_spectrum_analyzer_type_is_shared_with_the_apnet2_family(self) -> None:
        # RNDVoC consumes the same STFT views as the APNet2 lineage.
        loss: RndvocLoss = RndvocLoss(RndvocLossConfig())
        self.assertIs(loss.spectrum_analyzer_type, Apnet2SpectrumAnalyzer)


class RndvocOmniPhaseLossTest(unittest.TestCase):
    # Verifies the nine-neighbor anti-wrapped phase criterion.
    def setUp(self) -> None:
        # Nine bins by seven frames gives every kernel direction interior
        # positions to act on, so the eight neighbour comparisons are
        # exercised away from the padded border rather than only against it.
        # The two surfaces are drawn independently so they genuinely differ,
        # which is what lets a full-turn offset be shown to cost nothing
        # while the underlying disagreement still does.
        self._loss: RndvocOmniPhaseLoss = RndvocOmniPhaseLoss()
        torch.manual_seed(801)
        self._reference_phase: torch.Tensor = torch.randn(1, 9, 7)
        self._candidate_phase: torch.Tensor = torch.randn(1, 9, 7)

    def test_loss_is_zero_for_a_matching_phase(self) -> None:
        # Every neighborhood difference agrees, so nothing is charged.
        value: torch.Tensor = self._loss(self._reference_phase, self._reference_phase)
        self.assertAlmostEqual(
            float(value.item()),
            0.0,
            places=6,
            msg="A matching phase plane must carry no cost"
        )

    def test_full_turn_offset_leaves_the_loss_unchanged(self) -> None:
        # The anti-wrap absorbs a whole-turn offset across every stencil tap.
        plain: torch.Tensor = self._loss(self._reference_phase, self._candidate_phase)
        shifted: torch.Tensor = self._loss(
            self._reference_phase,
            self._candidate_phase + 2.0 * math.pi
        )
        self.assertAlmostEqual(float(plain.item()), float(shifted.item()), places=5)

    def test_double_turn_offset_leaves_the_loss_unchanged(self) -> None:
        # The projection absorbs any integer number of turns, not only one.
        plain: torch.Tensor = self._loss(self._reference_phase, self._candidate_phase)
        shifted: torch.Tensor = self._loss(
            self._reference_phase,
            self._candidate_phase - 4.0 * math.pi
        )
        self.assertAlmostEqual(float(plain.item()), float(shifted.item()), places=5)

    def test_partial_turn_offset_is_penalized(self) -> None:
        # A half-turn offset is a genuine phase error and must be charged.
        matched: torch.Tensor = self._loss(self._reference_phase, self._reference_phase)
        shifted: torch.Tensor = self._loss(
            self._reference_phase,
            self._reference_phase + math.pi
        )
        self.assertGreater(float(shifted.item()), float(matched.item()))

    def test_loss_returns_a_finite_non_negative_scalar(self) -> None:
        # The stencil reduction collapses the plane pair to one number.
        value: torch.Tensor = self._loss(self._reference_phase, self._candidate_phase)
        self.assertEqual(value.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(value).item())
        self.assertGreater(float(value.item()), 0.0)

    def test_over_long_candidate_planes_are_cropped_to_the_reference(self) -> None:
        # The paired crop tolerates a trailing frame instead of failing.
        extended_candidate: torch.Tensor = torch.cat(
            [self._candidate_phase, self._candidate_phase[..., :1]],
            dim=-1
        )
        plain: torch.Tensor = self._loss(self._reference_phase, self._candidate_phase)
        extended: torch.Tensor = self._loss(self._reference_phase, extended_candidate)
        self.assertAlmostEqual(float(plain.item()), float(extended.item()), places=5)

    def test_loss_propagates_gradient_to_the_candidate_phase(self) -> None:
        # The criterion must train the predicted phase plane.
        candidate_phase: torch.Tensor = self._candidate_phase.clone().requires_grad_(True)
        value: torch.Tensor = self._loss(self._reference_phase, candidate_phase)
        value.backward()
        self.assertIsNotNone(candidate_phase.grad, msg="The candidate phase must receive gradient")
        self.assertTrue(torch.isfinite(candidate_phase.grad).all().item())


class RndvocGeneratorObjectiveTest(unittest.TestCase):
    # Verifies the generator composition over the reconstruction and
    # adversarial terms.
    def setUp(self) -> None:
        # The reference spectrum is seeded so each supervised domain receives
        # a non-degenerate target identical across runs. The configuration is
        # retained because the composition cases read its weights back to
        # rebuild the expected total, which matters more here than in the
        # sibling families: this recipe applies the resolution weight inside
        # the adversarial group and the outer weights on top of it, so the
        # arithmetic is two-staged.
        self._builder: RndvocFixtureBuilder = RndvocFixtureBuilder(member_count=2, layer_count=2)
        self._configuration: RndvocLossConfig = RndvocLossConfig()
        self._loss: RndvocLoss = RndvocLoss(self._configuration)
        self._spectrum: Apnet2Spectrum = self._builder.spectrum(seed=811)

    def test_generator_loss_is_zero_at_perfect_reconstruction_and_full_fooling(self) -> None:
        # Matching every spectral view with saturated logits is the optimum.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_generator_loss(
            reference_spectrum=self._spectrum,
            candidate_log_amplitude=self._spectrum.log_amplitude,
            candidate_phase=self._spectrum.phase,
            candidate_real_spectrum=self._spectrum.real_spectrum,
            candidate_imaginary_spectrum=self._spectrum.imaginary_spectrum,
            final_spectrum=self._spectrum,
            reference_mel=self._builder.mel(0.0),
            candidate_mel=self._builder.mel(0.0),
            fake_period_logits=self._builder.logits(2.0),
            fake_resolution_logits=self._builder.logits(2.0),
            real_period_features=self._builder.feature_maps(0.5),
            fake_period_features=self._builder.feature_maps(0.5),
            real_resolution_features=self._builder.feature_maps(0.5),
            fake_resolution_features=self._builder.feature_maps(0.5)
        )
        self.assertAlmostEqual(
            float(total.item()),
            0.0,
            places=5,
            msg="A perfect generator must pay nothing under the RNDVoC composition"
        )
        self.assertAlmostEqual(components["generator_loss_phase"], 0.0, places=6)
        self.assertAlmostEqual(components["generator_loss_consistency"], 0.0, places=6)
        self.assertAlmostEqual(components["generator_loss_adversarial"], 0.0, places=6)

    def test_generator_total_equals_the_weighted_sum_of_its_components(self) -> None:
        # The composition arithmetic must match the reported per-term values.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._compute_generator()
        expected: float = (
            self._configuration.amplitude_weight * components["generator_loss_amplitude"]
            + self._configuration.phase_weight * components["generator_loss_phase"]
            + self._configuration.consistency_weight * components["generator_loss_consistency"]
            + self._configuration.real_imaginary_weight * components["generator_loss_real_imaginary"]
            + self._configuration.mel_weight * components["generator_loss_mel"]
            + self._configuration.adversarial_weight * components["generator_loss_adversarial"]
            + self._configuration.feature_matching_weight * components["generator_loss_feature_matching"]
        )
        self.assertAlmostEqual(float(total.item()), expected, places=3)
        self.assertAlmostEqual(components["generator_loss_total"], float(total.item()), places=4)

    def test_generator_components_expose_the_reference_keys(self) -> None:
        # The logged component panel is part of the training contract.
        components: dict[str, float]
        _, components = self._compute_generator()
        expected_keys: set[str] = {
            "generator_loss_total",
            "generator_loss_amplitude",
            "generator_loss_phase",
            "generator_loss_consistency",
            "generator_loss_real_imaginary",
            "generator_loss_mel",
            "generator_loss_adversarial",
            "generator_loss_feature_matching"
        }
        self.assertEqual(set(components), expected_keys)

    def test_generator_loss_propagates_gradient_to_the_candidate_views(self) -> None:
        # Amplitude, phase, and spectrum paths all reach the generator.
        candidate_log_amplitude: torch.Tensor = self._builder.spectral_plane(seed=821).requires_grad_(True)
        candidate_phase: torch.Tensor = self._builder.spectral_plane(seed=822).requires_grad_(True)
        candidate_real: torch.Tensor = self._builder.spectral_plane(seed=823).requires_grad_(True)
        total: torch.Tensor
        total, _ = self._loss.compute_generator_loss(
            reference_spectrum=self._spectrum,
            candidate_log_amplitude=candidate_log_amplitude,
            candidate_phase=candidate_phase,
            candidate_real_spectrum=candidate_real,
            candidate_imaginary_spectrum=self._builder.spectral_plane(seed=824),
            final_spectrum=self._spectrum,
            reference_mel=self._builder.mel(0.0),
            candidate_mel=self._builder.mel(0.5),
            fake_period_logits=self._builder.logits(0.0),
            fake_resolution_logits=self._builder.logits(0.0),
            real_period_features=self._builder.feature_maps(1.0),
            fake_period_features=self._builder.feature_maps(0.0),
            real_resolution_features=self._builder.feature_maps(1.0),
            fake_resolution_features=self._builder.feature_maps(0.0)
        )
        total.backward()
        self.assertIsNotNone(candidate_log_amplitude.grad, msg="The amplitude view must receive gradient")
        self.assertIsNotNone(candidate_phase.grad, msg="The phase view must receive gradient")
        self.assertIsNotNone(candidate_real.grad, msg="The real spectrum view must receive gradient")
        self.assertTrue(torch.isfinite(candidate_phase.grad).all().item())

    def _compute_generator(self) -> tuple[torch.Tensor, dict[str, float]]:
        # Runs the generator objective on one mismatched fixture set.
        return self._loss.compute_generator_loss(
            reference_spectrum=self._spectrum,
            candidate_log_amplitude=self._builder.spectral_plane(seed=812),
            candidate_phase=self._builder.spectral_plane(seed=813),
            candidate_real_spectrum=self._builder.spectral_plane(seed=814),
            candidate_imaginary_spectrum=self._builder.spectral_plane(seed=815),
            final_spectrum=self._builder.spectrum(seed=816),
            reference_mel=self._builder.mel(0.0),
            candidate_mel=self._builder.mel(0.5),
            fake_period_logits=self._builder.logits(0.25),
            fake_resolution_logits=self._builder.logits(-0.5),
            real_period_features=self._builder.feature_maps(1.0),
            fake_period_features=self._builder.feature_maps(0.0),
            real_resolution_features=self._builder.feature_maps(1.0),
            fake_resolution_features=self._builder.feature_maps(0.25)
        )


class RndvocDiscriminatorObjectiveTest(unittest.TestCase):
    # Verifies the hinge separation term and the resolution weighting.
    def setUp(self) -> None:
        # No spectrum is built, because the discriminator path consumes only
        # logits. The configuration is retained for the resolution weight,
        # which at its default of one means the two ensembles contribute
        # equally here, unlike the APNet2 and Vocos recipes that attenuate
        # the resolution ensemble.
        self._builder: RndvocFixtureBuilder = RndvocFixtureBuilder(member_count=2, layer_count=2)
        self._configuration: RndvocLossConfig = RndvocLossConfig()
        self._loss: RndvocLoss = RndvocLoss(self._configuration)

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
        # RNDVoC gives the resolution ensemble the same weight as the period one.
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
            + self._configuration.resolution_discriminator_weight
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
        narrow_builder: RndvocFixtureBuilder = RndvocFixtureBuilder(member_count=1, layer_count=2)
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


class RndvocValidationObjectiveTest(unittest.TestCase):
    # Verifies the adversarial-free validation composition and its renamed panel.
    def setUp(self) -> None:
        # Uses its own seed so this class's targets are distinct from the
        # generator class's, which matters because both paths run the same
        # shared reconstruction method: a value accidentally carried between
        # them would otherwise go unnoticed. The ensemble size is irrelevant
        # on this path, since validation evaluates no discriminator.
        self._builder: RndvocFixtureBuilder = RndvocFixtureBuilder(member_count=2, layer_count=2)
        self._configuration: RndvocLossConfig = RndvocLossConfig()
        self._loss: RndvocLoss = RndvocLoss(self._configuration)
        self._spectrum: Apnet2Spectrum = self._builder.spectrum(seed=831)

    def test_validation_loss_is_zero_at_perfect_reconstruction(self) -> None:
        # Matching every spectral view leaves nothing to charge.
        total: torch.Tensor
        total, _ = self._loss.compute_validation_loss(
            reference_spectrum=self._spectrum,
            candidate_log_amplitude=self._spectrum.log_amplitude,
            candidate_phase=self._spectrum.phase,
            candidate_real_spectrum=self._spectrum.real_spectrum,
            candidate_imaginary_spectrum=self._spectrum.imaginary_spectrum,
            final_spectrum=self._spectrum,
            reference_mel=self._builder.mel(0.0),
            candidate_mel=self._builder.mel(0.0)
        )
        self.assertAlmostEqual(float(total.item()), 0.0, places=5)

    def test_validation_total_equals_the_weighted_sum_of_its_components(self) -> None:
        # Validation drops the adversarial terms and keeps the reconstruction ones.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._compute_validation()
        expected: float = (
            self._configuration.amplitude_weight * components["loss_amplitude"]
            + self._configuration.phase_weight * components["loss_phase"]
            + self._configuration.consistency_weight * components["loss_consistency"]
            + self._configuration.real_imaginary_weight * components["loss_real_imaginary"]
            + self._configuration.mel_weight * components["loss_mel"]
        )
        self.assertAlmostEqual(float(total.item()), expected, places=3)

    def test_validation_components_expose_the_renamed_keys(self) -> None:
        # The validation panel renames the shared reconstruction terms.
        components: dict[str, float]
        _, components = self._compute_validation()
        expected_keys: set[str] = {
            "loss_total",
            "loss_amplitude",
            "loss_phase",
            "loss_consistency",
            "loss_real_imaginary",
            "loss_mel"
        }
        self.assertEqual(set(components), expected_keys)

    def test_validation_reconstruction_terms_match_the_generator_call(self) -> None:
        # Both entry points share one reconstruction path on identical inputs.
        generator_total: torch.Tensor
        generator_components: dict[str, float]
        generator_total, generator_components = self._loss.compute_generator_loss(
            reference_spectrum=self._spectrum,
            candidate_log_amplitude=self._builder.spectral_plane(seed=832),
            candidate_phase=self._builder.spectral_plane(seed=833),
            candidate_real_spectrum=self._builder.spectral_plane(seed=834),
            candidate_imaginary_spectrum=self._builder.spectral_plane(seed=835),
            final_spectrum=self._builder.spectrum(seed=836),
            reference_mel=self._builder.mel(0.0),
            candidate_mel=self._builder.mel(0.5),
            fake_period_logits=self._builder.logits(0.25),
            fake_resolution_logits=self._builder.logits(-0.5),
            real_period_features=self._builder.feature_maps(1.0),
            fake_period_features=self._builder.feature_maps(0.0),
            real_resolution_features=self._builder.feature_maps(1.0),
            fake_resolution_features=self._builder.feature_maps(0.25)
        )
        validation_total: torch.Tensor
        validation_components: dict[str, float]
        validation_total, validation_components = self._compute_validation()
        self.assertAlmostEqual(
            validation_components["loss_amplitude"],
            generator_components["generator_loss_amplitude"],
            places=5
        )
        self.assertAlmostEqual(
            validation_components["loss_phase"],
            generator_components["generator_loss_phase"],
            places=5
        )
        adversarial_contribution: float = (
            self._configuration.adversarial_weight * generator_components["generator_loss_adversarial"]
            + self._configuration.feature_matching_weight
            * generator_components["generator_loss_feature_matching"]
        )
        self.assertAlmostEqual(
            float(generator_total.item()) - adversarial_contribution,
            float(validation_total.item()),
            places=3,
            msg="Validation must equal the generator total without its adversarial terms"
        )

    def _compute_validation(self) -> tuple[torch.Tensor, dict[str, float]]:
        # Runs the validation objective on one mismatched fixture set.
        return self._loss.compute_validation_loss(
            reference_spectrum=self._spectrum,
            candidate_log_amplitude=self._builder.spectral_plane(seed=832),
            candidate_phase=self._builder.spectral_plane(seed=833),
            candidate_real_spectrum=self._builder.spectral_plane(seed=834),
            candidate_imaginary_spectrum=self._builder.spectral_plane(seed=835),
            final_spectrum=self._builder.spectrum(seed=836),
            reference_mel=self._builder.mel(0.0),
            candidate_mel=self._builder.mel(0.5)
        )
