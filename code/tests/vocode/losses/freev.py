# This module:
# 1. Verifies that the FreeV loss configuration and spectrum records inherit
#    the APNet2 lineage: the same reference weights, the same frozen behavior,
#    and a genuine subtype relationship
# 2. Verifies the delegation contract of FreevLoss: the generator,
#    discriminator, and validation results must equal those of an APNet2 loss
#    built from the same weights, on identical inputs
# 3. Verifies that a non-default weight survives the configuration rebuild the
#    delegation performs, so the prior-refined recipe is not silently reset to
#    the APNet2 defaults
#
# Design decisions:
# - Equality against APNet2 is asserted on the totals and on every reported
#   component, because this module is a pure forwarding shim and any behavioral
#   difference is by definition a defect
# - The weight-propagation test contrasts a custom-weight APNet2 loss against a
#   default-weight one, so an override that failed to reach the delegate would
#   fail the test instead of matching by coincidence
# - Fixtures are synthetic spectra rather than analyzer output, because the
#   forwarding path is independent of how the spectra were produced
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError

from vocode.losses.apnet2 import Apnet2Loss, Apnet2LossConfig, Apnet2Spectrum
from vocode.losses.freev import FreevLoss, FreevLossConfig, FreevSpectrum


class FreevFixtureBuilder:
    # Builds the spectra, logits, feature maps, and mels both losses consume.
    def __init__(self, member_count: int, layer_count: int) -> None:
        # Binds the constructor inputs into this component's state.
        self._member_count: int = member_count
        self._layer_count: int = layer_count
        self._bin_count: int = 9
        self._frame_count: int = 7
        self._logit_shape: tuple[int, int] = (1, 4)
        self._feature_shape: tuple[int, int, int] = (1, 2, 3)
        self._mel_shape: tuple[int, int, int] = (1, 6, 7)

    def spectrum(self, seed: int) -> FreevSpectrum:
        # Builds a seeded synthetic spectrum record of the FreeV subtype.
        torch.manual_seed(seed)
        return FreevSpectrum(
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


class FreevLossConfigurationTest(unittest.TestCase):
    # Verifies that the FreeV weight record is the APNet2 record by inheritance.
    def setUp(self) -> None:
        # Constructs the FreeV record from its defaults. Because the subclass
        # declares no field of its own, every value read here originates in
        # the APNet2 record, which is precisely the inheritance these cases
        # assert.
        self._configuration: FreevLossConfig = FreevLossConfig()

    def test_configuration_is_an_apnet2_configuration(self) -> None:
        # FreeV reuses the APNet2 objective, so it reuses its weight contract.
        self.assertIsInstance(self._configuration, Apnet2LossConfig)

    def test_default_weights_match_the_apnet2_lineage(self) -> None:
        # No weight is re-tuned when the prior refinement is added.
        self.assertEqual(self._configuration.model_dump(), Apnet2LossConfig().model_dump())

    def test_configuration_rejects_mutation_after_construction(self) -> None:
        # The frozen behavior is inherited alongside the fields.
        with self.assertRaises(ValidationError):
            self._configuration.amplitude_weight: float = 1.0

    def test_configuration_rejects_an_unknown_weight(self) -> None:
        # The inherited record still forbids fields it does not declare.
        with self.assertRaises(ValidationError):
            FreevLossConfig(prior_weight=1.0)

    def test_configuration_property_returns_the_freev_record(self) -> None:
        # The composite exposes its own record, not the rebuilt delegate record.
        configuration: FreevLossConfig = FreevLossConfig(mel_weight=1.0)
        loss: FreevLoss = FreevLoss(configuration)
        self.assertIs(loss.configuration, configuration)
        self.assertIsInstance(loss.configuration, FreevLossConfig)


class FreevSpectrumRecordTest(unittest.TestCase):
    # Verifies the FreeV spectrum record and its APNet2 subtype relationship.
    def setUp(self) -> None:
        # One plane is reused for all four spectrum fields, because these
        # cases assert the record's typing and subtype relationship rather
        # than any numeric behavior; distinct values would add nothing and
        # would obscure that the shape, not the content, is what matters.
        self._plane: torch.Tensor = torch.zeros(1, 5, 4)
        self._spectrum: FreevSpectrum = FreevSpectrum(
            log_amplitude=self._plane,
            phase=self._plane,
            real_spectrum=self._plane,
            imaginary_spectrum=self._plane
        )

    def test_spectrum_is_an_apnet2_spectrum(self) -> None:
        # The delegate consumes APNet2 spectra, so the subtype must hold.
        self.assertIsInstance(self._spectrum, Apnet2Spectrum)

    def test_spectrum_retains_the_tensors_it_was_built_from(self) -> None:
        # The record is a view holder, so the tensors pass through untouched.
        self.assertIs(self._spectrum.real_spectrum, self._plane)

    def test_spectrum_rejects_mutation_after_construction(self) -> None:
        # The frozen behavior is inherited alongside the fields.
        with self.assertRaises(ValidationError):
            self._spectrum.phase: torch.Tensor = torch.ones(1, 5, 4)


class FreevDelegationTest(unittest.TestCase):
    # Verifies that every FreeV objective forwards to the APNet2 objective
    # without changing a single reported number.
    def setUp(self) -> None:
        # Builds both objectives side by side from equal default records, so
        # each case can run the same fixture through the wrapper and through
        # a directly constructed APNet2 loss and require identical numbers.
        # That paired construction is the only way to prove delegation
        # changes nothing, since the wrapper reports the delegate's own
        # component panel and would otherwise be indistinguishable from a
        # reimplementation.
        self._builder: FreevFixtureBuilder = FreevFixtureBuilder(member_count=2, layer_count=2)
        self._freev_loss: FreevLoss = FreevLoss(FreevLossConfig())
        self._apnet2_loss: Apnet2Loss = Apnet2Loss(Apnet2LossConfig())
        self._spectrum: FreevSpectrum = self._builder.spectrum(seed=501)

    def test_generator_loss_matches_the_apnet2_objective(self) -> None:
        # The forwarded generator call must reproduce the delegate exactly.
        freev_total: torch.Tensor
        freev_components: dict[str, float]
        freev_total, freev_components = self._compute_generator(self._freev_loss)
        apnet2_total: torch.Tensor
        apnet2_components: dict[str, float]
        apnet2_total, apnet2_components = self._compute_generator(self._apnet2_loss)
        self.assertAlmostEqual(float(freev_total.item()), float(apnet2_total.item()), places=5)
        self.assertEqual(freev_components, apnet2_components)

    def test_discriminator_loss_matches_the_apnet2_objective(self) -> None:
        # The forwarded discriminator call must reproduce the delegate exactly.
        freev_total: torch.Tensor
        freev_components: dict[str, float]
        freev_total, freev_components = self._freev_loss.compute_discriminator_loss(
            real_period_logits=self._builder.logits(0.5),
            fake_period_logits=self._builder.logits(-0.25),
            real_resolution_logits=self._builder.logits(0.75),
            fake_resolution_logits=self._builder.logits(0.1)
        )
        apnet2_total: torch.Tensor
        apnet2_components: dict[str, float]
        apnet2_total, apnet2_components = self._apnet2_loss.compute_discriminator_loss(
            real_period_logits=self._builder.logits(0.5),
            fake_period_logits=self._builder.logits(-0.25),
            real_resolution_logits=self._builder.logits(0.75),
            fake_resolution_logits=self._builder.logits(0.1)
        )
        self.assertAlmostEqual(float(freev_total.item()), float(apnet2_total.item()), places=5)
        self.assertEqual(freev_components, apnet2_components)

    def test_validation_loss_matches_the_apnet2_objective(self) -> None:
        # The forwarded validation call must reproduce the delegate exactly.
        freev_total: torch.Tensor
        freev_components: dict[str, float]
        freev_total, freev_components = self._compute_validation(self._freev_loss)
        apnet2_total: torch.Tensor
        apnet2_components: dict[str, float]
        apnet2_total, apnet2_components = self._compute_validation(self._apnet2_loss)
        self.assertAlmostEqual(float(freev_total.item()), float(apnet2_total.item()), places=5)
        self.assertEqual(freev_components, apnet2_components)

    def test_custom_weights_survive_the_delegate_rebuild(self) -> None:
        # The delegate is rebuilt from the record, so an override must reach it.
        custom_freev: FreevLoss = FreevLoss(FreevLossConfig(mel_weight=1.0))
        custom_apnet2: Apnet2Loss = Apnet2Loss(Apnet2LossConfig(mel_weight=1.0))
        custom_total: torch.Tensor
        custom_total, _ = self._compute_generator(custom_freev, candidate_mel_value=0.5)
        matched_total: torch.Tensor
        matched_total, _ = self._compute_generator(custom_apnet2, candidate_mel_value=0.5)
        default_total: torch.Tensor
        default_total, _ = self._compute_generator(self._apnet2_loss, candidate_mel_value=0.5)
        self.assertAlmostEqual(float(custom_total.item()), float(matched_total.item()), places=5)
        self.assertNotAlmostEqual(
            float(custom_total.item()),
            float(default_total.item()),
            places=3,
            msg="A custom mel weight must change the total, proving it reached the delegate"
        )

    def test_generator_loss_propagates_gradient_through_the_delegation(self) -> None:
        # Forwarding must not detach the candidate views from autograd.
        candidate_phase: torch.Tensor = self._builder.spectral_plane(seed=511).requires_grad_(True)
        total: torch.Tensor
        total, _ = self._freev_loss.compute_generator_loss(
            reference_spectrum=self._spectrum,
            candidate_log_amplitude=self._spectrum.log_amplitude,
            candidate_phase=candidate_phase,
            candidate_real_spectrum=self._spectrum.real_spectrum,
            candidate_imaginary_spectrum=self._spectrum.imaginary_spectrum,
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
        self.assertIsNotNone(candidate_phase.grad, msg="The phase view must receive gradient")
        self.assertTrue(torch.isfinite(candidate_phase.grad).all().item())

    def _compute_generator(
        self,
        loss: FreevLoss | Apnet2Loss,
        candidate_mel_value: float = 0.0
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Runs one generator objective on the shared fixture set.
        return loss.compute_generator_loss(
            reference_spectrum=self._spectrum,
            candidate_log_amplitude=self._builder.spectral_plane(seed=502),
            candidate_phase=self._builder.spectral_plane(seed=503),
            candidate_real_spectrum=self._builder.spectral_plane(seed=504),
            candidate_imaginary_spectrum=self._builder.spectral_plane(seed=505),
            final_spectrum=self._builder.spectrum(seed=506),
            reference_mel=self._builder.mel(0.0),
            candidate_mel=self._builder.mel(candidate_mel_value),
            fake_period_logits=self._builder.logits(0.25),
            fake_resolution_logits=self._builder.logits(-0.5),
            real_period_features=self._builder.feature_maps(1.0),
            fake_period_features=self._builder.feature_maps(0.0),
            real_resolution_features=self._builder.feature_maps(1.0),
            fake_resolution_features=self._builder.feature_maps(0.25)
        )

    def _compute_validation(self, loss: FreevLoss | Apnet2Loss) -> tuple[torch.Tensor, dict[str, float]]:
        # Runs one validation objective on the shared fixture set.
        return loss.compute_validation_loss(
            reference_spectrum=self._spectrum,
            candidate_log_amplitude=self._builder.spectral_plane(seed=507),
            candidate_phase=self._builder.spectral_plane(seed=508),
            candidate_real_spectrum=self._builder.spectral_plane(seed=509),
            candidate_imaginary_spectrum=self._builder.spectral_plane(seed=510),
            final_spectrum=self._builder.spectrum(seed=506),
            reference_mel=self._builder.mel(0.0),
            candidate_mel=self._builder.mel(0.25)
        )
