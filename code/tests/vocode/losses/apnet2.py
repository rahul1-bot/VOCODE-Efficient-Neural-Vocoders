# This module:
# 1. Verifies the APNet2 spectrum analyzer: the accepted waveform layouts, the
#    internal consistency of the four returned spectral views, the silence
#    anchor, and the rejection of unsupported ranks
# 2. Verifies the frozen spectrum record and the frozen loss weight record
# 3. Verifies the APNet2 generator composition: the analytic zero at perfect
#    reconstruction with full fooling, the weighted sum of the reported
#    components, the deliberate cropping of over-long candidates, and the
#    gradient path
# 4. Verifies the anti-wrapping property of the phase losses, where a full-turn
#    phase offset must cost exactly nothing
# 5. Verifies the hinge discriminator composition and the validation composition
#
# Design decisions:
# - The anti-wrapping property is asserted on synthetic phase tensors rather
#   than analyzer output, so the comparison is exact rather than accumulating
#   matmul rounding across a large spectrogram
# - The silence anchor pins the analyzer's numeric floor: a zero waveform must
#   produce the logarithm of the epsilon and a zero phase, which fails if the
#   floor is ever moved
# - The analyzer runs on a short synthetic waveform with a small transform, so
#   the whole file stays inside the CPU runtime budget
# - Field relationships (phase against atan2, amplitude against the magnitude)
#   are asserted instead of re-deriving the STFT, which would restate the
#   implementation rather than test it
#
# Author: Rahul Sawhney

import math
import unittest

import torch
from pydantic import ValidationError

from vocode.losses.apnet2 import Apnet2Loss, Apnet2LossConfig, Apnet2Spectrum, Apnet2SpectrumAnalyzer


class SpectrumFixtureBuilder:
    # Builds synthetic spectra, logits, feature maps, and mels for the composite.
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


class Apnet2SpectrumAnalysisTest(unittest.TestCase):
    # Verifies the STFT analyzer that produces the views the loss compares.
    def setUp(self) -> None:
        # The two expected extents are the framing settings' consequences,
        # recorded here so a case asserts a derived shape rather than a
        # magic number. A 64-point transform yields 33 non-redundant bins,
        # and 512 samples at a hop of 16 under centred framing yield 33
        # frames; either expectation changing without the settings changing
        # would mean the analyzer stopped honouring them.
        self._analyzer: Apnet2SpectrumAnalyzer = Apnet2SpectrumAnalyzer(n_fft=64, hop_size=16, win_size=64)
        self._sample_count: int = 512
        self._expected_bin_count: int = 33
        self._expected_frame_count: int = 33

    def test_single_channel_waveform_is_promoted_to_a_batch(self) -> None:
        # A bare time axis is accepted and analyzed as one utterance.
        torch.manual_seed(111)
        waveform: torch.Tensor = torch.randn(self._sample_count)
        spectrum: Apnet2Spectrum = self._analyzer.analyze(waveform)
        self.assertEqual(
            spectrum.log_amplitude.shape,
            torch.Size([1, self._expected_bin_count, self._expected_frame_count])
        )

    def test_batched_waveform_keeps_its_batch_dimension(self) -> None:
        # The batch axis is carried through the analysis unchanged.
        torch.manual_seed(112)
        waveform: torch.Tensor = torch.randn(2, self._sample_count)
        spectrum: Apnet2Spectrum = self._analyzer.analyze(waveform)
        self.assertEqual(spectrum.phase.shape[0], 2)
        self.assertEqual(spectrum.phase.shape[1], self._expected_bin_count)

    def test_single_channel_axis_is_squeezed(self) -> None:
        # The mono channel axis of a synthesis output is removed, not analyzed.
        torch.manual_seed(113)
        waveform: torch.Tensor = torch.randn(2, 1, self._sample_count)
        spectrum: Apnet2Spectrum = self._analyzer.analyze(waveform)
        self.assertEqual(
            spectrum.real_spectrum.shape,
            torch.Size([2, self._expected_bin_count, self._expected_frame_count])
        )

    def test_unsupported_rank_is_rejected(self) -> None:
        # A layout the analysis cannot interpret must fail closed.
        with self.assertRaisesRegex(ValueError, "Expected waveform shape"):
            self._analyzer.analyze(torch.zeros(2, 3, 4, 5))

    def test_multi_channel_waveform_is_rejected(self) -> None:
        # A genuine stereo input is not silently collapsed.
        with self.assertRaisesRegex(ValueError, "Expected waveform shape"):
            self._analyzer.analyze(torch.zeros(2, 2, self._sample_count))

    def test_phase_view_agrees_with_the_returned_spectrum_parts(self) -> None:
        # The phase is the argument of the returned real and imaginary parts.
        torch.manual_seed(114)
        spectrum: Apnet2Spectrum = self._analyzer.analyze(torch.randn(1, self._sample_count))
        expected_phase: torch.Tensor = torch.atan2(spectrum.imaginary_spectrum, spectrum.real_spectrum)
        self.assertTrue(torch.allclose(spectrum.phase, expected_phase, atol=1e-6))

    def test_phase_stays_inside_the_principal_interval(self) -> None:
        # The analyzer emits wrapped phase, which is what the anti-wrap losses assume.
        torch.manual_seed(115)
        spectrum: Apnet2Spectrum = self._analyzer.analyze(torch.randn(1, self._sample_count))
        self.assertGreaterEqual(float(spectrum.phase.min().item()), -math.pi - 1e-5)
        self.assertLessEqual(float(spectrum.phase.max().item()), math.pi + 1e-5)

    def test_log_amplitude_agrees_with_the_returned_spectrum_magnitude(self) -> None:
        # The amplitude view is the floored logarithm of the magnitude.
        torch.manual_seed(116)
        spectrum: Apnet2Spectrum = self._analyzer.analyze(torch.randn(1, self._sample_count))
        magnitude: torch.Tensor = torch.sqrt(
            spectrum.real_spectrum.pow(2) + spectrum.imaginary_spectrum.pow(2)
        )
        recovered: torch.Tensor = torch.exp(spectrum.log_amplitude)
        self.assertTrue(torch.allclose(recovered, magnitude + 1e-5, atol=1e-4))
        self.assertTrue(torch.isfinite(spectrum.log_amplitude).all().item())

    def test_silence_reaches_the_numeric_floor_of_the_analyzer(self) -> None:
        # A zero waveform has no magnitude, so the epsilon floor is exposed.
        spectrum: Apnet2Spectrum = self._analyzer.analyze(torch.zeros(1, self._sample_count))
        self.assertAlmostEqual(float(spectrum.log_amplitude.max().item()), math.log(1e-5), places=4)
        self.assertAlmostEqual(float(spectrum.phase.abs().max().item()), 0.0, places=6)


class Apnet2SpectrumRecordTest(unittest.TestCase):
    # Verifies the frozen record carrying the four spectral views.
    def setUp(self) -> None:
        # One plane is reused for all four fields, because these cases assert
        # the record's typing and immutability rather than any numeric
        # behavior; distinct values would add nothing and would obscure that
        # the record imposes no relationship between its four views.
        self._plane: torch.Tensor = torch.zeros(1, 5, 4)
        self._spectrum: Apnet2Spectrum = Apnet2Spectrum(
            log_amplitude=self._plane,
            phase=self._plane,
            real_spectrum=self._plane,
            imaginary_spectrum=self._plane
        )

    def test_record_retains_the_tensors_it_was_built_from(self) -> None:
        # The record is a view holder, so the tensors pass through untouched.
        self.assertIs(self._spectrum.log_amplitude, self._plane)
        self.assertIs(self._spectrum.imaginary_spectrum, self._plane)

    def test_record_rejects_mutation_after_construction(self) -> None:
        # The analyzed spectrum is a value object and cannot be rewritten.
        with self.assertRaises(ValidationError):
            self._spectrum.phase: torch.Tensor = torch.ones(1, 5, 4)


class Apnet2LossConfigurationTest(unittest.TestCase):
    # Verifies the frozen weight record behind the APNet2 composition.
    def setUp(self) -> None:
        # Constructs the record from its defaults with no arguments, so the
        # cases below assert the seven reference weights the shipped
        # configuration actually applies rather than values restated at the
        # call site.
        self._configuration: Apnet2LossConfig = Apnet2LossConfig()

    def test_default_weights_match_the_reference_recipe(self) -> None:
        # The phase term dominates the reference APNet2 weighting.
        self.assertEqual(self._configuration.amplitude_weight, 45.0)
        self.assertEqual(self._configuration.phase_weight, 100.0)
        self.assertEqual(self._configuration.spectrum_weight, 20.0)
        self.assertEqual(self._configuration.mel_weight, 45.0)
        self.assertEqual(self._configuration.resolution_discriminator_weight, 0.1)
        self.assertEqual(self._configuration.resolution_feature_matching_weight, 0.1)
        self.assertEqual(self._configuration.real_imaginary_weight, 2.25)

    def test_configuration_rejects_mutation_after_construction(self) -> None:
        # The record is frozen so a run cannot drift from its logged weights.
        with self.assertRaises(ValidationError):
            self._configuration.phase_weight: float = 1.0

    def test_configuration_rejects_an_unknown_weight(self) -> None:
        # An unknown field is a typo, never a silently ignored setting.
        with self.assertRaises(ValidationError):
            Apnet2LossConfig(consistency_weight=45.0)

    def test_configuration_rejects_a_non_positive_weight(self) -> None:
        # Every weight is strictly positive in the reference composition.
        with self.assertRaises(ValidationError):
            Apnet2LossConfig(phase_weight=-100.0)

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The composite exposes the exact record it was constructed with.
        configuration: Apnet2LossConfig = Apnet2LossConfig(mel_weight=1.0)
        loss: Apnet2Loss = Apnet2Loss(configuration)
        self.assertIs(loss.configuration, configuration)


class Apnet2GeneratorObjectiveTest(unittest.TestCase):
    # Verifies the generator composition over amplitude, phase, spectrum,
    # and adversarial terms.
    def setUp(self) -> None:
        # The reference spectrum is seeded so the four supervised domains
        # each receive a non-degenerate target that is identical across runs;
        # a constant spectrum would let a term that ignored its input pass.
        # The configuration is retained as an attribute because the
        # composition cases read its weights back to rebuild the expected
        # total from the reported components.
        self._builder: SpectrumFixtureBuilder = SpectrumFixtureBuilder(member_count=2, layer_count=2)
        self._configuration: Apnet2LossConfig = Apnet2LossConfig()
        self._loss: Apnet2Loss = Apnet2Loss(self._configuration)
        self._spectrum: Apnet2Spectrum = self._builder.spectrum(seed=201)

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
            places=5,
            msg="A perfect generator must pay nothing under the APNet2 composition"
        )
        self.assertAlmostEqual(components["generator_loss_amplitude"], 0.0, places=6)
        self.assertAlmostEqual(components["generator_loss_phase"], 0.0, places=6)
        self.assertAlmostEqual(components["generator_loss_spectrum"], 0.0, places=6)

    def test_generator_total_equals_the_weighted_sum_of_its_components(self) -> None:
        # The composition arithmetic must match the reported per-term values.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_generator_loss(
            reference_spectrum=self._spectrum,
            candidate_log_amplitude=self._builder.spectral_plane(seed=202),
            candidate_phase=self._builder.spectral_plane(seed=203),
            candidate_real_spectrum=self._builder.spectral_plane(seed=204),
            candidate_imaginary_spectrum=self._builder.spectral_plane(seed=205),
            final_spectrum=self._builder.spectrum(seed=206),
            reference_mel=self._builder.mel(0.0),
            candidate_mel=self._builder.mel(0.5),
            fake_period_logits=self._builder.logits(0.25),
            fake_resolution_logits=self._builder.logits(-0.5),
            real_period_features=self._builder.feature_maps(1.0),
            fake_period_features=self._builder.feature_maps(0.0),
            real_resolution_features=self._builder.feature_maps(1.0),
            fake_resolution_features=self._builder.feature_maps(0.25)
        )
        waveform_terms: float = (
            components["generator_loss_adversarial_period"]
            + self._configuration.resolution_discriminator_weight
            * components["generator_loss_adversarial_resolution"]
            + components["generator_loss_feature_matching_period"]
            + self._configuration.resolution_feature_matching_weight
            * components["generator_loss_feature_matching_resolution"]
            + self._configuration.mel_weight * components["generator_loss_mel"]
        )
        expected: float = (
            self._configuration.amplitude_weight * components["generator_loss_amplitude"]
            + self._configuration.phase_weight * components["generator_loss_phase"]
            + self._configuration.spectrum_weight * components["generator_loss_spectrum"]
            + waveform_terms
        )
        self.assertAlmostEqual(float(total.item()), expected, places=3)
        self.assertAlmostEqual(components["generator_loss_total"], float(total.item()), places=4)

    def test_generator_components_expose_the_reference_keys(self) -> None:
        # The logged component panel is part of the training contract.
        components: dict[str, float]
        _, components = self._loss.compute_generator_loss(
            reference_spectrum=self._spectrum,
            candidate_log_amplitude=self._builder.spectral_plane(seed=211),
            candidate_phase=self._builder.spectral_plane(seed=212),
            candidate_real_spectrum=self._builder.spectral_plane(seed=213),
            candidate_imaginary_spectrum=self._builder.spectral_plane(seed=214),
            final_spectrum=self._spectrum,
            reference_mel=self._builder.mel(0.0),
            candidate_mel=self._builder.mel(0.0),
            fake_period_logits=self._builder.logits(0.0),
            fake_resolution_logits=self._builder.logits(0.0),
            real_period_features=self._builder.feature_maps(0.0),
            fake_period_features=self._builder.feature_maps(0.0),
            real_resolution_features=self._builder.feature_maps(0.0),
            fake_resolution_features=self._builder.feature_maps(0.0)
        )
        expected_keys: set[str] = {
            "generator_loss_total",
            "generator_loss_amplitude",
            "generator_loss_phase",
            "generator_loss_spectrum",
            "generator_loss_mel",
            "generator_loss_adversarial_period",
            "generator_loss_adversarial_resolution",
            "generator_loss_feature_matching_period",
            "generator_loss_feature_matching_resolution"
        }
        self.assertEqual(set(components), expected_keys)

    def test_generator_loss_propagates_gradient_to_the_candidate_views(self) -> None:
        # Amplitude, phase, and spectrum paths all reach the generator.
        candidate_log_amplitude: torch.Tensor = self._builder.spectral_plane(seed=221).requires_grad_(True)
        candidate_phase: torch.Tensor = self._builder.spectral_plane(seed=222).requires_grad_(True)
        candidate_real: torch.Tensor = self._builder.spectral_plane(seed=223).requires_grad_(True)
        total: torch.Tensor
        total, _ = self._loss.compute_generator_loss(
            reference_spectrum=self._spectrum,
            candidate_log_amplitude=candidate_log_amplitude,
            candidate_phase=candidate_phase,
            candidate_real_spectrum=candidate_real,
            candidate_imaginary_spectrum=self._builder.spectral_plane(seed=224),
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

    def test_over_long_candidate_views_are_cropped_to_the_reference(self) -> None:
        # The paired crop tolerates a trailing frame instead of failing.
        candidate_amplitude: torch.Tensor = self._builder.spectral_plane(seed=231)
        extended_amplitude: torch.Tensor = torch.cat(
            [candidate_amplitude, candidate_amplitude[..., :1]],
            dim=-1
        )
        cropped_total: torch.Tensor
        cropped_total, _ = self._compute_amplitude_only(candidate_amplitude)
        extended_total: torch.Tensor
        extended_total, _ = self._compute_amplitude_only(extended_amplitude)
        self.assertAlmostEqual(float(cropped_total.item()), float(extended_total.item()), places=4)

    def _compute_amplitude_only(self, candidate_log_amplitude: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        # Runs the generator objective varying only the amplitude view.
        return self._loss.compute_generator_loss(
            reference_spectrum=self._spectrum,
            candidate_log_amplitude=candidate_log_amplitude,
            candidate_phase=self._spectrum.phase,
            candidate_real_spectrum=self._spectrum.real_spectrum,
            candidate_imaginary_spectrum=self._spectrum.imaginary_spectrum,
            final_spectrum=self._spectrum,
            reference_mel=self._builder.mel(0.0),
            candidate_mel=self._builder.mel(0.0),
            fake_period_logits=self._builder.logits(1.0),
            fake_resolution_logits=self._builder.logits(1.0),
            real_period_features=self._builder.feature_maps(0.0),
            fake_period_features=self._builder.feature_maps(0.0),
            real_resolution_features=self._builder.feature_maps(0.0),
            fake_resolution_features=self._builder.feature_maps(0.0)
        )


class Apnet2AntiWrappingPhaseTest(unittest.TestCase):
    # Verifies that the phase terms charge nothing for a full-turn offset.
    def setUp(self) -> None:
        # A single-member ensemble suffices because these cases isolate the
        # phase terms, which never touch the discriminators; the smallest
        # ensemble keeps the adversarial contribution constant across the
        # comparison so any difference in the total is attributable to phase
        # alone. A separately seeded candidate phase gives a genuinely
        # different surface to offset by a full turn.
        self._builder: SpectrumFixtureBuilder = SpectrumFixtureBuilder(member_count=1, layer_count=1)
        self._loss: Apnet2Loss = Apnet2Loss(Apnet2LossConfig())
        self._spectrum: Apnet2Spectrum = self._builder.spectrum(seed=301)
        self._candidate_phase: torch.Tensor = self._builder.spectral_plane(seed=302)

    def test_full_turn_offset_leaves_the_phase_term_unchanged(self) -> None:
        # A whole-turn shift is perceptually identical and must cost nothing extra.
        plain: float = self._phase_component(self._candidate_phase)
        shifted: float = self._phase_component(self._candidate_phase + 2.0 * math.pi)
        self.assertAlmostEqual(
            plain,
            shifted,
            places=4,
            msg="The sawtooth projection must absorb a full-turn phase offset"
        )

    def test_double_turn_offset_leaves_the_phase_term_unchanged(self) -> None:
        # The projection absorbs any integer number of turns, not only one.
        plain: float = self._phase_component(self._candidate_phase)
        shifted: float = self._phase_component(self._candidate_phase - 4.0 * math.pi)
        self.assertAlmostEqual(plain, shifted, places=4)

    def test_matching_phase_is_the_optimum_of_the_phase_term(self) -> None:
        # An exact match costs nothing, and a mismatch costs strictly more.
        matched: float = self._phase_component(self._spectrum.phase)
        mismatched: float = self._phase_component(self._candidate_phase)
        self.assertAlmostEqual(matched, 0.0, places=6)
        self.assertGreater(mismatched, matched)

    def test_partial_turn_offset_is_penalized(self) -> None:
        # A half-turn offset is a genuine phase error and must be charged.
        plain: float = self._phase_component(self._candidate_phase)
        shifted: float = self._phase_component(self._candidate_phase + math.pi)
        self.assertNotAlmostEqual(plain, shifted, places=3)

    def _phase_component(self, candidate_phase: torch.Tensor) -> float:
        # Runs the generator objective varying only the phase view.
        components: dict[str, float]
        _, components = self._loss.compute_generator_loss(
            reference_spectrum=self._spectrum,
            candidate_log_amplitude=self._spectrum.log_amplitude,
            candidate_phase=candidate_phase,
            candidate_real_spectrum=self._spectrum.real_spectrum,
            candidate_imaginary_spectrum=self._spectrum.imaginary_spectrum,
            final_spectrum=self._spectrum,
            reference_mel=self._builder.mel(0.0),
            candidate_mel=self._builder.mel(0.0),
            fake_period_logits=self._builder.logits(1.0),
            fake_resolution_logits=self._builder.logits(1.0),
            real_period_features=self._builder.feature_maps(0.0),
            fake_period_features=self._builder.feature_maps(0.0),
            real_resolution_features=self._builder.feature_maps(0.0),
            fake_resolution_features=self._builder.feature_maps(0.0)
        )
        return components["generator_loss_phase"]


class Apnet2DiscriminatorObjectiveTest(unittest.TestCase):
    # Verifies the hinge separation term and the resolution weighting.
    def setUp(self) -> None:
        # No spectrum is built, because the discriminator path consumes only
        # logits; a spectral fixture here would be dead weight and would
        # suggest the separation term depends on reconstruction, which it
        # does not. Two members keeps this family's unnormalized summation
        # distinguishable from a mean.
        self._builder: SpectrumFixtureBuilder = SpectrumFixtureBuilder(member_count=2, layer_count=2)
        self._configuration: Apnet2LossConfig = Apnet2LossConfig()
        self._loss: Apnet2Loss = Apnet2Loss(self._configuration)

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
            + self._configuration.resolution_discriminator_weight
            * components["discriminator_loss_resolution"]
        )
        self.assertAlmostEqual(float(total.item()), expected, places=5)

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
        narrow_builder: SpectrumFixtureBuilder = SpectrumFixtureBuilder(member_count=1, layer_count=2)
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


class Apnet2ValidationObjectiveTest(unittest.TestCase):
    # Verifies the adversarial-free validation composition.
    def setUp(self) -> None:
        # A single-member ensemble is enough because the validation path
        # evaluates no discriminator at all; the builder is retained only for
        # its spectral fixtures. Its own seed keeps this class's targets
        # distinct from the generator class's, so a value accidentally shared
        # between the two paths would not go unnoticed.
        self._builder: SpectrumFixtureBuilder = SpectrumFixtureBuilder(member_count=1, layer_count=1)
        self._configuration: Apnet2LossConfig = Apnet2LossConfig()
        self._loss: Apnet2Loss = Apnet2Loss(self._configuration)
        self._spectrum: Apnet2Spectrum = self._builder.spectrum(seed=401)

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
        # Validation drops the adversarial terms and keeps the spectral ones.
        total: torch.Tensor
        components: dict[str, float]
        total, components = self._loss.compute_validation_loss(
            reference_spectrum=self._spectrum,
            candidate_log_amplitude=self._builder.spectral_plane(seed=402),
            candidate_phase=self._builder.spectral_plane(seed=403),
            candidate_real_spectrum=self._builder.spectral_plane(seed=404),
            candidate_imaginary_spectrum=self._builder.spectral_plane(seed=405),
            final_spectrum=self._builder.spectrum(seed=406),
            reference_mel=self._builder.mel(0.0),
            candidate_mel=self._builder.mel(0.5)
        )
        expected: float = (
            self._configuration.amplitude_weight * components["loss_amplitude"]
            + self._configuration.phase_weight * components["loss_phase"]
            + self._configuration.spectrum_weight * components["loss_spectrum"]
            + self._configuration.mel_weight * components["loss_mel"]
        )
        self.assertAlmostEqual(float(total.item()), expected, places=3)

    def test_validation_components_expose_the_reference_keys(self) -> None:
        # The validation panel is distinct from the training panel.
        components: dict[str, float]
        _, components = self._loss.compute_validation_loss(
            reference_spectrum=self._spectrum,
            candidate_log_amplitude=self._spectrum.log_amplitude,
            candidate_phase=self._spectrum.phase,
            candidate_real_spectrum=self._spectrum.real_spectrum,
            candidate_imaginary_spectrum=self._spectrum.imaginary_spectrum,
            final_spectrum=self._spectrum,
            reference_mel=self._builder.mel(0.0),
            candidate_mel=self._builder.mel(0.0)
        )
        expected_keys: set[str] = {
            "loss_total",
            "loss_amplitude",
            "loss_phase",
            "loss_spectrum",
            "loss_mel"
        }
        self.assertEqual(set(components), expected_keys)
