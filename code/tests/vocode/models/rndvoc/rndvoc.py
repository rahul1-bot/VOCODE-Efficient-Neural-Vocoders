# This module:
# 1. Verifies the RNDVoC configuration record: the Andong-Li 22.05 kHz
#    LJSpeech values, immutability, and the rejection of unknown fields
# 2. Verifies the RNDVoC module surface: manual optimization, the synthesis
#    and prediction contracts, the protocol properties, and the optimizer
#    and scheduler declaration
#
# Design decisions:
# - The module assertions run on a miniature refinement stack that keeps the
#   published transform geometry and mel protocols, because the band split
#   encodes the 1024-point bin count while the stage count and feature width
#   are free; the published values are asserted on the configuration record
# - Training is never executed: training_step and validation_step call the
#   harness logging surface and on_train_epoch_end reaches for attached
#   schedulers, so all three require a trainer and stay out of scope
# - Prediction lengths are derived from the module's own mel protocol rather
#   than hard-coded, so the assertion states the frame-to-sample contract
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError

from syntheticmind.core.optimizer import OptimizationConfiguration

from vocode.metrics.pesq import PesqConfig
from vocode.metrics.stoi import StoiConfig
from vocode.models.rndvoc.network import RndvocNetwork
from vocode.models.rndvoc.rndvoc import Rndvoc, RndvocConfig
from vocode.transforms.mel import MelConfig, MelSpectrogram


class MiniatureModuleRecipe:
    # Builds the miniature RNDVoC configuration used by the module-surface
    # assertions. The reduction touches only the decoder's depth and widths;
    # the transform geometry and both mel protocols stay at published values,
    # because the band split hard-codes the bin count they imply and the
    # decomposition is derived from the conditioning filterbank.
    def build(self) -> RndvocConfig:
        # Returns the reduced recipe: the published transform geometry with
        # one refinement stage instead of six. Both mel protocols are supplied
        # from their published factories rather than reduced, so the
        # conditioning and measurement records genuinely differ here as they
        # do in a study run.
        return RndvocConfig(
            sample_rate=22050,
            input_mel_channels=80,
            n_fft=1024,
            hop_size=256,
            win_size=1024,
            fmin=0.0,
            fmax=8000.0,
            null_stage_count=1,
            repeat_count=1,
            input_dimension=8,
            squeeze_dimension=4,
            hidden_dimension=8,
            kernel_size=3,
            mel_protocol=MelConfig.rndvoc_andong(),
            reconstruction_mel_protocol=MelConfig.rndvoc_andong_reconstruction(),
            pesq_protocol=PesqConfig(),
            stoi_protocol=StoiConfig()
        )


class RndvocPublishedConfigurationTest(unittest.TestCase):
    # Verifies the Andong-Li 22.05 kHz LJSpeech configuration values. These
    # are transcribed from the published recipe, so each assertion is part of
    # what this study claims to reproduce. The transform and mel geometry
    # additionally carries structural weight beyond citation, since the
    # decomposition and the band partition are both derived from it.
    def setUp(self) -> None:
        # Reads the published recipe record under test.
        self._configuration: RndvocConfig = RndvocConfig.andong_22k()

    def test_transform_geometry_matches_the_published_recipe(self) -> None:
        # The published recipe analyzes at 1024 points with hop 256 at 22.05 kHz.
        self.assertEqual(self._configuration.sample_rate, 22050)
        self.assertEqual(self._configuration.n_fft, 1024)
        self.assertEqual(self._configuration.hop_size, 256)
        self.assertEqual(self._configuration.win_size, 1024)

    def test_mel_band_geometry_matches_the_published_recipe(self) -> None:
        # Conditioning uses eighty bands between zero and eight kilohertz.
        # This geometry is what defines the decomposition: eighty bands
        # against a 513-bin spectrum is what makes the mel map rank-deficient,
        # and the upper cutoff means content above it lies entirely in the
        # null space, since no filter responds to it at all.
        self.assertEqual(self._configuration.input_mel_channels, 80)
        self.assertEqual(self._configuration.fmin, 0.0)
        self.assertEqual(self._configuration.fmax, 8000.0)

    def test_null_space_decoder_matches_the_published_recipe(self) -> None:
        # Six refinement stages of two repeats each form the published null-space decoder.
        self.assertEqual(self._configuration.null_stage_count, 6)
        self.assertEqual(self._configuration.repeat_count, 2)
        self.assertEqual(self._configuration.input_dimension, 256)
        self.assertEqual(self._configuration.squeeze_dimension, 64)
        self.assertEqual(self._configuration.hidden_dimension, 256)
        self.assertEqual(self._configuration.kernel_size, 7)

    def test_optimization_defaults_match_the_published_recipe(self) -> None:
        # Learning rate, Adam moments, and the per-epoch decay follow the reference settings.
        self.assertEqual(self._configuration.learning_rate, 0.0002)
        self.assertEqual(self._configuration.adam_beta_1, 0.8)
        self.assertEqual(self._configuration.adam_beta_2, 0.99)
        self.assertEqual(self._configuration.learning_rate_decay, 0.999)
        self.assertEqual(self._configuration.gradient_clip_norm, 1000.0)

    def test_conditioning_and_reconstruction_protocols_differ(self) -> None:
        # The reconstruction protocol is a separate record from the
        # conditioning protocol, which is a distinguishing property of this
        # family: the other mel-conditioned architectures return one record
        # from both properties. The separation is forced rather than
        # stylistic, since the conditioning protocol is bound to the
        # filterbank the decomposition inverts and cannot be adjusted to suit
        # measurement.
        self.assertNotEqual(
            self._configuration.mel_protocol,
            self._configuration.reconstruction_mel_protocol,
            msg="The reconstruction mel protocol must not collapse onto the conditioning protocol"
        )

    def test_configuration_is_immutable(self) -> None:
        # Frozen settings cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.null_stage_count = 1

    def test_unknown_field_is_rejected(self) -> None:
        # Extra fields are forbidden so silent typos cannot enter an experiment record.
        record: dict[str, object] = dict(self._configuration)
        record["unknown_setting"] = 1
        with self.assertRaises(ValidationError):
            RndvocConfig(**record)

    def test_published_record_round_trips_through_its_own_fields(self) -> None:
        # Control for the rejection above. Proving that the unperturbed record
        # rebuilds cleanly is what attributes that failure to the extra key
        # alone rather than to some unrelated reconstruction problem.
        self.assertEqual(RndvocConfig(**dict(self._configuration)), self._configuration)


class RndvocModuleSurfaceTest(unittest.TestCase):
    # Verifies module construction, the synthesis contract, and the exposed
    # protocol properties. The synthesis assertions derive their expected
    # lengths from the module's own configuration and mel protocol rather than
    # hard-coding them, so each states the frame-to-sample contract itself
    # instead of a number that would need editing whenever the reduced recipe
    # changed.
    def setUp(self) -> None:
        # Builds the miniature module and the synthetic reference waveform under a fixed seed.
        torch.manual_seed(1234)
        self._configuration: RndvocConfig = MiniatureModuleRecipe().build()
        self._module: Rndvoc = Rndvoc(self._configuration)
        self._waveform: torch.Tensor = torch.randn(1, 4096) * 0.1

    def test_automatic_optimization_is_disabled(self) -> None:
        # The adversarial recipe alternates the two optimizers by batch parity
        # from inside the training step, a schedule the loop-owned path has no
        # way to express, so manual optimization is required rather than
        # merely convenient.
        self.assertFalse(self._module.automatic_optimization)

    def test_generator_network_is_the_rndvoc_network(self) -> None:
        # The measurement stack and the optimizer split both reach for this attribute.
        self.assertIsInstance(self._module.network, RndvocNetwork)

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The module exposes exactly the record it was constructed with.
        self.assertIs(self._module.configuration, self._configuration)

    def test_mel_protocols_are_exposed_for_the_measurement_stack(self) -> None:
        # Conditioning and metric protocols are published separately to the metric components.
        self.assertEqual(self._module.mel_protocol, self._configuration.mel_protocol)
        self.assertEqual(self._module.metric_mel_protocol, self._configuration.reconstruction_mel_protocol)

    def test_synthesize_returns_the_flat_waveform_layout(self) -> None:
        # The inverse transform emits one flat sample row per batch element,
        # with no channel axis, unlike the families whose reconstruction head
        # introduces one. Asserting that the protocol entry point and the
        # module's forward pass return the identical tensor is what guarantees
        # the measurement stack and a direct caller observe the same synthesis
        # rather than two paths that could drift apart.
        mel: torch.Tensor = torch.randn(1, 80, 6)
        with torch.no_grad():
            synthesized: torch.Tensor = self._module.synthesize(mel)
            forwarded: torch.Tensor = self._module(mel)
        self.assertEqual(tuple(synthesized.shape), (1, (6 - 1) * self._configuration.hop_size))
        self.assertTrue(bool(torch.equal(synthesized, forwarded)))

    def test_prediction_returns_the_synthesized_waveform_mapping(self) -> None:
        # The measurement stack consumes the synthesized_waveform key.
        mel_transform: MelSpectrogram = MelSpectrogram(self._module.mel_protocol)
        with torch.no_grad():
            frame_count: int = int(mel_transform(self._waveform).shape[-1])
            prediction: dict[str, torch.Tensor] = self._module.predict_step({"waveform": self._waveform}, 0)
        self.assertIn("synthesized_waveform", prediction)
        self.assertEqual(
            tuple(prediction["synthesized_waveform"].shape),
            (1, (frame_count - 1) * self._configuration.hop_size)
        )

    def test_test_step_returns_the_prediction_output(self) -> None:
        # Test evaluation measures exactly the prediction path. Exact tensor
        # equality can be asserted here, rather than only matching shapes,
        # because this family's synthesis draws no randomness: two calls on
        # the same input produce identical output without any seed being
        # reset.
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._module.predict_step({"waveform": self._waveform}, 0)
            evaluation: dict[str, torch.Tensor] = self._module.test_step({"waveform": self._waveform}, 0)
        self.assertEqual(list(evaluation.keys()), list(prediction.keys()))
        self.assertTrue(
            bool(torch.equal(evaluation["synthesized_waveform"], prediction["synthesized_waveform"]))
        )

    def test_prediction_rejects_a_non_mapping_batch(self) -> None:
        # The batch contract is enforced before any synthesis happens.
        with self.assertRaisesRegex(TypeError, "Expected dict batch"):
            self._module.predict_step(self._waveform, 0)

    def test_prediction_rejects_a_non_tensor_waveform(self) -> None:
        # A missing or mistyped waveform entry fails loudly rather than synthesizing noise.
        with self.assertRaisesRegex(TypeError, "must be Tensor"):
            self._module.predict_step({"waveform": [0.0, 1.0]}, 0)


class RndvocOptimizationDeclarationTest(unittest.TestCase):
    # Verifies the paired generator and discriminator optimizers and their
    # exponential schedulers. The declaration is asserted rather than a
    # training run, because the training step reaches for the harness logging
    # surface and a trainer-attached epoch loop; what can be checked without a
    # trainer is that the declaration has the two-element list shape the
    # alternating step later indexes into by position.
    def setUp(self) -> None:
        # Builds the miniature module whose optimization declaration is under test.
        torch.manual_seed(1234)
        self._configuration: RndvocConfig = MiniatureModuleRecipe().build()
        self._module: Rndvoc = Rndvoc(self._configuration)

    def test_two_optimizers_and_two_schedulers_are_declared(self) -> None:
        # The adversarial recipe pairs one scheduler with each optimizer by position.
        declaration: OptimizationConfiguration = self._module.configure_optimizers()
        self.assertIsInstance(declaration.optimizer, list)
        self.assertIsInstance(declaration.scheduler, list)
        self.assertEqual(len(declaration.optimizer), 2)
        self.assertEqual(len(declaration.scheduler), 2)

    def test_generator_optimizer_covers_only_the_generator_parameters(self) -> None:
        # The first optimizer owns the generator, which is what the toggled
        # update relies on: the toggle restricts gradient tracking to the
        # stepping optimizer's parameter set, so an optimizer covering the
        # wrong parameters would silently train or freeze the wrong half.
        # Position matters as well as coverage, since the training step
        # unpacks the pair by index.
        declaration: OptimizationConfiguration = self._module.configure_optimizers()
        generator_optimizer: torch.optim.Optimizer = declaration.optimizer[0]
        self.assertEqual(
            len(generator_optimizer.param_groups[0]["params"]),
            len(list(self._module.network.parameters()))
        )

    def test_optimizers_carry_the_configured_learning_rate_and_moments(self) -> None:
        # Both optimizers start from the configured rate and Adam moment pair.
        declaration: OptimizationConfiguration = self._module.configure_optimizers()
        optimizer: torch.optim.Optimizer
        for optimizer in declaration.optimizer:
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], self._configuration.learning_rate)
            self.assertEqual(
                optimizer.param_groups[0]["betas"],
                (self._configuration.adam_beta_1, self._configuration.adam_beta_2)
            )

    def test_schedulers_decay_at_the_configured_gamma(self) -> None:
        # Both exponential schedulers share the reference per-epoch decay factor.
        declaration: OptimizationConfiguration = self._module.configure_optimizers()
        scheduler: torch.optim.lr_scheduler.ExponentialLR
        for scheduler in declaration.scheduler:
            self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
            self.assertEqual(scheduler.gamma, self._configuration.learning_rate_decay)
