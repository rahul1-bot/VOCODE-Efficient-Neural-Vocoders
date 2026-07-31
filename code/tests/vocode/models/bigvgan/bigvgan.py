# This module:
# 1. Verifies the BigVGAN configuration record: the published NVIDIA
#    base 24 kHz 100-band values, immutability, and the rejection of
#    unknown or loosely typed fields
# 2. Verifies the BigVGAN module surface: manual optimization, the
#    synthesis and prediction contracts, the protocol properties, and the
#    optimizer and scheduler declaration
#
# Design decisions:
# - The module assertions run on a miniature recipe that keeps the
#   published mel protocols but shrinks the generator and both
#   discriminator ensembles, because only the wiring is under test here
#   while the published generator topology is covered in the network tests
# - Training is never executed: training_step and validation_step call the
#   harness logging surface, which requires an attached trainer, so this
#   file asserts construction, prediction, and optimizer declaration only
# - Prediction lengths are derived from the module's own mel protocol
#   rather than hard-coded, so the assertion states the frame-to-sample
#   contract instead of a magic number
#
# Author: Rahul Sawhney

import math
import unittest

import torch
from pydantic import ValidationError

from syntheticmind.core.optimizer import OptimizationConfiguration

from vocode.metrics.pesq import PesqConfig
from vocode.metrics.stoi import StoiConfig
from vocode.models.bigvgan.bigvgan import Bigvgan, BigvganConfig
from vocode.models.bigvgan.network import BigvganNetwork
from vocode.transforms.mel import MelConfig, MelSpectrogram


class MiniatureModuleRecipe:
    # Builds the miniature BigVGAN configuration used by the module-surface assertions.
    # The generator and both ensembles are shrunk while the two mel protocols
    # are left at their published values, because the module's wiring is what
    # is under test here and that wiring is defined against the real protocols:
    # the conditioning band count, the hop, and the split between conditioning
    # and reconstruction all have to be the production ones for the assertions
    # to mean anything.
    def build(self) -> BigvganConfig:
        # Returns the reduced recipe, which keeps the published mel protocols only.
        # The mel band count stays at one hundred because the conditioning
        # protocol produces that many and the generator's entry convolution
        # would otherwise reject its own conditioning.
        #
        # Returns:
        #     A valid configuration whose generator expands each frame by four
        #     samples and whose ensembles hold one period and three reduced
        #     resolutions.
        return BigvganConfig(
            input_mel_channels=100,
            upsample_initial_channel=16,
            upsample_rates=(2, 2),
            upsample_kernel_sizes=(4, 4),
            resblock_kernel_sizes=(3,),
            resblock_dilation_sizes=((1, 3),),
            resblock="1",
            activation="snakebeta",
            snake_logscale=True,
            resolutions=((32, 8, 32), (64, 16, 64), (16, 4, 16)),
            mpd_reshapes=(2,),
            use_spectral_norm=False,
            discriminator_channel_multiplier=0.125,
            mel_protocol=MelConfig.bigvgan_nvidia_base_24khz_100band(),
            reconstruction_mel_protocol=MelConfig.bigvgan_nvidia_base_reconstruction_24khz_100band(),
            pesq_protocol=PesqConfig(),
            stoi_protocol=StoiConfig()
        )


class BigvganPublishedConfigurationTest(unittest.TestCase):
    # Verifies the published NVIDIA base 24 kHz 100-band configuration values.
    def setUp(self) -> None:
        # Reads the published recipe record under test.
        self._configuration: BigvganConfig = BigvganConfig.nvidia_base_24khz_100band()

    def test_conditioning_uses_one_hundred_mel_bands(self) -> None:
        # The base recipe conditions on the 100-band 24 kHz protocol.
        self.assertEqual(self._configuration.input_mel_channels, 100)
        self.assertEqual(self._configuration.mel_protocol.n_mels, 100)
        self.assertEqual(self._configuration.mel_protocol.sample_rate, 24000)

    def test_upsampling_rates_multiply_to_the_published_hop(self) -> None:
        # The four upsampling stages reconstruct 256 samples per conditioning frame.
        total_expansion: int = math.prod(self._configuration.upsample_rates)
        self.assertEqual(total_expansion, 256)
        self.assertEqual(self._configuration.upsample_rates, (8, 8, 2, 2))
        self.assertEqual(self._configuration.upsample_kernel_sizes, (16, 16, 4, 4))

    def test_residual_geometry_matches_the_published_recipe(self) -> None:
        # Three kernel sizes with the shared dilation triple define the residual stack.
        self.assertEqual(self._configuration.resblock_kernel_sizes, (3, 7, 11))
        self.assertEqual(self._configuration.resblock_dilation_sizes, ((1, 3, 5), (1, 3, 5), (1, 3, 5)))
        self.assertEqual(self._configuration.resblock, "1")

    def test_activation_is_snakebeta_with_log_scale_parameters(self) -> None:
        # The periodic activation is architecture identity rather than a tunable.
        self.assertEqual(self._configuration.activation, "snakebeta")
        self.assertTrue(self._configuration.snake_logscale)

    def test_discriminator_settings_match_the_published_recipe(self) -> None:
        # Five period views and three analysis resolutions form the adversarial ensembles.
        self.assertEqual(self._configuration.mpd_reshapes, (2, 3, 5, 7, 11))
        self.assertEqual(self._configuration.resolutions, ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240)))
        self.assertFalse(self._configuration.use_spectral_norm)
        self.assertEqual(self._configuration.discriminator_channel_multiplier, 1.0)

    def test_optimization_defaults_match_the_published_recipe(self) -> None:
        # Learning rate, Adam moments, and the exponential decay follow the reference settings.
        self.assertEqual(self._configuration.learning_rate, 0.0001)
        self.assertEqual(self._configuration.adam_beta_1, 0.8)
        self.assertEqual(self._configuration.adam_beta_2, 0.99)
        self.assertEqual(self._configuration.learning_rate_decay, 0.9999996)
        self.assertEqual(self._configuration.freeze_discriminator_steps, 0)

    def test_conditioning_and_reconstruction_protocols_differ(self) -> None:
        # The reconstruction protocol is a separate record from the conditioning protocol.
        self.assertNotEqual(
            self._configuration.mel_protocol,
            self._configuration.reconstruction_mel_protocol,
            msg="The reconstruction mel protocol must not collapse onto the conditioning protocol"
        )


class BigvganConfigurationValidationTest(unittest.TestCase):
    # Verifies the immutability and validation behavior of the configuration record.
    def setUp(self) -> None:
        # Reads the published record the single-field perturbations below start from.
        self._configuration: BigvganConfig = BigvganConfig.nvidia_base_24khz_100band()

    def test_configuration_is_immutable(self) -> None:
        # Frozen settings cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.learning_rate = 0.5

    def test_unknown_field_is_rejected(self) -> None:
        # Extra fields are forbidden so silent typos cannot enter an experiment record.
        record: dict[str, object] = dict(self._configuration)
        record["unknown_setting"] = 1
        with self.assertRaises(ValidationError):
            BigvganConfig(**record)

    def test_unlisted_residual_variant_is_rejected(self) -> None:
        # The residual variant is a closed literal vocabulary, not a free string.
        record: dict[str, object] = dict(self._configuration)
        record["resblock"] = "3"
        with self.assertRaises(ValidationError):
            BigvganConfig(**record)

    def test_published_record_round_trips_through_its_own_fields(self) -> None:
        # The rejections above perturb one field of a record that is otherwise valid.
        # Without this control, each rejection could be passing because the
        # reconstructed record was invalid for some unrelated reason; the round
        # trip establishes that the perturbed field is the only cause.
        self.assertEqual(BigvganConfig(**dict(self._configuration)), self._configuration)


class BigvganModuleSurfaceTest(unittest.TestCase):
    # Verifies module construction, the synthesis contract, and the exposed protocol properties.
    def setUp(self) -> None:
        # Builds the miniature module and the synthetic reference waveform under a fixed seed.
        torch.manual_seed(1234)
        self._recipe: MiniatureModuleRecipe = MiniatureModuleRecipe()
        self._configuration: BigvganConfig = self._recipe.build()
        self._module: Bigvgan = Bigvgan(self._configuration)
        self._waveform: torch.Tensor = torch.randn(1, 4096) * 0.1

    def test_automatic_optimization_is_disabled(self) -> None:
        # The adversarial recipe drives both optimizers manually.
        self.assertFalse(self._module.automatic_optimization)

    def test_generator_network_is_the_bigvgan_network(self) -> None:
        # The published weight adapter loads onto this attribute, so its type is contractual.
        self.assertIsInstance(self._module.network, BigvganNetwork)

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The module exposes exactly the record it was constructed with.
        self.assertIs(self._module.configuration, self._configuration)

    def test_mel_protocols_are_exposed_for_the_measurement_stack(self) -> None:
        # Conditioning and metric protocols are published separately to the metric components.
        self.assertEqual(self._module.mel_protocol, self._configuration.mel_protocol)
        self.assertEqual(self._module.metric_mel_protocol, self._configuration.reconstruction_mel_protocol)

    def test_synthesize_matches_the_forward_pass(self) -> None:
        # Synthesis routes through forward and expands every frame by the upsample product.
        mel: torch.Tensor = torch.randn(1, 100, 5)
        with torch.no_grad():
            synthesized: torch.Tensor = self._module.synthesize(mel)
            forwarded: torch.Tensor = self._module(mel)
        expected_samples: int = 5 * math.prod(self._configuration.upsample_rates)
        self.assertEqual(tuple(synthesized.shape), (1, 1, expected_samples))
        self.assertTrue(bool(torch.equal(synthesized, forwarded)))

    def test_prediction_returns_the_synthesized_waveform_mapping(self) -> None:
        # The measurement stack consumes the squeezed synthesized_waveform key.
        # The expected length is derived by running the module's own mel
        # protocol over the fixture and multiplying by this recipe's upsample
        # product, so the assertion states the frame-to-sample contract rather
        # than a constant that would have to be recomputed by hand whenever
        # the protocol or the rates change.
        mel_transform: MelSpectrogram = MelSpectrogram(self._module.mel_protocol)
        with torch.no_grad():
            frame_count: int = int(mel_transform(self._waveform).shape[-1])
            prediction: dict[str, torch.Tensor] = self._module.predict_step({"waveform": self._waveform}, 0)
        self.assertIn("synthesized_waveform", prediction)
        self.assertEqual(tuple(prediction["synthesized_waveform"].shape), (1, frame_count * 4))

    def test_test_step_returns_the_prediction_output(self) -> None:
        # Test evaluation measures exactly the prediction path.
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._module.predict_step({"waveform": self._waveform}, 0)
            evaluation: dict[str, torch.Tensor] = self._module.test_step({"waveform": self._waveform}, 0)
        self.assertEqual(
            tuple(evaluation["synthesized_waveform"].shape),
            tuple(prediction["synthesized_waveform"].shape)
        )

    def test_prediction_rejects_a_non_mapping_batch(self) -> None:
        # The batch contract is enforced before any synthesis happens.
        with self.assertRaisesRegex(TypeError, "Expected dict batch"):
            self._module.predict_step(self._waveform, 0)

    def test_prediction_rejects_a_non_tensor_waveform(self) -> None:
        # A missing or mistyped waveform entry fails loudly rather than synthesizing noise.
        with self.assertRaisesRegex(TypeError, "must be Tensor"):
            self._module.predict_step({"waveform": [0.0, 1.0]}, 0)


class BigvganOptimizationDeclarationTest(unittest.TestCase):
    # Verifies the paired generator and discriminator optimizers and their exponential schedulers.
    def setUp(self) -> None:
        # Builds the miniature module whose optimization declaration is under test.
        torch.manual_seed(1234)
        self._configuration: BigvganConfig = MiniatureModuleRecipe().build()
        self._module: Bigvgan = Bigvgan(self._configuration)

    def test_two_optimizers_and_two_schedulers_are_declared(self) -> None:
        # The adversarial recipe pairs one scheduler with each optimizer by position.
        declaration: OptimizationConfiguration = self._module.configure_optimizers()
        self.assertIsInstance(declaration.optimizer, list)
        self.assertIsInstance(declaration.scheduler, list)
        self.assertEqual(len(declaration.optimizer), 2)
        self.assertEqual(len(declaration.scheduler), 2)

    def test_generator_optimizer_covers_only_the_generator_parameters(self) -> None:
        # The first optimizer owns the generator, which is what the toggled update relies on.
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
        # Both exponential schedulers share the published per-step decay factor.
        # The factor is this close to unity precisely because the training step
        # advances both schedules once per optimizer step rather than once per
        # epoch; reading it as an epoch factor would understate the decay by
        # the number of steps in an epoch.
        declaration: OptimizationConfiguration = self._module.configure_optimizers()
        scheduler: torch.optim.lr_scheduler.ExponentialLR
        for scheduler in declaration.scheduler:
            self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
            self.assertEqual(scheduler.gamma, self._configuration.learning_rate_decay)
