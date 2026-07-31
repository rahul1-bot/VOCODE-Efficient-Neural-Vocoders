# This module:
# 1. Verifies the frozen MelGAN configuration: the Seungwon Park
#    reference recipe, its single mel protocol, its adversarial
#    hyperparameters, and the strict validation the record enforces
# 2. Verifies module construction: manual optimization, generator wiring
#    from the configuration, and the one protocol this family both
#    conditions and measures with
# 3. Verifies the evaluation surface: mel-to-waveform synthesis, the
#    prediction and test step contracts, and the paired Adam optimizer
#    declaration that carries no scheduler
#
# Design decisions:
# - Training and validation steps are not executed: both call the harness
#   metric log, which is only operational under an attached trainer, and
#   the training step additionally drives two optimizers; their input
#   guards and their fail-fast behavior without optimizers are verified
#   instead
# - Batches are fabricated as short synthetic waveforms rather than read
#   from the LJSpeech tree, so the suite carries no dataset dependency
# - Checkpoint configuration stamping is not verified for this family:
#   unlike HiFi-GAN, the module declares no checkpoint hooks of its own,
#   so there is no family-level stamp to assert
#
# Author: Rahul Sawhney

import math
import unittest

import torch
from pydantic import PositiveInt, ValidationError

from syntheticmind.core.optimizer import OptimizationConfiguration

from vocode.models.melgan.melgan import Melgan, MelganConfig
from vocode.models.melgan.network import MelganNetwork
from vocode.transforms.mel import MelConfig


class SyntheticWaveformBatch:
    # Builds the collated batch mapping the module reads, holding one
    # short synthetic waveform in place of an LJSpeech sample.
    def __init__(self, sample_count: int = 8192) -> None:
        # Binds the waveform length every emitted batch carries.
        self._sample_count: int = sample_count

    def batched(self) -> dict[str, torch.Tensor]:
        # Returns a collated record whose waveform carries a batch axis; the
        # amplitude scale keeps the material inside the waveform range.
        return {"waveform": torch.randn(1, self._sample_count) * 0.1}

    def unbatched(self) -> dict[str, torch.Tensor]:
        # Returns a record whose waveform carries no batch axis.
        return {"waveform": torch.randn(self._sample_count) * 0.1}

    @property
    def sample_count(self) -> int:
        # Returns the waveform length of every emitted record.
        return self._sample_count


class MelganConfigFieldPayload:
    # Reproduces the reference configuration as a keyword payload so
    # validation tests state only the single field under test.
    def __init__(self, configuration: MelganConfig) -> None:
        # Reads every declared field off the record, so a new recipe field is
        # carried into the payload without editing this builder.
        self._fields: dict[str, object] = {
            name: getattr(configuration, name) for name in type(configuration).model_fields
        }

    def valid(self) -> dict[str, object]:
        # Returns a copy of the accepted payload.
        return dict(self._fields)

    def with_field(self, field_name: str, field_value: object) -> dict[str, object]:
        # Returns the accepted payload with exactly one field replaced.
        mutated_fields: dict[str, object] = dict(self._fields)
        mutated_fields[field_name] = field_value
        return mutated_fields


class MelganReferenceRecipeTest(unittest.TestCase):
    # Verifies the named factory reproduces the Seungwon Park LJSpeech
    # recipe: generator topology, mel protocol, and Adam hyperparameters.
    def setUp(self) -> None:
        # Builds the single reference recipe this family publishes.
        self._configuration: MelganConfig = MelganConfig.seungwon()

    def test_recipe_declares_the_reference_generator_topology(self) -> None:
        # The reference generator is the lightweight four-stage upsampler.
        self.assertEqual(self._configuration.input_mel_channels, 80)
        self.assertEqual(self._configuration.ngf, 32)
        self.assertEqual(self._configuration.upsample_factors, (8, 8, 2, 2))
        self.assertEqual(self._configuration.leaky_relu_slope, 0.2)

    def test_recipe_upsamples_to_the_mel_hop_length(self) -> None:
        # The upsample factors multiply to the protocol hop length.
        upsampling_factor: int = math.prod(self._configuration.upsample_factors)
        self.assertEqual(upsampling_factor, self._configuration.mel_protocol.hop_length)

    def test_recipe_uses_the_seungwon_mel_protocol(self) -> None:
        # Conditioning follows the protocol the released checkpoint assumes.
        self.assertEqual(self._configuration.mel_protocol, MelConfig.melgan_seungwon())

    def test_recipe_declares_the_reference_adam_hyperparameters(self) -> None:
        # The reference recipe trains at a lower rate with lower betas than HiFi-GAN.
        self.assertEqual(self._configuration.learning_rate, 0.0001)
        self.assertEqual(self._configuration.adam_beta_1, 0.5)
        self.assertEqual(self._configuration.adam_beta_2, 0.9)


class MelganConfigValidationTest(unittest.TestCase):
    # Verifies the frozen configuration rejects loose types, non-positive
    # topology values, extra fields, and mutation.
    def setUp(self) -> None:
        # Builds the accepted reference payload each rejection case mutates once.
        self._payload: MelganConfigFieldPayload = MelganConfigFieldPayload(MelganConfig.seungwon())

    def test_configuration_rebuilds_from_its_own_field_values(self) -> None:
        # The payload used by the rejection cases is itself valid.
        self.assertEqual(MelganConfig(**self._payload.valid()), MelganConfig.seungwon())

    def test_configuration_rejects_a_string_mel_channel_count(self) -> None:
        # strict=True refuses a string where an integer is declared.
        with self.assertRaises(ValidationError):
            MelganConfig(**self._payload.with_field("input_mel_channels", "80"))

    def test_configuration_rejects_a_zero_growth_factor(self) -> None:
        # A generator with no channels cannot be built.
        with self.assertRaises(ValidationError):
            MelganConfig(**self._payload.with_field("ngf", 0))

    def test_configuration_rejects_a_negative_learning_rate(self) -> None:
        # A negative rate would invert every gradient step.
        with self.assertRaises(ValidationError):
            MelganConfig(**self._payload.with_field("learning_rate", -0.0001))

    def test_configuration_rejects_an_extra_field(self) -> None:
        # extra="forbid" blocks undeclared recipe knobs.
        with self.assertRaises(ValidationError):
            MelganConfig(**self._payload.with_field("spectral_loss_weight", 1.0))

    def test_configuration_rejects_mutation_after_construction(self) -> None:
        # frozen=True keeps a recipe fixed for the whole run.
        configuration: MelganConfig = MelganConfig.seungwon()
        with self.assertRaises(ValidationError):
            configuration.ngf: PositiveInt = 64


class MelganModuleConstructionTest(unittest.TestCase):
    # Verifies the module disables automatic optimization, wires the
    # generator from its configuration, and publishes one mel protocol.
    def setUp(self) -> None:
        # Builds one seeded module at the reference recipe.
        torch.manual_seed(0)
        self._configuration: MelganConfig = MelganConfig.seungwon()
        self._module: Melgan = Melgan(self._configuration)

    def test_module_disables_automatic_optimization(self) -> None:
        # The adversarial recipe drives two optimizers manually.
        self.assertFalse(self._module.automatic_optimization)

    def test_module_exposes_the_generator_network(self) -> None:
        # The network member is the synthesis chain the metrics consume.
        self.assertIsInstance(self._module.network, MelganNetwork)

    def test_module_returns_the_configuration_it_was_built_with(self) -> None:
        # The immutable configuration travels with the module.
        self.assertIs(self._module.configuration, self._configuration)

    def test_module_conditions_and_measures_with_one_protocol(self) -> None:
        # MelGAN declares a single mel protocol for both roles.
        self.assertEqual(self._module.mel_protocol, self._configuration.mel_protocol)
        self.assertEqual(self._module.metric_mel_protocol, self._module.mel_protocol)

    def test_generator_width_follows_the_growth_factor(self) -> None:
        # The configured growth factor reaches the constructed network.
        narrow_module: Melgan = Melgan(
            self._configuration.model_copy(update={"ngf": 8})
        )
        narrow_parameters: int = sum(
            parameter.numel() for parameter in narrow_module.network.parameters()
        )
        reference_parameters: int = sum(
            parameter.numel() for parameter in self._module.network.parameters()
        )
        self.assertLess(narrow_parameters, reference_parameters)


class MelganSynthesisTest(unittest.TestCase):
    # Verifies mel-to-waveform synthesis through the module surface the
    # metric and profiling components call.
    def setUp(self) -> None:
        # Builds the module in evaluation mode with one sixteen-frame mel.
        torch.manual_seed(0)
        self._module: Melgan = Melgan(MelganConfig.seungwon()).eval()
        self._mel: torch.Tensor = torch.randn(1, 80, 16)

    def test_synthesis_upsamples_the_mel_by_the_hop_length(self) -> None:
        # Each mel frame becomes one hop of waveform samples.
        with torch.no_grad():
            waveform: torch.Tensor = self._module.synthesize(self._mel)
        self.assertEqual(tuple(waveform.shape), (1, 1, 16 * 256))

    def test_synthesis_matches_the_forward_pass(self) -> None:
        # synthesize is the protocol name for the generator forward.
        with torch.no_grad():
            synthesized: torch.Tensor = self._module.synthesize(self._mel)
            forwarded: torch.Tensor = self._module(self._mel)
        self.assertTrue(bool(torch.equal(synthesized, forwarded)))

    def test_synthesis_returns_a_bounded_finite_waveform(self) -> None:
        # The tanh output bound holds for every synthesized sample.
        with torch.no_grad():
            waveform: torch.Tensor = self._module.synthesize(self._mel)
        self.assertEqual(waveform.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(waveform).all()))
        self.assertGreaterEqual(float(waveform.min()), -1.0)
        self.assertLessEqual(float(waveform.max()), 1.0)

    def test_synthesis_accepts_a_batch_of_two(self) -> None:
        # Synthesis is batch-agnostic along the leading dimension.
        with torch.no_grad():
            waveform: torch.Tensor = self._module.synthesize(torch.randn(2, 80, 8))
        self.assertEqual(tuple(waveform.shape), (2, 1, 8 * 256))


class MelganEvaluationStepTest(unittest.TestCase):
    # Verifies the prediction and test step contracts and the input
    # guards every step method applies to the collated batch.
    def setUp(self) -> None:
        # Builds the module detached from any trainer and its batch source.
        torch.manual_seed(0)
        self._module: Melgan = Melgan(MelganConfig.seungwon()).eval()
        self._batch_builder: SyntheticWaveformBatch = SyntheticWaveformBatch()

    def test_prediction_returns_the_synthesized_waveform_mapping(self) -> None:
        # The measurement stack reads synthesis under this exact key.
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._module.predict_step(
                self._batch_builder.batched(),
                0
            )
        self.assertIn("synthesized_waveform", prediction)
        self.assertIsInstance(prediction["synthesized_waveform"], torch.Tensor)

    def test_prediction_returns_a_two_dimensional_waveform(self) -> None:
        # The channel axis is squeezed out for the metric panel.
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._module.predict_step(
                self._batch_builder.batched(),
                0
            )
        waveform: torch.Tensor = prediction["synthesized_waveform"]
        self.assertEqual(waveform.ndim, 2)
        self.assertEqual(waveform.shape[0], 1)
        self.assertTrue(bool(torch.isfinite(waveform).all()))

    def test_prediction_length_follows_the_centered_mel_frame_count(self) -> None:
        # The centered protocol yields one frame past the final hop.
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._module.predict_step(
                self._batch_builder.batched(),
                0
            )
        hop_length: int = self._module.mel_protocol.hop_length
        expected_samples: int = (self._batch_builder.sample_count // hop_length + 1) * hop_length
        self.assertEqual(prediction["synthesized_waveform"].shape[-1], expected_samples)

    def test_prediction_accepts_an_unbatched_waveform(self) -> None:
        # A single-sample waveform yields a batch of one after extraction.
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._module.predict_step(
                self._batch_builder.unbatched(),
                0
            )
        self.assertEqual(prediction["synthesized_waveform"].shape[0], 1)

    def test_test_step_returns_the_prediction_contract(self) -> None:
        # Test evaluation measures exactly the prediction path.
        with torch.no_grad():
            step_output: dict[str, torch.Tensor] = self._module.test_step(
                self._batch_builder.batched(),
                0
            )
        self.assertEqual(list(step_output.keys()), ["synthesized_waveform"])
        self.assertIsInstance(step_output["synthesized_waveform"], torch.Tensor)

    def test_every_step_rejects_a_non_mapping_batch(self) -> None:
        # The collated batch contract is a mapping, not a sequence.
        with self.assertRaisesRegex(TypeError, "Expected dict batch"):
            self._module.predict_step([1, 2, 3], 0)
        with self.assertRaisesRegex(TypeError, "Expected dict batch"):
            self._module.validation_step([1, 2, 3], 0)
        with self.assertRaisesRegex(TypeError, "Expected dict batch"):
            self._module.training_step([1, 2, 3], 0)

    def test_prediction_rejects_a_batch_without_a_waveform(self) -> None:
        # A missing waveform key fails loudly rather than synthesizing noise.
        with self.assertRaisesRegex(TypeError, "waveform"):
            self._module.predict_step({}, 0)

    def test_prediction_rejects_a_non_tensor_waveform(self) -> None:
        # The waveform entry must be a tensor, not an audio path.
        with self.assertRaisesRegex(TypeError, "waveform"):
            self._module.predict_step({"waveform": "sample.wav"}, 0)

    def test_training_step_requires_the_optimizer_pair(self) -> None:
        # Without an attached trainer the adversarial step fails fast.
        with self.assertRaisesRegex(RuntimeError, "generator and discriminator optimizers"):
            self._module.training_step(self._batch_builder.batched(), 0)


class MelganOptimizerDeclarationTest(unittest.TestCase):
    # Verifies the paired Adam optimizers carry the configured
    # hyperparameters and that the reference recipe declares no scheduler.
    def setUp(self) -> None:
        # Collects the declaration as a record; no optimizer is ever stepped.
        torch.manual_seed(0)
        self._configuration: MelganConfig = MelganConfig.seungwon()
        self._module: Melgan = Melgan(self._configuration)
        self._declaration: OptimizationConfiguration = self._module.configure_optimizers()

    def test_declaration_pairs_two_adam_optimizers(self) -> None:
        # The generator and discriminator each own an Adam optimizer.
        self.assertIsInstance(self._declaration.optimizer, list)
        self.assertEqual(len(self._declaration.optimizer), 2)
        optimizer: torch.optim.Optimizer
        for optimizer in self._declaration.optimizer:
            self.assertIsInstance(optimizer, torch.optim.Adam)

    def test_both_optimizers_carry_the_configured_rate_and_betas(self) -> None:
        # Rate and betas come from the frozen configuration, not defaults.
        optimizer: torch.optim.Optimizer
        for optimizer in self._declaration.optimizer:
            self.assertEqual(optimizer.param_groups[0]["lr"], self._configuration.learning_rate)
            self.assertEqual(
                optimizer.param_groups[0]["betas"],
                (self._configuration.adam_beta_1, self._configuration.adam_beta_2)
            )

    def test_the_first_optimizer_owns_exactly_the_generator_parameters(self) -> None:
        # Declaration order is generator first, discriminator second.
        declared_parameters: int = sum(
            parameter.numel()
            for group in self._declaration.optimizer[0].param_groups
            for parameter in group["params"]
        )
        network_parameters: int = sum(
            parameter.numel() for parameter in self._module.network.parameters()
        )
        self.assertEqual(declared_parameters, network_parameters)

    def test_the_second_optimizer_excludes_the_generator_parameters(self) -> None:
        # The discriminator optimizer must never step the generator.
        declared_parameters: int = sum(
            parameter.numel()
            for group in self._declaration.optimizer[1].param_groups
            for parameter in group["params"]
        )
        network_parameters: int = sum(
            parameter.numel() for parameter in self._module.network.parameters()
        )
        module_parameters: int = sum(
            parameter.numel() for parameter in self._module.parameters()
        )
        self.assertEqual(declared_parameters, module_parameters - network_parameters)

    def test_declaration_carries_no_scheduler(self) -> None:
        # The reference MelGAN recipe holds the rate constant.
        self.assertIsNone(self._declaration.scheduler)


if __name__ == "__main__":
    unittest.main()
