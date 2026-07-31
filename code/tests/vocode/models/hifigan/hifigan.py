# This module:
# 1. Verifies the frozen HiFi-GAN configuration: the V1, V2, and V3
#    reference recipes, the project half-width control, and the strict
#    validation the record enforces
# 2. Verifies module construction: manual optimization, generator wiring
#    from the configuration, and the two distinct mel protocols the family
#    conditions and measures with
# 3. Verifies the evaluation surface: mel-to-waveform synthesis, the
#    prediction and test step contracts, the paired optimizer and
#    scheduler declaration, and the checkpoint configuration stamp
#
# Design decisions:
# - Training and validation steps are not executed: both call the harness
#   metric log, which is only operational under an attached trainer, and
#   the training step additionally drives two optimizers; their input
#   guards and their fail-fast behavior without optimizers are verified
#   instead, and the fitted paths remain out of scope for this suite
# - Batches are fabricated as short synthetic waveforms rather than read
#   from the LJSpeech tree, so the suite carries no dataset dependency
# - Optimizer and scheduler declarations are inspected as records, never
#   stepped, because a step is training
#
# Author: Rahul Sawhney

import math
import unittest

import torch
from pydantic import PositiveFloat, ValidationError

from syntheticmind.core.optimizer import OptimizationConfiguration, SchedulerConfig

from vocode.models.hifigan.hifigan import Hifigan, HifiganConfig
from vocode.models.hifigan.network import HifiganNetwork
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


class HifiganConfigFieldPayload:
    # Reproduces a reference configuration as a keyword payload so
    # validation tests state only the single field under test.
    def __init__(self, configuration: HifiganConfig) -> None:
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


class HifiganReferenceRecipeTest(unittest.TestCase):
    # Verifies the named factories reproduce the published V1, V2, and V3
    # topologies and the project-defined half-width control.
    def test_v1_declares_the_published_full_width_topology(self) -> None:
        # V1 is the full-width generator with the three-dilation resblock.
        configuration: HifiganConfig = HifiganConfig.v1()
        self.assertEqual(configuration.input_mel_channels, 80)
        self.assertEqual(configuration.upsample_initial_channels, 512)
        self.assertEqual(configuration.upsample_rates, (8, 8, 2, 2))
        self.assertEqual(configuration.upsample_kernel_sizes, (16, 16, 4, 4))
        self.assertEqual(configuration.resblock_kernel_sizes, (3, 7, 11))
        self.assertEqual(configuration.resblock_kind, "1")
        self.assertEqual(configuration.channel_multiplier, 1.0)

    def test_v2_narrows_v1_without_changing_the_topology(self) -> None:
        # V2 differs from V1 only in generator width.
        first_configuration: HifiganConfig = HifiganConfig.v1()
        second_configuration: HifiganConfig = HifiganConfig.v2()
        self.assertEqual(second_configuration.upsample_initial_channels, 128)
        self.assertEqual(second_configuration.upsample_rates, first_configuration.upsample_rates)
        self.assertEqual(second_configuration.resblock_kernel_sizes, first_configuration.resblock_kernel_sizes)
        self.assertEqual(second_configuration.resblock_kind, first_configuration.resblock_kind)

    def test_v3_uses_three_stages_and_the_light_resblock(self) -> None:
        # V3 reaches the same hop in three stages with two-dilation blocks.
        configuration: HifiganConfig = HifiganConfig.v3()
        self.assertEqual(configuration.upsample_initial_channels, 256)
        self.assertEqual(configuration.upsample_rates, (8, 8, 4))
        self.assertEqual(configuration.upsample_kernel_sizes, (16, 16, 8))
        self.assertEqual(configuration.resblock_kernel_sizes, (3, 5, 7))
        self.assertEqual(configuration.resblock_dilation_sizes, ((1, 2), (2, 6), (3, 12)))
        self.assertEqual(configuration.resblock_kind, "2")

    def test_every_recipe_upsamples_to_the_reference_hop(self) -> None:
        # All three recipes multiply their rates to the mel hop length.
        configuration: HifiganConfig
        for configuration in (HifiganConfig.v1(), HifiganConfig.v2(), HifiganConfig.v3()):
            upsampling_factor: int = math.prod(configuration.upsample_rates)
            self.assertEqual(upsampling_factor, configuration.conditioning_mel_protocol.hop_length)

    def test_half_width_control_differs_from_v1_only_in_multiplier(self) -> None:
        # The project control halves capacity without touching the recipe.
        reference_configuration: HifiganConfig = HifiganConfig.v1()
        control_configuration: HifiganConfig = HifiganConfig.half_width_v1()
        self.assertEqual(control_configuration.channel_multiplier, 0.5)
        self.assertEqual(
            control_configuration.model_copy(update={"channel_multiplier": 1.0}),
            reference_configuration
        )

    def test_every_recipe_shares_the_family_optimizer_defaults(self) -> None:
        # Learning rate, Adam betas, and decay are fleet-uniform.
        configuration: HifiganConfig
        for configuration in (HifiganConfig.v1(), HifiganConfig.v2(), HifiganConfig.v3()):
            self.assertEqual(configuration.learning_rate, 0.0002)
            self.assertEqual(configuration.adam_beta_1, 0.8)
            self.assertEqual(configuration.adam_beta_2, 0.99)
            self.assertEqual(configuration.learning_rate_decay, 0.999)

    def test_conditioning_and_reconstruction_protocols_are_distinct(self) -> None:
        # Conditioning is band-limited while reconstruction is full-band.
        configuration: HifiganConfig = HifiganConfig.v1()
        self.assertEqual(configuration.conditioning_mel_protocol, MelConfig.hifigan_conditioning())
        self.assertEqual(configuration.reconstruction_mel_protocol, MelConfig.hifigan_reconstruction())
        self.assertNotEqual(
            configuration.conditioning_mel_protocol,
            configuration.reconstruction_mel_protocol
        )


class HifiganConfigValidationTest(unittest.TestCase):
    # Verifies the frozen configuration rejects loose types, non-positive
    # topology values, extra fields, and mutation.
    def setUp(self) -> None:
        # Builds the accepted V1 payload each rejection case mutates once.
        self._payload: HifiganConfigFieldPayload = HifiganConfigFieldPayload(HifiganConfig.v1())

    def test_configuration_rebuilds_from_its_own_field_values(self) -> None:
        # The payload used by the rejection cases is itself valid.
        self.assertEqual(HifiganConfig(**self._payload.valid()), HifiganConfig.v1())

    def test_configuration_rejects_a_string_mel_channel_count(self) -> None:
        # strict=True refuses a string where an integer is declared.
        with self.assertRaises(ValidationError):
            HifiganConfig(**self._payload.with_field("input_mel_channels", "80"))

    def test_configuration_rejects_a_zero_mel_channel_count(self) -> None:
        # A generator conditioned on zero bands cannot be built.
        with self.assertRaises(ValidationError):
            HifiganConfig(**self._payload.with_field("input_mel_channels", 0))

    def test_configuration_rejects_a_negative_learning_rate(self) -> None:
        # A negative rate would invert every gradient step.
        with self.assertRaises(ValidationError):
            HifiganConfig(**self._payload.with_field("learning_rate", -0.0002))

    def test_configuration_rejects_an_unsupported_resblock_kind(self) -> None:
        # The residual-block vocabulary is closed to the two variants.
        with self.assertRaises(ValidationError):
            HifiganConfig(**self._payload.with_field("resblock_kind", "3"))

    def test_configuration_rejects_an_extra_field(self) -> None:
        # extra="forbid" blocks undeclared recipe knobs.
        with self.assertRaises(ValidationError):
            HifiganConfig(**self._payload.with_field("gradient_clip", 1.0))

    def test_configuration_rejects_mutation_after_construction(self) -> None:
        # frozen=True keeps a recipe fixed for the whole run.
        configuration: HifiganConfig = HifiganConfig.v1()
        with self.assertRaises(ValidationError):
            configuration.learning_rate: PositiveFloat = 0.001


class HifiganModuleConstructionTest(unittest.TestCase):
    # Verifies the module disables automatic optimization, wires the
    # generator from its configuration, and publishes both mel protocols.
    def setUp(self) -> None:
        # Builds the narrowest reference recipe, which is the cheapest to construct.
        torch.manual_seed(0)
        self._configuration: HifiganConfig = HifiganConfig.v3()
        self._module: Hifigan = Hifigan(self._configuration)

    def test_module_disables_automatic_optimization(self) -> None:
        # The adversarial recipe drives two optimizers manually.
        self.assertFalse(self._module.automatic_optimization)

    def test_module_exposes_the_generator_network(self) -> None:
        # The network member is the synthesis chain the metrics consume.
        self.assertIsInstance(self._module.network, HifiganNetwork)

    def test_module_returns_the_configuration_it_was_built_with(self) -> None:
        # The immutable configuration travels with the module.
        self.assertIs(self._module.configuration, self._configuration)

    def test_module_conditions_on_the_band_limited_protocol(self) -> None:
        # Generator conditioning uses the band-limited HiFi-GAN protocol.
        self.assertEqual(self._module.mel_protocol, self._configuration.conditioning_mel_protocol)
        self.assertEqual(self._module.mel_protocol.fmax, 8000.0)

    def test_module_measures_with_the_full_band_protocol(self) -> None:
        # The mel-error metric extracts with the reconstruction protocol.
        self.assertEqual(
            self._module.metric_mel_protocol,
            self._configuration.reconstruction_mel_protocol
        )
        self.assertIsNone(self._module.metric_mel_protocol.fmax)

    def test_generator_width_follows_the_channel_multiplier(self) -> None:
        # The half-width control reaches the constructed network.
        half_width_module: Hifigan = Hifigan(HifiganConfig.half_width_v1())
        full_width_module: Hifigan = Hifigan(HifiganConfig.v1())
        half_width_parameters: int = sum(
            parameter.numel() for parameter in half_width_module.network.parameters()
        )
        full_width_parameters: int = sum(
            parameter.numel() for parameter in full_width_module.network.parameters()
        )
        self.assertLess(half_width_parameters, full_width_parameters)


class HifiganSynthesisTest(unittest.TestCase):
    # Verifies mel-to-waveform synthesis through the module surface the
    # metric and profiling components call.
    def setUp(self) -> None:
        # Builds the V3 module in evaluation mode with one sixteen-frame mel.
        torch.manual_seed(0)
        self._module: Hifigan = Hifigan(HifiganConfig.v3()).eval()
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

    def test_every_reference_recipe_synthesizes_the_same_waveform_length(self) -> None:
        # V1, V2, and V3 share the hop length despite differing topologies.
        configuration: HifiganConfig
        for configuration in (HifiganConfig.v1(), HifiganConfig.v2(), HifiganConfig.v3()):
            module: Hifigan = Hifigan(configuration).eval()
            with torch.no_grad():
                waveform: torch.Tensor = module.synthesize(torch.randn(1, 80, 8))
            self.assertEqual(
                tuple(waveform.shape),
                (1, 1, 8 * 256),
                msg=f"recipe with {configuration.upsample_initial_channels} channels changed the hop"
            )


class HifiganEvaluationStepTest(unittest.TestCase):
    # Verifies the prediction and test step contracts and the input
    # guards every step method applies to the collated batch.
    def setUp(self) -> None:
        # Builds the V3 module detached from any trainer and its batch source.
        torch.manual_seed(0)
        self._module: Hifigan = Hifigan(HifiganConfig.v3()).eval()
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

    def test_prediction_accepts_an_unbatched_waveform(self) -> None:
        # A single-sample waveform is promoted to a batch of one.
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

    def test_epoch_end_requires_the_scheduler_pair(self) -> None:
        # Both exponential schedulers must exist before stepping them.
        with self.assertRaisesRegex(RuntimeError, "generator and discriminator schedulers"):
            self._module.on_train_epoch_end()


class HifiganOptimizerDeclarationTest(unittest.TestCase):
    # Verifies the paired AdamW optimizers and their paired per-epoch
    # exponential schedulers carry the configured hyperparameters.
    def setUp(self) -> None:
        # Collects the declaration as a record; no optimizer is ever stepped.
        torch.manual_seed(0)
        self._configuration: HifiganConfig = HifiganConfig.v3()
        self._module: Hifigan = Hifigan(self._configuration)
        self._declaration: OptimizationConfiguration = self._module.configure_optimizers()

    def test_declaration_pairs_two_optimizers(self) -> None:
        # The generator and discriminator each own an optimizer.
        self.assertIsInstance(self._declaration.optimizer, list)
        self.assertEqual(len(self._declaration.optimizer), 2)

    def test_both_optimizers_are_adamw(self) -> None:
        # The reference recipe optimizes with decoupled weight decay.
        optimizer: torch.optim.Optimizer
        for optimizer in self._declaration.optimizer:
            self.assertIsInstance(optimizer, torch.optim.AdamW)

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
        # Declaration order is generator first, discriminators second.
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

    def test_declaration_pairs_two_epoch_exponential_schedulers(self) -> None:
        # Both schedulers decay once per epoch at the configured gamma.
        self.assertIsInstance(self._declaration.scheduler, list)
        self.assertEqual(len(self._declaration.scheduler), 2)
        scheduler: SchedulerConfig
        for scheduler in self._declaration.scheduler:
            self.assertEqual(scheduler.name, "exponential")
            self.assertEqual(scheduler.interval, "epoch")
            self.assertEqual(scheduler.gamma, self._configuration.learning_rate_decay)


class HifiganCheckpointStampTest(unittest.TestCase):
    # Verifies the checkpoint hooks stamp the configuration dump and
    # refuse a resume against a mismatched configuration.
    def setUp(self) -> None:
        # Builds the V3 module whose recipe the checkpoint stamp must record.
        torch.manual_seed(0)
        self._configuration: HifiganConfig = HifiganConfig.v3()
        self._module: Hifigan = Hifigan(self._configuration)

    def test_saving_stamps_the_configuration_dump(self) -> None:
        # Reproducibility metadata travels inside the checkpoint payload.
        checkpoint: dict[str, object] = {}
        self._module.on_save_checkpoint(checkpoint)
        self.assertEqual(
            checkpoint["hifigan_configuration"],
            self._configuration.model_dump(mode="json")
        )

    def test_loading_accepts_a_matching_configuration(self) -> None:
        # A resume against the same recipe proceeds and changes nothing: the
        # hook is a compatibility gate, never a configuration restore path.
        checkpoint: dict[str, object] = {}
        self._module.on_save_checkpoint(checkpoint)
        self._module.on_load_checkpoint(checkpoint)
        self.assertIs(self._module.configuration, self._configuration)
        self.assertEqual(
            checkpoint["hifigan_configuration"],
            self._configuration.model_dump(mode="json")
        )

    def test_loading_accepts_a_checkpoint_without_the_stamp(self) -> None:
        # An unstamped checkpoint predates the hook; the guard neither rejects
        # it nor backfills a stamp into it.
        unstamped_checkpoint: dict[str, object] = {}
        self._module.on_load_checkpoint(unstamped_checkpoint)
        self.assertEqual(unstamped_checkpoint, {})
        self.assertIs(self._module.configuration, self._configuration)

    def test_loading_rejects_a_mismatched_configuration(self) -> None:
        # A resume against a different recipe would corrupt the evidence.
        foreign_checkpoint: dict[str, object] = {
            "hifigan_configuration": HifiganConfig.v1().model_dump(mode="json")
        }
        with self.assertRaisesRegex(ValueError, "does not match"):
            self._module.on_load_checkpoint(foreign_checkpoint)

    def test_the_stamp_distinguishes_the_half_width_control(self) -> None:
        # The control and its reference must not resume from each other.
        control_module: Hifigan = Hifigan(HifiganConfig.half_width_v1())
        control_checkpoint: dict[str, object] = {}
        control_module.on_save_checkpoint(control_checkpoint)
        reference_module: Hifigan = Hifigan(HifiganConfig.v1())
        with self.assertRaises(ValueError):
            reference_module.on_load_checkpoint(control_checkpoint)


if __name__ == "__main__":
    unittest.main()
