# This module:
# 1. Verifies the VocosFormer configuration record: the trimmed backbone and
#    position-network overrides, and that every inherited Vocos training
#    field is carried through unchanged
# 2. Verifies the controlled-variable design of the module: the generator is
#    the attention-augmented network while the discriminators, loss, and
#    optimization machinery remain the reproduced Vocos ones
# 3. Verifies that the inherited inference surface synthesizes on the Vocos
#    conditioning protocol
#
# Design decisions:
# - The controlled-variable claim is tested through the non-generator
#   parameter budget: the module total minus the generator total must be
#   identical for VocosFormer and Vocos, which proves the surrounding
#   machinery is untouched without reaching into private attributes
# - Inference runs on one 4096-sample synthetic waveform with the module
#   detached from any trainer, keeping the whole file well under a second of
#   compute per test
# - Training and validation steps are not executed here: they are inherited
#   byte-identical from Vocos and are covered by the Vocos module tests
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import PositiveInt, ValidationError

from vocode.models.vocos.network import VocosNetwork
from vocode.models.vocos.vocos import Vocos, VocosConfig
from vocode.models.vocosformer.network import VocosformerNetwork
from vocode.models.vocosformer.vocosformer import Vocosformer, VocosformerConfig


class ReferenceBatchBuilder:
    # Produces deterministic LJSpeech-shaped batch mappings at the Vocos conditioning rate.
    def __init__(self, sample_count: int, sample_rate: int) -> None:
        # Binds the waveform length and corpus rate this builder stamps on every batch.
        self._sample_count: int = sample_count
        self._sample_rate: int = sample_rate

    def build(self, seed: int) -> dict[str, torch.Tensor | int]:
        # Seeds the generator and returns one batched reference record.
        torch.manual_seed(seed)
        return {"waveform": torch.randn(1, self._sample_count), "sample_rate": self._sample_rate}

    @property
    def sample_count(self) -> int:
        # Returns the waveform length of every emitted record.
        return self._sample_count


class VocosformerConfigurationTest(unittest.TestCase):
    # Verifies that the configuration overrides only the backbone and position-network fields.
    def setUp(self) -> None:
        # Builds the adaptation record and the Vocos baseline record it is compared against.
        self._configuration: VocosformerConfig = VocosformerConfig()
        self._baseline: VocosConfig = VocosConfig()

    def test_configuration_extends_the_vocos_record(self) -> None:
        # Inheriting the Vocos record is what keeps the training recipe shared.
        self.assertIsInstance(self._configuration, VocosConfig)

    def test_backbone_and_position_fields_are_overridden(self) -> None:
        # The trimmed stack and the position network are the declared deltas.
        self.assertEqual(self._configuration.intermediate_dimension, 1344)
        self.assertEqual(self._configuration.layer_count, 4)
        self.assertEqual(self._configuration.position_group_count, 32)
        self.assertAlmostEqual(self._configuration.position_dropout, 0.1)

    def test_training_recipe_is_inherited_unchanged(self) -> None:
        # Optimizer, schedule, and loss settings must not differ from the baseline row.
        self.assertAlmostEqual(self._configuration.learning_rate, self._baseline.learning_rate)
        self.assertAlmostEqual(self._configuration.adam_beta_1, self._baseline.adam_beta_1)
        self.assertAlmostEqual(self._configuration.adam_beta_2, self._baseline.adam_beta_2)
        self.assertAlmostEqual(self._configuration.gradient_clip_norm, self._baseline.gradient_clip_norm)
        self.assertEqual(self._configuration.scheduler_total_steps, self._baseline.scheduler_total_steps)
        self.assertEqual(self._configuration.loss_configuration, self._baseline.loss_configuration)

    def test_conditioning_protocol_is_inherited_unchanged(self) -> None:
        # Both rows condition on the same mel protocol, so the comparison stays controlled.
        self.assertEqual(self._configuration.mel_protocol, self._baseline.mel_protocol)
        self.assertEqual(self._configuration.input_channels, self._baseline.input_channels)
        self.assertEqual(self._configuration.hidden_dimension, self._baseline.hidden_dimension)
        self.assertEqual(self._configuration.n_fft, self._baseline.n_fft)
        self.assertEqual(self._configuration.hop_length, self._baseline.hop_length)

    def test_matched_factory_returns_the_default_record(self) -> None:
        # The named factory is the documented entry point to the matched recipe.
        self.assertEqual(VocosformerConfig.matched_vocos_24khz(), self._configuration)

    def test_configuration_record_is_frozen(self) -> None:
        # Recipe fields cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.layer_count: PositiveInt = 8

    def test_unknown_field_is_rejected(self) -> None:
        # Extra fields are forbidden so a misspelled experiment setting fails loudly.
        with self.assertRaises(ValidationError):
            VocosformerConfig(position_dropouts=0.1)


class VocosformerControlledVariableTest(unittest.TestCase):
    # Verifies that only the generator differs from the reproduced Vocos module.
    def setUp(self) -> None:
        # Builds the adaptation module and the Vocos baseline module.
        torch.manual_seed(20260810)
        self._model: Vocosformer = Vocosformer(VocosformerConfig())
        self._baseline: Vocos = Vocos(VocosConfig())

    def test_module_extends_the_vocos_module(self) -> None:
        # Inheritance is what makes the training and evaluation machinery shared.
        self.assertIsInstance(self._model, Vocos)

    def test_generator_is_the_attention_augmented_network(self) -> None:
        # The Vocos network built during construction is replaced by the adaptation.
        self.assertIsInstance(self._model.network, VocosformerNetwork)
        self.assertNotIsInstance(self._model.network, VocosNetwork)

    def test_non_generator_parameter_budget_is_identical_to_the_baseline(self) -> None:
        # Discriminators and loss modules must be untouched for the comparison to be controlled.
        adapted_surrounding: int = (
            sum(parameter.numel() for parameter in self._model.parameters())
            - sum(parameter.numel() for parameter in self._model.network.parameters())
        )
        baseline_surrounding: int = (
            sum(parameter.numel() for parameter in self._baseline.parameters())
            - sum(parameter.numel() for parameter in self._baseline.network.parameters())
        )
        self.assertEqual(
            adapted_surrounding,
            baseline_surrounding,
            msg=f"Non-generator capacity differs: {adapted_surrounding} against {baseline_surrounding}"
        )

    def test_manual_optimization_is_inherited(self) -> None:
        # The adversarial recipe requires the module to own its stepping.
        self.assertFalse(self._model.automatic_optimization)


class VocosformerSynthesisTest(unittest.TestCase):
    # Verifies the inherited inference surface on the Vocos conditioning protocol.
    def setUp(self) -> None:
        # Builds the adaptation module in evaluation mode, detached from any trainer.
        self._configuration: VocosformerConfig = VocosformerConfig()
        self._builder: ReferenceBatchBuilder = ReferenceBatchBuilder(sample_count=4096, sample_rate=24000)
        torch.manual_seed(20260811)
        self._model: Vocosformer = Vocosformer(self._configuration)
        self._model.eval()

    def test_predict_step_returns_the_synthesized_waveform_mapping(self) -> None:
        # The measurement stack consumes exactly this key at the conditioning rate.
        batch: dict[str, torch.Tensor | int] = self._builder.build(seed=61)
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._model.predict_step(batch, 0)
        waveform: torch.Tensor = prediction["synthesized_waveform"]
        self.assertEqual(tuple(waveform.shape), (1, self._builder.sample_count))
        self.assertEqual(waveform.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(waveform).all()))

    def test_synthesize_routes_through_the_adapted_generator(self) -> None:
        # The synthesis entry point consumed by the metric stack must use the replaced network.
        torch.manual_seed(62)
        mel: torch.Tensor = torch.randn(1, 100, 16)
        with torch.no_grad():
            synthesized: torch.Tensor = self._model.synthesize(mel)
            generated: torch.Tensor = self._model.network(mel)
        self.assertEqual(tuple(synthesized.shape), (1, 15 * self._configuration.hop_length))
        self.assertTrue(torch.equal(synthesized, generated))

    def test_both_mel_protocols_are_the_conditioning_protocol(self) -> None:
        # The adaptation measures mel error on the same protocol it conditions on.
        self.assertEqual(self._model.mel_protocol, self._configuration.mel_protocol)
        self.assertEqual(self._model.metric_mel_protocol, self._configuration.mel_protocol)

    def test_configuration_property_returns_the_adapted_record(self) -> None:
        # The property is narrowed to the adaptation's record type.
        self.assertIs(self._model.configuration, self._configuration)
        self.assertIsInstance(self._model.configuration, VocosformerConfig)

    def test_training_step_requires_trainer_owned_optimizers(self) -> None:
        # The inherited adversarial step cannot run without the optimizer pair.
        batch: dict[str, torch.Tensor | int] = self._builder.build(seed=63)
        with self.assertRaises(RuntimeError):
            self._model.training_step(batch, 0)


if __name__ == "__main__":
    unittest.main()
