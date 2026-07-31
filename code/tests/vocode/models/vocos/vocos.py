# This module:
# 1. Verifies the Vocos configuration record: the published adversarial and
#    topology defaults, the charactr factory, the attached mel protocol, and
#    the frozen extra-forbidding validation contract
# 2. Verifies the module's inference surface: synthesis length arithmetic,
#    the predict and test step output contract, waveform layout promotion,
#    the corpus-rate resampling branch, and the batch validation guards
# 3. Verifies the declared optimization: two AdamW optimizers bound to the
#    generator and the discriminator ensembles, each paired with a per-step
#    cosine schedule
#
# Design decisions:
# - Every forward runs on a single 4096-sample synthetic waveform with the
#   module detached from any trainer, so construction plus synthesis of the
#   full recipe stays a fraction of a second on CPU
# - Training is never executed. The training step is exercised only through
#   its detached-module guard, because a real step requires trainer-owned
#   optimizers and a backward pass, both outside the test budget
# - The validation step is not exercised: it logs val_loss through the
#   harness logging surface, which by contract refuses calls from a module
#   that no trainer has attached, so no assertion about the model itself
#   could be drawn from it
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import PositiveFloat, ValidationError

from syntheticmind.core.optimizer import OptimizationConfiguration, SchedulerConfig

from vocode.models.vocos.network import VocosNetwork
from vocode.models.vocos.vocos import Vocos, VocosConfig


class ReferenceBatchBuilder:
    # Produces deterministic LJSpeech-shaped batch mappings for the inference-surface checks.
    def __init__(self, sample_count: int, sample_rate: int) -> None:
        # Binds the waveform length and corpus rate this builder stamps on every batch.
        self._sample_count: int = sample_count
        self._sample_rate: int = sample_rate

    def build(self, seed: int) -> dict[str, torch.Tensor | int]:
        # Seeds the generator and returns one batched reference record.
        torch.manual_seed(seed)
        return {"waveform": torch.randn(1, self._sample_count), "sample_rate": self._sample_rate}

    def build_with_rate(self, seed: int, sample_rate: int) -> dict[str, torch.Tensor | int]:
        # Returns a record stamped with a corpus rate that differs from the protocol rate.
        torch.manual_seed(seed)
        return {"waveform": torch.randn(1, self._sample_count), "sample_rate": sample_rate}

    def build_unbatched(self, seed: int) -> dict[str, torch.Tensor | int]:
        # Returns a record whose waveform carries no batch axis.
        torch.manual_seed(seed)
        return {"waveform": torch.randn(self._sample_count), "sample_rate": self._sample_rate}

    @property
    def sample_count(self) -> int:
        # Returns the waveform length of every emitted record.
        return self._sample_count


class VocosConfigurationTest(unittest.TestCase):
    # Verifies that the Vocos configuration pins the published recipe and its mel protocol.
    def setUp(self) -> None:
        # Builds the unmodified recipe record under test.
        self._configuration: VocosConfig = VocosConfig()

    def test_default_record_matches_the_published_recipe(self) -> None:
        # The unmodified record is the charactr 24 kHz training recipe.
        self.assertEqual(self._configuration.input_channels, 100)
        self.assertEqual(self._configuration.hidden_dimension, 512)
        self.assertEqual(self._configuration.intermediate_dimension, 1536)
        self.assertEqual(self._configuration.layer_count, 8)
        self.assertEqual(self._configuration.n_fft, 1024)
        self.assertEqual(self._configuration.hop_length, 256)
        self.assertAlmostEqual(self._configuration.learning_rate, 0.0005)
        self.assertAlmostEqual(self._configuration.adam_beta_1, 0.8)
        self.assertAlmostEqual(self._configuration.adam_beta_2, 0.9)
        self.assertEqual(self._configuration.scheduler_total_steps, 249600)

    def test_charactr_factory_returns_the_default_record(self) -> None:
        # The named factory is the documented entry point to the same recipe.
        self.assertEqual(VocosConfig.charactr_mel_24khz(), self._configuration)

    def test_mel_protocol_is_the_charactr_twenty_four_kilohertz_grid(self) -> None:
        # Vocos conditions on a hundred-band HTK mel at 24 kHz with no frequency ceiling.
        self.assertEqual(self._configuration.mel_protocol.sample_rate, 24000)
        self.assertEqual(self._configuration.mel_protocol.n_mels, 100)
        self.assertEqual(self._configuration.mel_protocol.hop_length, 256)
        self.assertEqual(self._configuration.mel_protocol.mel_scale, "htk")
        self.assertIsNone(self._configuration.mel_protocol.fmax)

    def test_configuration_record_is_frozen(self) -> None:
        # Recipe fields cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.learning_rate: PositiveFloat = 0.1

    def test_unknown_field_is_rejected(self) -> None:
        # Extra fields are forbidden so a misspelled experiment setting fails loudly.
        with self.assertRaises(ValidationError):
            VocosConfig(learning_rates=0.0005)


class VocosSynthesisTest(unittest.TestCase):
    # Verifies the detached module's synthesis and prediction contract on minimal seeded batches.
    def setUp(self) -> None:
        # Builds the recipe module in evaluation mode, detached from any trainer.
        self._configuration: VocosConfig = VocosConfig()
        self._builder: ReferenceBatchBuilder = ReferenceBatchBuilder(sample_count=4096, sample_rate=24000)
        torch.manual_seed(20260803)
        self._model: Vocos = Vocos(self._configuration)
        self._model.eval()

    def test_network_is_the_vocos_generator(self) -> None:
        # The module's generator is the ConvNeXt and inverse-STFT network.
        self.assertIsInstance(self._model.network, VocosNetwork)

    def test_synthesize_matches_the_forward_pass(self) -> None:
        # The synthesis entry point consumed by the metric stack routes through forward.
        torch.manual_seed(41)
        mel: torch.Tensor = torch.randn(1, 100, 16)
        with torch.no_grad():
            synthesized: torch.Tensor = self._model.synthesize(mel)
            forwarded: torch.Tensor = self._model(mel)
        self.assertEqual(tuple(synthesized.shape), (1, 15 * self._configuration.hop_length))
        self.assertTrue(torch.equal(synthesized, forwarded))

    def test_predict_step_returns_the_synthesized_waveform_mapping(self) -> None:
        # The measurement stack consumes exactly this key at the conditioning rate.
        batch: dict[str, torch.Tensor | int] = self._builder.build(seed=42)
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._model.predict_step(batch, 0)
        waveform: torch.Tensor = prediction["synthesized_waveform"]
        self.assertEqual(tuple(waveform.shape), (1, self._builder.sample_count))
        self.assertEqual(waveform.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(waveform).all()))

    def test_unbatched_waveform_is_promoted_before_synthesis(self) -> None:
        # A record without a batch axis still synthesizes one waveform.
        batch: dict[str, torch.Tensor | int] = self._builder.build_unbatched(seed=43)
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._model.predict_step(batch, 0)
        self.assertEqual(tuple(prediction["synthesized_waveform"].shape), (1, self._builder.sample_count))

    def test_corpus_rate_below_the_protocol_rate_is_resampled(self) -> None:
        # A 22.05 kHz corpus record is lifted to the 24 kHz protocol, lengthening the synthesis.
        batch: dict[str, torch.Tensor | int] = self._builder.build_with_rate(seed=44, sample_rate=22050)
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._model.predict_step(batch, 0)
        waveform: torch.Tensor = prediction["synthesized_waveform"]
        self.assertGreater(
            waveform.shape[-1],
            self._builder.sample_count,
            msg="Resampling to the higher protocol rate must lengthen the synthesis"
        )

    def test_tensor_sample_rate_is_accepted(self) -> None:
        # Collated batches carry the corpus rate as a tensor, which must read as an integer rate.
        torch.manual_seed(45)
        batch: dict[str, torch.Tensor] = {
            "waveform": torch.randn(1, self._builder.sample_count),
            "sample_rate": torch.tensor([24000])
        }
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._model.predict_step(batch, 0)
        self.assertEqual(tuple(prediction["synthesized_waveform"].shape), (1, self._builder.sample_count))

    def test_test_step_delegates_to_the_prediction_path(self) -> None:
        # Test evaluation must measure exactly what prediction produces.
        batch: dict[str, torch.Tensor | int] = self._builder.build(seed=46)
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._model.predict_step(batch, 0)
            test_output: dict[str, torch.Tensor] = self._model.test_step(batch, 0)
        self.assertIn("synthesized_waveform", test_output)
        self.assertTrue(torch.equal(test_output["synthesized_waveform"], prediction["synthesized_waveform"]))

    def test_non_mapping_batch_is_rejected(self) -> None:
        # The step contract accepts only collated mappings.
        with self.assertRaises(TypeError):
            self._model.predict_step([torch.zeros(1, 16)], 0)

    def test_missing_waveform_entry_is_rejected(self) -> None:
        # A batch without the waveform tensor cannot condition synthesis.
        with self.assertRaises(TypeError):
            self._model.predict_step({"sample_rate": 24000}, 0)

    def test_missing_sample_rate_entry_is_rejected(self) -> None:
        # Without the corpus rate the resampling decision cannot be made.
        with self.assertRaises(TypeError):
            self._model.predict_step({"waveform": torch.zeros(1, 4096)}, 0)

    def test_multichannel_waveform_layout_is_rejected(self) -> None:
        # Only single-channel material matches the reference conditioning protocol.
        with self.assertRaises(ValueError):
            self._model.predict_step({"waveform": torch.zeros(1, 2, 4096), "sample_rate": 24000}, 0)

    def test_training_step_requires_trainer_owned_optimizers(self) -> None:
        # The adversarial recipe cannot step without the generator and discriminator pair.
        batch: dict[str, torch.Tensor | int] = self._builder.build(seed=47)
        with self.assertRaises(RuntimeError):
            self._model.training_step(batch, 0)

    def test_both_mel_protocols_are_the_conditioning_protocol(self) -> None:
        # Vocos measures mel error on the same protocol it conditions on.
        self.assertEqual(self._model.mel_protocol, self._configuration.mel_protocol)
        self.assertEqual(self._model.metric_mel_protocol, self._configuration.mel_protocol)

    def test_configuration_property_returns_the_bound_record(self) -> None:
        # The module exposes exactly the record it was constructed from.
        self.assertIs(self._model.configuration, self._configuration)


class VocosOptimizationDeclarationTest(unittest.TestCase):
    # Verifies the declared adversarial optimizer pair and their per-step cosine schedules.
    def setUp(self) -> None:
        # Builds the recipe module and collects its optimization declaration.
        self._configuration: VocosConfig = VocosConfig()
        torch.manual_seed(20260804)
        self._model: Vocos = Vocos(self._configuration)
        self._declaration: OptimizationConfiguration = self._model.configure_optimizers()

    def test_manual_optimization_is_selected(self) -> None:
        # Two optimizers require the module to own its stepping.
        self.assertFalse(self._model.automatic_optimization)

    def test_two_optimizers_are_declared(self) -> None:
        # The generator and the discriminator ensembles are optimized separately.
        self.assertIsInstance(self._declaration.optimizer, list)
        self.assertEqual(len(self._declaration.optimizer), 2)

    def test_generator_optimizer_covers_exactly_the_generator_parameters(self) -> None:
        # The first optimizer must own the synthesis network and nothing else.
        generator_optimizer: torch.optim.Optimizer = self._declaration.optimizer[0]
        owned_tensor_count: int = sum(len(group["params"]) for group in generator_optimizer.param_groups)
        self.assertEqual(owned_tensor_count, len(list(self._model.network.parameters())))

    def test_optimizer_hyperparameters_match_the_recipe(self) -> None:
        # Both optimizers use the recipe learning rate and beta pair.
        optimizer: torch.optim.Optimizer
        for optimizer in self._declaration.optimizer:
            self.assertIsInstance(optimizer, torch.optim.AdamW)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], self._configuration.learning_rate)
            self.assertEqual(
                optimizer.param_groups[0]["betas"],
                (self._configuration.adam_beta_1, self._configuration.adam_beta_2)
            )

    def test_two_per_step_cosine_schedules_are_declared(self) -> None:
        # Both schedules advance per optimizer step over the recipe horizon.
        self.assertIsInstance(self._declaration.scheduler, list)
        self.assertEqual(len(self._declaration.scheduler), 2)
        scheduler_declaration: SchedulerConfig
        for scheduler_declaration in self._declaration.scheduler:
            self.assertEqual(scheduler_declaration.name, "cosine")
            self.assertEqual(scheduler_declaration.interval, "step")
            self.assertEqual(scheduler_declaration.t_max, self._configuration.scheduler_total_steps)
            self.assertAlmostEqual(scheduler_declaration.eta_min, 0.0)


if __name__ == "__main__":
    unittest.main()
