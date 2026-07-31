# This module:
# 1. Verifies the APNet2 configuration record: the mandatory topology fields,
#    the redmist328 factory values, the split between the conditioning and
#    reconstruction mel protocols, and the frozen validation contract
# 2. Verifies the module's inference surface: synthesis length arithmetic,
#    the predict and test step output contract, the batch validation guards,
#    and the detached-module behavior of the training step and the epoch-end
#    scheduler hook
# 3. Verifies the declared optimization: two AdamW optimizers bound to the
#    generator and the discriminator ensembles, each paired with an
#    exponential decay schedule
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
#   that no trainer has attached
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError

from syntheticmind.core.optimizer import OptimizationConfiguration

from vocode.models.apnet2.apnet2 import Apnet2, Apnet2Config
from vocode.models.apnet2.network import Apnet2Network


class ReferenceBatchBuilder:
    # Produces deterministic LJSpeech-shaped batch mappings for the inference-surface checks.
    # Only the waveform entry is emitted, because the module derives its
    # conditioning mel from that entry and reads nothing else from a batch;
    # a fixture carrying more would overstate the step's real input contract.
    def __init__(self, sample_count: int) -> None:
        # Binds the waveform length this builder stamps on every batch.
        #
        # Args:
        #     sample_count: Samples per emitted waveform. It must be a whole
        #         number of hops for the round trip through the mel and back
        #         to return the same length the batch went in with.
        self._sample_count: int = sample_count

    def build(self, seed: int) -> dict[str, torch.Tensor]:
        # Seeds the generator and returns one batched reference record.
        # Seeding immediately before sampling makes each test's input a
        # function of its own seed alone, so tests cannot influence one another
        # through the global generator regardless of execution order.
        #
        # Args:
        #     seed: Global generator seed applied before sampling.
        #
        # Returns:
        #     A mapping holding one ``[1, sample_count]`` waveform under the
        #     key the step contract reads.
        torch.manual_seed(seed)
        return {"waveform": torch.randn(1, self._sample_count)}

    @property
    def sample_count(self) -> int:
        # Returns the waveform length of every emitted record.
        return self._sample_count


class Apnet2ConfigurationTest(unittest.TestCase):
    # Verifies that the configuration pins the reference recipe and its two mel protocols.
    def setUp(self) -> None:
        # Builds the reference recipe record under test.
        self._configuration: Apnet2Config = Apnet2Config.redmist328()

    def test_topology_fields_are_mandatory(self) -> None:
        # There is no default architecture: a record must state its full topology.
        with self.assertRaises(ValidationError):
            Apnet2Config()

    def test_redmist_factory_matches_the_reference_recipe(self) -> None:
        # The named factory is the documented entry point to the reference topology.
        self.assertEqual(self._configuration.input_mel_channels, 80)
        self.assertEqual(self._configuration.n_fft, 1024)
        self.assertEqual(self._configuration.hop_size, 256)
        self.assertEqual(self._configuration.win_size, 1024)
        self.assertEqual(self._configuration.asp_channel, 512)
        self.assertEqual(self._configuration.psp_channel, 512)
        self.assertEqual(self._configuration.convnext_layer_count, 8)
        self.assertEqual(self._configuration.convnext_intermediate_dimension, 1536)

    def test_adversarial_hyperparameters_match_the_reference_recipe(self) -> None:
        # Learning rate, betas, and decay come from the published training script.
        self.assertAlmostEqual(self._configuration.learning_rate, 0.0002)
        self.assertAlmostEqual(self._configuration.adam_beta_1, 0.8)
        self.assertAlmostEqual(self._configuration.adam_beta_2, 0.99)
        self.assertAlmostEqual(self._configuration.learning_rate_decay, 0.999)

    def test_conditioning_and_reconstruction_protocols_differ_only_in_the_ceiling(self) -> None:
        # The reference conditions on a band-limited mel while its loss uses the full band.
        self.assertAlmostEqual(self._configuration.mel_protocol.fmax, 8000.0)
        self.assertIsNone(self._configuration.reconstruction_mel_protocol.fmax)
        self.assertEqual(
            self._configuration.mel_protocol.sample_rate,
            self._configuration.reconstruction_mel_protocol.sample_rate
        )
        self.assertEqual(
            self._configuration.mel_protocol.n_mels,
            self._configuration.reconstruction_mel_protocol.n_mels
        )
        self.assertEqual(
            self._configuration.mel_protocol.hop_length,
            self._configuration.reconstruction_mel_protocol.hop_length
        )

    def test_conditioning_protocol_is_the_centered_twenty_two_kilohertz_grid(self) -> None:
        # APNet2 conditions on an eighty-band Slaney mel at 22.05 kHz with centered frames.
        self.assertEqual(self._configuration.mel_protocol.sample_rate, 22050)
        self.assertEqual(self._configuration.mel_protocol.n_mels, 80)
        self.assertEqual(self._configuration.mel_protocol.mel_scale, "slaney")
        self.assertTrue(self._configuration.mel_protocol.center)

    def test_configuration_record_is_frozen(self) -> None:
        # Recipe fields cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.hop_size = 512

    def test_unknown_field_is_rejected(self) -> None:
        # Extra fields are forbidden so a misspelled experiment setting fails loudly.
        record: dict[str, object] = dict(self._configuration)
        record["asp_channels"] = 512
        with self.assertRaises(ValidationError):
            Apnet2Config(**record)


class Apnet2SynthesisTest(unittest.TestCase):
    # Verifies the detached module's synthesis and prediction contract on minimal seeded batches.
    def setUp(self) -> None:
        # Builds the recipe module in evaluation mode, detached from any trainer.
        self._configuration: Apnet2Config = Apnet2Config.redmist328()
        self._builder: ReferenceBatchBuilder = ReferenceBatchBuilder(sample_count=4096)
        torch.manual_seed(20260817)
        self._model: Apnet2 = Apnet2(self._configuration)
        self._model.eval()

    def test_network_is_the_apnet2_generator(self) -> None:
        # The module's generator is the parallel amplitude and phase network.
        self.assertIsInstance(self._model.network, Apnet2Network)

    def test_synthesize_matches_the_forward_pass(self) -> None:
        # The synthesis entry point consumed by the metric stack routes through forward.
        torch.manual_seed(101)
        mel: torch.Tensor = torch.randn(1, 80, 16)
        with torch.no_grad():
            synthesized: torch.Tensor = self._model.synthesize(mel)
            forwarded: torch.Tensor = self._model(mel)
        self.assertEqual(tuple(synthesized.shape), (1, 1, 15 * self._configuration.hop_size))
        self.assertTrue(torch.equal(synthesized, forwarded))

    def test_predict_step_returns_a_channel_free_waveform_mapping(self) -> None:
        # The measurement stack consumes a [batch, time] synthesis under this key.
        # The synthesis returns exactly the input sample count here because
        # 4096 samples is a whole number of hops, so the centered analysis and
        # the centered inverse transform cancel; the metric components compare
        # signals elementwise and would otherwise be handed a length they must
        # truncate.
        batch: dict[str, torch.Tensor] = self._builder.build(seed=102)
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._model.predict_step(batch, 0)
        waveform: torch.Tensor = prediction["synthesized_waveform"]
        self.assertEqual(tuple(waveform.shape), (1, self._builder.sample_count))
        self.assertEqual(waveform.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(waveform).all()))

    def test_test_step_delegates_to_the_prediction_path(self) -> None:
        # Test evaluation must measure exactly what prediction produces.
        batch: dict[str, torch.Tensor] = self._builder.build(seed=103)
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._model.predict_step(batch, 0)
            test_output: dict[str, torch.Tensor] = self._model.test_step(batch, 0)
        self.assertIn("synthesized_waveform", test_output)
        self.assertTrue(torch.equal(test_output["synthesized_waveform"], prediction["synthesized_waveform"]))

    def test_non_mapping_batch_is_rejected(self) -> None:
        # The step contract accepts only collated mappings.
        with self.assertRaises(TypeError):
            self._model.predict_step([torch.zeros(1, 4096)], 0)

    def test_missing_waveform_entry_is_rejected(self) -> None:
        # A batch without the waveform tensor cannot condition synthesis.
        with self.assertRaises(TypeError):
            self._model.predict_step({"sample_rate": 22050}, 0)

    def test_training_step_requires_trainer_owned_optimizers(self) -> None:
        # The adversarial recipe cannot step without the generator and discriminator pair.
        # A detached module reports no optimizers, and the step refuses on that
        # rather than proceeding to a partial update, so a misconfigured run
        # fails at its first batch instead of training only half the model.
        batch: dict[str, torch.Tensor] = self._builder.build(seed=104)
        with self.assertRaises(RuntimeError):
            self._model.training_step(batch, 0)

    def test_epoch_end_hook_is_inert_without_schedulers(self) -> None:
        # The decay hook must tolerate a detached module rather than raising.
        self.assertIsNone(self._model.lr_schedulers())
        self._model.on_train_epoch_end()

    def test_metric_protocol_is_the_full_band_reconstruction_protocol(self) -> None:
        # Mel error is measured on the full band while conditioning stays band-limited.
        self.assertEqual(self._model.mel_protocol, self._configuration.mel_protocol)
        self.assertEqual(self._model.metric_mel_protocol, self._configuration.reconstruction_mel_protocol)
        self.assertIsNone(self._model.metric_mel_protocol.fmax)

    def test_configuration_property_returns_the_bound_record(self) -> None:
        # The module exposes exactly the record it was constructed from.
        self.assertIs(self._model.configuration, self._configuration)


class Apnet2OptimizationDeclarationTest(unittest.TestCase):
    # Verifies the declared adversarial optimizer pair and their exponential decay schedules.
    def setUp(self) -> None:
        # Builds the recipe module and collects its optimization declaration.
        self._configuration: Apnet2Config = Apnet2Config.redmist328()
        torch.manual_seed(20260818)
        self._model: Apnet2 = Apnet2(self._configuration)
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
        # Position is contractual: the training step unpacks the declaration as
        # generator first, and the optimizer toggle restricts gradients to the
        # stepping optimizer's own parameters, so a first optimizer holding the
        # wrong set would silently train the wrong half of the model.
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

    def test_two_exponential_schedules_are_declared(self) -> None:
        # Both schedules decay by the recipe factor once per epoch.
        self.assertIsInstance(self._declaration.scheduler, list)
        self.assertEqual(len(self._declaration.scheduler), 2)
        scheduler: torch.optim.lr_scheduler.LRScheduler
        for scheduler in self._declaration.scheduler:
            self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
            self.assertAlmostEqual(scheduler.gamma, self._configuration.learning_rate_decay)
