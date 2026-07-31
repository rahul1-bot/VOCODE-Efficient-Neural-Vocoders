# This module:
# 1. Verifies the FreeV configuration record: the mandatory topology fields,
#    the official factory values, the split between the conditioning and
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

from vocode.models.apnet2.discriminator import Apnet2MultiPeriodDiscriminator, Apnet2MultiResolutionDiscriminator
from vocode.models.freev.freev import Freev, FreevConfig
from vocode.models.freev.network import FreevNetwork


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


class FreevConfigurationTest(unittest.TestCase):
    # Verifies that the configuration pins the official recipe and its two mel protocols.
    def setUp(self) -> None:
        # Builds the official recipe record under test.
        self._configuration: FreevConfig = FreevConfig.official()

    def test_topology_fields_are_mandatory(self) -> None:
        # There is no default architecture: a record must state its full topology.
        with self.assertRaises(ValidationError):
            FreevConfig()

    def test_official_factory_matches_the_reference_recipe(self) -> None:
        # The named factory is the documented entry point to the reference topology.
        self.assertEqual(self._configuration.input_mel_channels, 80)
        self.assertEqual(self._configuration.n_fft, 1024)
        self.assertEqual(self._configuration.hop_size, 256)
        self.assertEqual(self._configuration.win_size, 1024)
        self.assertEqual(self._configuration.sampling_rate, 22050)
        self.assertEqual(self._configuration.psp_channel, 512)
        self.assertEqual(self._configuration.convnext_layer_count, 8)
        self.assertEqual(self._configuration.amplitude_refinement_layer_count, 1)
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

    def test_conditioning_protocol_matches_the_network_sampling_rate(self) -> None:
        # The pseudo-inverse prior is built from this basis, so the rates must agree.
        # The record carries the rate, the band count, and the hop in two
        # places: on the mel protocol the module extracts with, and as
        # standalone fields the network reconstructs the mel basis from. A
        # disagreement would leave the prior inverting a basis the network is
        # never fed, which degrades the amplitude estimate silently rather than
        # raising, so the agreement is asserted rather than assumed.
        self.assertEqual(self._configuration.mel_protocol.sample_rate, self._configuration.sampling_rate)
        self.assertEqual(self._configuration.mel_protocol.n_mels, self._configuration.input_mel_channels)
        self.assertEqual(self._configuration.mel_protocol.hop_length, self._configuration.hop_size)
        self.assertTrue(self._configuration.mel_protocol.center)

    def test_configuration_record_is_frozen(self) -> None:
        # Recipe fields cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.hop_size = 512

    def test_unknown_field_is_rejected(self) -> None:
        # Extra fields are forbidden so a misspelled experiment setting fails loudly.
        record: dict[str, object] = dict(self._configuration)
        record["psp_channels"] = 512
        with self.assertRaises(ValidationError):
            FreevConfig(**record)


class FreevSynthesisTest(unittest.TestCase):
    # Verifies the detached module's synthesis and prediction contract on minimal seeded batches.
    def setUp(self) -> None:
        # Builds the recipe module in evaluation mode, detached from any trainer.
        self._configuration: FreevConfig = FreevConfig.official()
        self._builder: ReferenceBatchBuilder = ReferenceBatchBuilder(sample_count=4096)
        torch.manual_seed(20260824)
        self._model: Freev = Freev(self._configuration)
        self._model.eval()

    def test_network_is_the_freev_generator(self) -> None:
        # The module's generator is the prior-seeded amplitude and phase network.
        self.assertIsInstance(self._model.network, FreevNetwork)

    def test_synthesize_matches_the_forward_pass(self) -> None:
        # The synthesis entry point consumed by the metric stack routes through forward.
        torch.manual_seed(121)
        mel: torch.Tensor = torch.randn(1, 80, 16)
        with torch.no_grad():
            synthesized: torch.Tensor = self._model.synthesize(mel)
            forwarded: torch.Tensor = self._model(mel)
        self.assertEqual(tuple(synthesized.shape), (1, 1, 15 * self._configuration.hop_size))
        self.assertTrue(torch.equal(synthesized, forwarded))

    def test_predict_step_returns_a_channel_free_waveform_mapping(self) -> None:
        # The measurement stack consumes a [batch, time] synthesis under this key.
        batch: dict[str, torch.Tensor] = self._builder.build(seed=122)
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._model.predict_step(batch, 0)
        waveform: torch.Tensor = prediction["synthesized_waveform"]
        self.assertEqual(tuple(waveform.shape), (1, self._builder.sample_count))
        self.assertEqual(waveform.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(waveform).all()))

    def test_test_step_delegates_to_the_prediction_path(self) -> None:
        # Test evaluation must measure exactly what prediction produces.
        batch: dict[str, torch.Tensor] = self._builder.build(seed=123)
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
        batch: dict[str, torch.Tensor] = self._builder.build(seed=124)
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

    def test_adversarial_machinery_is_shared_with_apnet2(self) -> None:
        # FreeV trains against the APNet2 discriminator ensembles rather than its own.
        # This is a comparability claim, not an implementation detail: the two
        # architectures are judged by identically constructed critics, so a
        # measured difference between them is attributable to the generator and
        # its loss composition alone. Total element count is compared rather
        # than object identity, because each model constructs its own
        # ensembles and identity would never hold; a divergence in the critics'
        # capacity would still change the sum.
        declaration: OptimizationConfiguration = self._model.configure_optimizers()
        discriminator_optimizer: torch.optim.Optimizer = declaration.optimizer[1]
        owned_element_count: int = sum(
            int(parameter.numel())
            for group in discriminator_optimizer.param_groups
            for parameter in group["params"]
        )
        reference_element_count: int = (
            sum(parameter.numel() for parameter in Apnet2MultiPeriodDiscriminator().parameters())
            + sum(parameter.numel() for parameter in Apnet2MultiResolutionDiscriminator().parameters())
        )
        self.assertEqual(
            owned_element_count,
            reference_element_count,
            msg=f"Discriminator capacity is {owned_element_count} against the APNet2 {reference_element_count}"
        )

    def test_configuration_property_returns_the_bound_record(self) -> None:
        # The module exposes exactly the record it was constructed from.
        self.assertIs(self._model.configuration, self._configuration)


class FreevOptimizationDeclarationTest(unittest.TestCase):
    # Verifies the declared adversarial optimizer pair and their exponential decay schedules.
    def setUp(self) -> None:
        # Builds the recipe module and collects its optimization declaration.
        self._configuration: FreevConfig = FreevConfig.official()
        torch.manual_seed(20260825)
        self._model: Freev = Freev(self._configuration)
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

    def test_two_exponential_schedules_are_declared(self) -> None:
        # Both schedules decay by the recipe factor once per epoch.
        self.assertIsInstance(self._declaration.scheduler, list)
        self.assertEqual(len(self._declaration.scheduler), 2)
        scheduler: torch.optim.lr_scheduler.LRScheduler
        for scheduler in self._declaration.scheduler:
            self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
            self.assertAlmostEqual(scheduler.gamma, self._configuration.learning_rate_decay)
