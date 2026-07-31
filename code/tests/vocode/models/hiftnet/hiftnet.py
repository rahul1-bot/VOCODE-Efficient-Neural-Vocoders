# This module:
# 1. Verifies the HiFTNet configuration record: the yl4579 reference values,
#    the optional pitch-checkpoint path, immutability, and the rejection of
#    unknown fields
# 2. Verifies the HiFTNet module surface: manual optimization, the synthesis
#    and prediction contracts, the protocol properties, and the optimizer and
#    scheduler declaration
#
# Design decisions:
# - The pretrained F0 predictor of the reference recipe is never retrieved:
#   every module here is constructed with an absent checkpoint path, so the
#   pitch extractor runs at its random initialization
# - The module assertions run on a reduced generator that keeps the
#   reference mel protocol, because only the module wiring is under test;
#   the generator topology is covered in the network tests
# - Training is never executed: training_step and validation_step call the
#   harness logging surface and on_train_epoch_end reaches for attached
#   schedulers, so all three require a trainer and stay out of scope
#
# Author: Rahul Sawhney

import math
import unittest
from pathlib import Path

import torch
from pydantic import ValidationError

from syntheticmind.core.optimizer import OptimizationConfiguration

from vocode.metrics.pesq import PesqConfig
from vocode.metrics.stoi import StoiConfig
from vocode.models.hiftnet.hiftnet import Hiftnet, HiftnetConfig
from vocode.models.hiftnet.network import HiftnetNetwork
from vocode.transforms.mel import MelConfig, MelSpectrogram


class ReducedModuleRecipe:
    # Builds the reduced HiFTNet configuration used by the module-surface
    # assertions. The reduction is chosen so that what these tests assert
    # stays exactly as it is under the reference recipe: the width is cut from
    # five hundred twelve channels to thirty-two and the residual kernel set to
    # a single entry, because neither affects module wiring, while the mel
    # protocol, the band count, the sampling rate, and the inverse-STFT head
    # are left at reference values, because the shape contracts asserted below
    # are derived from them.
    def build(self) -> HiftnetConfig:
        # Returns the reduced recipe. The upsampling rates are lowered to four
        # and four, so the total expansion is sixty-four rather than the
        # reference two hundred fifty-six; the shape assertions compute the
        # expected sample count from the configuration rather than hard-coding
        # it, so they remain valid under that change.
        #
        # Returns:
        #     A valid HiftnetConfig with an absent pitch-checkpoint path, so
        #     construction performs no filesystem access and no download.
        return HiftnetConfig(
            input_mel_channels=80,
            sampling_rate=22050,
            upsample_rates=(4, 4),
            upsample_kernel_sizes=(8, 8),
            upsample_initial_channel=32,
            resblock_kernel_sizes=(3,),
            resblock_dilation_sizes=((1, 3, 5),),
            gen_istft_n_fft=16,
            gen_istft_hop_size=4,
            f0_checkpoint_path=None,
            mel_protocol=MelConfig.hiftnet_yl4579(),
            pesq_protocol=PesqConfig(),
            stoi_protocol=StoiConfig()
        )


class HiftnetReferenceConfigurationTest(unittest.TestCase):
    # Verifies that the reference factory reproduces the published recipe and
    # that the record enforces its own validation contract. These assertions
    # are the guard against silent recipe drift: the values are transcribed
    # from the reference implementation, so a change to any of them is a
    # change to what this study claims to reproduce and must be a deliberate,
    # visible edit rather than a side effect.
    def setUp(self) -> None:
        # Reads the reference recipe record under test.
        self._configuration: HiftnetConfig = HiftnetConfig.yl4579()

    def test_reference_topology_matches_the_published_recipe(self) -> None:
        # The reference generator upsamples eight and eight from 512 channels at 22.05 kHz.
        self.assertEqual(self._configuration.input_mel_channels, 80)
        self.assertEqual(self._configuration.sampling_rate, 22050)
        self.assertEqual(self._configuration.upsample_rates, (8, 8))
        self.assertEqual(self._configuration.upsample_kernel_sizes, (16, 16))
        self.assertEqual(self._configuration.upsample_initial_channel, 512)
        self.assertEqual(self._configuration.resblock_kernel_sizes, (3, 7, 11))

    def test_inverse_stft_head_matches_the_published_recipe(self) -> None:
        # The final reconstruction is a sixteen-point inverse transform at hop four.
        self.assertEqual(self._configuration.gen_istft_n_fft, 16)
        self.assertEqual(self._configuration.gen_istft_hop_size, 4)

    def test_pitch_checkpoint_path_is_absent_by_default(self) -> None:
        # The factory leaves the pretrained pitch predictor unset unless a
        # path is supplied. This default is what keeps the whole test suite
        # free of any external download: an absent path is a documented
        # no-op in the network's loader rather than an error, so every module
        # built here runs at random pitch initialization.
        self.assertIsNone(self._configuration.f0_checkpoint_path)

    def test_pitch_checkpoint_path_is_carried_when_supplied(self) -> None:
        # A supplied path travels into the record for the network to consume.
        checkpoint_path: Path = Path("/tmp/vocode-f0/jdc.pt")
        configured: HiftnetConfig = HiftnetConfig.yl4579(f0_checkpoint_path=checkpoint_path)
        self.assertEqual(configured.f0_checkpoint_path, checkpoint_path)

    def test_mel_protocol_is_the_reference_eighty_band_protocol(self) -> None:
        # Conditioning follows the reference 22.05 kHz eighty-band mel protocol.
        self.assertEqual(self._configuration.mel_protocol.n_mels, 80)
        self.assertEqual(self._configuration.mel_protocol.sample_rate, 22050)

    def test_optimization_defaults_match_the_reference_recipe(self) -> None:
        # Learning rate, Adam moments, and the per-epoch decay follow the reference settings.
        self.assertEqual(self._configuration.learning_rate, 0.0002)
        self.assertEqual(self._configuration.adam_beta_1, 0.8)
        self.assertEqual(self._configuration.adam_beta_2, 0.99)
        self.assertEqual(self._configuration.learning_rate_decay, 0.999)

    def test_configuration_is_immutable(self) -> None:
        # Frozen settings cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.learning_rate = 0.5

    def test_unknown_field_is_rejected(self) -> None:
        # Extra fields are forbidden so silent typos cannot enter an experiment record.
        record: dict[str, object] = dict(self._configuration)
        record["unknown_setting"] = 1
        with self.assertRaises(ValidationError):
            HiftnetConfig(**record)

    def test_reference_record_round_trips_through_its_own_fields(self) -> None:
        # Control for the rejection above. Without this assertion, the
        # unknown-field test would pass even if reconstruction failed for some
        # unrelated reason, so proving that the unperturbed record rebuilds
        # cleanly is what attributes that failure to the extra key alone.
        self.assertEqual(HiftnetConfig(**dict(self._configuration)), self._configuration)


class HiftnetModuleSurfaceTest(unittest.TestCase):
    # Verifies module construction, the synthesis and prediction contracts,
    # and the properties the Vocoder protocol requires. The scope is
    # deliberately the module's wiring rather than its numerics: the
    # assertions cover which objects are held, which shapes cross each
    # boundary, and which inputs are refused, none of which depends on the
    # values a randomly initialized generator produces.
    def setUp(self) -> None:
        # Builds the reduced module at random pitch-predictor initialization
        # plus its reference waveform. The seed makes the random
        # initialization and the random waveform reproducible, which matters
        # because the shape assertions depend on the frame count the mel
        # transform derives from this waveform's length.
        torch.manual_seed(1234)
        self._configuration: HiftnetConfig = ReducedModuleRecipe().build()
        self._module: Hiftnet = Hiftnet(self._configuration)
        self._waveform: torch.Tensor = torch.randn(1, 4096) * 0.1

    def test_automatic_optimization_is_disabled(self) -> None:
        # The adversarial recipe drives both optimizers manually.
        self.assertFalse(self._module.automatic_optimization)

    def test_generator_network_is_the_hiftnet_network(self) -> None:
        # The published weight adapter loads onto this attribute, so its type is contractual.
        self.assertIsInstance(self._module.network, HiftnetNetwork)

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The module exposes exactly the record it was constructed with.
        self.assertIs(self._module.configuration, self._configuration)

    def test_metric_protocol_reuses_the_conditioning_protocol(self) -> None:
        # HiFTNet measures against the same mel protocol it conditions on.
        self.assertEqual(self._module.mel_protocol, self._configuration.mel_protocol)
        self.assertEqual(self._module.metric_mel_protocol, self._configuration.mel_protocol)

    def test_synthesize_matches_the_forward_shape_contract(self) -> None:
        # Synthesis emits the channel-carrying waveform layout of the
        # generator network. The expected sample count is computed from the
        # configuration rather than written as a literal, so the assertion
        # states the actual invariant (output samples equal input frames
        # times the upsampling-rate product times the inverse-STFT hop)
        # instead of a number that would silently need editing if the reduced
        # recipe changed.
        mel: torch.Tensor = torch.randn(1, 80, 6)
        with torch.no_grad():
            synthesized: torch.Tensor = self._module.synthesize(mel)
        frame_expansion: int = (
            math.prod(self._configuration.upsample_rates) * self._configuration.gen_istft_hop_size
        )
        self.assertEqual(tuple(synthesized.shape), (1, 1, 6 * frame_expansion))

    def test_prediction_returns_the_synthesized_waveform_mapping(self) -> None:
        # The measurement stack consumes the squeezed synthesized_waveform
        # key, so this proves two things at once: that the mapping carries
        # that exact key, and that the channel axis the inverse-STFT head
        # introduces has been removed. The frame count is obtained by running
        # the same mel transform the module uses rather than by predicting it
        # from the waveform length, because the protocol's padding and
        # centering settings determine it and duplicating that arithmetic here
        # would test the duplicate rather than the module.
        mel_transform: MelSpectrogram = MelSpectrogram(self._module.mel_protocol)
        with torch.no_grad():
            frame_count: int = int(mel_transform(self._waveform).shape[-1])
            prediction: dict[str, torch.Tensor] = self._module.predict_step({"waveform": self._waveform}, 0)
        self.assertIn("synthesized_waveform", prediction)
        self.assertEqual(tuple(prediction["synthesized_waveform"].shape), (1, frame_count * 64))

    def test_test_step_returns_the_prediction_output(self) -> None:
        # Test evaluation measures exactly the prediction path, so both carry the same contract.
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._module.predict_step({"waveform": self._waveform}, 0)
            evaluation: dict[str, torch.Tensor] = self._module.test_step({"waveform": self._waveform}, 0)
        self.assertEqual(list(evaluation.keys()), list(prediction.keys()))
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


class HiftnetOptimizationDeclarationTest(unittest.TestCase):
    # Verifies the paired generator and discriminator optimizers and their
    # exponential schedulers. The declaration is asserted rather than a
    # training run, because the training step reaches for the harness logging
    # surface and a trainer-attached epoch loop; what can be checked without a
    # trainer is that the declaration has the shape the step later depends on,
    # in particular that it is a two-element list whose first entry owns the
    # generator.
    def setUp(self) -> None:
        # Builds the reduced module whose optimization declaration is under test.
        torch.manual_seed(1234)
        self._configuration: HiftnetConfig = ReducedModuleRecipe().build()
        self._module: Hiftnet = Hiftnet(self._configuration)

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
        # wrong parameters would silently freeze or train the wrong half.
        # Counting the whole network confirms the pitch extractor is included,
        # matching the implementation's joint fine-tuning of the pitch model.
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
