# This module:
# 1. Verifies the LPCNet configuration record: the xiph reference values,
#    immutability, and the rejection of unknown fields
# 2. Verifies the LPCNet module surface: manual optimization, the copy
#    synthesis contract from a reference waveform, the waveform layout
#    normalization, and the single-optimizer declaration
#
# Design decisions:
# - Copy synthesis is autoregressive at sample rate, so every synthesis here
#   runs a miniature network at frame size sixteen over half a second of
#   synthetic audio at most; the reference geometry (384-unit recurrent core
#   at frame size 160 over 16 kHz utterances) is never synthesized because a
#   single reference utterance is tens of thousands of sequential steps
# - Training is never executed: training_step applies the post-step protocol
#   through the harness optimizer surface, which requires an attached
#   trainer, and validation_step logs through the same surface
# - Waveforms are synthetic noise at low amplitude; no dataset audio is read
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError

from syntheticmind.core.optimizer import OptimizationConfiguration

from vocode.metrics.pesq import PesqConfig
from vocode.metrics.stoi import StoiConfig
from vocode.models.lpcnet.lpcnet import Lpcnet, LpcnetConfig
from vocode.models.lpcnet.network import LpcnetNetwork
from vocode.transforms.lpc import LpcnetFeatureConfig
from vocode.transforms.mel import MelConfig


class MiniatureModuleRecipe:
    # Builds the miniature LPCNet configuration whose synthesis loop is short
    # enough to run. Two things are held at reference values because the
    # assertions depend on them: the sampling rate and the filter order, which
    # together with the frame size determine the analysis geometry the shape
    # contracts are read against. The widths are cut freely, since the module
    # surface under test is indifferent to them.
    def build(self) -> LpcnetConfig:
        # Returns the reduced recipe. The frame size is reduced from the
        # reference and the feature configuration is given the matching value,
        # which is essential rather than cosmetic: the module derives its
        # sample count from the frame count the extractor reports, so a
        # mismatch between the two frame sizes would silently truncate
        # synthesis. Training noise is disabled so the teacher-forced path
        # stays deterministic.
        return LpcnetConfig(
            sample_rate=16000,
            feature_dimension=20,
            condition_dimension=8,
            embedding_dimension=4,
            first_gru_dimension=8,
            second_gru_dimension=4,
            lpc_order=16,
            frame_size=16,
            preemphasis_coefficient=0.85,
            training_noise_standard_deviation=0.0,
            feature_configuration=LpcnetFeatureConfig(frame_size=16),
            metric_mel_protocol=MelConfig.lpcnet_metric_16khz(),
            pesq_protocol=PesqConfig(),
            stoi_protocol=StoiConfig()
        )


class LpcnetReferenceConfigurationTest(unittest.TestCase):
    # Verifies the xiph reference configuration values and the validation
    # behavior of the record. These values are transcribed from the reference
    # recipe, so each assertion is a claim about what this study reproduces
    # and a change to any of them narrows or breaks that claim. The nested
    # protocol assertion additionally proves the defaults of the feature,
    # sparsifier, and sampler records travel with the configuration rather
    # than having to be supplied at every construction site.
    def setUp(self) -> None:
        # Reads the reference recipe record under test.
        self._configuration: LpcnetConfig = LpcnetConfig.xiph_reference()

    def test_analysis_geometry_matches_the_reference_recipe(self) -> None:
        # The reference operates at 16 kHz with 160-sample frames and a sixteenth-order filter.
        self.assertEqual(self._configuration.sample_rate, 16000)
        self.assertEqual(self._configuration.frame_size, 160)
        self.assertEqual(self._configuration.lpc_order, 16)
        self.assertEqual(self._configuration.feature_dimension, 20)

    def test_network_dimensions_match_the_reference_deployment_geometry(self) -> None:
        # The sparsification target is the 384-unit first recurrent core of
        # the reference. That width is not free: it must divide evenly by both
        # pruning block dimensions for the block reshape to succeed, so it is
        # a constraint of the deployment kernel rather than a tuning choice.
        # The second core is an order of magnitude narrower because it runs
        # unpruned and therefore costs full density at inference.
        self.assertEqual(self._configuration.condition_dimension, 128)
        self.assertEqual(self._configuration.embedding_dimension, 128)
        self.assertEqual(self._configuration.first_gru_dimension, 384)
        self.assertEqual(self._configuration.second_gru_dimension, 16)

    def test_signal_conditioning_matches_the_reference_recipe(self) -> None:
        # Preemphasis and the teacher-forcing noise follow the reference training recipe.
        self.assertEqual(self._configuration.preemphasis_coefficient, 0.85)
        self.assertEqual(self._configuration.training_noise_standard_deviation, 0.3)

    def test_optimization_defaults_match_the_reference_recipe(self) -> None:
        # A single Adam optimizer with inverse-time decay drives the reference training run.
        self.assertEqual(self._configuration.learning_rate, 0.001)
        self.assertEqual(self._configuration.adam_beta_1, 0.5)
        self.assertEqual(self._configuration.adam_beta_2, 0.8)
        self.assertEqual(self._configuration.learning_rate_step_decay, 5e-5)
        self.assertEqual(self._configuration.recurrent_clip_value, 0.992)

    def test_nested_protocols_default_to_the_reference_records(self) -> None:
        # Feature analysis, sparsification, and sampling defaults travel with the configuration.
        self.assertEqual(self._configuration.feature_configuration.frame_size, 160)
        self.assertEqual(self._configuration.sparsifier_configuration.schedule_end_step, 20000)
        self.assertEqual(self._configuration.sampler_configuration.probability_floor, 0.002)
        self.assertEqual(self._configuration.metric_mel_protocol.sample_rate, 16000)

    def test_configuration_is_immutable(self) -> None:
        # Frozen settings cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.learning_rate = 0.5

    def test_unknown_field_is_rejected(self) -> None:
        # Extra fields are forbidden so silent typos cannot enter an experiment record.
        record: dict[str, object] = dict(self._configuration)
        record["unknown_setting"] = 1
        with self.assertRaises(ValidationError):
            LpcnetConfig(**record)

    def test_reference_record_round_trips_through_its_own_fields(self) -> None:
        # Control for the rejection above. Proving that the unperturbed record
        # rebuilds cleanly is what attributes that failure to the extra key
        # alone rather than to some unrelated reconstruction problem.
        self.assertEqual(LpcnetConfig(**dict(self._configuration)), self._configuration)


class LpcnetModuleSurfaceTest(unittest.TestCase):
    # Verifies module construction, the copy synthesis contract, and the
    # exposed properties. The synthesis assertions are structural: an
    # untrained autoregressive model produces noise, so what is established is
    # that analysis and resynthesis round-trip to the expected sample count,
    # that the result is finite, and that the input contract is enforced
    # before any sampling begins.
    def setUp(self) -> None:
        # Builds the miniature module in evaluation mode over half a thousand
        # reference samples. Evaluation mode is required rather than tidy: it
        # is what disables the network's training-time noise injection, so the
        # synthesis path under test is the one inference actually uses.
        torch.manual_seed(1234)
        self._configuration: LpcnetConfig = MiniatureModuleRecipe().build()
        self._module: Lpcnet = Lpcnet(self._configuration)
        self._module.eval()
        self._waveform: torch.Tensor = torch.randn(1, 512) * 0.05

    def test_automatic_optimization_is_disabled(self) -> None:
        # The reference post-step protocol runs after a manual optimizer step.
        self.assertFalse(self._module.automatic_optimization)

    def test_network_is_the_lpcnet_network(self) -> None:
        # The sparsifier and the sampler both resolve this attribute.
        self.assertIsInstance(self._module.network, LpcnetNetwork)

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The module exposes exactly the record it was constructed with.
        self.assertIs(self._module.configuration, self._configuration)

    def test_metric_protocol_is_published_for_the_measurement_stack(self) -> None:
        # LPCNet conditions on LPC analysis features rather than a mel, so it
        # publishes only the measurement protocol and no conditioning
        # counterpart. This asymmetry is what places the family outside the
        # Vocoder structural protocol the mel-conditioned architectures
        # satisfy.
        self.assertEqual(self._module.metric_mel_protocol, self._configuration.metric_mel_protocol)

    def test_copy_synthesis_returns_the_reference_sample_count(self) -> None:
        # Analysis frames are unrolled back into the same number of waveform
        # samples. Exact equality holds here only because the reference length
        # is a whole multiple of the frame size; an input that ended mid-frame
        # would come back shorter, since synthesis covers whole frames only.
        with torch.no_grad():
            synthesized: torch.Tensor = self._module(self._waveform)
        self.assertEqual(tuple(synthesized.shape), (1, 512))
        self.assertTrue(bool(torch.isfinite(synthesized).all()))

    def test_single_dimensional_waveform_is_promoted_to_a_batch(self) -> None:
        # A bare sample vector is treated as a one-element batch.
        with torch.no_grad():
            synthesized: torch.Tensor = self._module(torch.randn(512) * 0.05)
        self.assertEqual(tuple(synthesized.shape), (1, 512))

    def test_waveform_with_unsupported_rank_is_rejected(self) -> None:
        # A four-dimensional input has no defined sample axis and fails before synthesis.
        with self.assertRaisesRegex(ValueError, "Expected waveform shape"):
            self._module(torch.randn(1, 1, 1, 512))

    def test_prediction_returns_the_synthesized_waveform_mapping(self) -> None:
        # The measurement stack consumes the synthesized_waveform key.
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._module.predict_step({"waveform": self._waveform}, 0)
        self.assertIn("synthesized_waveform", prediction)
        self.assertEqual(tuple(prediction["synthesized_waveform"].shape), (1, 512))

    def test_prediction_rejects_a_non_mapping_batch(self) -> None:
        # The batch contract is enforced before any synthesis happens.
        with self.assertRaisesRegex(TypeError, "Expected dict batch"):
            self._module.predict_step(self._waveform, 0)

    def test_prediction_rejects_a_non_tensor_waveform(self) -> None:
        # A missing or mistyped waveform entry fails loudly rather than synthesizing noise.
        with self.assertRaisesRegex(TypeError, "must be Tensor"):
            self._module.predict_step({"waveform": [0.0, 1.0]}, 0)

    def test_test_step_returns_the_prediction_output(self) -> None:
        # Test evaluation measures exactly the prediction path, so both carry the same contract.
        with torch.no_grad():
            prediction: dict[str, torch.Tensor] = self._module.predict_step({"waveform": self._waveform}, 0)
            evaluation: dict[str, torch.Tensor] = self._module.test_step({"waveform": self._waveform}, 0)
        self.assertEqual(list(evaluation.keys()), list(prediction.keys()))
        self.assertEqual(tuple(evaluation["synthesized_waveform"].shape), (1, 512))
        self.assertEqual(
            tuple(evaluation["synthesized_waveform"].shape),
            tuple(prediction["synthesized_waveform"].shape)
        )


class LpcnetOptimizationDeclarationTest(unittest.TestCase):
    # Verifies the single-optimizer declaration of the reference recipe. The
    # absence of a scheduler is as much a contract as the optimizer's
    # presence: this family drives its learning rate from inside the training
    # step, so a declared schedule would contend with that rule for ownership
    # of the parameter group's rate.
    def setUp(self) -> None:
        # Builds the miniature module whose optimization declaration is under test.
        torch.manual_seed(1234)
        self._configuration: LpcnetConfig = MiniatureModuleRecipe().build()
        self._module: Lpcnet = Lpcnet(self._configuration)

    def test_exactly_one_optimizer_is_declared(self) -> None:
        # LPCNet is not adversarial, so it declares one optimizer and no scheduler object.
        declaration: OptimizationConfiguration = self._module.configure_optimizers()
        self.assertIsInstance(declaration.optimizer, torch.optim.Adam)
        self.assertIsNone(declaration.scheduler)

    def test_optimizer_carries_the_configured_learning_rate_and_moments(self) -> None:
        # The reference recipe uses low Adam moments with the configured learning rate.
        declaration: OptimizationConfiguration = self._module.configure_optimizers()
        optimizer: torch.optim.Optimizer = declaration.optimizer
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], self._configuration.learning_rate)
        self.assertEqual(
            optimizer.param_groups[0]["betas"],
            (self._configuration.adam_beta_1, self._configuration.adam_beta_2)
        )

    def test_optimizer_covers_the_network_parameters(self) -> None:
        # The single optimizer owns the whole network, which the post-step protocol then constrains.
        declaration: OptimizationConfiguration = self._module.configure_optimizers()
        optimizer: torch.optim.Optimizer = declaration.optimizer
        self.assertEqual(
            len(optimizer.param_groups[0]["params"]),
            len(list(self._module.network.parameters()))
        )
