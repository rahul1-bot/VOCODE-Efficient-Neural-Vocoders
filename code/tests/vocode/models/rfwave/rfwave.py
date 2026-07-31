# This module:
# 1. Verifies the RFWave configuration record: the reference optimization,
#    sampling, and protocol defaults, immutability, and the rejection of
#    unknown fields
# 2. Verifies the RFWave module surface: manual optimization, the synthesis
#    and prediction contracts, and the protocol properties
# 3. Verifies the optimization declaration: the single AdamW optimizer and
#    the cosine schedule with linear warmup, asserted at the schedule's
#    defining points
#
# Design decisions:
# - The module assertions run on a reduced backbone and band geometry while
#   keeping the reference mel protocol, because only the module wiring is
#   under test; the reference backbone values are asserted on the
#   configuration record itself
# - Training is never executed: training_step drives the harness optimizer
#   surface, which requires an attached trainer
# - The learning-rate schedule is a closed-form function of the step, so it
#   is asserted exactly at zero, at the end of warmup, at the midpoint, and
#   past the end of the cosine decay
#
# Author: Rahul Sawhney

import unittest
from collections.abc import Callable

import torch
from pydantic import ValidationError

from syntheticmind.core.optimizer import OptimizationConfiguration

from vocode.models.rfwave.flow import RfwaveRectifiedFlow
from vocode.models.rfwave.network import RfwaveNetworkConfig
from vocode.models.rfwave.rfwave import Rfwave, RfwaveConfig


class ReducedModuleRecipe:
    # Builds the reduced RFWave configuration used by the module-surface
    # assertions. Two reductions are applied for different reasons: the
    # backbone and band geometry shrink the per-pass cost, and the sampling
    # step count shrinks how many passes a synthesis performs. The second is
    # specific to this family, since every other module here synthesizes in
    # one pass regardless of configuration.
    def build(self) -> RfwaveConfig:
        # Returns the reduced recipe at two Euler steps rather than the
        # reference ten, which cuts every synthesis in this file to a fifth of
        # its cost. Two steps is retained rather than one because a
        # single-step schedule would not exercise the state carry-over between
        # iterations. The mel protocol is left at its reference default, so
        # the conditioning contract under test is the real one.
        return RfwaveConfig(
            network=RfwaveNetworkConfig(
                hidden_dimension=32,
                intermediate_dimension=64,
                layer_count=2,
                band_count=4,
                n_fft=64,
                hop_length=16,
                output_channels=24,
                left_overlap=2,
                right_overlap=2,
                pqmf_taps=16
            ),
            sampling_step_count=2
        )


class RfwaveReferenceConfigurationTest(unittest.TestCase):
    # Verifies the reference optimization, sampling, and protocol defaults of
    # the configuration. Because every field of this record already defaults
    # to its reference value, the defaults are the recipe rather than merely a
    # convenience, and these assertions are the only thing standing between an
    # accidental default change and a silently altered reproduction.
    def setUp(self) -> None:
        # Reads the default record, which is the published 24 kHz recipe. No
        # factory call is needed, since construction with no arguments already
        # yields the reference configuration.
        self._configuration: RfwaveConfig = RfwaveConfig()

    def test_optimization_defaults_match_the_reference_recipe(self) -> None:
        # One AdamW optimizer at 2e-4 with clip norm five drives the reference training run.
        self.assertEqual(self._configuration.learning_rate, 2e-4)
        self.assertEqual(self._configuration.adam_beta_1, 0.9)
        self.assertEqual(self._configuration.adam_beta_2, 0.999)
        self.assertEqual(self._configuration.gradient_clip_norm, 5.0)

    def test_schedule_defaults_match_the_reference_recipe(self) -> None:
        # The reference warms up over 20000 steps inside a 125000-step cosine schedule.
        self.assertEqual(self._configuration.scheduler_warmup_steps, 20000)
        self.assertEqual(self._configuration.scheduler_total_steps, 125000)

    def test_sampling_defaults_to_ten_euler_steps(self) -> None:
        # Inference integrates the learned flow in ten steps by default.
        self.assertEqual(self._configuration.sampling_step_count, 10)

    def test_mel_protocol_is_the_twenty_four_kilohertz_hundred_band_protocol(self) -> None:
        # RFWave conditions on the shared 24 kHz hundred-band protocol.
        self.assertEqual(self._configuration.mel_protocol.sample_rate, 24000)
        self.assertEqual(self._configuration.mel_protocol.n_mels, 100)
        self.assertEqual(self._configuration.network.mel_channels, 100)

    def test_named_factory_returns_the_reference_configuration(self) -> None:
        # The published 24 kHz recipe is the default record under a descriptive name.
        self.assertEqual(RfwaveConfig.bfs18_24khz(), self._configuration)

    def test_configuration_is_immutable(self) -> None:
        # Frozen settings cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.sampling_step_count = 2

    def test_unknown_field_is_rejected(self) -> None:
        # Extra fields are forbidden so silent typos cannot enter an experiment record.
        with self.assertRaises(ValidationError):
            RfwaveConfig(unknown_setting=1)


class RfwaveModuleSurfaceTest(unittest.TestCase):
    # Verifies module construction, the synthesis contract, and the exposed protocol properties.
    def setUp(self) -> None:
        # Builds the reduced module in evaluation mode plus its synthetic reference waveform.
        torch.manual_seed(1234)
        self._configuration: RfwaveConfig = ReducedModuleRecipe().build()
        self._module: Rfwave = Rfwave(self._configuration)
        self._module.eval()
        self._waveform: torch.Tensor = torch.randn(1, 4096) * 0.1

    def test_automatic_optimization_is_disabled(self) -> None:
        # The reference gradient hygiene wraps the optimizer step manually.
        # The reason differs from the adversarial families: this module drives
        # a single optimizer, but it must inspect gradients between backward
        # and the step in order to abandon a step whose gradients are not
        # finite, and the loop-owned path exposes no hook at that point.
        self.assertFalse(self._module.automatic_optimization)

    def test_network_is_the_rectified_flow(self) -> None:
        # The sampler and the losses both operate on this attribute.
        self.assertIsInstance(self._module.network, RfwaveRectifiedFlow)

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The module exposes exactly the record it was constructed with.
        self.assertIs(self._module.configuration, self._configuration)

    def test_metric_protocol_reuses_the_conditioning_protocol(self) -> None:
        # RFWave measures against the same mel protocol it conditions on.
        self.assertEqual(self._module.mel_protocol, self._configuration.mel_protocol)
        self.assertEqual(self._module.metric_mel_protocol, self._configuration.mel_protocol)

    def test_synthesize_integrates_the_flow_to_the_waveform(self) -> None:
        # Synthesis routes through the ODE sampler and spans one hop per frame
        # after the first, the leading frame being consumed by the transform's
        # edge behavior. Only shape and finiteness are asserted, because the
        # output of an untrained flow integrated from noise is noise; what can
        # be established is that the integration completes and lands in the
        # waveform domain at the expected length.
        mel: torch.Tensor = torch.randn(1, 100, 5)
        synthesized: torch.Tensor = self._module.synthesize(mel)
        self.assertEqual(tuple(synthesized.shape), (1, (5 - 1) * self._configuration.network.hop_length))
        self.assertTrue(bool(torch.isfinite(synthesized).all()))

    def test_prediction_returns_the_synthesized_waveform_mapping(self) -> None:
        # The measurement stack consumes the synthesized_waveform key.
        prediction: dict[str, torch.Tensor] = self._module.predict_step({"waveform": self._waveform}, 0)
        self.assertIn("synthesized_waveform", prediction)
        self.assertEqual(prediction["synthesized_waveform"].ndim, 2)
        self.assertEqual(prediction["synthesized_waveform"].shape[0], 1)

    def test_test_step_returns_the_prediction_output(self) -> None:
        # Test evaluation measures exactly the prediction path, so both carry
        # the same contract. Only shapes are compared rather than values,
        # because each call draws a fresh noise endpoint: two syntheses of the
        # same input legitimately differ, so exact equality would be the wrong
        # assertion for this family.
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


class RfwaveOptimizationDeclarationTest(unittest.TestCase):
    # Verifies the single optimizer and the cosine schedule with linear
    # warmup. The schedule is a closed-form function of the step count, so it
    # is asserted at the points where its value is known exactly rather than
    # by stepping it forward: the start, the end of warmup, the midpoint of
    # decay, and past the planned end. Those four points together pin both
    # segments and the clamping behavior beyond the plan.
    def setUp(self) -> None:
        # Builds the reduced module and collects its optimization declaration once.
        torch.manual_seed(1234)
        self._configuration: RfwaveConfig = ReducedModuleRecipe().build()
        self._module: Rfwave = Rfwave(self._configuration)
        self._declaration: OptimizationConfiguration = self._module.configure_optimizers()

    def test_exactly_one_optimizer_and_one_scheduler_are_declared(self) -> None:
        # Flow matching is not adversarial, so one optimizer carries the whole network.
        self.assertIsInstance(self._declaration.optimizer, torch.optim.AdamW)
        self.assertIsInstance(self._declaration.scheduler, torch.optim.lr_scheduler.LambdaLR)

    def test_optimizer_carries_the_configured_base_rate_and_moments(self) -> None:
        # The configured rate is the schedule's base rate, and the moments are the reference pair.
        optimizer: torch.optim.Optimizer = self._declaration.optimizer
        self.assertAlmostEqual(optimizer.param_groups[0]["initial_lr"], self._configuration.learning_rate)
        self.assertEqual(
            optimizer.param_groups[0]["betas"],
            (self._configuration.adam_beta_1, self._configuration.adam_beta_2)
        )

    def test_live_rate_starts_at_the_warmup_floor(self) -> None:
        # Attaching the schedule applies its first factor immediately, so the
        # live rate is zero before any step is taken while the base rate
        # asserted above is unchanged. Distinguishing the two is the point:
        # the base rate is what the schedule scales, and reading the live rate
        # at construction would otherwise look like a misconfigured optimizer.
        optimizer: torch.optim.Optimizer = self._declaration.optimizer
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0, places=9)

    def test_optimizer_covers_the_network_parameters(self) -> None:
        # The single optimizer owns the whole flow network, which the clip norm then bounds.
        optimizer: torch.optim.Optimizer = self._declaration.optimizer
        self.assertEqual(
            len(optimizer.param_groups[0]["params"]),
            len(list(self._module.network.parameters()))
        )

    def test_schedule_starts_at_zero_and_reaches_one_after_warmup(self) -> None:
        # Linear warmup rises from zero at the first step to the full rate at the warmup step.
        scheduler: torch.optim.lr_scheduler.LambdaLR = self._declaration.scheduler
        schedule_factor: Callable[[int], float] = scheduler.lr_lambdas[0]
        self.assertAlmostEqual(schedule_factor(0), 0.0, places=6)
        self.assertAlmostEqual(schedule_factor(self._configuration.scheduler_warmup_steps), 1.0, places=6)

    def test_schedule_halves_at_the_cosine_midpoint(self) -> None:
        # A cosine decay passes through one half at the middle of the post-warmup span.
        scheduler: torch.optim.lr_scheduler.LambdaLR = self._declaration.scheduler
        schedule_factor: Callable[[int], float] = scheduler.lr_lambdas[0]
        warmup_steps: int = self._configuration.scheduler_warmup_steps
        total_steps: int = self._configuration.scheduler_total_steps
        midpoint_step: int = warmup_steps + (total_steps - warmup_steps) // 2
        self.assertAlmostEqual(schedule_factor(midpoint_step), 0.5, places=5)

    def test_schedule_decays_to_zero_and_stays_there(self) -> None:
        # The cosine reaches zero at the total step count and is clamped beyond it.
        scheduler: torch.optim.lr_scheduler.LambdaLR = self._declaration.scheduler
        schedule_factor: Callable[[int], float] = scheduler.lr_lambdas[0]
        total_steps: int = self._configuration.scheduler_total_steps
        self.assertAlmostEqual(schedule_factor(total_steps), 0.0, places=6)
        self.assertAlmostEqual(schedule_factor(total_steps * 2), 0.0, places=6)
