# This module:
# 1. Verifies the frozen step-reduction setting: the closed step-count
#    domain, immutability, and strict typing
# 2. Verifies the step-count-to-variant-name decision for the three
#    registered reduction points
# 3. Verifies the sampler exchange itself: the baseline schedule captured
#    from the module, the reduced schedule installed in its place, the
#    untouched weights, and the recorded step-count trajectory
# 4. Verifies that the technique refuses modules carrying no ordinary
#    differential equation sampler
#
# Design decisions:
# - The baseline schedule under test is the sampler's own default of ten
#   Euler steps, so the recorded trajectory is the registered ten to
#   eight, four, and two reduction the study claims
# - No synthesis is executed: the technique exchanges a collaborator and
#   integration itself belongs to the flow model's own suite
# - The refusal path is exercised with both an absent sampler and a
#   wrong-typed one, because the recorded error names the offending type
#
# Author: Rahul Sawhney

import unittest
from typing import override

import torch
from pydantic import ValidationError
from torch import nn

from syntheticmind.core.module import Module
from syntheticmind.utilities.exceptions import MisconfigurationError

from vocode.models.rfwave.sampling import RfwaveOdeSampler
from vocode.optimization.sampling import OdeStepReduction, OdeStepReductionConfig


class TinyLinearNetwork(nn.Module):
    # Minimal stand-in for the trained flow network whose weights stay untouched.
    def __init__(self) -> None:
        # Builds the single projection whose weights must survive the exchange.
        super().__init__()
        self.projection: nn.Linear = nn.Linear(4, 4)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Projects the conditioning mel through the only layer of the network.
        return self.projection(mel)


class TinyHarnessModule(Module):
    # Minimal harness Module without any sampler collaborator.
    def __init__(self, network: nn.Module) -> None:
        # Binds the synthesis network without attaching any sampler collaborator.
        super().__init__()
        self.network: nn.Module = network

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Delegates synthesis to the bound network.
        return self.network(mel)


class SamplerCarryingModuleBuilder:
    # Builds a fresh harness module carrying the default ten-step sampler.
    # A fresh module per case matters because the technique records the baseline
    # schedule it found; a shared module would let one case observe another's
    # exchange and report a reduced schedule as the baseline.
    def build(self) -> TinyHarnessModule:
        # Seeds construction and attaches a default sampler under its private name.
        # The private attribute name is the seam the technique reads, so the
        # sampler is attached under exactly that name rather than a public one.
        # The sampler is left at its own default schedule, which is what makes
        # the recorded trajectory the registered ten-step starting point.
        #
        # Returns:
        #     A module carrying a dense network and a default ten-step
        #     sampler.
        torch.manual_seed(0)
        module: TinyHarnessModule = TinyHarnessModule(TinyLinearNetwork())
        setattr(module, "_sampler", RfwaveOdeSampler())
        return module


class OdeStepReductionConfigurationTest(unittest.TestCase):
    # Verifies the closed step-count domain of the intervention setting.
    def setUp(self) -> None:
        # Binds one accepted setting record for the mutation case to operate on.
        self._configuration: OdeStepReductionConfig = OdeStepReductionConfig(step_count=8)

    def test_registered_step_counts_are_accepted(self) -> None:
        # The registered reduction points halve the baseline schedule progressively.
        step_count: int
        for step_count in (8, 4, 2):
            self.assertEqual(OdeStepReductionConfig(step_count=step_count).step_count, step_count)

    def test_unregistered_step_counts_are_refused(self) -> None:
        # Only the three registered reduction points exist in the study.
        # The refused values are chosen to cover the ways an unregistered count
        # could look plausible: the baseline schedule itself, an intermediate
        # count, an odd count, and the degenerate single-step and zero-step
        # cases.
        step_count: int
        for step_count in (10, 6, 3, 1, 0):
            with self.assertRaises(ValidationError):
                OdeStepReductionConfig(step_count=step_count)

    def test_step_count_is_required(self) -> None:
        # The intervention has no default schedule; the reduction point is explicit.
        with self.assertRaises(ValidationError):
            OdeStepReductionConfig()

    def test_configuration_is_strictly_typed(self) -> None:
        # A string standing in for the step count is refused.
        with self.assertRaises(ValidationError):
            OdeStepReductionConfig(step_count="8")

    def test_configuration_rejects_mutation_and_unknown_fields(self) -> None:
        # The intervention setting is frozen and closed to extra keys.
        with self.assertRaises(ValidationError):
            self._configuration.step_count = 4
        with self.assertRaises(ValidationError):
            OdeStepReductionConfig(step_count=8, unknown_field=1)


class OdeStepReductionNamingTest(unittest.TestCase):
    # Verifies the step-count-to-variant-name decision.
    def setUp(self) -> None:
        # Binds a technique at the first reduction point for the dump assertion.
        self._technique: OdeStepReduction = OdeStepReduction(OdeStepReductionConfig(step_count=8))

    def test_each_registered_step_count_names_its_own_variant(self) -> None:
        # The variant name encodes the reduced step count.
        expected_names: dict[int, str] = {8: "ode_steps_8", 4: "ode_steps_4", 2: "ode_steps_2"}
        step_count: int
        expected_name: str
        for step_count, expected_name in expected_names.items():
            technique: OdeStepReduction = OdeStepReduction(OdeStepReductionConfig(step_count=step_count))
            self.assertEqual(technique.name, expected_name)

    def test_bound_configuration_is_reported_unchanged(self) -> None:
        # The supplied setting record is the one the technique reports.
        configuration: OdeStepReductionConfig = OdeStepReductionConfig(step_count=2)
        self.assertIs(OdeStepReduction(configuration).configuration, configuration)

    def test_configuration_dump_before_apply_leaves_the_baseline_unrecorded(self) -> None:
        # The starting point is evidence gathered at apply time, not a constant.
        self.assertEqual(
            self._technique.configuration_dump(),
            {
                "technique": "ode_step_reduction",
                "baseline_step_count": None,
                "reduced_step_count": 8
            }
        )


class OdeSamplerExchangeTest(unittest.TestCase):
    # Verifies the sampler exchange and the recorded step-count trajectory.
    def setUp(self) -> None:
        # Binds the builder producing a fresh sampler-carrying module per case.
        self._builder: SamplerCarryingModuleBuilder = SamplerCarryingModuleBuilder()

    def test_default_schedule_is_the_ten_step_baseline(self) -> None:
        # The intervention starts from the sampler's registered ten-step schedule.
        module: TinyHarnessModule = self._builder.build()
        self.assertEqual(getattr(module, "_sampler").step_count, 10)

    def test_each_reduction_point_installs_a_new_sampler_with_its_step_count(self) -> None:
        # The exchange replaces the collaborator rather than mutating the existing one.
        step_count: int
        for step_count in (8, 4, 2):
            module: TinyHarnessModule = self._builder.build()
            original_sampler: RfwaveOdeSampler = getattr(module, "_sampler")
            technique: OdeStepReduction = OdeStepReduction(OdeStepReductionConfig(step_count=step_count))
            returned: Module = technique.apply(module)
            reduced_sampler: RfwaveOdeSampler = getattr(returned, "_sampler")
            self.assertIs(returned, module)
            self.assertIsInstance(reduced_sampler, RfwaveOdeSampler)
            self.assertIsNot(reduced_sampler, original_sampler)
            self.assertEqual(reduced_sampler.step_count, step_count)
            self.assertEqual(original_sampler.step_count, 10)

    def test_recorded_trajectory_states_the_baseline_and_the_reduction(self) -> None:
        # The recipe records where the schedule started and where it landed.
        module: TinyHarnessModule = self._builder.build()
        technique: OdeStepReduction = OdeStepReduction(OdeStepReductionConfig(step_count=4))
        technique.apply(module)
        self.assertEqual(
            technique.configuration_dump(),
            {
                "technique": "ode_step_reduction",
                "baseline_step_count": 10,
                "reduced_step_count": 4
            }
        )

    def test_weights_are_never_touched_by_the_sampling_intervention(self) -> None:
        # The intervention is structural; the checkpoint under test stays identical.
        module: TinyHarnessModule = self._builder.build()
        original_weight: torch.Tensor = getattr(module.network, "projection").weight.detach().clone()
        original_network: nn.Module = module.network
        technique: OdeStepReduction = OdeStepReduction(OdeStepReductionConfig(step_count=2))
        technique.apply(module)
        self.assertIs(module.network, original_network)
        self.assertTrue(torch.equal(getattr(module.network, "projection").weight.detach(), original_weight))


class OdeSamplerRequirementTest(unittest.TestCase):
    # Verifies that the intervention refuses architectures without an integration schedule.
    def setUp(self) -> None:
        # Seeds construction and binds a module deliberately built without a sampler.
        torch.manual_seed(0)
        self._technique: OdeStepReduction = OdeStepReduction(OdeStepReductionConfig(step_count=8))
        self._module: TinyHarnessModule = TinyHarnessModule(TinyLinearNetwork())

    def test_module_without_a_sampler_is_refused(self) -> None:
        # Applying a sampling-schedule intervention to a single-pass architecture is meaningless.
        with self.assertRaisesRegex(MisconfigurationError, "NoneType"):
            self._technique.apply(self._module)

    def test_module_with_a_wrongly_typed_sampler_is_refused(self) -> None:
        # The recorded failure names the offending collaborator type.
        setattr(self._module, "_sampler", "ten-step-schedule")
        with self.assertRaisesRegex(MisconfigurationError, "str"):
            self._technique.apply(self._module)

    def test_refusal_states_the_required_architecture(self) -> None:
        # The refusal states that the technique belongs to the iterative-flow architecture.
        with self.assertRaisesRegex(MisconfigurationError, "iterative-flow architecture"):
            self._technique.apply(self._module)


if __name__ == "__main__":
    unittest.main()
