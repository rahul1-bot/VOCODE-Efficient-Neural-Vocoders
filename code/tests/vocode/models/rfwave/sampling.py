# This module:
# 1. Verifies the Euler sampler's integration schedule: one velocity
#    evaluation per configured step, evaluated on the uniform time grid that
#    starts at zero and stops one step short of one
# 2. Verifies the integration itself: a zero velocity field leaves the noise
#    endpoint untouched, and a constant field advances the state by exactly
#    one unit of that field over the whole path
# 3. Verifies the sampling seam: the step count is the sampler's only knob,
#    and changing it changes the synthesis while the weights are held fixed
#
# Design decisions:
# - The schedule and integration assertions use flow subclasses that replace
#   only the velocity field, so the arithmetic under test is the solver's and
#   not the backbone's; every other component stays the real implementation
# - The noise endpoint is stochastic, so comparisons against an expected
#   state re-draw it under the same fixed seed rather than assuming
#   determinism
# - Geometry runs at four bands over a 64-point transform, and step counts
#   stay at or below four, keeping every integration inside the budget
#
# Author: Rahul Sawhney

import unittest
from typing import override

import torch

from vocode.models.rfwave.flow import RfwaveRectifiedFlow
from vocode.models.rfwave.network import RfwaveNetworkConfig
from vocode.models.rfwave.sampling import RfwaveOdeSampler


class ReducedFlowRecipe:
    # Builds the reduced rectified-flow configuration the sampler integrates
    # over. Reduction matters more here than elsewhere in the suite, because
    # every assertion runs the integration loop and therefore pays the model
    # cost once per step rather than once in total.
    def build(self) -> RfwaveNetworkConfig:
        # Returns the four-band 64-point geometry that keeps every integration
        # inside the budget. The output width is not free: one row carries a
        # band's real and imaginary slabs, so it must equal twice the slab
        # width implied by the transform size, band count, and overlaps. The
        # values here satisfy that relation, which is why they cannot be
        # varied independently of one another.
        return RfwaveNetworkConfig(
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
        )


class ZeroVelocityFlow(RfwaveRectifiedFlow):
    # Replaces the learned velocity field with zero and records the schedule
    # the sampler walks.
    #
    # Subclassing the real flow and overriding only the velocity prediction is
    # what makes these assertions about the solver rather than about the
    # backbone. Every other component stays the genuine implementation, so the
    # endpoint construction, the projection, and the reconstruction all behave
    # exactly as they do in a real synthesis; only the one quantity the solver
    # consumes is made predictable. A wholesale mock would have proved nothing
    # about how the solver interacts with the real flow.
    #
    # A zero field additionally makes the expected result exact: the state can
    # only remain where it started, whatever schedule the solver chooses.
    def __init__(self, configuration: RfwaveNetworkConfig) -> None:
        # Opens the evaluation counter and the observed-time log this flow
        # records into, so the schedule is observable from outside.
        super().__init__(configuration)
        self._evaluation_count: int = 0
        self._observed_times: list[float] = []

    @override
    def predict_velocity(
        self,
        noisy_state: torch.Tensor,
        time_values: torch.Tensor,
        expanded_mel: torch.Tensor,
        band_index: torch.Tensor
    ) -> torch.Tensor:
        # Records the integration time and returns no field, leaving the state
        # where it was. Only the first row's time is logged, which is
        # sufficient because the solver evaluates every row at the same time;
        # were that to change, the log would silently narrow rather than fail.
        self._evaluation_count: int = self._evaluation_count + 1
        self._observed_times.append(float(time_values[0]))
        return torch.zeros_like(noisy_state)

    @property
    def evaluation_count(self) -> int:
        # Returns how many times the sampler evaluated the field.
        return self._evaluation_count

    @property
    def observed_times(self) -> list[float]:
        # Returns a copy of the integration times, in the order the solver visited them.
        return list(self._observed_times)


class ConstantVelocityFlow(RfwaveRectifiedFlow):
    # Replaces the learned velocity field with one fixed field, so the
    # integral is known in closed form.
    #
    # A constant field is the one case where Euler integration is exact, and
    # exact in a way that does not depend on the step count: the step sizes
    # sum to the length of the unit interval, so the state advances by exactly
    # one field unit whether the solver takes one step or a hundred. That
    # makes it possible to assert the integration result against a closed-form
    # expectation rather than against a tolerance chosen to accommodate solver
    # error.
    #
    # This is also the property the architecture relies on. Training defines a
    # constant velocity along a straight path, so a well-trained field is
    # nearly constant and few-step integration is nearly exact; this fixture
    # tests the solver under precisely the conditions the design assumes.
    def __init__(self, configuration: RfwaveNetworkConfig, velocity_field: torch.Tensor) -> None:
        # Copies the field so the fixture cannot be perturbed through the
        # caller's tensor after construction.
        super().__init__(configuration)
        self._velocity_field: torch.Tensor = velocity_field.clone()

    @override
    def predict_velocity(
        self,
        noisy_state: torch.Tensor,
        time_values: torch.Tensor,
        expanded_mel: torch.Tensor,
        band_index: torch.Tensor
    ) -> torch.Tensor:
        # Returns the same field at every integration time, independent of the state.
        return self._velocity_field

    @property
    def velocity_field(self) -> torch.Tensor:
        # Returns a copy of the constant field the closed-form expectation is built from.
        return self._velocity_field.clone()


class RfwaveOdeSamplerScheduleTest(unittest.TestCase):
    # Verifies the number of solver steps and the uniform time grid they are
    # evaluated on. The evaluation count is the substantive assertion: each
    # evaluation is one full backbone forward pass, so it is the unit this
    # family's inference cost is counted in, and the sampling-step study's
    # cost axis is exactly this number.
    def setUp(self) -> None:
        # Builds the recording zero-velocity flow so only the solver's schedule is observed.
        torch.manual_seed(1234)
        self._configuration: RfwaveNetworkConfig = ReducedFlowRecipe().build()
        self._flow: ZeroVelocityFlow = ZeroVelocityFlow(self._configuration)
        self._flow.eval()
        self._mel: torch.Tensor = torch.randn(1, 100, 5)

    def test_default_step_count_matches_the_reference_sampler(self) -> None:
        # The reference synthesizes with ten Euler steps, which is the
        # baseline every reported cost and quality figure for this family is
        # measured against. Pinning it here means a change to the default
        # cannot silently reinterpret those figures.
        self.assertEqual(RfwaveOdeSampler().step_count, 10)

    def test_step_count_property_reports_the_configured_steps(self) -> None:
        # The step count is the sampler's only knob and is published for the run record.
        self.assertEqual(RfwaveOdeSampler(step_count=3).step_count, 3)

    def test_velocity_is_evaluated_once_per_step(self) -> None:
        # Euler integration performs exactly one field evaluation per interval.
        sampler: RfwaveOdeSampler = RfwaveOdeSampler(step_count=4)
        sampler.synthesize(self._flow, self._mel)
        self.assertEqual(self._flow.evaluation_count, 4)

    def test_velocity_is_evaluated_on_the_uniform_left_endpoint_grid(self) -> None:
        # The forward Euler scheme evaluates at the left endpoint of every
        # interval, so the final evaluation is at three quarters rather than
        # at one: the state reaches the path's end, but the field is never
        # queried there. Asserting the grid explicitly distinguishes forward
        # Euler from the midpoint or right-endpoint variants, which would
        # visit different times while producing an equally plausible waveform.
        sampler: RfwaveOdeSampler = RfwaveOdeSampler(step_count=4)
        sampler.synthesize(self._flow, self._mel)
        observed: list[float] = self._flow.observed_times
        expected: list[float] = [0.0, 0.25, 0.5, 0.75]
        step_index: int
        expected_time: float
        for step_index, expected_time in enumerate(expected):
            self.assertAlmostEqual(
                observed[step_index],
                expected_time,
                places=6,
                msg=f"Unexpected integration time at solver step {step_index}"
            )

    def test_single_step_evaluates_only_the_path_start(self) -> None:
        # A one-step schedule collapses to a single evaluation at time zero.
        sampler: RfwaveOdeSampler = RfwaveOdeSampler(step_count=1)
        sampler.synthesize(self._flow, self._mel)
        self.assertEqual(self._flow.evaluation_count, 1)
        self.assertAlmostEqual(self._flow.observed_times[0], 0.0, places=6)


class RfwaveOdeIntegrationTest(unittest.TestCase):
    # Verifies that the sampler integrates the velocity field and reconstructs
    # the waveform. Each expectation is re-derived under a repeated seed
    # rather than assumed, because the noise endpoint is drawn fresh on every
    # synthesis call; without resetting the seed the comparison would be
    # against a different starting point.
    def setUp(self) -> None:
        # Binds the reduced geometry and conditioning mel; each test builds
        # its own flow variant, since the variants differ in what they
        # override rather than in configuration.
        torch.manual_seed(1234)
        self._configuration: RfwaveNetworkConfig = ReducedFlowRecipe().build()
        self._mel: torch.Tensor = torch.randn(1, 100, 5)

    def test_zero_velocity_returns_the_noise_endpoint(self) -> None:
        # With no field to follow the state stays at the noise endpoint it started from.
        flow: ZeroVelocityFlow = ZeroVelocityFlow(self._configuration)
        flow.eval()
        sampler: RfwaveOdeSampler = RfwaveOdeSampler(step_count=3)
        torch.manual_seed(5)
        synthesized: torch.Tensor = sampler.synthesize(flow, self._mel)
        torch.manual_seed(5)
        with torch.no_grad():
            expected: torch.Tensor = flow.waveform_from_joint(flow.noise_endpoint(self._mel))
        self.assertTrue(bool(torch.allclose(synthesized, expected, atol=1e-5)))

    def test_constant_velocity_advances_the_state_by_one_field_unit(self) -> None:
        # The Euler steps sum to the full unit interval, so the state moves by
        # exactly the projected field regardless of how many steps were taken.
        # The expectation applies the projection before adding, mirroring the
        # solver: projecting is not optional bookkeeping but part of what is
        # actually integrated, so an expectation built from the raw field
        # would not match.
        field: torch.Tensor = torch.randn(4, 24, 5)
        flow: ConstantVelocityFlow = ConstantVelocityFlow(self._configuration, field)
        flow.eval()
        sampler: RfwaveOdeSampler = RfwaveOdeSampler(step_count=4)
        torch.manual_seed(5)
        synthesized: torch.Tensor = sampler.synthesize(flow, self._mel)
        torch.manual_seed(5)
        with torch.no_grad():
            noise_state: torch.Tensor = flow.noise_endpoint(self._mel)
            advanced_state: torch.Tensor = noise_state + flow.project_to_consistent_spectrum(
                flow.velocity_field
            )
            expected: torch.Tensor = flow.waveform_from_joint(advanced_state)
        self.assertTrue(
            bool(torch.allclose(synthesized, expected, atol=1e-4)),
            msg="Euler integration over a constant field must advance the state by exactly that field"
        )

    def test_synthesis_returns_the_full_rate_waveform(self) -> None:
        # Integration ends in the waveform domain at one hop per conditioning frame.
        flow: RfwaveRectifiedFlow = RfwaveRectifiedFlow(self._configuration)
        flow.eval()
        sampler: RfwaveOdeSampler = RfwaveOdeSampler(step_count=2)
        synthesized: torch.Tensor = sampler.synthesize(flow, self._mel)
        self.assertEqual(tuple(synthesized.shape), (1, 64))
        self.assertTrue(bool(torch.isfinite(synthesized).all()))

    def test_synthesis_runs_without_building_a_gradient_graph(self) -> None:
        # Sampling is inference, so the synthesized waveform carries no autograd graph.
        flow: RfwaveRectifiedFlow = RfwaveRectifiedFlow(self._configuration)
        flow.eval()
        sampler: RfwaveOdeSampler = RfwaveOdeSampler(step_count=2)
        synthesized: torch.Tensor = sampler.synthesize(flow, self._mel)
        self.assertFalse(synthesized.requires_grad)

    def test_batched_conditioning_is_supported(self) -> None:
        # Batched mels expand into band-major flow states and return one waveform per element.
        flow: RfwaveRectifiedFlow = RfwaveRectifiedFlow(self._configuration)
        flow.eval()
        sampler: RfwaveOdeSampler = RfwaveOdeSampler(step_count=2)
        synthesized: torch.Tensor = sampler.synthesize(flow, torch.randn(2, 100, 5))
        self.assertEqual(tuple(synthesized.shape), (2, 64))

    def test_step_count_changes_the_synthesis_at_fixed_weights(self) -> None:
        # The step count is the seam the sampling-step study exchanges, and
        # this assertion is what makes that study well posed. One flow object
        # is used for both syntheses and the seed is reset between them, so
        # the weights and the noise endpoint are identical; the only
        # difference is the integration schedule. That the outputs then differ
        # establishes the step count as a genuine inference-time variable
        # rather than a setting the solver happens to ignore, while the
        # matching shapes confirm the two remain comparable as audio.
        flow: RfwaveRectifiedFlow = RfwaveRectifiedFlow(self._configuration)
        flow.eval()
        torch.manual_seed(9)
        coarse_waveform: torch.Tensor = RfwaveOdeSampler(step_count=1).synthesize(flow, self._mel)
        torch.manual_seed(9)
        fine_waveform: torch.Tensor = RfwaveOdeSampler(step_count=4).synthesize(flow, self._mel)
        self.assertEqual(tuple(coarse_waveform.shape), tuple(fine_waveform.shape))
        self.assertFalse(
            bool(torch.equal(coarse_waveform, fine_waveform)),
            msg="A different integration schedule must produce a different synthesis"
        )
