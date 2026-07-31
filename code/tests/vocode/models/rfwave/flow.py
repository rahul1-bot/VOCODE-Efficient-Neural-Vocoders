# This module:
# 1. Verifies the joint subband geometry: the overlapping gather, its exact
#    inverse placement, the per-band splitting, and the reported band count
#    and overlap width
# 2. Verifies the rectified-flow endpoints and training tuple: the noise and
#    target constructions, the shared time value per sample, the band index
#    tiling, and the straight-path interpolation identity
# 3. Verifies the loss-facing inversions: the velocity error waveform, the
#    implied endpoints, and the consistent-spectrum projection
#
# Design decisions:
# - The flow is exercised in evaluation mode wherever an endpoint is
#   recomputed, because the equalizer updates running subband statistics in
#   training mode and would otherwise return a different projection on the
#   second call; that training-mode behavior is asserted in the network tests
# - The straight-path identity is checked by reconstructing the data endpoint
#   from the noisy state and the velocity target, which is exact arithmetic
#   up to float32 rounding rather than a learned approximation
# - Geometry runs at four bands over a 64-point transform so the slab
#   arithmetic stays checkable by hand; the reference geometry is asserted in
#   the network configuration tests
#
# Author: Rahul Sawhney

import unittest

import torch

from vocode.models.rfwave.flow import RfwaveRectifiedFlow, RfwaveTrainTuple
from vocode.models.rfwave.network import RfwaveNetworkConfig


class ReducedFlowRecipe:
    # Builds the reduced rectified-flow configuration used by the geometry assertions.
    def build(self) -> RfwaveNetworkConfig:
        # Returns the four-band 64-point geometry whose slab arithmetic stays checkable by hand.
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


class RfwaveSubbandGeometryTest(unittest.TestCase):
    # Verifies the overlapping subband gather and its exact inverse placement.
    # The inversion is the load-bearing property of the whole decomposition:
    # the flow operates entirely on slabs, so a placement that did not
    # perfectly undo the gather would corrupt every reconstruction by an
    # amount no other assertion in the suite would isolate.
    def setUp(self) -> None:
        # Builds the flow in evaluation mode and one analysed spectrum to
        # gather and replace. The spectrum is produced by the flow's own
        # transform rather than fabricated, so it is a genuinely consistent
        # spectrum with the exact bin count the geometry expects.
        torch.manual_seed(1234)
        self._configuration: RfwaveNetworkConfig = ReducedFlowRecipe().build()
        self._flow: RfwaveRectifiedFlow = RfwaveRectifiedFlow(self._configuration)
        self._flow.eval()
        self._spectrum: torch.Tensor = self._flow.spectral_transform.stft(torch.randn(2, 128))

    def test_band_count_and_overlap_are_reported(self) -> None:
        # The sampler and the losses read the band geometry from these properties.
        self.assertEqual(self._flow.band_count, 4)
        self.assertEqual(self._flow.overlap, 4)

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The flow exposes exactly the record it was constructed with.
        self.assertIs(self._flow.configuration, self._configuration)

    def test_gather_widens_each_band_by_the_overlap(self) -> None:
        # Every band carries its own bins plus the shared overlap, in both spectral parts.
        with torch.no_grad():
            joint: torch.Tensor = self._flow.gather_joint_subbands(self._spectrum)
        self.assertEqual(tuple(joint.shape), (2, 96, 9))

    def test_placement_inverts_the_gather_exactly(self) -> None:
        # Trimming the overlaps restores the stacked spectrum exactly.
        # Exact equality is asserted rather than closeness, and it is
        # achievable because the pair performs only slicing, padding, and
        # concatenation, never arithmetic; any tolerance here would mask a
        # genuine defect, since the operations cannot introduce rounding at
        # all. This is also what proves the asymmetric padding and the last
        # band's extra bin cooperate exactly rather than approximately.
        with torch.no_grad():
            joint: torch.Tensor = self._flow.gather_joint_subbands(self._spectrum)
            placed: torch.Tensor = self._flow.place_joint_subbands(joint)
        self.assertTrue(
            bool(torch.equal(placed, self._spectrum)),
            msg="The subband placement must be the exact inverse of the overlapping gather"
        )

    def test_split_returns_one_slab_pair_per_band(self) -> None:
        # The loss consumes per-band real and imaginary slabs of the flattened prediction.
        joint: torch.Tensor = torch.randn(2 * 4, 24, 9)
        with torch.no_grad():
            real_slabs, imaginary_slabs = self._flow.split_band_lists(joint)
        self.assertEqual(len(real_slabs), 4)
        self.assertEqual(len(imaginary_slabs), 4)
        self.assertEqual(tuple(real_slabs[0].shape), (2, 12, 9))


class RfwaveFlowEndpointTest(unittest.TestCase):
    # Verifies the noise and data endpoints and the flattened per-band
    # flow-state layout. The two endpoints have deliberately opposite
    # determinism properties, and both are asserted: the noise endpoint must
    # vary between calls, since a flow trained from a fixed starting point
    # would learn a single trajectory, while the data endpoint must not, since
    # it is the fixed destination the velocity target is defined against.
    def setUp(self) -> None:
        # Builds the flow in evaluation mode so the equalizer statistics stay
        # frozen between calls. This is required rather than tidy: in training
        # mode the equalizer updates its running statistics on every call, so
        # a second target endpoint would be computed under different
        # statistics and the determinism assertion below would fail for a
        # reason unrelated to the endpoint construction.
        torch.manual_seed(1234)
        self._flow: RfwaveRectifiedFlow = RfwaveRectifiedFlow(ReducedFlowRecipe().build())
        self._flow.eval()
        self._mel: torch.Tensor = torch.randn(1, 100, 5)
        self._waveform: torch.Tensor = torch.randn(1, 64)

    def test_noise_endpoint_is_flattened_over_bands(self) -> None:
        # The flow state stacks bands into the batch axis so one backbone serves them all.
        with torch.no_grad():
            noise_state: torch.Tensor = self._flow.noise_endpoint(self._mel)
        self.assertEqual(tuple(noise_state.shape), (4, 24, 5))

    def test_noise_endpoint_frame_count_follows_the_conditioning_mel(self) -> None:
        # The noise waveform spans one hop per conditioning frame after the first.
        with torch.no_grad():
            noise_state: torch.Tensor = self._flow.noise_endpoint(torch.randn(1, 100, 9))
        self.assertEqual(noise_state.shape[2], 9)

    def test_noise_endpoint_is_stochastic(self) -> None:
        # The endpoint is Gaussian noise, so two draws differ.
        with torch.no_grad():
            first_state: torch.Tensor = self._flow.noise_endpoint(self._mel)
            second_state: torch.Tensor = self._flow.noise_endpoint(self._mel)
        self.assertFalse(bool(torch.equal(first_state, second_state)))

    def test_target_endpoint_is_deterministic_in_evaluation_mode(self) -> None:
        # With the equalizer statistics frozen the data endpoint is a pure function of the waveform.
        with torch.no_grad():
            first_state: torch.Tensor = self._flow.target_endpoint(self._waveform)
            second_state: torch.Tensor = self._flow.target_endpoint(self._waveform)
        self.assertTrue(bool(torch.equal(first_state, second_state)))

    def test_target_endpoint_shares_the_flow_state_layout(self) -> None:
        # Both endpoints live in the same flattened band space, which is the
        # precondition for interpolating between them at all: the path is a
        # convex combination taken elementwise, so a shape disagreement would
        # either broadcast silently into something meaningless or fail deep
        # inside the tuple construction.
        with torch.no_grad():
            target_state: torch.Tensor = self._flow.target_endpoint(self._waveform)
            noise_state: torch.Tensor = self._flow.noise_endpoint(self._mel)
        self.assertEqual(tuple(target_state.shape), tuple(noise_state.shape))

    def test_waveform_reconstruction_spans_one_hop_per_frame(self) -> None:
        # Placing the slabs back and inverting the transform returns the full-rate waveform.
        with torch.no_grad():
            target_state: torch.Tensor = self._flow.target_endpoint(self._waveform)
            reconstructed: torch.Tensor = self._flow.waveform_from_joint(target_state)
        self.assertEqual(tuple(reconstructed.shape), (1, 64))


class RfwaveTrainTupleTest(unittest.TestCase):
    # Verifies the joint-parallel training tuple and the straight-path
    # interpolation it encodes. The row-ordering assertions matter as much as
    # the arithmetic one: the time values, the band indices, and the expanded
    # mel are built by three different expansion operations, and if any
    # disagreed with the others the backbone would receive a row conditioned
    # on the wrong band or the wrong utterance while every shape still
    # matched.
    def setUp(self) -> None:
        # Builds one two-element training tuple, giving eight band rows. Two
        # samples is the minimum that makes the ordering observable, since a
        # single sample cannot distinguish the two expansion patterns.
        torch.manual_seed(1234)
        self._flow: RfwaveRectifiedFlow = RfwaveRectifiedFlow(ReducedFlowRecipe().build())
        self._flow.eval()
        self._mel: torch.Tensor = torch.randn(2, 100, 5)
        self._waveform: torch.Tensor = torch.randn(2, 64)
        self._tuple: RfwaveTrainTuple = self._flow.build_train_tuple(self._mel, self._waveform)

    def test_tuple_carries_the_flattened_band_batch(self) -> None:
        # Every field is expanded to one row per band of every batch element.
        self.assertIsInstance(self._tuple, RfwaveTrainTuple)
        self.assertEqual(tuple(self._tuple.noisy_state.shape), (8, 24, 5))
        self.assertEqual(tuple(self._tuple.velocity_target.shape), (8, 24, 5))
        self.assertEqual(tuple(self._tuple.expanded_mel.shape), (8, 100, 5))
        self.assertEqual(tuple(self._tuple.time_values.shape), (8,))
        self.assertEqual(tuple(self._tuple.band_index.shape), (8,))

    def test_time_value_is_shared_across_the_bands_of_one_sample(self) -> None:
        # All bands of a sample advance together, which is what joint-parallel training means.
        grouped_times: torch.Tensor = self._tuple.time_values.view(2, 4)
        sample_index: int
        for sample_index in range(2):
            self.assertTrue(
                bool(torch.equal(grouped_times[sample_index], grouped_times[sample_index][0].expand(4))),
                msg=f"Sample {sample_index} must carry one flow time across all of its bands"
            )

    def test_time_values_lie_inside_the_unit_interval(self) -> None:
        # Flow time is sampled uniformly along the straight path.
        self.assertGreaterEqual(float(self._tuple.time_values.min()), 0.0)
        self.assertLessEqual(float(self._tuple.time_values.max()), 1.0)

    def test_band_index_tiles_the_band_range_per_sample(self) -> None:
        # The backbone distinguishes bands through this index, so its ordering is contractual.
        self.assertTrue(bool(torch.equal(self._tuple.band_index, torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))))

    def test_expanded_mel_repeats_each_sample_across_its_bands(self) -> None:
        # Every band of a sample conditions on that sample's mel. Both halves
        # of the assertion are needed: rows within one sample's band block
        # must agree, and rows across the block boundary must not, which
        # together pin the mel to the block-repeating expansion rather than to
        # the cycling one the band index uses.
        self.assertTrue(bool(torch.equal(self._tuple.expanded_mel[0], self._tuple.expanded_mel[3])))
        self.assertFalse(bool(torch.equal(self._tuple.expanded_mel[0], self._tuple.expanded_mel[4])))

    def test_noisy_state_lies_on_the_straight_path(self) -> None:
        # The data endpoint is recovered from the noisy state and the velocity
        # target, which is the identity the whole formulation rests on:
        # advancing the state by the remaining fraction of the constant
        # velocity lands exactly on the destination. This is what guarantees
        # that a network predicting the target perfectly would be integrated
        # to the true data in one step, and therefore why few-step synthesis
        # is expected to work at all.
        #
        # The comparison uses a tolerance only because the two sides traverse
        # different float32 operation orders; the identity itself is exact.
        time_shaped: torch.Tensor = self._tuple.time_values.view(-1, 1, 1)
        with torch.no_grad():
            implied_target: torch.Tensor = self._tuple.noisy_state + (1.0 - time_shaped) * self._tuple.velocity_target
            actual_target: torch.Tensor = self._flow.target_endpoint(self._waveform)
        self.assertTrue(
            bool(torch.allclose(implied_target, actual_target, atol=1e-5)),
            msg="The noised state must interpolate linearly between the noise and data endpoints"
        )


class RfwaveLossInversionTest(unittest.TestCase):
    # Verifies the inversions the losses consume: the velocity error waveform,
    # the implied endpoints, and the consistency projection. Each is checked
    # at the point where its behavior is known exactly rather than at a random
    # input, which is why the perfect-prediction case recurs: feeding the
    # target back as the prediction makes the expected result zero or exact
    # agreement, with no tolerance to negotiate.
    def setUp(self) -> None:
        # Builds one single-element training tuple whose velocity target
        # doubles as the perfect prediction in the assertions below.
        torch.manual_seed(1234)
        self._flow: RfwaveRectifiedFlow = RfwaveRectifiedFlow(ReducedFlowRecipe().build())
        self._flow.eval()
        self._mel: torch.Tensor = torch.randn(1, 100, 5)
        self._waveform: torch.Tensor = torch.randn(1, 64)
        self._tuple: RfwaveTrainTuple = self._flow.build_train_tuple(self._mel, self._waveform)

    def test_velocity_error_waveform_vanishes_for_a_perfect_prediction(self) -> None:
        # A prediction equal to the target places zero error back onto the waveform.
        with torch.no_grad():
            error_waveform: torch.Tensor = self._flow.velocity_error_waveform(
                self._tuple.velocity_target,
                self._tuple.velocity_target
            )
        self.assertEqual(tuple(error_waveform.shape), (1, 64))
        self.assertEqual(float(error_waveform.abs().max()), 0.0)

    def test_velocity_error_waveform_is_non_zero_for_a_wrong_prediction(self) -> None:
        # A mismatched prediction produces a measurable waveform-domain error.
        # This is the necessary counterpart to the vanishing case above:
        # without it, a transformation that returned zero unconditionally
        # would satisfy that assertion and silently reduce the primary
        # training term to a constant.
        with torch.no_grad():
            error_waveform: torch.Tensor = self._flow.velocity_error_waveform(
                torch.zeros_like(self._tuple.velocity_target),
                self._tuple.velocity_target
            )
        self.assertGreater(float(error_waveform.abs().max()), 0.0)

    def test_implied_endpoints_agree_for_a_perfect_prediction(self) -> None:
        # With the exact velocity the implied and true data endpoints
        # coincide, and do so exactly: both are computed by the same
        # expression from the same inputs, so any difference would indicate
        # the recovery treats the prediction and the target asymmetrically
        # rather than merely rounding differently.
        with torch.no_grad():
            predicted_endpoint, target_endpoint = self._flow.implied_endpoints(
                self._tuple.noisy_state,
                self._tuple.time_values,
                self._tuple.velocity_target,
                self._tuple.velocity_target
            )
        self.assertTrue(bool(torch.equal(predicted_endpoint, target_endpoint)))

    def test_implied_endpoints_are_placed_in_the_full_spectrum(self) -> None:
        # The magnitude loss operates on the reassembled spectrum rather than on slabs.
        with torch.no_grad():
            predicted_endpoint, target_endpoint = self._flow.implied_endpoints(
                self._tuple.noisy_state,
                self._tuple.time_values,
                self._tuple.velocity_target,
                self._tuple.velocity_target
            )
        self.assertEqual(tuple(predicted_endpoint.shape), (1, 66, 5))
        self.assertEqual(tuple(target_endpoint.shape), (1, 66, 5))

    def test_consistent_projection_preserves_the_flow_state_layout(self) -> None:
        # The projection round-trips through the waveform domain without changing the layout.
        with torch.no_grad():
            projected: torch.Tensor = self._flow.project_to_consistent_spectrum(self._tuple.noisy_state)
        self.assertEqual(tuple(projected.shape), tuple(self._tuple.noisy_state.shape))

    def test_consistent_projection_is_idempotent(self) -> None:
        # Projecting an already consistent spectrum leaves it where it is,
        # which is the defining property of a projection and the reason the
        # sampler can apply it at every step without accumulating drift. Were
        # it not idempotent, repeated application across ten steps would move
        # the state systematically rather than merely correcting it.
        with torch.no_grad():
            once_projected: torch.Tensor = self._flow.project_to_consistent_spectrum(self._tuple.noisy_state)
            twice_projected: torch.Tensor = self._flow.project_to_consistent_spectrum(once_projected)
        self.assertTrue(
            bool(torch.allclose(twice_projected, once_projected, atol=1e-5)),
            msg="A projection onto the consistent-spectrum manifold must be idempotent"
        )
