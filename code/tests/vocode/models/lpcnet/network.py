# This module:
# 1. Verifies the LPCNet building blocks: the fractional signal embedding,
#    the dual fully connected output gate, the hierarchical tree-probability
#    expansion, and the frame-to-sample linear prediction
# 2. Verifies the LPCNet network surface: the frame-rate conditioning
#    encoder, the teacher-forced sample-rate transformation, and the
#    training-noise injection that is active only in training mode
#
# Design decisions:
# - The tree expansion, the linear prediction, and the embedding
#   interpolation are exact deterministic mathematics, so they are asserted
#   against hand-computed values rather than tolerance bands
# - The network is exercised at miniature dimensions over a handful of
#   frames; the reference geometry (384-unit recurrent core at frame size
#   160) is a training-scale object and is covered by the module tests
# - Autoregressive synthesis is not reached from this module, so no
#   sample-rate loop runs here
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError

from vocode.models.lpcnet.network import (
    LpcnetDualFullyConnected,
    LpcnetFractionalEmbedding,
    LpcnetLinearPrediction,
    LpcnetNetwork,
    LpcnetNetworkOutput,
    LpcnetTreeProbability,
)
from vocode.transforms.lpc import LpcnetMuLaw


class MiniatureNetworkRecipe:
    # Builds the miniature LPCNet network used by the conditioning and forward
    # assertions. The noise level is the recipe's only parameter because it is
    # the only setting the tests need to vary: everything else is reduced once
    # and held fixed.
    def __init__(self, training_noise_standard_deviation: float) -> None:
        # Binds the teacher-forcing noise level this recipe varies.
        #
        # Args:
        #     training_noise_standard_deviation: Noise injected into the
        #         teacher-forced inputs. Zero makes the forward pass
        #         deterministic in either mode, which the surface assertions
        #         require; the reference level is used by the noise
        #         assertions.
        self._training_noise_standard_deviation: float = training_noise_standard_deviation

    def build(self) -> LpcnetNetwork:
        # Returns the miniature network at the bound noise level. Frame size
        # two is the smallest value that still exercises the frame-to-sample
        # expansion, since a frame size of one would make the repeat a no-op
        # and hide any error in it.
        return LpcnetNetwork(
            feature_dimension=6,
            condition_dimension=8,
            embedding_dimension=4,
            first_gru_dimension=8,
            second_gru_dimension=4,
            lpc_order=2,
            frame_size=2,
            training_noise_standard_deviation=self._training_noise_standard_deviation
        )


class FrameFeatureBuilder:
    # Builds the frame-rate conditioning inputs the miniature network consumes.
    def __init__(self, batch_size: int, frame_count: int) -> None:
        # Binds the batch and frame geometry every emitted input carries.
        self._batch_size: int = batch_size
        self._frame_count: int = frame_count

    def build_conditioning(self) -> torch.Tensor:
        # Returns one six-dimensional feature vector per frame.
        return torch.randn(self._batch_size, self._frame_count, 6)

    def build_pitch_index(self) -> torch.Tensor:
        # Returns one pitch-table index per frame, inside the 256-entry table.
        return torch.randint(0, 256, (self._batch_size, self._frame_count))

    def build_lpc_coefficients(self) -> torch.Tensor:
        # Returns small second-order coefficients so the prediction filter stays stable.
        return torch.randn(self._batch_size, self._frame_count, 2) * 0.1

    def build_target_samples(self) -> torch.Tensor:
        # Returns the teacher-forced sample stream at frame size two, scaled
        # to the int16 domain. The scaling matters because the mu-law
        # companding constants assume that domain: unit-scaled samples would
        # all compand to indices clustered at the centre of the table, so the
        # embedding would be exercised over a negligible part of its range.
        return torch.randn(self._batch_size, self._frame_count * 2, 1) * 1000.0


class LpcnetFractionalEmbeddingTest(unittest.TestCase):
    # Verifies the interpolating signal embedding and its deterministic
    # reference initialization. The interpolation is asserted at three
    # positions that together characterize it: a whole index, which must
    # return a table row untouched, a half index, which must return the exact
    # midpoint, and both saturation bounds. The half-index case is the one
    # that matters most in practice, since companded mu-law values are almost
    # never integral.
    def setUp(self) -> None:
        # Builds the 256-entry embedding table at width four.
        torch.manual_seed(1234)
        self._embedding: LpcnetFractionalEmbedding = LpcnetFractionalEmbedding(256, 4)

    def test_embedding_appends_the_feature_dimension(self) -> None:
        # Each mu-law index expands into one embedding row.
        indices: torch.Tensor = torch.zeros(2, 5, 3)
        with torch.no_grad():
            embedded: torch.Tensor = self._embedding(indices)
        self.assertEqual(tuple(embedded.shape), (2, 5, 3, 4))

    def test_integer_index_returns_the_table_row(self) -> None:
        # A whole index carries no interpolation weight onto the next row.
        with torch.no_grad():
            embedded: torch.Tensor = self._embedding(torch.tensor([[7.0]]))
        self.assertTrue(bool(torch.allclose(embedded[0, 0], self._embedding.weight[7])))

    def test_half_index_returns_the_midpoint_of_adjacent_rows(self) -> None:
        # Continuous mu-law inputs interpolate linearly between neighbouring rows.
        with torch.no_grad():
            embedded: torch.Tensor = self._embedding(torch.tensor([[7.5]]))
        expected: torch.Tensor = 0.5 * (self._embedding.weight[7] + self._embedding.weight[8])
        self.assertTrue(bool(torch.allclose(embedded[0, 0], expected, atol=1e-6)))

    def test_index_below_the_table_is_clamped_to_the_first_row(self) -> None:
        # Out-of-range inputs saturate rather than indexing outside the table.
        with torch.no_grad():
            embedded: torch.Tensor = self._embedding(torch.tensor([[-9.0]]))
        self.assertTrue(bool(torch.allclose(embedded[0, 0], self._embedding.weight[0])))

    def test_index_above_the_table_is_clamped_to_the_last_row(self) -> None:
        # The upper saturation lands exactly on the final table row. This is
        # the boundary the second clamp exists for: the lower row is held at
        # the penultimate entry so that reading the row above it stays in
        # range, and the interpolation weight then resolves to one, landing on
        # the last row rather than indexing past the table.
        with torch.no_grad():
            embedded: torch.Tensor = self._embedding(torch.tensor([[999.0]]))
        self.assertTrue(bool(torch.allclose(embedded[0, 0], self._embedding.weight[255])))

    def test_initialization_is_reproducible_across_instances(self) -> None:
        # The reference table is seeded internally, so two instances start
        # identically regardless of what the global generator has produced.
        # That independence is what makes this table's contribution to a run
        # reproducible even if unrelated code changes how much randomness is
        # drawn before construction.
        other_embedding: LpcnetFractionalEmbedding = LpcnetFractionalEmbedding(256, 4)
        self.assertTrue(bool(torch.equal(self._embedding.weight, other_embedding.weight)))

    def test_initialization_carries_the_signal_ramp(self) -> None:
        # The reference initialization biases the table along the amplitude
        # ordering, so neighboring mu-law indices begin as neighbors in
        # embedding space. A wide table is used because the ramp is a shared
        # component added to independent noise: at width four the noise can
        # dominate the comparison, whereas averaging over sixty-four
        # dimensions leaves the ramp clearly visible.
        wide_embedding: LpcnetFractionalEmbedding = LpcnetFractionalEmbedding(256, 64)
        table: torch.Tensor = wide_embedding.weight.detach()
        self.assertGreater(float(table[255].mean()), float(table[0].mean()))


class LpcnetDualFullyConnectedTest(unittest.TestCase):
    # Verifies the two-branch output gate that produces the binary-tree node activations.
    def setUp(self) -> None:
        # Builds the output gate mapping six features onto the 256 tree nodes.
        torch.manual_seed(1234)
        self._gate: LpcnetDualFullyConnected = LpcnetDualFullyConnected(6, 256)

    def test_output_width_matches_the_node_count(self) -> None:
        # The gate emits one activation per binary-tree node.
        with torch.no_grad():
            activations: torch.Tensor = self._gate(torch.randn(2, 3, 6))
        self.assertEqual(tuple(activations.shape), (2, 3, 256))

    def test_activations_are_probabilities(self) -> None:
        # The sigmoid gate keeps every node activation inside the open unit interval.
        with torch.no_grad():
            activations: torch.Tensor = self._gate(torch.randn(2, 3, 6))
        self.assertGreater(float(activations.min()), 0.0)
        self.assertLess(float(activations.max()), 1.0)

    def test_branch_factors_start_at_one(self) -> None:
        # Both branch weights begin balanced, matching the reference initialization.
        self.assertTrue(bool(torch.equal(self._gate.first_factor, torch.ones(256))))
        self.assertTrue(bool(torch.equal(self._gate.second_factor, torch.ones(256))))


class LpcnetTreeProbabilityTest(unittest.TestCase):
    # Verifies that the hierarchical node activations expand into a normalized
    # 256-level distribution. Normalization is the load-bearing property: the
    # sampler draws from this output directly, so an expansion that did not
    # sum to one would bias sampling without raising anywhere. It holds
    # structurally rather than by construction, since no explicit division is
    # performed; every leaf receives exactly one factor from each level, and
    # each level's two branch probabilities are complementary.
    def setUp(self) -> None:
        # Builds the hierarchical expansion that turns node activations into a distribution.
        torch.manual_seed(1234)
        self._expansion: LpcnetTreeProbability = LpcnetTreeProbability()

    def test_expansion_produces_one_probability_per_mulaw_level(self) -> None:
        # The eight-level tree covers the full 256-level mu-law alphabet.
        node_outputs: torch.Tensor = torch.rand(2, 3, 256)
        with torch.no_grad():
            probabilities: torch.Tensor = self._expansion(node_outputs)
        self.assertEqual(tuple(probabilities.shape), (2, 3, 256))

    def test_expansion_is_a_normalized_distribution(self) -> None:
        # A complete binary tree of independent splits sums to one at every position.
        node_outputs: torch.Tensor = torch.rand(2, 3, 256)
        with torch.no_grad():
            probabilities: torch.Tensor = self._expansion(node_outputs)
        self.assertTrue(bool(torch.allclose(probabilities.sum(dim=-1), torch.ones(2, 3), atol=1e-5)))

    def test_expansion_is_non_negative(self) -> None:
        # Products of split probabilities can never be negative.
        node_outputs: torch.Tensor = torch.rand(2, 3, 256)
        with torch.no_grad():
            probabilities: torch.Tensor = self._expansion(node_outputs)
        self.assertGreaterEqual(float(probabilities.min()), 0.0)

    def test_balanced_nodes_expand_to_the_uniform_distribution(self) -> None:
        # Every split at one half assigns identical mass to all 256 levels.
        # This is the one input whose expansion is known in closed form, so it
        # is the only case that can be checked against an exact expected value
        # rather than a summation property; the tight tolerance reflects that
        # the result is a product of exact halves rather than an approximation.
        node_outputs: torch.Tensor = torch.full((1, 1, 256), 0.5)
        with torch.no_grad():
            probabilities: torch.Tensor = self._expansion(node_outputs)
        self.assertTrue(
            bool(torch.allclose(probabilities, torch.full((1, 1, 256), 1.0 / 256.0), atol=1e-9)),
            msg="Balanced tree nodes must yield the uniform excitation distribution"
        )


class LpcnetLinearPredictionTest(unittest.TestCase):
    # Verifies the per-sample linear prediction driven by frame-rate
    # coefficients. This component is exact deterministic arithmetic with no
    # parameters, so it is asserted against hand-computed values rather than
    # against shape or tolerance properties; that also makes it the one place
    # where the tap ordering and the sign convention can be pinned precisely.
    def setUp(self) -> None:
        # Fixes four samples and two frames of coefficients so the prediction
        # is hand-computable. The two frames carry deliberately different
        # coefficients, which is what makes the frame-to-sample expansion
        # observable: samples one and two must use the first frame's
        # coefficients and samples three and four the second's, so a broken
        # expansion changes the expected values rather than merely their
        # shape. The second frame zeroes its lag coefficient, isolating the
        # leading tap.
        self._prediction: LpcnetLinearPrediction = LpcnetLinearPrediction(lpc_order=2, frame_size=2)
        self._samples: torch.Tensor = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        self._coefficients: torch.Tensor = torch.tensor([[[0.5, 0.25], [1.0, 0.0]]])

    def test_prediction_keeps_the_sample_layout(self) -> None:
        # One prediction is produced per input sample.
        with torch.no_grad():
            predicted: torch.Tensor = self._prediction(self._samples, self._coefficients)
        self.assertEqual(tuple(predicted.shape), (1, 4, 1))

    def test_prediction_is_the_negated_lagged_combination(self) -> None:
        # Coefficients are repeated across the frame and applied to the
        # sequence's own current position and its lags. The leading tap
        # multiplies the current entry of the supplied sequence rather than
        # the one before it, which is correct because the caller passes an
        # already-shifted sequence: inside the network, position n of that
        # argument holds the true sample n minus one. The negation follows the
        # inverse-filter convention the analysis produces its coefficients in.
        with torch.no_grad():
            predicted: torch.Tensor = self._prediction(self._samples, self._coefficients)
        expected: torch.Tensor = torch.tensor([[[-0.5], [-1.25], [-3.0], [-4.0]]])
        self.assertTrue(bool(torch.allclose(predicted, expected, atol=1e-6)))

    def test_history_before_the_first_sample_is_zero(self) -> None:
        # The lag buffer starts at silence, so the first prediction draws on
        # the leading tap alone. The zero padding is what makes the opening
        # samples well defined; without it the sequence would wrap and the
        # first prediction would depend on the end of the utterance.
        with torch.no_grad():
            predicted: torch.Tensor = self._prediction(self._samples, self._coefficients)
        self.assertAlmostEqual(float(predicted[0, 0, 0]), -0.5, places=6)


class LpcnetNetworkSurfaceTest(unittest.TestCase):
    # Verifies the conditioning encoder, the teacher-forced transformation, and the exposed properties.
    def setUp(self) -> None:
        # Builds the noise-free miniature network and one three-frame conditioning set.
        torch.manual_seed(1234)
        self._network: LpcnetNetwork = MiniatureNetworkRecipe(training_noise_standard_deviation=0.0).build()
        self._features: FrameFeatureBuilder = FrameFeatureBuilder(batch_size=1, frame_count=3)
        self._conditioning: torch.Tensor = self._features.build_conditioning()
        self._pitch_index: torch.Tensor = self._features.build_pitch_index()
        self._lpc_coefficients: torch.Tensor = self._features.build_lpc_coefficients()
        self._target_samples: torch.Tensor = self._features.build_target_samples()

    def test_frame_size_property_reports_the_conditioning_span(self) -> None:
        # One conditioning frame covers the configured number of waveform samples.
        self.assertEqual(self._network.frame_size, 2)

    def test_mulaw_property_exposes_the_shared_codec(self) -> None:
        # Training, loss, and synthesis paths share one companding codec instance.
        self.assertIsInstance(self._network.mulaw, LpcnetMuLaw)

    def test_conditioning_encoder_emits_one_vector_per_frame(self) -> None:
        # The frame-rate encoder maps features and pitch onto the conditioning width.
        with torch.no_grad():
            encoded: torch.Tensor = self._network.encode_conditioning(self._conditioning, self._pitch_index)
        self.assertEqual(tuple(encoded.shape), (1, 3, 8))

    def test_conditioning_vectors_are_bounded_by_the_output_activation(self) -> None:
        # The encoder ends in a hyperbolic tangent, so its outputs stay inside the unit interval.
        with torch.no_grad():
            encoded: torch.Tensor = self._network.encode_conditioning(self._conditioning, self._pitch_index)
        self.assertLess(float(encoded.abs().max()), 1.0)

    def test_expanded_tree_probabilities_are_normalized(self) -> None:
        # The network exposes the tree expansion the sampler and the loss both consume.
        node_outputs: torch.Tensor = torch.rand(1, 2, 256)
        with torch.no_grad():
            probabilities: torch.Tensor = self._network.expand_tree_probabilities(node_outputs)
        self.assertTrue(bool(torch.allclose(probabilities.sum(dim=-1), torch.ones(1, 2), atol=1e-5)))

    def test_forward_returns_prediction_and_node_outputs(self) -> None:
        # The teacher-forced pass returns the linear prediction beside the excitation nodes.
        with torch.no_grad():
            output: LpcnetNetworkOutput = self._network(
                self._target_samples,
                self._conditioning,
                self._pitch_index,
                self._lpc_coefficients
            )
        self.assertIsInstance(output, LpcnetNetworkOutput)
        self.assertEqual(tuple(output.prediction.shape), (1, 6, 1))
        self.assertEqual(tuple(output.node_outputs.shape), (1, 6, 256))

    def test_forward_node_outputs_are_probabilities(self) -> None:
        # Node activations feed the hierarchical expansion and must stay inside the unit interval.
        with torch.no_grad():
            output: LpcnetNetworkOutput = self._network(
                self._target_samples,
                self._conditioning,
                self._pitch_index,
                self._lpc_coefficients
            )
        self.assertGreater(float(output.node_outputs.min()), 0.0)
        self.assertLess(float(output.node_outputs.max()), 1.0)

    def test_forward_output_is_immutable(self) -> None:
        # The frozen bundle cannot be edited between the forward pass and the loss.
        with torch.no_grad():
            output: LpcnetNetworkOutput = self._network(
                self._target_samples,
                self._conditioning,
                self._pitch_index,
                self._lpc_coefficients
            )
        with self.assertRaises(ValidationError):
            output.prediction = torch.zeros(1)

    def test_batch_dimension_is_preserved(self) -> None:
        # Batched features produce one sample stream per batch element.
        batched: FrameFeatureBuilder = FrameFeatureBuilder(batch_size=2, frame_count=3)
        with torch.no_grad():
            output: LpcnetNetworkOutput = self._network(
                batched.build_target_samples(),
                batched.build_conditioning(),
                batched.build_pitch_index(),
                batched.build_lpc_coefficients()
            )
        self.assertEqual(output.node_outputs.shape[0], 2)


class LpcnetTrainingNoiseTest(unittest.TestCase):
    # Verifies that the teacher-forcing noise is injected in training mode
    # only. The three assertions bracket the mechanism completely: evaluation
    # mode must repeat exactly, training mode must not, and the analytic
    # prediction must be identical in both. That third assertion is the
    # precise one, since it establishes where the noise enters. Perturbing the
    # prediction as well would corrupt the excitation target the objective
    # derives from it, turning a robustness measure into label noise.
    def setUp(self) -> None:
        # Builds the miniature network at the reference noise level and one
        # conditioning set shared by all three assertions, so mode is the only
        # variable between them.
        torch.manual_seed(1234)
        self._network: LpcnetNetwork = MiniatureNetworkRecipe(training_noise_standard_deviation=0.3).build()
        self._features: FrameFeatureBuilder = FrameFeatureBuilder(batch_size=1, frame_count=3)
        self._conditioning: torch.Tensor = self._features.build_conditioning()
        self._pitch_index: torch.Tensor = self._features.build_pitch_index()
        self._lpc_coefficients: torch.Tensor = self._features.build_lpc_coefficients()
        self._target_samples: torch.Tensor = self._features.build_target_samples()

    def test_evaluation_mode_is_deterministic(self) -> None:
        # With noise disabled by evaluation mode the transformation repeats
        # exactly, without any seed being reset between the two calls. That is
        # a stronger claim than seeded reproducibility: it establishes the
        # evaluation path draws no randomness at all, which is what synthesis
        # relies on when it invokes the network's components directly.
        self._network.eval()
        with torch.no_grad():
            first: LpcnetNetworkOutput = self._network(
                self._target_samples,
                self._conditioning,
                self._pitch_index,
                self._lpc_coefficients
            )
            second: LpcnetNetworkOutput = self._network(
                self._target_samples,
                self._conditioning,
                self._pitch_index,
                self._lpc_coefficients
            )
        self.assertTrue(bool(torch.equal(first.node_outputs, second.node_outputs)))

    def test_training_mode_perturbs_the_teacher_forced_inputs(self) -> None:
        # Training injects the configured noise so the model learns robustness to its own errors.
        self._network.train()
        with torch.no_grad():
            first: LpcnetNetworkOutput = self._network(
                self._target_samples,
                self._conditioning,
                self._pitch_index,
                self._lpc_coefficients
            )
            second: LpcnetNetworkOutput = self._network(
                self._target_samples,
                self._conditioning,
                self._pitch_index,
                self._lpc_coefficients
            )
        self.assertFalse(bool(torch.equal(first.node_outputs, second.node_outputs)))

    def test_linear_prediction_is_unaffected_by_the_input_noise(self) -> None:
        # Noise enters the embedded inputs only; the analytic prediction stays
        # exact. Running under training mode alongside the previous assertion
        # is what makes the pair conclusive: the same two calls that produce
        # differing node outputs produce identical predictions, localizing the
        # perturbation to the embedded path and proving the supervision target
        # is untouched.
        self._network.train()
        with torch.no_grad():
            first: LpcnetNetworkOutput = self._network(
                self._target_samples,
                self._conditioning,
                self._pitch_index,
                self._lpc_coefficients
            )
            second: LpcnetNetworkOutput = self._network(
                self._target_samples,
                self._conditioning,
                self._pitch_index,
                self._lpc_coefficients
            )
        self.assertTrue(bool(torch.equal(first.prediction, second.prediction)))
