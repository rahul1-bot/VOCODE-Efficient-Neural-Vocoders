# This module:
# 1. Verifies the LPCNet loss configuration: the probability floor, the frozen
#    record, and the rejection of unknown or non-positive settings
# 2. Verifies the excitation cross entropy against its closed form: a uniform
#    node table costs the entropy of the 256-level distribution, a table that
#    is certain about the target path costs nothing, and a constant branch
#    confidence costs minus eight times its logarithm
# 3. Verifies the probability floor, which replaces an impossible branch with
#    the configured epsilon instead of producing an infinite loss
# 4. Verifies the gradient contract: the node probabilities carry the training
#    signal while the sample path through the mu-law index does not
#
# Design decisions:
# - Node tables are built by walking the same eight-level binary path the loss
#   walks, so each anchor states which branch probability is being asserted
#   rather than hiding it behind a random table
# - The zero residual is used as the reference target because it lands on the
#   mu-law midpoint, an index whose branch path is stated explicitly by a test
#   rather than assumed
# - The closed forms are exact natural logarithms, so these are pinned to full
#   float32 precision instead of being bounded
#
# Author: Rahul Sawhney

import math
import unittest

import torch
from pydantic import ValidationError

from vocode.losses.lpcnet import LpcnetLoss, LpcnetLossConfig
from vocode.transforms.lpc import LpcnetMuLaw


class ExcitationTreeBuilder:
    # Builds teacher-forced sample sequences and node probability tables over
    # the eight-level mu-law tree the loss traverses.
    def __init__(self, batch_size: int, time_step_count: int) -> None:
        # Binds the constructor inputs into this component's state.
        self._batch_size: int = batch_size
        self._time_step_count: int = time_step_count
        self._level_count: int = 8
        self._node_count: int = 256

    def samples(self, value: float) -> torch.Tensor:
        # Builds a constant int16-scaled sample sequence.
        return torch.full((self._batch_size, self._time_step_count, 1), value)

    def uniform_nodes(self, probability: float) -> torch.Tensor:
        # Builds a node table holding one probability at every tree node.
        return torch.full((self._batch_size, self._time_step_count, self._node_count), probability)

    def path_nodes(self, excitation_index: int, branch_confidence: float) -> torch.Tensor:
        # Builds a table giving the target's branch the requested confidence at every level.
        table: torch.Tensor = torch.zeros(self._batch_size, self._time_step_count, self._node_count)
        level: int
        for level in range(self._level_count):
            prefix: int = excitation_index >> (self._level_count - level)
            node_index: int = (1 << level) + prefix
            branch_bit: int = (excitation_index >> (self._level_count - 1 - level)) & 1
            table[..., node_index] = branch_confidence if branch_bit == 1 else 1.0 - branch_confidence
        return table

    def narrow_nodes(self, node_count: int) -> torch.Tensor:
        # Builds a table too narrow to address the deepest tree level.
        return torch.full((self._batch_size, self._time_step_count, node_count), 0.5)

    @property
    def level_count(self) -> int:
        # Returns the depth of the binary excitation tree.
        return self._level_count


class LpcnetLossConfigurationTest(unittest.TestCase):
    # Verifies the frozen settings record behind the excitation cross entropy.
    def setUp(self) -> None:
        # Constructs the record from its defaults with no arguments, so the
        # cases below assert the probability floor the shipped configuration
        # actually applies rather than a value restated at the call site.
        self._configuration: LpcnetLossConfig = LpcnetLossConfig()

    def test_default_probability_floor_matches_the_reference_recipe(self) -> None:
        # The floor keeps an impossible branch from producing an infinite loss.
        self.assertEqual(self._configuration.probability_epsilon, 1e-8)

    def test_configuration_rejects_mutation_after_construction(self) -> None:
        # The record is frozen so a run cannot drift from its logged settings.
        with self.assertRaises(ValidationError):
            self._configuration.probability_epsilon: float = 1e-4

    def test_configuration_rejects_an_unknown_setting(self) -> None:
        # An unknown field is a typo, never a silently ignored setting.
        with self.assertRaises(ValidationError):
            LpcnetLossConfig(level_count=8)

    def test_configuration_rejects_a_non_positive_floor(self) -> None:
        # A zero floor would reintroduce the infinite-loss failure mode.
        with self.assertRaises(ValidationError):
            LpcnetLossConfig(probability_epsilon=0.0)

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The criterion exposes the exact record it was constructed with.
        configuration: LpcnetLossConfig = LpcnetLossConfig(probability_epsilon=1e-6)
        loss: LpcnetLoss = LpcnetLoss(configuration)
        self.assertIs(loss.configuration, configuration)


class LpcnetExcitationCrossEntropyTest(unittest.TestCase):
    # Verifies the tree-structured cross entropy against its closed forms.
    def setUp(self) -> None:
        # Samples and predictions are both held at zero so the residual is
        # zero, which mu-law companding maps to index 128, the midpoint of
        # the 0-255 domain. Fixing the target index this way is what makes a
        # closed form available: the eight branch decisions along a known
        # path can be reasoned about exactly, whereas an arbitrary residual
        # would leave the expected likelihood to be recomputed rather than
        # derived.
        self._builder: ExcitationTreeBuilder = ExcitationTreeBuilder(batch_size=2, time_step_count=6)
        self._loss: LpcnetLoss = LpcnetLoss(LpcnetLossConfig())
        self._samples: torch.Tensor = self._builder.samples(0.0)
        self._midpoint_index: int = 128

    def test_zero_residual_lands_on_the_mu_law_midpoint(self) -> None:
        # The target index used by every anchor below is stated here explicitly.
        mulaw: LpcnetMuLaw = LpcnetMuLaw()
        companded: torch.Tensor = mulaw.linear_to_mulaw(torch.zeros(1))
        self.assertAlmostEqual(float(companded.item()), float(self._midpoint_index), places=5)

    def test_uniform_node_table_costs_the_full_distribution_entropy(self) -> None:
        # Eight coin flips over 256 levels cost exactly eight natural logarithms of two.
        value: torch.Tensor = self._loss.compute_loss(
            target_samples=self._samples,
            prediction=self._samples,
            node_outputs=self._builder.uniform_nodes(0.5)
        )
        expected: float = float(self._builder.level_count) * math.log(2.0)
        self.assertAlmostEqual(float(value.item()), expected, places=5)

    def test_certain_node_table_costs_nothing(self) -> None:
        # A table certain about every branch of the target path is the optimum.
        value: torch.Tensor = self._loss.compute_loss(
            target_samples=self._samples,
            prediction=self._samples,
            node_outputs=self._builder.path_nodes(self._midpoint_index, branch_confidence=1.0)
        )
        self.assertAlmostEqual(
            float(value.item()),
            0.0,
            places=6,
            msg="A perfectly confident excitation distribution must cost nothing"
        )

    def test_constant_branch_confidence_costs_minus_eight_logarithms(self) -> None:
        # Each of the eight levels contributes the logarithm of the branch probability.
        value: torch.Tensor = self._loss.compute_loss(
            target_samples=self._samples,
            prediction=self._samples,
            node_outputs=self._builder.path_nodes(self._midpoint_index, branch_confidence=0.9)
        )
        expected: float = -float(self._builder.level_count) * math.log(0.9)
        self.assertAlmostEqual(float(value.item()), expected, places=5)

    def test_loss_decreases_as_the_target_branch_gains_probability(self) -> None:
        # The criterion is strictly decreasing in the confidence on the true path.
        low: torch.Tensor = self._loss.compute_loss(
            target_samples=self._samples,
            prediction=self._samples,
            node_outputs=self._builder.path_nodes(self._midpoint_index, branch_confidence=0.3)
        )
        high: torch.Tensor = self._loss.compute_loss(
            target_samples=self._samples,
            prediction=self._samples,
            node_outputs=self._builder.path_nodes(self._midpoint_index, branch_confidence=0.8)
        )
        self.assertGreater(float(low.item()), float(high.item()))

    def test_loss_returns_a_finite_scalar(self) -> None:
        # The criterion reduces the whole teacher-forced batch to one number.
        value: torch.Tensor = self._loss.compute_loss(
            target_samples=self._samples,
            prediction=self._samples,
            node_outputs=self._builder.uniform_nodes(0.25)
        )
        self.assertEqual(value.shape, torch.Size([]), msg="The criterion must reduce to a scalar")
        self.assertTrue(torch.isfinite(value).item())
        self.assertGreater(float(value.item()), 0.0)

    def test_loss_is_a_mean_and_ignores_the_sequence_length(self) -> None:
        # Averaging over samples keeps the value comparable across chunk sizes.
        long_builder: ExcitationTreeBuilder = ExcitationTreeBuilder(batch_size=2, time_step_count=24)
        short_value: torch.Tensor = self._loss.compute_loss(
            target_samples=self._samples,
            prediction=self._samples,
            node_outputs=self._builder.uniform_nodes(0.5)
        )
        long_value: torch.Tensor = self._loss.compute_loss(
            target_samples=long_builder.samples(0.0),
            prediction=long_builder.samples(0.0),
            node_outputs=long_builder.uniform_nodes(0.5)
        )
        self.assertAlmostEqual(float(short_value.item()), float(long_value.item()), places=5)

    def test_node_table_narrower_than_the_tree_is_rejected(self) -> None:
        # The deepest level addresses node 255, so a shorter table cannot be indexed.
        with self.assertRaises(RuntimeError):
            self._loss.compute_loss(
                target_samples=self._samples,
                prediction=self._samples,
                node_outputs=self._builder.narrow_nodes(node_count=128)
            )


class LpcnetProbabilityFloorTest(unittest.TestCase):
    # Verifies that an impossible branch is floored rather than diverging.
    def setUp(self) -> None:
        # Deliberately builds no loss instance, because these cases construct
        # their own with a chosen floor to show the bound moving with it. The
        # smallest fixture that still exercises the eight-level walk is used,
        # since the floor's effect is per branch and does not depend on how
        # many positions are averaged.
        self._builder: ExcitationTreeBuilder = ExcitationTreeBuilder(batch_size=1, time_step_count=4)
        self._samples: torch.Tensor = self._builder.samples(0.0)
        self._midpoint_index: int = 128

    def test_impossible_path_costs_the_floored_logarithm_at_every_level(self) -> None:
        # A zero-probability branch is replaced by the configured epsilon.
        loss: LpcnetLoss = LpcnetLoss(LpcnetLossConfig())
        value: torch.Tensor = loss.compute_loss(
            target_samples=self._samples,
            prediction=self._samples,
            node_outputs=self._builder.path_nodes(self._midpoint_index, branch_confidence=0.0)
        )
        expected: float = -float(self._builder.level_count) * math.log(1e-8)
        self.assertTrue(torch.isfinite(value).item(), msg="The floor must prevent an infinite loss")
        self.assertAlmostEqual(float(value.item()), expected, places=3)

    def test_a_larger_floor_lowers_the_worst_case_cost(self) -> None:
        # The ceiling on a single level is set by the configured epsilon alone.
        coarse_loss: LpcnetLoss = LpcnetLoss(LpcnetLossConfig(probability_epsilon=1e-4))
        value: torch.Tensor = coarse_loss.compute_loss(
            target_samples=self._samples,
            prediction=self._samples,
            node_outputs=self._builder.path_nodes(self._midpoint_index, branch_confidence=0.0)
        )
        expected: float = -float(self._builder.level_count) * math.log(1e-4)
        self.assertAlmostEqual(float(value.item()), expected, places=3)


class LpcnetGradientPathTest(unittest.TestCase):
    # Verifies which inputs the excitation cross entropy can train.
    def setUp(self) -> None:
        # Records no midpoint index, because these cases assert which inputs
        # receive gradient rather than what the loss evaluates to, and
        # gradient reachability does not depend on which target index the
        # residual selects.
        self._builder: ExcitationTreeBuilder = ExcitationTreeBuilder(batch_size=2, time_step_count=4)
        self._loss: LpcnetLoss = LpcnetLoss(LpcnetLossConfig())
        self._samples: torch.Tensor = self._builder.samples(0.0)

    def test_backward_populates_gradient_on_the_node_probabilities(self) -> None:
        # The node table is the learned distribution and carries the signal.
        node_outputs: torch.Tensor = self._builder.uniform_nodes(0.5).requires_grad_(True)
        value: torch.Tensor = self._loss.compute_loss(
            target_samples=self._samples,
            prediction=self._samples,
            node_outputs=node_outputs
        )
        value.backward()
        self.assertIsNotNone(node_outputs.grad, msg="The node probabilities must receive gradient")
        self.assertEqual(node_outputs.grad.shape, node_outputs.shape)
        self.assertTrue(torch.isfinite(node_outputs.grad).all().item())

    def test_backward_leaves_the_linear_prediction_without_gradient(self) -> None:
        # The residual only selects a tree path, which is a non-differentiable lookup.
        prediction: torch.Tensor = self._builder.samples(0.0).requires_grad_(True)
        node_outputs: torch.Tensor = self._builder.uniform_nodes(0.5).requires_grad_(True)
        value: torch.Tensor = self._loss.compute_loss(
            target_samples=self._samples,
            prediction=prediction,
            node_outputs=node_outputs
        )
        value.backward()
        self.assertIsNone(
            prediction.grad,
            msg="The mu-law index path is a rounded lookup and must not pretend to be trainable"
        )
        self.assertIsNotNone(node_outputs.grad)
