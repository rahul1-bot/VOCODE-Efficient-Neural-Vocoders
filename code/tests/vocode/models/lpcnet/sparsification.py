# This module:
# 1. Verifies the LPCNet sparsifier configuration record: the reference
#    schedule bounds, per-gate densities, block geometry, immutability, and
#    the rejection of unknown fields
# 2. Verifies the sparsification schedule itself: the inactive window before
#    the start step, the interval gating, the per-gate final densities, the
#    always-retained diagonal, and the untouched input weights
# 3. Verifies the recurrent weight clipper: the pairwise magnitude
#    constraint, its in-place application, and its neutrality on weights
#    that already satisfy the constraint
#
# Design decisions:
# - Assertions run on a small recurrent layer whose hidden size divides both
#   block dimensions, because the mask reshape requires that divisibility
# - Sparsity is asserted through surviving-weight counts and gate ordering
#   rather than pinned magnitudes, since the surviving blocks depend on the
#   random initialization
# - The clipper contract is asserted as the pairwise sum bound it enforces,
#   which is exact arithmetic rather than a tolerance band
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError
from torch import nn

from vocode.models.lpcnet.sparsification import LpcnetSparsifier, LpcnetSparsifierConfig, LpcnetWeightClipper


class RecurrentLayerBuilder:
    # Builds the small recurrent layer the sparsification schedule is applied
    # to. A real first layer is far wider, but the schedule's behavior does not
    # depend on width beyond the divisibility the block reshape requires, so a
    # small layer exercises every branch at a fraction of the cost.
    def __init__(self, hidden_size: int) -> None:
        # Binds the hidden size, which must divide both block dimensions of
        # the mask reshape.
        #
        # Args:
        #     hidden_size: Recurrent width. A value not divisible by both
        #         block dimensions would make the block reshape fail rather
        #         than merely prune differently.
        self._hidden_size: int = hidden_size

    def build(self) -> nn.GRU:
        # Returns the layer under a local seed so its initialization is fixed
        # across tests. Seeding inside the builder rather than relying on the
        # caller's seed is what makes the surviving blocks deterministic:
        # pruning ranks blocks by magnitude, so which ones survive is entirely
        # a function of the initialization.
        torch.manual_seed(5)
        return nn.GRU(8, self._hidden_size, batch_first=True)


class GateSliceReader:
    # Reads the reset, update, and state gate blocks out of a recurrent weight
    # matrix. The three gates are stacked into one parameter, so isolating
    # them is what allows the per-gate densities to be asserted separately
    # rather than only as an aggregate sparsity.
    def __init__(self, hidden_size: int) -> None:
        # Binds the hidden size that sets each gate's row span in the stacked
        # matrix.
        self._hidden_size: int = hidden_size

    def read_gates(self, recurrent_weight: torch.Tensor) -> dict[str, torch.Tensor]:
        # Returns all three gate blocks keyed by name, in the order the matrix
        # stacks them.
        #
        # Returns:
        #     Views into the supplied matrix rather than copies, so a caller
        #     must read them before any further in-place modification.
        return {
            "reset": self.read_reset_gate(recurrent_weight),
            "update": self.read_update_gate(recurrent_weight),
            "state": self.read_state_gate(recurrent_weight)
        }

    def read_reset_gate(self, recurrent_weight: torch.Tensor) -> torch.Tensor:
        # Returns the first hidden-size rows.
        return recurrent_weight[: self._hidden_size]

    def read_update_gate(self, recurrent_weight: torch.Tensor) -> torch.Tensor:
        # Returns the second hidden-size rows.
        return recurrent_weight[self._hidden_size: 2 * self._hidden_size]

    def read_state_gate(self, recurrent_weight: torch.Tensor) -> torch.Tensor:
        # Returns the third hidden-size rows.
        return recurrent_weight[2 * self._hidden_size: 3 * self._hidden_size]


class LpcnetSparsifierConfigurationTest(unittest.TestCase):
    # Verifies the reference schedule settings and the validation behavior of the record.
    def setUp(self) -> None:
        # Reads the default sparsifier record, which is the reference schedule.
        self._configuration: LpcnetSparsifierConfig = LpcnetSparsifierConfig()

    def test_schedule_bounds_follow_the_reference_recipe(self) -> None:
        # Sparsification ramps between step 2000 and step 20000 and reapplies every 400 steps.
        self.assertEqual(self._configuration.schedule_start_step, 2000)
        self.assertEqual(self._configuration.schedule_end_step, 20000)
        self.assertEqual(self._configuration.application_interval, 400)

    def test_state_gate_keeps_a_higher_final_density(self) -> None:
        # The reference retains four times more state-gate weight than gate weight.
        self.assertEqual(self._configuration.reset_gate_density, 0.05)
        self.assertEqual(self._configuration.update_gate_density, 0.05)
        self.assertEqual(self._configuration.state_gate_density, 0.2)

    def test_block_geometry_matches_the_deployment_layout(self) -> None:
        # Pruning operates on four-by-eight blocks, the reference deployment geometry.
        self.assertEqual(self._configuration.block_rows, 4)
        self.assertEqual(self._configuration.block_columns, 8)

    def test_configuration_is_immutable(self) -> None:
        # Frozen settings cannot drift during a training run.
        with self.assertRaises(ValidationError):
            self._configuration.state_gate_density = 0.9

    def test_unknown_field_is_rejected(self) -> None:
        # Extra fields are forbidden so silent typos cannot enter an experiment record.
        with self.assertRaises(ValidationError):
            LpcnetSparsifierConfig(unknown_setting=1)

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The sparsifier publishes the schedule it enforces.
        sparsifier: LpcnetSparsifier = LpcnetSparsifier(self._configuration)
        self.assertIs(sparsifier.configuration, self._configuration)


class LpcnetSparsificationScheduleTest(unittest.TestCase):
    # Verifies the schedule gating and the per-gate block sparsity of the
    # recurrent weights. The gating assertions bracket the schedule from both
    # sides: three steps that must leave the weights untouched and two that
    # must prune. Because pruning is destructive, each assertion compares
    # against a snapshot taken before any application rather than against a
    # recomputed expectation.
    def setUp(self) -> None:
        # Builds the recurrent layer and records its unpruned weight for every
        # comparison below. The snapshot is cloned rather than aliased,
        # because the sparsifier writes through the live parameter and an
        # alias would silently track those edits.
        torch.manual_seed(1234)
        self._configuration: LpcnetSparsifierConfig = LpcnetSparsifierConfig()
        self._sparsifier: LpcnetSparsifier = LpcnetSparsifier(self._configuration)
        self._builder: RecurrentLayerBuilder = RecurrentLayerBuilder(hidden_size=32)
        self._recurrent_layer: nn.GRU = self._builder.build()
        self._initial_weight: torch.Tensor = self._recurrent_layer.weight_hh_l0.detach().clone()
        self._reader: GateSliceReader = GateSliceReader(hidden_size=32)

    def test_no_mask_is_applied_before_the_start_step(self) -> None:
        # Sparsification stays inactive while the model is still establishing a signal.
        self._sparsifier.apply(self._recurrent_layer, 1999)
        self.assertTrue(bool(torch.equal(self._recurrent_layer.weight_hh_l0, self._initial_weight)))

    def test_no_mask_is_applied_at_the_start_step(self) -> None:
        # The schedule opens strictly after the start step. The boundary is
        # asserted explicitly because the interval test measures elapsed steps
        # from the start: at the start step itself that elapsed count is zero
        # and divides evenly, so without the strict comparison the very first
        # eligible step would prune.
        self._sparsifier.apply(self._recurrent_layer, 2000)
        self.assertTrue(bool(torch.equal(self._recurrent_layer.weight_hh_l0, self._initial_weight)))

    def test_no_mask_is_applied_between_intervals(self) -> None:
        # Between application intervals the optimizer is left alone.
        self._sparsifier.apply(self._recurrent_layer, 2500)
        self.assertTrue(bool(torch.equal(self._recurrent_layer.weight_hh_l0, self._initial_weight)))

    def test_mask_is_applied_on_an_interval_step(self) -> None:
        # On an interval boundary the annealed density prunes the weakest
        # blocks. The step chosen is exactly one interval past the start, the
        # earliest step at which any pruning may occur. Only the presence of
        # zeros is asserted, not how many: this early in the cubic ramp the
        # density is still close to one, so the count is small and would be a
        # brittle expectation.
        self._sparsifier.apply(self._recurrent_layer, 2400)
        self.assertFalse(bool(torch.equal(self._recurrent_layer.weight_hh_l0, self._initial_weight)))
        self.assertGreater(int((self._recurrent_layer.weight_hh_l0 == 0.0).sum()), 0)

    def test_final_step_prunes_the_majority_of_the_recurrent_weight(self) -> None:
        # At the end of the schedule the reference densities leave a sparse
        # matrix behind. The bound is deliberately loose: the target densities
        # imply a far sparser result than half, but the retained diagonal and
        # the block granularity both raise the realized density above the
        # nominal one, so asserting the exact figure would encode arithmetic
        # that the block geometry, not the schedule, determines.
        self._sparsifier.apply(self._recurrent_layer, 20000)
        surviving: int = int((self._recurrent_layer.weight_hh_l0 != 0.0).sum())
        self.assertLess(surviving, self._initial_weight.numel() // 2)

    def test_state_gate_retains_more_weight_than_the_reset_gate(self) -> None:
        # The per-gate densities are honoured separately, not averaged into one schedule.
        self._sparsifier.apply(self._recurrent_layer, 20000)
        recurrent_weight: torch.Tensor = self._recurrent_layer.weight_hh_l0.detach()
        reset_survivors: int = int((self._reader.read_reset_gate(recurrent_weight) != 0.0).sum())
        state_survivors: int = int((self._reader.read_state_gate(recurrent_weight) != 0.0).sum())
        self.assertGreater(
            state_survivors,
            reset_survivors,
            msg="The state gate density of 0.2 must survive more weight than the 0.05 gate densities"
        )

    def test_diagonal_blocks_are_always_retained(self) -> None:
        # The reference keeps every gate's diagonal so each hidden unit's own
        # recurrence survives pruning. Exact equality against the initial
        # diagonal is asserted rather than mere non-zeroness, which proves the
        # diagonal passes through untouched and is not merely rescaled by the
        # mask arithmetic. The check runs for all three gates, since the
        # exemption is applied per gate slice.
        self._sparsifier.apply(self._recurrent_layer, 20000)
        recurrent_weight: torch.Tensor = self._recurrent_layer.weight_hh_l0.detach()
        pruned_gates: dict[str, torch.Tensor] = self._reader.read_gates(recurrent_weight)
        initial_gates: dict[str, torch.Tensor] = self._reader.read_gates(self._initial_weight)
        gate_name: str
        pruned_gate: torch.Tensor
        for gate_name, pruned_gate in pruned_gates.items():
            self.assertTrue(
                bool(torch.equal(torch.diag(pruned_gate), torch.diag(initial_gates[gate_name]))),
                msg=f"The {gate_name} gate diagonal must survive the block mask"
            )

    def test_input_weights_are_never_masked(self) -> None:
        # Sparsification targets the recurrent kernel only, per the deployment geometry.
        initial_input_weight: torch.Tensor = self._recurrent_layer.weight_ih_l0.detach().clone()
        self._sparsifier.apply(self._recurrent_layer, 20000)
        self.assertTrue(bool(torch.equal(self._recurrent_layer.weight_ih_l0, initial_input_weight)))

    def test_mask_is_applied_in_place(self) -> None:
        # Masks mutate the live parameter rather than rebinding it. Comparing
        # storage addresses is what makes this precise: a replacement tensor
        # holding the same values would satisfy any value-based check while
        # detaching the optimizer's accumulated state, which is keyed to the
        # original parameter object.
        weight_identity: int = self._recurrent_layer.weight_hh_l0.data_ptr()
        self._sparsifier.apply(self._recurrent_layer, 20000)
        self.assertEqual(self._recurrent_layer.weight_hh_l0.data_ptr(), weight_identity)


class LpcnetWeightClipperTest(unittest.TestCase):
    # Verifies the pairwise magnitude constraint that keeps recurrent weights
    # representable by the quantized inference path. The constraint is asserted
    # as the exact bound it enforces on adjacent column pairs, together with
    # the two properties that make it a rescaling rather than a truncation:
    # signs survive, and already-compliant weights are untouched.
    def setUp(self) -> None:
        # Builds the clipper and a weight scaled well past the constraint so
        # clipping is exercised. The threefold scaling guarantees essentially
        # every pair violates the bound, so the tests measure the constraint
        # itself rather than accidentally sampling compliant weights.
        torch.manual_seed(1234)
        self._clipper: LpcnetWeightClipper = LpcnetWeightClipper()
        self._weight: torch.Tensor = torch.randn(4, 8) * 3.0

    def test_default_clip_value_matches_the_reference(self) -> None:
        # The reference constant keeps adjacent pairs inside the quantized
        # inference range. The tolerance covers floating-point error in the
        # rescaling division only; the constraint itself is exact, so the
        # margin is set at the scale of representation error rather than as a
        # band chosen to accommodate approximate behavior.
        self._clipper.apply(self._weight)
        pair_sums: torch.Tensor = self._weight[:, 0::2].abs() + self._weight[:, 1::2].abs()
        self.assertLessEqual(float(pair_sums.max()), 0.992 + 1e-6)

    def test_configured_clip_value_bounds_the_pair_sums(self) -> None:
        # An explicit clip value replaces the reference constant.
        tight_clipper: LpcnetWeightClipper = LpcnetWeightClipper(clip_value=0.5)
        tight_clipper.apply(self._weight)
        pair_sums: torch.Tensor = self._weight[:, 0::2].abs() + self._weight[:, 1::2].abs()
        self.assertLessEqual(float(pair_sums.max()), 0.5 + 1e-6)

    def test_compliant_weights_are_left_unchanged(self) -> None:
        # Pairs already inside the constraint are not rescaled. This is the
        # assertion that distinguishes a bound from a normalization: were the
        # division applied unconditionally, these pairs would be scaled up to
        # meet the limit exactly, inflating every small weight in the matrix.
        compliant_weight: torch.Tensor = torch.full((2, 4), 0.1)
        expected_weight: torch.Tensor = compliant_weight.clone()
        self._clipper.apply(compliant_weight)
        self.assertTrue(bool(torch.allclose(compliant_weight, expected_weight, atol=1e-7)))

    def test_signs_are_preserved(self) -> None:
        # The constraint rescales magnitudes and never flips a weight's
        # direction, because it divides by a strictly positive quantity. Sign
        # preservation is what makes the constraint safe to apply after every
        # optimizer step: it shrinks the update the optimizer produced without
        # ever reversing it, so it cannot fight the descent direction.
        original_signs: torch.Tensor = torch.sign(self._weight).clone()
        self._clipper.apply(self._weight)
        self.assertTrue(bool(torch.equal(torch.sign(self._weight), original_signs)))

    def test_clip_is_applied_in_place(self) -> None:
        # The constraint mutates the live parameter rather than returning a copy.
        weight_identity: int = self._weight.data_ptr()
        self._clipper.apply(self._weight)
        self.assertEqual(self._weight.data_ptr(), weight_identity)
