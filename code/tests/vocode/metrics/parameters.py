# This module:
# 1. Verifies the residual parameter count: the total of registered
#    floating-point parameters over handcrafted networks whose counts are known
#    exactly by construction, including nested and parameter-free cases
# 2. Verifies the deployable count, which adds the packed quantized weights
#    that live outside torch parameter registration to the residual count
# 3. Verifies the analytic packed-GRU geometry across the layer count,
#    direction count, and bias switches that drive it
#
# Design decisions:
# - Counts are exact integers derived from the declared tensor geometry, so
#   every assertion pins a computed value rather than a tolerance
# - Real torch.ao quantized modules cannot be constructed in this environment
#   because no quantization engine is registered (quantized::linear_prepack
#   reports NoQEngine), so the packed strata are exercised through stand-in
#   subclasses of the real quantized classes that skip the prepacking
#   constructor and expose only the geometry and weight-bias accessor the
#   counter reads; the counting arithmetic under test is the production code
# - Buffers are asserted to stay outside both counts, because a buffer is
#   state rather than a weight and would inflate compression ratios
#
# Author: Rahul Sawhney

import unittest
from typing import override

import torch
import torch.ao.nn.quantized as quantized_nn
import torch.ao.nn.quantized.dynamic as quantized_dynamic_nn
from torch import Tensor, nn

from vocode.metrics.parameters import ParameterCount


class PackedLinearStandIn(quantized_nn.Linear):
    # Quantized Linear stand-in that bypasses the prepacking constructor and
    # serves fixed tensors through the weight-bias accessor the counter reads.
    def __init__(self, packed_weight: Tensor, packed_bias: Tensor | None) -> None:
        # Initializes through nn.Module directly, skipping the quantized
        # constructor that would demand a registered quantization engine.
        nn.Module.__init__(self)
        self._packed_weight: Tensor = packed_weight
        self._packed_bias: Tensor | None = packed_bias

    @override
    def _weight_bias(self) -> tuple[Tensor, Tensor | None]:
        # Serves the fixed tensors through the accessor the counter reads.
        return (self._packed_weight, self._packed_bias)


class PackedGruStandIn(quantized_dynamic_nn.GRU):
    # Dynamic quantized GRU stand-in that bypasses the prepacking constructor
    # and declares only the layer geometry the analytic count consumes.
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        bias: bool
    ) -> None:
        # Initializes through nn.Module directly and declares only the geometry
        # fields the analytic packed count reads.
        nn.Module.__init__(self)
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.bidirectional: bool = bidirectional
        self.bias: bool = bias


class BufferedLinearNetwork(nn.Module):
    # Network pairing one registered linear layer with a registered buffer, so
    # buffer exclusion from the parameter counts is observable.
    def __init__(self, input_features: int, output_features: int, buffer_length: int) -> None:
        # Registers one weighted layer beside one buffer of the requested length.
        super().__init__()
        self.projection: nn.Linear = nn.Linear(input_features, output_features)
        self.register_buffer("running_scale", torch.zeros(buffer_length))


class PackedHybridNetwork(nn.Module):
    # Network mixing a registered linear layer with packed quantized Linear and
    # GRU stand-ins, so residual and packed strata are counted together.
    def __init__(self) -> None:
        # Registers one unquantized layer alongside both packed strata, so a
        # single network exercises the residual and deployable counts together.
        super().__init__()
        self.projection: nn.Linear = nn.Linear(4, 3)
        self.packed_projection: PackedLinearStandIn = PackedLinearStandIn(
            torch.zeros(6, 5),
            torch.zeros(6)
        )
        self.packed_recurrence: PackedGruStandIn = PackedGruStandIn(4, 5, 2, False, True)


class ParameterResidualCountTest(unittest.TestCase):
    # Verifies the residual count over registered floating-point parameters.
    def setUp(self) -> None:
        # Prepares the counter shared by the residual checks.
        self._counter: ParameterCount = ParameterCount()

    def test_single_linear_layer_counts_its_weight_and_bias(self) -> None:
        # A linear layer contributes in_features * out_features weights plus one bias per output.
        network: nn.Linear = nn.Linear(4, 3)
        self.assertEqual(self._counter.residual_count(network), 4 * 3 + 3)

    def test_biasless_linear_layer_counts_only_its_weight(self) -> None:
        # Dropping the bias removes exactly out_features parameters.
        network: nn.Linear = nn.Linear(4, 3, bias=False)
        self.assertEqual(self._counter.residual_count(network), 4 * 3)

    def test_nested_submodules_are_traversed(self) -> None:
        # The count reaches every registered parameter in the module tree.
        network: nn.Sequential = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        self.assertEqual(self._counter.residual_count(network), (4 * 3 + 3) + (3 * 2 + 2))

    def test_parameter_free_network_counts_zero(self) -> None:
        # An activation-only network holds no weights at all.
        network: nn.ReLU = nn.ReLU()
        self.assertEqual(self._counter.residual_count(network), 0)

    def test_registered_buffers_are_excluded(self) -> None:
        # Buffers are state, not weights, and must not enter the count.
        network: BufferedLinearNetwork = BufferedLinearNetwork(4, 3, 64)
        self.assertEqual(self._counter.residual_count(network), 4 * 3 + 3)

    def test_convolutional_geometry_is_counted_exactly(self) -> None:
        # A convolution contributes out * in * kernel weights plus one bias per output channel.
        network: nn.Conv1d = nn.Conv1d(8, 4, kernel_size=3)
        self.assertEqual(self._counter.residual_count(network), 4 * 8 * 3 + 4)

    def test_frozen_parameters_still_count(self) -> None:
        # Deployability depends on stored weights, not on gradient requirements.
        network: nn.Linear = nn.Linear(4, 3)
        network.weight.requires_grad_(False)
        self.assertEqual(self._counter.residual_count(network), 4 * 3 + 3)


class DeployableParameterCountTest(unittest.TestCase):
    # Verifies the deployable count: the residual count plus packed quantized
    # weights that torch parameter registration does not expose.
    def setUp(self) -> None:
        # Prepares the counter shared by the deployable checks.
        self._counter: ParameterCount = ParameterCount()

    def test_deployable_count_equals_residual_count_without_packed_weights(self) -> None:
        # An unquantized network has no packed stratum to add.
        network: nn.Sequential = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        self.assertEqual(self._counter(network), self._counter.residual_count(network))

    def test_packed_linear_weight_and_bias_join_the_count(self) -> None:
        # The packed tensors are read through the module's weight-bias accessor.
        network: PackedLinearStandIn = PackedLinearStandIn(torch.zeros(6, 5), torch.zeros(6))
        self.assertEqual(self._counter(network), 6 * 5 + 6)

    def test_packed_linear_without_bias_counts_only_its_weight(self) -> None:
        # An absent packed bias contributes nothing rather than failing.
        network: PackedLinearStandIn = PackedLinearStandIn(torch.zeros(6, 5), None)
        self.assertEqual(self._counter(network), 6 * 5)

    def test_packed_weights_are_invisible_to_the_residual_count(self) -> None:
        # The residual count reports the unquantized remainder alone.
        network: PackedLinearStandIn = PackedLinearStandIn(torch.zeros(6, 5), torch.zeros(6))
        self.assertEqual(self._counter.residual_count(network), 0)

    def test_residual_and_packed_strata_sum_into_the_deployable_count(self) -> None:
        # A hybrid network reports registered weights plus both packed strata.
        network: PackedHybridNetwork = PackedHybridNetwork()
        residual_total: int = 4 * 3 + 3
        packed_linear_total: int = 6 * 5 + 6
        gate_dimension: int = 3 * 5
        first_gru_layer: int = gate_dimension * 4 + gate_dimension * 5 + 2 * gate_dimension
        later_gru_layer: int = gate_dimension * 5 + gate_dimension * 5 + 2 * gate_dimension
        self.assertEqual(self._counter.residual_count(network), residual_total)
        self.assertEqual(
            self._counter(network),
            residual_total + packed_linear_total + first_gru_layer + later_gru_layer
        )


class PackedRecurrentGeometryTest(unittest.TestCase):
    # Verifies the analytic packed-GRU count across layer count, direction
    # count, and bias, where each gate stratum is three times the hidden size.
    def setUp(self) -> None:
        # Prepares the counter shared by the packed-geometry checks.
        self._counter: ParameterCount = ParameterCount()

    def test_single_layer_counts_input_recurrent_and_bias_strata(self) -> None:
        # One unidirectional layer holds gate * input, gate * hidden, and two gate biases.
        network: PackedGruStandIn = PackedGruStandIn(4, 5, 1, False, True)
        gate_dimension: int = 3 * 5
        self.assertEqual(
            self._counter(network),
            gate_dimension * 4 + gate_dimension * 5 + 2 * gate_dimension
        )

    def test_biasless_layer_omits_both_gate_bias_strata(self) -> None:
        # Removing the bias removes exactly two gate-width vectors per direction.
        network: PackedGruStandIn = PackedGruStandIn(4, 5, 1, False, False)
        gate_dimension: int = 3 * 5
        self.assertEqual(self._counter(network), gate_dimension * 4 + gate_dimension * 5)

    def test_bidirectional_layer_doubles_the_single_layer_geometry(self) -> None:
        # Both directions carry a full weight set over the same input width.
        unidirectional: PackedGruStandIn = PackedGruStandIn(4, 5, 1, False, True)
        bidirectional: PackedGruStandIn = PackedGruStandIn(4, 5, 1, True, True)
        self.assertEqual(self._counter(bidirectional), 2 * self._counter(unidirectional))

    def test_later_layers_take_the_stacked_hidden_width_as_input(self) -> None:
        # Layer zero reads the input size while later layers read the hidden output.
        network: PackedGruStandIn = PackedGruStandIn(4, 5, 2, False, True)
        gate_dimension: int = 3 * 5
        first_layer: int = gate_dimension * 4 + gate_dimension * 5 + 2 * gate_dimension
        later_layer: int = gate_dimension * 5 + gate_dimension * 5 + 2 * gate_dimension
        self.assertEqual(self._counter(network), first_layer + later_layer)

    def test_bidirectional_later_layers_read_both_direction_outputs(self) -> None:
        # A bidirectional stack feeds later layers hidden_size times two.
        network: PackedGruStandIn = PackedGruStandIn(4, 5, 2, True, True)
        gate_dimension: int = 3 * 5
        first_layer: int = 2 * (gate_dimension * 4 + gate_dimension * 5 + 2 * gate_dimension)
        later_layer: int = 2 * (
            gate_dimension * (5 * 2) + gate_dimension * 5 + 2 * gate_dimension
        )
        self.assertEqual(self._counter(network), first_layer + later_layer)

    def test_packed_recurrence_contributes_no_registered_parameters(self) -> None:
        # The packed cell parameters are outside torch parameter registration.
        network: PackedGruStandIn = PackedGruStandIn(4, 5, 2, True, True)
        self.assertEqual(self._counter.residual_count(network), 0)


if __name__ == "__main__":
    unittest.main()
