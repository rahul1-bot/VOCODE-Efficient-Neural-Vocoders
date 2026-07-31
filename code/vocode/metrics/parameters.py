# This module:
# 1. Counts model parameters at the deployable definition: registered
#    floating-point parameters plus packed quantized weights that live
#    outside torch parameter registration
# 2. Exposes the residual count (registered floating parameters alone)
#    separately, so quantized networks report both their logical weight
#    count and their unquantized remainder
#
# Design decisions:
# - Packed torch.ao quantized Linear weights are counted through the
#   module's weight-bias accessor, the only faithful view of the packed
#   tensor
# - Packed dynamic quantized GRU weights are counted analytically from the
#   layer geometry, because the packed cell parameters expose no tensor
#   traversal
# - Without this accounting, a quantized network would appear to have lost
#   its weights rather than repacked them, corrupting compression ratios
#
# Author: Rahul Sawhney

import torch.ao.nn.quantized as quantized_nn
import torch.ao.nn.quantized.dynamic as quantized_dynamic_nn
from torch import Tensor, nn

__all__: list[str] = ["ParameterCount"]


class ParameterCount:
    # Deployable parameter counting over a torch.nn.Module network,
    # quantization-aware on both the Linear and dynamic GRU strata.
    #
    # Two counts are exposed because a quantized network has two honest
    # answers. The residual count is the registered floating-point
    # parameters torch still traverses; the deployable count adds the
    # packed quantized weights that quantization moved outside parameter
    # registration. Reporting only the first would make a quantized network
    # appear to have lost its weights rather than repacked them, and any
    # compression ratio derived from that would be fabricated.
    #
    # Both counts measure weights alone. Registered buffers are excluded
    # from each, because a buffer is state rather than a weight, and frozen
    # parameters are included in each, because deployability depends on
    # what ships rather than on what trains. The class holds no state, so
    # one instance counts any number of networks.
    def __call__(self, network: nn.Module) -> int:
        # Reports the deployable count: registered floating parameters plus
        # packed quantized weights.
        #
        # Args:
        #     network: The network to traverse, including every submodule.
        #
        # Returns:
        #     The total weight count a deployment of this network would
        #     carry. It equals the residual count exactly when nothing in
        #     the network is quantized.
        return self.residual_count(network) + self._packed_count(network)

    def residual_count(self, network: nn.Module) -> int:
        # Counts only the registered floating-point parameters remaining on the network.
        #
        # This is the unquantized remainder, reported beside the deployable
        # count so a partially quantized network states how much of it was
        # left in floating point.
        return sum(parameter.numel() for parameter in network.parameters())

    def _packed_count(self, network: nn.Module) -> int:
        # Counts packed quantized weights across every module of the network.
        packed_total: int = 0
        for module in network.modules():
            packed_total += self._packed_linear_count(module)
            packed_total += self._packed_gru_count(module)
        return packed_total

    def _packed_linear_count(self, module: nn.Module) -> int:
        # Counts the packed weight and bias of one quantized Linear module.
        if not isinstance(module, quantized_nn.Linear):
            return 0
        weight_and_bias: tuple[Tensor, Tensor | None] = module._weight_bias()
        packed_weight: Tensor = weight_and_bias[0]
        packed_bias: Tensor | None = weight_and_bias[1]
        bias_count: int = packed_bias.numel() if packed_bias is not None else 0
        return packed_weight.numel() + bias_count

    def _packed_gru_count(self, module: nn.Module) -> int:
        # Counts the packed recurrent weights of one dynamic quantized GRU analytically,
        # because the packed cell parameters do not expose a tensor traversal.
        #
        # The geometry is reconstructed from the module's declared shape.
        # Each gate stratum is three times the hidden size, covering the
        # update, reset, and candidate gates. Each direction of each layer
        # holds one input-to-gate matrix and one hidden-to-gate matrix, and
        # a biased layer adds two gate-width vectors per direction. Layer
        # zero reads the module's input width, while every later layer
        # reads the stacked hidden output, which is the hidden size once
        # for a unidirectional stack and twice for a bidirectional one.
        if not isinstance(module, quantized_dynamic_nn.GRU):
            return 0
        direction_count: int = 2 if module.bidirectional else 1
        gate_dimension: int = 3 * module.hidden_size
        packed_total: int = 0
        for layer_index in range(module.num_layers):
            layer_input_dimension: int = (
                module.input_size if layer_index == 0 else module.hidden_size * direction_count
            )
            for _ in range(direction_count):
                packed_total += gate_dimension * layer_input_dimension
                packed_total += gate_dimension * module.hidden_size
                if module.bias:
                    packed_total += 2 * gate_dimension
        return packed_total
