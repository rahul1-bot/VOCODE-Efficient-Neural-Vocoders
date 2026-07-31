# This module:
# 1. Verifies the backbone scope resolver: its lookup precedence, the Linear
#    selection inside the resolved scope, and its closed failures
# 2. Verifies the structured arm on minimal Linear backbones: exact achieved
#    sparsity at the registered curve levels, exact covered parameter
#    fraction, head exclusion, and mask baking into literal zeros
# 3. Verifies the global unstructured arm: the collected weight scope across
#    convolution, linear, recurrent, and weight-normed modules, the exact
#    pooled sparsity, and the propagation of direction-parameter zeros into
#    the effective weight
# 4. Verifies the evaluation-time techniques built on the arms: the masked
#    curve with its sparsity validation, the pruned-and-recovered
#    verification floor, and the dense-continuation causal control
#
# Design decisions:
# - Layer widths are chosen so every registered curve level lands on a whole
#   number of output channels, which makes the achieved sparsity an exact
#   fraction rather than a rounding-tolerant approximation
# - Sparse fixtures are produced by zeroing weights directly instead of by
#   running the pruning arms, so the verification techniques are tested
#   against an independent artifact rather than against their own output
# - Baking is asserted through the absence of mask and original-weight
#   entries in the state dictionary, because the deployment claim is that a
#   recovered checkpoint carries no pruning-hook dependency
# - No recovery fine-tuning is performed anywhere in this file; the
#   verification techniques transform nothing by contract
#
# Author: Rahul Sawhney

import unittest
from typing import override

import torch
from pydantic import ValidationError
from torch import nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm

from syntheticmind.core.module import Module
from syntheticmind.utilities.exceptions import MisconfigurationError

from vocode.optimization.pruning import (
    DenseContinuedIdentity,
    MaskedMagnitudePruning,
    PrunedBackboneScopeResolver,
    PrunedRecoveredVerification,
    StructuredMagnitudePruning,
    StructuredPruningConfig,
    UnstructuredMagnitudePruning,
    UnstructuredPruningConfig,
    UnstructuredWeightScopeCollector,
)


class LinearBackboneNetwork(nn.Module):
    # Backbone-bearing network whose head projection stays outside the pruning scope.
    # Both backbone projections carry ten output channels, so each registered
    # curve level lands on a whole number of channels (three, five, and seven)
    # and the achieved sparsity is an exact fraction rather than a rounded one.
    # Its parameter budget is hand-countable: the backbone weights are eighty
    # plus one hundred, its biases ten plus ten, and the head contributes twenty
    # weights and two biases, giving one hundred eighty covered elements out of
    # two hundred twenty-two parameters.
    def __init__(self) -> None:
        # Builds a two-projection backbone and the head that stays outside the scope.
        super().__init__()
        self.backbone: nn.Sequential = nn.Sequential(nn.Linear(8, 10), nn.Linear(10, 10))
        self.head: nn.Linear = nn.Linear(10, 2)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Runs the backbone and then the excluded head.
        return self.head(self.backbone(mel))


class CoarseChannelBackboneNetwork(nn.Module):
    # Backbone whose channel granularity is too coarse to reach a registered sparsity level.
    # Structured pruning removes a whole number of output channels, so the
    # single-channel projection loses none at all at the thirty-percent level
    # while holding fifty of the scope's seventy weights; the second projection
    # can contribute at most six zeros, leaving the achieved figure far below the
    # registered level and triggering the shortfall refusal.
    def __init__(self) -> None:
        # Builds a backbone whose one-channel projection no fraction can reach.
        super().__init__()
        self.backbone: nn.Sequential = nn.Sequential(nn.Linear(50, 1), nn.Linear(2, 10))

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Runs the coarse backbone.
        return self.backbone(mel)


class ConvolutionalNetwork(nn.Module):
    # Convolution-dominant network taking the global unstructured arm.
    # Its pooled budget is hand-countable: forty-eight convolution weights and
    # seventy-two projection weights make one hundred twenty pooled elements, and
    # the four convolution biases plus nine projection biases bring the network
    # to one hundred thirty-three parameters. Half of the pool is therefore
    # exactly sixty elements.
    def __init__(self) -> None:
        # Builds a convolution and a projection, both pooled by the global arm.
        super().__init__()
        self.convolution: nn.Conv1d = nn.Conv1d(4, 4, 3)
        self.projection: nn.Linear = nn.Linear(8, 9)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Runs the convolution and then the projection.
        return self.projection(self.convolution(mel))


class WeightNormalizedNetwork(nn.Module):
    # Network whose convolution is weight-normed, so masks must hook the direction parameter.
    def __init__(self) -> None:
        # Builds a weight-normed convolution, whose weight is a parametrization.
        super().__init__()
        self.convolution: nn.Module = weight_norm(nn.Conv1d(4, 4, 3))

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Runs the weight-normed convolution.
        return self.convolution(mel)


class RecurrentNetwork(nn.Module):
    # Recurrent network contributing its gate weight matrices to the global pool.
    def __init__(self) -> None:
        # Builds the recurrent layer whose gate matrices join the pool.
        super().__init__()
        self.recurrent: nn.GRU = nn.GRU(4, 4)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Returns the sequence output and discards the final hidden state.
        sequence_output: torch.Tensor
        sequence_output, _ = self.recurrent(mel)
        return sequence_output


class ParameterFreeNetwork(nn.Module):
    # Network without any prunable weight tensor.
    def __init__(self) -> None:
        # Builds a passthrough carrying no prunable weight tensor.
        super().__init__()
        self.passthrough: nn.Identity = nn.Identity()

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Returns the input unchanged.
        return self.passthrough(mel)


class UnderscoreBackboneNetwork(nn.Module):
    # Network exposing the private backbone name alongside a later fallback name.
    def __init__(self) -> None:
        # Builds the private backbone name alongside a later fallback name.
        super().__init__()
        self._backbone: nn.Linear = nn.Linear(4, 4)
        self.convnext: nn.Linear = nn.Linear(4, 4)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Runs the private backbone that wins the lookup.
        return self._backbone(mel)


class ConvnextBackboneNetwork(nn.Module):
    # Network exposing only the final fallback backbone name.
    def __init__(self) -> None:
        # Builds only the final fallback backbone name.
        super().__init__()
        self.convnext: nn.Linear = nn.Linear(4, 4)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Runs the fallback backbone.
        return self.convnext(mel)


class LinearFreeBackboneNetwork(nn.Module):
    # Backbone-bearing network whose scope contains no Linear projection.
    def __init__(self) -> None:
        # Builds a backbone holding a convolution instead of a Linear projection.
        super().__init__()
        self.backbone: nn.Conv1d = nn.Conv1d(4, 4, 3)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Runs the Linear-free backbone.
        return self.backbone(mel)


class TinyHarnessModule(Module):
    # Minimal harness Module exposing the network attribute techniques transform.
    def __init__(self, network: nn.Module) -> None:
        # Binds the network the pruning techniques transform.
        super().__init__()
        self.network: nn.Module = network

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Delegates synthesis to the bound network.
        return self.network(mel)


class LinearBackboneModuleBuilder:
    # Builds a fresh harness module carrying the Linear backbone network.
    def build(self) -> TinyHarnessModule:
        # Seeds construction and returns a fresh dense Linear-backbone module.
        torch.manual_seed(0)
        return TinyHarnessModule(LinearBackboneNetwork())


class ConvolutionalModuleBuilder:
    # Builds a fresh harness module carrying the convolution-dominant network.
    def build(self) -> TinyHarnessModule:
        # Seeds construction and returns a fresh dense convolutional module.
        torch.manual_seed(0)
        return TinyHarnessModule(ConvolutionalNetwork())


class SparseLinearBackboneModuleBuilder:
    # Builds a module whose backbone rows are half zeroed, standing in for a
    # loaded pruned-and-recovered checkpoint rather than for pruning output.
    # Zeroing the weights directly rather than running the structured arm is
    # deliberate: it makes the verification technique face an artifact it did not
    # produce, which is the situation a real recovered checkpoint presents.
    def build(self) -> TinyHarnessModule:
        # Seeds construction, then zeroes the leading half of every backbone row.
        # Whole rows are zeroed because the structured arm removes whole output
        # channels, so the fixture is sparse in the same shape a real recovered
        # checkpoint is, and the resulting figure is exactly one half.
        #
        # Returns:
        #     A module whose backbone reads as half sparse and whose head
        #     stays dense.
        torch.manual_seed(0)
        module: TinyHarnessModule = TinyHarnessModule(LinearBackboneNetwork())
        backbone: nn.Sequential = getattr(module.network, "backbone")
        with torch.no_grad():
            linear_module: nn.Linear
            for linear_module in backbone:
                zeroed_rows: int = linear_module.weight.shape[0] // 2
                linear_module.weight[:zeroed_rows] = 0.0
        return module


class SparseConvolutionalModuleBuilder:
    # Builds a convolution-dominant module whose pooled weights are half zeroed.
    # It is the global-arm counterpart of the sparse backbone builder and is
    # likewise produced by direct zeroing rather than by running the arm.
    def build(self) -> TinyHarnessModule:
        # Seeds construction, then zeroes the leading half of each pooled tensor.
        # Each tensor is zeroed in its flattened view because the global arm
        # removes individual weights rather than whole channels; halving both
        # tensors makes the pooled figure exactly one half regardless of how the
        # pool weighs them.
        #
        # Returns:
        #     A module whose pooled weight tensors read as half sparse.
        torch.manual_seed(0)
        module: TinyHarnessModule = TinyHarnessModule(ConvolutionalNetwork())
        weight_owners: tuple[nn.Module, ...] = (
            getattr(module.network, "convolution"),
            getattr(module.network, "projection")
        )
        with torch.no_grad():
            weight_owner: nn.Module
            for weight_owner in weight_owners:
                flattened: torch.Tensor = weight_owner.weight.view(-1)
                flattened[: flattened.numel() // 2] = 0.0
        return module


class StructuredPruningConfigurationTest(unittest.TestCase):
    # Verifies the frozen structured-arm settings and their validated domains.
    def setUp(self) -> None:
        # Binds the default structured settings the default and mutation cases read.
        self._configuration: StructuredPruningConfig = StructuredPruningConfig()

    def test_default_configuration_prunes_half_the_output_channels_by_l2_norm(self) -> None:
        # The default arm ranks output channels by the second norm at half sparsity.
        self.assertEqual(self._configuration.pruning_amount, 0.5)
        self.assertEqual(self._configuration.norm_degree, 2)
        self.assertEqual(self._configuration.channel_dimension, 0)
        self.assertEqual(self._configuration.minimum_accepted_sparsity, 0.45)

    def test_degenerate_pruning_amounts_are_refused(self) -> None:
        # A pruning amount must remain strictly inside the open unit interval.
        with self.assertRaises(ValidationError):
            StructuredPruningConfig(pruning_amount=0.0)
        with self.assertRaises(ValidationError):
            StructuredPruningConfig(pruning_amount=1.0)

    def test_non_positive_norm_degree_is_refused(self) -> None:
        # The ranking norm degree must be a positive integer.
        with self.assertRaises(ValidationError):
            StructuredPruningConfig(norm_degree=0)

    def test_configuration_rejects_mutation_and_unknown_fields(self) -> None:
        # The arm settings are frozen and closed to extra keys.
        with self.assertRaises(ValidationError):
            self._configuration.pruning_amount = 0.7
        with self.assertRaises(ValidationError):
            StructuredPruningConfig(unknown_field=1)


class UnstructuredPruningConfigurationTest(unittest.TestCase):
    # Verifies the frozen global unstructured-arm settings and their validated domains.
    def setUp(self) -> None:
        # Binds the default global settings the default and mutation cases read.
        self._configuration: UnstructuredPruningConfig = UnstructuredPruningConfig()

    def test_default_configuration_prunes_half_the_pooled_weights(self) -> None:
        # The default global arm zeroes half the pooled magnitude distribution.
        self.assertEqual(self._configuration.pruning_amount, 0.5)
        self.assertEqual(self._configuration.minimum_accepted_sparsity, 0.45)

    def test_degenerate_pruning_amounts_are_refused(self) -> None:
        # A pruning amount must remain strictly inside the open unit interval.
        with self.assertRaises(ValidationError):
            UnstructuredPruningConfig(pruning_amount=0.0)
        with self.assertRaises(ValidationError):
            UnstructuredPruningConfig(pruning_amount=1.0)

    def test_configuration_rejects_mutation_and_unknown_fields(self) -> None:
        # The arm settings are frozen and closed to extra keys.
        with self.assertRaises(ValidationError):
            self._configuration.pruning_amount = 0.3
        with self.assertRaises(ValidationError):
            UnstructuredPruningConfig(unknown_field=1)


class PrunedBackboneScopeResolutionTest(unittest.TestCase):
    # Verifies the backbone lookup precedence and the Linear selection inside it.
    def setUp(self) -> None:
        # Seeds construction and binds the scope resolver under test.
        torch.manual_seed(0)
        self._resolver: PrunedBackboneScopeResolver = PrunedBackboneScopeResolver()

    def test_private_backbone_name_wins_over_the_later_fallback(self) -> None:
        # The lookup order is private backbone, public backbone, then the convnext scope.
        network: UnderscoreBackboneNetwork = UnderscoreBackboneNetwork()
        self.assertIs(self._resolver.resolve(network), getattr(network, "_backbone"))

    def test_public_backbone_name_is_resolved(self) -> None:
        # A public backbone attribute is the resolved pruning scope.
        network: LinearBackboneNetwork = LinearBackboneNetwork()
        self.assertIs(self._resolver.resolve(network), getattr(network, "backbone"))

    def test_final_fallback_scope_is_resolved(self) -> None:
        # The convnext scope is the last registered backbone name.
        network: ConvnextBackboneNetwork = ConvnextBackboneNetwork()
        self.assertIs(self._resolver.resolve(network), getattr(network, "convnext"))

    def test_network_without_any_registered_backbone_name_fails_closed(self) -> None:
        # An unrecognised network is a configuration error, not a silently empty scope.
        with self.assertRaisesRegex(MisconfigurationError, "No prunable backbone scope found"):
            self._resolver.resolve(ParameterFreeNetwork())

    def test_selected_linear_modules_are_named_within_the_backbone_scope(self) -> None:
        # Selection returns the named Linear projections of the resolved scope only.
        # The names are the sequential positions inside the backbone rather than
        # network-level paths, which is the evidence that the walk starts at the
        # resolved scope and not at the network root.
        network: LinearBackboneNetwork = LinearBackboneNetwork()
        selected: list[tuple[str, nn.Linear]] = self._resolver.select_linear_modules(network)
        self.assertEqual([module_name for module_name, _ in selected], ["0", "1"])
        self.assertNotIn(
            getattr(network, "head"),
            [linear_module for _, linear_module in selected],
            msg="The head projection must stay structurally outside the pruning scope."
        )

    def test_backbone_without_linear_projections_fails_closed(self) -> None:
        # The structured lane does not apply to a scope with no Linear modules.
        with self.assertRaisesRegex(MisconfigurationError, "no Linear modules"):
            self._resolver.select_linear_modules(LinearFreeBackboneNetwork())


class StructuredMagnitudePruningTest(unittest.TestCase):
    # Verifies achieved sparsity, coverage, and mask baking of the structured arm.
    def setUp(self) -> None:
        # Seeds construction and binds one network with the default structured arm.
        torch.manual_seed(0)
        self._network: LinearBackboneNetwork = LinearBackboneNetwork()
        self._pruning: StructuredMagnitudePruning = StructuredMagnitudePruning(StructuredPruningConfig())

    def test_applied_masks_report_the_pruned_module_paths(self) -> None:
        # The arm reports exactly the backbone projections it masked.
        self.assertEqual(self._pruning.apply_masks(self._network), ["0", "1"])

    def test_registered_levels_achieve_their_exact_channel_sparsity(self) -> None:
        # Whole-channel pruning lands exactly on each registered curve level.
        # Twelve decimal places rather than a loose tolerance is the right
        # comparison because the achieved figure is a rational number over ten
        # channels, not an approximation: a level that misses is a real defect,
        # never rounding noise.
        pruning_amount: float
        for pruning_amount in (0.3, 0.5, 0.7):
            torch.manual_seed(0)
            network: LinearBackboneNetwork = LinearBackboneNetwork()
            pruning: StructuredMagnitudePruning = StructuredMagnitudePruning(
                StructuredPruningConfig(pruning_amount=pruning_amount)
            )
            pruning.apply_masks(network)
            pruning.bake_masks(network)
            self.assertAlmostEqual(
                pruning.measured_sparsity(network),
                pruning_amount,
                places=12,
                msg=f"Structured pruning at {pruning_amount} must zero exactly that channel fraction."
            )

    def test_baking_removes_the_pruning_hook_state(self) -> None:
        # A baked checkpoint carries literal zeros and no pruning-hook dependency.
        self._pruning.apply_masks(self._network)
        masked_keys: list[str] = list(self._network.state_dict().keys())
        self.assertTrue(any(key.endswith("weight_mask") for key in masked_keys))
        self._pruning.bake_masks(self._network)
        baked_keys: list[str] = list(self._network.state_dict().keys())
        self.assertFalse(any(key.endswith("weight_mask") for key in baked_keys))
        self.assertFalse(any(key.endswith("weight_orig") for key in baked_keys))
        self.assertIsInstance(getattr(self._network, "backbone")[0].weight, nn.Parameter)

    def test_head_projection_stays_dense_after_pruning(self) -> None:
        # Output-channel pruning of a spectral head is unrecoverable, so the head is excluded.
        self._pruning.apply_masks(self._network)
        self._pruning.bake_masks(self._network)
        head_weight: torch.Tensor = getattr(self._network, "head").weight.detach()
        self.assertEqual(int((head_weight == 0.0).sum().item()), 0)

    def test_covered_parameter_fraction_is_the_exact_backbone_weight_share(self) -> None:
        # Coverage is the exact parameter share the pruning scope reaches.
        self.assertAlmostEqual(
            self._pruning.covered_parameter_fraction(self._network),
            180.0 / 222.0,
            places=12,
            msg="The backbone weights are 180 of the network's 222 parameters."
        )

    def test_sparsity_measurement_on_an_unresolvable_network_fails_closed(self) -> None:
        # Measurement requires the resolved scope; an unrecognised network is an error.
        with self.assertRaises(MisconfigurationError):
            self._pruning.measured_sparsity(ParameterFreeNetwork())

    def test_configuration_dump_records_the_arm_scope_and_settings(self) -> None:
        # The recipe states the exact structured configuration and its target scope.
        self.assertEqual(
            self._pruning.configuration_dump(),
            {
                "technique": "structured_magnitude_pruning",
                "pruning_amount": 0.5,
                "norm_degree": 2,
                "channel_dimension": 0,
                "target_module_kind": "torch.nn.Linear",
                "target_scope": "generator_backbone"
            }
        )


class UnstructuredWeightScopeCollectionTest(unittest.TestCase):
    # Verifies which weight tensors enter the single global magnitude pool.
    def setUp(self) -> None:
        # Seeds construction and binds the weight-scope collector under test.
        torch.manual_seed(0)
        self._collector: UnstructuredWeightScopeCollector = UnstructuredWeightScopeCollector()

    def test_convolution_and_linear_weights_enter_the_pool(self) -> None:
        # Both convolutional and linear weight matrices are pooled network-wide.
        collected: list[tuple[str, nn.Module, str]] = self._collector.collect(ConvolutionalNetwork())
        self.assertEqual(
            [(module_path, parameter_name) for module_path, _, parameter_name in collected],
            [("convolution", "weight"), ("projection", "weight")]
        )

    def test_weight_normed_module_contributes_its_direction_parameter(self) -> None:
        # Masks hook the direction parameter, whose zeros propagate into the effective weight.
        network: WeightNormalizedNetwork = WeightNormalizedNetwork()
        collected: list[tuple[str, nn.Module, str]] = self._collector.collect(network)
        self.assertEqual(len(collected), 1)
        module_path: str
        scoped_module: nn.Module
        parameter_name: str
        module_path, scoped_module, parameter_name = collected[0]
        self.assertEqual(module_path, "convolution.parametrizations.weight")
        self.assertEqual(parameter_name, "original1")
        self.assertIs(scoped_module, getattr(network, "convolution").parametrizations["weight"])
        self.assertTrue(parametrize.is_parametrized(getattr(network, "convolution"), "weight"))

    def test_recurrent_module_contributes_its_gate_weight_matrices_in_order(self) -> None:
        # Recurrent gate matrices join the pool under their logical names, sorted.
        collected: list[tuple[str, nn.Module, str]] = self._collector.collect(RecurrentNetwork())
        self.assertEqual(
            [(module_path, parameter_name) for module_path, _, parameter_name in collected],
            [("recurrent", "weight_hh_l0"), ("recurrent", "weight_ih_l0")]
        )

    def test_network_without_prunable_weights_fails_closed(self) -> None:
        # An empty global scope is a configuration error, not a silent no-op.
        with self.assertRaisesRegex(MisconfigurationError, "no prunable weight tensors"):
            self._collector.collect(ParameterFreeNetwork())


class UnstructuredMagnitudePruningTest(unittest.TestCase):
    # Verifies the pooled sparsity, coverage, and baking of the global unstructured arm.
    def setUp(self) -> None:
        # Seeds construction and binds one network with the default global arm.
        torch.manual_seed(0)
        self._network: ConvolutionalNetwork = ConvolutionalNetwork()
        self._pruning: UnstructuredMagnitudePruning = UnstructuredMagnitudePruning(UnstructuredPruningConfig())

    def test_applied_masks_report_the_pruned_parameter_paths(self) -> None:
        # The arm reports every pooled parameter path it masked.
        self.assertEqual(
            self._pruning.apply_masks(self._network),
            ["convolution.weight", "projection.weight"]
        )

    def test_pooled_sparsity_matches_the_registered_level_exactly(self) -> None:
        # One global magnitude pool zeroes exactly the registered fraction of pooled weights.
        self._pruning.apply_masks(self._network)
        self._pruning.bake_masks(self._network)
        self.assertAlmostEqual(self._pruning.measured_sparsity(self._network), 0.5, places=12)

    def test_pooled_sparsity_is_global_rather_than_per_tensor(self) -> None:
        # The pool is shared, so individual tensors need not each reach the level.
        # The assertion is deliberately on the summed zero count over the summed
        # element count rather than on either tensor alone, because that total is
        # the only quantity the global arm actually controls.
        self._pruning.apply_masks(self._network)
        self._pruning.bake_masks(self._network)
        convolution_weight: torch.Tensor = getattr(self._network, "convolution").weight.detach()
        projection_weight: torch.Tensor = getattr(self._network, "projection").weight.detach()
        total_zeros: int = int((convolution_weight == 0.0).sum().item() + (projection_weight == 0.0).sum().item())
        self.assertEqual(total_zeros, 60)
        self.assertEqual(convolution_weight.numel() + projection_weight.numel(), 120)

    def test_baking_removes_the_pruning_hook_state(self) -> None:
        # The baked network deploys without any pruning-hook dependency.
        self._pruning.apply_masks(self._network)
        self._pruning.bake_masks(self._network)
        baked_keys: list[str] = list(self._network.state_dict().keys())
        self.assertFalse(any(key.endswith("weight_mask") for key in baked_keys))
        self.assertFalse(any(key.endswith("weight_orig") for key in baked_keys))

    def test_direction_parameter_zeros_propagate_into_the_effective_weight(self) -> None:
        # Pruning the direction parameter of a weight-normed module zeroes the real weight.
        torch.manual_seed(0)
        network: WeightNormalizedNetwork = WeightNormalizedNetwork()
        self._pruning.apply_masks(network)
        self._pruning.bake_masks(network)
        parametrization_list: nn.Module = getattr(network, "convolution").parametrizations["weight"]
        direction: torch.Tensor = getattr(parametrization_list, "original1").detach()
        effective_weight: torch.Tensor = getattr(network, "convolution").weight.detach()
        self.assertAlmostEqual(
            float((direction == 0.0).sum().item()) / direction.numel(),
            0.5,
            places=12
        )
        self.assertTrue(
            torch.equal(direction == 0.0, effective_weight == 0.0),
            msg="Direction-parameter zeros must map exactly onto effective-weight zeros."
        )

    def test_covered_parameter_fraction_is_the_exact_pooled_weight_share(self) -> None:
        # Coverage is the exact parameter share the global scope reaches.
        self.assertAlmostEqual(
            self._pruning.covered_parameter_fraction(self._network),
            120.0 / 133.0,
            places=12,
            msg="The pooled weights are 120 of the network's 133 parameters."
        )

    def test_configuration_dump_records_the_arm_scope_and_settings(self) -> None:
        # The recipe states the exact global configuration and its target scope.
        self.assertEqual(
            self._pruning.configuration_dump(),
            {
                "technique": "unstructured_magnitude_pruning",
                "pruning_amount": 0.5,
                "target_module_kinds": "conv_linear_gru_weights",
                "target_scope": "global_network"
            }
        )


class MaskedMagnitudePruningCurveTest(unittest.TestCase):
    # Verifies the evaluation-time masked curve on both architecture-resolved arms.
    def setUp(self) -> None:
        # Binds one dense builder per arm, so each case starts from an unpruned module.
        self._structured_builder: LinearBackboneModuleBuilder = LinearBackboneModuleBuilder()
        self._unstructured_builder: ConvolutionalModuleBuilder = ConvolutionalModuleBuilder()

    def test_technique_reports_the_produced_variant_name(self) -> None:
        # The technique carries the curve point it produces.
        technique: MaskedMagnitudePruning = MaskedMagnitudePruning("linear_structured", 0.5, "pruned_50")
        self.assertEqual(technique.name, "pruned_50")

    def test_structured_arm_reaches_every_registered_curve_level(self) -> None:
        # The registered curve levels are achieved exactly on the structured arm.
        curve_levels: dict[str, float] = {"pruned_30": 0.3, "pruned_50": 0.5, "pruned_70": 0.7}
        variant_name: str
        pruning_amount: float
        for variant_name, pruning_amount in curve_levels.items():
            module: TinyHarnessModule = self._structured_builder.build()
            technique: MaskedMagnitudePruning = MaskedMagnitudePruning(
                "linear_structured",
                pruning_amount,
                variant_name
            )
            returned: Module = technique.apply(module)
            self.assertIs(returned, module)
            self.assertAlmostEqual(
                float(technique.configuration_dump()["measured_sparsity"]),
                pruning_amount,
                places=12,
                msg=f"{variant_name} must reach its registered sparsity level exactly."
            )

    def test_unstructured_arm_reaches_every_registered_curve_level(self) -> None:
        # The registered curve levels are achieved exactly on the global arm.
        curve_levels: dict[str, float] = {"pruned_30": 0.3, "pruned_50": 0.5, "pruned_70": 0.7}
        variant_name: str
        pruning_amount: float
        for variant_name, pruning_amount in curve_levels.items():
            module: TinyHarnessModule = self._unstructured_builder.build()
            technique: MaskedMagnitudePruning = MaskedMagnitudePruning(
                "global_unstructured",
                pruning_amount,
                variant_name
            )
            technique.apply(module)
            self.assertAlmostEqual(
                float(technique.configuration_dump()["measured_sparsity"]),
                pruning_amount,
                places=12,
                msg=f"{variant_name} must reach its registered sparsity level exactly."
            )

    def test_masks_are_baked_so_the_evaluated_object_carries_literal_zeros(self) -> None:
        # The evaluated object is the masked network itself, without pruning hooks.
        module: TinyHarnessModule = self._structured_builder.build()
        technique: MaskedMagnitudePruning = MaskedMagnitudePruning("linear_structured", 0.5, "pruned_50")
        technique.apply(module)
        state_keys: list[str] = list(module.state_dict().keys())
        self.assertFalse(any(key.endswith("weight_mask") for key in state_keys))
        self.assertFalse(any(key.endswith("weight_orig") for key in state_keys))

    def test_configuration_dump_records_the_arm_level_sparsity_and_coverage(self) -> None:
        # The recipe records what was pruned, how much, and over which parameter share.
        module: TinyHarnessModule = self._structured_builder.build()
        technique: MaskedMagnitudePruning = MaskedMagnitudePruning("linear_structured", 0.5, "pruned_50")
        technique.apply(module)
        dump: dict[str, object] = technique.configuration_dump()
        self.assertEqual(dump["technique"], "masked_magnitude_pruning")
        self.assertEqual(dump["pruning_arm"], "linear_structured")
        self.assertEqual(dump["pruning_amount"], 0.5)
        self.assertAlmostEqual(float(dump["covered_parameter_fraction"]), 180.0 / 222.0, places=12)

    def test_configuration_dump_before_apply_leaves_measurements_unrecorded(self) -> None:
        # Nothing is claimed about a transformation that has not run.
        technique: MaskedMagnitudePruning = MaskedMagnitudePruning("global_unstructured", 0.3, "pruned_30")
        self.assertEqual(
            technique.configuration_dump(),
            {
                "technique": "masked_magnitude_pruning",
                "pruning_arm": "global_unstructured",
                "pruning_amount": 0.3,
                "measured_sparsity": None,
                "covered_parameter_fraction": None
            }
        )


class MaskedMagnitudePruningValidationTest(unittest.TestCase):
    # Verifies that a curve point falling short of its registered level fails closed.
    def setUp(self) -> None:
        # Seeds construction and binds the module whose channels are too coarse.
        torch.manual_seed(0)
        self._module: TinyHarnessModule = TinyHarnessModule(CoarseChannelBackboneNetwork())

    def test_sparsity_shortfall_is_a_configuration_error(self) -> None:
        # Channel granularity too coarse for the registered level is an error, not a smaller experiment.
        # Refusing rather than proceeding is what stops a curve point from being
        # published under a level it never reached, which would corrupt the
        # robustness curve while looking like a completed run.
        technique: MaskedMagnitudePruning = MaskedMagnitudePruning("linear_structured", 0.3, "pruned_30")
        with self.assertRaisesRegex(MisconfigurationError, "below the"):
            technique.apply(self._module)


class DenseContinuedIdentityTest(unittest.TestCase):
    # Verifies the causal control that proves a continuation checkpoint stayed dense.
    def setUp(self) -> None:
        # Binds a dense builder per arm, a sparse builder, and the structured control.
        self._structured_builder: LinearBackboneModuleBuilder = LinearBackboneModuleBuilder()
        self._unstructured_builder: ConvolutionalModuleBuilder = ConvolutionalModuleBuilder()
        self._sparse_builder: SparseLinearBackboneModuleBuilder = SparseLinearBackboneModuleBuilder()
        self._technique: DenseContinuedIdentity = DenseContinuedIdentity("linear_structured")

    def test_technique_reports_its_canonical_variant_name(self) -> None:
        # The technique produces the dense-continuation control row.
        self.assertEqual(self._technique.name, "dense_continued")

    def test_dense_checkpoint_passes_and_is_transformed_by_nothing(self) -> None:
        # The control applies no transformation; it only proves density.
        module: TinyHarnessModule = self._structured_builder.build()
        original_weight: torch.Tensor = getattr(module.network, "backbone")[0].weight.detach().clone()
        returned: Module = self._technique.apply(module)
        self.assertIs(returned, module)
        self.assertTrue(torch.equal(getattr(module.network, "backbone")[0].weight.detach(), original_weight))
        self.assertEqual(self._technique.configuration_dump()["measured_sparsity"], 0.0)

    def test_dense_checkpoint_passes_on_the_global_arm(self) -> None:
        # Density is measured over whichever scope the resolved arm covers.
        technique: DenseContinuedIdentity = DenseContinuedIdentity("global_unstructured")
        technique.apply(self._unstructured_builder.build())
        self.assertEqual(technique.configuration_dump()["measured_sparsity"], 0.0)

    def test_sparse_checkpoint_is_rejected_as_a_causal_control(self) -> None:
        # A sparse continuation checkpoint cannot serve as the dense causal control.
        with self.assertRaisesRegex(MisconfigurationError, "must stay dense"):
            self._technique.apply(self._sparse_builder.build())

    def test_configuration_dump_records_the_arm_and_measurement(self) -> None:
        # The recipe states the arm the density claim was measured over.
        self._technique.apply(self._structured_builder.build())
        self.assertEqual(
            self._technique.configuration_dump(),
            {
                "technique": "dense_continued_identity",
                "pruning_arm": "linear_structured",
                "measured_sparsity": 0.0
            }
        )


class PrunedRecoveredVerificationTest(unittest.TestCase):
    # Verifies the load-time proof that a recovered checkpoint is as sparse as its name claims.
    def setUp(self) -> None:
        # Binds dense and sparse builders per arm and the structured verification.
        self._dense_builder: LinearBackboneModuleBuilder = LinearBackboneModuleBuilder()
        self._sparse_builder: SparseLinearBackboneModuleBuilder = SparseLinearBackboneModuleBuilder()
        self._sparse_convolutional_builder: SparseConvolutionalModuleBuilder = SparseConvolutionalModuleBuilder()
        self._technique: PrunedRecoveredVerification = PrunedRecoveredVerification("linear_structured")

    def test_default_technique_produces_the_full_budget_recovery_name(self) -> None:
        # The default recovery arm is the full-budget recovered variant.
        self.assertEqual(self._technique.name, "pruned_50_recovered")

    def test_half_budget_technique_produces_the_ablation_name(self) -> None:
        # The half-budget arm is the recovery-budget ablation row.
        technique: PrunedRecoveredVerification = PrunedRecoveredVerification(
            "linear_structured",
            produced_variant_name="pruned_50_recovered_half"
        )
        self.assertEqual(technique.name, "pruned_50_recovered_half")

    def test_recovered_checkpoint_passes_and_is_transformed_by_nothing(self) -> None:
        # Verification transforms nothing; it proves a property of the loaded weights.
        module: TinyHarnessModule = self._sparse_builder.build()
        original_weight: torch.Tensor = getattr(module.network, "backbone")[0].weight.detach().clone()
        returned: Module = self._technique.apply(module)
        self.assertIs(returned, module)
        self.assertTrue(torch.equal(getattr(module.network, "backbone")[0].weight.detach(), original_weight))
        self.assertAlmostEqual(float(self._technique.configuration_dump()["measured_sparsity"]), 0.5, places=12)

    def test_recovered_checkpoint_passes_on_the_global_arm(self) -> None:
        # Sparsity is measured over whichever scope the resolved arm covers.
        technique: PrunedRecoveredVerification = PrunedRecoveredVerification("global_unstructured")
        technique.apply(self._sparse_convolutional_builder.build())
        self.assertAlmostEqual(float(technique.configuration_dump()["measured_sparsity"]), 0.5, places=12)

    def test_dense_checkpoint_is_rejected_below_the_accepted_floor(self) -> None:
        # A dense artifact is not a pruned-and-recovered artifact.
        with self.assertRaisesRegex(MisconfigurationError, "below the accepted minimum"):
            self._technique.apply(self._dense_builder.build())

    def test_configuration_dump_records_the_floor_and_the_arm_configuration(self) -> None:
        # The recipe carries the accepted floor and the arm's own configuration.
        self._technique.apply(self._sparse_builder.build())
        dump: dict[str, object] = self._technique.configuration_dump()
        self.assertEqual(dump["technique"], "pruned_recovered_verification")
        self.assertEqual(dump["produced_variant_name"], "pruned_50_recovered")
        self.assertEqual(dump["pruning_arm"], "linear_structured")
        self.assertEqual(dump["minimum_accepted_sparsity"], 0.45)
        self.assertEqual(
            dump["pruning_configuration"],
            {
                "technique": "structured_magnitude_pruning",
                "pruning_amount": 0.5,
                "norm_degree": 2,
                "channel_dimension": 0,
                "target_module_kind": "torch.nn.Linear",
                "target_scope": "generator_backbone"
            }
        )


if __name__ == "__main__":
    unittest.main()
