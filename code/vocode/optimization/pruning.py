# This module:
# 1. Implements the two magnitude-pruning arms of Study 2: structured
#    channel pruning over the backbone Linear projections, and global
#    unstructured pruning pooling convolution, linear, and recurrent
#    weights network-wide
# 2. Implements the evaluation-time techniques built on those arms: the
#    masked pruning curve (30/50/70), the pruned-and-recovered
#    verification, and the dense-continuation causal control
#
# Harness contract (syntheticmind):
# - Every technique maps a harness Module onto a harness Module; masks
#   are applied and baked on the module's network in place, and the
#   verification techniques transform nothing, proving properties of the
#   loaded checkpoint instead
#
# Design decisions:
# - Masks are baked into literal zero weights immediately, so evaluated
#   and recovered checkpoints deploy without any pruning-hook dependency
# - Spectral head projections stay structurally out of the pruning scope
#   because output-channel pruning of a spectral head removes frequency
#   bins, which no fine-tuning can recover
# - Weight-normed modules are pruned through their direction parameter,
#   whose zeros propagate exactly into the effective weight, so masks
#   always hook real parameter tensors
# - Achieved sparsity is measured after baking and validated against the
#   registered level; the masked lane claims a robustness curve only and
#   no deployment acceleration, because dense zeros do not accelerate
#   dense kernels
#
# Author: Rahul Sawhney

from typing import ClassVar

import torch
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from torch import nn
from torch.nn.utils import parametrize, prune

from syntheticmind.core.module import Module
from syntheticmind.utilities.exceptions import MisconfigurationError

from vocode.optimization.registry import OptimizationTechnique, OptimizationVariantName, PruningArmName

__all__: list[str] = [
    "DenseContinuedIdentity",
    "MaskedMagnitudePruning",
    "PrunedBackboneScopeResolver",
    "PrunedRecoveredVerification",
    "StructuredMagnitudePruning",
    "StructuredPruningConfig",
    "UnstructuredMagnitudePruning",
    "UnstructuredPruningConfig"
]


class StructuredPruningConfig(BaseModel):
    # Frozen structured-arm settings: pruning amount, ranking norm, channel
    # dimension, and the accepted sparsity floor for recovered checkpoints.
    #
    # Fields:
    #     pruning_amount: Fraction of output channels zeroed in every
    #         covered Linear projection, constrained to the open unit
    #         interval because neither an empty nor a total intervention is
    #         a curve point. Default: ``0.5``.
    #     norm_degree: Degree of the norm that ranks output channels by
    #         magnitude; the smallest-norm channels are the ones zeroed.
    #         Default: ``2``.
    #     channel_dimension: Weight-tensor dimension along which whole
    #         slices are removed; dimension zero is the Linear output-channel
    #         axis. Default: ``0``.
    #     minimum_accepted_sparsity: Sparsity floor a loaded checkpoint must
    #         reach before the pruned-and-recovered verification accepts it
    #         as a recovered artifact. It sits below the pruning amount
    #         because recovery fine-tuning is free to leave a small number
    #         of masked weights numerically indistinguishable from zero.
    #         This setting governs verification only and never the arm's own
    #         masking. Default: ``0.45``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    pruning_amount: float = Field(default=0.5, gt=0.0, lt=1.0)
    norm_degree: PositiveInt = 2
    channel_dimension: int = 0
    minimum_accepted_sparsity: float = Field(default=0.45, gt=0.0, lt=1.0)


class PrunedBackboneScopeResolver:
    # Resolver locating the prunable backbone scope inside a vocoder network.
    # The head projections stay out of scope structurally because output-channel pruning
    # of a spectral head removes frequency bins, which is unrecoverable by fine-tuning.
    #
    # Integration: the exclusion of the head is enforced by the shape of the
    # lookup rather than by a filter. The resolver returns a submodule and the
    # selection then walks only inside it, so anything the network exposes
    # outside the resolved backbone attribute is structurally unreachable by the
    # structured arm. A network exposing none of the three registered backbone
    # names is refused rather than defaulted to the whole network, because
    # defaulting would silently place the synthesis head inside the scope.
    def resolve(self, network: nn.Module) -> nn.Module:
        # Resolves the backbone submodule whose Linear projections are the pruning targets.
        # The three registered attribute names are tried in a fixed precedence
        # order, so a network exposing more than one of them always resolves the
        # same scope.
        #
        # Args:
        #     network: The generator network whose prunable scope is sought.
        #
        # Raises:
        #     MisconfigurationError: If the network exposes none of the
        #         registered backbone attribute names, which means the
        #         structured arm has no reviewed scope on this architecture.
        #
        # Returns:
        #     The backbone submodule the structured arm is confined to.
        if hasattr(network, "_backbone"):
            return getattr(network, "_backbone")
        if hasattr(network, "backbone"):
            return getattr(network, "backbone")
        if hasattr(network, "convnext"):
            return getattr(network, "convnext")
        raise MisconfigurationError(
            "No prunable backbone scope found: the network exposes none of _backbone, "
            "backbone, or convnext; extend PrunedBackboneScopeResolver deliberately."
        )

    def select_linear_modules(self, network: nn.Module) -> list[tuple[str, nn.Linear]]:
        # Returns the named Linear modules inside the resolved backbone scope.
        # Names are relative to the backbone, not to the network, and the
        # traversal is recursive, so Linear projections nested inside container
        # modules of the backbone are covered. This selection defines the
        # structured arm's scope for masking, baking, sparsity measurement, and
        # coverage measurement alike, which is why all four report over exactly
        # the same set of tensors.
        #
        # Args:
        #     network: The generator network whose backbone scope is walked.
        #
        # Raises:
        #     MisconfigurationError: If the network exposes no registered
        #         backbone name, or if the resolved scope holds no Linear
        #         module, in which case the structured arm does not apply to
        #         this architecture as configured.
        #
        # Returns:
        #     The (backbone-relative name, module) pairs of every Linear
        #     projection inside the resolved scope.
        backbone: nn.Module = self.resolve(network)
        selected: list[tuple[str, nn.Linear]] = [
            (module_name, module)
            for module_name, module in backbone.named_modules()
            if isinstance(module, nn.Linear)
        ]
        if not selected:
            raise MisconfigurationError(
                "The resolved backbone scope contains no Linear modules; the structured "
                "pruning lane does not apply to this architecture as configured."
            )
        return selected


class StructuredMagnitudePruning:
    # Structured magnitude pruning over the backbone Linear projections.
    # Masks zero whole output channels by L-norm ranking; baking makes the zeros literal
    # so the recovered checkpoint deploys without any pruning hook dependency.
    #
    # Scope: the arm covers exactly the Linear projections the resolver finds
    # inside the generator backbone, and nothing else in the network. Pruning is
    # structured, meaning whole output channels are removed rather than
    # individual weights: every covered projection independently loses the same
    # fraction of its output channels, ranked by the norm of each channel. Per
    # tensor achieved sparsity therefore equals the registered amount exactly
    # whenever the channel count admits the fraction, and the arm is the one
    # that applies to Linear-bearing backbones.
    #
    # Mask-baking semantics: applying a mask leaves torch's pruning
    # reparametrization in place, so the module then carries a weight_orig
    # parameter, a weight_mask buffer, and a forward pre-hook that recomputes
    # the effective weight. Baking multiplies the two together, restores weight
    # as an ordinary parameter holding literal zeros, and drops the hook and both
    # auxiliary entries from the state dictionary. Every artifact this study
    # persists or evaluates is baked, which is what lets a recovered checkpoint
    # load into an unmodified network; recovery training is the single phase
    # where masks stay unbaked, because the mask must hold the zeros in place
    # while the optimizer runs.
    def __init__(self, configuration: StructuredPruningConfig) -> None:
        # Binds the arm settings and the backbone scope resolver.
        self._configuration: StructuredPruningConfig = configuration
        self._scope_resolver: PrunedBackboneScopeResolver = PrunedBackboneScopeResolver()

    @property
    def configuration(self) -> StructuredPruningConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    def apply_masks(self, network: nn.Module) -> list[str]:
        # Applies structured magnitude masks and returns the pruned module paths.
        # Each covered projection is masked independently, so the amount is a
        # per-tensor fraction of output channels rather than a network-wide
        # pool. The masks remain live reparametrizations after this call; the
        # network holds literal zeros only once bake_masks has run.
        #
        # Args:
        #     network: The generator network whose backbone projections are
        #         masked in place.
        #
        # Raises:
        #     MisconfigurationError: If the network exposes no resolvable
        #         backbone scope, or if that scope holds no Linear module.
        #
        # Returns:
        #     The backbone-relative names of every masked projection, in
        #     traversal order, which the recovery recipe records as its mask
        #     inventory.
        pruned_paths: list[str] = []
        for module_name, linear_module in self._scope_resolver.select_linear_modules(network):
            prune.ln_structured(
                linear_module,
                name="weight",
                amount=self._configuration.pruning_amount,
                n=self._configuration.norm_degree,
                dim=self._configuration.channel_dimension
            )
            pruned_paths.append(module_name)
        return pruned_paths

    def bake_masks(self, network: nn.Module) -> None:
        # Bakes the masks into literal zero weights and removes the pruning hooks.
        # After this call the covered projections carry an ordinary weight
        # parameter whose masked entries are exactly zero, and their state
        # dictionaries no longer contain the weight_orig or weight_mask entries,
        # so the baked network serializes and loads without any pruning
        # dependency.
        #
        # Args:
        #     network: The masked network whose covered projections are baked
        #         in place.
        for _, linear_module in self._scope_resolver.select_linear_modules(network):
            prune.remove(linear_module, name="weight")

    def measured_sparsity(self, network: nn.Module) -> float:
        # Measures the achieved zero fraction over the covered Linear weights.
        # The measurement counts literal zeros in the effective weight, so it is
        # meaningful both before masking (reporting the density of a loaded
        # checkpoint) and after baking (reporting what the intervention
        # achieved); read on an unbaked network it reflects the masked effective
        # weight the pre-hook produces.
        #
        # Args:
        #     network: The network whose covered projections are counted.
        #
        # Raises:
        #     MisconfigurationError: If the scope is unresolvable, holds no
        #         Linear module, or covers no weight elements at all.
        #
        # Returns:
        #     Zero-valued elements as a fraction of all covered elements.
        zero_count: int = 0
        element_count: int = 0
        for _, linear_module in self._scope_resolver.select_linear_modules(network):
            weight: torch.Tensor = linear_module.weight.detach()
            zero_count += int((weight == 0.0).sum().item())
            element_count += weight.numel()
        if element_count == 0:
            raise MisconfigurationError("Sparsity measurement found no covered weight elements.")
        return zero_count / element_count

    def covered_parameter_fraction(self, network: nn.Module) -> float:
        # Reports the fraction of network parameters the pruning scope covers.
        # Coverage bounds what the arm's sparsity figure can mean for the whole
        # model: an arm reaching half sparsity over a scope covering four fifths
        # of the parameters is a materially different intervention from the same
        # sparsity over a tenth, so both numbers enter the recipe together. The
        # denominator is every parameter of the network, biases included, while
        # the numerator counts covered weight tensors only.
        #
        # Args:
        #     network: The network whose scope share is measured.
        #
        # Raises:
        #     MisconfigurationError: If the scope is unresolvable, holds no
        #         Linear module, or the network has no parameters.
        #
        # Returns:
        #     Covered weight elements as a fraction of all network parameters.
        covered: int = sum(
            linear_module.weight.numel()
            for _, linear_module in self._scope_resolver.select_linear_modules(network)
        )
        total: int = sum(parameter.numel() for parameter in network.parameters())
        if total == 0:
            raise MisconfigurationError("Coverage measurement found an empty network.")
        return covered / total

    def configuration_dump(self) -> dict[str, object]:
        # Returns the exact transformation configuration for the optimization recipe.
        return {
            "technique": "structured_magnitude_pruning",
            "pruning_amount": self._configuration.pruning_amount,
            "norm_degree": self._configuration.norm_degree,
            "channel_dimension": self._configuration.channel_dimension,
            "target_module_kind": "torch.nn.Linear",
            "target_scope": "generator_backbone"
        }


class UnstructuredPruningConfig(BaseModel):
    # Frozen unstructured-arm settings: pruning amount and the accepted
    # sparsity floor for recovered checkpoints.
    #
    # Fields:
    #     pruning_amount: Fraction of the pooled weight elements zeroed
    #         across the whole network, constrained to the open unit
    #         interval. Because the pool is global rather than per tensor,
    #         individual tensors reach different sparsities and only the
    #         pooled figure equals this amount. Default: ``0.5``.
    #     minimum_accepted_sparsity: Sparsity floor a loaded checkpoint must
    #         reach before the pruned-and-recovered verification accepts it
    #         as a recovered artifact; it governs verification only and
    #         never the arm's own masking. Default: ``0.45``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    pruning_amount: float = Field(default=0.5, gt=0.0, lt=1.0)
    minimum_accepted_sparsity: float = Field(default=0.45, gt=0.0, lt=1.0)


class UnstructuredWeightScopeCollector:
    # Collector locating every prunable weight tensor for the global unstructured arm.
    # Convolution, linear, and recurrent weight matrices join one global magnitude pool,
    # matching the phase-zero magnitude-pruning method at full-fleet rigor.
    #
    # Integration: the collected triples are what torch's global pruning call is
    # given, and the same collection is re-walked for baking, sparsity, and
    # coverage, so all four operations describe one identical tensor set. Biases,
    # normalization parameters, embeddings, and the weight-norm magnitude
    # parameter are outside the scope by construction: they are either too few to
    # matter or, in the magnitude parameter's case, would rescale surviving
    # weights rather than remove any.
    def collect(self, network: nn.Module) -> list[tuple[str, nn.Module, str]]:
        # Returns (module path, module, parameter name) triples across the prunable scope.
        # Weight-normed modules contribute their direction parameter, whose zeros propagate
        # exactly into the effective weight, so masks always hook real nn.Parameter tensors.
        # A recurrent module contributes each of its gate weight matrices under
        # the logical name, with any pruning suffix already stripped, so
        # re-collecting an already-masked network yields the same names it did
        # before masking. Recurrent names are sorted for a deterministic pool
        # order across processes.
        #
        # Args:
        #     network: The generator network walked for prunable tensors.
        #
        # Raises:
        #     MisconfigurationError: If the walk finds no prunable weight
        #         tensor, which would otherwise make the global arm a silent
        #         no-op.
        #
        # Returns:
        #     The (module path, owning module, parameter name) triples the
        #     global magnitude pool is formed from. For a weight-normed module
        #     the owning module is its parametrization list and the parameter
        #     name is the direction parameter, not the module itself.
        collected: list[tuple[str, nn.Module, str]] = []
        for module_path, candidate_module in network.named_modules():
            if isinstance(candidate_module, nn.Conv1d | nn.Conv2d | nn.ConvTranspose1d | nn.Linear):
                # A weight-normed module has no plain weight parameter to hook,
                # so the direction parameter inside its parametrization list is
                # collected instead.
                if parametrize.is_parametrized(candidate_module, "weight"):
                    parametrization_list: nn.Module = candidate_module.parametrizations["weight"]
                    collected.append(
                        (f"{module_path}.parametrizations.weight", parametrization_list, "original1")
                    )
                    continue
                collected.append((module_path, candidate_module, "weight"))
                continue
            # A recurrent module holds several gate weight matrices as direct
            # parameters; each is collected under its logical name so a
            # re-collection after masking yields the same pool.
            if isinstance(candidate_module, nn.GRU):
                gru_weight_names: set[str] = set()
                for parameter_name, _ in candidate_module.named_parameters(recurse=False):
                    if not parameter_name.startswith("weight"):
                        continue
                    logical_name: str = parameter_name.removesuffix("_orig")
                    gru_weight_names.add(logical_name)
                for logical_name in sorted(gru_weight_names):
                    collected.append((module_path, candidate_module, logical_name))
        if not collected:
            raise MisconfigurationError(
                "The global unstructured scope found no prunable weight tensors in this network."
            )
        return collected


class UnstructuredMagnitudePruning:
    # Global unstructured magnitude pruning across every prunable weight tensor.
    # A single L1 magnitude pool zeroes the smallest weights network-wide; baking makes
    # the zeros literal so the recovered checkpoint deploys without pruning hooks.
    #
    # Scope: the arm covers every convolutional, linear, and recurrent weight
    # tensor the collector finds anywhere in the network, with no backbone
    # restriction, and it is the arm that applies to convolution-dominant
    # generators where no Linear stratum worth pruning exists. Pruning is
    # unstructured, meaning individual weights are zeroed wherever they fall, so
    # no channel, filter, or gate is removed as a unit and the resulting tensors
    # keep their original shapes.
    #
    # Pooling semantics: all covered tensors form one magnitude distribution and
    # the globally smallest elements are zeroed until the registered fraction of
    # the pool is reached. Sparsity is therefore a property of the pool and not
    # of any single tensor: a tensor whose weights are large relative to the rest
    # of the network may lose almost nothing while another loses most of its
    # entries, and only the pooled figure equals the registered amount. Baking
    # carries the same semantics as on the structured arm, replacing every masked
    # reparametrization with literal zeros in ordinary parameters so the artifact
    # deploys with no pruning dependency.
    def __init__(self, configuration: UnstructuredPruningConfig) -> None:
        # Binds the arm settings and the global weight-scope collector.
        self._configuration: UnstructuredPruningConfig = configuration
        self._scope_collector: UnstructuredWeightScopeCollector = UnstructuredWeightScopeCollector()

    @property
    def configuration(self) -> UnstructuredPruningConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    def apply_masks(self, network: nn.Module) -> list[str]:
        # Applies one global magnitude mask pool and returns the pruned parameter paths.
        # The whole collected scope enters a single ranking, which is what makes
        # this arm global rather than a per-tensor sweep. The masks remain live
        # reparametrizations after this call; the network holds literal zeros
        # only once bake_masks has run.
        #
        # Args:
        #     network: The generator network whose collected weight tensors are
        #         masked in place.
        #
        # Raises:
        #     MisconfigurationError: If the walk finds no prunable weight
        #         tensor in this network.
        #
        # Returns:
        #     The dotted paths of every pooled parameter, which the recovery
        #     recipe records as its mask inventory.
        scope: list[tuple[str, nn.Module, str]] = self._scope_collector.collect(network)
        prune.global_unstructured(
            [(scoped_module, parameter_name) for _, scoped_module, parameter_name in scope],
            pruning_method=prune.L1Unstructured,
            amount=self._configuration.pruning_amount
        )
        return [f"{module_path}.{parameter_name}" for module_path, _, parameter_name in scope]

    def bake_masks(self, network: nn.Module) -> None:
        # Bakes the masks into literal zero weights and removes the pruning hooks.
        # For a weight-normed module the baked zeros land in the direction
        # parameter, from which they propagate exactly into the effective weight,
        # so the deployed artifact is sparse in the tensor the convolution
        # actually uses.
        #
        # Args:
        #     network: The masked network whose pooled tensors are baked in
        #         place.
        for _, scoped_module, parameter_name in self._scope_collector.collect(network):
            prune.remove(scoped_module, name=parameter_name)

    def measured_sparsity(self, network: nn.Module) -> float:
        # Measures the achieved zero fraction over the covered weight tensors.
        # The figure is pooled exactly as the masking was, so it is comparable
        # with the registered amount; per-tensor sparsities are deliberately not
        # reported because they are not what the arm controls.
        #
        # Args:
        #     network: The network whose pooled tensors are counted.
        #
        # Raises:
        #     MisconfigurationError: If the walk finds no prunable weight
        #         tensor, or the pool covers no elements.
        #
        # Returns:
        #     Zero-valued elements as a fraction of all pooled elements.
        zero_count: int = 0
        element_count: int = 0
        for _, scoped_module, parameter_name in self._scope_collector.collect(network):
            weight: torch.Tensor = getattr(scoped_module, parameter_name).detach()
            zero_count += int((weight == 0.0).sum().item())
            element_count += weight.numel()
        if element_count == 0:
            raise MisconfigurationError("Sparsity measurement found no covered weight elements.")
        return zero_count / element_count

    def covered_parameter_fraction(self, network: nn.Module) -> float:
        # Reports the fraction of network parameters the global scope covers.
        # The global arm reaches a much larger share than the structured arm
        # does, which is why the two arms' sparsity figures are never compared
        # without their coverage figures beside them.
        #
        # Args:
        #     network: The network whose scope share is measured.
        #
        # Raises:
        #     MisconfigurationError: If the walk finds no prunable weight
        #         tensor, or the network has no parameters.
        #
        # Returns:
        #     Pooled weight elements as a fraction of all network parameters.
        covered: int = sum(
            getattr(scoped_module, parameter_name).numel()
            for _, scoped_module, parameter_name in self._scope_collector.collect(network)
        )
        total: int = sum(parameter.numel() for parameter in network.parameters())
        if total == 0:
            raise MisconfigurationError("Coverage measurement found an empty network.")
        return covered / total

    def configuration_dump(self) -> dict[str, object]:
        # Returns the exact transformation configuration for the optimization recipe.
        return {
            "technique": "unstructured_magnitude_pruning",
            "pruning_amount": self._configuration.pruning_amount,
            "target_module_kinds": "conv_linear_gru_weights",
            "target_scope": "global_network"
        }


class MaskedMagnitudePruning(OptimizationTechnique):
    # Evaluation-time masked magnitude pruning at one pre-registered sparsity level.
    # The arm's masks are applied to the loaded baseline and baked immediately, so the
    # evaluated object is the masked network itself; this measures the robustness curve
    # without recovery and claims no deployment acceleration from dense zeros.
    def __init__(
        self,
        pruning_arm: PruningArmName,
        pruning_amount: float,
        produced_variant_name: OptimizationVariantName
    ) -> None:
        # Binds the resolved arm, the registered sparsity level, and the
        # produced variant name, and prepares both arm implementations. Both are
        # constructed because construction is free and the arm selection happens
        # per call site; only the resolved one ever touches a network.
        #
        # Args:
        #     pruning_arm: The arm the registry resolved for this
        #         architecture, which selects between the backbone-Linear
        #         scope and the global pooled scope.
        #     pruning_amount: The registered curve level, applied as the
        #         per-tensor channel fraction on the structured arm and as
        #         the pooled fraction on the global arm.
        #     produced_variant_name: The curve point this technique produces,
        #         which is the identity the capsule and CSV row carry.
        self._pruning_arm: PruningArmName = pruning_arm
        self._pruning_amount: float = pruning_amount
        self._produced_variant_name: OptimizationVariantName = produced_variant_name
        self._structured_pruning: StructuredMagnitudePruning = StructuredMagnitudePruning(
            StructuredPruningConfig(pruning_amount=pruning_amount)
        )
        self._unstructured_pruning: UnstructuredMagnitudePruning = UnstructuredMagnitudePruning(
            UnstructuredPruningConfig(pruning_amount=pruning_amount)
        )
        self._measured_sparsity: float | None = None
        self._measured_coverage: float | None = None

    @property
    def name(self) -> OptimizationVariantName:
        # Returns the canonical variant name this technique produces.
        return self._produced_variant_name

    def apply(self, module: Module) -> Module:
        # Applies and immediately bakes the resolved arm's masks, then
        # validates that the achieved sparsity reaches the registered level
        # within tolerance; falling short is a configuration error, not a
        # smaller experiment. Baking happens inside apply because this lane
        # performs no recovery training, so the evaluated object is the masked
        # network itself and needs no live mask to hold its zeros.
        #
        # Args:
        #     module: The harness Module carrying the restored baseline
        #         weights, whose network is masked in place.
        #
        # Raises:
        #     MisconfigurationError: If the resolved arm cannot resolve its
        #         scope on this network, or if the achieved sparsity falls
        #         more than five hundredths below the registered level, which
        #         happens when the channel granularity is too coarse to admit
        #         the fraction.
        #
        # Returns:
        #     The same module, now carrying the masked and baked network.
        network: nn.Module = getattr(module, "network")
        match self._pruning_arm:
            case "linear_structured":
                self._structured_pruning.apply_masks(network)
                self._structured_pruning.bake_masks(network)
                self._measured_sparsity: float | None = self._structured_pruning.measured_sparsity(network)
                self._measured_coverage: float | None = self._structured_pruning.covered_parameter_fraction(network)
            case "global_unstructured":
                self._unstructured_pruning.apply_masks(network)
                self._unstructured_pruning.bake_masks(network)
                self._measured_sparsity: float | None = self._unstructured_pruning.measured_sparsity(network)
                self._measured_coverage: float | None = self._unstructured_pruning.covered_parameter_fraction(network)
        # Both measurements are taken after baking, so they describe the literal
        # zeros of the object about to be evaluated rather than the mask that
        # produced them.
        if self._measured_sparsity is None or self._measured_sparsity < self._pruning_amount - 0.05:
            raise MisconfigurationError(
                f"Masked pruning reached sparsity {self._measured_sparsity} below the "
                f"registered level {self._pruning_amount:.2f} for the {self._pruning_arm} arm."
            )
        return module

    def configuration_dump(self) -> dict[str, object]:
        # Returns the exact transformation configuration for the optimization recipe.
        return {
            "technique": "masked_magnitude_pruning",
            "pruning_arm": self._pruning_arm,
            "pruning_amount": self._pruning_amount,
            "measured_sparsity": self._measured_sparsity,
            "covered_parameter_fraction": self._measured_coverage
        }


class DenseContinuedIdentity(OptimizationTechnique):
    # Evaluation-time verification technique for the matched dense-continuation control.
    # The continued checkpoint already carries its transformation, so evaluation applies
    # nothing and instead proves the loaded weights stayed dense, which is what makes
    # this arm the causal control for the recovery budget.
    #
    # Executed status: the registered arm is the control the recovery design
    # calls for, and the reported study admitted no dense-continuation group.
    # Recovery rows in the reported evidence therefore support descriptive
    # rather than causal claims, and this technique's presence in the registry
    # states the intended control rather than an executed one.
    def __init__(self, pruning_arm: PruningArmName) -> None:
        # Binds the resolved arm and prepares both arm implementations for
        # scope-correct sparsity measurement.
        #
        # Args:
        #     pruning_arm: The arm the registry resolved for this
        #         architecture, which fixes the scope the density claim is
        #         measured over; the control must be dense over exactly the
        #         scope the treatment would have masked.
        self._pruning_arm: PruningArmName = pruning_arm
        self._structured_pruning: StructuredMagnitudePruning = StructuredMagnitudePruning(StructuredPruningConfig())
        self._unstructured_pruning: UnstructuredMagnitudePruning = UnstructuredMagnitudePruning(UnstructuredPruningConfig())
        self._measured_sparsity: float | None = None

    @property
    def name(self) -> OptimizationVariantName:
        # Returns the canonical variant name this technique produces.
        return "dense_continued"

    def apply(self, module: Module) -> Module:
        # Transforms nothing; proves the loaded continuation checkpoint
        # stayed dense over the arm's scope, which is what qualifies it as
        # the causal control. Density is measured over the same scope the
        # pruned arm would have masked, so the control and the treatment are
        # compared over identical tensors.
        #
        # Args:
        #     module: The harness Module carrying the loaded continuation
        #         checkpoint.
        #
        # Raises:
        #     MisconfigurationError: If measured sparsity exceeds five
        #         hundredths, which means the loaded artifact is not a dense
        #         continuation and cannot serve as the causal control.
        #
        # Returns:
        #     The same module, untransformed.
        network: nn.Module = getattr(module, "network")
        measured: float = self._measure_arm_sparsity(network)
        if measured > 0.05:
            raise MisconfigurationError(
                f"Dense-continuation checkpoint reports sparsity {measured:.4f} over the "
                f"{self._pruning_arm} scope; the causal control must stay dense."
            )
        self._measured_sparsity: float | None = measured
        return module

    def configuration_dump(self) -> dict[str, object]:
        # Returns the exact transformation configuration for the optimization recipe.
        return {
            "technique": "dense_continued_identity",
            "pruning_arm": self._pruning_arm,
            "measured_sparsity": self._measured_sparsity
        }

    def _measure_arm_sparsity(self, network: nn.Module) -> float:
        # Measures sparsity over the scope the resolved pruning arm actually covers.
        match self._pruning_arm:
            case "linear_structured":
                return self._structured_pruning.measured_sparsity(network)
            case "global_unstructured":
                return self._unstructured_pruning.measured_sparsity(network)


class PrunedRecoveredVerification(OptimizationTechnique):
    # Evaluation-time verification technique for the pruned-and-recovered variants.
    # The recovered checkpoint already carries its transformation, so evaluation applies
    # nothing and instead proves the loaded weights are as sparse as the variant name
    # claims, measured over the architecture-resolved pruning arm's scope.
    def __init__(
        self,
        pruning_arm: PruningArmName,
        produced_variant_name: OptimizationVariantName = "pruned_50_recovered"
    ) -> None:
        # Binds the resolved arm and produced variant name, and prepares
        # both arm implementations for scope-correct sparsity measurement.
        # The arms are constructed with their default settings because only
        # their scope resolution and their accepted sparsity floor are used
        # here; no masking configuration of theirs is exercised.
        #
        # Args:
        #     pruning_arm: The arm the registry resolved for this
        #         architecture, which fixes the scope sparsity is measured
        #         over and the floor it is compared against.
        #     produced_variant_name: Which recovered arm this verification
        #         represents, distinguishing the full-budget row from the
        #         half-budget ablation. Default: ``"pruned_50_recovered"``.
        self._pruning_arm: PruningArmName = pruning_arm
        self._produced_variant_name: OptimizationVariantName = produced_variant_name
        self._structured_pruning: StructuredMagnitudePruning = StructuredMagnitudePruning(StructuredPruningConfig())
        self._unstructured_pruning: UnstructuredMagnitudePruning = UnstructuredMagnitudePruning(UnstructuredPruningConfig())
        self._measured_sparsity: float | None = None

    @property
    def name(self) -> OptimizationVariantName:
        # Returns the canonical variant name this technique produces.
        return self._produced_variant_name

    def apply(self, module: Module) -> Module:
        # Transforms nothing; proves the loaded checkpoint is as sparse as
        # the variant name claims over the arm's scope, rejecting artifacts
        # below the accepted floor. The transformation this variant names
        # happened during recovery training, and the checkpoint written there
        # was baked, so the zeros are already literal by the time they are
        # counted here.
        #
        # Args:
        #     module: The harness Module carrying the loaded recovered
        #         checkpoint.
        #
        # Raises:
        #     MisconfigurationError: If measured sparsity is below the arm's
        #         accepted floor, which means the loaded artifact is not the
        #         pruned-and-recovered checkpoint the variant name claims.
        #
        # Returns:
        #     The same module, untransformed.
        network: nn.Module = getattr(module, "network")
        measured: float = self._measure_arm_sparsity(network)
        minimum: float = self._arm_minimum_sparsity()
        if measured < minimum:
            raise MisconfigurationError(
                f"Loaded checkpoint sparsity {measured:.4f} is below the accepted minimum "
                f"{minimum:.4f} for the {self._pruning_arm} arm; this checkpoint is not a "
                f"{self._produced_variant_name} artifact."
            )
        self._measured_sparsity: float | None = measured
        return module

    def configuration_dump(self) -> dict[str, object]:
        # Returns the exact transformation configuration for the optimization recipe.
        return {
            "technique": "pruned_recovered_verification",
            "produced_variant_name": self._produced_variant_name,
            "pruning_arm": self._pruning_arm,
            "measured_sparsity": self._measured_sparsity,
            "minimum_accepted_sparsity": self._arm_minimum_sparsity(),
            "pruning_configuration": self._arm_configuration_dump()
        }

    def _measure_arm_sparsity(self, network: nn.Module) -> float:
        # Measures sparsity over the scope the resolved pruning arm actually covers.
        match self._pruning_arm:
            case "linear_structured":
                return self._structured_pruning.measured_sparsity(network)
            case "global_unstructured":
                return self._unstructured_pruning.measured_sparsity(network)

    def _arm_minimum_sparsity(self) -> float:
        # Returns the accepted sparsity floor declared by the resolved arm's configuration.
        match self._pruning_arm:
            case "linear_structured":
                return self._structured_pruning.configuration.minimum_accepted_sparsity
            case "global_unstructured":
                return self._unstructured_pruning.configuration.minimum_accepted_sparsity

    def _arm_configuration_dump(self) -> dict[str, object]:
        # Returns the resolved arm's transformation configuration for the recipe.
        match self._pruning_arm:
            case "linear_structured":
                return self._structured_pruning.configuration_dump()
            case "global_unstructured":
                return self._unstructured_pruning.configuration_dump()

