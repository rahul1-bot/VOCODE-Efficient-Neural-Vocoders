# This module:
# 1. Defines the closed Study 2 variant vocabulary (nineteen names spanning
#    baselines, compilation, quantization, storage precision, ONNX
#    deployment, the pruning curve with recovery controls, and ODE sampling
#    reduction) and the per-architecture technique-arm resolution table
# 2. Decides applicability for every (architecture, variant) cell with a
#    recorded reason, resolves the registered hardware lane per variant, and
#    constructs the technique instance for supported cells
# 3. Defines the OptimizationTechnique base contract and the BaselineIdentity
#    null technique that measures denominators under the identical protocol
#
# Design decisions:
# - Unsupported cells fail closed with a recorded scientific reason rather
#   than being silently skipped, so the study's coverage gaps are evidence,
#   not accidents
# - Applicability derives from a declarative per-architecture table
#   (operator profile, measured screening outcomes, and registered scope),
#   keeping every decision reviewable in one place
# - Measurement lanes are fixed per variant for denominator continuity;
#   the hardware-lane validation in the evaluator enforces them
# - Technique modules are imported inside the builder so importing the
#   registry never drags optional backend dependencies into processes that
#   only need applicability decisions
#
# Author: Rahul Sawhney

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from syntheticmind.core.module import Module
from syntheticmind.utilities.exceptions import MisconfigurationError

from vocode.configs.layout import HardwareName
from vocode.models.vocoder import ArchitectureName

__all__: list[str] = [
    "ArchitectureTechniqueResolution",
    "BaselineIdentity",
    "OptimizationTechnique",
    "OptimizationVariantName",
    "OptimizationVariantRecord",
    "OptimizationVariantRegistry",
    "OptimizationVariantSupport",
    "PruningArmName",
    "QuantizedModuleKind",
    "TechniqueResolutionTable"
]

type OptimizationVariantName = Literal[
    "baseline_cpu",
    "baseline_b200",
    "torch_compile",
    "torch_compile_overhead",
    "int8_dynamic",
    "int8_weight_only",
    "int4_weight_only",
    "fp16_weights",
    "onnx_fp32",
    "onnx_int8_static",
    "pruned_30",
    "pruned_50",
    "pruned_70",
    "pruned_50_recovered",
    "pruned_50_recovered_half",
    "dense_continued",
    "ode_steps_8",
    "ode_steps_4",
    "ode_steps_2"
]
type PruningArmName = Literal["linear_structured", "global_unstructured"]
type QuantizedModuleKind = Literal["linear", "gru"]


class ArchitectureTechniqueResolution(BaseModel):
    # Frozen per-architecture technique-arm declaration: the pruning arm,
    # quantized module kinds, per-family support flags, and whether the
    # architecture carries the causal recovery controls. One row exists per
    # architecture that enters Study 2; the row is the single place where a
    # technique family's applicability to that architecture is decided, and
    # every applicability decision the registry publishes is derived from it.
    #
    # Fields:
    #     architecture_name: The architecture this row declares technique
    #         arms for; rows are unique per architecture.
    #     pruning_arm: The magnitude-pruning arm the architecture takes.
    #         Linear-bearing backbones take ``"linear_structured"`` and
    #         convolution-dominant generators take
    #         ``"global_unstructured"``.
    #     int8_module_kinds: The torch module kinds dynamic INT8
    #         quantization covers on this architecture; the recurrent
    #         reference deployment adds ``"gru"`` to the Linear kind.
    #     int8_dynamic_supported: Whether the dynamic INT8 cell is
    #         registered; screening measured zero executable coverage on the
    #         convolution-dominant generators, which record it as False.
    #     weight_only_supported: Whether the architecture's operator profile
    #         grants the Linear stratum meaningful weight-only coverage.
    #     fp16_supported: Whether the half-precision storage lane executes;
    #         a measured dtype incompatibility records it as False.
    #     compile_supported: Whether module-level compilation covers the
    #         executed synthesis path; an architecture whose sampler drives
    #         submodules directly records it as False.
    #     onnx_export_supported: Whether the two ONNX deployment lanes are
    #         registered for this architecture.
    #     ode_step_reduction_supported: Whether the architecture integrates
    #         an ordinary differential equation sampling schedule that a
    #         step-count intervention can reduce.
    #     causal_control_selected: Whether the architecture carries the
    #         dense-continuation and half-budget recovery controls, which
    #         are registered for two selected architectures only.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    architecture_name: ArchitectureName
    pruning_arm: PruningArmName
    int8_module_kinds: tuple[QuantizedModuleKind, ...]
    int8_dynamic_supported: bool
    weight_only_supported: bool
    fp16_supported: bool
    compile_supported: bool
    onnx_export_supported: bool
    ode_step_reduction_supported: bool
    causal_control_selected: bool


class OptimizationVariantSupport(BaseModel):
    # Frozen applicability decision for one (architecture, variant) cell,
    # carrying the recorded reason in both the supported and unsupported
    # outcomes. A decision record is produced for every cell of the matrix,
    # so a coverage gap in the study is an explicit refusal with a stated
    # scientific cause rather than a missing row.
    #
    # Fields:
    #     architecture_name: The architecture of the decided cell; it
    #         echoes the architecture the decision was requested for.
    #     variant_name: The variant of the decided cell; it echoes the
    #         variant the decision was requested for.
    #     supported: Whether the cell may be constructed and measured.
    #     reason: The recorded cause of the decision. Supported cells carry
    #         the marker ``"registered"``; unsupported cells carry the
    #         scientific reason the cell is out of scope, which the registry
    #         raises verbatim when construction of that cell is attempted.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    architecture_name: ArchitectureName
    variant_name: OptimizationVariantName
    supported: bool
    reason: str


class TechniqueResolutionTable:
    # Declarative table resolving which technique arms apply to each architecture.
    # Linear-structured pruning, weight-only quantization, and Linear-kind dynamic INT8
    # require Linear-bearing backbones; convolution-dominant generators take the global
    # unstructured arm, and LPCNet adds GRU-kind quantization because its reference
    # deployment is an eight-bit GRU. Support flags encode measured screening evidence
    # and the registered study scope so unsupported cells fail closed with recorded reasons.
    #
    # Integration: the twelve rows are the study's declarative applicability
    # source. Every architecture that enters Study 2 holds exactly one row, and
    # HiFTNet holds none because it is the documented non-executed exclusion.
    # Extending the study to a new architecture means adding one row here, which
    # is what keeps every applicability decision reviewable in a single place;
    # the registry never infers arms or support from a model object.
    resolutions: ClassVar[tuple[ArchitectureTechniqueResolution, ...]] = (
        ArchitectureTechniqueResolution(architecture_name="vocos", pruning_arm="linear_structured", int8_module_kinds=("linear",), int8_dynamic_supported=True, weight_only_supported=True, fp16_supported=True, compile_supported=True, onnx_export_supported=False, ode_step_reduction_supported=False, causal_control_selected=True),
        ArchitectureTechniqueResolution(architecture_name="freev", pruning_arm="linear_structured", int8_module_kinds=("linear",), int8_dynamic_supported=True, weight_only_supported=True, fp16_supported=False, compile_supported=True, onnx_export_supported=False, ode_step_reduction_supported=False, causal_control_selected=False),
        ArchitectureTechniqueResolution(architecture_name="vocosformer", pruning_arm="linear_structured", int8_module_kinds=("linear",), int8_dynamic_supported=True, weight_only_supported=True, fp16_supported=True, compile_supported=True, onnx_export_supported=False, ode_step_reduction_supported=False, causal_control_selected=False),
        ArchitectureTechniqueResolution(architecture_name="rfwave", pruning_arm="linear_structured", int8_module_kinds=("linear",), int8_dynamic_supported=True, weight_only_supported=True, fp16_supported=True, compile_supported=True, onnx_export_supported=False, ode_step_reduction_supported=True, causal_control_selected=True),
        ArchitectureTechniqueResolution(architecture_name="hifigan_v1", pruning_arm="global_unstructured", int8_module_kinds=("linear",), int8_dynamic_supported=False, weight_only_supported=False, fp16_supported=True, compile_supported=True, onnx_export_supported=True, ode_step_reduction_supported=False, causal_control_selected=False),
        ArchitectureTechniqueResolution(architecture_name="hifigan_v2", pruning_arm="global_unstructured", int8_module_kinds=("linear",), int8_dynamic_supported=False, weight_only_supported=False, fp16_supported=True, compile_supported=True, onnx_export_supported=True, ode_step_reduction_supported=False, causal_control_selected=False),
        ArchitectureTechniqueResolution(architecture_name="hifigan_v3", pruning_arm="global_unstructured", int8_module_kinds=("linear",), int8_dynamic_supported=False, weight_only_supported=False, fp16_supported=True, compile_supported=True, onnx_export_supported=True, ode_step_reduction_supported=False, causal_control_selected=False),
        ArchitectureTechniqueResolution(architecture_name="melgan", pruning_arm="global_unstructured", int8_module_kinds=("linear",), int8_dynamic_supported=False, weight_only_supported=False, fp16_supported=True, compile_supported=True, onnx_export_supported=True, ode_step_reduction_supported=False, causal_control_selected=False),
        ArchitectureTechniqueResolution(architecture_name="bigvgan", pruning_arm="global_unstructured", int8_module_kinds=("linear",), int8_dynamic_supported=False, weight_only_supported=False, fp16_supported=True, compile_supported=True, onnx_export_supported=False, ode_step_reduction_supported=False, causal_control_selected=False),
        ArchitectureTechniqueResolution(architecture_name="apnet2", pruning_arm="global_unstructured", int8_module_kinds=("linear",), int8_dynamic_supported=True, weight_only_supported=False, fp16_supported=True, compile_supported=True, onnx_export_supported=False, ode_step_reduction_supported=False, causal_control_selected=False),
        ArchitectureTechniqueResolution(architecture_name="rndvoc", pruning_arm="global_unstructured", int8_module_kinds=("linear",), int8_dynamic_supported=False, weight_only_supported=False, fp16_supported=True, compile_supported=True, onnx_export_supported=False, ode_step_reduction_supported=False, causal_control_selected=False),
        ArchitectureTechniqueResolution(architecture_name="lpcnet", pruning_arm="global_unstructured", int8_module_kinds=("linear", "gru"), int8_dynamic_supported=True, weight_only_supported=False, fp16_supported=False, compile_supported=False, onnx_export_supported=False, ode_step_reduction_supported=False, causal_control_selected=False)
    )

    def get(self, architecture_name: ArchitectureName) -> ArchitectureTechniqueResolution:
        # Resolves the technique arms declared for one architecture. Lookup is a
        # linear scan of the declared rows; an architecture without a row is a
        # configuration error rather than a defaulted resolution, because
        # silently defaulting would card an architecture into cells whose
        # applicability was never reviewed.
        #
        # Args:
        #     architecture_name: The architecture whose declared technique
        #         arms are requested.
        #
        # Raises:
        #     MisconfigurationError: If no row declares this architecture,
        #         which is the state of the documented exclusion and of any
        #         architecture added to the fleet without a reviewed row.
        #
        # Returns:
        #     The single declared resolution row for this architecture.
        matching: tuple[ArchitectureTechniqueResolution, ...] = tuple(
            resolution for resolution in self.resolutions
            if resolution.architecture_name == architecture_name
        )
        if not matching:
            raise MisconfigurationError(
                f"No technique resolution declared for architecture {architecture_name}; "
                f"extend TechniqueResolutionTable deliberately before carding it."
            )
        return matching[0]


class OptimizationTechnique:
    # Base contract for one model transformation inside the Study 2
    # intervention lane. A technique maps a trained harness Module onto a
    # harness Module and reports its exact configuration for the
    # optimization recipe; it never touches the reproduction code path.
    # The three members are unimplemented here and every one of them raises,
    # so a subclass that forgets a member fails at its first use instead of
    # producing an unnamed or unrecorded intervention.
    #
    # Integration: this is the subclassing surface every registered technique
    # implements, and the Module-to-Module signature of apply is what keeps the
    # optimized-variant evaluator technique-agnostic: the evaluator loads a
    # checkpoint, calls apply once, and measures whatever module comes back
    # through the unchanged harness prediction and test loops. A subclass
    # therefore transforms the module in place or returns a replacement, but
    # never changes the type the harness receives. Techniques that need run-level
    # context beyond the module (the ONNX deployment lane needs the run capsule's
    # directories) declare an additional bind method that the evaluator calls
    # before apply.
    #
    # Example::
    #
    #     class HalfPrecisionExample(OptimizationTechnique):
    #         @property
    #         def name(self) -> OptimizationVariantName:
    #             return "fp16_weights"
    #
    #         def apply(self, module: Module) -> Module:
    #             module.network = module.network.half()
    #             return module
    #
    #         def configuration_dump(self) -> dict[str, object]:
    #             return {"technique": "example", "storage_dtype": "float16"}
    @property
    def name(self) -> OptimizationVariantName:
        # Returns the canonical variant name this technique produces. The name
        # is the identity the capsule, the CSV row, and the recipe are written
        # under, so a technique constructed for two lanes reports the lane it
        # was constructed for rather than a family name.
        #
        # Raises:
        #     NotImplementedError: Always, on the unimplemented base contract.
        raise NotImplementedError("Subclasses must implement the name property.")

    def apply(self, module: Module) -> Module:
        # Applies the transformation to the trained module and returns it. The
        # module arrives with the base checkpoint already restored, so a
        # technique operates on trained weights; verification techniques
        # transform nothing and instead prove a property of those weights.
        #
        # Args:
        #     module: The harness Module carrying the restored baseline
        #         weights, whose network attribute is the transformation
        #         target.
        #
        # Raises:
        #     NotImplementedError: Always, on the unimplemented base contract.
        #
        # Returns:
        #     The harness Module the evaluator measures, which is the same
        #     object for in-place transformations.
        raise NotImplementedError("Subclasses must implement apply.")

    def configuration_dump(self) -> dict[str, object]:
        # Returns the exact transformation configuration for the optimization recipe.
        # The dump is read after apply and must therefore carry both the
        # settings the technique was constructed with and whatever the
        # transformation measured or resolved at apply time; before apply those
        # measured entries read as None, which is how the recipe states that
        # nothing has been claimed yet.
        #
        # Raises:
        #     NotImplementedError: Always, on the unimplemented base contract.
        raise NotImplementedError("Subclasses must implement configuration_dump.")


class BaselineIdentity(OptimizationTechnique):
    # Null technique producing the unmodified baseline row on a measurement lane.
    # It exists so every speed claim has its denominator measured under the identical
    # capsule protocol and the identical software stack as the compared intervention.
    def __init__(self, produced_variant_name: OptimizationVariantName = "baseline_cpu") -> None:
        # Binds which baseline lane name this identity row produces. The two
        # baseline lanes are distinct denominators measured on different
        # hardware, so the lane is part of the technique's identity rather than
        # a property of the run that uses it.
        #
        # Args:
        #     produced_variant_name: The baseline lane this identity row
        #         measures. Default: ``"baseline_cpu"``.
        self._produced_variant_name: OptimizationVariantName = produced_variant_name

    @property
    def name(self) -> OptimizationVariantName:
        # Returns the canonical variant name this technique produces.
        return self._produced_variant_name

    def apply(self, module: Module) -> Module:
        # Returns the module unchanged; the baseline is the identity
        # transformation by definition.
        return module

    def configuration_dump(self) -> dict[str, object]:
        # Returns the exact transformation configuration for the optimization recipe.
        return {
            "technique": "baseline_identity",
            "produced_variant_name": self._produced_variant_name
        }


class OptimizationVariantRecord(BaseModel):
    # Frozen binding of one supported cell: the variant name, its base
    # architecture, the constructed technique, and the lane-interpretation
    # note recorded into the run capsule. Freezing the record is what
    # guarantees that the technique the evaluator writes into its recipe is
    # the technique it applied, because the binding cannot be substituted
    # after construction.
    #
    # Fields:
    #     variant_name: The cell's variant identity, which the capsule, the
    #         experiment CSV row, and both recipe scopes are written under.
    #     base_architecture: The architecture whose trained checkpoint the
    #         technique transforms.
    #     technique: The constructed OptimizationTechnique instance carrying
    #         this cell's architecture-resolved configuration; arbitrary
    #         types are permitted on this model so the live technique object
    #         can be bound directly.
    #     measurement_note: The lane-interpretation sentence recorded into
    #         the run capsule, stating how the resulting numbers must be
    #         read (which denominator applies, and whether a speedup is
    #         expected at all).
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True
    )
    variant_name: OptimizationVariantName
    base_architecture: ArchitectureName
    technique: OptimizationTechnique
    measurement_note: str


class OptimizationVariantRegistry:
    # Registry resolving (base architecture, variant name) to a validated variant record.
    # Any training-level registry model qualifies as a baseline; the per-architecture
    # technique arms, applicability decisions, and hardware lanes come from the
    # declarative resolution table so unsupported cells fail before construction.
    # The registry is stateless: it holds no cache and no cross-call state, so
    # every caller may construct its own instance and two instances always
    # decide a cell identically.
    #
    # Integration: this is the single entry point every Study 2 consumer uses,
    # and it enforces the recorded-unsupported-reason contract. Applicability is
    # published through variant_support for every cell of the matrix, including
    # cells that will never be measured, so the study's coverage gaps are
    # evidence rather than accidents. Construction through get is admitted only
    # for a supported cell; an unsupported cell raises with the recorded reason
    # carried verbatim into the failure message, which means a caller can never
    # obtain a technique for a cell whose exclusion has no stated scientific
    # cause. supported_hardware fixes the measurement lane per variant, and the
    # optimized-variant evaluator validates the run's declared lane against it
    # before measuring, because the study's ratios are only meaningful when the
    # numerator and the denominator were measured on the same lane.
    #
    # Relation to the reported admission contract: a supported cell here is a
    # cell eligible for execution, not an admitted group. Admission is decided
    # downstream on the executed evidence and additionally requires an
    # executable transformed path, nonzero operator or parameter coverage, the
    # complete evaluation partition under every execution, and a same-profile
    # pre-transformation measurement. This registry supplies the first of those
    # conditions by refusing unsupported cells before execution expansion, and
    # an attempt that fails a later condition becomes an exclusion record rather
    # than a result row.
    #
    # Example::
    #
    #     registry: OptimizationVariantRegistry = OptimizationVariantRegistry()
    #     support: OptimizationVariantSupport = registry.variant_support(
    #         "hifigan_v1",
    #         "int8_dynamic"
    #     )
    #     if support.supported:
    #         record: OptimizationVariantRecord = registry.get(
    #             "hifigan_v1",
    #             "int8_dynamic"
    #         )
    #     else:
    #         # support.reason states why the cell is out of the study's scope.
    #         ...
    def get(
        self,
        architecture_name: ArchitectureName,
        variant_name: OptimizationVariantName
    ) -> OptimizationVariantRecord:
        # Resolves a validated variant record with architecture-resolved technique arms.
        # The applicability decision is consulted first, so construction of an
        # unsupported cell is refused before any technique module is imported or
        # any technique object exists.
        #
        # Args:
        #     architecture_name: The architecture whose trained checkpoint the
        #         variant transforms.
        #     variant_name: The registered Study 2 variant to construct.
        #
        # Raises:
        #     MisconfigurationError: If the cell is unsupported, carrying the
        #         recorded reason verbatim; or if the architecture has no
        #         declared row in the resolution table.
        #
        # Returns:
        #     The frozen record binding the constructed technique, the cell's
        #     identity, and the lane-interpretation note.
        support: OptimizationVariantSupport = self.variant_support(architecture_name, variant_name)
        if not support.supported:
            raise MisconfigurationError(
                f"Variant {variant_name} is not supported for {architecture_name}: "
                f"{support.reason}"
            )
        resolution: ArchitectureTechniqueResolution = TechniqueResolutionTable().get(architecture_name)
        technique: OptimizationTechnique = self._build_technique(variant_name, resolution)
        return OptimizationVariantRecord(
            variant_name=variant_name,
            base_architecture=architecture_name,
            technique=technique,
            measurement_note=self._measurement_note(variant_name)
        )

    def variant_support(
        self,
        architecture_name: ArchitectureName,
        variant_name: OptimizationVariantName
    ) -> OptimizationVariantSupport:
        # Resolves the applicability decision and its recorded reason for
        # one cell. HiFTNet is rejected before table resolution because it
        # is the documented non-executed exclusion of the study contract;
        # every other decision derives from the architecture's declared
        # technique arms. The match is exhaustive over the closed variant
        # vocabulary, so extending that vocabulary without extending this
        # decision is a type error rather than a silently undecided cell.
        #
        # Args:
        #     architecture_name: The architecture of the cell being decided.
        #     variant_name: The variant of the cell being decided.
        #
        # Raises:
        #     MisconfigurationError: If the architecture is not the
        #         documented exclusion and has no declared row in the
        #         resolution table.
        #
        # Returns:
        #     The frozen decision record echoing the requested cell and
        #     carrying either the ``"registered"`` marker or the scientific
        #     reason the cell is out of scope.
        if architecture_name == "hiftnet":
            return self._support(
                architecture_name,
                variant_name,
                supported=False,
                reason="HiFTNet is the documented non-executed exclusion and enters no Study 2 cell."
            )
        resolution: ArchitectureTechniqueResolution = TechniqueResolutionTable().get(architecture_name)
        match variant_name:
            case "baseline_cpu" | "baseline_b200" | "pruned_30" | "pruned_50" | "pruned_70" | "pruned_50_recovered":
                return self._support(architecture_name, variant_name, supported=True, reason="registered")
            case "torch_compile" | "torch_compile_overhead":
                if resolution.compile_supported:
                    return self._support(architecture_name, variant_name, supported=True, reason="registered")
                return self._support(
                    architecture_name,
                    variant_name,
                    supported=False,
                    reason=(
                        "The autoregressive sampler drives submodules directly, so module-level "
                        "compilation never covers the executed synthesis path."
                    )
                )
            case "int8_dynamic":
                if resolution.int8_dynamic_supported:
                    return self._support(architecture_name, variant_name, supported=True, reason="registered")
                return self._support(
                    architecture_name,
                    variant_name,
                    supported=False,
                    reason=(
                        "Dynamic INT8 covers only Linear and GRU modules; screening measured "
                        "zero executable coverage on this convolution-dominant generator."
                    )
                )
            case "int8_weight_only":
                # The 22 July 2026 aligned-runtime probe (torch 2.13.0+cu130,
                # torchao 0.17.0, Python 3.14.2) executed INT8 weight-only
                # quantization end to end, so the cell is supported wherever
                # the architecture's operator profile grants coverage.
                if not resolution.weight_only_supported:
                    return self._support(
                        architecture_name,
                        variant_name,
                        supported=False,
                        reason=(
                            "Weight-only integer quantization targets the Linear stratum and has no "
                            "meaningful coverage on this architecture's operator profile."
                        )
                    )
                return self._support(architecture_name, variant_name, supported=True, reason="registered")
            case "int4_weight_only":
                if not resolution.weight_only_supported:
                    return self._support(
                        architecture_name,
                        variant_name,
                        supported=False,
                        reason=(
                            "Weight-only integer quantization targets the Linear stratum and has no "
                            "meaningful coverage on this architecture's operator profile."
                        )
                    )
                return self._support(
                    architecture_name,
                    variant_name,
                    supported=False,
                    reason=(
                        "TorchAO 0.17.0 INT4 weight-only requires the mslk kernel package "
                        "(ImportError: Requires mslk >= 1.0.0, measured 22 July 2026 under the "
                        "aligned torch 2.13.0 runtime), which the evaluated image did not provide."
                    )
                )
            case "fp16_weights":
                if resolution.fp16_supported:
                    return self._support(architecture_name, variant_name, supported=True, reason="registered")
                return self._support(
                    architecture_name,
                    variant_name,
                    supported=False,
                    reason="Screening measured a dtype incompatibility in the half-precision lane."
                )
            case "onnx_fp32" | "onnx_int8_static":
                if resolution.onnx_export_supported:
                    return self._support(architecture_name, variant_name, supported=True, reason="registered")
                return self._support(
                    architecture_name,
                    variant_name,
                    supported=False,
                    reason=self._onnx_unsupported_reason(architecture_name)
                )
            case "ode_steps_8" | "ode_steps_4" | "ode_steps_2":
                if resolution.ode_step_reduction_supported:
                    return self._support(architecture_name, variant_name, supported=True, reason="registered")
                return self._support(
                    architecture_name,
                    variant_name,
                    supported=False,
                    reason="Sampling-step reduction applies only to the iterative-flow architecture."
                )
            case "dense_continued" | "pruned_50_recovered_half":
                if resolution.causal_control_selected:
                    return self._support(architecture_name, variant_name, supported=True, reason="registered")
                return self._support(
                    architecture_name,
                    variant_name,
                    supported=False,
                    reason=(
                        "Causal recovery controls are registered for the two selected architectures "
                        "only; recovery claims for the remainder stay bounded to their one recovered "
                        "checkpoint."
                    )
                )

    def supported_hardware(self, variant_name: OptimizationVariantName) -> tuple[HardwareName, ...]:
        # Resolves the registered measurement lanes for one variant name. The
        # lane is a property of the variant alone and not of the architecture,
        # because a technique and the baseline it is compared against must be
        # measured on the same hardware for the ratio to mean anything. The
        # recovery arms carry a second lane because their fine-tuning ran on
        # separate accelerator capacity.
        #
        # Args:
        #     variant_name: The registered Study 2 variant whose measurement
        #         lanes are requested.
        #
        # Returns:
        #     The non-empty tuple of hardware lanes registered for this
        #     variant; the optimized-variant evaluator refuses a run whose
        #     declared lane is absent from it.
        match variant_name:
            case "baseline_cpu" | "int8_dynamic" | "onnx_fp32" | "onnx_int8_static":
                return ("cpu",)
            case "baseline_b200" | "fp16_weights" | "torch_compile" | "torch_compile_overhead" | "int8_weight_only" | "int4_weight_only" | "pruned_30" | "pruned_50" | "pruned_70" | "ode_steps_8" | "ode_steps_4" | "ode_steps_2":
                return ("b200",)
            case "pruned_50_recovered" | "pruned_50_recovered_half" | "dense_continued":
                return ("b200", "l40s")

    def _support(
        self,
        architecture_name: ArchitectureName,
        variant_name: OptimizationVariantName,
        supported: bool,
        reason: str
    ) -> OptimizationVariantSupport:
        # Builds one immutable applicability decision record.
        return OptimizationVariantSupport(
            architecture_name=architecture_name,
            variant_name=variant_name,
            supported=supported,
            reason=reason
        )

    def _onnx_unsupported_reason(self, architecture_name: ArchitectureName) -> str:
        # Returns the recorded per-architecture reason an ONNX lane is not registered.
        # The reason is architecture-specific because the three causes are
        # scientifically distinct: an unexportable spectral synthesis operator,
        # a sampling loop that is not a static graph, and an export that was
        # never attempted under the registered presentation budget.
        match architecture_name:
            case "vocos" | "freev" | "vocosformer" | "apnet2" | "rfwave":
                return "The torch.istft synthesis head has no ONNX operator mapping."
            case "lpcnet":
                return "The autoregressive Python sampling loop is not a static exportable graph."
            case _:
                return (
                    "Export was not attempted under the registered presentation budget; "
                    "the operator graph remains unvalidated for ONNX."
                )

    def _build_technique(
        self,
        variant_name: OptimizationVariantName,
        resolution: ArchitectureTechniqueResolution
    ) -> OptimizationTechnique:
        # Builds the technique instance carrying its architecture-resolved configuration.
        # Imports are function-local so that importing this module for
        # applicability decisions alone never drags the optional quantization,
        # sampling, or runtime backends into the process; the two variants whose
        # backends are heaviest are imported inside their own match arms.
        #
        # Args:
        #     variant_name: The supported variant to construct a technique for.
        #     resolution: The architecture's declared row, which supplies the
        #         pruning arm and the quantized module kinds the technique is
        #         parameterized with.
        #
        # Returns:
        #     The constructed technique, already carrying every setting the
        #     recipe records for this cell.
        from vocode.optimization.compilation import TorchCompileAcceleration, TorchCompileAccelerationConfig
        from vocode.optimization.pruning import (
            DenseContinuedIdentity,
            MaskedMagnitudePruning,
            PrunedRecoveredVerification,
        )
        from vocode.optimization.quantization import (
            DynamicInt8Quantization,
            DynamicInt8QuantizationConfig,
            WeightOnlyFp16Storage,
            WeightOnlyIntQuantization,
            WeightOnlyIntQuantizationConfig,
        )
        match variant_name:
            case "baseline_cpu":
                return BaselineIdentity()
            case "baseline_b200":
                return BaselineIdentity(produced_variant_name="baseline_b200")
            case "torch_compile":
                return TorchCompileAcceleration()
            case "torch_compile_overhead":
                return TorchCompileAcceleration(
                    TorchCompileAccelerationConfig(mode="reduce-overhead")
                )
            case "int8_dynamic":
                return DynamicInt8Quantization(
                    DynamicInt8QuantizationConfig(quantized_module_kinds=resolution.int8_module_kinds)
                )
            case "int8_weight_only":
                return WeightOnlyIntQuantization(WeightOnlyIntQuantizationConfig(bit_width=8))
            case "int4_weight_only":
                return WeightOnlyIntQuantization(WeightOnlyIntQuantizationConfig(bit_width=4))
            case "fp16_weights":
                return WeightOnlyFp16Storage()
            case "pruned_30":
                return MaskedMagnitudePruning(resolution.pruning_arm, 0.3, "pruned_30")
            case "pruned_50":
                return MaskedMagnitudePruning(resolution.pruning_arm, 0.5, "pruned_50")
            case "pruned_70":
                return MaskedMagnitudePruning(resolution.pruning_arm, 0.7, "pruned_70")
            case "pruned_50_recovered":
                return PrunedRecoveredVerification(resolution.pruning_arm)
            case "pruned_50_recovered_half":
                return PrunedRecoveredVerification(
                    resolution.pruning_arm,
                    produced_variant_name="pruned_50_recovered_half"
                )
            case "dense_continued":
                return DenseContinuedIdentity(resolution.pruning_arm)
            case "ode_steps_8" | "ode_steps_4" | "ode_steps_2":
                from vocode.optimization.sampling import OdeStepReduction, OdeStepReductionConfig
                step_count: int = int(variant_name.removeprefix("ode_steps_"))
                return OdeStepReduction(OdeStepReductionConfig(step_count=step_count))
            case "onnx_fp32" | "onnx_int8_static":
                from vocode.optimization.deployment import OnnxRuntimeDeployment, OnnxRuntimeDeploymentConfig
                quantize_static: bool = variant_name == "onnx_int8_static"
                return OnnxRuntimeDeployment(
                    OnnxRuntimeDeploymentConfig(static_int8=quantize_static)
                )

    def _measurement_note(self, variant_name: OptimizationVariantName) -> str:
        # Returns the lane-interpretation note recorded into the run capsule.
        # The note travels with the record so a reader of the artifact tree
        # learns how the row must be read without consulting this source: which
        # denominator the ratio is against, whether the weights were altered,
        # and whether a speedup is expected at all. The three masked-pruning
        # curve points deliberately share one note because they are one curve.
        match variant_name:
            case "baseline_cpu":
                return "Unmodified fp32 baseline on the CPU lane; denominator for CPU speed claims."
            case "baseline_b200":
                return "Unmodified fp32 baseline re-measured on the B200 lane under the current runtime; same-stack denominator for compiled GPU lanes."
            case "torch_compile":
                return "Compiled execution; weights unchanged; quality must match baseline rows within seed spread."
            case "torch_compile_overhead":
                return "Reduce-overhead compiled execution; weights unchanged; second registered compiler configuration."
            case "int8_dynamic":
                return "Dynamic INT8 linear layers; CPU-lane technique; quality degradation measured, not assumed."
            case "int8_weight_only":
                return "TorchAO INT8 weight-only Linear stratum; GPU lane; quality degradation measured, not assumed."
            case "int4_weight_only":
                return "TorchAO INT4 weight-only Linear stratum behind a bf16 boundary; GPU lane; lower-bit candidate."
            case "fp16_weights":
                return "Half-precision weights and compute behind an fp32 interface; GPU lane measured."
            case "onnx_fp32":
                return "Exported ONNX Runtime CPU execution; same-lane denominator for the static INT8 deployment claim."
            case "onnx_int8_static":
                return "Calibrated static INT8 QDQ execution on ONNX Runtime CPU; deployment-lane measurement."
            case "pruned_30" | "pruned_50" | "pruned_70":
                return "Masked magnitude pruning without recovery; robustness curve point; no speedup expected from masked zeros."
            case "pruned_50_recovered":
                return "Pruned-and-recovered weights verified at load; no speedup expected from masked zeros."
            case "pruned_50_recovered_half":
                return "Half-budget pruned-and-recovered weights verified at load; recovery-budget ablation arm."
            case "dense_continued":
                return "Unpruned continuation with the matched recovery budget; causal control for recovery claims."
            case "ode_steps_8" | "ode_steps_4" | "ode_steps_2":
                return "Reduced ODE sampling schedule on the flow model; structural inference-step intervention."
