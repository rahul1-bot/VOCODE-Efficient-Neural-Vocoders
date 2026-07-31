# This module:
# 1. Implements the post-training quantization and reduced-precision storage
#    techniques of Study 2: dynamic INT8 over the architecture-resolved
#    module kinds (CPU lane), TorchAO INT8 and INT4 weight-only quantization
#    of the Linear stratum (GPU lane), and half-precision weight storage
# 2. Provides the dtype-boundary adapters that keep every transformed
#    network behind an fp32 interface, so the surrounding module, metrics,
#    and data pipeline remain untouched
#
# Harness contract (syntheticmind):
# - Every technique maps a harness Module onto a harness Module by
#   replacing its network (or the network's backbone) with the transformed
#   or adapter-wrapped object; the harness loops run unchanged
#
# Design decisions:
# - Parameter coverage is measured before every transformation and recorded
#   into the recipe; a weight-only cell measuring zero Linear coverage
#   refuses execution because it belongs in the recorded-unsupported lane
# - INT4 weight-only requires bf16 activations, so its quantized scope is
#   prepared in bf16 and wrapped in a boundary that casts at the interface;
#   the iterative-flow backbone keeps its flow, STFT, and filter-bank
#   plumbing in fp32 with only floating inputs cast
# - The dynamic-INT8 backend engine is resolved from a fixed preference
#   order and recorded, because the executing engine is part of the
#   measurement's identity
# - Storage-halving claims come with per-lane measured speed and quality
#   consequences, never assumptions, per the Study 2 contract
#
# Author: Rahul Sawhney

from typing import ClassVar, Literal

import torch
from pydantic import BaseModel, ConfigDict, PositiveInt
from torch import nn
from torch.ao.quantization import quantize_dynamic

from syntheticmind.core.module import Module

from vocode.optimization.registry import OptimizationTechnique, OptimizationVariantName, QuantizedModuleKind

__all__: list[str] = [
    "DynamicInt8Quantization",
    "DynamicInt8QuantizationConfig",
    "HalfPrecisionNetworkAdapter",
    "WeightOnlyFp16Storage",
    "WeightOnlyFp16StorageConfig",
    "WeightOnlyIntQuantization",
    "WeightOnlyIntQuantizationConfig"
]


class HalfPrecisionRfwaveBackboneAdapter(nn.Module):
    # Keeps RFWave's flow, STFT, and PQMF plumbing in fp32 while the learned backbone
    # stores and executes its weights in fp16 behind an fp32 multi-input interface.
    #
    # Integration: the adapter is installed in the backbone position of the
    # iterative-flow network, so the surrounding synthesis code keeps calling four
    # positional tensors and receiving one, unaware that the learned scope now
    # runs at half precision. Scoping the conversion to the backbone is what
    # keeps the spectral transform and filter-bank stages numerically intact,
    # since those stages are the ones a half-precision cast degrades.
    def __init__(self, backbone: nn.Module) -> None:
        # Converts the learned backbone to half precision at construction, which
        # is what makes the storage-halving claim a property of the object rather
        # than of the call path.
        #
        # Args:
        #     backbone: The learned submodule to convert and wrap; the
        #         conversion is in place on the supplied module.
        super().__init__()
        self._backbone: nn.Module = backbone.half()

    def forward(
        self,
        noisy_state: torch.Tensor,
        time_values: torch.Tensor,
        mel: torch.Tensor,
        band_index: torch.Tensor
    ) -> torch.Tensor:
        # Casts only floating model inputs; the categorical band index remains integral.
        # The band index selects a frequency band rather than carrying a
        # magnitude, so casting it would corrupt an index rather than reduce a
        # precision. The output is restored to single precision at the boundary,
        # which is what leaves the surrounding flow arithmetic untouched.
        #
        # Args:
        #     noisy_state: Current state of the integration, cast to half.
        #     time_values: Integration time points, cast to half.
        #     mel: Conditioning mel batch, cast to half.
        #     band_index: Categorical band selector, passed through unchanged.
        #
        # Returns:
        #     The backbone's prediction restored to single precision.
        return self._backbone(
            noisy_state.half(),
            time_values.half(),
            mel.half(),
            band_index
        ).float()


class Bfloat16InputBoundary(nn.Module):
    # Keeps an fp32 interface around an already bf16-prepared quantized network.
    # Inputs are cast to bf16 at the boundary and outputs restored to fp32 so the
    # surrounding module, metrics, and data pipeline remain untouched.
    # The boundary exists because the four-bit weight-only kernels require
    # brain-float activations; it is a dtype adapter and never a quantizer.
    def __init__(self, network: nn.Module) -> None:
        # Wraps the prepared network; no further conversion happens here. Both
        # the brain-float preparation and the quantization have already been
        # applied to the network by the technique before it is wrapped, which is
        # why this constructor converts nothing.
        #
        # Args:
        #     network: The already prepared and quantized network to wrap.
        super().__init__()
        self._network: nn.Module = network

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Runs the wrapped bf16 quantized network behind the fp32 interface.
        return self._network(mel.to(torch.bfloat16)).float()


class Bfloat16RfwaveBackboneBoundary(nn.Module):
    # Keeps RFWave's flow, STFT, and PQMF plumbing in fp32 while the already
    # bf16-prepared quantized backbone executes behind an fp32 multi-input interface.
    # It is the multi-input counterpart of the single-input brain-float boundary,
    # installed in the backbone position of the iterative-flow network.
    def __init__(self, backbone: nn.Module) -> None:
        # Wraps the prepared backbone; no further conversion happens here.
        #
        # Args:
        #     backbone: The already prepared and quantized backbone to wrap.
        super().__init__()
        self._backbone: nn.Module = backbone

    def forward(
        self,
        noisy_state: torch.Tensor,
        time_values: torch.Tensor,
        mel: torch.Tensor,
        band_index: torch.Tensor
    ) -> torch.Tensor:
        # Casts only floating model inputs; the categorical band index remains integral.
        return self._backbone(
            noisy_state.to(torch.bfloat16),
            time_values.to(torch.bfloat16),
            mel.to(torch.bfloat16),
            band_index
        ).float()


class DynamicInt8QuantizationConfig(BaseModel):
    # Frozen settings: the module kinds entering quantization and the
    # integer dtype label recorded in the recipe.
    #
    # Fields:
    #     quantized_module_kinds: The module kinds the transformation
    #         covers, resolved per architecture by the registry. Only the
    #         Linear and recurrent kinds exist, because those are the two
    #         dynamic quantization implements. Default: ``("linear",)``.
    #     quantized_dtype: Label of the integer representation recorded into
    #         the recipe. It documents the transformation rather than
    #         selecting it; the executed dtype is signed eight-bit by
    #         construction. Default: ``"qint8"``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    quantized_module_kinds: tuple[QuantizedModuleKind, ...] = ("linear",)
    quantized_dtype: str = "qint8"


class DynamicInt8Quantization(OptimizationTechnique):
    # Post-training dynamic INT8 quantization over the architecture-resolved module kinds.
    # Dynamic quantization executes on CPU targets; the measured parameter coverage and
    # the resolved backend engine are framework facts recorded in the recipe, and audio
    # degradation is measured on the identical test split.
    #
    # Mechanism: dynamic quantization stores the covered modules' weights as
    # eight-bit integers and derives each activation's scale at inference time
    # from the tensor actually presented, which is what removes the need for a
    # calibration pass and what confines the technique to the processor lane.
    # Only whole modules of the covered kinds are replaced, so an architecture
    # whose parameters live in convolutions is left essentially untransformed;
    # that is why coverage is measured and recorded rather than assumed, and why
    # the registry records zero-coverage cells as unsupported instead of
    # measuring them.
    #
    # The preference order below is tried against the engines the host actually
    # offers, so the resolved engine depends on the machine; it is recorded into
    # the recipe because the executing kernel implementation is part of the
    # measurement's identity.
    _engine_preference: tuple[str, ...] = ("x86", "fbgemm", "qnnpack")

    def __init__(self, configuration: DynamicInt8QuantizationConfig | None = None) -> None:
        # Binds the settings (defaulting to Linear-kind coverage) and clears
        # the engine and coverage records.
        self._configuration: DynamicInt8QuantizationConfig = (
            configuration if configuration is not None else DynamicInt8QuantizationConfig()
        )
        self._resolved_engine: str | None = None
        self._measured_coverage: float | None = None

    @property
    def name(self) -> OptimizationVariantName:
        # Returns the canonical variant name this technique produces.
        return "int8_dynamic"

    def apply(self, module: Module) -> Module:
        # Resolves and installs the quantized backend engine, measures the
        # parameter coverage of the configured kinds, and replaces the
        # network with its dynamically quantized form. The engine is installed
        # process-wide before the transformation because the conversion binds
        # the engine's kernels into the produced modules. Coverage is measured
        # before the transformation, while the original module types are still
        # present to count.
        #
        # Args:
        #     module: The harness Module carrying the restored baseline
        #         weights; its network attribute is replaced.
        #
        # Raises:
        #     RuntimeError: If the host offers no supported quantized engine,
        #         or if the network has no parameters to measure coverage over.
        #
        # Returns:
        #     The same module, now carrying the dynamically quantized network.
        self._resolved_engine: str | None = self._select_quantized_engine()
        torch.backends.quantized.engine = self._resolved_engine
        covered_module_classes: set[type[nn.Module]] = self._covered_module_classes()
        self._measured_coverage: float | None = self._measure_parameter_coverage(
            module.network,
            covered_module_classes
        )
        quantized_network: nn.Module = quantize_dynamic(
            module.network,
            covered_module_classes,
            dtype=torch.qint8
        )
        module.network = quantized_network
        return module

    @property
    def configuration(self) -> DynamicInt8QuantizationConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    def _covered_module_classes(self) -> set[type[nn.Module]]:
        # Maps the configured module kinds onto their torch module classes.
        kind_classes: dict[QuantizedModuleKind, type[nn.Module]] = {
            "linear": nn.Linear,
            "gru": nn.GRU
        }
        return {kind_classes[kind] for kind in self._configuration.quantized_module_kinds}

    def _measure_parameter_coverage(
        self,
        network: nn.Module,
        covered_module_classes: set[type[nn.Module]]
    ) -> float:
        # Measures the fraction of network parameters the configured kinds actually cover.
        # Membership is tested by exact type rather than by subclass, so a
        # subclass of a covered kind is not counted; that matches what the
        # conversion itself replaces. Parameters are counted without recursion,
        # so a covered module's own weights and biases are counted exactly once.
        #
        # Args:
        #     network: The untransformed network whose parameters are counted.
        #     covered_module_classes: The torch classes the transformation
        #         replaces.
        #
        # Raises:
        #     RuntimeError: If the network holds no parameters, so no coverage
        #         fraction is defined.
        #
        # Returns:
        #     Parameters of covered modules as a fraction of all network
        #     parameters.
        covered_count: int = sum(
            parameter.numel()
            for candidate_module in network.modules()
            if type(candidate_module) in covered_module_classes
            for parameter in candidate_module.parameters(recurse=False)
        )
        total_count: int = sum(parameter.numel() for parameter in network.parameters())
        if total_count == 0:
            raise RuntimeError("Coverage measurement found an empty network.")
        return covered_count / total_count

    def _select_quantized_engine(self) -> str:
        # Resolves the first available quantized backend engine for this host.
        # The fixed preference order makes the choice deterministic for a given
        # host rather than dependent on enumeration order, and a host offering
        # none of the three is refused rather than silently left on whatever
        # default is installed.
        #
        # Raises:
        #     RuntimeError: If none of the preferred engines is available,
        #         naming what the host does offer.
        #
        # Returns:
        #     The name of the engine installed for this transformation.
        supported_engines: list[str] = list(torch.backends.quantized.supported_engines)
        for candidate_engine in self._engine_preference:
            if candidate_engine in supported_engines:
                return candidate_engine
        raise RuntimeError(
            f"No supported quantized engine available; host offers {supported_engines}."
        )

    def configuration_dump(self) -> dict[str, object]:
        # Returns the exact transformation configuration for the optimization recipe.
        return {
            "technique": "dynamic_int8",
            "quantized_engine": self._resolved_engine,
            "measured_parameter_coverage": self._measured_coverage,
            **self._configuration.model_dump(mode="json")
        }


class WeightOnlyIntQuantizationConfig(BaseModel):
    # Frozen settings: the registered bit width and the INT4 quantization
    # group size.
    #
    # Fields:
    #     bit_width: Integer width the Linear weights are stored at, which
    #         also decides which of the two weight-only variant names the
    #         technique reports and whether the brain-float preparation path
    #         is taken. Default: ``8``.
    #     int4_group_size: Number of weights sharing one scale in the
    #         four-bit lane; smaller groups cost more scale storage and lose
    #         less accuracy. The eight-bit lane does not read this field.
    #         Default: ``128``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    bit_width: Literal[4, 8] = 8
    int4_group_size: PositiveInt = 128


class WeightOnlyIntQuantization(OptimizationTechnique):
    # TorchAO weight-only integer quantization over the Linear stratum on the GPU lane.
    # INT8 weight-only keeps fp32 activations; INT4 weight-only requires bf16 activations,
    # so the quantized scope is prepared in bf16 and executes behind an fp32 boundary.
    # Coverage is measured before transformation and recorded into the recipe.
    #
    # Mechanism: weight-only quantization stores the Linear weights at reduced
    # integer width and dequantizes them inside the kernel, leaving activations
    # in floating point throughout. The gain is therefore in memory traffic and
    # footprint rather than in integer arithmetic, which is what makes it an
    # accelerator-lane technique while dynamic INT8 is a processor-lane one. The
    # transformation reaches only Linear modules, so an architecture whose
    # parameters live elsewhere gains nothing; zero measured coverage is refused
    # outright rather than measured, because such a cell belongs in the recorded
    # unsupported lane.
    def __init__(self, configuration: WeightOnlyIntQuantizationConfig | None = None) -> None:
        # Binds the settings (defaulting to INT8) and clears the coverage
        # and scope records.
        self._configuration: WeightOnlyIntQuantizationConfig = (
            configuration if configuration is not None else WeightOnlyIntQuantizationConfig()
        )
        self._measured_coverage: float | None = None
        self._quantized_scope: str | None = None

    @property
    def name(self) -> OptimizationVariantName:
        # Returns the canonical variant name this technique produces.
        if self._configuration.bit_width == 8:
            return "int8_weight_only"
        return "int4_weight_only"

    def apply(self, module: Module) -> Module:
        # Measures Linear coverage (refusing zero-coverage cells), then
        # quantizes: INT8 in place over the fp32 network, INT4 over a
        # bf16-prepared scope wrapped in the fp32 casting boundary, with
        # the backbone-only scope taken where the network exposes one. The
        # backend import is function-local, so an unsupported cell can be decided
        # and a technique constructed without the backend installed.
        #
        # Args:
        #     module: The harness Module carrying the restored baseline
        #         weights; its network or that network's backbone is
        #         transformed.
        #
        # Raises:
        #     RuntimeError: If the network has no parameters, or if the Linear
        #         stratum covers none of them, which is the state that belongs
        #         in the recorded unsupported lane rather than in a measurement.
        #     ModuleNotFoundError: If the quantization backend is absent from
        #         the environment.
        #
        # Returns:
        #     The same module, now carrying the quantized network; on the
        #     four-bit lane the transformed scope sits behind a casting
        #     boundary so the module's interface stays single precision.
        from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
        self._measured_coverage: float | None = self._measure_linear_coverage(module.network)
        if self._measured_coverage == 0.0:
            raise RuntimeError(
                "Weight-only quantization found zero Linear parameter coverage; "
                "this cell must be recorded as unsupported instead of executed."
            )
        # The eight-bit lane keeps single-precision activations, so the whole
        # network is quantized in place and needs no dtype boundary around it.
        if self._configuration.bit_width == 8:
            quantize_(module.network, Int8WeightOnlyConfig())
            self._quantized_scope: str | None = "network"
            return module
        # The four-bit lane requires brain-float activations, so from here the
        # scope is converted before quantization and wrapped in a casting
        # boundary afterwards, keeping the module's own interface unchanged.
        int4_configuration: Int4WeightOnlyConfig = Int4WeightOnlyConfig(
            group_size=self._configuration.int4_group_size
        )
        if hasattr(module.network, "backbone"):
            backbone: nn.Module = getattr(module.network, "backbone")
            prepared_backbone: nn.Module = backbone.to(torch.bfloat16)
            quantize_(prepared_backbone, int4_configuration)
            setattr(module.network, "backbone", Bfloat16RfwaveBackboneBoundary(prepared_backbone))
            self._quantized_scope: str | None = "backbone"
            return module
        prepared_network: nn.Module = module.network.to(torch.bfloat16)
        quantize_(prepared_network, int4_configuration)
        module.network = Bfloat16InputBoundary(prepared_network)
        self._quantized_scope: str | None = "network"
        return module

    @property
    def configuration(self) -> WeightOnlyIntQuantizationConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    def _measure_linear_coverage(self, network: nn.Module) -> float:
        # Measures the fraction of network parameters the Linear stratum covers.
        # The measurement is taken before any transformation, while the Linear
        # modules are still present to count, and it is exact rather than
        # approximate because it is integer arithmetic over parameter counts.
        #
        # Args:
        #     network: The untransformed network whose parameters are counted.
        #
        # Raises:
        #     RuntimeError: If the network holds no parameters at all.
        #
        # Returns:
        #     Linear-module parameters as a fraction of all network parameters.
        covered_count: int = sum(
            parameter.numel()
            for candidate_module in network.modules()
            if type(candidate_module) is nn.Linear
            for parameter in candidate_module.parameters(recurse=False)
        )
        total_count: int = sum(parameter.numel() for parameter in network.parameters())
        if total_count == 0:
            raise RuntimeError("Coverage measurement found an empty network.")
        return covered_count / total_count

    def configuration_dump(self) -> dict[str, object]:
        # Returns the exact transformation configuration for the optimization recipe.
        return {
            "technique": "weight_only_int",
            "measured_parameter_coverage": self._measured_coverage,
            "quantized_scope": self._quantized_scope,
            **self._configuration.model_dump(mode="json")
        }


class HalfPrecisionNetworkAdapter(nn.Module):
    # Casting adapter keeping an fp32 interface around a half-precision network.
    # Inputs are cast to half at the boundary and outputs restored to fp32 so the
    # surrounding module, metrics, and data pipeline remain untouched.
    # It is the single-input counterpart of the multi-input backbone adapter,
    # installed in the network position of architectures exposing no backbone.
    def __init__(self, network: nn.Module) -> None:
        # Wraps the already-converted network; the conversion happened in the
        # technique, so this constructor changes no dtype.
        #
        # Args:
        #     network: The already half-precision network to wrap.
        super().__init__()
        self._network: nn.Module = network

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Runs the wrapped half-precision network behind the fp32 interface.
        return self._network(mel.half()).float()


class WeightOnlyFp16StorageConfig(BaseModel):
    # Frozen dtype labels recorded in the recipe: half-precision storage
    # behind an fp32 interface. Both fields document the transformation for
    # the recipe rather than selecting it; the conversion and the casting
    # boundary are fixed by construction.
    #
    # Fields:
    #     storage_dtype: Label of the precision the converted scope stores
    #         and computes in. Default: ``"float16"``.
    #     interface_dtype: Label of the precision the module's interface
    #         continues to present to the surrounding pipeline.
    #         Default: ``"float32"``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    storage_dtype: str = "float16"
    interface_dtype: str = "float32"


class WeightOnlyFp16Storage(OptimizationTechnique):
    # Half-precision weight storage and compute behind the module's fp32 interface.
    # On-disk footprint halves by construction; speed and quality consequences are
    # measured per lane rather than assumed, per the Study 2 contract.
    def __init__(self, configuration: WeightOnlyFp16StorageConfig | None = None) -> None:
        # Binds the dtype labels and clears the converted-scope record.
        self._configuration: WeightOnlyFp16StorageConfig = (
            configuration if configuration is not None else WeightOnlyFp16StorageConfig()
        )
        self._converted_scope: str | None = None

    @property
    def name(self) -> OptimizationVariantName:
        # Returns the canonical variant name this technique produces.
        return "fp16_weights"

    def apply(self, module: Module) -> Module:
        # Converts the appropriate scope to half precision behind the fp32
        # casting adapter: the backbone where one exists, the whole network
        # otherwise. The backbone scope is taken where available because the
        # surrounding synthesis stages of that architecture are numerically
        # fragile at half precision, and the recorded scope tells a reader of
        # the recipe which of the two was actually converted.
        #
        # Args:
        #     module: The harness Module carrying the restored baseline
        #         weights; the converted scope is installed in place.
        #
        # Returns:
        #     The same module, now storing and computing the converted scope
        #     at half precision behind an unchanged single-precision
        #     interface.
        if hasattr(module.network, "backbone"):
            backbone: nn.Module = getattr(module.network, "backbone")
            setattr(
                module.network,
                "backbone",
                HalfPrecisionRfwaveBackboneAdapter(backbone)
            )
            self._converted_scope: str | None = "backbone"
            return module
        half_network: nn.Module = module.network.half()
        module.network = HalfPrecisionNetworkAdapter(half_network)
        self._converted_scope: str | None = "network"
        return module

    def configuration_dump(self) -> dict[str, object]:
        # Returns the exact transformation configuration for the optimization recipe.
        return {
            "technique": "weight_only_fp16",
            "converted_scope": self._converted_scope,
            **self._configuration.model_dump(mode="json")
        }

    @property
    def configuration(self) -> WeightOnlyFp16StorageConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration
