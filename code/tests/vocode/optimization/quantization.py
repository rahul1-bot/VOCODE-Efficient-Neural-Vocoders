# This module:
# 1. Verifies dynamic INT8 quantization on minimal Linear and recurrent
#    networks: the transformed module types, output finiteness, the resolved
#    backend engine, and the measured parameter coverage recorded into the
#    recipe
# 2. Verifies the weight-only integer quantization configuration, its
#    bit-width-to-variant-name decision, and its recipe dump
# 3. Verifies half-precision weight storage on both scopes: the installed
#    casting adapter, the halved parameter dtype, and the fp32 interface
# 4. Verifies the dtype-boundary adapters, including that the categorical
#    band index of the multi-input backbone is never cast
#
# Design decisions:
# - The dynamic quantization tests install and restore the process-wide
#   quantized backend engine, because the technique resolves and installs it
#   as part of the measurement's identity
# - Coverage is asserted as an exact parameter fraction on hand-counted
#   minimal networks rather than as a tolerance, because the measurement is
#   integer arithmetic over parameter counts
# - Quantized numerics are bounded by dtype, shape, and finiteness only;
#   pinning quantized activations to float values would encode the backend
#   engine of the executing host into the suite
# - The weight-only apply path is asserted behind a dependency probe:
#   torchao is not installed in this environment, so the assertion is that
#   the missing backend surfaces rather than being silently skipped; INT4
#   preparation additionally requires accelerator kernels and is never
#   invoked here
#
# Author: Rahul Sawhney

import importlib.util
import unittest
from typing import ClassVar, override

import torch
from pydantic import BaseModel, ConfigDict, ValidationError
from torch import nn
from torch.ao.nn.quantized.dynamic import GRU as DynamicQuantizedGru
from torch.ao.nn.quantized.dynamic import Linear as DynamicQuantizedLinear

from syntheticmind.core.module import Module

from vocode.optimization.quantization import (
    Bfloat16InputBoundary,
    Bfloat16RfwaveBackboneBoundary,
    DynamicInt8Quantization,
    DynamicInt8QuantizationConfig,
    HalfPrecisionNetworkAdapter,
    HalfPrecisionRfwaveBackboneAdapter,
    WeightOnlyFp16Storage,
    WeightOnlyFp16StorageConfig,
    WeightOnlyIntQuantization,
    WeightOnlyIntQuantizationConfig,
)


class TinyLinearNetwork(nn.Module):
    # Minimal Linear-only stand-in for a trained generator network.
    def __init__(self) -> None:
        # Builds the single projection the Linear stratum covers in full.
        super().__init__()
        self.projection: nn.Linear = nn.Linear(4, 4)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Projects the conditioning mel through the only layer of the network.
        return self.projection(mel)


class TinyMixedNetwork(nn.Module):
    # Minimal network mixing a Linear projection with a convolutional stage.
    # Its parameter budget is hand-countable: the projection contributes sixteen
    # weights and four biases while the convolution contributes twelve weights
    # and two biases, so the Linear stratum covers twenty of thirty-four
    # parameters and the coverage fraction is exact integer arithmetic.
    def __init__(self) -> None:
        # Builds a projection and a convolution the Linear stratum cannot reach.
        super().__init__()
        self.projection: nn.Linear = nn.Linear(4, 4)
        self.convolution: nn.Conv1d = nn.Conv1d(2, 2, 3, padding=1)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Runs the projection and then the convolution.
        return self.convolution(self.projection(mel))


class TinyRecurrentNetwork(nn.Module):
    # Minimal recurrent network standing in for the eight-bit reference deployment.
    def __init__(self) -> None:
        # Builds the recurrent layer the GRU kind covers.
        super().__init__()
        self.recurrent: nn.GRU = nn.GRU(4, 4, batch_first=True)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Returns the sequence output and discards the final hidden state.
        sequence_output: torch.Tensor
        sequence_output, _ = self.recurrent(mel)
        return sequence_output


class ParameterFreeNetwork(nn.Module):
    # Parameter-free network used to prove the coverage measurement fails closed.
    def __init__(self) -> None:
        # Builds a passthrough carrying no parameters at all.
        super().__init__()
        self.passthrough: nn.Identity = nn.Identity()

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Returns the input unchanged.
        return self.passthrough(mel)


class DtypeRecordingBackbone(nn.Module):
    # Four-input backbone stand-in recording the dtypes its inputs arrived in.
    # Recording at the far side of the boundary is the only way to prove which
    # inputs were cast: inspecting the boundary's output alone would show the
    # restored interface dtype and say nothing about what the wrapped module saw.
    def __init__(self) -> None:
        # Builds the projection and opens the empty dtype record.
        super().__init__()
        self.projection: nn.Linear = nn.Linear(4, 4)
        self.received_dtypes: tuple[torch.dtype, ...] = ()

    @override
    def forward(
        self,
        noisy_state: torch.Tensor,
        time_values: torch.Tensor,
        mel: torch.Tensor,
        band_index: torch.Tensor
    ) -> torch.Tensor:
        # Records the dtype each of the four inputs arrived in, then projects.
        self.received_dtypes: tuple[torch.dtype, ...] = (
            noisy_state.dtype,
            time_values.dtype,
            mel.dtype,
            band_index.dtype
        )
        return self.projection(noisy_state)


class TinyBackboneNetwork(nn.Module):
    # Minimal network exposing a backbone submodule, the narrower transformed scope.
    def __init__(self, backbone: nn.Module) -> None:
        # Binds the backbone submodule that is the narrower transformed scope.
        super().__init__()
        self.backbone: nn.Module = backbone

    @override
    def forward(
        self,
        noisy_state: torch.Tensor,
        time_values: torch.Tensor,
        mel: torch.Tensor,
        band_index: torch.Tensor
    ) -> torch.Tensor:
        # Forwards the four-tensor batch to the backbone unchanged.
        return self.backbone(noisy_state, time_values, mel, band_index)


class TinyHarnessModule(Module):
    # Minimal harness Module exposing the network attribute techniques transform.
    def __init__(self, network: nn.Module) -> None:
        # Binds the network the quantization techniques transform.
        super().__init__()
        self.network: nn.Module = network

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Delegates synthesis to the bound network.
        return self.network(mel)


class MultiInputBatch(BaseModel):
    # Frozen four-tensor input batch of the multi-input backbone interface,
    # whose band index is categorical and must survive every dtype boundary.
    #
    # Fields:
    #     noisy_state: The integration state tensor, which the boundaries are
    #         expected to cast.
    #     time_values: The integration time points, which the boundaries are
    #         expected to cast.
    #     mel: The conditioning batch, which the boundaries are expected to
    #         cast.
    #     band_index: The categorical band selector, built from an integer
    #         range so any cast would change its dtype visibly; it is the one
    #         input every boundary must leave untouched.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True
    )
    noisy_state: torch.Tensor
    time_values: torch.Tensor
    mel: torch.Tensor
    band_index: torch.Tensor


class MultiInputBatchBuilder:
    # Produces one deterministic instance of the multi-input batch.
    def build(self) -> MultiInputBatch:
        # Seeds construction and returns one deterministic four-tensor batch.
        torch.manual_seed(0)
        return MultiInputBatch(
            noisy_state=torch.randn(2, 4),
            time_values=torch.rand(2),
            mel=torch.randn(2, 4),
            band_index=torch.arange(2)
        )


class BackendDependencyProbe:
    # Reports whether an optional quantization backend is importable in this environment.
    #
    # Integration: the probe lets a case assert one contract under either
    # environment instead of skipping. Where the backend is absent, the assertion
    # is that its absence surfaces as an import failure and leaves no partial
    # transformation behind; where it is present, the assertion is the
    # transformation itself. Neither branch is a silent pass.
    def is_installed(self, module_name: str) -> bool:
        # Reports whether the named backend resolves, without importing it.
        # Resolution rather than import keeps the probe free of the backend's
        # own import cost and side effects.
        #
        # Args:
        #     module_name: Top-level module name of the optional backend.
        #
        # Returns:
        #     Whether an import of that name would resolve in this
        #     environment.
        return importlib.util.find_spec(module_name) is not None


class DynamicInt8ConfigurationTest(unittest.TestCase):
    # Verifies the frozen dynamic-INT8 settings record.
    def setUp(self) -> None:
        # Binds the default settings record the default and mutation cases read.
        self._configuration: DynamicInt8QuantizationConfig = DynamicInt8QuantizationConfig()

    def test_default_configuration_covers_the_linear_kind_in_eight_bit(self) -> None:
        # The default coverage is the Linear stratum labelled as eight-bit integers.
        self.assertEqual(self._configuration.quantized_module_kinds, ("linear",))
        self.assertEqual(self._configuration.quantized_dtype, "qint8")

    def test_recurrent_kind_is_accepted_alongside_the_linear_kind(self) -> None:
        # The recurrent deployment adds the GRU kind to the covered set.
        configuration: DynamicInt8QuantizationConfig = DynamicInt8QuantizationConfig(
            quantized_module_kinds=("linear", "gru")
        )
        self.assertEqual(configuration.quantized_module_kinds, ("linear", "gru"))

    def test_unregistered_module_kind_is_refused(self) -> None:
        # Dynamic INT8 covers only the two registered module kinds.
        with self.assertRaises(ValidationError):
            DynamicInt8QuantizationConfig(quantized_module_kinds=("conv1d",))

    def test_configuration_rejects_mutation_and_unknown_fields(self) -> None:
        # The settings record is frozen and closed to extra keys.
        with self.assertRaises(ValidationError):
            self._configuration.quantized_dtype = "qint4"
        with self.assertRaises(ValidationError):
            DynamicInt8QuantizationConfig(unknown_field=1)

    def test_technique_reports_its_canonical_variant_name(self) -> None:
        # The technique produces the single dynamic-INT8 variant.
        self.assertEqual(DynamicInt8Quantization().name, "int8_dynamic")

    def test_omitted_configuration_binds_the_default_record(self) -> None:
        # Constructing without settings binds Linear-kind coverage.
        self.assertEqual(DynamicInt8Quantization().configuration, DynamicInt8QuantizationConfig())


class DynamicInt8TransformationTest(unittest.TestCase):
    # Verifies the transformed module types, execution, and recorded facts of dynamic INT8.
    def setUp(self) -> None:
        # Seeds construction and records the process-wide engine the technique replaces.
        # The engine is process-wide state the technique installs as part of its
        # transformation, so it is captured here and restored afterwards rather
        # than being left changed for whichever suite runs next.
        torch.manual_seed(0)
        self._original_engine: str = torch.backends.quantized.engine
        self._technique: DynamicInt8Quantization = DynamicInt8Quantization()

    def tearDown(self) -> None:
        # Restores the engine recorded before the technique installed its own.
        # The restoration is guarded because a recorded engine this host does not
        # support cannot be reinstalled, and failing here would mask the result
        # of the case that has run.
        if self._original_engine in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = self._original_engine

    def test_linear_modules_become_dynamically_quantized_modules(self) -> None:
        # The covered Linear stratum is replaced by its dynamically quantized counterpart.
        module: TinyHarnessModule = TinyHarnessModule(TinyLinearNetwork())
        original_network: nn.Module = module.network
        returned: Module = self._technique.apply(module)
        self.assertIs(returned, module)
        self.assertIsNot(returned.network, original_network)
        self.assertIsInstance(getattr(returned.network, "projection"), DynamicQuantizedLinear)

    def test_quantized_linear_network_produces_finite_output_of_the_expected_shape(self) -> None:
        # The quantized network still executes end to end behind the fp32 interface.
        # Shape and finiteness are the right bound here: pinning the output to
        # float values would encode the executing host's backend kernels into the
        # suite, so quality loss is measured on the study's own test split rather
        # than asserted here.
        module: TinyHarnessModule = TinyHarnessModule(TinyLinearNetwork())
        self._technique.apply(module)
        output: torch.Tensor = module.network(torch.randn(2, 4))
        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertTrue(torch.isfinite(output).all(), msg="Quantized output must stay finite.")

    def test_recurrent_modules_become_dynamically_quantized_modules(self) -> None:
        # The recurrent kind is covered when the architecture resolves it.
        technique: DynamicInt8Quantization = DynamicInt8Quantization(
            DynamicInt8QuantizationConfig(quantized_module_kinds=("linear", "gru"))
        )
        module: TinyHarnessModule = TinyHarnessModule(TinyRecurrentNetwork())
        technique.apply(module)
        self.assertIsInstance(getattr(module.network, "recurrent"), DynamicQuantizedGru)

    def test_quantized_recurrent_network_produces_finite_output(self) -> None:
        # The quantized recurrent network executes on its sequence input.
        technique: DynamicInt8Quantization = DynamicInt8Quantization(
            DynamicInt8QuantizationConfig(quantized_module_kinds=("linear", "gru"))
        )
        module: TinyHarnessModule = TinyHarnessModule(TinyRecurrentNetwork())
        technique.apply(module)
        output: torch.Tensor = module.network(torch.randn(1, 3, 4))
        self.assertEqual(tuple(output.shape), (1, 3, 4))
        self.assertTrue(torch.isfinite(output).all())

    def test_resolved_engine_is_installed_and_recorded(self) -> None:
        # The executing backend engine is part of the measurement identity.
        module: TinyHarnessModule = TinyHarnessModule(TinyLinearNetwork())
        self._technique.apply(module)
        dump: dict[str, object] = self._technique.configuration_dump()
        self.assertIn(dump["quantized_engine"], list(torch.backends.quantized.supported_engines))
        self.assertEqual(torch.backends.quantized.engine, dump["quantized_engine"])

    def test_full_linear_network_measures_complete_parameter_coverage(self) -> None:
        # A Linear-only network is covered in full by the Linear stratum.
        module: TinyHarnessModule = TinyHarnessModule(TinyLinearNetwork())
        self._technique.apply(module)
        self.assertEqual(self._technique.configuration_dump()["measured_parameter_coverage"], 1.0)

    def test_mixed_network_measures_the_exact_covered_parameter_fraction(self) -> None:
        # Coverage is the exact fraction of parameters the configured kinds reach.
        module: TinyHarnessModule = TinyHarnessModule(TinyMixedNetwork())
        self._technique.apply(module)
        self.assertAlmostEqual(
            float(self._technique.configuration_dump()["measured_parameter_coverage"]),
            20.0 / 34.0,
            places=12,
            msg="Linear parameters are 20 of the network's 34 parameters."
        )

    def test_parameter_free_network_fails_the_coverage_measurement(self) -> None:
        # A network without parameters cannot yield a meaningful coverage fraction.
        module: TinyHarnessModule = TinyHarnessModule(ParameterFreeNetwork())
        with self.assertRaisesRegex(RuntimeError, "empty network"):
            self._technique.apply(module)

    def test_configuration_dump_before_apply_leaves_engine_and_coverage_unrecorded(self) -> None:
        # Nothing is claimed about a transformation that has not run.
        self.assertEqual(
            self._technique.configuration_dump(),
            {
                "technique": "dynamic_int8",
                "quantized_engine": None,
                "measured_parameter_coverage": None,
                "quantized_module_kinds": ["linear"],
                "quantized_dtype": "qint8"
            }
        )


class WeightOnlyIntConfigurationTest(unittest.TestCase):
    # Verifies the weight-only integer settings, naming decision, and recipe dump.
    def setUp(self) -> None:
        # Binds a technique built without settings, which is the eight-bit lane.
        self._technique: WeightOnlyIntQuantization = WeightOnlyIntQuantization()

    def test_default_configuration_is_eight_bit_with_the_registered_group_size(self) -> None:
        # The default weight-only lane is eight-bit; the group size serves the four-bit lane.
        self.assertEqual(self._technique.configuration.bit_width, 8)
        self.assertEqual(self._technique.configuration.int4_group_size, 128)

    def test_registered_bit_widths_map_onto_their_variant_names(self) -> None:
        # Each registered bit width produces its own variant name.
        eight_bit: WeightOnlyIntQuantization = WeightOnlyIntQuantization(
            WeightOnlyIntQuantizationConfig(bit_width=8)
        )
        four_bit: WeightOnlyIntQuantization = WeightOnlyIntQuantization(
            WeightOnlyIntQuantizationConfig(bit_width=4)
        )
        self.assertEqual(eight_bit.name, "int8_weight_only")
        self.assertEqual(four_bit.name, "int4_weight_only")

    def test_unregistered_bit_width_is_refused(self) -> None:
        # Only the two registered bit widths exist in the study.
        with self.assertRaises(ValidationError):
            WeightOnlyIntQuantizationConfig(bit_width=16)

    def test_non_positive_group_size_is_refused(self) -> None:
        # The quantization group size must be a positive integer.
        with self.assertRaises(ValidationError):
            WeightOnlyIntQuantizationConfig(int4_group_size=0)

    def test_configuration_rejects_mutation_and_unknown_fields(self) -> None:
        # The settings record is frozen and closed to extra keys.
        with self.assertRaises(ValidationError):
            self._technique.configuration.bit_width = 4
        with self.assertRaises(ValidationError):
            WeightOnlyIntQuantizationConfig(unknown_field=1)

    def test_configuration_dump_before_apply_leaves_coverage_and_scope_unrecorded(self) -> None:
        # Nothing is claimed about a transformation that has not run.
        self.assertEqual(
            self._technique.configuration_dump(),
            {
                "technique": "weight_only_int",
                "measured_parameter_coverage": None,
                "quantized_scope": None,
                "bit_width": 8,
                "int4_group_size": 128
            }
        )

    def test_missing_backend_surfaces_when_the_eight_bit_lane_is_applied(self) -> None:
        # The weight-only lane depends on an optional backend; its absence must surface.
        module: TinyHarnessModule = TinyHarnessModule(TinyLinearNetwork())
        probe: BackendDependencyProbe = BackendDependencyProbe()
        if not probe.is_installed("torchao"):
            with self.assertRaises(ModuleNotFoundError):
                self._technique.apply(module)
            self.assertIsNone(self._technique.configuration_dump()["quantized_scope"])
            return
        returned: Module = self._technique.apply(module)
        self.assertIs(returned, module)
        self.assertEqual(self._technique.configuration_dump()["quantized_scope"], "network")
        self.assertEqual(self._technique.configuration_dump()["measured_parameter_coverage"], 1.0)


class HalfPrecisionStorageTest(unittest.TestCase):
    # Verifies half-precision weight storage behind the module's fp32 interface.
    def setUp(self) -> None:
        # Seeds construction and binds the half-precision storage technique.
        torch.manual_seed(0)
        self._technique: WeightOnlyFp16Storage = WeightOnlyFp16Storage()

    def test_technique_reports_its_canonical_variant_name(self) -> None:
        # The technique produces the half-precision storage variant.
        self.assertEqual(self._technique.name, "fp16_weights")

    def test_default_configuration_records_the_storage_and_interface_dtypes(self) -> None:
        # Storage halves while the interface stays single precision.
        self.assertEqual(self._technique.configuration.storage_dtype, "float16")
        self.assertEqual(self._technique.configuration.interface_dtype, "float32")

    def test_configuration_rejects_mutation_and_unknown_fields(self) -> None:
        # The dtype labels are frozen and closed to extra keys.
        with self.assertRaises(ValidationError):
            self._technique.configuration.storage_dtype = "bfloat16"
        with self.assertRaises(ValidationError):
            WeightOnlyFp16StorageConfig(unknown_field=1)

    def test_network_scope_is_wrapped_in_the_casting_adapter(self) -> None:
        # Without a backbone the whole network is converted behind the casting adapter.
        module: TinyHarnessModule = TinyHarnessModule(TinyLinearNetwork())
        returned: Module = self._technique.apply(module)
        self.assertIs(returned, module)
        self.assertIsInstance(returned.network, HalfPrecisionNetworkAdapter)
        self.assertEqual(self._technique.configuration_dump()["converted_scope"], "network")

    def test_converted_network_stores_every_parameter_in_half_precision(self) -> None:
        # The storage claim is a parameter-dtype fact, not an assumption.
        module: TinyHarnessModule = TinyHarnessModule(TinyLinearNetwork())
        self._technique.apply(module)
        parameter_dtypes: set[torch.dtype] = {
            parameter.dtype for parameter in module.network.parameters()
        }
        self.assertEqual(parameter_dtypes, {torch.float16})

    def test_casting_adapter_keeps_a_single_precision_interface(self) -> None:
        # Inputs are cast at the boundary and outputs restored, so the pipeline is untouched.
        module: TinyHarnessModule = TinyHarnessModule(TinyLinearNetwork())
        self._technique.apply(module)
        output: torch.Tensor = module.network(torch.randn(2, 4))
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertTrue(torch.isfinite(output).all())

    def test_backbone_scope_is_wrapped_when_a_backbone_is_exposed(self) -> None:
        # A backbone-bearing network converts only that submodule.
        module: TinyHarnessModule = TinyHarnessModule(TinyBackboneNetwork(DtypeRecordingBackbone()))
        original_network: nn.Module = module.network
        self._technique.apply(module)
        self.assertIs(module.network, original_network)
        self.assertIsInstance(getattr(module.network, "backbone"), HalfPrecisionRfwaveBackboneAdapter)
        self.assertEqual(self._technique.configuration_dump()["converted_scope"], "backbone")

    def test_configuration_dump_before_apply_leaves_the_scope_unrecorded(self) -> None:
        # Nothing is claimed about a conversion that has not run.
        self.assertEqual(
            self._technique.configuration_dump(),
            {
                "technique": "weight_only_fp16",
                "converted_scope": None,
                "storage_dtype": "float16",
                "interface_dtype": "float32"
            }
        )


class DtypeBoundaryAdapterTest(unittest.TestCase):
    # Verifies the casting adapters that keep transformed networks behind an fp32 interface.
    def setUp(self) -> None:
        # Seeds construction and binds the builder of the multi-input batch.
        torch.manual_seed(0)
        self._batch_builder: MultiInputBatchBuilder = MultiInputBatchBuilder()

    def test_half_precision_adapter_restores_single_precision_output(self) -> None:
        # The adapter casts inputs down and outputs back up.
        adapter: HalfPrecisionNetworkAdapter = HalfPrecisionNetworkAdapter(TinyLinearNetwork().half())
        output: torch.Tensor = adapter(torch.randn(2, 4))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())

    def test_half_precision_backbone_adapter_leaves_the_band_index_integral(self) -> None:
        # Only floating model inputs are cast; the categorical band index is not.
        backbone: DtypeRecordingBackbone = DtypeRecordingBackbone()
        adapter: HalfPrecisionRfwaveBackboneAdapter = HalfPrecisionRfwaveBackboneAdapter(backbone)
        batch: MultiInputBatch = self._batch_builder.build()
        output: torch.Tensor = adapter(
            batch.noisy_state,
            batch.time_values,
            batch.mel,
            batch.band_index
        )
        self.assertEqual(
            backbone.received_dtypes,
            (torch.float16, torch.float16, torch.float16, batch.band_index.dtype)
        )
        self.assertEqual(output.dtype, torch.float32)

    def test_half_precision_backbone_adapter_converts_the_backbone_at_construction(self) -> None:
        # The learned backbone stores its weights in half precision from construction on.
        adapter: HalfPrecisionRfwaveBackboneAdapter = HalfPrecisionRfwaveBackboneAdapter(
            DtypeRecordingBackbone()
        )
        parameter_dtypes: set[torch.dtype] = {parameter.dtype for parameter in adapter.parameters()}
        self.assertEqual(parameter_dtypes, {torch.float16})

    def test_bfloat16_boundary_restores_single_precision_output(self) -> None:
        # The prepared network runs in bfloat16 behind a single-precision interface.
        boundary: Bfloat16InputBoundary = Bfloat16InputBoundary(
            TinyLinearNetwork().to(torch.bfloat16)
        )
        output: torch.Tensor = boundary(torch.randn(2, 4))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())

    def test_bfloat16_boundary_performs_no_further_conversion(self) -> None:
        # The boundary wraps an already-prepared network; it converts nothing itself.
        boundary: Bfloat16InputBoundary = Bfloat16InputBoundary(TinyLinearNetwork())
        parameter_dtypes: set[torch.dtype] = {parameter.dtype for parameter in boundary.parameters()}
        self.assertEqual(parameter_dtypes, {torch.float32})

    def test_bfloat16_backbone_boundary_leaves_the_band_index_integral(self) -> None:
        # Only floating model inputs are cast at the multi-input boundary.
        backbone: DtypeRecordingBackbone = DtypeRecordingBackbone()
        boundary: Bfloat16RfwaveBackboneBoundary = Bfloat16RfwaveBackboneBoundary(
            backbone.to(torch.bfloat16)
        )
        batch: MultiInputBatch = self._batch_builder.build()
        output: torch.Tensor = boundary(
            batch.noisy_state,
            batch.time_values,
            batch.mel,
            batch.band_index
        )
        self.assertEqual(
            backbone.received_dtypes,
            (torch.bfloat16, torch.bfloat16, torch.bfloat16, batch.band_index.dtype)
        )
        self.assertEqual(output.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
