# This module:
# 1. Verifies the closed Study 2 variant vocabulary: the exact nineteen
#    variant names, the two pruning arm names, and the two quantized module
#    kinds
# 2. Verifies the declarative per-architecture resolution table: its exact
#    row set, the technique arms and support flags of representative rows,
#    and the failure of an undeclared architecture
# 3. Verifies the applicability decision for every (architecture, variant)
#    cell: the exact supported count per architecture, the recorded reason
#    exposed on unsupported cells, and the documented HiFTNet exclusion
# 4. Verifies the registered measurement lanes per variant, the technique
#    instance built for supported cells, the lane-interpretation notes, the
#    BaselineIdentity null technique, and the base technique contract
#
# Design decisions:
# - The vocabulary is read from the literal type alias itself rather than
#   restated from the registry, so a silent addition or removal of a variant
#   name fails the membership assertion
# - Applicability is asserted through exact per-architecture supported
#   counts plus targeted reason-substring checks, which catches both a cell
#   flipping state and a reason losing its scientific content
# - Technique construction is asserted through public type, configuration,
#   and configuration_dump surfaces only; no technique is applied here
#   because the transformation behaviour belongs to the per-technique files
# - Building every supported cell imports no optional backend (torchao and
#   onnxruntime stay unimported), which is the registry's own stated design
#   guarantee and is exercised by constructing all supported records
#
# Author: Rahul Sawhney

import unittest
from typing import TypeAliasType, get_args, override

import torch
from pydantic import ValidationError
from torch import nn

from syntheticmind.core.module import Module
from syntheticmind.utilities.exceptions import MisconfigurationError

from vocode.models.vocoder import ArchitectureName
from vocode.optimization.compilation import TorchCompileAcceleration
from vocode.optimization.deployment import OnnxRuntimeDeployment
from vocode.optimization.pruning import DenseContinuedIdentity, MaskedMagnitudePruning, PrunedRecoveredVerification
from vocode.optimization.quantization import DynamicInt8Quantization, WeightOnlyFp16Storage, WeightOnlyIntQuantization
from vocode.optimization.registry import (
    ArchitectureTechniqueResolution,
    BaselineIdentity,
    OptimizationTechnique,
    OptimizationVariantName,
    OptimizationVariantRecord,
    OptimizationVariantRegistry,
    OptimizationVariantSupport,
    PruningArmName,
    QuantizedModuleKind,
    TechniqueResolutionTable,
)
from vocode.optimization.sampling import OdeStepReduction


class LiteralMemberReader:
    # Reads the member tuple standing behind a literal type alias.
    #
    # Integration: reading the vocabulary from the alias itself, rather than
    # restating it in this file, is what makes the vocabulary assertions
    # meaningful. A name added to or removed from a closed type immediately
    # changes what this reader returns, so the membership assertions fail rather
    # than silently continuing to check a stale list.
    def read(self, alias: TypeAliasType) -> tuple[str, ...]:
        # Unwraps the alias and returns its literal members as text, in order.
        # Declaration order is preserved, so an assertion may check both the
        # membership and the ordering of a closed vocabulary.
        #
        # Args:
        #     alias: The literal type alias whose members are read.
        #
        # Returns:
        #     The alias members as text, in declaration order.
        return tuple(str(member) for member in get_args(alias.__value__))


class TinyLinearNetwork(nn.Module):
    # Minimal Linear-only stand-in for a trained generator network.
    def __init__(self) -> None:
        # Builds the single projection of the stand-in network.
        super().__init__()
        self.projection: nn.Linear = nn.Linear(4, 4)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Projects the conditioning mel through the only layer of the network.
        return self.projection(mel)


class TinyHarnessModule(Module):
    # Minimal harness Module exposing the network attribute techniques transform.
    def __init__(self, network: nn.Module) -> None:
        # Binds the network a technique would transform.
        super().__init__()
        self.network: nn.Module = network

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Delegates synthesis to the bound network.
        return self.network(mel)


class OptimizationVocabularyTest(unittest.TestCase):
    # Verifies the closed variant, pruning arm, and quantized module kind vocabularies.
    def setUp(self) -> None:
        # Binds the reader resolving each closed vocabulary.
        self._reader: LiteralMemberReader = LiteralMemberReader()

    def test_variant_vocabulary_holds_the_nineteen_registered_names_in_order(self) -> None:
        # The variant vocabulary is exactly the registered nineteen names.
        expected_variants: tuple[str, ...] = (
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
        )
        self.assertEqual(
            self._reader.read(OptimizationVariantName),
            expected_variants,
            msg="The Study 2 variant vocabulary changed; the study scope is a registered contract."
        )

    def test_pruning_arm_vocabulary_holds_the_two_registered_arms(self) -> None:
        # Exactly two pruning arms exist: linear structured and global unstructured.
        self.assertEqual(
            self._reader.read(PruningArmName),
            ("linear_structured", "global_unstructured")
        )

    def test_quantized_module_kind_vocabulary_holds_linear_and_recurrent_kinds(self) -> None:
        # Dynamic INT8 covers exactly the Linear and GRU module kinds.
        self.assertEqual(self._reader.read(QuantizedModuleKind), ("linear", "gru"))


class TechniqueResolutionTableTest(unittest.TestCase):
    # Verifies the declarative per-architecture technique-arm resolution rows.
    def setUp(self) -> None:
        # Binds the resolution table under test and the vocabulary reader.
        self._table: TechniqueResolutionTable = TechniqueResolutionTable()
        self._reader: LiteralMemberReader = LiteralMemberReader()

    def test_table_declares_every_architecture_except_the_excluded_one(self) -> None:
        # Twelve rows cover the whole architecture vocabulary except HiFTNet.
        declared: tuple[str, ...] = tuple(
            resolution.architecture_name for resolution in self._table.resolutions
        )
        architecture_names: tuple[str, ...] = self._reader.read(ArchitectureName)
        self.assertEqual(len(declared), 12)
        self.assertEqual(len(set(declared)), 12, msg="Resolution rows must be unique per architecture.")
        self.assertEqual(
            set(architecture_names) - set(declared),
            {"hiftnet"},
            msg="HiFTNet is the only architecture without a declared technique resolution."
        )

    def test_transformer_backbones_take_the_linear_structured_arm(self) -> None:
        # Linear-bearing backbones resolve to the structured pruning arm.
        linear_bearing_names: tuple[str, ...] = ("vocos", "freev", "vocosformer", "rfwave")
        architecture_name: str
        for architecture_name in linear_bearing_names:
            resolution: ArchitectureTechniqueResolution = self._table.get(architecture_name)
            self.assertEqual(
                resolution.pruning_arm,
                "linear_structured",
                msg=f"{architecture_name} must take the Linear-structured pruning arm."
            )

    def test_convolutional_generators_take_the_global_unstructured_arm(self) -> None:
        # Convolution-dominant generators resolve to the global unstructured arm.
        convolution_dominant_names: tuple[str, ...] = (
            "hifigan_v1",
            "hifigan_v2",
            "hifigan_v3",
            "melgan",
            "bigvgan",
            "apnet2",
            "rndvoc",
            "lpcnet"
        )
        architecture_name: str
        for architecture_name in convolution_dominant_names:
            resolution: ArchitectureTechniqueResolution = self._table.get(architecture_name)
            self.assertEqual(
                resolution.pruning_arm,
                "global_unstructured",
                msg=f"{architecture_name} must take the global unstructured pruning arm."
            )

    def test_recurrent_architecture_adds_the_gru_quantization_kind(self) -> None:
        # LPCNet is the only architecture carrying the GRU quantization kind.
        gru_bearing: tuple[str, ...] = tuple(
            resolution.architecture_name
            for resolution in self._table.resolutions
            if "gru" in resolution.int8_module_kinds
        )
        self.assertEqual(gru_bearing, ("lpcnet",))
        self.assertEqual(self._table.get("lpcnet").int8_module_kinds, ("linear", "gru"))

    def test_onnx_export_scope_is_registered_for_the_four_convolutional_rows(self) -> None:
        # ONNX export is registered exactly for the four convolutional architectures.
        exportable: set[str] = {
            resolution.architecture_name
            for resolution in self._table.resolutions
            if resolution.onnx_export_supported
        }
        self.assertEqual(exportable, {"hifigan_v1", "hifigan_v2", "hifigan_v3", "melgan"})

    def test_step_reduction_and_causal_controls_are_narrowly_registered(self) -> None:
        # Step reduction is RFWave only; causal recovery controls are Vocos and RFWave.
        step_reducible: set[str] = {
            resolution.architecture_name
            for resolution in self._table.resolutions
            if resolution.ode_step_reduction_supported
        }
        causal_controlled: set[str] = {
            resolution.architecture_name
            for resolution in self._table.resolutions
            if resolution.causal_control_selected
        }
        self.assertEqual(step_reducible, {"rfwave"})
        self.assertEqual(causal_controlled, {"vocos", "rfwave"})

    def test_half_precision_and_compilation_exclusions_are_recorded(self) -> None:
        # FreeV and LPCNet carry the recorded fp16 exclusion; LPCNet also refuses compilation.
        fp16_excluded: set[str] = {
            resolution.architecture_name
            for resolution in self._table.resolutions
            if not resolution.fp16_supported
        }
        compile_excluded: set[str] = {
            resolution.architecture_name
            for resolution in self._table.resolutions
            if not resolution.compile_supported
        }
        self.assertEqual(fp16_excluded, {"freev", "lpcnet"})
        self.assertEqual(compile_excluded, {"lpcnet"})

    def test_undeclared_architecture_fails_closed(self) -> None:
        # An architecture without a declared row is a configuration error, not a default.
        with self.assertRaisesRegex(MisconfigurationError, "No technique resolution declared"):
            self._table.get("hiftnet")

    def test_resolution_rows_reject_mutation_and_unknown_fields(self) -> None:
        # The resolution record is frozen, closed, and strictly typed.
        resolution: ArchitectureTechniqueResolution = self._table.get("vocos")
        with self.assertRaises(ValidationError):
            resolution.fp16_supported = False
        with self.assertRaises(ValidationError):
            ArchitectureTechniqueResolution(
                architecture_name="vocos",
                pruning_arm="linear_structured",
                int8_module_kinds=("linear",),
                int8_dynamic_supported=True,
                weight_only_supported=True,
                fp16_supported=True,
                compile_supported=True,
                onnx_export_supported=False,
                ode_step_reduction_supported=False,
                causal_control_selected=True,
                unknown_field=1
            )


class VariantApplicabilityDecisionTest(unittest.TestCase):
    # Verifies the per-cell applicability decisions and their exact per-architecture counts.
    def setUp(self) -> None:
        # Binds the registry, the reader, and the variant vocabulary swept per case.
        self._registry: OptimizationVariantRegistry = OptimizationVariantRegistry()
        self._reader: LiteralMemberReader = LiteralMemberReader()
        self._variant_names: tuple[str, ...] = self._reader.read(OptimizationVariantName)

    def test_supported_cell_count_per_architecture_is_exact(self) -> None:
        # The study's coverage map is a registered fact, asserted cell count by cell count.
        # Each count is the six fleet-wide cells (the two baseline lanes, the
        # three masked curve points, and the full-budget recovered arm) plus
        # whatever that architecture's declared support flags add: two for
        # compilation, one each for dynamic and eight-bit weight-only
        # quantization and for half precision, two for the export lanes, three
        # for step reduction, and two for the causal recovery controls. The
        # four-bit cell contributes nowhere. Asserting the whole map at once
        # makes any single flag change surface here rather than in a downstream
        # analysis.
        expected_counts: dict[str, int] = {
            "hifigan_v1": 11,
            "hifigan_v2": 11,
            "hifigan_v3": 11,
            "melgan": 11,
            "vocos": 13,
            "bigvgan": 9,
            "apnet2": 10,
            "freev": 10,
            "hiftnet": 0,
            "lpcnet": 7,
            "rndvoc": 9,
            "vocosformer": 11,
            "rfwave": 16
        }
        measured_counts: dict[str, int] = {
            architecture_name: sum(
                1
                for variant_name in self._variant_names
                if self._registry.variant_support(architecture_name, variant_name).supported
            )
            for architecture_name in self._reader.read(ArchitectureName)
        }
        self.assertEqual(
            measured_counts,
            expected_counts,
            msg="A cell changed applicability; Study 2 coverage is a pre-registered contract."
        )

    def test_every_cell_carries_a_decision_and_a_non_empty_reason(self) -> None:
        # Every (architecture, variant) pair resolves to a recorded decision that
        # echoes the cell it was asked about; the cell count guards the sweep
        # itself against a vocabulary that silently shrank.
        architecture_names: tuple[str, ...] = self._reader.read(ArchitectureName)
        decided_cells: dict[tuple[str, str], OptimizationVariantSupport] = {
            (architecture_name, variant_name): self._registry.variant_support(
                architecture_name,
                variant_name
            )
            for architecture_name in architecture_names
            for variant_name in self._variant_names
        }
        self.assertEqual(
            len(decided_cells),
            len(architecture_names) * len(self._variant_names)
        )
        cell_key: tuple[str, str]
        support: OptimizationVariantSupport
        for cell_key, support in decided_cells.items():
            self.assertEqual((support.architecture_name, support.variant_name), cell_key)
            self.assertTrue(support.reason, msg=f"Cell {cell_key} recorded an empty reason.")

    def test_supported_cells_record_the_registered_reason(self) -> None:
        # A supported cell records the registered marker rather than prose.
        support: OptimizationVariantSupport = self._registry.variant_support("vocos", "int8_dynamic")
        self.assertTrue(support.supported)
        self.assertEqual(support.reason, "registered")

    def test_excluded_architecture_is_unsupported_in_every_cell(self) -> None:
        # HiFTNet is the documented non-executed exclusion and enters no cell.
        variant_name: str
        for variant_name in self._variant_names:
            support: OptimizationVariantSupport = self._registry.variant_support("hiftnet", variant_name)
            self.assertFalse(support.supported, msg=f"HiFTNet must not enter cell {variant_name}.")
            self.assertIn("documented non-executed exclusion", support.reason)

    def test_baseline_and_masked_pruning_cells_are_supported_everywhere_in_the_table(self) -> None:
        # Baselines and the masked pruning curve apply to every declared architecture.
        fleet_wide_names: tuple[str, ...] = (
            "baseline_cpu",
            "baseline_b200",
            "pruned_30",
            "pruned_50",
            "pruned_70",
            "pruned_50_recovered"
        )
        architecture_name: str
        for architecture_name in self._reader.read(ArchitectureName):
            if architecture_name == "hiftnet":
                continue
            variant_name: str
            for variant_name in fleet_wide_names:
                support: OptimizationVariantSupport = self._registry.variant_support(
                    architecture_name,
                    variant_name
                )
                self.assertTrue(
                    support.supported,
                    msg=f"({architecture_name}, {variant_name}) must be a registered cell."
                )

    def test_lower_bit_weight_only_cell_is_unsupported_everywhere(self) -> None:
        # INT4 weight-only is unsupported for every architecture under the evaluated image.
        architecture_name: str
        for architecture_name in self._reader.read(ArchitectureName):
            support: OptimizationVariantSupport = self._registry.variant_support(
                architecture_name,
                "int4_weight_only"
            )
            self.assertFalse(support.supported)

    def test_step_reduction_cells_are_supported_only_for_the_flow_architecture(self) -> None:
        # Sampling-step reduction applies only to the iterative-flow architecture.
        variant_name: str
        for variant_name in ("ode_steps_8", "ode_steps_4", "ode_steps_2"):
            self.assertTrue(self._registry.variant_support("rfwave", variant_name).supported)
            self.assertFalse(self._registry.variant_support("vocos", variant_name).supported)
            self.assertFalse(self._registry.variant_support("hifigan_v1", variant_name).supported)

    def test_causal_control_cells_are_supported_only_for_the_selected_architectures(self) -> None:
        # The recovery-budget controls exist for the two selected architectures only.
        variant_name: str
        for variant_name in ("dense_continued", "pruned_50_recovered_half"):
            self.assertTrue(self._registry.variant_support("vocos", variant_name).supported)
            self.assertTrue(self._registry.variant_support("rfwave", variant_name).supported)
            self.assertFalse(self._registry.variant_support("hifigan_v1", variant_name).supported)
            self.assertFalse(self._registry.variant_support("lpcnet", variant_name).supported)


class UnsupportedReasonExposureTest(unittest.TestCase):
    # Verifies that every unsupported cell exposes its recorded scientific reason.
    # Each case targets one distinct cause and asserts a substring carrying that
    # cause's technical content rather than the whole sentence, so the wording
    # may be revised while a reason that loses its substance still fails. The
    # causes are deliberately kept apart: two architectures may both be
    # unsupported for one variant for entirely different scientific reasons, and
    # collapsing them would erase exactly the evidence the study reports.
    def setUp(self) -> None:
        # Binds the registry whose recorded refusal reasons are read.
        self._registry: OptimizationVariantRegistry = OptimizationVariantRegistry()

    def test_spectral_head_architectures_record_the_missing_operator_mapping(self) -> None:
        # The inverse short-time Fourier head has no ONNX operator mapping.
        architecture_name: str
        for architecture_name in ("vocos", "freev", "vocosformer", "apnet2", "rfwave"):
            support: OptimizationVariantSupport = self._registry.variant_support(
                architecture_name,
                "onnx_fp32"
            )
            self.assertFalse(support.supported)
            self.assertIn("torch.istft", support.reason)

    def test_autoregressive_architecture_records_the_non_static_graph_reason(self) -> None:
        # The autoregressive sampling loop is not a static exportable graph.
        support: OptimizationVariantSupport = self._registry.variant_support("lpcnet", "onnx_int8_static")
        self.assertFalse(support.supported)
        self.assertIn("autoregressive Python sampling loop", support.reason)

    def test_unattempted_export_records_the_presentation_budget_reason(self) -> None:
        # Architectures outside the export scope record an unvalidated operator graph.
        support: OptimizationVariantSupport = self._registry.variant_support("bigvgan", "onnx_fp32")
        self.assertFalse(support.supported)
        self.assertIn("Export was not attempted", support.reason)

    def test_compilation_exclusion_records_the_uncovered_synthesis_path(self) -> None:
        # Module-level compilation never covers the executed autoregressive path.
        support: OptimizationVariantSupport = self._registry.variant_support("lpcnet", "torch_compile")
        self.assertFalse(support.supported)
        self.assertIn("module-level", support.reason)

    def test_dynamic_int8_exclusion_records_measured_zero_coverage(self) -> None:
        # Dynamic INT8 measured zero executable coverage on convolution-dominant generators.
        support: OptimizationVariantSupport = self._registry.variant_support("hifigan_v1", "int8_dynamic")
        self.assertFalse(support.supported)
        self.assertIn("zero executable coverage", support.reason)

    def test_lower_bit_exclusion_records_the_missing_kernel_package(self) -> None:
        # On weight-only capable architectures the INT4 refusal is the missing kernel package.
        support: OptimizationVariantSupport = self._registry.variant_support("vocos", "int4_weight_only")
        self.assertFalse(support.supported)
        self.assertIn("mslk", support.reason)

    def test_lower_bit_exclusion_falls_back_to_the_operator_profile_reason(self) -> None:
        # Without Linear coverage the INT4 refusal is the operator profile, not the kernel.
        support: OptimizationVariantSupport = self._registry.variant_support("hifigan_v1", "int4_weight_only")
        self.assertFalse(support.supported)
        self.assertIn("Linear stratum", support.reason)
        self.assertNotIn("mslk", support.reason)

    def test_half_precision_exclusion_records_the_measured_dtype_incompatibility(self) -> None:
        # The fp16 exclusion is a measured screening outcome.
        support: OptimizationVariantSupport = self._registry.variant_support("freev", "fp16_weights")
        self.assertFalse(support.supported)
        self.assertIn("dtype incompatibility", support.reason)

    def test_step_reduction_exclusion_records_the_architecture_restriction(self) -> None:
        # Step reduction is meaningless outside the iterative-flow architecture.
        support: OptimizationVariantSupport = self._registry.variant_support("vocos", "ode_steps_4")
        self.assertFalse(support.supported)
        self.assertIn("iterative-flow architecture", support.reason)

    def test_causal_control_exclusion_records_the_bounded_recovery_claim(self) -> None:
        # Unselected architectures keep their recovery claims bounded to one checkpoint.
        support: OptimizationVariantSupport = self._registry.variant_support("melgan", "dense_continued")
        self.assertFalse(support.supported)
        self.assertIn("one recovered", support.reason)

    def test_support_records_reject_mutation(self) -> None:
        # The applicability decision is an immutable record.
        support: OptimizationVariantSupport = self._registry.variant_support("vocos", "baseline_cpu")
        with self.assertRaises(ValidationError):
            support.supported = False


class SupportedHardwareLaneTest(unittest.TestCase):
    # Verifies the registered measurement lane resolved for every variant name.
    def setUp(self) -> None:
        # Binds the registry and the reader enumerating every variant.
        self._registry: OptimizationVariantRegistry = OptimizationVariantRegistry()
        self._reader: LiteralMemberReader = LiteralMemberReader()

    def test_processor_lane_variants_resolve_to_the_processor_lane(self) -> None:
        # Deployment and dynamic-INT8 lanes are processor lanes by registration.
        variant_name: str
        for variant_name in ("baseline_cpu", "int8_dynamic", "onnx_fp32", "onnx_int8_static"):
            self.assertEqual(self._registry.supported_hardware(variant_name), ("cpu",))

    def test_accelerator_lane_variants_resolve_to_the_single_accelerator_lane(self) -> None:
        # The compiled, weight-only, half-precision, masked, and step-reduced lanes
        # are accelerator lanes.
        accelerator_lane_names: tuple[str, ...] = (
            "baseline_b200",
            "fp16_weights",
            "torch_compile",
            "torch_compile_overhead",
            "int8_weight_only",
            "int4_weight_only",
            "pruned_30",
            "pruned_50",
            "pruned_70",
            "ode_steps_8",
            "ode_steps_4",
            "ode_steps_2"
        )
        variant_name: str
        for variant_name in accelerator_lane_names:
            self.assertEqual(self._registry.supported_hardware(variant_name), ("b200",))

    def test_recovery_variants_resolve_to_two_registered_lanes(self) -> None:
        # The recovery arms carry a second registered lane.
        variant_name: str
        for variant_name in ("pruned_50_recovered", "pruned_50_recovered_half", "dense_continued"):
            self.assertEqual(self._registry.supported_hardware(variant_name), ("b200", "l40s"))

    def test_every_variant_resolves_a_non_empty_lane_tuple(self) -> None:
        # No variant may reach measurement without a registered lane.
        variant_name: str
        for variant_name in self._reader.read(OptimizationVariantName):
            lanes: tuple[str, ...] = self._registry.supported_hardware(variant_name)
            self.assertTrue(lanes, msg=f"Variant {variant_name} resolved no measurement lane.")


class VariantRecordConstructionTest(unittest.TestCase):
    # Verifies the technique instance and note bound into each supported variant record.
    def setUp(self) -> None:
        # Binds the registry and the reader enumerating every cell.
        self._registry: OptimizationVariantRegistry = OptimizationVariantRegistry()
        self._reader: LiteralMemberReader = LiteralMemberReader()

    def test_unsupported_cell_refuses_construction_with_its_reason(self) -> None:
        # An unsupported cell fails closed and carries its recorded reason into the failure.
        with self.assertRaisesRegex(MisconfigurationError, "torch.istft"):
            self._registry.get("vocos", "onnx_fp32")
        with self.assertRaisesRegex(MisconfigurationError, "documented non-executed exclusion"):
            self._registry.get("hiftnet", "baseline_cpu")

    def test_every_supported_cell_builds_a_named_record(self) -> None:
        # Every supported cell constructs a record whose fields match the requested cell.
        # Sweeping the whole matrix also exercises the registry's stated import
        # guarantee: constructing every supported cell must not pull an optional
        # backend into this process, and a technique module that imported one at
        # module scope would surface here as a collection error.
        architecture_name: str
        for architecture_name in self._reader.read(ArchitectureName):
            variant_name: str
            for variant_name in self._reader.read(OptimizationVariantName):
                if not self._registry.variant_support(architecture_name, variant_name).supported:
                    continue
                record: OptimizationVariantRecord = self._registry.get(architecture_name, variant_name)
                self.assertEqual(record.variant_name, variant_name)
                self.assertEqual(record.base_architecture, architecture_name)
                self.assertTrue(record.measurement_note)
                self.assertIsInstance(record.technique, OptimizationTechnique)

    def test_baseline_cells_build_the_null_technique_on_their_own_lane(self) -> None:
        # Both baseline lanes build the identity technique under their own variant name.
        processor_record: OptimizationVariantRecord = self._registry.get("vocos", "baseline_cpu")
        accelerator_record: OptimizationVariantRecord = self._registry.get("vocos", "baseline_b200")
        self.assertIsInstance(processor_record.technique, BaselineIdentity)
        self.assertIsInstance(accelerator_record.technique, BaselineIdentity)
        self.assertEqual(processor_record.technique.name, "baseline_cpu")
        self.assertEqual(accelerator_record.technique.name, "baseline_b200")

    def test_compilation_cells_build_the_two_registered_compiler_configurations(self) -> None:
        # The two compilation cells differ exactly by compiler mode.
        default_technique: OptimizationTechnique = self._registry.get("vocos", "torch_compile").technique
        overhead_technique: OptimizationTechnique = self._registry.get("vocos", "torch_compile_overhead").technique
        self.assertIsInstance(default_technique, TorchCompileAcceleration)
        self.assertIsInstance(overhead_technique, TorchCompileAcceleration)
        self.assertEqual(default_technique.configuration.mode, "default")
        self.assertEqual(overhead_technique.configuration.mode, "reduce-overhead")

    def test_dynamic_quantization_cell_receives_the_architecture_resolved_module_kinds(self) -> None:
        # The recurrent architecture receives both quantized module kinds.
        recurrent_technique: OptimizationTechnique = self._registry.get("lpcnet", "int8_dynamic").technique
        linear_technique: OptimizationTechnique = self._registry.get("vocos", "int8_dynamic").technique
        self.assertIsInstance(recurrent_technique, DynamicInt8Quantization)
        self.assertEqual(recurrent_technique.configuration.quantized_module_kinds, ("linear", "gru"))
        self.assertEqual(linear_technique.configuration.quantized_module_kinds, ("linear",))

    def test_weight_only_and_half_precision_cells_build_their_techniques(self) -> None:
        # The weight-only cell carries its bit width; the storage cell is the fp16 technique.
        weight_only_technique: OptimizationTechnique = self._registry.get("vocos", "int8_weight_only").technique
        storage_technique: OptimizationTechnique = self._registry.get("vocos", "fp16_weights").technique
        self.assertIsInstance(weight_only_technique, WeightOnlyIntQuantization)
        self.assertEqual(weight_only_technique.configuration.bit_width, 8)
        self.assertIsInstance(storage_technique, WeightOnlyFp16Storage)

    def test_deployment_cells_build_the_runtime_technique_with_its_quantization_switch(self) -> None:
        # The static INT8 deployment cell differs from the fp32 cell by its switch.
        fp32_technique: OptimizationTechnique = self._registry.get("hifigan_v1", "onnx_fp32").technique
        int8_technique: OptimizationTechnique = self._registry.get("hifigan_v1", "onnx_int8_static").technique
        self.assertIsInstance(fp32_technique, OnnxRuntimeDeployment)
        self.assertFalse(fp32_technique.configuration.static_int8)
        self.assertTrue(int8_technique.configuration.static_int8)

    def test_pruning_curve_cells_carry_their_level_and_resolved_arm(self) -> None:
        # Each curve point binds its registered sparsity level and architecture-resolved arm.
        expected_levels: dict[str, float] = {"pruned_30": 0.3, "pruned_50": 0.5, "pruned_70": 0.7}
        variant_name: str
        expected_level: float
        for variant_name, expected_level in expected_levels.items():
            structured_technique: OptimizationTechnique = self._registry.get("vocos", variant_name).technique
            unstructured_technique: OptimizationTechnique = self._registry.get("hifigan_v1", variant_name).technique
            self.assertIsInstance(structured_technique, MaskedMagnitudePruning)
            self.assertEqual(structured_technique.configuration_dump()["pruning_amount"], expected_level)
            self.assertEqual(structured_technique.configuration_dump()["pruning_arm"], "linear_structured")
            self.assertEqual(unstructured_technique.configuration_dump()["pruning_arm"], "global_unstructured")

    def test_recovery_cells_build_verification_and_control_techniques(self) -> None:
        # The recovered arms verify a loaded checkpoint; the control proves density.
        recovered_technique: OptimizationTechnique = self._registry.get("vocos", "pruned_50_recovered").technique
        half_technique: OptimizationTechnique = self._registry.get("vocos", "pruned_50_recovered_half").technique
        control_technique: OptimizationTechnique = self._registry.get("vocos", "dense_continued").technique
        self.assertIsInstance(recovered_technique, PrunedRecoveredVerification)
        self.assertIsInstance(half_technique, PrunedRecoveredVerification)
        self.assertIsInstance(control_technique, DenseContinuedIdentity)
        self.assertEqual(recovered_technique.name, "pruned_50_recovered")
        self.assertEqual(half_technique.name, "pruned_50_recovered_half")
        self.assertEqual(control_technique.name, "dense_continued")

    def test_step_reduction_cells_carry_their_registered_step_counts(self) -> None:
        # The three step-reduction cells decode their step count from the variant name.
        expected_steps: dict[str, int] = {"ode_steps_8": 8, "ode_steps_4": 4, "ode_steps_2": 2}
        variant_name: str
        expected_step_count: int
        for variant_name, expected_step_count in expected_steps.items():
            technique: OptimizationTechnique = self._registry.get("rfwave", variant_name).technique
            self.assertIsInstance(technique, OdeStepReduction)
            self.assertEqual(technique.configuration.step_count, expected_step_count)
            self.assertEqual(technique.name, variant_name)

    def test_measurement_notes_state_the_lane_interpretation(self) -> None:
        # The note recorded into the capsule states how the lane must be read.
        self.assertIn("denominator", self._registry.get("vocos", "baseline_cpu").measurement_note)
        self.assertIn("same-stack denominator", self._registry.get("vocos", "baseline_b200").measurement_note)
        self.assertIn("no speedup expected", self._registry.get("vocos", "pruned_50").measurement_note)
        self.assertIn("causal control", self._registry.get("vocos", "dense_continued").measurement_note)
        self.assertIn("Reduced ODE sampling schedule", self._registry.get("rfwave", "ode_steps_2").measurement_note)

    def test_pruning_curve_points_share_one_note_and_baselines_do_not(self) -> None:
        # The curve points read identically; the two baselines are distinct denominators.
        curve_notes: set[str] = {
            self._registry.get("vocos", variant_name).measurement_note
            for variant_name in ("pruned_30", "pruned_50", "pruned_70")
        }
        self.assertEqual(len(curve_notes), 1)
        self.assertNotEqual(
            self._registry.get("vocos", "baseline_cpu").measurement_note,
            self._registry.get("vocos", "baseline_b200").measurement_note
        )

    def test_variant_records_reject_mutation_and_unknown_fields(self) -> None:
        # The variant record is a frozen, closed binding.
        record: OptimizationVariantRecord = self._registry.get("vocos", "baseline_cpu")
        with self.assertRaises(ValidationError):
            record.variant_name = "pruned_50"
        with self.assertRaises(ValidationError):
            OptimizationVariantRecord(
                variant_name="baseline_cpu",
                base_architecture="vocos",
                technique=BaselineIdentity(),
                measurement_note="note",
                unknown_field=1
            )


class BaselineIdentityTest(unittest.TestCase):
    # Verifies the null technique that measures every denominator under the shared protocol.
    def setUp(self) -> None:
        # Seeds construction and binds the null technique with its subject module.
        torch.manual_seed(0)
        self._technique: BaselineIdentity = BaselineIdentity()
        self._module: TinyHarnessModule = TinyHarnessModule(TinyLinearNetwork())

    def test_default_identity_produces_the_processor_baseline_name(self) -> None:
        # The default identity names the processor baseline lane.
        self.assertEqual(self._technique.name, "baseline_cpu")

    def test_accelerator_identity_produces_the_declared_lane_name(self) -> None:
        # The identity row names whichever baseline lane it was constructed for.
        self.assertEqual(BaselineIdentity(produced_variant_name="baseline_b200").name, "baseline_b200")

    def test_apply_returns_the_module_untransformed(self) -> None:
        # The baseline is the identity transformation by definition.
        original_network: nn.Module = self._module.network
        returned: Module = self._technique.apply(self._module)
        self.assertIs(returned, self._module)
        self.assertIs(returned.network, original_network)

    def test_configuration_dump_records_the_identity_and_its_lane(self) -> None:
        # The recipe records the null technique and the baseline lane it produced.
        self.assertEqual(
            self._technique.configuration_dump(),
            {"technique": "baseline_identity", "produced_variant_name": "baseline_cpu"}
        )


class OptimizationTechniqueContractTest(unittest.TestCase):
    # Verifies that the base technique contract refuses use without an implementation.
    def setUp(self) -> None:
        # Binds the unimplemented base contract and a module it must refuse.
        self._technique: OptimizationTechnique = OptimizationTechnique()
        self._module: TinyHarnessModule = TinyHarnessModule(TinyLinearNetwork())

    def test_unimplemented_name_is_refused(self) -> None:
        # The base contract carries no variant name.
        with self.assertRaises(NotImplementedError):
            self._technique.name

    def test_unimplemented_apply_is_refused(self) -> None:
        # The base contract performs no transformation.
        with self.assertRaises(NotImplementedError):
            self._technique.apply(self._module)

    def test_unimplemented_configuration_dump_is_refused(self) -> None:
        # The base contract exposes no recipe configuration.
        with self.assertRaises(NotImplementedError):
            self._technique.configuration_dump()


if __name__ == "__main__":
    unittest.main()
