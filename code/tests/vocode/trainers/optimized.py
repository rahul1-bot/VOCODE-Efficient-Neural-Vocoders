# This module:
# 1. Verifies OptimizedVariantEvaluator construction and the one-variant-per-
#    evaluator responsibility boundary: each evaluator opens its own variant
#    registry, resolves exactly one declared variant, and binds it in a frozen
#    spec that cannot be substituted after construction
# 2. Verifies the guards that run in front of measurement: the test-stage gate,
#    the required variant name, the registered hardware-lane check, the
#    required base checkpoint, the checkpoint payload check, and the
#    transformed-execution assertion
# 3. Verifies the evidence the capsule writes and the Trainers it builds: the
#    variant-level and run-level optimization recipes, the base-checkpoint
#    SHA-256, peak-memory metrics, evidence identity, the metrics snapshot, and
#    the prediction and test callback wiring
#
# Design decisions:
# - A harness entry point is replaced by a recording stand-in inside a scoped
#   context manager, so the Trainer the evaluator actually builds is inspected
#   while no prediction or test loop ever executes; the original entry point is
#   restored on both the success and the failure path
# - Evaluation subjects are assembled directly as OptimizedVariantModuleSpec
#   records around a registry variant record and a minimal harness Module stub,
#   so no reference architecture is built and no checkpoint weights are loaded
# - The peak-accelerator assertion is expressed against torch.cuda.is_available
#   rather than a fixed expectation, so it states the same contract on a CPU
#   runner and on an accelerator host
# - Boundaries excluded: _build_module_spec, _build_base_module, and the
#   post-guard body of run(), because they construct a reference architecture
#   through the model registry, apply a technique to trained weights, and then
#   drive harness loops over the real corpus
#
# Author: Rahul Sawhney

import hashlib
import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

import torch
import yaml
from torch import nn

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.runtime_profiler import RuntimeProfiler
from syntheticmind.core.module import Module
from syntheticmind.core.trainer import Trainer
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.types import HyperparameterDict

from vocode.configs.layout import EvidenceCategory, ExperimentArtifactLayout, ExperimentStage
from vocode.configs.run import ExperimentConfiguration
from vocode.data.ljspeech_datamodule import LJSpeechDataConfig, LJSpeechDataModule
from vocode.loggers.experiment import ExperimentLogger
from vocode.metrics.registry import MetricSelection
from vocode.metrics.rtf import RealTimeFactorConfig, RealTimeFactorMonitor
from vocode.metrics.sequence import MetricSequence
from vocode.optimization.registry import (
    BaselineIdentity,
    OptimizationVariantName,
    OptimizationVariantRecord,
    OptimizationVariantRegistry,
)
from vocode.trainers.optimized import OptimizedVariantEvaluator, OptimizedVariantModuleSpec


class NetworkProvider:
    # Callable stand-in resolving the network object a sampler would synthesize
    # with, matching the provider seam the transformed-execution assertion reads.
    # It is a callable class rather than a closure because the assertion tests
    # the seam for callability before invoking it, and a class states that
    # requirement in the stub's own shape.
    def __init__(self, network: nn.Module) -> None:
        # Binds the network object this provider resolves.
        #
        # Args:
        #     network: The object this provider will resolve, which a case
        #         sets either to the module's own network or to a different
        #         one.
        self._network: nn.Module = network

    def __call__(self) -> nn.Module:
        # Resolves the bound network, matching the provider seam of a real sampler.
        #
        # Returns:
        #     The bound network object.
        return self._network


class StubSampler:
    # Sampler stand-in exposing the network provider the assertion inspects.
    # It carries the private provider attribute and nothing else, because that
    # attribute is the entire surface the assertion reads.
    def __init__(self, network_provider: NetworkProvider) -> None:
        # Binds the provider the transformed-execution assertion reads.
        #
        # Args:
        #     network_provider: The provider whose resolution the assertion
        #         compares against the module's current network.
        self._network_provider: NetworkProvider = network_provider


class StubTransformedModule(Module):
    # Harness Module stub whose sampler resolves the module's current network,
    # which is the state a correctly transformed capsule is in.
    def __init__(self) -> None:
        # Builds the network and a sampler whose provider resolves that same object.
        super().__init__()
        self.network: nn.Module = nn.Identity()
        self._sampler: StubSampler = StubSampler(NetworkProvider(self.network))


class StubStaleSamplerModule(Module):
    # Harness Module stub whose sampler retains a pre-transformation network,
    # which is the defect the transformed-execution assertion must catch.
    # It stands for the real failure mode: a technique that replaces the network
    # object rather than mutating it in place leaves any sampler holding the
    # original, and the capsule would then time one model while scoring another.
    def __init__(self) -> None:
        # Builds the network and a sampler retaining a different, stale object.
        # The two objects are of the same type and differ only in identity,
        # so the case proves the assertion compares identity rather than kind.
        super().__init__()
        self.network: nn.Module = nn.Identity()
        self._sampler: StubSampler = StubSampler(NetworkProvider(nn.Identity()))


class StubSamplerlessModule(Module):
    # Harness Module stub without a provider-backed sampler, which is the
    # single-network case the assertion admits unconditionally.
    def __init__(self) -> None:
        # Builds the network alone, with no sampler seam to diverge from.
        super().__init__()
        self.network: nn.Module = nn.Identity()


class OptimizedRunConfigurationFactory:
    # Builds valid optimized-variant ExperimentConfiguration records rooted in
    # one temporary artifact tree, with distinct batch limits per stage so
    # forwarding can be told apart from cross-wiring.
    #
    # Integration: the test and prediction limits differ from one another, so an
    # assertion that a stage's Trainer carries the right limit proves the
    # evaluator forwarded that stage's own value rather than one that merely
    # happens to match. Every configuration is the real validated record and
    # every path descends from the per-case temporary root. The layout is fixed
    # to the processor lane, which makes the accelerator-lane baseline the
    # natural way to reach the hardware-lane refusal without inventing an
    # unregistered lane name.
    def __init__(self, temporary_root: Path, base_checkpoint_path: Path) -> None:
        # Binds the temporary root and the base checkpoint every configuration cites.
        #
        # Args:
        #     temporary_root: The per-case temporary directory all built paths
        #         descend from.
        #     base_checkpoint_path: The written checkpoint the optimized
        #         configurations cite as their base identity.
        self._temporary_root: Path = temporary_root
        self._base_checkpoint_path: Path = base_checkpoint_path

    def build_variant_run(self, variant_name: OptimizationVariantName) -> ExperimentConfiguration:
        # Builds a test-stage run description for the requested variant.
        return self._compose_optimized("test", variant_name, None)

    def build_variant_run_for_stage(
        self,
        variant_name: OptimizationVariantName,
        stage: ExperimentStage
    ) -> ExperimentConfiguration:
        # Builds a run description for the requested variant on the requested stage.
        return self._compose_optimized(stage, variant_name, None)

    def build_variant_run_without_real_time_factor(
        self,
        variant_name: OptimizationVariantName
    ) -> ExperimentConfiguration:
        # Builds a run description whose metric panel omits the timing metric.
        return self._compose_optimized("test", variant_name, MetricSelection(names=("pesq", "mel")))

    def build_run_without_optimization_provenance(self) -> ExperimentConfiguration:
        # Builds a reproduction run description, which cites neither variant nor
        # base checkpoint. This is the only way to reach both the missing-variant
        # and the missing-checkpoint refusals, because the optimized evidence
        # category requires those fields to validate.
        #
        # Returns:
        #     A validated reproduction run description carrying no
        #     optimization provenance at all.
        layout: ExperimentArtifactLayout = self._compose_layout(
            "project_trained_reproduction",
            None
        )
        return self._compose(
            layout,
            "project_trained_reproduction",
            "test",
            None,
            None,
            MetricSelection()
        )

    def _compose_optimized(
        self,
        stage: ExperimentStage,
        variant_name: OptimizationVariantName,
        metric_selection: MetricSelection | None
    ) -> ExperimentConfiguration:
        # Composes an optimized-variant run description around the varied fields.
        layout: ExperimentArtifactLayout = self._compose_layout(
            "project_optimized_variants",
            variant_name
        )
        resolved_metric_selection: MetricSelection = (
            metric_selection if metric_selection is not None else MetricSelection()
        )
        return self._compose(
            layout,
            "project_optimized_variants",
            stage,
            variant_name,
            self._base_checkpoint_path,
            resolved_metric_selection
        )

    def _compose_layout(
        self,
        evidence_category: EvidenceCategory,
        variant_name: str | None
    ) -> ExperimentArtifactLayout:
        # Composes the artifact layout of the requested evidence category.
        return ExperimentArtifactLayout(
            artifact_root=self._temporary_root / "artifacts",
            evidence_category=evidence_category,
            architecture_name="hifigan_v1",
            hardware_name="cpu",
            precision_name="fp32",
            seed=5,
            run_id="run_optimized",
            variant_name=variant_name
        )

    def _compose(
        self,
        layout: ExperimentArtifactLayout,
        evidence_category: EvidenceCategory,
        stage: ExperimentStage,
        variant_name: str | None,
        project_checkpoint_path: Path | None,
        metric_selection: MetricSelection
    ) -> ExperimentConfiguration:
        # Composes the full run description from the layout and the varied fields.
        # The hypothesis identifier and the commit hash are present exactly when
        # a variant is declared, because those two fields are optimization
        # provenance and a reproduction run carries neither.
        #
        # Args:
        #     layout: The artifact layout of the requested evidence category.
        #     evidence_category: The evidence family the run declares.
        #     stage: The stage the run declares, including ones the gate must
        #         refuse.
        #     variant_name: The optimization variant, absent on the
        #         reproduction description.
        #     project_checkpoint_path: The base checkpoint, absent on the
        #         reproduction description.
        #     metric_selection: The metric panel deciding whether the timing
        #         monitor enters the prediction pass.
        #
        # Returns:
        #     A validated run description anchored in the temporary root.
        return ExperimentConfiguration(
            experiment_name="vocode_optimization",
            run_id="run_optimized",
            hypothesis="Registered techniques trade quality for synthesis speed.",
            interpretation_notes="Synthetic configuration used for wiring verification.",
            evidence_category=evidence_category,
            stage=stage,
            dataset_split_name=stage,
            architecture_name="hifigan_v1",
            seed=5,
            artifact_layout=layout,
            published_weights_root=self._temporary_root / "published_weights",
            project_checkpoint_path=project_checkpoint_path,
            optimization_variant_name=variant_name,
            optimization_hypothesis_id="H2" if variant_name is not None else None,
            code_commit_hash="0f1e2d3c" if variant_name is not None else None,
            data_configuration=LJSpeechDataConfig(dataset_root=self._temporary_root / "corpus"),
            metric_selection=metric_selection,
            real_time_factor_configuration=RealTimeFactorConfig(),
            limit_test_batches=8,
            limit_predict_batches=9,
            accelerator="cpu",
            runtime_profiling_enabled=True,
            runtime_profile_interval_steps=30
        )


class TrainerEntryPointInterception:
    # Scoped replacement of one harness Trainer entry point with a recording
    # stand-in, capturing the Trainer instances an evaluator builds while
    # keeping every harness loop unexecuted.
    #
    # Integration: this is what makes the evaluator's Trainer assembly observable
    # without a corpus. The evaluator is driven for real, so it builds its
    # callbacks and its Trainer exactly as it would in production, and only the
    # final loop invocation is replaced. The replacement is made on the Trainer
    # class itself and is therefore process-wide while the block is open, so the
    # original is restored in the exit path on both the success and the failure
    # branch.
    def __init__(self, entry_point_name: str) -> None:
        # Retains the original entry point and opens the recording list.
        # The original is captured at construction rather than at entry, so a
        # nested or repeated use restores what was actually replaced.
        #
        # Args:
        #     entry_point_name: The harness entry point to intercept, which on
        #         this path is the predict or the test method.
        self._entry_point_name: str = entry_point_name
        self._original_entry_point: Callable[..., object] = getattr(Trainer, entry_point_name)
        self._recorded_trainers: list[Trainer] = []

    @property
    def recorded_trainer(self) -> Trainer:
        # Returns the Trainer captured by the first recorded invocation.
        # A case that reads this without an invocation having happened fails on
        # the empty list, which is the correct outcome: it means the dispatch
        # under test never reached its entry point.
        return self._recorded_trainers[0]

    @property
    def recorded_trainer_count(self) -> int:
        # Returns how many invocations were recorded. Asserting the count is
        # what proves a dispatch method built one Trainer rather than several.
        return len(self._recorded_trainers)

    def __enter__(self) -> TrainerEntryPointInterception:
        # Installs the recording stand-in in place of the harness entry point.
        # The recording list is bound into the closure rather than reached
        # through the instance, because the stand-in is installed on the class
        # and receives the calling Trainer as its first argument, not this
        # interception.
        recorded_trainers: list[Trainer] = self._recorded_trainers

        def record_invocation(
            trainer: Trainer,
            *invocation_arguments: object,
            **keyword_arguments: object
        ) -> None:
            # Records the calling Trainer and discards the invocation arguments.
            del invocation_arguments, keyword_arguments
            recorded_trainers.append(trainer)

        setattr(Trainer, self._entry_point_name, record_invocation)
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback_object: TracebackType | None
    ) -> None:
        # Restores the original entry point on both the success and failure path.
        del exception_type, exception_value, traceback_object
        setattr(Trainer, self._entry_point_name, self._original_entry_point)


class OptimizedVariantEvaluatorConstructionTest(unittest.TestCase):
    # Verifies construction and the one-variant-per-evaluator responsibility
    # boundary: private registry, private counter, single resolved variant.
    def setUp(self) -> None:
        # Writes a well-formed base checkpoint and binds one evaluator on the
        # processor baseline lane.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._temporary_root: Path = Path(self._temporary_directory.name)
        self._base_checkpoint_path: Path = self._temporary_root / "base.ckpt"
        torch.save({"model_state_dict": {}}, self._base_checkpoint_path)
        self._factory: OptimizedRunConfigurationFactory = OptimizedRunConfigurationFactory(
            self._temporary_root,
            self._base_checkpoint_path
        )
        self._configuration: ExperimentConfiguration = self._factory.build_variant_run(
            "baseline_cpu"
        )
        self._evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(self._configuration)

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_construction_writes_no_rows(self) -> None:
        # A freshly constructed evaluator has produced no experiment rows.
        self.assertEqual(
            self._evaluator.row_count,
            0,
            msg="Construction must not write evidence rows"
        )

    def test_construction_creates_no_artifact_directories(self) -> None:
        # Construction alone leaves the artifact tree untouched.
        self.assertFalse(self._configuration.run_directory.exists())

    def test_each_evaluator_opens_its_own_variant_registry(self) -> None:
        # One evaluator owns one optimization registry; no state is shared.
        other_evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(self._configuration)
        self.assertIsInstance(self._evaluator._registry, OptimizationVariantRegistry)
        self.assertIsInstance(other_evaluator._registry, OptimizationVariantRegistry)
        self.assertIsNot(other_evaluator._registry, self._evaluator._registry)

    def test_evaluator_resolves_exactly_one_declared_variant(self) -> None:
        # The evaluator's variant identity is fixed by its configuration.
        self.assertEqual(self._evaluator._resolve_variant_name(), "baseline_cpu")
        self.assertEqual(self._evaluator._resolve_variant_name(), "baseline_cpu")

    def test_two_evaluators_keep_their_own_variant_identity(self) -> None:
        # Two capsules in one process never share a resolved variant name.
        other_evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(
            self._factory.build_variant_run("int8_dynamic")
        )
        self.assertEqual(self._evaluator._resolve_variant_name(), "baseline_cpu")
        self.assertEqual(other_evaluator._resolve_variant_name(), "int8_dynamic")


class OptimizedVariantModuleSpecTest(unittest.TestCase):
    # Verifies the frozen binding of the transformed module to its registry
    # record and the identity of the checkpoint it came from.
    def setUp(self) -> None:
        # Binds a registry record and the exact module bound into the spec under test.
        self._record: OptimizationVariantRecord = OptimizationVariantRegistry().get(
            "hifigan_v1",
            "baseline_cpu"
        )
        self._module: StubTransformedModule = StubTransformedModule()
        self._module_spec: OptimizedVariantModuleSpec = OptimizedVariantModuleSpec(
            record=self._record,
            module=self._module,
            base_checkpoint_path=Path("checkpoints/base.ckpt"),
            base_checkpoint_sha256="1" * 64
        )

    def test_spec_binds_record_module_and_checkpoint_identity(self) -> None:
        # The exact measured module travels with the record and checkpoint it came from.
        self.assertIs(self._module_spec.record, self._record)
        self.assertIs(self._module_spec.module, self._module)
        self.assertEqual(self._module_spec.base_checkpoint_path, Path("checkpoints/base.ckpt"))
        self.assertEqual(self._module_spec.base_checkpoint_sha256, "1" * 64)

    def test_spec_refuses_post_construction_substitution(self) -> None:
        # Freezing prevents swapping what was transformed for what gets measured.
        with self.assertRaises(ValueError):
            self._module_spec.module: Module = StubTransformedModule()

    def test_spec_refuses_unknown_fields(self) -> None:
        # The closed record rejects keys that would carry unvalidated state.
        with self.assertRaises(ValueError):
            OptimizedVariantModuleSpec(
                record=self._record,
                module=StubTransformedModule(),
                base_checkpoint_path=Path("checkpoints/base.ckpt"),
                base_checkpoint_sha256="1" * 64,
                speedup_ratio=2.0
            )

    def test_baseline_record_carries_the_identity_technique(self) -> None:
        # The baseline lane measures the denominator through the null technique.
        self.assertIsInstance(self._record.technique, BaselineIdentity)
        self.assertEqual(self._record.base_architecture, "hifigan_v1")
        self.assertEqual(self._record.variant_name, "baseline_cpu")


class OptimizedVariantGuardTest(unittest.TestCase):
    # Verifies every guard that must fire before an optimized capsule is
    # allowed to measure anything.
    def setUp(self) -> None:
        # Writes a well-formed base checkpoint and opens the configuration factory.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._temporary_root: Path = Path(self._temporary_directory.name)
        self._base_checkpoint_path: Path = self._temporary_root / "base.ckpt"
        torch.save({"model_state_dict": {}}, self._base_checkpoint_path)
        self._factory: OptimizedRunConfigurationFactory = OptimizedRunConfigurationFactory(
            self._temporary_root,
            self._base_checkpoint_path
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_non_test_stage_is_refused(self) -> None:
        # Optimized variants are measurement subjects, never training subjects.
        evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(
            self._factory.build_variant_run_for_stage("baseline_cpu", "validation")
        )
        with self.assertRaisesRegex(MisconfigurationError, "only stage='test'"):
            evaluator.run()

    def test_refused_stage_writes_no_evidence(self) -> None:
        # The stage gate fires before seeding and before any artifact write.
        configuration: ExperimentConfiguration = self._factory.build_variant_run_for_stage(
            "baseline_cpu",
            "validation"
        )
        evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(configuration)
        with self.assertRaises(MisconfigurationError):
            evaluator.run()
        self.assertEqual(evaluator.row_count, 0)
        self.assertFalse(configuration.run_directory.exists())

    def test_missing_variant_name_is_refused(self) -> None:
        # An unstated technique would make the capsule's identity ambiguous.
        evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(
            self._factory.build_run_without_optimization_provenance()
        )
        with self.assertRaisesRegex(MisconfigurationError, "optimization_variant_name is required"):
            evaluator._resolve_variant_name()

    def test_registered_hardware_lane_is_accepted(self) -> None:
        # The CPU baseline lane is registered for the CPU-lane variant.
        evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(
            self._factory.build_variant_run("baseline_cpu")
        )
        evaluator._validate_hardware_lane("baseline_cpu")

    def test_unregistered_hardware_lane_is_refused(self) -> None:
        # Measurement lanes are fixed per variant for denominator continuity.
        evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(
            self._factory.build_variant_run("baseline_cpu")
        )
        with self.assertRaisesRegex(MisconfigurationError, "registered for hardware lanes"):
            evaluator._validate_hardware_lane("baseline_b200")

    def test_lane_guard_stops_a_run_before_any_evidence_is_written(self) -> None:
        # A GPU-lane variant declared on the CPU lane never reaches measurement.
        # The summary CSV is asserted absent as well as the run directory,
        # because the schema preflight touches that file and runs immediately
        # after this guard; its absence is what proves the ordering.
        configuration: ExperimentConfiguration = self._factory.build_variant_run("baseline_b200")
        evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(configuration)
        with self.assertRaisesRegex(MisconfigurationError, "registered for hardware lanes"):
            evaluator.run()
        self.assertEqual(evaluator.row_count, 0)
        self.assertFalse(configuration.run_directory.exists())
        self.assertFalse(configuration.summary_csv_path.exists())

    def test_missing_base_checkpoint_is_refused(self) -> None:
        # An optimized variant without a stated base checkpoint has no identity.
        evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(
            self._factory.build_run_without_optimization_provenance()
        )
        with self.assertRaisesRegex(MisconfigurationError, "project_checkpoint_path is required"):
            evaluator._resolve_base_checkpoint_path()

    def test_declared_base_checkpoint_is_returned(self) -> None:
        # The configured checkpoint path is the capsule's base identity.
        evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(
            self._factory.build_variant_run("baseline_cpu")
        )
        self.assertEqual(evaluator._resolve_base_checkpoint_path(), self._base_checkpoint_path)

    def test_checkpoint_without_a_model_state_dictionary_is_refused(self) -> None:
        # A checkpoint missing the model_state_dict entry cannot seed a transformation.
        incomplete_checkpoint_path: Path = self._temporary_root / "incomplete.ckpt"
        torch.save({"optimizer_state_dict": {}}, incomplete_checkpoint_path)
        evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(
            self._factory.build_variant_run("baseline_cpu")
        )
        with self.assertRaisesRegex(MisconfigurationError, "does not contain a model_state_dict"):
            evaluator._load_base_checkpoint(StubSamplerlessModule(), incomplete_checkpoint_path)

    def test_well_formed_checkpoint_restores_the_module(self) -> None:
        # A harness checkpoint with a model state dictionary restores without raising.
        evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(
            self._factory.build_variant_run("baseline_cpu")
        )
        evaluator._load_base_checkpoint(StubSamplerlessModule(), self._base_checkpoint_path)
        self.assertEqual(evaluator.row_count, 0)


class OptimizedVariantTransformedExecutionAssertionTest(unittest.TestCase):
    # Verifies the assertion proving the sampler synthesizes with the
    # transformed network rather than a retained pre-transformation object.
    def setUp(self) -> None:
        # Binds the evaluator whose assertion each sampler stub is presented to.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._temporary_root: Path = Path(self._temporary_directory.name)
        self._base_checkpoint_path: Path = self._temporary_root / "base.ckpt"
        torch.save({"model_state_dict": {}}, self._base_checkpoint_path)
        self._factory: OptimizedRunConfigurationFactory = OptimizedRunConfigurationFactory(
            self._temporary_root,
            self._base_checkpoint_path
        )
        self._evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(
            self._factory.build_variant_run("baseline_cpu")
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_sampler_resolving_the_current_network_is_admitted(self) -> None:
        # A provider that resolves module.network describes one single model.
        self._evaluator._assert_transformed_network_execution(StubTransformedModule())

    def test_module_without_a_provider_backed_sampler_is_admitted(self) -> None:
        # Architectures with no sampler seam have nothing to diverge from.
        self._evaluator._assert_transformed_network_execution(StubSamplerlessModule())

    def test_sampler_resolving_a_stale_network_is_refused(self) -> None:
        # Two different network objects would make the measurement describe two models.
        with self.assertRaisesRegex(RuntimeError, "Transformed-execution assertion failed"):
            self._evaluator._assert_transformed_network_execution(StubStaleSamplerModule())


class OptimizedVariantRecipeWritingTest(unittest.TestCase):
    # Verifies the two-scope optimization recipe: run-invariant facts at the
    # variant scope, capsule facts at the run scope, joinable by path.
    def setUp(self) -> None:
        # Binds an evaluator and the module spec whose two recipe scopes are written.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._temporary_root: Path = Path(self._temporary_directory.name)
        self._base_checkpoint_path: Path = self._temporary_root / "base.ckpt"
        torch.save({"model_state_dict": {}}, self._base_checkpoint_path)
        self._factory: OptimizedRunConfigurationFactory = OptimizedRunConfigurationFactory(
            self._temporary_root,
            self._base_checkpoint_path
        )
        self._configuration: ExperimentConfiguration = self._factory.build_variant_run(
            "baseline_cpu"
        )
        self._evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(self._configuration)
        self._module_spec: OptimizedVariantModuleSpec = OptimizedVariantModuleSpec(
            record=OptimizationVariantRegistry().get("hifigan_v1", "baseline_cpu"),
            module=StubTransformedModule(),
            base_checkpoint_path=self._base_checkpoint_path,
            base_checkpoint_sha256="2" * 64
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_variant_recipe_sits_above_the_run_capsules(self) -> None:
        # The variant recipe is shared by every run of that variant.
        self.assertEqual(
            self._evaluator._variant_optimization_recipe_path(),
            self._configuration.run_directory.parents[1] / "optimization_recipe.yaml"
        )

    def test_run_recipe_sits_inside_the_run_capsule(self) -> None:
        # The run recipe is immutable evidence of this capsule alone.
        self.assertEqual(
            self._evaluator._run_optimization_recipe_path(),
            self._configuration.run_directory / "optimization_recipe.yaml"
        )

    def test_variant_recipe_holds_only_run_invariant_facts(self) -> None:
        # Per-run fields would make a shared variant recipe lie about other runs.
        # The absences are asserted alongside the contents, because the whole
        # point of the variant scope is what it must not contain: a run
        # identifier, a seed, or a hardware lane in a document every run of that
        # variant overwrites would describe one of them and misdescribe the rest.
        self._evaluator._write_optimization_recipe(self._module_spec)
        variant_recipe: dict[str, object] = yaml.safe_load(
            self._evaluator._variant_optimization_recipe_path().read_text(encoding="utf-8")
        )
        self.assertEqual(variant_recipe["recipe_scope"], "variant")
        self.assertEqual(variant_recipe["variant_name"], "baseline_cpu")
        self.assertEqual(variant_recipe["base_architecture"], "hifigan_v1")
        self.assertEqual(variant_recipe["base_checkpoint_sha256"], "2" * 64)
        self.assertNotIn("run_id", variant_recipe)
        self.assertNotIn("seed", variant_recipe)
        self.assertNotIn("hardware_name", variant_recipe)

    def test_run_recipe_adds_the_capsule_identity(self) -> None:
        # The run recipe pins lane, precision, seed, hypothesis, and commit.
        self._evaluator._write_optimization_recipe(self._module_spec)
        run_recipe: dict[str, object] = yaml.safe_load(
            self._evaluator._run_optimization_recipe_path().read_text(encoding="utf-8")
        )
        self.assertEqual(run_recipe["recipe_scope"], "run")
        self.assertEqual(run_recipe["hardware_name"], "cpu")
        self.assertEqual(run_recipe["precision_name"], "fp32")
        self.assertEqual(run_recipe["run_id"], "run_optimized")
        self.assertEqual(run_recipe["seed"], 5)
        self.assertEqual(run_recipe["stage"], "test")
        self.assertEqual(run_recipe["hypothesis_id"], "H2")
        self.assertEqual(run_recipe["code_commit_hash"], "0f1e2d3c")

    def test_run_recipe_references_the_variant_recipe(self) -> None:
        # The two scopes remain joinable from the artifact tree alone.
        self._evaluator._write_optimization_recipe(self._module_spec)
        run_recipe: dict[str, object] = yaml.safe_load(
            self._evaluator._run_optimization_recipe_path().read_text(encoding="utf-8")
        )
        self.assertEqual(
            run_recipe["variant_recipe_path"],
            str(self._evaluator._variant_optimization_recipe_path())
        )
        self.assertEqual(run_recipe["base_checkpoint_sha256"], "2" * 64)

    def test_both_recipes_record_the_technique_configuration(self) -> None:
        # The exact transformation configuration is recorded at both scopes.
        self._evaluator._write_optimization_recipe(self._module_spec)
        variant_recipe: dict[str, object] = yaml.safe_load(
            self._evaluator._variant_optimization_recipe_path().read_text(encoding="utf-8")
        )
        run_recipe: dict[str, object] = yaml.safe_load(
            self._evaluator._run_optimization_recipe_path().read_text(encoding="utf-8")
        )
        expected_configuration: dict[str, object] = (
            self._module_spec.record.technique.configuration_dump()
        )
        self.assertEqual(variant_recipe["technique_configuration"], expected_configuration)
        self.assertEqual(run_recipe["technique_configuration"], expected_configuration)


class OptimizedVariantCheckpointHashTest(unittest.TestCase):
    # Verifies the base-checkpoint content hash recorded into the recipes.
    def setUp(self) -> None:
        # Writes the checkpoint whose bytes the hashing assertions digest.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._temporary_root: Path = Path(self._temporary_directory.name)
        self._base_checkpoint_path: Path = self._temporary_root / "base.ckpt"
        torch.save({"model_state_dict": {}}, self._base_checkpoint_path)
        self._factory: OptimizedRunConfigurationFactory = OptimizedRunConfigurationFactory(
            self._temporary_root,
            self._base_checkpoint_path
        )
        self._evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(
            self._factory.build_variant_run("baseline_cpu")
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_hash_matches_the_reference_digest_of_the_file(self) -> None:
        # The recorded hash is the SHA-256 of the checkpoint bytes.
        expected_digest: str = hashlib.sha256(
            self._base_checkpoint_path.read_bytes()
        ).hexdigest()
        self.assertEqual(self._evaluator._compute_sha256(self._base_checkpoint_path), expected_digest)

    def test_hash_is_stable_across_repeated_reads(self) -> None:
        # Hashing twice records the same identity for the same bytes.
        first_digest: str = self._evaluator._compute_sha256(self._base_checkpoint_path)
        second_digest: str = self._evaluator._compute_sha256(self._base_checkpoint_path)
        self.assertEqual(first_digest, second_digest)

    def test_hash_covers_files_larger_than_one_chunk(self) -> None:
        # The chunked read path digests the whole file, not the first chunk.
        # The payload is deliberately larger than the one-mebibyte read size, so
        # a hash implementation that stopped after its first chunk would differ
        # from the reference digest; a smaller file could not tell the two apart.
        large_checkpoint_path: Path = self._temporary_root / "large.ckpt"
        large_payload: bytes = b"vocode" * 400000
        large_checkpoint_path.write_bytes(large_payload)
        self.assertEqual(
            self._evaluator._compute_sha256(large_checkpoint_path),
            hashlib.sha256(large_payload).hexdigest()
        )

    def test_different_bytes_produce_different_hashes(self) -> None:
        # Two checkpoints are never recorded under one identity.
        other_checkpoint_path: Path = self._temporary_root / "other.ckpt"
        torch.save({"model_state_dict": {"weight": torch.zeros(2)}}, other_checkpoint_path)
        self.assertNotEqual(
            self._evaluator._compute_sha256(self._base_checkpoint_path),
            self._evaluator._compute_sha256(other_checkpoint_path)
        )


class OptimizedVariantEvidenceIdentityTest(unittest.TestCase):
    # Verifies the identity and provenance an optimized row carries into the
    # shared summary CSV and the hyperparameter dump.
    def setUp(self) -> None:
        # Binds an evaluator and the module spec whose identity fields are read.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._temporary_root: Path = Path(self._temporary_directory.name)
        self._base_checkpoint_path: Path = self._temporary_root / "base.ckpt"
        torch.save({"model_state_dict": {}}, self._base_checkpoint_path)
        self._factory: OptimizedRunConfigurationFactory = OptimizedRunConfigurationFactory(
            self._temporary_root,
            self._base_checkpoint_path
        )
        self._configuration: ExperimentConfiguration = self._factory.build_variant_run(
            "baseline_cpu"
        )
        self._evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(self._configuration)
        self._module_spec: OptimizedVariantModuleSpec = OptimizedVariantModuleSpec(
            record=OptimizationVariantRegistry().get("hifigan_v1", "baseline_cpu"),
            module=StubTransformedModule(),
            base_checkpoint_path=self._base_checkpoint_path,
            base_checkpoint_sha256="3" * 64
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_unique_id_composes_architecture_variant_category_stage_seed_and_run(self) -> None:
        # The row identifier is unique across the intervention matrix.
        self.assertEqual(
            self._evaluator._build_unique_id(self._module_spec),
            "hifigan_v1_baseline_cpu_project_optimized_variants_test_seed5_run_optimized"
        )

    def test_experiment_logger_carries_the_optimized_variant_prefix(self) -> None:
        # Optimized rows stay distinguishable from project-trained and published rows.
        experiment_logger: ExperimentLogger = self._evaluator._build_experiment_logger(
            self._module_spec
        )
        self.assertEqual(experiment_logger._variant_name, "optimized_baseline_cpu")
        self.assertEqual(
            experiment_logger.unique_id,
            self._evaluator._build_unique_id(self._module_spec)
        )

    def test_hyperparameter_summary_states_the_base_checkpoint(self) -> None:
        # The summary column names the checkpoint the variant transformed.
        summary: str = self._evaluator._summarize_hyperparameters(self._module_spec)
        self.assertIn("architecture=hifigan_v1", summary)
        self.assertIn("variant=baseline_cpu", summary)
        self.assertIn(f"base_checkpoint={self._base_checkpoint_path}", summary)
        self.assertIn("stage=test", summary)

    def test_hyperparameter_dump_traces_the_variant_to_its_checkpoint_bytes(self) -> None:
        # The dump records the base checkpoint path and its content hash.
        dump: HyperparameterDict = self._evaluator._build_hyperparameter_dump(self._module_spec)
        self.assertEqual(dump["architecture_name"], "hifigan_v1")
        self.assertEqual(dump["variant_name"], "optimized_baseline_cpu")
        self.assertEqual(dump["optimization_variant_name"], "baseline_cpu")
        self.assertEqual(dump["base_checkpoint_path"], str(self._base_checkpoint_path))
        self.assertEqual(dump["base_checkpoint_sha256"], "3" * 64)
        self.assertEqual(
            dump["technique_configuration"],
            self._module_spec.record.technique.configuration_dump()
        )

    def test_hyperparameter_dump_records_the_measurement_limits(self) -> None:
        # Every batch limit of the invocation enters the dump.
        dump: HyperparameterDict = self._evaluator._build_hyperparameter_dump(self._module_spec)
        self.assertEqual(dump["limit_test_batches"], 8)
        self.assertEqual(dump["limit_predict_batches"], 9)
        self.assertIsNone(dump["limit_train_batches"])
        self.assertEqual(dump["evidence_category"], "project_optimized_variants")
        self.assertEqual(dump["run_directory"], str(self._configuration.run_directory))


class OptimizedVariantPeakMemoryTest(unittest.TestCase):
    # Verifies the peak-memory reporting bracketing this run only.
    def setUp(self) -> None:
        # Binds an evaluator and the logger its peak-memory metrics are published to.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._temporary_root: Path = Path(self._temporary_directory.name)
        self._base_checkpoint_path: Path = self._temporary_root / "base.ckpt"
        torch.save({"model_state_dict": {}}, self._base_checkpoint_path)
        self._factory: OptimizedRunConfigurationFactory = OptimizedRunConfigurationFactory(
            self._temporary_root,
            self._base_checkpoint_path
        )
        self._configuration: ExperimentConfiguration = self._factory.build_variant_run(
            "baseline_cpu"
        )
        self._evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(self._configuration)
        self._experiment_logger: ExperimentLogger = self._evaluator._build_experiment_logger(
            OptimizedVariantModuleSpec(
                record=OptimizationVariantRegistry().get("hifigan_v1", "baseline_cpu"),
                module=StubTransformedModule(),
                base_checkpoint_path=self._base_checkpoint_path,
                base_checkpoint_sha256="4" * 64
            )
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_peak_host_memory_is_reported_in_megabytes(self) -> None:
        # The host peak is a positive megabyte figure for the current process.
        self._evaluator._log_peak_memory(self._experiment_logger)
        metric_buffer: dict[str, float | int] = self._experiment_logger.metric_buffer
        self.assertIn("peak_host_memory_megabytes", metric_buffer)
        self.assertGreater(metric_buffer["peak_host_memory_megabytes"], 0.0)

    def test_accelerator_peak_is_reported_only_with_an_accelerator(self) -> None:
        # The accelerator counter is absent on a host without CUDA.
        # The expectation is written against the runtime's own availability
        # rather than pinned to either outcome, so the case states one contract
        # on a processor-only runner and on an accelerator host alike.
        self._evaluator._log_peak_memory(self._experiment_logger)
        metric_buffer: dict[str, float | int] = self._experiment_logger.metric_buffer
        self.assertEqual(
            "peak_accelerator_memory_megabytes" in metric_buffer,
            torch.cuda.is_available()
        )

    def test_peak_memory_reset_is_safe_without_an_accelerator(self) -> None:
        # The counter reset is a no-op when no CUDA device is present.
        self._evaluator._reset_peak_memory_statistics()
        self.assertEqual(self._evaluator.row_count, 0)


class OptimizedVariantDispatchWiringTest(unittest.TestCase):
    # Verifies the single-pass Trainers and callback ordering of the prediction
    # and test dispatch methods, including the per-utterance metric table.
    def setUp(self) -> None:
        # Binds an evaluator with the module, logger, and datamodule the dispatch
        # methods are called with.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._temporary_root: Path = Path(self._temporary_directory.name)
        self._base_checkpoint_path: Path = self._temporary_root / "base.ckpt"
        torch.save({"model_state_dict": {}}, self._base_checkpoint_path)
        self._factory: OptimizedRunConfigurationFactory = OptimizedRunConfigurationFactory(
            self._temporary_root,
            self._base_checkpoint_path
        )
        self._configuration: ExperimentConfiguration = self._factory.build_variant_run(
            "baseline_cpu"
        )
        self._evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(self._configuration)
        self._module: StubTransformedModule = StubTransformedModule()
        self._experiment_logger: ExperimentLogger = self._evaluator._build_experiment_logger(
            OptimizedVariantModuleSpec(
                record=OptimizationVariantRegistry().get("hifigan_v1", "baseline_cpu"),
                module=self._module,
                base_checkpoint_path=self._base_checkpoint_path,
                base_checkpoint_sha256="5" * 64
            )
        )
        self._datamodule: LJSpeechDataModule = LJSpeechDataModule(
            self._configuration.data_configuration
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_prediction_dispatch_builds_a_single_pass_trainer(self) -> None:
        # Prediction runs one pass under the run's prediction batch limit.
        with TrainerEntryPointInterception("predict") as interception:
            self._evaluator._dispatch_prediction(
                self._module,
                self._datamodule,
                self._experiment_logger
            )
        trainer: Trainer = interception.recorded_trainer
        self.assertEqual(interception.recorded_trainer_count, 1)
        self.assertEqual(trainer.max_epochs, 1)
        self.assertIs(trainer.logger, self._experiment_logger)
        self.assertEqual(
            trainer.predict_loop.limit_batches,
            self._configuration.limit_predict_batches
        )

    def test_prediction_dispatch_places_the_timing_monitor_ahead_of_the_profiler(self) -> None:
        # The real-time-factor monitor brackets the synthesis calls it times.
        with TrainerEntryPointInterception("predict") as interception:
            self._evaluator._dispatch_prediction(
                self._module,
                self._datamodule,
                self._experiment_logger
            )
        trainer: Trainer = interception.recorded_trainer
        self.assertIsInstance(trainer.callbacks[0], RealTimeFactorMonitor)
        self.assertIsInstance(trainer.callbacks[1], RuntimeProfiler)

    def test_prediction_dispatch_omits_the_monitor_when_rtf_is_unselected(self) -> None:
        # A metric panel without rtf carries no timing monitor into prediction.
        configuration: ExperimentConfiguration = (
            self._factory.build_variant_run_without_real_time_factor("baseline_cpu")
        )
        evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(configuration)
        experiment_logger: ExperimentLogger = evaluator._build_experiment_logger(
            OptimizedVariantModuleSpec(
                record=OptimizationVariantRegistry().get("hifigan_v1", "baseline_cpu"),
                module=self._module,
                base_checkpoint_path=self._base_checkpoint_path,
                base_checkpoint_sha256="6" * 64
            )
        )
        with TrainerEntryPointInterception("predict") as interception:
            evaluator._dispatch_prediction(
                self._module,
                LJSpeechDataModule(configuration.data_configuration),
                experiment_logger
            )
        monitors: list[Callback] = [
            callback for callback in interception.recorded_trainer.callbacks
            if isinstance(callback, RealTimeFactorMonitor)
        ]
        self.assertEqual(monitors, [], msg="rtf must be absent from the prediction callbacks")

    def test_test_dispatch_leads_with_the_metric_sequence(self) -> None:
        # The objective metric panel is the first callback of the test pass.
        with TrainerEntryPointInterception("test") as interception:
            self._evaluator._dispatch_test(
                self._module,
                self._datamodule,
                self._experiment_logger
            )
        trainer: Trainer = interception.recorded_trainer
        self.assertIsInstance(trainer.callbacks[0], MetricSequence)
        self.assertIsInstance(trainer.callbacks[1], RuntimeProfiler)
        self.assertEqual(trainer.max_epochs, 1)
        self.assertEqual(trainer.test_loop.limit_batches, self._configuration.limit_test_batches)

    def test_test_dispatch_requests_the_per_utterance_metric_table(self) -> None:
        # Distribution-level analysis needs the per-utterance table inside the capsule.
        with TrainerEntryPointInterception("test") as interception:
            self._evaluator._dispatch_test(
                self._module,
                self._datamodule,
                self._experiment_logger
            )
        self.assertIsInstance(interception.recorded_trainer.callbacks[0], MetricSequence)
        metric_sequence: MetricSequence = interception.recorded_trainer.callbacks[0]
        self.assertEqual(
            metric_sequence._per_utterance_csv_path,
            self._configuration.artifact_layout.metrics_directory / "metrics_per_utterance.csv"
        )


class OptimizedVariantMetricsSnapshotTest(unittest.TestCase):
    # Verifies the per-run metrics snapshot written from the logger buffer.
    def setUp(self) -> None:
        # Binds an evaluator and the logger whose buffer the snapshot is written from.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._temporary_root: Path = Path(self._temporary_directory.name)
        self._base_checkpoint_path: Path = self._temporary_root / "base.ckpt"
        torch.save({"model_state_dict": {}}, self._base_checkpoint_path)
        self._factory: OptimizedRunConfigurationFactory = OptimizedRunConfigurationFactory(
            self._temporary_root,
            self._base_checkpoint_path
        )
        self._configuration: ExperimentConfiguration = self._factory.build_variant_run(
            "baseline_cpu"
        )
        self._evaluator: OptimizedVariantEvaluator = OptimizedVariantEvaluator(self._configuration)
        self._experiment_logger: ExperimentLogger = self._evaluator._build_experiment_logger(
            OptimizedVariantModuleSpec(
                record=OptimizationVariantRegistry().get("hifigan_v1", "baseline_cpu"),
                module=StubTransformedModule(),
                base_checkpoint_path=self._base_checkpoint_path,
                base_checkpoint_sha256="7" * 64
            )
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_snapshot_matches_the_reduced_metric_buffer(self) -> None:
        # The snapshot is exactly the logger's reduced buffer.
        self._experiment_logger.log_metrics({"real_time_factor": 0.02, "pesq": 3.7}, step=0)
        self._evaluator._write_metrics_snapshot(self._experiment_logger)
        snapshot: dict[str, float] = json.loads(
            self._configuration.metrics_path.read_text(encoding="utf-8")
        )
        self.assertEqual(snapshot, self._experiment_logger.metric_buffer)

    def test_snapshot_keys_are_sorted_for_diff_stability(self) -> None:
        # Sorted keys keep two runs of the same capsule textually comparable.
        self._experiment_logger.log_metrics(
            {"real_time_factor": 0.02, "pesq": 3.7, "mel": 0.2},
            step=0
        )
        self._evaluator._write_metrics_snapshot(self._experiment_logger)
        snapshot: dict[str, float] = json.loads(
            self._configuration.metrics_path.read_text(encoding="utf-8")
        )
        self.assertEqual(list(snapshot.keys()), sorted(snapshot.keys()))

    def test_snapshot_is_written_inside_the_run_capsule(self) -> None:
        # The snapshot lands on the layout-owned metrics path of this capsule.
        self._experiment_logger.log_metrics({"pesq": 3.7}, step=0)
        self._evaluator._write_metrics_snapshot(self._experiment_logger)
        self.assertTrue(self._configuration.metrics_path.exists())
        self.assertEqual(self._configuration.metrics_path.name, "metrics.json")


if __name__ == "__main__":
    unittest.main()
