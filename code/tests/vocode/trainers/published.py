# This module:
# 1. Verifies PublishedWeightsEvaluator construction and the stage gate that
#    restricts author-released weights to the test stage, because training on
#    top of a published release would contaminate the comparison against the
#    project-trained reproductions
# 2. Verifies the harness Trainers the prediction and test dispatch methods
#    build: single-pass epoch budget, batch limits, and the callback ordering
#    that places the timing monitor and the metric panel first in their passes
# 3. Verifies the evidence identity of a published row: the published_ variant
#    prefix, the composed unique id, the retrieval-provenance fields in the
#    summary and hyperparameter dump, and the metrics snapshot
#
# Design decisions:
# - A harness entry point is replaced by a recording stand-in inside a scoped
#   context manager, so the Trainer the evaluator actually builds is inspected
#   while no prediction or test loop ever executes; the original entry point is
#   restored on both the success and the failure path
# - Evaluation subjects are assembled directly as PublishedWeightsModuleSpec
#   records around a synthetic provenance record and a minimal harness Module
#   stub, so no author checkpoint is retrieved, hashed, or loaded
# - Boundaries excluded: _prepare_module_spec and the post-gate body of run(),
#   because both build a reference architecture through the model registry and
#   then load author-released weights from the local weights root, which is a
#   retained-binary dependency this suite must not require
#
# Author: Rahul Sawhney

import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.runtime_profiler import RuntimeProfiler
from syntheticmind.core.module import Module
from syntheticmind.core.trainer import Trainer
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.types import HyperparameterDict, HyperparameterValue

from vocode.configs.layout import ExperimentArtifactLayout, ExperimentStage
from vocode.configs.run import ExperimentConfiguration
from vocode.data.ljspeech_datamodule import LJSpeechDataConfig, LJSpeechDataModule
from vocode.loggers.experiment import ExperimentLogger
from vocode.metrics.registry import MetricSelection
from vocode.metrics.rtf import RealTimeFactorConfig, RealTimeFactorMonitor
from vocode.metrics.sequence import MetricSequence
from vocode.models.vocoder import ArchitectureName, PublishedWeightProvenance
from vocode.trainers.published import PublishedWeightsEvaluator, PublishedWeightsModuleSpec


class StubArchitectureModule(Module):
    # Minimal harness Module standing in for a registry-built architecture
    # carrying author weights, so dispatch wiring needs no author checkpoint.
    def __init__(self) -> None:
        # Opens the harness Module without registering any submodule.
        super().__init__()


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


class PublishedRunConfigurationFactory:
    # Builds valid published-weights ExperimentConfiguration records rooted in
    # one temporary artifact tree, with distinct batch limits per stage so
    # forwarding can be told apart from cross-wiring.
    #
    # Integration: the test and prediction limits differ from one another, so an
    # assertion that a stage's Trainer carries the right limit proves the
    # evaluator forwarded that stage's own value rather than one that merely
    # happens to match. Every configuration is the real validated record and
    # every path descends from the per-case temporary root, so no case reaches
    # the corpus, the author weights, or the real artifact tree.
    def __init__(self, temporary_root: Path) -> None:
        # Binds the temporary root every built configuration is anchored in.
        #
        # Args:
        #     temporary_root: The per-case temporary directory all built paths
        #         descend from.
        self._temporary_root: Path = temporary_root

    def build_test_run(self) -> ExperimentConfiguration:
        # Builds the one admitted test-stage run description.
        return self._compose("test", None)

    def build_run_for_stage(self, stage: ExperimentStage) -> ExperimentConfiguration:
        # Builds a run description on the requested stage, including refused ones.
        return self._compose(stage, None)

    def build_test_run_without_real_time_factor(self) -> ExperimentConfiguration:
        # Builds a run description whose metric panel omits the timing metric.
        return self._compose("test", MetricSelection(names=("pesq", "stoi")))

    def build_test_run_without_runtime_profiling(self) -> ExperimentConfiguration:
        # Builds a run description with runtime telemetry switched off.
        return self._compose("test", None, runtime_profiling_enabled=False)

    def _compose(
        self,
        stage: ExperimentStage,
        metric_selection: MetricSelection | None,
        runtime_profiling_enabled: bool = True
    ) -> ExperimentConfiguration:
        # Composes the full run description from the varied stage and metric panel.
        # Everything the cases do not vary is fixed here, so two configurations
        # differ only in what the case under test is about.
        #
        # Args:
        #     stage: The stage the run declares, including the ones the gate
        #         must refuse.
        #     metric_selection: The metric panel, defaulting to the full one so
        #         the timing monitor is present unless a case removes it.
        #     runtime_profiling_enabled: Whether the telemetry callback is
        #         built. Default: ``True``.
        #
        # Returns:
        #     A validated published-weights run description anchored in the
        #     temporary root.
        layout: ExperimentArtifactLayout = ExperimentArtifactLayout(
            artifact_root=self._temporary_root / "artifacts",
            evidence_category="published_checkpoint_evaluation",
            architecture_name="hifigan_v1",
            hardware_name="cpu",
            precision_name="fp32",
            seed=21,
            run_id="run_published"
        )
        resolved_metric_selection: MetricSelection = (
            metric_selection if metric_selection is not None else MetricSelection()
        )
        return ExperimentConfiguration(
            experiment_name="vocode_published",
            run_id="run_published",
            hypothesis="Author releases anchor the reproduction comparison.",
            interpretation_notes="Synthetic configuration used for wiring verification.",
            evidence_category="published_checkpoint_evaluation",
            stage=stage,
            dataset_split_name=stage,
            architecture_name="hifigan_v1",
            seed=21,
            artifact_layout=layout,
            published_weights_root=self._temporary_root / "published_weights",
            data_configuration=LJSpeechDataConfig(dataset_root=self._temporary_root / "corpus"),
            metric_selection=resolved_metric_selection,
            real_time_factor_configuration=RealTimeFactorConfig(),
            limit_test_batches=6,
            limit_predict_batches=7,
            accelerator="cpu",
            runtime_profiling_enabled=runtime_profiling_enabled,
            runtime_profile_interval_steps=40
        )


class PublishedProvenanceFactory:
    # Builds synthetic release-identity records so provenance handling can be
    # verified without retrieving or hashing an author checkpoint.
    # The record's shape is real even though its content is synthetic, which is
    # what lets the evidence assertions check that every provenance field
    # survives into the row rather than only the two the summary column names.
    def build(self, architecture_name: ArchitectureName) -> PublishedWeightProvenance:
        # Builds a synthetic release-identity record for the requested architecture.
        # The author source is a real project URI while the retrieval fields and
        # the expected hash are obviously synthetic, so no case can be mistaken
        # for a claim about a genuine release.
        #
        # Args:
        #     architecture_name: The architecture the synthetic release names.
        #
        # Returns:
        #     A validated provenance record carrying every field a real
        #     release identity holds.
        return PublishedWeightProvenance(
            architecture_name=architecture_name,
            variant_name="v1",
            author_source_uri="https://github.com/jik876/hifi-gan",
            retrieval_kind="google_drive_folder",
            retrieval_uri="https://drive.google.com/drive/folders/synthetic",
            retrieval_filename="generator_v1",
            expected_sha256="9" * 64,
            local_relative_path=Path("hifigan_v1/generator_v1"),
            serialization="checkpoint_dict",
            state_dict_key="generator"
        )


class PublishedWeightsEvaluatorConstructionTest(unittest.TestCase):
    # Verifies that construction binds the configuration and opens per-evaluator
    # state without producing any evidence row.
    def setUp(self) -> None:
        # Binds one evaluator over an admitted test-stage configuration.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._factory: PublishedRunConfigurationFactory = PublishedRunConfigurationFactory(
            Path(self._temporary_directory.name)
        )
        self._configuration: ExperimentConfiguration = self._factory.build_test_run()
        self._evaluator: PublishedWeightsEvaluator = PublishedWeightsEvaluator(self._configuration)

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

    def test_each_evaluator_owns_its_row_counter_and_model_registry(self) -> None:
        # Two evaluators over the same configuration share no counter and no registry.
        other_evaluator: PublishedWeightsEvaluator = PublishedWeightsEvaluator(self._configuration)
        self.assertEqual(other_evaluator.row_count, 0)
        self.assertEqual(self._evaluator.row_count, 0)
        self.assertIsNot(other_evaluator._registry, self._evaluator._registry)


class PublishedWeightsModuleSpecTest(unittest.TestCase):
    # Verifies the frozen binding of the evaluation subject to its release
    # provenance and architecture configuration dump.
    def setUp(self) -> None:
        # Binds the provenance and the exact module bound into the spec under test.
        self._provenance: PublishedWeightProvenance = PublishedProvenanceFactory().build("hifigan_v1")
        self._module: StubArchitectureModule = StubArchitectureModule()
        self._module_spec: PublishedWeightsModuleSpec = PublishedWeightsModuleSpec(
            provenance=self._provenance,
            module=self._module,
            configuration_dump={"learning_rate": 0.0002}
        )

    def test_spec_binds_provenance_module_and_configuration(self) -> None:
        # The spec carries the exact loaded module together with the release identity.
        self.assertIs(self._module_spec.provenance, self._provenance)
        self.assertIs(self._module_spec.module, self._module)
        self.assertEqual(self._module_spec.configuration_dump, {"learning_rate": 0.0002})

    def test_spec_refuses_post_construction_substitution(self) -> None:
        # Freezing prevents swapping what was loaded for what gets measured.
        with self.assertRaises(ValueError):
            self._module_spec.configuration_dump: dict[str, object] = {}

    def test_spec_refuses_unknown_fields(self) -> None:
        # The closed record rejects keys that would carry unvalidated state.
        with self.assertRaises(ValueError):
            PublishedWeightsModuleSpec(
                provenance=self._provenance,
                module=StubArchitectureModule(),
                configuration_dump={},
                checkpoint_epoch=7
            )


class PublishedWeightsStageGateTest(unittest.TestCase):
    # Verifies that only the test stage is admitted, so published weights are
    # never trained on or silently evaluated under another split label.
    def setUp(self) -> None:
        # Opens the factory each refused stage draws its configuration from.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._factory: PublishedRunConfigurationFactory = PublishedRunConfigurationFactory(
            Path(self._temporary_directory.name)
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_training_stage_is_refused(self) -> None:
        # Training on an author release would contaminate the reference anchor.
        evaluator: PublishedWeightsEvaluator = PublishedWeightsEvaluator(
            self._factory.build_run_for_stage("train")
        )
        with self.assertRaisesRegex(MisconfigurationError, "only stage='test'"):
            evaluator.run()

    def test_validation_stage_is_refused(self) -> None:
        # The validation split is not an admitted published-weights stage.
        evaluator: PublishedWeightsEvaluator = PublishedWeightsEvaluator(
            self._factory.build_run_for_stage("validation")
        )
        with self.assertRaisesRegex(MisconfigurationError, "only stage='test'"):
            evaluator.run()

    def test_refused_stage_writes_no_evidence(self) -> None:
        # The gate fires before seeding, module construction, and any artifact write.
        configuration: ExperimentConfiguration = self._factory.build_run_for_stage("train")
        evaluator: PublishedWeightsEvaluator = PublishedWeightsEvaluator(configuration)
        with self.assertRaises(MisconfigurationError):
            evaluator.run()
        self.assertEqual(evaluator.row_count, 0)
        self.assertFalse(configuration.run_directory.exists())


class PublishedWeightsDispatchWiringTest(unittest.TestCase):
    # Verifies the single-pass Trainers and callback ordering of the prediction
    # and test dispatch methods.
    def setUp(self) -> None:
        # Binds an evaluator with the module, logger, and datamodule the dispatch
        # methods are called with.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._factory: PublishedRunConfigurationFactory = PublishedRunConfigurationFactory(
            Path(self._temporary_directory.name)
        )
        self._configuration: ExperimentConfiguration = self._factory.build_test_run()
        self._evaluator: PublishedWeightsEvaluator = PublishedWeightsEvaluator(self._configuration)
        self._module: StubArchitectureModule = StubArchitectureModule()
        self._module_spec: PublishedWeightsModuleSpec = PublishedWeightsModuleSpec(
            provenance=PublishedProvenanceFactory().build("hifigan_v1"),
            module=self._module,
            configuration_dump={}
        )
        self._experiment_logger: ExperimentLogger = self._evaluator._build_experiment_logger(
            self._module_spec
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
            self._factory.build_test_run_without_real_time_factor()
        )
        evaluator: PublishedWeightsEvaluator = PublishedWeightsEvaluator(configuration)
        experiment_logger: ExperimentLogger = evaluator._build_experiment_logger(self._module_spec)
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

    def test_disabled_profiling_removes_the_telemetry_callback(self) -> None:
        # A run without profiling carries only the metric panel into the test pass.
        configuration: ExperimentConfiguration = (
            self._factory.build_test_run_without_runtime_profiling()
        )
        evaluator: PublishedWeightsEvaluator = PublishedWeightsEvaluator(configuration)
        experiment_logger: ExperimentLogger = evaluator._build_experiment_logger(self._module_spec)
        self.assertEqual(evaluator._build_runtime_profiler_callbacks(), [])
        with TrainerEntryPointInterception("test") as interception:
            evaluator._dispatch_test(
                self._module,
                LJSpeechDataModule(configuration.data_configuration),
                experiment_logger
            )
        profilers: list[Callback] = [
            callback for callback in interception.recorded_trainer.callbacks
            if isinstance(callback, RuntimeProfiler)
        ]
        self.assertEqual(profilers, [])

    def test_runtime_profiler_carries_the_configured_interval(self) -> None:
        # The profiler is built once with the run's profiling interval.
        profiler_callbacks: list[Callback] = self._evaluator._build_runtime_profiler_callbacks()
        self.assertEqual(len(profiler_callbacks), 1)
        self.assertIsInstance(profiler_callbacks[0], RuntimeProfiler)
        profiler: RuntimeProfiler = profiler_callbacks[0]
        self.assertEqual(
            profiler._profile_every_n_steps,
            self._configuration.runtime_profile_interval_steps
        )


class PublishedWeightsEvidenceIdentityTest(unittest.TestCase):
    # Verifies the identity and retrieval provenance a published row carries
    # into the shared summary CSV and the hyperparameter dump.
    def setUp(self) -> None:
        # Binds an evaluator and the module spec whose identity fields are read.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._factory: PublishedRunConfigurationFactory = PublishedRunConfigurationFactory(
            Path(self._temporary_directory.name)
        )
        self._configuration: ExperimentConfiguration = self._factory.build_test_run()
        self._evaluator: PublishedWeightsEvaluator = PublishedWeightsEvaluator(self._configuration)
        self._provenance: PublishedWeightProvenance = PublishedProvenanceFactory().build("hifigan_v1")
        self._module_spec: PublishedWeightsModuleSpec = PublishedWeightsModuleSpec(
            provenance=self._provenance,
            module=StubArchitectureModule(),
            configuration_dump={"learning_rate": 0.0002}
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_unique_id_composes_architecture_category_stage_seed_and_run(self) -> None:
        # The row identifier is unique across the study by construction.
        self.assertEqual(
            self._evaluator._build_unique_id(self._module_spec),
            "hifigan_v1_published_checkpoint_evaluation_test_seed21_run_published"
        )

    def test_experiment_logger_carries_the_published_variant_prefix(self) -> None:
        # Published rows stay distinguishable from project-trained rows in the shared CSV.
        experiment_logger: ExperimentLogger = self._evaluator._build_experiment_logger(
            self._module_spec
        )
        self.assertEqual(experiment_logger._variant_name, "published_v1")
        self.assertEqual(
            experiment_logger.unique_id,
            self._evaluator._build_unique_id(self._module_spec)
        )

    def test_hyperparameter_summary_states_both_weight_sources(self) -> None:
        # The summary column names the author source and the retrieval source.
        summary: str = self._evaluator._summarize_hyperparameters(self._module_spec)
        self.assertIn(f"author_source={self._provenance.author_source_uri}", summary)
        self.assertIn(f"retrieval_source={self._provenance.retrieval_uri}", summary)
        self.assertIn("architecture=hifigan_v1", summary)
        self.assertIn("stage=test", summary)

    def test_hyperparameter_dump_records_the_release_identity(self) -> None:
        # The dump traces the row back to the exact release it measured.
        dump: HyperparameterDict = self._evaluator._build_hyperparameter_dump(self._module_spec)
        self.assertEqual(dump["architecture_name"], "hifigan_v1")
        self.assertEqual(dump["variant_name"], "v1")
        self.assertEqual(dump["author_source_uri"], self._provenance.author_source_uri)
        self.assertEqual(dump["retrieval_uri"], self._provenance.retrieval_uri)
        self.assertEqual(dump["evidence_category"], "published_checkpoint_evaluation")
        self.assertEqual(dump["seed"], 21)

    def test_hyperparameter_dump_embeds_the_full_provenance_record(self) -> None:
        # The complete provenance dump travels with the row for audit.
        # Its nesting is asserted before its content, because a provenance
        # flattened into a string would still contain the same substrings while
        # ceasing to be machine-readable evidence; the fields checked afterwards
        # are the ones no summary column carries, so the dump is the only place
        # an auditor can recover them.
        dump: HyperparameterDict = self._evaluator._build_hyperparameter_dump(self._module_spec)
        provenance_dump: HyperparameterValue = dump["published_weight_provenance"]
        self.assertIsInstance(
            provenance_dump,
            dict,
            msg="The provenance must travel as a nested record, not as a flattened string."
        )
        self.assertEqual(provenance_dump["expected_sha256"], self._provenance.expected_sha256)
        self.assertEqual(provenance_dump["serialization"], "checkpoint_dict")
        self.assertEqual(provenance_dump["state_dict_key"], "generator")
        self.assertEqual(provenance_dump["retrieval_kind"], "google_drive_folder")

    def test_hyperparameter_dump_records_the_measurement_limits(self) -> None:
        # Every batch limit of the invocation enters the dump.
        dump: HyperparameterDict = self._evaluator._build_hyperparameter_dump(self._module_spec)
        self.assertEqual(dump["limit_test_batches"], 6)
        self.assertEqual(dump["limit_predict_batches"], 7)
        self.assertIsNone(dump["limit_train_batches"])
        self.assertEqual(dump["metrics"], list(self._configuration.metric_selection.names))
        self.assertEqual(dump["run_directory"], str(self._configuration.run_directory))

    def test_hyperparameter_dump_copies_the_architecture_configuration(self) -> None:
        # The architecture configuration dump is recorded alongside the results.
        dump: HyperparameterDict = self._evaluator._build_hyperparameter_dump(self._module_spec)
        self.assertEqual(dump["model_configuration"], {"learning_rate": 0.0002})


class PublishedWeightsMetricsSnapshotTest(unittest.TestCase):
    # Verifies the per-run metrics snapshot written from the logger buffer.
    def setUp(self) -> None:
        # Binds an evaluator and the logger whose buffer the snapshot is written from.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._factory: PublishedRunConfigurationFactory = PublishedRunConfigurationFactory(
            Path(self._temporary_directory.name)
        )
        self._configuration: ExperimentConfiguration = self._factory.build_test_run()
        self._evaluator: PublishedWeightsEvaluator = PublishedWeightsEvaluator(self._configuration)
        self._module_spec: PublishedWeightsModuleSpec = PublishedWeightsModuleSpec(
            provenance=PublishedProvenanceFactory().build("hifigan_v1"),
            module=StubArchitectureModule(),
            configuration_dump={}
        )
        self._experiment_logger: ExperimentLogger = self._evaluator._build_experiment_logger(
            self._module_spec
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_snapshot_matches_the_reduced_metric_buffer(self) -> None:
        # The snapshot is exactly the logger's reduced buffer.
        self._experiment_logger.log_metrics({"pesq": 3.9, "stoi": 0.94}, step=0)
        self._evaluator._write_metrics_snapshot(self._experiment_logger)
        snapshot: dict[str, float] = json.loads(
            self._configuration.metrics_path.read_text(encoding="utf-8")
        )
        self.assertEqual(snapshot, self._experiment_logger.metric_buffer)

    def test_snapshot_keys_are_sorted_for_diff_stability(self) -> None:
        # Sorted keys keep two runs of the same cell textually comparable.
        self._experiment_logger.log_metrics({"stoi": 0.94, "mel": 0.18, "pesq": 3.9}, step=0)
        self._evaluator._write_metrics_snapshot(self._experiment_logger)
        snapshot: dict[str, float] = json.loads(
            self._configuration.metrics_path.read_text(encoding="utf-8")
        )
        self.assertEqual(list(snapshot.keys()), sorted(snapshot.keys()))

    def test_snapshot_is_written_inside_the_run_capsule(self) -> None:
        # The snapshot lands on the layout-owned metrics path of this run.
        self._experiment_logger.log_metrics({"pesq": 3.9}, step=0)
        self._evaluator._write_metrics_snapshot(self._experiment_logger)
        self.assertTrue(self._configuration.metrics_path.exists())
        self.assertEqual(self._configuration.metrics_path.name, "metrics.json")


if __name__ == "__main__":
    unittest.main()
