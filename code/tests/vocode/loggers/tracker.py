# This module:
# 1. Verifies the opening half of the run capsule lifecycle: entering the tracker
#    creates the run directory tree, writes the resolved configuration and the
#    run manifest before any stage executes, and opens a self-identifying
#    execution log
# 2. Verifies the closing half: a normal exit records a completed status with the
#    reported row count, and a failing exit records a failed status carrying the
#    exception identity while letting the original exception propagate
# 3. Verifies that the execution-log sink is detached on exit, so a finished
#    capsule stops absorbing unrelated log output
#
# Design decisions:
# - Configurations are built from the real ExperimentConfiguration and
#   ExperimentArtifactLayout records against a temporary artifact root, so the
#   asserted paths are the paths the layout actually computes
# - The execution log is read only after the context has exited, because the
#   loguru sink is attached with enqueue enabled and is flushed when the tracker
#   detaches it
# - No trainer, dataset, or model is constructed; the dataset root inside the
#   data configuration is a temporary path that is never opened, since the
#   tracker only serializes the configuration it was handed
#
# Author: Rahul Sawhney

import tempfile
import unittest
from pathlib import Path

import yaml
from loguru import logger as log

from vocode.configs.layout import ExperimentArtifactLayout
from vocode.configs.run import ExperimentConfiguration
from vocode.data.ljspeech_datamodule import LJSpeechDataConfig
from vocode.loggers.tracker import RunTracker
from vocode.metrics.rtf import RealTimeFactorConfig


class ExperimentConfigurationFactory:
    # Builds validated ExperimentConfiguration records against a temporary
    # artifact root: a plain reproduction capsule and an optimized-variant
    # capsule that carries the extra provenance fields.
    def __init__(self, root: Path) -> None:
        # Binds the temporary root every artifact path is computed under.
        self._root: Path = root

    def reproduction_capsule(self, run_id: str) -> ExperimentConfiguration:
        # Builds a plain reproduction configuration, which carries no variant
        # provenance and indexes into the first-schema summary surface.
        layout: ExperimentArtifactLayout = ExperimentArtifactLayout(
            artifact_root=self._root / "artifacts",
            evidence_category="project_trained_reproduction",
            architecture_name="hifigan_v1",
            hardware_name="cpu",
            precision_name="fp32",
            seed=11,
            run_id=run_id
        )
        return ExperimentConfiguration(
            experiment_name="reproduction_sweep",
            run_id=run_id,
            hypothesis="the reproduction matches the published quality panel",
            interpretation_notes="capsule note",
            evidence_category="project_trained_reproduction",
            stage="test",
            dataset_split_name="test",
            architecture_name="hifigan_v1",
            seed=11,
            artifact_layout=layout,
            published_weights_root=self._root / "published_weights",
            data_configuration=LJSpeechDataConfig(dataset_root=self._root / "corpus"),
            real_time_factor_configuration=RealTimeFactorConfig()
        )

    def optimized_variant_capsule(self, run_id: str) -> ExperimentConfiguration:
        # Builds an optimized-variant configuration carrying the checkpoint,
        # hypothesis identifier, and commit hash that a variant run must record.
        layout: ExperimentArtifactLayout = ExperimentArtifactLayout(
            artifact_root=self._root / "artifacts",
            evidence_category="project_optimized_variants",
            architecture_name="hifigan_v1",
            hardware_name="cpu",
            precision_name="fp32",
            seed=11,
            run_id=run_id,
            variant_name="channels_last_fp32"
        )
        return ExperimentConfiguration(
            experiment_name="optimization_sweep",
            run_id=run_id,
            hypothesis="the optimized variant preserves quality at lower latency",
            interpretation_notes="capsule note",
            evidence_category="project_optimized_variants",
            stage="test",
            dataset_split_name="test",
            architecture_name="hifigan_v1",
            seed=11,
            artifact_layout=layout,
            published_weights_root=self._root / "published_weights",
            project_checkpoint_path=self._root / "checkpoints" / "best.ckpt",
            optimization_variant_name="channels_last_fp32",
            optimization_hypothesis_id="opt-0001",
            code_commit_hash="0123456789abcdef0123456789abcdef01234567",
            data_configuration=LJSpeechDataConfig(dataset_root=self._root / "corpus"),
            real_time_factor_configuration=RealTimeFactorConfig(),
            accelerator="cpu"
        )


class CapsuleArtifactReader:
    # Reads the durable records of one run capsule back from disk: the resolved
    # configuration, the run manifest, the lifecycle status, and the log text.
    def __init__(self, configuration: ExperimentConfiguration) -> None:
        # Binds the configuration whose layout computes every artifact path,
        # so assertions read the paths the run itself would use.
        self._configuration: ExperimentConfiguration = configuration

    @property
    def status_path(self) -> Path:
        # Returns the lifecycle status file written on exit.
        return self._configuration.run_directory / "run_status.yaml"

    @property
    def resolved_configuration(self) -> dict[str, object]:
        # Parses the settings the capsule was launched with.
        return yaml.safe_load(
            self._configuration.resolved_configuration_path.read_text(encoding="utf-8")
        )

    @property
    def run_manifest(self) -> dict[str, object]:
        # Parses the navigation document written before any stage runs.
        return yaml.safe_load(self._configuration.run_manifest_path.read_text(encoding="utf-8"))

    @property
    def run_status(self) -> dict[str, object]:
        # Parses the recorded outcome, which is what a fleet audit reads.
        return yaml.safe_load(self.status_path.read_text(encoding="utf-8"))

    @property
    def execution_log_text(self) -> str:
        # Reads the execution log; the tracker flushes it when it detaches the
        # sink, so this is only meaningful after the context has exited.
        return self._configuration.artifact_layout.execution_log_path.read_text(encoding="utf-8")


class RunCapsuleOpeningTest(unittest.TestCase):
    # Verifies what entering the tracker produces before any experiment stage
    # runs: directories, resolved configuration, manifest, and log header.
    def setUp(self) -> None:
        # Builds a reproduction configuration and a reader bound to it, so
        # every asserted path is the one the layout computes rather than one
        # restated here. Nothing is created on disk yet: the directory tree is
        # the tracker's own responsibility on entry, which is what these cases
        # exercise. Each class in this file uses a distinct run identifier so
        # no two capsules could ever resolve to the same directory.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._factory: ExperimentConfigurationFactory = ExperimentConfigurationFactory(self._root)
        self._configuration: ExperimentConfiguration = self._factory.reproduction_capsule("run-0001")
        self._reader: CapsuleArtifactReader = CapsuleArtifactReader(self._configuration)

    def tearDown(self) -> None:
        # Removes the temporary artifact root and the whole capsule beneath
        # it. The tracker detaches its own log sink on exit, so nothing here
        # needs to unwind loguru state.
        self._temporary_directory.cleanup()

    def test_entering_creates_the_run_directory_tree(self) -> None:
        # Every evidence lane produces the same capsule anatomy.
        with RunTracker(self._configuration):
            self.assertTrue(self._configuration.run_directory.is_dir())
            self.assertTrue(self._configuration.checkpoints_directory.is_dir())
            self.assertTrue(self._configuration.artifact_layout.logs_directory.is_dir())
            self.assertTrue(self._configuration.artifact_layout.metrics_directory.is_dir())
            self.assertTrue(self._configuration.summary_csv_path.parent.is_dir())

    def test_entering_returns_the_tracker_itself(self) -> None:
        # The context variable is the tracker, so row counts can be reported.
        tracker: RunTracker = RunTracker(self._configuration)
        with tracker as entered_tracker:
            self.assertIs(entered_tracker, tracker)

    def test_configuration_and_manifest_exist_before_any_stage_runs(self) -> None:
        # A capsule that crashes immediately still documents what it was asked
        # to do.
        with RunTracker(self._configuration):
            self.assertTrue(self._configuration.resolved_configuration_path.exists())
            self.assertTrue(self._configuration.run_manifest_path.exists())

    def test_resolved_configuration_matches_the_validated_record(self) -> None:
        # The exact resolved settings are reproducible from the capsule without
        # the launching command.
        with RunTracker(self._configuration):
            pass
        self.assertEqual(
            self._reader.resolved_configuration,
            self._configuration.model_dump(mode="json"),
            msg="the persisted configuration must equal the validated record"
        )

    def test_run_manifest_records_identity_and_artifact_locations(self) -> None:
        # A reader can navigate a downloaded capsule from this one document.
        with RunTracker(self._configuration):
            pass
        run_manifest: dict[str, object] = self._reader.run_manifest
        self.assertEqual(
            tuple(run_manifest.keys()),
            (
                "experiment_name",
                "run_id",
                "evidence_category",
                "stage",
                "dataset_split_name",
                "limit_train_batches",
                "limit_val_batches",
                "limit_test_batches",
                "limit_predict_batches",
                "architecture_name",
                "seed",
                "run_directory",
                "execution_log",
                "metrics_file",
                "metric_history_file",
                "runtime_profile_file",
                "runtime_profile_summary_file",
                "runtime_profile_config_file",
                "metrics",
                "runtime_profiling_enabled",
                "runtime_profile_interval_steps",
                "created_at_utc"
            )
        )
        self.assertEqual(run_manifest["experiment_name"], "reproduction_sweep")
        self.assertEqual(run_manifest["run_id"], "run-0001")
        self.assertEqual(run_manifest["run_directory"], str(self._configuration.run_directory))
        self.assertEqual(run_manifest["execution_log"], "logs/execution.log")
        self.assertEqual(run_manifest["metrics_file"], "metrics/metrics.json")
        self.assertEqual(run_manifest["metric_history_file"], "metrics/metric_history.jsonl")

    def test_run_manifest_records_the_metric_selection(self) -> None:
        # The manifest states which metrics the capsule was asked to produce.
        with RunTracker(self._configuration):
            pass
        self.assertEqual(
            self._reader.run_manifest["metrics"],
            list(self._configuration.metric_selection.names)
        )

    def test_manifest_paths_resolve_inside_the_capsule(self) -> None:
        # The relative locations in the manifest point at real capsule files.
        with RunTracker(self._configuration):
            pass
        recorded_log_path: Path = self._configuration.run_directory / "logs" / "execution.log"
        self.assertEqual(
            recorded_log_path,
            self._configuration.artifact_layout.execution_log_path
        )
        self.assertTrue(recorded_log_path.exists())

    def test_execution_log_opens_with_the_run_identity(self) -> None:
        # The log is self-identifying from its first lines.
        with RunTracker(self._configuration):
            pass
        log_text: str = self._reader.execution_log_text
        self.assertIn("Experiment reproduction_sweep run_id=run-0001", log_text)
        self.assertIn("architecture=hifigan_v1", log_text)
        self.assertIn("seed=11", log_text)
        self.assertIn("evidence_category=project_trained_reproduction", log_text)

    def test_execution_log_states_the_hypothesis_under_test(self) -> None:
        # The capsule records what the run was trying to establish.
        with RunTracker(self._configuration):
            pass
        self.assertIn(
            "Hypothesis: the reproduction matches the published quality panel",
            self._reader.execution_log_text
        )


class RunCapsuleClosingTest(unittest.TestCase):
    # Verifies the normal-exit half of the lifecycle: the completed status file,
    # the reported row count, the finalization log line, and sink detachment.
    def setUp(self) -> None:
        # Same reproduction fixture as the opening cases under its own run
        # identifier. These cases enter and leave the context normally, so the
        # status file they read is written by the success branch of the exit
        # path.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._factory: ExperimentConfigurationFactory = ExperimentConfigurationFactory(self._root)
        self._configuration: ExperimentConfiguration = self._factory.reproduction_capsule("run-0002")
        self._reader: CapsuleArtifactReader = CapsuleArtifactReader(self._configuration)

    def tearDown(self) -> None:
        # Removes the temporary artifact root and the whole capsule beneath
        # it. The tracker detaches its own log sink on exit, so nothing here
        # needs to unwind loguru state.
        self._temporary_directory.cleanup()

    def test_completed_status_is_written_on_normal_exit(self) -> None:
        # Auditing a fleet means parsing outcomes, not scraping log lines.
        with RunTracker(self._configuration):
            pass
        self.assertEqual(self._reader.run_status["status"], "completed")

    def test_completed_status_carries_the_run_identity(self) -> None:
        # A status file is attributable without opening any other artifact.
        with RunTracker(self._configuration):
            pass
        run_status: dict[str, object] = self._reader.run_status
        self.assertEqual(run_status["experiment_name"], "reproduction_sweep")
        self.assertEqual(run_status["run_id"], "run-0002")
        self.assertEqual(run_status["evidence_category"], "project_trained_reproduction")
        self.assertEqual(run_status["stage"], "test")
        self.assertEqual(run_status["dataset_split_name"], "test")
        self.assertEqual(run_status["architecture_name"], "hifigan_v1")
        self.assertEqual(run_status["seed"], 11)

    def test_reported_row_count_reaches_the_status(self) -> None:
        # The lane runner's row count is part of the audited outcome.
        with RunTracker(self._configuration) as tracker:
            tracker.record_completed_rows(3)
        self.assertEqual(self._reader.run_status["completed_row_count"], 3)

    def test_status_records_a_non_negative_elapsed_time(self) -> None:
        # The lifecycle window is measured, not assumed.
        with RunTracker(self._configuration):
            pass
        run_status: dict[str, object] = self._reader.run_status
        self.assertIn("started_at_utc", run_status)
        self.assertIn("finished_at_utc", run_status)
        self.assertGreaterEqual(run_status["elapsed_seconds"], 0.0)

    def test_optimization_variant_is_absent_for_a_reproduction_capsule(self) -> None:
        # Variant provenance appears only where a variant exists.
        with RunTracker(self._configuration):
            pass
        self.assertNotIn("optimization_variant_name", self._reader.run_status)

    def test_finalization_line_is_appended_to_the_execution_log(self) -> None:
        # The log closes with the row count and the elapsed time.
        with RunTracker(self._configuration) as tracker:
            tracker.record_completed_rows(3)
        self.assertIn(
            "Experiment reproduction_sweep finished. Rows written: 3.",
            self._reader.execution_log_text
        )

    def test_log_sink_is_detached_after_exit(self) -> None:
        # A finished capsule stops absorbing unrelated log output.
        with RunTracker(self._configuration):
            pass
        log.info("unrelated line emitted after the capsule closed")
        self.assertNotIn(
            "unrelated line emitted after the capsule closed",
            self._reader.execution_log_text
        )

    def test_configuration_property_returns_the_bound_record(self) -> None:
        # The tracker exposes the immutable configuration it was constructed with.
        tracker: RunTracker = RunTracker(self._configuration)
        self.assertIs(tracker.configuration, self._configuration)


class RunCapsuleFailureTest(unittest.TestCase):
    # Verifies the failure exit path: the original exception propagates and the
    # status file records the failure with the exception identity.
    def setUp(self) -> None:
        # Same reproduction fixture under its own run identifier. These cases
        # raise inside the context, so the status file they read is written by
        # the failure branch and the original exception must still reach the
        # assertion.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._factory: ExperimentConfigurationFactory = ExperimentConfigurationFactory(self._root)
        self._configuration: ExperimentConfiguration = self._factory.reproduction_capsule("run-0003")
        self._reader: CapsuleArtifactReader = CapsuleArtifactReader(self._configuration)

    def tearDown(self) -> None:
        # Removes the temporary artifact root and the whole capsule beneath
        # it. The tracker detaches its own log sink on exit, so nothing here
        # needs to unwind loguru state.
        self._temporary_directory.cleanup()

    def test_exception_propagates_out_of_the_context(self) -> None:
        # The tracker records the failure without swallowing it.
        with self.assertRaisesRegex(RuntimeError, "synthesis stage aborted"):
            with RunTracker(self._configuration):
                raise RuntimeError("synthesis stage aborted")

    def test_failed_status_is_written(self) -> None:
        # A failed capsule is auditable as failed, not merely incomplete.
        with self.assertRaises(RuntimeError):
            with RunTracker(self._configuration):
                raise RuntimeError("synthesis stage aborted")
        self.assertEqual(self._reader.run_status["status"], "failed")

    def test_failed_status_records_the_exception_identity(self) -> None:
        # The status carries what went wrong, so triage needs no log parsing.
        with self.assertRaises(ValueError):
            with RunTracker(self._configuration):
                raise ValueError("metric selection produced no rows")
        run_status: dict[str, object] = self._reader.run_status
        self.assertEqual(run_status["exception_type"], "ValueError")
        self.assertEqual(run_status["exception_message"], "metric selection produced no rows")

    def test_failed_capsule_still_carries_its_configuration_and_manifest(self) -> None:
        # Records written on entry survive a failure during the run.
        with self.assertRaises(RuntimeError):
            with RunTracker(self._configuration):
                raise RuntimeError("synthesis stage aborted")
        self.assertTrue(self._configuration.resolved_configuration_path.exists())
        self.assertTrue(self._configuration.run_manifest_path.exists())

    def test_failed_capsule_omits_the_finalization_line(self) -> None:
        # A failed run must not claim a normal finish.
        with self.assertRaises(RuntimeError):
            with RunTracker(self._configuration):
                raise RuntimeError("synthesis stage aborted")
        self.assertNotIn("finished. Rows written", self._reader.execution_log_text)


class RunCapsuleOptimizedVariantTest(unittest.TestCase):
    # Verifies the optimized-variant capsule: its status carries the variant
    # provenance that distinguishes it from a plain reproduction run.
    def setUp(self) -> None:
        # Uses the optimized-variant configuration instead of the plain
        # reproduction one, because the variant provenance fields are exactly
        # what these cases assert reach the capsule's records; a reproduction
        # capsule carries none of them and would prove nothing here.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._factory: ExperimentConfigurationFactory = ExperimentConfigurationFactory(self._root)
        self._configuration: ExperimentConfiguration = self._factory.optimized_variant_capsule(
            "run-0004"
        )
        self._reader: CapsuleArtifactReader = CapsuleArtifactReader(self._configuration)

    def tearDown(self) -> None:
        # Removes the temporary artifact root and the whole capsule beneath
        # it. The tracker detaches its own log sink on exit, so nothing here
        # needs to unwind loguru state.
        self._temporary_directory.cleanup()

    def test_status_records_the_optimization_variant(self) -> None:
        # The variant name is part of the capsule's scientific identity.
        with RunTracker(self._configuration):
            pass
        self.assertEqual(
            self._reader.run_status["optimization_variant_name"],
            "channels_last_fp32"
        )

    def test_capsule_is_written_under_the_variant_summary_surface(self) -> None:
        # Optimized variants index into the versioned summary file, leaving the
        # first-schema surface frozen.
        with RunTracker(self._configuration):
            pass
        self.assertEqual(self._configuration.summary_csv_path.name, "experiments_v2.csv")
        self.assertTrue(self._configuration.summary_csv_path.parent.is_dir())

    def test_resolved_configuration_records_the_variant_provenance(self) -> None:
        # Checkpoint, hypothesis identifier, and commit hash travel with the capsule.
        with RunTracker(self._configuration):
            pass
        resolved_configuration: dict[str, object] = self._reader.resolved_configuration
        self.assertEqual(resolved_configuration["optimization_variant_name"], "channels_last_fp32")
        self.assertEqual(resolved_configuration["optimization_hypothesis_id"], "opt-0001")
        self.assertEqual(
            resolved_configuration["code_commit_hash"],
            "0123456789abcdef0123456789abcdef01234567"
        )
        self.assertEqual(resolved_configuration["accelerator"], "cpu")
