# This module:
# 1. Verifies the metric buffering contract of ExperimentLogger: finite values
#    accumulate under last-write-wins semantics, non-finite values are dropped
#    before they can poison a strict JSON artifact, and the exposed buffer is a
#    defensive copy
# 2. Verifies the metric artifacts inside the run capsule: the append-only
#    history file, the latest-snapshot pair, and the sentinel step recorded when
#    a run finalizes without ever logging a metric
# 3. Verifies the hyperparameter record: declaration order is preserved in YAML
#    and a rewrite replaces rather than appends
# 4. Verifies the summary flush: exactly one row per run, repeated flushes are
#    no-ops, and an incompatible summary surface fails the flush closed
#
# Design decisions:
# - The logger is driven through its public harness surface (log_metrics,
#    log_hyperparams, finalize, flush_to_experiments_csv) with plain float
#   metrics, because reduction happens upstream in the harness and no trainer is
#   constructed here
# - Every artifact is verified by reading the file back and parsing it, so the
#   assertions bind to what a downloaded capsule would contain
# - All output is confined to a per-test temporary directory; the real artifact
#   tree and the shared study summary are never touched
#
# Author: Rahul Sawhney

import csv
import json
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path

import yaml

from vocode.loggers.experiment import ExperimentLogger
from vocode.loggers.result import ExperimentResultRow


class LoggerFactory:
    # Builds ExperimentLogger instances bound to a temporary run capsule, with
    # the summary surface and hyperparameter record under the same root.
    def __init__(self, root: Path) -> None:
        # Binds the temporary root every capsule and summary lives under.
        self._root: Path = root

    def for_run(self, run_name: str) -> ExperimentLogger:
        # Builds a logger for one run against the shared study summary.
        return self.targeting_summary(run_name, self._root / "summary" / "experiments.csv")

    def targeting_summary(self, run_name: str, summary_csv_path: Path) -> ExperimentLogger:
        # Builds a logger aimed at a chosen summary file, which is how the
        # incompatible-schema path is reached.
        run_directory: Path = self._root / run_name
        return ExperimentLogger(
            run_directory=run_directory,
            summary_csv_path=summary_csv_path,
            hyperparameters_path=run_directory / "hyperparameters.yaml",
            architecture_name="hifigan_v1",
            variant_name="project_trained_reproduction_baseline",
            seed=11,
            unique_id=f"hifigan_v1_{run_name}",
            dataset_name="ljspeech",
            hyperparameters_summary="learning_rate=0.0002;batch_size=16",
            interpretation_notes="capsule note"
        )


class MetricArtifactReader:
    # Reads the metric artifacts of one run capsule back from disk: the
    # append-only history records and the two latest-snapshot files.
    def __init__(self, run_directory: Path) -> None:
        # Locates the metrics subdirectory of one run capsule.
        self._metrics_directory: Path = run_directory / "metrics"

    @property
    def directory(self) -> Path:
        # Returns the metrics directory the logger is expected to create.
        return self._metrics_directory

    @property
    def history_path(self) -> Path:
        # Returns the append-only scalar trajectory file.
        return self._metrics_directory / "metric_history.jsonl"

    @property
    def latest_path(self) -> Path:
        # Returns the stamped latest-snapshot file.
        return self._metrics_directory / "latest_metrics.json"

    @property
    def snapshot_path(self) -> Path:
        # Returns the bare scalar-mapping snapshot file.
        return self._metrics_directory / "metrics.json"

    @property
    def history_records(self) -> list[dict[str, float | int | dict[str, float]]]:
        # Parses one record per non-empty line; an absent file reads as no
        # history, which is what an ignored batch must leave behind.
        if not self.history_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.history_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    @property
    def latest_record(self) -> dict[str, float | int | dict[str, float]]:
        # Returns the stamped snapshot carrying the step and the whole buffer.
        return json.loads(self.latest_path.read_text(encoding="utf-8"))

    @property
    def snapshot_record(self) -> dict[str, float]:
        # Returns the bare metric mapping without the surrounding stamp.
        return json.loads(self.snapshot_path.read_text(encoding="utf-8"))


class SummaryFileReader:
    # Reads a summary CSV back from disk as its header columns and its parsed
    # row mappings.
    def __init__(self, summary_path: Path) -> None:
        # Binds the summary file the flush is expected to append to.
        self._summary_path: Path = summary_path

    @property
    def header_columns(self) -> tuple[str, ...]:
        # Returns the first line as parsed columns; an empty file reads as no
        # columns rather than raising.
        with self._summary_path.open(newline="", encoding="utf-8") as summary_file:
            header_reader: Iterator[list[str]] = csv.reader(summary_file)
            return tuple(next(header_reader, ()))

    @property
    def rows(self) -> list[dict[str, str]]:
        # Returns every data row keyed by column name.
        with self._summary_path.open(newline="", encoding="utf-8") as summary_file:
            return list(csv.DictReader(summary_file))


class ExperimentMetricBufferTest(unittest.TestCase):
    # Verifies which incoming metric values reach the run buffer and how the
    # buffer is exposed to callers.
    def setUp(self) -> None:
        # Builds one logger per case against a fresh capsule. Constructing the
        # logger already creates the metrics directory, so cases that assert
        # an artifact is absent are asserting that no file was written into an
        # existing directory, not that the directory itself is missing.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._factory: LoggerFactory = LoggerFactory(self._root)
        self._logger: ExperimentLogger = self._factory.for_run("run_a")

    def tearDown(self) -> None:
        # Removes the temporary root, discarding the whole run capsule and the
        # summary surface written under it, so no case leaks artifacts into
        # the real evidence tree.
        self._temporary_directory.cleanup()

    def test_finite_metrics_are_buffered(self) -> None:
        # Reduced scalar metrics accumulate for the run's summary row.
        self._logger.log_metrics({"pesq": 3.25, "stoi": 0.91}, step=0)
        self.assertEqual(self._logger.metric_buffer, {"pesq": 3.25, "stoi": 0.91})

    def test_non_finite_metrics_are_dropped(self) -> None:
        # A poisoned buffer would abort strict JSON writing at flush time.
        self._logger.log_metrics(
            {"pesq": 3.25, "stoi": float("nan"), "mel_error": float("inf")},
            step=0
        )
        self.assertEqual(self._logger.metric_buffer, {"pesq": 3.25})

    def test_batch_of_only_non_finite_metrics_writes_no_artifact(self) -> None:
        # An entirely non-finite batch is ignored, leaving no history record and
        # no snapshot behind.
        self._logger.log_metrics({"pesq": float("nan")}, step=4)
        reader: MetricArtifactReader = MetricArtifactReader(self._root / "run_a")
        self.assertEqual(self._logger.metric_buffer, {})
        self.assertEqual(reader.history_records, [])
        self.assertFalse(
            reader.latest_path.exists(),
            msg="an ignored batch must not produce a latest-metrics snapshot"
        )

    def test_later_values_overwrite_earlier_values_per_key(self) -> None:
        # The buffer holds the final value of each metric key.
        self._logger.log_metrics({"pesq": 3.0, "stoi": 0.90}, step=0)
        self._logger.log_metrics({"pesq": 3.5}, step=1)
        self.assertEqual(self._logger.metric_buffer, {"pesq": 3.5, "stoi": 0.90})

    def test_metric_buffer_property_returns_a_defensive_copy(self) -> None:
        # A caller inspecting the buffer cannot mutate the run's state.
        self._logger.log_metrics({"pesq": 3.25}, step=0)
        exposed_buffer: dict[str, float | int] = self._logger.metric_buffer
        exposed_buffer["injected"] = 1.0
        self.assertEqual(self._logger.metric_buffer, {"pesq": 3.25})

    def test_unique_id_property_returns_the_bound_identifier(self) -> None:
        # The run identifier written into artifacts is readable by the runner.
        self.assertEqual(self._logger.unique_id, "hifigan_v1_run_a")


class ExperimentMetricArtifactTest(unittest.TestCase):
    # Verifies the on-disk metric artifacts of one run capsule: directory
    # creation, the append-only history, and the latest-snapshot pair.
    def setUp(self) -> None:
        # Pairs the logger with a reader aimed at the same capsule, so every
        # assertion in this class reads the artifact back from disk rather
        # than trusting the logger's in-memory state.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._factory: LoggerFactory = LoggerFactory(self._root)
        self._logger: ExperimentLogger = self._factory.for_run("run_a")
        self._reader: MetricArtifactReader = MetricArtifactReader(self._root / "run_a")

    def tearDown(self) -> None:
        # Removes the temporary root, discarding the whole run capsule and the
        # summary surface written under it, so no case leaks artifacts into
        # the real evidence tree.
        self._temporary_directory.cleanup()

    def test_metrics_directory_is_created_at_construction(self) -> None:
        # Artifact writers never have to prepare their own directory.
        self.assertTrue(self._reader.directory.is_dir())

    def test_metric_history_is_append_only(self) -> None:
        # An interrupted run retains its full scalar trajectory for audit.
        self._logger.log_metrics({"pesq": 3.0}, step=0)
        self._logger.log_metrics({"pesq": 3.5}, step=1)
        history_records: list[dict[str, float | int | dict[str, float]]] = self._reader.history_records
        self.assertEqual(len(history_records), 2)
        self.assertEqual(history_records[0]["step"], 0)
        self.assertEqual(history_records[0]["metrics"], {"pesq": 3.0})
        self.assertEqual(history_records[1]["step"], 1)
        self.assertEqual(history_records[1]["metrics"], {"pesq": 3.5})

    def test_history_records_carry_a_wall_clock_stamp(self) -> None:
        # Each record is placed in time so trajectories can be aligned.
        self._logger.log_metrics({"pesq": 3.0}, step=0)
        history_record: dict[str, float | int | dict[str, float]] = self._reader.history_records[0]
        self.assertEqual(set(history_record.keys()), {"time_unix", "step", "metrics"})
        self.assertGreater(history_record["time_unix"], 0.0)

    def test_history_records_only_the_finite_subset_of_a_batch(self) -> None:
        # The durable history matches what the buffer accepted.
        self._logger.log_metrics({"pesq": 3.0, "stoi": float("nan")}, step=0)
        self.assertEqual(self._reader.history_records[0]["metrics"], {"pesq": 3.0})

    def test_latest_snapshot_reflects_the_whole_buffer(self) -> None:
        # The snapshot is cumulative, not per-batch.
        self._logger.log_metrics({"pesq": 3.0}, step=0)
        self._logger.log_metrics({"stoi": 0.91}, step=1)
        latest_record: dict[str, float | int | dict[str, float]] = self._reader.latest_record
        self.assertEqual(set(latest_record.keys()), {"time_unix", "step", "metrics"})
        self.assertEqual(latest_record["step"], 1)
        self.assertEqual(latest_record["metrics"], {"pesq": 3.0, "stoi": 0.91})

    def test_metrics_snapshot_holds_the_bare_buffer(self) -> None:
        # The metrics file is the plain scalar mapping a reader expects.
        self._logger.log_metrics({"pesq": 3.0, "stoi": 0.91}, step=0)
        self.assertEqual(self._reader.snapshot_record, {"pesq": 3.0, "stoi": 0.91})

    def test_finalize_preserves_the_last_recorded_step(self) -> None:
        # Teardown rewrites the snapshot without inventing a new step.
        self._logger.log_metrics({"pesq": 3.0}, step=7)
        self._logger.finalize()
        self.assertEqual(self._reader.latest_record["step"], 7)
        self.assertEqual(self._reader.latest_record["metrics"], {"pesq": 3.0})

    def test_finalize_without_metrics_records_the_sentinel_step(self) -> None:
        # A run that never recorded a metric is marked, not silently empty.
        self._logger.finalize()
        self.assertEqual(self._reader.latest_record["step"], -1)
        self.assertEqual(self._reader.latest_record["metrics"], {})

    def test_finalize_after_an_ignored_batch_keeps_the_sentinel_step(self) -> None:
        # A batch dropped as non-finite never becomes the recorded last step.
        self._logger.log_metrics({"pesq": float("inf")}, step=5)
        self._logger.finalize()
        self.assertEqual(self._reader.latest_record["step"], -1)


class ExperimentHyperparameterRecordTest(unittest.TestCase):
    # Verifies the one-time hyperparameter record: YAML ordering, replacement
    # on rewrite, and directory creation.
    def setUp(self) -> None:
        # Records the hyperparameter path the factory gave the logger, so the
        # cases can read the written YAML back without reaching into the
        # logger for its own configuration.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._factory: LoggerFactory = LoggerFactory(self._root)
        self._logger: ExperimentLogger = self._factory.for_run("run_a")
        self._hyperparameters_path: Path = self._root / "run_a" / "hyperparameters.yaml"

    def tearDown(self) -> None:
        # Removes the temporary root, discarding the whole run capsule and the
        # summary surface written under it, so no case leaks artifacts into
        # the real evidence tree.
        self._temporary_directory.cleanup()

    def test_hyperparameters_are_written_in_declaration_order(self) -> None:
        # Provenance reads in the order the runner declared it, not alphabetically.
        self._logger.log_hyperparams({"seed": 11, "architecture_name": "hifigan_v1", "batch_size": 16})
        written_lines: list[str] = self._hyperparameters_path.read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(written_lines, ["seed: 11", "architecture_name: hifigan_v1", "batch_size: 16"])

    def test_nested_hyperparameters_round_trip(self) -> None:
        # Structured provenance survives the YAML round trip intact.
        declared_parameters: dict[str, int | str | list[int] | dict[str, int]] = {
            "seed": 11,
            "mel": {"n_fft": 1024, "hop_length": 256},
            "segment_sizes": [8192, 16384]
        }
        self._logger.log_hyperparams(declared_parameters)
        parsed_parameters: dict[str, int | str | list[int] | dict[str, int]] = yaml.safe_load(
            self._hyperparameters_path.read_text(encoding="utf-8")
        )
        self.assertEqual(parsed_parameters, declared_parameters)

    def test_rewrite_replaces_the_previous_record(self) -> None:
        # The record is the current provenance, not an accumulating log.
        self._logger.log_hyperparams({"seed": 11})
        self._logger.log_hyperparams({"seed": 12})
        parsed_parameters: dict[str, int] = yaml.safe_load(
            self._hyperparameters_path.read_text(encoding="utf-8")
        )
        self.assertEqual(parsed_parameters, {"seed": 12})

    def test_missing_parent_directory_is_created(self) -> None:
        # The record can be pointed anywhere inside the capsule.
        nested_path: Path = self._root / "run_a" / "provenance" / "hyperparameters.yaml"
        nested_logger: ExperimentLogger = ExperimentLogger(
            run_directory=self._root / "run_a",
            summary_csv_path=self._root / "summary" / "experiments.csv",
            hyperparameters_path=nested_path,
            architecture_name="hifigan_v1",
            variant_name="baseline",
            seed=11,
            unique_id="hifigan_v1_nested",
            dataset_name="ljspeech",
            hyperparameters_summary="",
            interpretation_notes=""
        )
        nested_logger.log_hyperparams({"seed": 11})
        self.assertTrue(nested_path.exists())


class ExperimentSummaryFlushTest(unittest.TestCase):
    # Verifies that a run materializes into exactly one summary row carrying its
    # identity and buffered measurements, and fails closed on a foreign schema.
    def setUp(self) -> None:
        # Aims the reader at the same shared summary path the factory gives
        # every logger it builds, which is what lets a case flush two
        # independent runs and observe both rows in one file.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._factory: LoggerFactory = LoggerFactory(self._root)
        self._logger: ExperimentLogger = self._factory.for_run("run_a")
        self._summary_path: Path = self._root / "summary" / "experiments.csv"
        self._reader: SummaryFileReader = SummaryFileReader(self._summary_path)

    def tearDown(self) -> None:
        # Removes the temporary root, discarding the whole run capsule and the
        # summary surface written under it, so no case leaks artifacts into
        # the real evidence tree.
        self._temporary_directory.cleanup()

    def test_flush_writes_exactly_one_row(self) -> None:
        # The summary is an index over capsules, one row per run.
        self._logger.log_metrics({"pesq": 3.25}, step=0)
        self._logger.flush_to_experiments_csv()
        self.assertEqual(len(self._reader.rows), 1)

    def test_repeated_flush_does_not_double_count_the_run(self) -> None:
        # A runner that flushes twice cannot inflate the study index.
        self._logger.log_metrics({"pesq": 3.25}, step=0)
        self._logger.flush_to_experiments_csv()
        self._logger.flush_to_experiments_csv()
        self.assertEqual(len(self._reader.rows), 1)

    def test_flushed_row_carries_the_bound_run_identity(self) -> None:
        # Identity columns come from the logger's construction arguments.
        self._logger.flush_to_experiments_csv()
        recorded_row: dict[str, str] = self._reader.rows[0]
        self.assertEqual(recorded_row["unique_id"], "hifigan_v1_run_a")
        self.assertEqual(recorded_row["architecture_name"], "hifigan_v1")
        self.assertEqual(recorded_row["dataset_name"], "ljspeech")
        self.assertEqual(recorded_row["variant_name"], "project_trained_reproduction_baseline")
        self.assertEqual(int(recorded_row["seed"]), 11)
        self.assertEqual(recorded_row["hyperparameters_summary"], "learning_rate=0.0002;batch_size=16")
        self.assertEqual(recorded_row["interpretation_notes"], "capsule note")

    def test_buffered_metrics_populate_their_summary_columns(self) -> None:
        # Buffer keys reach the columns of the same name.
        self._logger.log_metrics(
            {"pesq": 3.25, "stoi": 0.91, "parameter_count": 13500000.0, "pesq_failure_count": 2.0},
            step=0
        )
        self._logger.flush_to_experiments_csv()
        recorded_row: dict[str, str] = self._reader.rows[0]
        self.assertEqual(float(recorded_row["pesq"]), 3.25)
        self.assertEqual(float(recorded_row["stoi"]), 0.91)
        self.assertEqual(int(recorded_row["parameter_count"]), 13500000)
        self.assertEqual(int(recorded_row["pesq_failure_count"]), 2)

    def test_unproduced_metrics_remain_empty_cells(self) -> None:
        # Absence is recorded as an empty cell, never as a fabricated zero.
        self._logger.log_metrics({"pesq": 3.25}, step=0)
        self._logger.flush_to_experiments_csv()
        recorded_row: dict[str, str] = self._reader.rows[0]
        self.assertEqual(recorded_row["mcd"], "")
        self.assertEqual(recorded_row["real_time_factor"], "")

    def test_two_runs_append_to_the_same_summary(self) -> None:
        # Independent capsules share one append-only index.
        self._logger.flush_to_experiments_csv()
        second_logger: ExperimentLogger = self._factory.for_run("run_b")
        second_logger.flush_to_experiments_csv()
        recorded_identifiers: list[str] = [row["unique_id"] for row in self._reader.rows]
        self.assertEqual(recorded_identifiers, ["hifigan_v1_run_a", "hifigan_v1_run_b"])

    def test_flush_into_an_incompatible_summary_fails_closed(self) -> None:
        # A legacy summary surface aborts the flush instead of being rewritten.
        legacy_path: Path = self._root / "experiments_legacy.csv"
        legacy_path.write_text("unique_id,date,pesq\nfirst_schema_run,2025-01-01,3.1\n", encoding="utf-8")
        content_before: str = legacy_path.read_text(encoding="utf-8")
        legacy_logger: ExperimentLogger = self._factory.targeting_summary("run_c", legacy_path)
        with self.assertRaisesRegex(ValueError, "Existing summary schema differs"):
            legacy_logger.flush_to_experiments_csv()
        self.assertEqual(legacy_path.read_text(encoding="utf-8"), content_before)

    def test_summary_header_is_the_locked_result_schema(self) -> None:
        # The logger writes through the schema authority, not its own column list.
        self._logger.flush_to_experiments_csv()
        self.assertEqual(self._reader.header_columns, ExperimentResultRow.csv_columns)
