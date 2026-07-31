# This module:
# 1. Implements the experiment logger for VOCODE runs: buffers reduced scalar
#    metrics, appends a durable metric history, and maintains latest-snapshot
#    JSON artifacts inside the run capsule
# 2. Materializes the buffered metrics into exactly one summary CSV row per
#    run through the result writer
#
# Harness contract (syntheticmind):
# - Subclasses the harness Logger, so Trainers deliver already-reduced float
#   metrics with a step index through log_metrics, the one-time
#   hyperparameter record through log_hyperparams, and the teardown flush
#   through finalize; metric reduction itself happens upstream in the harness
#
# Report alignment:
# - One flushed row corresponds to one execution of the study, so the
#   register the report publishes is an index over executions rather than
#   over configurations: a configuration measured under several inference
#   seeds contributes one row per execution, and grouping into
#   model-variant groups happens downstream of this writer
#
# Design decisions:
# - Non-finite values are dropped before buffering because every JSON
#   artifact is written with allow_nan disabled, and a poisoned buffer would
#   abort evidence writing at flush time
# - The metric history is an append-only JSONL file, so interrupted runs
#   retain their full scalar trajectory for audit
# - The summary row is flushed at most once per run because the CSV is an
#   append-only index over immutable capsules; repeating a row would
#   double-count the run
#
# Author: Rahul Sawhney

import json
import math
import time
from pathlib import Path
from typing import override

import yaml

from syntheticmind.loggers.logger import Logger
from syntheticmind.utilities.types import HyperparameterDict

from vocode.loggers.result import ExperimentResultRow
from vocode.loggers.writer import ExperimentResultWriter

__all__: list[str] = ["ExperimentLogger"]


class ExperimentLogger(Logger):
    # Harness-facing logger bound to one run's identity. It accumulates every
    # metric the harness or evaluator emits and converts the final buffer
    # into the run's summary CSV row. The buffer is last-value-wins per key,
    # so the row reports each metric's final observation rather than its
    # trajectory; the trajectory is preserved separately in the append-only
    # history file, which is the artifact to consult when a run's behavior
    # over time matters.
    #
    # Integration: the trainer drives log_metrics, log_hyperparams, and
    # finalize through the harness Logger contract and never calls the flush.
    # The runner owns flush_to_experiments_csv and calls it once the run's
    # evidence is complete, which is why the summary row is written outside
    # the harness lifecycle rather than from finalize: a run that fails
    # mid-stage still leaves its capsule artifacts but must not add a row
    # claiming a completed measurement.
    def __init__(
        self,
        run_directory: Path,
        summary_csv_path: Path,
        hyperparameters_path: Path,
        architecture_name: str,
        variant_name: str,
        seed: int,
        unique_id: str,
        dataset_name: str,
        hyperparameters_summary: str,
        interpretation_notes: str
    ) -> None:
        # Binds the run identity and artifact paths, prepares the empty
        # metric buffer, and creates the metrics directory.
        #
        # Args:
        #     run_directory: Root of the run capsule; textual logs live in
        #         its logs subdirectory and metric artifacts in its metrics
        #         subdirectory.
        #     summary_csv_path: Append-only experiment summary CSV that
        #         receives this run's single result row at flush.
        #     hyperparameters_path: YAML file receiving the one-time
        #         hyperparameter record.
        #     architecture_name: Architecture identity column of the
        #         result row.
        #     variant_name: Variant identity column, already prefixed with
        #         the evidence family by the runner.
        #     seed: Seed identity column of the result row.
        #     unique_id: Stable per-run identifier written to the result
        #         row and artifacts.
        #     dataset_name: Dataset identity column of the result row.
        #     hyperparameters_summary: Compact semicolon-separated summary
        #         written into the row's hyperparameters column.
        #     interpretation_notes: Analyst-facing note column carried
        #         through to the row.
        super().__init__(save_dir=run_directory, name="logs", version=None)
        self._run_directory: Path = run_directory
        self._metrics_directory: Path = run_directory / "metrics"
        self._metric_history_path: Path = self._metrics_directory / "metric_history.jsonl"
        self._latest_metrics_path: Path = self._metrics_directory / "latest_metrics.json"
        self._metrics_snapshot_path: Path = self._metrics_directory / "metrics.json"
        self._hyperparameters_path: Path = hyperparameters_path
        self._architecture_name: str = architecture_name
        self._variant_name: str = variant_name
        self._seed: int = seed
        self._unique_id: str = unique_id
        self._dataset_name: str = dataset_name
        self._hyperparameters_summary: str = hyperparameters_summary
        self._interpretation_notes: str = interpretation_notes
        self._metric_buffer: dict[str, float | int] = {}
        self._result_writer: ExperimentResultWriter = ExperimentResultWriter(
            experiments_csv=summary_csv_path
        )
        self._row_flushed: bool = False
        self._last_step: int | None = None
        self._metrics_directory.mkdir(parents=True, exist_ok=True)

    @override
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        # Merges the finite subset of the incoming metrics into the run
        # buffer (later values overwrite earlier ones per key), appends the
        # history record, and refreshes the latest-snapshot artifacts.
        #
        # A call whose metrics are entirely non-finite returns before any
        # state changes: the buffer, the recorded step, the history file, and
        # the snapshots are all left exactly as they were. That is deliberate
        # rather than incidental, because writing a history record with an
        # empty metric map would claim an observation that carries no
        # measurement, and advancing the recorded step would misdate the
        # snapshot written at teardown.
        #
        # Args:
        #     metrics: Already-reduced scalar values keyed by metric name.
        #         Reduction happens upstream in the harness, so values
        #         arrive here as plain floats.
        #     step: Step index this batch of metrics belongs to. It is
        #         recorded verbatim into the history entry and retained as
        #         the run's last observed step.
        finite_metrics: dict[str, float] = self._finite_metrics(metrics)
        if not finite_metrics:
            return
        for key, value in finite_metrics.items():
            self._metric_buffer[key] = value
        self._last_step: int | None = step
        self._append_metric_history(finite_metrics, step)
        self._write_latest_metrics(step)

    @override
    def log_hyperparams(self, params: HyperparameterDict) -> None:
        # Writes the run-provenance record as YAML in declaration order,
        # replacing any previous record at the same path.
        #
        # Declaration order is preserved rather than sorted because the
        # record is written to be read by a human auditing what the run was
        # configured to do, and the caller's grouping carries meaning that
        # alphabetical ordering would destroy. The write is a full
        # replacement, so a second call supersedes the first rather than
        # appending; the harness contract calls this once per run.
        #
        # Args:
        #     params: The run's hyperparameter record. Values must be
        #         YAML-representable through the safe dumper, which is the
        #         constraint that keeps the artifact loadable without
        #         executing arbitrary Python.
        self._hyperparameters_path.parent.mkdir(parents=True, exist_ok=True)
        self._hyperparameters_path.write_text(yaml.safe_dump(params, sort_keys=False, default_flow_style=False))

    @override
    def finalize(self) -> None:
        # Rewrites the latest-snapshot artifacts one final time at teardown
        # so the on-disk state reflects the complete buffer even after a
        # failure; a step of -1 marks that no metric was ever recorded.
        #
        # The harness invokes this on every teardown path, including
        # teardown after an exception, which is what makes the snapshot
        # trustworthy as the run's final state. The sentinel step is
        # distinguishable from any real step because the harness never
        # issues a negative index, so a reader can tell an empty run from one
        # that recorded metrics at step zero.
        self._write_latest_metrics(step=self._last_step if self._last_step is not None else -1)

    def flush_to_experiments_csv(self) -> None:
        # Materializes the buffer into the run's single summary CSV row;
        # repeated calls are no-ops so a runner cannot double-count the run.
        #
        # The guard is set only after the write returns, so a failed write
        # leaves the flag clear and a retry is still possible. The writer
        # takes its own exclusive lock, so this may be called concurrently
        # by separate runs sharing one summary file.
        #
        # Raises:
        #     ValueError: Propagated from the writer when the existing
        #         summary's header disagrees with the current result-row
        #         schema. The row is not written and the run remains
        #         unflushed.
        if self._row_flushed:
            return
        self._result_writer.write_row(self._build_result_row())
        self._row_flushed: bool = True

    @property
    def unique_id(self) -> str:
        # Returns the stable run identifier written to result rows and artifacts.
        return self._unique_id

    @property
    def metric_buffer(self) -> dict[str, float | int]:
        # Returns buffered metrics collected before CSV row materialization.
        # A copy is returned, so a caller inspecting the buffer cannot
        # mutate the run's evidence.
        return dict(self._metric_buffer)

    def _build_result_row(self) -> ExperimentResultRow:
        # Converts the accumulated metric buffer plus the bound run identity
        # into the typed result row that the writer serializes.
        return ExperimentResultRow.from_metric_buffer(
            unique_id=self._unique_id,
            architecture_name=self._architecture_name,
            dataset_name=self._dataset_name,
            variant_name=self._variant_name,
            seed=self._seed,
            hyperparameters_summary=self._hyperparameters_summary,
            interpretation_notes=self._interpretation_notes,
            metric_buffer=self._metric_buffer
        )

    def _append_metric_history(self, metrics: dict[str, float], step: int) -> None:
        # Appends a durable scalar-history record for interrupted-run auditability.
        # One JSON object per line, opened in append mode and closed
        # immediately, so a run killed at any point retains every record
        # written before the interruption and a reader can stream the file
        # without parsing it whole. Keys are sorted within each record so
        # two runs of the same configuration produce byte-comparable
        # history, and allow_nan is disabled so a non-finite value would
        # raise here rather than emit the non-standard NaN token that
        # strict JSON readers reject; the caller has already screened
        # non-finite values out, making this the second line of defence.
        payload: dict[str, float | int | dict[str, float]] = {
            "time_unix": time.time(),
            "step": step,
            "metrics": metrics
        }
        self._metric_history_path.parent.mkdir(parents=True, exist_ok=True)
        with self._metric_history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, sort_keys=True, allow_nan=False))
            file.write("\n")

    def _write_latest_metrics(self, step: int) -> None:
        # Writes the latest metric snapshot to stable per-run JSON artifacts.
        # Two files are refreshed at fixed paths, each a full replacement
        # rather than an append. The latest-metrics artifact wraps the buffer
        # with the wall-clock time and step, answering when the snapshot was
        # taken; the metrics artifact holds the bare buffer, so a consumer
        # that only wants the values needs no unwrapping. Fixed paths mean a
        # downloaded capsule always exposes its final metrics at the same
        # two locations regardless of how many steps the run took.
        payload: dict[str, float | int | dict[str, float | int]] = {
            "time_unix": time.time(),
            "step": step,
            "metrics": dict(self._metric_buffer)
        }
        latest_payload: str = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        self._latest_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._latest_metrics_path.write_text(latest_payload, encoding="utf-8")
        self._metrics_snapshot_path.write_text(
            json.dumps(self._metric_buffer, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8"
        )

    def _finite_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        # Keeps JSON metric artifacts strict by dropping non-finite scalar values.
        # Both infinities and NaN are removed, because every JSON artifact in
        # this module is written with allow_nan disabled and a single
        # poisoned entry would abort evidence writing at flush time, losing
        # the whole snapshot rather than the one bad metric. Dropping is
        # therefore preferred to substituting a sentinel, which would enter
        # the summary row as a real measurement.
        return {
            key: float(value)
            for key, value in metrics.items()
            if math.isfinite(float(value))
        }
