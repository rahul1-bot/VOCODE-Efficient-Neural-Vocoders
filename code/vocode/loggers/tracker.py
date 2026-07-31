# This module:
# 1. Owns the lifecycle record of one run capsule as a context manager:
#    entering creates the directories, attaches the execution-log sink, and
#    writes the resolved configuration and run manifest; exiting writes the
#    lifecycle status and detaches the sink
# 2. Distinguishes completed from failed runs in a machine-readable
#    run_status.yaml, including the exception identity on failure
#
# Design decisions:
# - The resolved configuration and manifest are written before any stage
#   runs, so even a capsule that crashes immediately documents what it was
#   asked to do
# - The status file is separate from the textual log because auditing a
#   fleet requires parsing outcomes, not scraping log lines
# - Status writing on the failure path is itself guarded, so a status-write
#   error can never mask the original exception
# - The log sink is attached with enqueue enabled, keeping writes safe
#   across worker processes
#
# Author: Rahul Sawhney

from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

import yaml
from loguru import logger as log

from vocode.configs.run import ExperimentConfiguration

__all__: list[str] = ["RunTracker"]


class RunTracker:
    # Experiment-tracking component owning one run's durable record anatomy.
    # Entering the tracker creates the run directories, attaches the execution log sink,
    # and writes the resolved configuration and run manifest; exiting finalizes the log
    # and detaches the sink, so every evidence lane produces identical run records.
    # The tracker records the run; it does not run it. Nothing here executes
    # a stage, and the tracker never suppresses an exception, so a failure
    # inside the block is documented and then propagates unchanged.
    #
    # Integration: a lane runner wraps its entire execution in one tracker,
    # calls record_completed_rows once it knows how many result rows it
    # produced, and lets the context manager close the record on both the
    # success and the failure path.
    #
    # Example::
    #
    #     with RunTracker(configuration) as tracker:
    #         rows_written: int = run_every_stage(configuration)
    #         tracker.record_completed_rows(rows_written)
    def __init__(self, configuration: ExperimentConfiguration) -> None:
        # Binds the configuration, marks the start time, and prepares the
        # sink identifier and row-count slots the lifecycle methods fill.
        # The start time is taken here rather than on entry, so the recorded
        # elapsed duration covers construction as well as execution.
        #
        # Args:
        #     configuration: The validated run configuration. It supplies
        #         the run identity, the artifact layout, and the limits
        #         written into the manifest, and it is republished
        #         unchanged through the configuration property.
        self._configuration: ExperimentConfiguration = configuration
        self._log_sink_identifier: int | None = None
        self._completed_row_count: int = 0
        self._start_time: datetime = datetime.now(tz=timezone.utc)

    def record_completed_rows(self, row_count: int) -> None:
        # Records how many result rows the lane runner wrote for the finalization log.
        # The count reaches both the closing log line and the run-status
        # file, so an auditor can tell a run that completed having produced
        # nothing from one that produced its expected rows. It defaults to
        # zero, which is therefore also what a run that failed before
        # reporting will record.
        #
        # Args:
        #     row_count: Number of summary rows the runner wrote for this
        #         run. The last call wins; the value is not accumulated.
        self._completed_row_count: int = row_count

    @property
    def configuration(self) -> ExperimentConfiguration:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    def _attach_log_sink(self) -> None:
        # Attaches the per-run execution log sink for durable run evidence.
        # Writes are enqueued so records from worker processes reach the file
        # safely, and both the backtrace and diagnose renderings are
        # disabled: an execution log is evidence meant to be read and
        # diffed, and variable-value dumps would make otherwise identical
        # runs produce divergent logs. The returned identifier is retained
        # because it is the only handle by which the sink can later be
        # detached.
        self._log_sink_identifier: int | None = log.add(
            self._configuration.artifact_layout.execution_log_path,
            level="INFO",
            enqueue=True,
            backtrace=False,
            diagnose=False
        )

    def _detach_log_sink(self) -> None:
        # Detaches the per-run execution log sink after finalization.
        # The identifier is cleared as well as removed, so a second call is
        # harmless and cannot remove a sink another tracker has since been
        # assigned the same identifier for. Detaching matters because the
        # loguru logger is process-global: a tracker that left its sink
        # attached would keep capturing records from every later run into
        # this run's log file.
        if self._log_sink_identifier is not None:
            log.remove(self._log_sink_identifier)
            self._log_sink_identifier: int | None = None

    def _write_resolved_configuration(self) -> None:
        # Serializes the complete validated configuration as YAML in the run
        # directory, so the exact resolved settings are reproducible from
        # the capsule without the launching command.
        # The dump is taken in JSON mode, which converts paths, enums, and
        # other rich types into primitives the safe dumper can represent, so
        # the artifact loads without executing arbitrary Python. Declaration
        # order is preserved rather than sorted, keeping the file's structure
        # aligned with the configuration model a reader would consult.
        resolved_configuration: dict[str, object] = self._configuration.model_dump(mode="json")
        self._configuration.resolved_configuration_path.write_text(
            yaml.safe_dump(resolved_configuration, sort_keys=False, default_flow_style=False)
        )

    def _write_run_manifest(self) -> None:
        # Writes the run manifest: identity, limits, the metric selection,
        # and the relative locations of every evidence file inside the
        # capsule, so a reader can navigate a downloaded run from this one
        # document.
        # The evidence-file entries are relative paths rather than absolute
        # ones, which is what lets a capsule stay navigable after being
        # copied off the machine that produced it; the run directory is
        # recorded absolutely alongside them purely as provenance. The
        # entries are declared unconditionally, so the manifest describes the
        # capsule's intended anatomy even for files a short or failed run
        # never produced.
        run_manifest: dict[str, object] = {
            "experiment_name": self._configuration.experiment_name,
            "run_id": self._configuration.run_id,
            "evidence_category": self._configuration.evidence_category,
            "stage": self._configuration.stage,
            "dataset_split_name": self._configuration.dataset_split_name,
            "limit_train_batches": self._configuration.limit_train_batches,
            "limit_val_batches": self._configuration.limit_val_batches,
            "limit_test_batches": self._configuration.limit_test_batches,
            "limit_predict_batches": self._configuration.limit_predict_batches,
            "architecture_name": self._configuration.architecture_name,
            "seed": self._configuration.seed,
            "run_directory": str(self._configuration.run_directory),
            "execution_log": "logs/execution.log",
            "metrics_file": "metrics/metrics.json",
            "metric_history_file": "metrics/metric_history.jsonl",
            "runtime_profile_file": "metrics/runtime_profile.jsonl",
            "runtime_profile_summary_file": "metrics/runtime_profile_summary.json",
            "runtime_profile_config_file": "metrics/runtime_profile_config.json",
            "metrics": list(self._configuration.metric_selection.names),
            "runtime_profiling_enabled": self._configuration.runtime_profiling_enabled,
            "runtime_profile_interval_steps": self._configuration.runtime_profile_interval_steps,
            "created_at_utc": self._start_time.isoformat(timespec="seconds")
        }
        self._configuration.run_manifest_path.write_text(
            yaml.safe_dump(run_manifest, sort_keys=False, default_flow_style=False)
        )

    def _announce_run(self) -> None:
        # Opens the execution log with the run identity and the hypothesis
        # under test, so the log is self-identifying from its first lines.
        # A log fragment quoted out of context is therefore still
        # attributable to a specific run, and a reviewer reading the file
        # learns what the run was trying to establish before reading what it
        # did.
        log.info(
            f"Experiment {self._configuration.experiment_name} run_id={self._configuration.run_id} "
            f"architecture={self._configuration.architecture_name} seed={self._configuration.seed} "
            f"evidence_category={self._configuration.evidence_category} stage={self._configuration.stage}"
        )
        log.info(f"Hypothesis: {self._configuration.hypothesis}")

    def _finalize_run_log(self) -> None:
        # Finalizes experiment logging after all selected run stages complete.
        # Emitted only on the success path, so the presence of this line is
        # itself the log-level signal that the run reached normal
        # completion; the machine-readable equivalent is the status file.
        elapsed_seconds: float = (datetime.now(tz=timezone.utc) - self._start_time).total_seconds()
        log.info(
            f"Experiment {self._configuration.experiment_name} finished. "
            f"Rows written: {self._completed_row_count}. Elapsed: {elapsed_seconds:.1f}s"
        )

    def _write_run_status(
        self,
        status: str,
        exception_type: type[BaseException] | None = None,
        exception_value: BaseException | None = None
    ) -> None:
        # Writes the lifecycle outcome separately from logs so failed capsules are auditable.
        # The payload always carries the run identity, both timestamps, the
        # elapsed duration, and the completed row count, so a fleet can be
        # audited by parsing these files alone without scraping log text.
        # The optimization variant is included only when the run has one,
        # keeping the record free of null placeholders. On the failure path
        # the exception's type name and message are added; the type is taken
        # from the argument when the caller supplied one and derived from the
        # value otherwise, so the field is populated even for an exception
        # raised without its class being passed through.
        #
        # Args:
        #     status: Lifecycle outcome, written verbatim as the payload's
        #         first key. The two values this class emits are
        #         "completed" and "failed".
        #     exception_type: Class of the failing exception when known.
        #         Default: ``None``.
        #     exception_value: The failing exception itself; its presence,
        #         not the status string, is what adds the exception fields
        #         to the payload. Default: ``None``.
        finished_at: datetime = datetime.now(tz=timezone.utc)
        payload: dict[str, object] = {
            "status": status,
            "experiment_name": self._configuration.experiment_name,
            "run_id": self._configuration.run_id,
            "evidence_category": self._configuration.evidence_category,
            "stage": self._configuration.stage,
            "dataset_split_name": self._configuration.dataset_split_name,
            "architecture_name": self._configuration.architecture_name,
            "seed": self._configuration.seed,
            "completed_row_count": self._completed_row_count,
            "started_at_utc": self._start_time.isoformat(timespec="seconds"),
            "finished_at_utc": finished_at.isoformat(timespec="seconds"),
            "elapsed_seconds": (finished_at - self._start_time).total_seconds()
        }
        if self._configuration.optimization_variant_name is not None:
            payload["optimization_variant_name"] = self._configuration.optimization_variant_name
        if exception_value is not None:
            payload["exception_type"] = (
                exception_type.__name__
                if exception_type is not None
                else type(exception_value).__name__
            )
            payload["exception_message"] = str(exception_value)
        status_path: Path = self._configuration.run_directory / "run_status.yaml"
        status_path.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))

    def _write_failed_run_status(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException
    ) -> None:
        # Preserves the original exception if status emission itself fails.
        # A status write can fail for reasons unrelated to the run, such as
        # a full or read-only filesystem. Letting that failure escape would
        # replace the exception the operator needs to see with one about
        # bookkeeping, so it is logged and swallowed instead. The broad
        # except is deliberate and is the reason this path is separated from
        # the unguarded success-path write.
        try:
            self._write_run_status("failed", exception_type, exception_value)
        except Exception:
            log.exception("Failed to write run_status.yaml for failed experiment run.")

    def __enter__(self) -> RunTracker:
        # Opens the run record: directories, log sink, resolved configuration, and manifest.
        # The order is load-bearing. Directories exist before the sink is
        # attached because the sink opens a file inside them; the sink is
        # attached before the documents are written so any failure while
        # writing them is captured in the execution log; and both documents
        # are written before the announcement so the capsule already
        # describes what it was asked to do even if the very first stage
        # crashes.
        #
        # Returns:
        #     This tracker, so the with statement can bind it and the body
        #     can call record_completed_rows.
        self._configuration.artifact_layout.create_run_directories()
        self._attach_log_sink()
        self._write_resolved_configuration()
        self._write_run_manifest()
        self._announce_run()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        exception_traceback: TracebackType | None
    ) -> None:
        # Closes the run record: failure logging, finalization line, and sink detachment.
        # The failure branch logs the traceback and writes a failed status
        # under its own guard; the success branch logs the closing summary
        # and writes a completed status. Both branches then detach the sink,
        # which is why detachment sits outside the conditional: a sink left
        # attached would capture later runs into this run's file.
        #
        # Args:
        #     exception_type: Class of the exception leaving the block, or
        #         ``None`` on the success path.
        #     exception_value: The exception itself, or ``None``. This is
        #         the value the branch tests, so an exception raised with no
        #         value would be treated as success.
        #     exception_traceback: The traceback, accepted to satisfy the
        #         context-manager protocol and not consulted; the loguru
        #         exception logger reads the active exception directly.
        #
        # Returns:
        #     ``None``, which is falsy, so an exception raised inside the
        #     block always propagates after being recorded. This tracker
        #     documents failures; it never suppresses them.
        if exception_value is not None:
            log.exception("Experiment run failed before normal finalization.")
            self._write_failed_run_status(exception_type, exception_value)
        else:
            self._finalize_run_log()
            self._write_run_status("completed")
        self._detach_log_sink()
