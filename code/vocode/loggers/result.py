# This module:
# 1. Defines ExperimentResultRow, the frozen typed record behind one summary
#    CSV row: run identity, the objective quality panel, speed and complexity
#    measurements, measurement-denominator counts, and per-metric failure
#    counts
# 2. Converts a run's metric buffer into that record and the record into the
#    stable CSV column mapping
#
# Report alignment:
# - Rows of this schema populate the result register that the report
#   publishes alongside the admission and exclusion registers. The quality
#   columns carry proxy measurements, never listener judgments, and the
#   timing columns are lane-conditional warm measurements, so a row states
#   what was measured under one profile rather than a hardware-independent
#   ranking
#
# Design decisions:
# - The column tuple is the single schema authority; the writer validates
#   existing files against it and serializes rows from it, so schema and
#   serialization cannot drift apart
# - Every measurement column is optional because evidence categories differ
#   in which metrics they can produce; absence is recorded as an empty cell,
#   never as a fabricated zero
# - Denominator and failure-count columns accompany the panel means so a
#   reviewer can judge each mean against how many utterances produced it and
#   how many failed
# - The row timestamp is UTC at construction, making rows totally ordered
#   across machines without local-timezone ambiguity
#
# Author: Rahul Sawhney

from datetime import datetime, timezone
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

__all__: list[str] = ["ExperimentResultRow"]


class ExperimentResultRow(BaseModel):
    # One frozen summary row. csv_columns is the schema authority consumed by
    # the writer for both header validation and serialization order.
    # The class variable and the field declarations are kept in the same
    # order deliberately: the writer compares an existing file's header
    # against the tuple and serializes through it, so the tuple is what a
    # reviewer must consult to know what a given column position means.
    # Changing it is a schema change that invalidates every existing summary
    # file, which is exactly why the writer refuses to append across a
    # mismatch rather than tolerating it.
    #
    # Fields:
    #     Run identity, always present:
    #         unique_id: Stable per-run identifier shared with the run
    #             capsule's artifacts, and the join key between this index
    #             and the capsule holding the authoritative evidence.
    #         date: UTC timestamp stamped at row construction, to seconds
    #             precision in ISO 8601 form.
    #         architecture_name: Vocoder family under test.
    #         dataset_name: Corpus the run evaluated against.
    #         variant_name: Variant label, already prefixed with the
    #             evidence family by the runner.
    #         seed: Seed governing the run.
    #         hyperparameters_summary: Compact semicolon-separated
    #             rendering of the settings that distinguish this run.
    #         interpretation_notes: Analyst-facing free text carried
    #             through unmodified.
    #     Objective quality panel, each optional:
    #         mel_error: Log-mel reconstruction error against the
    #             reference; lower is better.
    #         pesq: Perceptual evaluation of speech quality.
    #         stoi: Short-time objective intelligibility.
    #         multi_resolution_stft_error: Spectral error aggregated over
    #             several STFT resolutions.
    #         mcd: Mel-cepstral distortion.
    #         las_rmse: Log-amplitude-spectrum root-mean-square error in
    #             decibels.
    #         f0_rmse_cents: Fundamental-frequency error in cents, a
    #             pitch-relative unit so the figure is comparable across
    #             speakers.
    #         periodicity_rmse: Error in the periodicity envelope.
    #         vuv_f1: F1 score of the candidate's voiced/unvoiced decisions
    #             against the reference's.
    #         utmos_strong: UTMOS strong-model predicted mean opinion
    #             score. It is a no-reference predictor, so unlike every
    #             other entry in this panel it scores the candidate alone.
    #     Speed and complexity, each optional:
    #         parameter_count: Deployable parameter count, including packed
    #             quantized weights that carry no registered tensor.
    #         residual_parameter_count: Registered floating-point
    #             parameters alone, so a quantized network reports both its
    #             logical weight count and its unquantized remainder.
    #         model_size_megabytes: In-memory size of registered parameters
    #             and buffers.
    #         deployable_size_megabytes: Serialized size of the persisted
    #             state dictionary, which is what an artifact actually
    #             occupies.
    #         real_time_factor: Warm real-time factor, the synthesis
    #             duration divided by the audio duration, measured after
    #             warm-up batches are excluded. A value below one denotes
    #             synthesis faster than real time. The figure is conditional
    #             on the execution lane that produced it, and the report
    #             never compares a value from one lane against another.
    #         macs_per_second_audio: Multiply-accumulate operations per
    #             second of synthesized audio.
    #         latency_p50_ms: Median per-utterance synthesis latency in
    #             milliseconds, produced only when timed batches hold a
    #             single utterance each. The percentile is taken over the
    #             per-utterance means of the repeated timed calls, not over
    #             the individual calls, so one slow call cannot become a
    #             reported percentile on its own.
    #         latency_p95_ms: The corresponding ninety-fifth percentile over
    #             the same per-utterance means.
    #     Measurement denominators, each optional:
    #         timed_utterance_count: Utterances that entered the timing
    #             measurement.
    #         rtf_warmup_iterations: Warm-up passes excluded from timing.
    #         rtf_repetitions_executed: Measured repetitions actually run.
    #         test_utterance_count: Utterances the quality panel was
    #             computed over.
    #     Per-metric failure counts, each optional:
    #         pesq_failure_count, stoi_failure_count,
    #         mel_error_failure_count,
    #         multi_resolution_stft_error_failure_count, mcd_failure_count,
    #         las_rmse_failure_count, utmos_strong_failure_count:
    #             Utterances on which that metric could not be computed.
    #             Each accompanies its mean so a reviewer can judge the mean
    #             against how many utterances produced it and how many were
    #             lost, rather than reading a mean over an unknown
    #             denominator.
    #
    # Every measurement, denominator, and failure column is optional because
    # evidence categories differ in which metrics they can produce. An
    # absent value is None here and an empty cell in the CSV, never a
    # fabricated zero, so a metric that was never run stays distinguishable
    # from one that measured zero. The model is frozen, strict, and
    # extra-forbidding, so a row cannot be mutated after construction and a
    # misspelled column is rejected instead of silently ignored.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    csv_columns: ClassVar[tuple[str, ...]] = (
        "unique_id",
        "date",
        "architecture_name",
        "dataset_name",
        "variant_name",
        "seed",
        "hyperparameters_summary",
        "parameter_count",
        "model_size_megabytes",
        "real_time_factor",
        "mel_error",
        "pesq",
        "stoi",
        "multi_resolution_stft_error",
        "mcd",
        "las_rmse",
        "f0_rmse_cents",
        "periodicity_rmse",
        "vuv_f1",
        "utmos_strong",
        "macs_per_second_audio",
        "residual_parameter_count",
        "deployable_size_megabytes",
        "latency_p50_ms",
        "latency_p95_ms",
        "timed_utterance_count",
        "rtf_warmup_iterations",
        "rtf_repetitions_executed",
        "test_utterance_count",
        "pesq_failure_count",
        "stoi_failure_count",
        "mel_error_failure_count",
        "multi_resolution_stft_error_failure_count",
        "mcd_failure_count",
        "las_rmse_failure_count",
        "utmos_strong_failure_count",
        "interpretation_notes"
    )

    unique_id: str
    date: str
    architecture_name: str
    dataset_name: str
    variant_name: str
    seed: int
    hyperparameters_summary: str
    parameter_count: int | None
    model_size_megabytes: float | None
    real_time_factor: float | None
    mel_error: float | None
    pesq: float | None
    stoi: float | None
    multi_resolution_stft_error: float | None
    mcd: float | None
    las_rmse: float | None
    f0_rmse_cents: float | None
    periodicity_rmse: float | None
    vuv_f1: float | None
    utmos_strong: float | None
    macs_per_second_audio: float | None
    residual_parameter_count: int | None
    deployable_size_megabytes: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    timed_utterance_count: int | None
    rtf_warmup_iterations: int | None
    rtf_repetitions_executed: int | None
    test_utterance_count: int | None
    pesq_failure_count: int | None
    stoi_failure_count: int | None
    mel_error_failure_count: int | None
    multi_resolution_stft_error_failure_count: int | None
    mcd_failure_count: int | None
    las_rmse_failure_count: int | None
    utmos_strong_failure_count: int | None
    interpretation_notes: str

    @classmethod
    def from_metric_buffer(
        cls,
        *,
        unique_id: str,
        architecture_name: str,
        dataset_name: str,
        variant_name: str,
        seed: int,
        hyperparameters_summary: str,
        interpretation_notes: str,
        metric_buffer: dict[str, float | int]
    ) -> ExperimentResultRow:
        # Builds the row from the run's metric buffer: identity fields come
        # from the caller, measurement fields are read by their buffer keys
        # with absent keys mapping to None, and the timestamp is stamped in
        # UTC at this moment.
        #
        # The buffer is read by name rather than by position, so a metric the
        # run never produced is absent from the buffer and its column becomes
        # None. No key is required and no default is substituted, which is
        # what keeps an unproduced measurement distinguishable from a
        # measured zero all the way to the CSV cell. The parameter count is
        # converted explicitly rather than through the shared integer helper
        # because it is the one measurement whose column the buffer may hold
        # as a float.
        #
        # Args:
        #     unique_id: Stable per-run identifier for the identity column.
        #     architecture_name: Vocoder family under test.
        #     dataset_name: Corpus the run evaluated against.
        #     variant_name: Variant label, already prefixed with the
        #         evidence family by the runner.
        #     seed: Seed governing the run.
        #     hyperparameters_summary: Compact rendering of the settings
        #         that distinguish this run.
        #     interpretation_notes: Analyst-facing free text carried
        #         through to the row unmodified.
        #     metric_buffer: The run's accumulated metrics keyed by metric
        #         name. Keys the schema does not name are ignored, so an
        #         evaluator may buffer more than the summary records.
        #
        # Returns:
        #     A frozen ExperimentResultRow whose date field is the UTC
        #     instant of this call, not of the measurement, so rows are
        #     totally ordered across machines without local-timezone
        #     ambiguity.
        raw_parameter_count: float | int | None = metric_buffer.get("parameter_count")
        parameter_count: int | None = (
            int(raw_parameter_count) if raw_parameter_count is not None else None
        )
        return cls(
            unique_id=unique_id,
            date=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            architecture_name=architecture_name,
            dataset_name=dataset_name,
            variant_name=variant_name,
            seed=seed,
            hyperparameters_summary=hyperparameters_summary,
            parameter_count=parameter_count,
            model_size_megabytes=cls._optional_float(metric_buffer, "model_size_megabytes"),
            real_time_factor=cls._optional_float(metric_buffer, "real_time_factor"),
            mel_error=cls._optional_float(metric_buffer, "mel_error"),
            pesq=cls._optional_float(metric_buffer, "pesq"),
            stoi=cls._optional_float(metric_buffer, "stoi"),
            multi_resolution_stft_error=cls._optional_float(
                metric_buffer,
                "multi_resolution_stft_error"
            ),
            mcd=cls._optional_float(metric_buffer, "mcd"),
            las_rmse=cls._optional_float(metric_buffer, "las_rmse"),
            f0_rmse_cents=cls._optional_float(metric_buffer, "f0_rmse_cents"),
            periodicity_rmse=cls._optional_float(metric_buffer, "periodicity_rmse"),
            vuv_f1=cls._optional_float(metric_buffer, "vuv_f1"),
            utmos_strong=cls._optional_float(metric_buffer, "utmos_strong"),
            macs_per_second_audio=cls._optional_float(
                metric_buffer,
                "macs_per_second_audio"
            ),
            residual_parameter_count=cls._optional_int(metric_buffer, "residual_parameter_count"),
            deployable_size_megabytes=cls._optional_float(metric_buffer, "deployable_size_megabytes"),
            latency_p50_ms=cls._optional_float(metric_buffer, "latency_p50_ms"),
            latency_p95_ms=cls._optional_float(metric_buffer, "latency_p95_ms"),
            timed_utterance_count=cls._optional_int(metric_buffer, "timed_utterance_count"),
            rtf_warmup_iterations=cls._optional_int(metric_buffer, "rtf_warmup_iterations"),
            rtf_repetitions_executed=cls._optional_int(metric_buffer, "rtf_repetitions_executed"),
            test_utterance_count=cls._optional_int(metric_buffer, "test_utterance_count"),
            pesq_failure_count=cls._optional_int(metric_buffer, "pesq_failure_count"),
            stoi_failure_count=cls._optional_int(metric_buffer, "stoi_failure_count"),
            mel_error_failure_count=cls._optional_int(metric_buffer, "mel_error_failure_count"),
            multi_resolution_stft_error_failure_count=cls._optional_int(
                metric_buffer,
                "multi_resolution_stft_error_failure_count"
            ),
            mcd_failure_count=cls._optional_int(metric_buffer, "mcd_failure_count"),
            las_rmse_failure_count=cls._optional_int(metric_buffer, "las_rmse_failure_count"),
            utmos_strong_failure_count=cls._optional_int(metric_buffer, "utmos_strong_failure_count"),
            interpretation_notes=interpretation_notes
        )

    def to_csv_row(self) -> dict[str, str | int | float | None]:
        # Serializes the record into the csv_columns mapping; None values
        # become empty CSV cells, preserving the distinction between a
        # measured zero and an unproduced measurement.
        #
        # The mapping is written out explicitly rather than derived from the
        # model dump, so the serialization and the schema tuple are two
        # independent statements of the same contract and a field added to
        # one without the other is visible rather than silently dropped.
        #
        # Returns:
        #     A mapping whose keys are exactly csv_columns, ready for the
        #     writer's DictWriter. Values are left as their native Python
        #     types; the csv module renders None as an empty field.
        return {
            "unique_id": self.unique_id,
            "date": self.date,
            "architecture_name": self.architecture_name,
            "dataset_name": self.dataset_name,
            "variant_name": self.variant_name,
            "seed": self.seed,
            "hyperparameters_summary": self.hyperparameters_summary,
            "parameter_count": self.parameter_count,
            "model_size_megabytes": self.model_size_megabytes,
            "real_time_factor": self.real_time_factor,
            "mel_error": self.mel_error,
            "pesq": self.pesq,
            "stoi": self.stoi,
            "multi_resolution_stft_error": self.multi_resolution_stft_error,
            "mcd": self.mcd,
            "las_rmse": self.las_rmse,
            "f0_rmse_cents": self.f0_rmse_cents,
            "periodicity_rmse": self.periodicity_rmse,
            "vuv_f1": self.vuv_f1,
            "utmos_strong": self.utmos_strong,
            "macs_per_second_audio": self.macs_per_second_audio,
            "residual_parameter_count": self.residual_parameter_count,
            "deployable_size_megabytes": self.deployable_size_megabytes,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "timed_utterance_count": self.timed_utterance_count,
            "rtf_warmup_iterations": self.rtf_warmup_iterations,
            "rtf_repetitions_executed": self.rtf_repetitions_executed,
            "test_utterance_count": self.test_utterance_count,
            "pesq_failure_count": self.pesq_failure_count,
            "stoi_failure_count": self.stoi_failure_count,
            "mel_error_failure_count": self.mel_error_failure_count,
            "multi_resolution_stft_error_failure_count": self.multi_resolution_stft_error_failure_count,
            "mcd_failure_count": self.mcd_failure_count,
            "las_rmse_failure_count": self.las_rmse_failure_count,
            "utmos_strong_failure_count": self.utmos_strong_failure_count,
            "interpretation_notes": self.interpretation_notes
        }

    @staticmethod
    def _optional_float(
        metric_buffer: dict[str, float | int],
        metric_name: str
    ) -> float | None:
        # Reads one buffer key as a float, preserving absence as None. The
        # conversion is unconditional for a present value, so an integer
        # buffered under a float column becomes a float rather than
        # tripping the model's strict type checking.
        value: float | int | None = metric_buffer.get(metric_name)
        return float(value) if value is not None else None

    @staticmethod
    def _optional_int(
        metric_buffer: dict[str, float | int],
        metric_name: str
    ) -> int | None:
        # Reads one buffer key as an integer, preserving absence as None.
        # Counts reach the buffer as floats when they pass through the
        # harness metric path, so the conversion truncates toward zero
        # rather than rejecting them; the columns this serves are all
        # non-negative counts, where truncation and rounding agree for the
        # integral values actually produced.
        value: float | int | None = metric_buffer.get(metric_name)
        return int(value) if value is not None else None
