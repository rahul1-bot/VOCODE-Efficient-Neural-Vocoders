# This module:
# 1. Samples step time, dataloader wait time, iterator-creation time, CPU
#    utilization, and GPU memory and utilization counters on a configurable
#    batch cadence across all four execution stages
# 2. Persists three durable artifacts beside the experiment logs: an
#    append-only JSONL event stream, a continuously rewritten JSON summary of
#    per-metric statistics, and a configuration sidecar recording the sampling
#    cadence
# 3. Emits every sampled value through the experiment logger under runtime/
#    metric keys
#
# Design decisions:
# - Dataloader wait time prefers the trainer's measured fetch duration and
#   falls back to the gap since the previous batch end, so the reported wait
#   reflects true input starvation wherever instrumentation exists
# - The accelerator is synchronized before sampled timing reads, so queued
#   asynchronous kernels are charged to the batch that launched them; the
#   synchronization runs only on sampled batches to bound its overhead
# - The first batch of every stage is always sampled because it carries
#   one-time costs (worker startup, compilation, cache warmup) that a cadence
#   starting later would miss; training thereafter samples on the global-step
#   cadence and evaluation stages on the batch-index cadence
# - The summary file is rewritten after every emission and again at teardown
#   and on exceptions, so an interrupted run still leaves a valid summary; the
#   existing summary is reloaded at setup so sequential dispatches within one
#   run directory accumulate into one summary
# - GPU utilization and power draw are collected through nvidia-smi as a
#   bounded best-effort subprocess with a timeout, degrading silently where
#   the binary is unavailable
# - All JSON artifacts are written with NaN forbidden, and non-finite samples
#   are dropped at the emission boundary, keeping every artifact strictly
#   parseable
#
# Author: Rahul Sawhney

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Literal, override

import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.core.trainer import Trainer
from syntheticmind.utilities.distributed import is_rank_zero
from syntheticmind.utilities.types import Batch, ModelOutput, RunningStage, StepOutput, TrainerStage

__all__: list[str] = ["RuntimeProfiler"]

type ProfileStageName = Literal["train", "validation", "test", "prediction"]
type RuntimeRecordValue = bool | int | float | str | None


class RuntimeMetricAccumulator:
    # Streaming summary of one runtime metric: observation count, running
    # total for the mean, extremes, and the latest value. Only finite values
    # are admitted so the serialized summary is always strict JSON.
    def __init__(self) -> None:
        # Starts an empty summary; extremes and the latest value remain None
        # until the first finite observation arrives.
        self.count: int = 0
        self.total: float = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.latest: float | None = None

    def update(self, value: float) -> None:
        # Records one finite scalar value into the running summary.
        if not math.isfinite(value):
            return
        self.count += 1
        self.total += value
        self.latest: float | None = value
        self.minimum: float | None = value if self.minimum is None else min(self.minimum, value)
        self.maximum: float | None = value if self.maximum is None else max(self.maximum, value)

    def as_record(self) -> dict[str, float | int | None]:
        # Serializes this summary into a stable JSON-compatible mapping.
        mean_value: float | None = None
        if self.count > 0:
            mean_value: float | None = self.total / self.count
        return {
            "count": self.count,
            "mean": mean_value,
            "min": self.minimum,
            "max": self.maximum,
            "latest": self.latest
        }

    @classmethod
    def from_record(cls, record: dict[str, RuntimeRecordValue]) -> RuntimeMetricAccumulator:
        # Restores a summary accumulator from a prior profiler artifact.
        accumulator: RuntimeMetricAccumulator = cls()
        count_value: RuntimeRecordValue = record.get("count")
        mean_value: RuntimeRecordValue = record.get("mean")
        if not isinstance(count_value, int) or count_value < 1:
            return accumulator
        if not isinstance(mean_value, int | float):
            return accumulator
        accumulator.count: int = count_value
        accumulator.total: float = float(mean_value) * count_value
        minimum_value: RuntimeRecordValue = record.get("min")
        maximum_value: RuntimeRecordValue = record.get("max")
        latest_value: RuntimeRecordValue = record.get("latest")
        accumulator.minimum: float | None = float(minimum_value) if isinstance(minimum_value, int | float) else None
        accumulator.maximum: float | None = float(maximum_value) if isinstance(maximum_value, int | float) else None
        accumulator.latest: float | None = float(latest_value) if isinstance(latest_value, int | float) else None
        return accumulator


class RuntimeProfiler(Callback):
    # Runtime bottleneck profiler over the training, validation, test, and
    # prediction loops. Sampled batch events capture where wall-clock time
    # goes (step versus dataloader wait) alongside CPU and GPU resource
    # counters, and every artifact write is durable so interrupted cloud runs
    # still leave usable evidence.
    def __init__(
        self,
        profile_every_n_steps: int = 50,
        include_cpu_metrics: bool = True,
        include_gpu_metrics: bool = True
    ) -> None:
        # Validates the sampling cadence and prepares the per-stage timing
        # slots, the artifact path slots populated at setup, the metric
        # summaries, and the CPU-utilization baseline counters.
        super().__init__()
        if profile_every_n_steps < 1:
            from syntheticmind.utilities.exceptions import MisconfigurationError
            raise MisconfigurationError("profile_every_n_steps must be >= 1")
        self._profile_every_n_steps: int = profile_every_n_steps
        self._include_cpu_metrics: bool = include_cpu_metrics
        self._include_gpu_metrics: bool = include_gpu_metrics
        self._metrics_directory: Path | None = None
        self._profile_events_path: Path | None = None
        self._profile_summary_path: Path | None = None
        self._profile_config_path: Path | None = None
        self._stage_start_times: dict[ProfileStageName, float] = {}
        self._batch_start_times: dict[ProfileStageName, float] = {}
        self._last_batch_end_times: dict[ProfileStageName, float] = {}
        self._current_wait_times: dict[ProfileStageName, float] = {}
        self._current_iterator_creation_times: dict[ProfileStageName, float] = {}
        self._metric_summaries: dict[str, RuntimeMetricAccumulator] = {}
        self._last_resource_wall_time: float = time.perf_counter()
        self._last_process_cpu_time: float = time.process_time()
        self._cpu_count: int = max(1, os.cpu_count() or 1)

    @override
    def setup(self, trainer: Trainer, module: Module, stage: TrainerStage) -> None:
        # Resolves the artifact paths inside the logger's metrics directory,
        # reloads any existing summary so sequential dispatches accumulate,
        # and records the profiling configuration sidecar. Without a logger
        # the profiler stays disabled for the stage.
        if not is_rank_zero():
            return
        if trainer.logger is None:
            return
        self._metrics_directory: Path | None = trainer.logger.save_dir / "metrics"
        self._metrics_directory.mkdir(parents=True, exist_ok=True)
        self._profile_events_path: Path | None = self._metrics_directory / "runtime_profile.jsonl"
        self._profile_summary_path: Path | None = self._metrics_directory / "runtime_profile_summary.json"
        self._profile_config_path: Path | None = self._metrics_directory / "runtime_profile_config.json"
        self._load_existing_profile_summary()
        self._write_profile_config(stage)

    @override
    def teardown(self, trainer: Trainer, module: Module, stage: TrainerStage) -> None:
        # Writes the final summary at stage teardown so the artifact reflects
        # every sample the stage produced.
        self._write_profile_summary()

    @override
    def on_train_start(self, trainer: Trainer, module: Module) -> None:
        # Opens the training-stage duration measurement.
        self._mark_stage_start("train")

    @override
    def on_train_end(self, trainer: Trainer, module: Module) -> None:
        # Closes the training-stage measurement and emits its total duration.
        self._record_stage_end(trainer, module, "train", "runtime/train_time_ms")

    @override
    def on_train_epoch_start(self, trainer: Trainer, module: Module) -> None:
        # Re-baselines the inter-batch timestamp at the epoch boundary so the
        # first batch's wait measurement excludes epoch-end work.
        self._last_batch_end_times["train"] = time.perf_counter()

    @override
    def on_train_batch_start(
        self, trainer: Trainer, module: Module, batch: Batch, batch_idx: int
    ) -> None:
        # Marks the training batch start and captures its dataloader wait.
        self._mark_batch_start(trainer, "train")

    @override
    def on_train_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int
    ) -> None:
        # Records the sampled training batch event with its runtime metrics.
        self._record_batch_end(
            trainer=trainer,
            module=module,
            stage_name="train",
            event_name="train_batch",
            batch_idx=batch_idx,
            step_metric_name="runtime/train_step_time_ms",
            wait_metric_name="runtime/train_dataloader_wait_time_ms",
            iterator_creation_metric_name="runtime/train_dataloader_iterator_creation_time_ms"
        )

    @override
    def on_validation_start(self, trainer: Trainer, module: Module) -> None:
        # Opens the validation-stage measurement; sanity-check passes are
        # excluded so their bounded runs never contaminate the telemetry.
        if trainer._sanity_checking:
            return
        self._mark_stage_start("validation")
        self._last_batch_end_times["validation"] = time.perf_counter()

    @override
    def on_validation_end(self, trainer: Trainer, module: Module) -> None:
        # Closes the validation-stage measurement outside sanity checking.
        if trainer._sanity_checking:
            return
        self._record_stage_end(trainer, module, "validation", "runtime/validation_time_ms")

    @override
    def on_validation_batch_start(
        self, trainer: Trainer, module: Module, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Marks the validation batch start outside sanity checking.
        if trainer._sanity_checking:
            return
        self._mark_batch_start(trainer, "validation")

    @override
    def on_validation_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Records the sampled validation batch event outside sanity checking.
        if trainer._sanity_checking:
            return
        self._record_batch_end(
            trainer=trainer,
            module=module,
            stage_name="validation",
            event_name="validation_batch",
            batch_idx=batch_idx,
            step_metric_name="runtime/validation_step_time_ms",
            wait_metric_name="runtime/validation_dataloader_wait_time_ms",
            iterator_creation_metric_name="runtime/validation_dataloader_iterator_creation_time_ms"
        )

    @override
    def on_test_start(self, trainer: Trainer, module: Module) -> None:
        # Opens the test-stage measurement and baselines the batch timestamp.
        self._mark_stage_start("test")
        self._last_batch_end_times["test"] = time.perf_counter()

    @override
    def on_test_end(self, trainer: Trainer, module: Module) -> None:
        # Closes the test-stage measurement and emits its total duration.
        self._record_stage_end(trainer, module, "test", "runtime/test_time_ms")

    @override
    def on_test_batch_start(
        self, trainer: Trainer, module: Module, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Marks the test batch start and captures its dataloader wait.
        self._mark_batch_start(trainer, "test")

    @override
    def on_test_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Records the sampled test batch event with its runtime metrics.
        self._record_batch_end(
            trainer=trainer,
            module=module,
            stage_name="test",
            event_name="test_batch",
            batch_idx=batch_idx,
            step_metric_name="runtime/test_step_time_ms",
            wait_metric_name="runtime/test_dataloader_wait_time_ms",
            iterator_creation_metric_name="runtime/test_dataloader_iterator_creation_time_ms"
        )

    @override
    def on_predict_start(self, trainer: Trainer, module: Module) -> None:
        # Opens the prediction-stage measurement and baselines the batch
        # timestamp.
        self._mark_stage_start("prediction")
        self._last_batch_end_times["prediction"] = time.perf_counter()

    @override
    def on_predict_end(self, trainer: Trainer, module: Module) -> None:
        # Closes the prediction-stage measurement and emits its total
        # duration.
        self._record_stage_end(trainer, module, "prediction", "runtime/prediction_time_ms")

    @override
    def on_predict_batch_start(
        self, trainer: Trainer, module: Module, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Marks the prediction batch start and captures its dataloader wait.
        self._mark_batch_start(trainer, "prediction")

    @override
    def on_predict_batch_end(
        self, trainer: Trainer, module: Module, outputs: ModelOutput, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Records the sampled prediction batch event with its runtime
        # metrics.
        self._record_batch_end(
            trainer=trainer,
            module=module,
            stage_name="prediction",
            event_name="prediction_batch",
            batch_idx=batch_idx,
            step_metric_name="runtime/prediction_step_time_ms",
            wait_metric_name="runtime/prediction_dataloader_wait_time_ms",
            iterator_creation_metric_name="runtime/prediction_dataloader_iterator_creation_time_ms"
        )

    @override
    def on_exception(self, trainer: Trainer, module: Module, exception: BaseException) -> None:
        # Persists the summary on failure so the telemetry collected before
        # the exception survives the run.
        self._write_profile_summary()

    def _mark_stage_start(self, stage_name: ProfileStageName) -> None:
        # Records stage start time for later duration telemetry.
        self._stage_start_times[stage_name] = time.perf_counter()

    def _record_stage_end(
        self,
        trainer: Trainer,
        module: Module,
        stage_name: ProfileStageName,
        metric_name: str
    ) -> None:
        # Records total elapsed time for a trainer stage.
        start_time: float | None = self._stage_start_times.get(stage_name)
        if start_time is None:
            return
        self._synchronize_cuda_if_needed(trainer)
        duration_ms: float = (time.perf_counter() - start_time) * 1000.0
        metrics: dict[str, float] = {metric_name: duration_ms}
        self._emit_metrics(trainer, module, f"{stage_name}_end", None, metrics)

    def _mark_batch_start(self, trainer: Trainer, stage_name: ProfileStageName) -> None:
        # Records batch start and the elapsed time spent waiting for the dataloader.
        current_time: float = time.perf_counter()
        previous_end_time: float | None = self._last_batch_end_times.get(stage_name)
        wait_seconds: float = 0.0
        dataloader_fetch_seconds: float | None = trainer._get_dataloader_fetch_time(
            self._to_running_stage(stage_name)
        )
        dataloader_iterator_creation_seconds: float | None = (
            trainer._consume_dataloader_iterator_creation_time(self._to_running_stage(stage_name))
        )
        if dataloader_fetch_seconds is not None:
            wait_seconds: float = dataloader_fetch_seconds
        elif previous_end_time is not None:
            wait_seconds: float = max(0.0, current_time - previous_end_time)
        self._current_wait_times[stage_name] = wait_seconds
        if dataloader_iterator_creation_seconds is not None:
            self._current_iterator_creation_times[stage_name] = dataloader_iterator_creation_seconds
        self._batch_start_times[stage_name] = current_time

    def _to_running_stage(self, stage_name: ProfileStageName) -> RunningStage:
        # Maps profiler stage names to the trainer running-stage contract.
        match stage_name:
            case "train":
                return "training"
            case "validation":
                return "validating"
            case "test":
                return "testing"
            case "prediction":
                return "predicting"

    def _record_batch_end(
        self,
        trainer: Trainer,
        module: Module,
        stage_name: ProfileStageName,
        event_name: str,
        batch_idx: int,
        step_metric_name: str,
        wait_metric_name: str,
        iterator_creation_metric_name: str
    ) -> None:
        # Records one profiled batch event with optional resource telemetry.
        if not self._should_emit_batch_event(trainer, stage_name, batch_idx):
            self._last_batch_end_times[stage_name] = time.perf_counter()
            return
        self._synchronize_cuda_if_needed(trainer)
        current_time: float = time.perf_counter()
        batch_start_time: float | None = self._batch_start_times.get(stage_name)
        step_seconds: float = 0.0
        if batch_start_time is not None:
            step_seconds: float = max(0.0, current_time - batch_start_time)
        wait_seconds: float = self._current_wait_times.get(stage_name, 0.0)
        self._last_batch_end_times[stage_name] = current_time
        metrics: dict[str, float] = {
            step_metric_name: step_seconds * 1000.0,
            wait_metric_name: wait_seconds * 1000.0
        }
        iterator_creation_seconds: float | None = self._current_iterator_creation_times.pop(
            stage_name, None
        )
        if iterator_creation_seconds is not None:
            metrics[iterator_creation_metric_name] = iterator_creation_seconds * 1000.0
        metrics.update(self._collect_resource_metrics(trainer))
        self._emit_metrics(trainer, module, event_name, batch_idx, metrics)

    def _should_emit_batch_event(
        self,
        trainer: Trainer,
        stage_name: ProfileStageName,
        batch_idx: int
    ) -> bool:
        # Applies the configured sampling cadence to avoid excessive profiling overhead.
        if batch_idx == 0:
            return True
        if stage_name == "train":
            step: int = max(0, trainer.state.global_step)
            return step > 0 and step % self._profile_every_n_steps == 0
        return (batch_idx + 1) % self._profile_every_n_steps == 0

    def _collect_resource_metrics(self, trainer: Trainer) -> dict[str, float]:
        # Collects CPU and GPU resource counters for a profiler sample.
        metrics: dict[str, float] = {}
        if self._include_cpu_metrics:
            metrics.update(self._collect_cpu_metrics())
        if self._include_gpu_metrics:
            metrics.update(self._collect_gpu_metrics(trainer))
        return metrics

    def _collect_cpu_metrics(self) -> dict[str, float]:
        # Collects process CPU utilization and host load metrics.
        current_wall_time: float = time.perf_counter()
        current_process_time: float = time.process_time()
        elapsed_wall_time: float = max(current_wall_time - self._last_resource_wall_time, 1.0e-9)
        elapsed_process_time: float = max(current_process_time - self._last_process_cpu_time, 0.0)
        self._last_resource_wall_time: float = current_wall_time
        self._last_process_cpu_time: float = current_process_time
        metrics: dict[str, float] = {
            "runtime/process_cpu_utilization_percent": (
                elapsed_process_time / (elapsed_wall_time * self._cpu_count)
            ) * 100.0
        }
        if hasattr(os, "getloadavg"):
            load_average_1m: float = os.getloadavg()[0]
            metrics["runtime/cpu_load_average_1m"] = load_average_1m
            metrics["runtime/cpu_load_average_1m_per_core_percent"] = (
                load_average_1m / self._cpu_count
            ) * 100.0
        return metrics

    def _collect_gpu_metrics(self, trainer: Trainer) -> dict[str, float]:
        # Collects CUDA memory counters and best-effort device utilization.
        device: torch.device = trainer.strategy.root_device
        if device.type == "mps":
            return {
                "runtime/mps_memory_allocated_mb": torch.mps.current_allocated_memory() / 1048576.0,
                "runtime/mps_memory_driver_mb": torch.mps.driver_allocated_memory() / 1048576.0
            }
        if not torch.cuda.is_available():
            return {}
        if device.type != "cuda":
            return {}
        device_index: int = device.index if device.index is not None else torch.cuda.current_device()
        metrics: dict[str, float] = {
            "runtime/gpu_memory_allocated_mb": torch.cuda.memory_allocated(device_index) / 1048576.0,
            "runtime/gpu_memory_reserved_mb": torch.cuda.memory_reserved(device_index) / 1048576.0,
            "runtime/gpu_memory_max_allocated_mb": torch.cuda.max_memory_allocated(device_index) / 1048576.0
        }
        try:
            free_bytes: int
            total_bytes: int
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            metrics["runtime/gpu_memory_free_mb"] = free_bytes / 1048576.0
            metrics["runtime/gpu_memory_total_mb"] = total_bytes / 1048576.0
        except RuntimeError:
            pass
        metrics.update(self._collect_nvidia_smi_metrics(device_index))
        return metrics

    def _collect_nvidia_smi_metrics(self, device_index: int) -> dict[str, float]:
        # Collects utilization counters through nvidia-smi when the binary is available.
        command: list[str] = [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits"
        ]
        try:
            completed_process: subprocess.CompletedProcess[str] = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return {}
        if completed_process.returncode != 0:
            return {}
        stdout_lines: list[str] = completed_process.stdout.strip().splitlines()
        if not stdout_lines:
            return {}
        first_line: str = stdout_lines[0]
        raw_values: list[str] = [value.strip() for value in first_line.split(",")]
        if len(raw_values) != 4:
            return {}
        metric_names: tuple[str, str, str, str] = (
            "runtime/gpu_utilization_percent",
            "runtime/gpu_memory_used_mb",
            "runtime/gpu_memory_total_smi_mb",
            "runtime/gpu_power_draw_watts"
        )
        metrics: dict[str, float] = {}
        for metric_name, raw_value in zip(metric_names, raw_values, strict=True):
            try:
                metrics[metric_name] = float(raw_value)
            except ValueError:
                continue
        return metrics

    def _emit_metrics(
        self,
        trainer: Trainer,
        module: Module,
        event_name: str,
        batch_idx: int | None,
        metrics: dict[str, float]
    ) -> None:
        # Logs scalar telemetry and appends the profiler event artifact.
        finite_metrics: dict[str, float] = self._finite_metrics(metrics)
        if not finite_metrics:
            return
        for metric_name, value in finite_metrics.items():
            self._metric_summaries.setdefault(metric_name, RuntimeMetricAccumulator()).update(value)
        if trainer.logger is not None and is_rank_zero():
            trainer.logger.log_metrics(finite_metrics, step=trainer.state.global_step)
        self._append_profile_event(trainer, module, event_name, batch_idx, finite_metrics)
        self._write_profile_summary()

    def _append_profile_event(
        self,
        trainer: Trainer,
        module: Module,
        event_name: str,
        batch_idx: int | None,
        metrics: dict[str, float]
    ) -> None:
        # Writes one profiler JSONL event to the current run artifact directory.
        if not is_rank_zero() or self._profile_events_path is None:
            return
        payload: dict[str, RuntimeRecordValue | dict[str, float]] = {
            "event": event_name,
            "time_unix": time.time(),
            "epoch": module.current_epoch,
            "global_step": trainer.state.global_step,
            "batch_idx": batch_idx,
            "running_stage": trainer.state.running_stage,
            "metrics": metrics
        }
        self._profile_events_path.parent.mkdir(parents=True, exist_ok=True)
        with self._profile_events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, sort_keys=True, allow_nan=False))
            file.write("\n")

    def _write_profile_config(self, stage: TrainerStage) -> None:
        # Writes profiler configuration so telemetry cadence is auditable.
        if not is_rank_zero() or self._profile_config_path is None:
            return
        payload: dict[str, RuntimeRecordValue] = {
            "profile_every_n_steps": self._profile_every_n_steps,
            "include_cpu_metrics": self._include_cpu_metrics,
            "include_gpu_metrics": self._include_gpu_metrics,
            "stage_at_setup": stage,
            "cpu_count": self._cpu_count
        }
        self._profile_config_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8"
        )

    def _load_existing_profile_summary(self) -> None:
        # Loads prior summary state so sequential trainer dispatches share one run summary.
        if self._profile_summary_path is None or not self._profile_summary_path.exists():
            return
        try:
            loaded_payload: object = json.loads(self._profile_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(loaded_payload, dict):
            return
        for metric_name, record in loaded_payload.items():
            if not isinstance(metric_name, str) or not isinstance(record, dict):
                continue
            parsed_record: dict[str, RuntimeRecordValue] = {}
            for key, value in record.items():
                if isinstance(key, str) and isinstance(value, bool | int | float | str | None):
                    parsed_record[key] = value
            self._metric_summaries[metric_name] = RuntimeMetricAccumulator.from_record(parsed_record)

    def _write_profile_summary(self) -> None:
        # Writes the latest profiler summary into a stable JSON artifact.
        if not is_rank_zero() or self._profile_summary_path is None:
            return
        payload: dict[str, dict[str, float | int | None]] = {
            metric_name: metric_summary.as_record()
            for metric_name, metric_summary in sorted(self._metric_summaries.items())
        }
        self._profile_summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8"
        )

    def _finite_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        # Filters non-finite metrics before logging or writing strict JSON.
        return {
            key: float(value)
            for key, value in metrics.items()
            if math.isfinite(float(value))
        }

    def _synchronize_cuda_if_needed(self, trainer: Trainer) -> None:
        # Synchronizes the active accelerator before sampled timing reads so queued work is included.
        if not self._include_gpu_metrics:
            return
        device: torch.device = trainer.strategy.root_device
        if device.type == "mps":
            torch.mps.synchronize()
            return
        if not torch.cuda.is_available():
            return
        if device.type != "cuda":
            return
        torch.cuda.synchronize(device)
