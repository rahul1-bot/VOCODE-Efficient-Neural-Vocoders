# This module:
# 1. Measures the real-time factor over the prediction stage: warm-up
#    batches excluded, every timed batch executed for a configured number of
#    measured repetitions through the module's own prediction step
# 2. Reports the timing denominators alongside the ratio (timed utterance
#    count, warm-up count, executed repetitions) and adds latency
#    percentiles when batches contain single utterances
#
# Harness contract (syntheticmind):
# - Subclasses the harness Callback and rides the prediction loop; timing
#   state resets at predict start and the final ratio publishes through the
#   experiment logger at predict end
# - Repetitions call the module's predict_step directly under inference
#   mode, so the timed protocol is exactly the executed synthesis protocol
#
# Design decisions:
# - The accelerator is synchronized on both sides of every repetition, so
#   queued kernels fall inside the timed span rather than leaking into the
#   next measurement
# - The real-time factor is total inference time over total audio duration
#   across timed batches, which weights utterances by their length instead
#   of averaging per-utterance ratios
# - A run that produces zero timed samples fails loudly instead of
#   recording a zero, because a zero real-time factor is a claim, not an
#   absence
# - Latency percentiles are reported only for batch-size-one timing, where
#   a batch mean is a genuine per-utterance latency
#
# Author: Rahul Sawhney

import time
from typing import ClassVar, override

import torch
from loguru import logger as log
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.core.trainer import Trainer
from syntheticmind.utilities.types import Batch, ModelOutput

__all__: list[str] = ["RealTimeFactorConfig", "RealTimeFactorMonitor"]


class RealTimeFactorConfig(BaseModel):
    # Frozen timing protocol.
    #
    # Fields:
    #     warmup_iterations: Batches executed and discarded before timing
    #         starts, absorbing lazy initialization, allocator growth, and
    #         compilation. Default: ``5``.
    #     timed_repetitions: Measured synthesis repetitions per timed
    #         batch; the batch contributes the mean of its repetitions.
    #         Default: ``3``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    warmup_iterations: NonNegativeInt = 5
    timed_repetitions: PositiveInt = 3


class RealTimeFactorMonitor(Callback):
    # Prediction-stage timing driver. Warm-up batches pass through
    # untimed; each subsequent batch is re-synthesized the configured
    # number of times with accelerator synchronization around every
    # repetition, and the final ratio is total inference time over total
    # audio duration.
    #
    # The measurement mechanism runs entirely inside the batch-end hook.
    # The prediction loop has already synthesized the batch by the time the
    # hook fires, so the first warmup_iterations batches are consumed by
    # decrementing a countdown and nothing more. Once the countdown reaches
    # zero every batch is re-synthesized timed_repetitions further times
    # through the module's own predict_step under inference mode, which
    # makes the timed protocol the executed synthesis protocol rather than a
    # reconstruction of it. The accelerator is synchronized immediately
    # before the clock starts and again before it stops, so kernels queued
    # asynchronously are charged to the repetition that issued them instead
    # of leaking into the next one. A batch contributes the arithmetic mean
    # of its repetitions, not their sum, so the repetition count controls
    # measurement noise without changing the scale of the result.
    #
    # Reduction sums the per-batch mean times and divides by the summed
    # audio durations, which weights each utterance by its own length rather
    # than averaging per-utterance ratios; a pass that reached reduction
    # with no timed batch raises instead of publishing a zero. Because
    # every timed call follows an untimed prediction of the same input, the
    # published quantity is a warm real-time factor and characterizes warm
    # repeated synthesis only, never cold-start or first-inference cost.
    #
    # Latency percentiles accompany the ratio only when every timed batch
    # declared exactly one utterance, because only then is a batch mean a
    # genuine per-utterance latency. Under that batch-one condition a batch
    # mean is an utterance-level mean call time, and the reported endpoints
    # are the p50 and p95 synthesis-call latency over those post-warm-up
    # times in milliseconds: p50 states the typical utterance and p95
    # states the tail a deployment must absorb, while the maximum is
    # deliberately not reported because a single scheduling outlier would
    # dominate it.
    #
    # Every ratio produced here is conditional on the execution lane that
    # produced it, where a lane is a requested resource and software
    # profile rather than an identified physical host. Ratios, latencies,
    # and any speedup derived from them are therefore comparable only
    # within one lane and never across lanes. Warm batch timing further
    # omits streaming behaviour, queuing, and cold start, so these
    # quantities do not establish low-latency streaming performance.
    #
    # One instance measures one prediction pass. The predict-start hook
    # clears every accumulator and re-arms the countdown, so a reused
    # monitor never carries timings across passes.
    def __init__(self, configuration: RealTimeFactorConfig) -> None:
        # Binds the timing protocol and prepares the empty accumulators the
        # prediction hooks fill.
        super().__init__()
        self._configuration: RealTimeFactorConfig = configuration
        self._batch_mean_times_seconds: list[float] = []
        self._audio_durations_seconds: list[float] = []
        self._batch_utterance_counts: list[int] = []
        self._warmup_remaining: int = 0
        self._warmup_executed: int = 0
        self._final_rtf: float | None = None

    @override
    def on_predict_start(self, trainer: Trainer, module: Module) -> None:
        # Resets every accumulator and arms the warm-up countdown for a
        # fresh prediction pass.
        self._batch_mean_times_seconds: list[float] = []
        self._audio_durations_seconds: list[float] = []
        self._batch_utterance_counts: list[int] = []
        self._warmup_remaining: int = self._configuration.warmup_iterations
        self._warmup_executed: int = 0
        self._final_rtf: float | None = None

    @override
    def on_predict_batch_end(
        self,
        trainer: Trainer,
        module: Module,
        outputs: ModelOutput,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Consumes one prediction batch: warm-up batches only decrement the
        # countdown; timed batches must carry a positive audio duration and
        # contribute their repetition-mean time, duration, and utterance
        # count to the accumulators.
        #
        # Args:
        #     trainer: The running trainer, consulted only for the device
        #         its strategy places work on, which selects the
        #         accelerator synchronization performed around every
        #         repetition.
        #     module: The evaluated module whose predict_step is the timed
        #         operation.
        #     outputs: The prediction the loop already produced; discarded,
        #         because timing re-executes synthesis rather than
        #         measuring the loop's own call.
        #     batch: The collated prediction batch, required to carry a
        #         duration_seconds payload as a tensor, a scalar, or a
        #         sequence.
        #     batch_idx: Index of this batch within the prediction pass,
        #         forwarded unchanged to every timed repetition so the
        #         module observes the same index the loop reported.
        #     dataloader_idx: Prediction dataloader ordinal; discarded,
        #         because timing accumulates over the whole pass.
        #         Default: ``0``.
        #
        # Raises:
        #     RuntimeError: If a timed batch carries no positive audio
        #         duration, since the ratio would then have no denominator
        #         to normalize against.
        del outputs, dataloader_idx
        if self._warmup_remaining > 0:
            self._warmup_remaining: int = self._warmup_remaining - 1
            self._warmup_executed: int = self._warmup_executed + 1
            return
        audio_duration_seconds: float = self._extract_audio_duration(batch)
        if audio_duration_seconds <= 0.0:
            raise RuntimeError(
                f"RealTimeFactorMonitor received batch {batch_idx} without a positive "
                f"duration_seconds payload; timed batches require audio durations."
            )
        repetition_times_seconds: list[float] = self._execute_timed_repetitions(
            trainer,
            module,
            batch,
            batch_idx
        )
        batch_mean_seconds: float = sum(repetition_times_seconds) / len(repetition_times_seconds)
        self._batch_mean_times_seconds.append(batch_mean_seconds)
        self._audio_durations_seconds.append(audio_duration_seconds)
        self._batch_utterance_counts.append(self._extract_utterance_count(batch))

    @override
    def on_predict_end(self, trainer: Trainer, module: Module) -> None:
        # Reduces the pass to the final ratio, failing loudly when no timed
        # samples exist, and publishes the metric payload through the
        # experiment logger.
        #
        # The ratio is the summed per-batch mean times over the summed
        # audio durations, so a long utterance carries proportionally more
        # weight than a short one. Publication is a separate concern from
        # measurement: a trainer without a logger still records the ratio on
        # the monitor, and that ratio reaches no experiment backend.
        #
        # Raises:
        #     RuntimeError: If the pass produced no timed batch at all. The
        #         message names the warm-up batches actually consumed,
        #         because the usual cause is a warm-up count that exhausted
        #         every available prediction batch.
        del module
        if not self._batch_mean_times_seconds or not self._audio_durations_seconds:
            raise RuntimeError(
                f"RealTimeFactorMonitor collected no timed samples after "
                f"{self._warmup_executed} warm-up batches; raise limit_predict_batches "
                f"or lower warmup_iterations so timed measurement can execute."
            )
        total_inference: float = sum(self._batch_mean_times_seconds)
        total_audio: float = sum(self._audio_durations_seconds)
        self._final_rtf: float | None = total_inference / total_audio
        log.info(
            f"RealTimeFactor over {len(self._batch_mean_times_seconds)} timed batches with "
            f"{self._configuration.timed_repetitions} repetitions each: {self._final_rtf:.6f}"
        )
        if trainer.logger is not None:
            trainer.logger.log_metrics(self._final_metrics(), step=0)

    @property
    def configuration(self) -> RealTimeFactorConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    @property
    def final_rtf(self) -> float | None:
        # Returns the final real-time-factor value after prediction timing completes.
        return self._final_rtf

    def _execute_timed_repetitions(
        self,
        trainer: Trainer,
        module: Module,
        batch: Batch,
        batch_idx: int
    ) -> list[float]:
        # Executes the configured measured repetitions on the module's prediction step.
        #
        # Every repetition is bracketed by an accelerator synchronization so
        # the elapsed span covers the kernels this repetition issued and no
        # others. The whole sequence runs under inference mode, which
        # removes autograd bookkeeping from the measurement without altering
        # the arithmetic being timed. The caller reduces the returned list to
        # its mean.
        #
        # Returns:
        #     One elapsed duration in seconds per configured repetition, in
        #     execution order.
        repetition_times_seconds: list[float] = []
        with torch.inference_mode():
            for _ in range(self._configuration.timed_repetitions):
                self._synchronize_accelerator(trainer)
                repetition_start: float = time.perf_counter()
                module.predict_step(batch, batch_idx)
                self._synchronize_accelerator(trainer)
                repetition_times_seconds.append(time.perf_counter() - repetition_start)
        return repetition_times_seconds

    def _final_metrics(self) -> dict[str, float]:
        # Builds the logged metric payload, adding latency percentiles for batch-1 timing.
        #
        # The ratio always travels with the denominators that produced it:
        # the utterances actually timed, the warm-up batches actually
        # consumed (which falls below the configured count when the pass
        # ran out of batches first), and the repetitions each timed batch
        # executed. A bare ratio is not auditable evidence, so these three
        # counts are unconditional.
        #
        # Percentiles are appended only when no timed batch carried more
        # than one utterance, because a batch mean over several utterances
        # is a throughput figure rather than a latency. Under the batch-one
        # timing protocol those batch means are the post-warm-up
        # utterance-level mean call times, and the two reported endpoints
        # are the p50 and p95 synthesis-call latency over them, in
        # milliseconds. The quantiles are therefore taken over the
        # per-utterance repetition means and not over the individual timed
        # calls, whose raw durations are never retained.
        metrics: dict[str, float] = {
            "real_time_factor": float(self._final_rtf or 0.0),
            "timed_utterance_count": float(sum(self._batch_utterance_counts)),
            "rtf_warmup_iterations": float(self._warmup_executed),
            "rtf_repetitions_executed": float(self._configuration.timed_repetitions)
        }
        if not self._batch_utterance_counts or max(self._batch_utterance_counts) != 1:
            return metrics
        latencies_ms: torch.Tensor = torch.tensor(self._batch_mean_times_seconds) * 1000.0
        metrics["latency_p50_ms"] = float(torch.quantile(latencies_ms, 0.5).item())
        metrics["latency_p95_ms"] = float(torch.quantile(latencies_ms, 0.95).item())
        return metrics

    def _synchronize_accelerator(self, trainer: Trainer) -> None:
        # Synchronizes the active accelerator so queued kernels are inside the timed span.
        #
        # CUDA and Metal dispatch work asynchronously, so an unsynchronized
        # clock would record the time to enqueue rather than the time to
        # compute. CPU execution is already synchronous and needs no
        # barrier, which is why an unrecognized device type is a no-op
        # rather than an error.
        device: torch.device = trainer.strategy.root_device
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
            return
        if device.type == "mps":
            torch.mps.synchronize()

    def _extract_utterance_count(self, batch: Batch) -> int:
        # Counts the utterances in a batch from its duration payload,
        # accepting sequence, scalar, or tensor forms. The count decides
        # whether latency percentiles are reportable, so a payload the
        # collator never produces yields zero rather than a guess.
        if not isinstance(batch, dict):
            return 0
        durations: object = batch.get("duration_seconds")
        if isinstance(durations, (tuple, list)):
            return len(durations)
        if isinstance(durations, (int, float)):
            return 1
        if isinstance(durations, torch.Tensor):
            return int(durations.numel())
        return 0

    def _extract_audio_duration(self, batch: Batch) -> float:
        # Sums the batch's audio duration in seconds from its duration
        # payload, accepting tensor, scalar, or sequence forms. A payload
        # the collator never produces sums to zero, which the caller turns
        # into a loud rejection rather than a silently unnormalized batch.
        if not isinstance(batch, dict):
            return 0.0
        durations: object = batch.get("duration_seconds")
        if isinstance(durations, torch.Tensor):
            return float(durations.sum().item())
        if isinstance(durations, (int, float)):
            return float(durations)
        if isinstance(durations, (tuple, list)):
            return float(sum(float(duration_value) for duration_value in durations))
        return 0.0
