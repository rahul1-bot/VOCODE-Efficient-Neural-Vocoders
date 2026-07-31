# This module:
# 1. Verifies the timing protocol record: its defaults, immutability, and
#    validation boundaries
# 2. Verifies the warm-up contract, that warm-up batches execute no timed
#    repetitions and that timed batches execute exactly the configured count
#    through the module's own prediction step
# 3. Verifies the reduction: a pass without timed samples fails loudly, a batch
#    without a positive audio duration is rejected, and the retained ratio is
#    positive and finite
# 4. Verifies the published payload: the ratio, its denominators, and the
#    latency percentiles that appear only for single-utterance batches
#
# Design decisions:
# - No assertion depends on a wall-clock magnitude. The measured quantity is a
#   real elapsed duration on shared hardware, so only its sign, finiteness,
#   internal consistency, and the number of executed repetitions are pinned
# - The synthesizer stand-in counts its own invocations rather than performing
#   work, which makes the repetition contract observable without spending time
# - The logger stand-in records payloads in memory; its save directory is an
#   existing temporary directory and its log directory is never resolved, so no
#   file is created by these tests
# - Percentiles are asserted as ordered and finite rather than as values,
#   because they are quantiles over genuinely measured durations
#
# Author: Rahul Sawhney

import math
import tempfile
import unittest
from pathlib import Path
from typing import override

import torch
from pydantic import ValidationError

from syntheticmind.core.module import Module
from syntheticmind.core.trainer import Trainer
from syntheticmind.loggers.logger import Logger
from syntheticmind.utilities.types import Batch, HyperparameterDict, ModelOutput

from vocode.metrics.rtf import RealTimeFactorConfig, RealTimeFactorMonitor


class RecordingLogger(Logger):
    # Logger stand-in retaining every published metric payload in memory so the
    # timing callback's output is inspectable without touching the filesystem.
    def __init__(self, save_dir: Path) -> None:
        # Binds the logger identity and prepares the empty payload record.
        super().__init__(save_dir)
        self._payloads: list[dict[str, float]] = []

    @override
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        # Retains a copy of one published payload.
        self._payloads.append(dict(metrics))

    @override
    def log_hyperparams(self, params: HyperparameterDict) -> None:
        # Accepts hyperparameters without recording them; timing publishes none.
        pass

    @override
    def finalize(self) -> None:
        # Holds no open resource to release.
        pass

    @property
    def payloads(self) -> list[dict[str, float]]:
        # Returns a copy of every payload published so far.
        return list(self._payloads)


class CountingSynthesizer(Module):
    # Prediction-step stand-in that records how many times it was invoked and
    # under which batch index, making the repetition contract observable.
    #
    # Counting instead of synthesizing is what makes the warm-up and
    # repetition contracts assertable as exact integers: a warm-up batch
    # must leave the counter untouched and a timed batch must advance it by
    # exactly the configured repetition count. It also keeps every test in
    # this module inexpensive, since the timed operation performs no work
    # while still producing a genuine, strictly positive elapsed duration.
    def __init__(self) -> None:
        # Prepares the counters the repetition assertions read.
        super().__init__()
        self._invocation_count: int = 0
        self._last_batch_index: int = -1

    @override
    def predict_step(self, batch: Batch, batch_idx: int) -> ModelOutput:
        # Records the invocation and its batch index instead of synthesizing,
        # so a repetition costs no measurable time.
        self._invocation_count: int = self._invocation_count + 1
        self._last_batch_index: int = batch_idx
        return torch.zeros(1)

    @property
    def invocation_count(self) -> int:
        # Returns how often the timing driver called the prediction step.
        return self._invocation_count

    @property
    def last_batch_index(self) -> int:
        # Returns the batch index carried by the most recent invocation.
        return self._last_batch_index


class SingleUtteranceBatch:
    # Builds the collated payload of a one-utterance prediction batch carrying
    # its audio duration in the tensor form the dataloader produces.
    #
    # A one-element duration tensor is what declares the batch to hold a
    # single utterance, which is the condition under which latency
    # percentiles are reportable; the multi-utterance cases are built
    # inline where they are needed so the contrast stays visible at the
    # assertion.
    def __init__(self, duration_seconds: float) -> None:
        # Binds the audio duration the built batch declares.
        self._duration_seconds: float = duration_seconds

    def create(self) -> dict[str, Batch]:
        # Produces the collated mapping with the duration as a one-element tensor.
        return {"duration_seconds": torch.tensor([self._duration_seconds])}


class RealTimeFactorConfigurationTest(unittest.TestCase):
    # Verifies the frozen timing protocol and its validation boundaries.
    def test_default_protocol_discards_five_warmup_batches(self) -> None:
        # Warm-up absorbs lazy initialization, allocator growth, and compilation.
        self.assertEqual(RealTimeFactorConfig().warmup_iterations, 5)

    def test_default_protocol_times_three_repetitions_per_batch(self) -> None:
        # Each timed batch contributes the mean of its measured repetitions.
        self.assertEqual(RealTimeFactorConfig().timed_repetitions, 3)

    def test_configuration_is_frozen(self) -> None:
        # The timing protocol cannot drift while a pass is being measured.
        configuration: RealTimeFactorConfig = RealTimeFactorConfig()
        with self.assertRaises(ValidationError):
            configuration.timed_repetitions = 9

    def test_zero_warmup_is_allowed(self) -> None:
        # Skipping warm-up is a legal protocol, unlike skipping measurement.
        self.assertEqual(RealTimeFactorConfig(warmup_iterations=0).warmup_iterations, 0)

    def test_negative_warmup_is_rejected(self) -> None:
        # A negative countdown has no timing meaning.
        with self.assertRaises(ValidationError):
            RealTimeFactorConfig(warmup_iterations=-1)

    def test_zero_repetitions_is_rejected(self) -> None:
        # A timed batch with no repetitions would produce no measurement.
        with self.assertRaises(ValidationError):
            RealTimeFactorConfig(timed_repetitions=0)

    def test_extra_fields_are_rejected(self) -> None:
        # A misspelled setting must fail loudly rather than be ignored.
        with self.assertRaises(ValidationError):
            RealTimeFactorConfig.model_validate({"warmup_iteration": 5})

    def test_monitor_exposes_the_bound_configuration(self) -> None:
        # The monitor reports the exact protocol it was constructed with.
        configuration: RealTimeFactorConfig = RealTimeFactorConfig(warmup_iterations=1)
        self.assertIs(RealTimeFactorMonitor(configuration).configuration, configuration)


class RealTimeFactorWarmupTest(unittest.TestCase):
    # Verifies that warm-up batches pass through untimed and that timed batches
    # execute exactly the configured repetitions on the module's predict step.
    #
    # The hooks are driven directly rather than through a real prediction
    # run, because the contract under test is the callback's own state
    # machine: how the countdown is consumed, when repetitions execute, and
    # what a second pass inherits. Driving the hooks by hand also makes the
    # re-arming behaviour testable, which a single loop-driven pass could
    # not show.
    def setUp(self) -> None:
        # Prepares a CPU trainer, the counting module, and a two-second batch.
        self._logger: RecordingLogger = RecordingLogger(Path(tempfile.gettempdir()))
        self._trainer: Trainer = Trainer(
            accelerator="cpu",
            strategy="single_device",
            enable_progress_bar=False,
            logger=self._logger
        )
        self._module: CountingSynthesizer = CountingSynthesizer()
        self._batch: dict[str, Batch] = SingleUtteranceBatch(2.0).create()

    def test_warmup_batches_execute_no_timed_repetitions(self) -> None:
        # A warm-up batch only decrements the countdown; the loop already ran it.
        monitor: RealTimeFactorMonitor = RealTimeFactorMonitor(
            RealTimeFactorConfig(warmup_iterations=2, timed_repetitions=3)
        )
        monitor.on_predict_start(self._trainer, self._module)
        monitor.on_predict_batch_end(self._trainer, self._module, None, self._batch, 0)
        monitor.on_predict_batch_end(self._trainer, self._module, None, self._batch, 1)
        self.assertEqual(self._module.invocation_count, 0)

    def test_timed_batch_executes_the_configured_repetitions(self) -> None:
        # The timed protocol is the executed synthesis protocol, run three times.
        monitor: RealTimeFactorMonitor = RealTimeFactorMonitor(
            RealTimeFactorConfig(warmup_iterations=0, timed_repetitions=3)
        )
        monitor.on_predict_start(self._trainer, self._module)
        monitor.on_predict_batch_end(self._trainer, self._module, None, self._batch, 0)
        self.assertEqual(self._module.invocation_count, 3)

    def test_each_timed_batch_repeats_independently(self) -> None:
        # Repetitions accumulate per timed batch rather than per pass.
        monitor: RealTimeFactorMonitor = RealTimeFactorMonitor(
            RealTimeFactorConfig(warmup_iterations=0, timed_repetitions=2)
        )
        monitor.on_predict_start(self._trainer, self._module)
        monitor.on_predict_batch_end(self._trainer, self._module, None, self._batch, 0)
        monitor.on_predict_batch_end(self._trainer, self._module, None, self._batch, 1)
        monitor.on_predict_batch_end(self._trainer, self._module, None, self._batch, 2)
        self.assertEqual(self._module.invocation_count, 6)

    def test_repetitions_run_against_the_batch_under_measurement(self) -> None:
        # The measured call receives the same batch index the loop reported.
        monitor: RealTimeFactorMonitor = RealTimeFactorMonitor(
            RealTimeFactorConfig(warmup_iterations=0, timed_repetitions=1)
        )
        monitor.on_predict_start(self._trainer, self._module)
        monitor.on_predict_batch_end(self._trainer, self._module, None, self._batch, 7)
        self.assertEqual(self._module.last_batch_index, 7)

    def test_pass_without_timed_batches_fails_loudly(self) -> None:
        # A zero ratio is a claim, not an absence, so the run refuses to report one.
        monitor: RealTimeFactorMonitor = RealTimeFactorMonitor(
            RealTimeFactorConfig(warmup_iterations=2, timed_repetitions=1)
        )
        monitor.on_predict_start(self._trainer, self._module)
        monitor.on_predict_batch_end(self._trainer, self._module, None, self._batch, 0)
        with self.assertRaisesRegex(RuntimeError, "collected no timed samples"):
            monitor.on_predict_end(self._trainer, self._module)

    def test_predict_start_rearms_the_warmup_countdown(self) -> None:
        # A second pass repeats warm-up rather than inheriting a spent countdown.
        monitor: RealTimeFactorMonitor = RealTimeFactorMonitor(
            RealTimeFactorConfig(warmup_iterations=1, timed_repetitions=1)
        )
        monitor.on_predict_start(self._trainer, self._module)
        monitor.on_predict_batch_end(self._trainer, self._module, None, self._batch, 0)
        monitor.on_predict_batch_end(self._trainer, self._module, None, self._batch, 1)
        monitor.on_predict_end(self._trainer, self._module)
        monitor.on_predict_start(self._trainer, self._module)
        monitor.on_predict_batch_end(self._trainer, self._module, None, self._batch, 0)
        with self.assertRaisesRegex(RuntimeError, "collected no timed samples"):
            monitor.on_predict_end(self._trainer, self._module)

    def test_predict_start_clears_the_previous_ratio(self) -> None:
        # State from a completed pass must not survive into the next one.
        monitor: RealTimeFactorMonitor = RealTimeFactorMonitor(
            RealTimeFactorConfig(warmup_iterations=0, timed_repetitions=1)
        )
        monitor.on_predict_start(self._trainer, self._module)
        monitor.on_predict_batch_end(self._trainer, self._module, None, self._batch, 0)
        monitor.on_predict_end(self._trainer, self._module)
        monitor.on_predict_start(self._trainer, self._module)
        self.assertIsNone(monitor.final_rtf)


class RealTimeFactorMeasurementTest(unittest.TestCase):
    # Verifies the retained ratio and the audio-duration payload contract.
    def setUp(self) -> None:
        # Prepares a CPU trainer and a monitor that times every batch once.
        self._logger: RecordingLogger = RecordingLogger(Path(tempfile.gettempdir()))
        self._trainer: Trainer = Trainer(
            accelerator="cpu",
            strategy="single_device",
            enable_progress_bar=False,
            logger=self._logger
        )
        self._module: CountingSynthesizer = CountingSynthesizer()
        self._monitor: RealTimeFactorMonitor = RealTimeFactorMonitor(
            RealTimeFactorConfig(warmup_iterations=0, timed_repetitions=1)
        )
        self._batch: dict[str, Batch] = SingleUtteranceBatch(2.0).create()

    def test_ratio_is_unset_before_the_first_pass(self) -> None:
        # A monitor that has measured nothing reports no ratio.
        self.assertIsNone(self._monitor.final_rtf)

    def test_ratio_is_positive_and_finite_after_a_timed_pass(self) -> None:
        # Elapsed time over audio duration is a strictly positive real number.
        self._monitor.on_predict_start(self._trainer, self._module)
        self._monitor.on_predict_batch_end(self._trainer, self._module, None, self._batch, 0)
        self._monitor.on_predict_end(self._trainer, self._module)
        measured_ratio: float | None = self._monitor.final_rtf
        self.assertIsNotNone(measured_ratio)
        self.assertGreater(measured_ratio, 0.0)
        self.assertTrue(math.isfinite(measured_ratio))

    def test_batch_without_a_positive_duration_is_rejected(self) -> None:
        # Timed batches require an audio duration to normalize against.
        self._monitor.on_predict_start(self._trainer, self._module)
        with self.assertRaisesRegex(RuntimeError, "positive"):
            self._monitor.on_predict_batch_end(
                self._trainer,
                self._module,
                None,
                SingleUtteranceBatch(0.0).create(),
                0
            )

    def test_batch_without_a_duration_payload_is_rejected(self) -> None:
        # A batch carrying no duration key cannot be timed.
        self._monitor.on_predict_start(self._trainer, self._module)
        with self.assertRaisesRegex(RuntimeError, "duration_seconds"):
            self._monitor.on_predict_batch_end(self._trainer, self._module, None, {}, 0)

    def test_non_mapping_batch_is_rejected(self) -> None:
        # A batch that is not a mapping carries no readable duration.
        self._monitor.on_predict_start(self._trainer, self._module)
        with self.assertRaisesRegex(RuntimeError, "duration_seconds"):
            self._monitor.on_predict_batch_end(
                self._trainer,
                self._module,
                None,
                torch.zeros(4),
                0
            )

    def test_duration_payload_accepts_tensor_scalar_and_sequence_forms(self) -> None:
        # The collator may present durations as a tensor, a scalar, or a sequence.
        self._monitor.on_predict_start(self._trainer, self._module)
        self._monitor.on_predict_batch_end(self._trainer, self._module, None, self._batch, 0)
        self._monitor.on_predict_batch_end(
            self._trainer,
            self._module,
            None,
            {"duration_seconds": 4.0},
            1
        )
        self._monitor.on_predict_batch_end(
            self._trainer,
            self._module,
            None,
            {"duration_seconds": [1.0, 2.0]},
            2
        )
        self._monitor.on_predict_end(self._trainer, self._module)
        self.assertEqual(self._logger.payloads[-1]["timed_utterance_count"], 4.0)

    def test_pass_without_a_logger_still_records_the_ratio(self) -> None:
        # Logging is a publication channel, not a precondition for measurement.
        silent_trainer: Trainer = Trainer(
            accelerator="cpu",
            strategy="single_device",
            enable_progress_bar=False,
            logger=None
        )
        self._monitor.on_predict_start(silent_trainer, self._module)
        self._monitor.on_predict_batch_end(silent_trainer, self._module, None, self._batch, 0)
        self._monitor.on_predict_end(silent_trainer, self._module)
        measured_ratio: float | None = self._monitor.final_rtf
        self.assertIsNotNone(measured_ratio)
        self.assertGreater(measured_ratio, 0.0)


class RealTimeFactorPublicationTest(unittest.TestCase):
    # Verifies the published metric payload: the ratio, the timing
    # denominators, and the conditional latency percentiles.
    #
    # The percentiles are asserted as ordered and finite rather than as
    # values, because they are quantiles over durations genuinely measured
    # on shared hardware; the one property that must hold regardless of the
    # machine is that the median cannot exceed the ninety-fifth percentile.
    # Their absence for multi-utterance batches is asserted equally
    # explicitly, since silently publishing a batch mean as a latency would
    # be the more damaging failure.
    def setUp(self) -> None:
        # Prepares a CPU trainer with a recording logger and the counting module.
        self._logger: RecordingLogger = RecordingLogger(Path(tempfile.gettempdir()))
        self._trainer: Trainer = Trainer(
            accelerator="cpu",
            strategy="single_device",
            enable_progress_bar=False,
            logger=self._logger
        )
        self._module: CountingSynthesizer = CountingSynthesizer()

    def test_payload_carries_the_ratio_and_its_denominators(self) -> None:
        # A ratio without its denominators is not auditable evidence.
        monitor: RealTimeFactorMonitor = RealTimeFactorMonitor(
            RealTimeFactorConfig(warmup_iterations=1, timed_repetitions=2)
        )
        monitor.on_predict_start(self._trainer, self._module)
        monitor.on_predict_batch_end(
            self._trainer,
            self._module,
            None,
            SingleUtteranceBatch(2.0).create(),
            0
        )
        monitor.on_predict_batch_end(
            self._trainer,
            self._module,
            None,
            SingleUtteranceBatch(2.0).create(),
            1
        )
        monitor.on_predict_end(self._trainer, self._module)
        published: dict[str, float] = self._logger.payloads[-1]
        self.assertEqual(published["timed_utterance_count"], 1.0)
        self.assertEqual(published["rtf_warmup_iterations"], 1.0)
        self.assertEqual(published["rtf_repetitions_executed"], 2.0)

    def test_published_ratio_matches_the_retained_value(self) -> None:
        # The logged number and the readable property are one measurement.
        monitor: RealTimeFactorMonitor = RealTimeFactorMonitor(
            RealTimeFactorConfig(warmup_iterations=0, timed_repetitions=1)
        )
        monitor.on_predict_start(self._trainer, self._module)
        monitor.on_predict_batch_end(
            self._trainer,
            self._module,
            None,
            SingleUtteranceBatch(2.0).create(),
            0
        )
        monitor.on_predict_end(self._trainer, self._module)
        self.assertEqual(self._logger.payloads[-1]["real_time_factor"], monitor.final_rtf)

    def test_utterance_denominator_sums_every_timed_batch(self) -> None:
        # The denominator counts utterances measured, not batches measured.
        monitor: RealTimeFactorMonitor = RealTimeFactorMonitor(
            RealTimeFactorConfig(warmup_iterations=0, timed_repetitions=1)
        )
        monitor.on_predict_start(self._trainer, self._module)
        monitor.on_predict_batch_end(
            self._trainer,
            self._module,
            None,
            SingleUtteranceBatch(2.0).create(),
            0
        )
        monitor.on_predict_batch_end(
            self._trainer,
            self._module,
            None,
            SingleUtteranceBatch(3.0).create(),
            1
        )
        monitor.on_predict_end(self._trainer, self._module)
        self.assertEqual(self._logger.payloads[-1]["timed_utterance_count"], 2.0)

    def test_latency_percentiles_are_published_for_single_utterance_batches(self) -> None:
        # A batch mean is a genuine per-utterance latency only at batch size one.
        monitor: RealTimeFactorMonitor = RealTimeFactorMonitor(
            RealTimeFactorConfig(warmup_iterations=0, timed_repetitions=1)
        )
        monitor.on_predict_start(self._trainer, self._module)
        for batch_index in range(3):
            monitor.on_predict_batch_end(
                self._trainer,
                self._module,
                None,
                SingleUtteranceBatch(2.0).create(),
                batch_index
            )
        monitor.on_predict_end(self._trainer, self._module)
        published: dict[str, float] = self._logger.payloads[-1]
        self.assertIn("latency_p50_ms", published)
        self.assertIn("latency_p95_ms", published)

    def test_published_percentiles_are_ordered_and_finite(self) -> None:
        # The median latency can never exceed the ninety-fifth percentile.
        monitor: RealTimeFactorMonitor = RealTimeFactorMonitor(
            RealTimeFactorConfig(warmup_iterations=0, timed_repetitions=1)
        )
        monitor.on_predict_start(self._trainer, self._module)
        for batch_index in range(3):
            monitor.on_predict_batch_end(
                self._trainer,
                self._module,
                None,
                SingleUtteranceBatch(2.0).create(),
                batch_index
            )
        monitor.on_predict_end(self._trainer, self._module)
        published: dict[str, float] = self._logger.payloads[-1]
        self.assertGreater(published["latency_p50_ms"], 0.0)
        self.assertTrue(math.isfinite(published["latency_p95_ms"]))
        self.assertLessEqual(published["latency_p50_ms"], published["latency_p95_ms"])

    def test_latency_percentiles_are_withheld_for_multi_utterance_batches(self) -> None:
        # A batch mean over several utterances is not a per-utterance latency.
        monitor: RealTimeFactorMonitor = RealTimeFactorMonitor(
            RealTimeFactorConfig(warmup_iterations=0, timed_repetitions=1)
        )
        monitor.on_predict_start(self._trainer, self._module)
        monitor.on_predict_batch_end(
            self._trainer,
            self._module,
            None,
            {"duration_seconds": torch.tensor([1.0, 3.0])},
            0
        )
        monitor.on_predict_end(self._trainer, self._module)
        published: dict[str, float] = self._logger.payloads[-1]
        self.assertNotIn("latency_p50_ms", published)
        self.assertNotIn("latency_p95_ms", published)

    def test_payload_is_published_once_per_pass(self) -> None:
        # Timing publishes at reduction, not per batch.
        monitor: RealTimeFactorMonitor = RealTimeFactorMonitor(
            RealTimeFactorConfig(warmup_iterations=0, timed_repetitions=1)
        )
        monitor.on_predict_start(self._trainer, self._module)
        monitor.on_predict_batch_end(
            self._trainer,
            self._module,
            None,
            SingleUtteranceBatch(2.0).create(),
            0
        )
        self.assertEqual(len(self._logger.payloads), 0)
        monitor.on_predict_end(self._trainer, self._module)
        self.assertEqual(len(self._logger.payloads), 1)


if __name__ == "__main__":
    unittest.main()
