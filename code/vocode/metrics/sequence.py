# This module:
# 1. Drives the complete objective metric panel over a test pass as a single
#    callback: static complexity metrics once per run, per-utterance waveform
#    quality metrics with failure accounting, and stateful pitch-family
#    aggregates
# 2. Writes the per-utterance metric table for paired statistical analysis
#    and publishes the reduced panel through the experiment logger
#
# Harness contract (syntheticmind):
# - Subclasses the harness Callback and rides the test loop: on_test_start
#   prepares metric state and computes the static metrics, on_test_batch_end
#   consumes each batch, and on_test_end reduces and publishes
# - The evaluated module must expose network and metric_mel_protocol (the
#   MetricModel protocol), its test_step output must contain the
#   synthesized_waveform tensor, and the batch mapping must carry waveform,
#   waveform_length, and sample_rate as produced by the LJSpeech collator
#
# Design decisions:
# - Metrics divide into three families with different lifecycles: static
#   metrics (parameters, size, MACs) computed once from the network;
#   per-utterance metrics (PESQ, STOI, mel error, multi-resolution STFT,
#   MCD, LAS, UTMOS) averaged over utterances with per-metric failure
#   counts; and pitch-family aggregates (F0, periodicity, voicing) updated
#   incrementally over shared pitch extractions so pitch runs once per
#   utterance pair, not once per metric
# - Reference and candidate are truncated to their common length and
#   compared as float32 on the CPU, keeping metric arithmetic identical
#   across accelerator lanes; only UTMOS and pitch extraction run on the
#   trainer device for speed
# - A per-utterance metric failure is counted and logged, and its cell in
#   the per-utterance table stays empty; a metric whose value count reaches
#   zero fails the run at reduction, so a broken metric can never produce a
#   silently absent column
# - Parameter and size metrics prefer the module's declared deployable
#   quantities (packed weights, exported artifact bytes) over live-object
#   measurements, so transformed variants report what would actually ship
#
# Author: Rahul Sawhney

import csv
import math
from pathlib import Path
from typing import Protocol, cast, override

import torch
from loguru import logger as log

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.core.trainer import Trainer
from syntheticmind.utilities.types import Batch, StepOutput

from vocode.metrics.f0 import F0Rmse
from vocode.metrics.las import LogAmplitudeSpectrumRmse
from vocode.metrics.macs import MacsProfiler, MacsProfileResult
from vocode.metrics.mcd import MelCepstralDistortion
from vocode.metrics.mel import MelError
from vocode.metrics.parameters import ParameterCount
from vocode.metrics.periodicity import PeriodicityRmse
from vocode.metrics.pesq import Pesq
from vocode.metrics.pitch import PitchConfig, PitchExtractor, PitchFeatures
from vocode.metrics.registry import MetricName, MetricRegistry, MetricSelection
from vocode.metrics.size import ModelSize
from vocode.metrics.stft import MultiResolutionStftError
from vocode.metrics.stoi import Stoi
from vocode.metrics.utmos import UtmosPredictor
from vocode.metrics.voicing import VoicingF1
from vocode.transforms.mel import MelConfig, MelSpectrogram

__all__: list[str] = ["MetricSequence"]


class MetricModel(Protocol):
    # Structural contract the evaluated module must satisfy: the synthesis
    # network for complexity metrics and the mel protocol for the mel-error
    # metric.
    #
    # Integration: conformance is checked by attribute lookup at test start
    # rather than by declared inheritance, so any module exposing both
    # members qualifies without importing this protocol. The check runs
    # once, before any measurement consumes the module, so a module missing
    # a member fails the pass immediately instead of part-way through a
    # batch. Modules may additionally declare deployable_parameter_count
    # and deployable_artifact_bytes, which the static panel prefers over
    # its own live-object measurements; those two are optional and are
    # deliberately outside this protocol because an untransformed model
    # legitimately has neither.
    @property
    def network(self) -> torch.nn.Module: ...

    @property
    def metric_mel_protocol(self) -> MelConfig: ...


class MetricSequence(Callback):
    # The metric panel driver. One instance rides one test pass; its results
    # property retains the reduced panel after the pass ends.
    #
    # Composition is by ordered name. The run declares a validated tuple of
    # metric names and this class alone turns that tuple into work: at
    # construction the names are filtered against the registry's family
    # tuples into the three lifecycles the pass executes, preserving the
    # caller's order within each. A name absent from the selection is never
    # built, never computed, and produces no column, so the panel states
    # exactly what was asked for and nothing else.
    #
    # The three lifecycles differ in when they run and how they reduce.
    # Static complexity metrics are computed once at test start, straight
    # from the network, and published immediately so they survive a later
    # batch failure. Per-utterance metrics are computed for every reference
    # and candidate pair and reduced at test end to a mean plus a failure
    # count. Pitch-family aggregates hold state across the whole pass and
    # are finalized once at the end, which weights every frame equally
    # instead of averaging per-utterance values of unequal length.
    #
    # Selection order is retained rather than normalized: it fixes the
    # order in which per-utterance metrics are evaluated for each utterance
    # and the column order of the per-utterance table. It does not affect
    # the reduced panel, which is a mapping keyed by result-column name and
    # therefore identical however the selection was ordered.
    #
    # Failure is accounted rather than swallowed. A per-utterance metric
    # that raises, or returns a non-finite value, is counted against its
    # own failure column and leaves its table cell empty while the
    # remaining metrics and utterances continue; but a metric that reaches
    # reduction with no valid value at all fails the run, so a broken
    # metric can never disappear into a silently absent column. Those
    # per-metric failure counts are the recorded evidence for the study's
    # admission criterion that an admitted result group carries zero
    # mandatory metric failures, which is why they are published as
    # columns of the panel rather than retained as internal state.
    #
    # One instance is reusable across passes: test start clears every
    # accumulator, so a second pass measures its own utterances alone.
    def __init__(
        self,
        selection: MetricSelection,
        per_utterance_csv_path: Path | None = None
    ) -> None:
        # Partitions the selected metrics into their lifecycle families and
        # constructs the sample-rate-independent metric instances; the
        # sample-rate-dependent ones (pitch, MCD) and the device-bound one
        # (UTMOS) are built lazily at test start.
        #
        # The partition filters the selection against the registry's family
        # tuples, so membership is decided by the registry and never
        # restated here. Within each family the caller's declared order is
        # preserved. Construction is deliberately incomplete: metrics whose
        # settings depend on values not known until the pass begins are
        # left unbuilt, which is why the sample-rate-bound and device-bound
        # slots start empty rather than holding a provisional instance.
        #
        # Args:
        #     selection: Validated set of metric names to compute; names
        #         outside the waveform family are handled as static or
        #         aggregate metrics per the registry's classification.
        #     per_utterance_csv_path: Optional destination for the
        #         per-utterance metric table; ``None`` skips the table.
        #         Default: ``None``.
        super().__init__()
        self._selection: MetricSelection = selection
        self._per_utterance_csv_path: Path | None = per_utterance_csv_path
        self._per_utterance_rows: list[dict[str, float | int | str]] = []
        self._registry: MetricRegistry = MetricRegistry()
        self._waveform_names: tuple[MetricName, ...] = tuple(
            name for name in selection.names if name in self._registry.waveform_names
        )
        self._aggregate_names: tuple[MetricName, ...] = tuple(
            name for name in self._waveform_names if name in ("f0", "periodicity", "voicing")
        )
        self._utterance_names: tuple[MetricName, ...] = tuple(
            name for name in self._waveform_names if name not in self._aggregate_names
        )
        self._values: dict[MetricName, list[float]] = {}
        self._failures: dict[MetricName, int] = {}
        self._results: dict[str, float] = {}
        self._utterance_count: int = 0
        self._metric_device: str = "cpu"
        self._mel_transform: MelSpectrogram | None = None
        self._pitch_extractors: dict[int, PitchExtractor] = {}
        self._mcd_metrics: dict[int, MelCepstralDistortion] = {}
        self._mel_metric: MelError | None = self._build_optional("mel", MelError)
        self._pesq_metric: Pesq | None = self._build_optional("pesq", Pesq)
        self._stoi_metric: Stoi | None = self._build_optional("stoi", Stoi)
        self._stft_metric: MultiResolutionStftError | None = self._build_optional(
            "stft",
            MultiResolutionStftError
        )
        self._las_metric: LogAmplitudeSpectrumRmse | None = self._build_optional(
            "las",
            LogAmplitudeSpectrumRmse
        )
        self._f0_metric: F0Rmse | None = self._build_optional("f0", F0Rmse)
        self._periodicity_metric: PeriodicityRmse | None = self._build_optional(
            "periodicity",
            PeriodicityRmse
        )
        self._voicing_metric: VoicingF1 | None = self._build_optional("voicing", VoicingF1)
        self._utmos_metric: UtmosPredictor | None = None

    @override
    def on_test_start(self, trainer: Trainer, module: Module) -> None:
        # Resets all accumulation state for the pass, binds the metric
        # device from the strategy, builds the deferred metric instances,
        # verifies the module satisfies the metric-model contract, and
        # computes and logs the static metrics immediately so they exist
        # even if a later batch fails.
        #
        # Args:
        #     trainer: The running trainer, consulted for the strategy's
        #         device (which binds the device-sensitive metrics) and for
        #         the experiment logger the static panel publishes through.
        #     module: The module under evaluation, required to satisfy the
        #         metric-model contract and inspected here for the optional
        #         deployable declarations.
        #
        # Raises:
        #     TypeError: If the module exposes no synthesis network or no
        #         metric mel protocol, since every measurement downstream
        #         depends on one or the other.
        self._values: dict[MetricName, list[float]] = {
            name: [] for name in self._utterance_names
        }
        self._failures: dict[MetricName, int] = {
            name: 0 for name in self._utterance_names
        }
        self._results: dict[str, float] = {}
        self._utterance_count: int = 0
        self._per_utterance_rows: list[dict[str, float | int | str]] = []
        self._metric_device: str = str(trainer.strategy.root_device)
        self._pitch_extractors: dict[int, PitchExtractor] = {}
        self._mcd_metrics: dict[int, MelCepstralDistortion] = {}
        self._utmos_metric: UtmosPredictor | None = self._build_optional(
            "utmos",
            UtmosPredictor,
            device=self._metric_device
        )
        self._reset_aggregate_metrics()
        metric_model: MetricModel = self._require_metric_model(module)
        if "mel" in self._selection.names:
            self._mel_transform: MelSpectrogram | None = MelSpectrogram(
                metric_model.metric_mel_protocol
            )
        else:
            self._mel_transform: MelSpectrogram | None = None
        static_results: dict[str, float] = self._compute_static_metrics(metric_model, module)
        self._results.update(static_results)
        self._log_metrics(trainer, static_results)

    @override
    def on_test_batch_end(
        self,
        trainer: Trainer,
        module: Module,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Evaluates every utterance of the batch: reference and candidate
        # are truncated to their common true length (padding never enters a
        # measurement), moved to the CPU as float32, and passed through the
        # per-utterance and aggregate metric paths.
        #
        # The common length is the smaller of the reference's declared true
        # length and the candidate's synthesized length, so neither the
        # collator's padding nor a length mismatch between the two signals
        # can reach a metric. This realizes the true-length evaluation
        # semantics the study reports: when the batch declares
        # per-utterance lengths the result is the true-length value, and
        # without them the padded width stands in, which is the recorded
        # padded-batch value. The two diverge materially for a
        # padding-sensitive architecture. Comparing on the CPU in float32
        # keeps the arithmetic identical whichever accelerator produced the
        # audio.
        #
        # Args:
        #     trainer: Discarded; batch evaluation publishes nothing and
        #         the reduced panel is logged at test end instead.
        #     module: Discarded; the batch already carries both signals.
        #     outputs: The test-step output, required to carry the
        #         synthesized waveform under its declared key.
        #     batch: The collated batch, required to carry the reference
        #         waveform and the sample rate, and optionally the true
        #         per-utterance lengths.
        #     batch_idx: Index of this batch within the pass, recorded in
        #         each row's label so a per-utterance row is traceable back
        #         to the batch that produced it.
        #     dataloader_idx: Test dataloader ordinal; discarded, because
        #         the panel accumulates over the whole pass. Default: ``0``.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping, carries no
        #         reference waveform tensor, carries an unsupported length
        #         or sample-rate payload, or the step output carries no
        #         synthesized waveform tensor.
        #     ValueError: If the reference and candidate batch sizes
        #         disagree, if the declared length count does not match the
        #         batch size, or if an utterance's common length is empty.
        del trainer, module, dataloader_idx
        if not self._waveform_names:
            return
        reference_batch: torch.Tensor = self._extract_reference_batch(batch)
        candidate_batch: torch.Tensor = self._extract_candidate_batch(outputs)
        if reference_batch.shape[0] != candidate_batch.shape[0]:
            raise ValueError(
                f"Reference batch size {reference_batch.shape[0]} does not match "
                f"candidate batch size {candidate_batch.shape[0]}"
            )
        lengths: tuple[int, ...] = self._extract_lengths(batch, reference_batch)
        sample_rate: int = self._extract_sample_rate(batch)
        for utterance_index in range(reference_batch.shape[0]):
            reference_length: int = lengths[utterance_index]
            candidate_length: int = int(candidate_batch[utterance_index].shape[-1])
            common_length: int = min(reference_length, candidate_length)
            if common_length < 1:
                raise ValueError(
                    f"Empty waveform at batch={batch_idx}, utterance={utterance_index}"
                )
            reference: torch.Tensor = reference_batch[
                utterance_index,
                :common_length
            ].detach().float().cpu()
            candidate: torch.Tensor = candidate_batch[
                utterance_index,
                :common_length
            ].detach().float().cpu()
            self._evaluate_utterance(
                reference,
                candidate,
                sample_rate,
                f"batch={batch_idx};utterance={utterance_index}",
                self._utterance_count
            )
            self._utterance_count: int = self._utterance_count + 1

    @override
    def on_test_end(self, trainer: Trainer, module: Module) -> None:
        # Reduces the pass: each per-utterance metric becomes its mean plus
        # a failure-count column, a metric with zero valid values fails the
        # run loudly, the aggregates compute from their accumulated state,
        # the utterance denominator is recorded, and the per-utterance table
        # is written before the final panel is logged.
        #
        # Every reported mean is accompanied by the two denominators that
        # make it auditable: its own failure count and, once any waveform
        # metric ran, the number of utterance pairs evaluated. The static
        # results computed at test start are carried into the final panel
        # unchanged, so one mapping describes the whole pass.
        #
        # Args:
        #     trainer: The running trainer, used for its experiment logger
        #         and its global step; a trainer without a logger still
        #         gets a fully reduced panel on the results property.
        #     module: Discarded; reduction reads only accumulated state.
        #
        # Raises:
        #     RuntimeError: If a selected per-utterance metric produced no
        #         valid value across the entire pass. The message names the
        #         metric and its failure count, because an absent column is
        #         indistinguishable from an unselected metric downstream
        #         and must never be published as one.
        #     ValueError: Propagated from a pitch-family aggregate whose
        #         pass observed no frame it could score.
        del module
        final_metrics: dict[str, float] = dict(self._results)
        for name, values in self._values.items():
            if not values:
                raise RuntimeError(
                    f"Metric {name} produced no valid values; failure_count={self._failures[name]}"
                )
            output_name: str = self._registry.output_name(name)
            final_metrics[output_name] = sum(values) / len(values)
            final_metrics[f"{output_name}_failure_count"] = float(self._failures[name])
        final_metrics.update(self._compute_aggregate_metrics())
        if self._waveform_names:
            final_metrics["test_utterance_count"] = float(self._utterance_count)
        self._results: dict[str, float] = final_metrics
        self._write_per_utterance_rows()
        self._log_metrics(trainer, final_metrics)

    @property
    def results(self) -> dict[str, float]:
        # Returns a copy of the reduced panel from the completed pass.
        return dict(self._results)

    def _evaluate_utterance(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
        sample_rate: int,
        label: str,
        utterance_ordinal: int
    ) -> None:
        # Evaluates one reference-candidate pair: pitch features are
        # extracted once and shared by all aggregate metrics, then each
        # per-utterance metric computes independently. A failure is
        # counted, logged, and leaves an empty table cell without stopping
        # the remaining metrics or utterances. Non-finite values are treated
        # as failures rather than entering the mean.
        #
        # The pitch extraction is performed here rather than inside the
        # pitch-family metrics, and its two bundles are handed to every
        # active aggregate in turn. One utterance pair therefore costs
        # exactly two backbone invocations whether the run selected one
        # pitch-family metric or all three, and those metrics are
        # guaranteed to have judged identical evidence. Extraction is
        # skipped entirely when no aggregate is selected.
        #
        # Args:
        #     reference: The true signal, already cropped and on the CPU.
        #     candidate: The synthesized signal, cropped to the same
        #         length and on the CPU.
        #     sample_rate: The batch sample rate, which selects the cached
        #         rate-bound instances.
        #     label: Human-readable batch and utterance position, written
        #         into the table row and into any failure warning so a
        #         failure is traceable to the utterance that caused it.
        #     utterance_ordinal: Pass-level index of this pair, recorded as
        #         the row's primary key.
        if self._aggregate_names:
            extractor: PitchExtractor = self._pitch_extractor_for(sample_rate)
            reference_pitch: PitchFeatures = extractor.extract(reference)
            candidate_pitch: PitchFeatures = extractor.extract(candidate)
            self._update_aggregate_metrics(reference_pitch, candidate_pitch)
        utterance_row: dict[str, float | int | str] = {
            "utterance_index": utterance_ordinal,
            "label": label
        }
        for name in self._utterance_names:
            output_name: str = self._registry.output_name(name)
            try:
                value: float = self._compute_waveform_metric(
                    name,
                    reference,
                    candidate,
                    sample_rate
                )
                if not math.isfinite(value):
                    raise ValueError(f"Metric {name} returned a non-finite value")
                self._values[name].append(value)
                utterance_row[output_name] = value
            except Exception as caught:
                failure_count: int = self._failures[name] + 1
                self._failures[name] = failure_count
                utterance_row[output_name] = ""
                log.warning(f"Metric {name} failed for {label}: {caught}")
        self._per_utterance_rows.append(utterance_row)

    def _compute_waveform_metric(
        self,
        name: MetricName,
        reference: torch.Tensor,
        candidate: torch.Tensor,
        sample_rate: int
    ) -> float:
        # Dispatches one per-utterance metric by name. The mel path runs
        # both waveforms through the module's declared metric mel protocol;
        # MCD instances are cached per sample rate; UTMOS scores the
        # candidate alone because it is a no-reference predictor.
        match name:
            case "pesq":
                return self._require_instance(self._pesq_metric, Pesq)(
                    reference,
                    candidate,
                    sample_rate
                )
            case "stoi":
                return self._require_instance(self._stoi_metric, Stoi)(
                    reference,
                    candidate,
                    sample_rate
                )
            case "mel":
                mel_transform: MelSpectrogram = self._require_instance(
                    self._mel_transform,
                    MelSpectrogram
                )
                reference_mel: torch.Tensor = mel_transform(reference.unsqueeze(0))
                candidate_mel: torch.Tensor = mel_transform(candidate.unsqueeze(0))
                return self._require_instance(self._mel_metric, MelError)(
                    reference_mel,
                    candidate_mel
                )
            case "stft":
                return self._require_instance(self._stft_metric, MultiResolutionStftError)(
                    reference,
                    candidate
                )
            case "mcd":
                mcd_metric: MelCepstralDistortion = self._mcd_for(sample_rate)
                return mcd_metric(reference, candidate)
            case "las":
                return self._require_instance(self._las_metric, LogAmplitudeSpectrumRmse)(
                    reference,
                    candidate
                )
            case "utmos":
                return self._require_instance(self._utmos_metric, UtmosPredictor)(
                    candidate,
                    sample_rate
                )
            case _:
                raise ValueError(f"Metric {name} is not a waveform metric")

    def _compute_static_metrics(
        self,
        metric_model: MetricModel,
        module: Module
    ) -> dict[str, float]:
        # Computes the complexity metrics once per run. The parameter count
        # prefers the module's declared deployable count (packed sparse
        # weights) and always reports the residual non-network parameters
        # separately; size reports both the live-object size and the
        # deployable size; MACs are recorded only when the profiler
        # supports the module's execution path.
        metrics: dict[str, float] = {}
        if "parameters" in self._selection.names:
            counter: ParameterCount = self._build_required("parameters", ParameterCount)
            deployed_count: object = getattr(module, "deployable_parameter_count", None)
            if isinstance(deployed_count, int) and deployed_count > 0:
                metrics[self._registry.output_name("parameters")] = float(deployed_count)
                metrics["residual_parameter_count"] = float(
                    counter.residual_count(metric_model.network)
                )
            else:
                metrics[self._registry.output_name("parameters")] = float(
                    counter(metric_model.network)
                )
                metrics["residual_parameter_count"] = float(
                    counter.residual_count(metric_model.network)
                )
        if "size" in self._selection.names:
            size_metric: ModelSize = self._build_required("size", ModelSize)
            metrics[self._registry.output_name("size")] = size_metric(metric_model.network)
            metrics["deployable_size_megabytes"] = self._resolve_deployable_size(
                module,
                size_metric,
                metric_model
            )
        if "macs" in self._selection.names:
            profiler: MacsProfiler = self._build_required("macs", MacsProfiler)
            profile: MacsProfileResult = profiler.profile(module)
            log.info(f"macs_profile={profile.model_dump_json()}")
            if profile.giga_macs_per_audio_second is not None:
                metrics[self._registry.output_name("macs")] = (
                    profile.giga_macs_per_audio_second
                )
        return metrics

    def _resolve_deployable_size(
        self,
        module: Module,
        size_metric: ModelSize,
        metric_model: MetricModel
    ) -> float:
        # Prefers an exported deployment artifact size when the module declares one,
        # otherwise measures the exact serialized bytes of the live network state.
        artifact_bytes: object = getattr(module, "deployable_artifact_bytes", None)
        if isinstance(artifact_bytes, int) and artifact_bytes > 0:
            return artifact_bytes / (1024.0 * 1024.0)
        return size_metric.serialized_megabytes(metric_model.network)

    def _write_per_utterance_rows(self) -> None:
        # Persists the per-utterance metric table for paired statistical evidence.
        if self._per_utterance_csv_path is None or not self._per_utterance_rows:
            return
        output_columns: list[str] = ["utterance_index", "label"] + [
            self._registry.output_name(name) for name in self._utterance_names
        ]
        self._per_utterance_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self._per_utterance_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer: csv.DictWriter = csv.DictWriter(csv_file, fieldnames=output_columns)
            writer.writeheader()
            for utterance_row in self._per_utterance_rows:
                writer.writerow(utterance_row)
        log.info(
            f"Per-utterance metric table with {len(self._per_utterance_rows)} rows "
            f"written to {self._per_utterance_csv_path}"
        )

    def _reset_aggregate_metrics(self) -> None:
        # Clears the accumulated state of the pitch-family aggregates at the
        # start of a pass.
        if self._f0_metric is not None:
            self._f0_metric.reset()
        if self._periodicity_metric is not None:
            self._periodicity_metric.reset()
        if self._voicing_metric is not None:
            self._voicing_metric.reset()

    def _update_aggregate_metrics(
        self,
        reference: PitchFeatures,
        candidate: PitchFeatures
    ) -> None:
        # Feeds one utterance's shared pitch features to every active
        # aggregate metric.
        if self._f0_metric is not None:
            self._f0_metric.update(reference, candidate)
        if self._periodicity_metric is not None:
            self._periodicity_metric.update(reference, candidate)
        if self._voicing_metric is not None:
            self._voicing_metric.update(reference, candidate)

    def _compute_aggregate_metrics(self) -> dict[str, float]:
        # Finalizes the pitch-family aggregates from their whole-pass state.
        metrics: dict[str, float] = {}
        if self._f0_metric is not None:
            metrics[self._registry.output_name("f0")] = self._f0_metric.compute()
        if self._periodicity_metric is not None:
            metrics[self._registry.output_name("periodicity")] = (
                self._periodicity_metric.compute()
            )
        if self._voicing_metric is not None:
            metrics[self._registry.output_name("voicing")] = self._voicing_metric.compute()
        return metrics

    def _pitch_extractor_for(self, sample_rate: int) -> PitchExtractor:
        # Returns the pitch extractor for a sample rate, building it on
        # first use; caching avoids re-loading the pitch model per batch.
        #
        # The cache is keyed by sample rate because the extraction grid is
        # bound at construction, and it is populated lazily so a run that
        # selected no pitch-family metric never constructs one at all.
        cached_extractor: PitchExtractor | None = self._pitch_extractors.get(sample_rate)
        if cached_extractor is not None:
            return cached_extractor
        extractor: PitchExtractor = PitchExtractor(
            PitchConfig(sample_rate=sample_rate, device=self._metric_device)
        )
        self._pitch_extractors[sample_rate] = extractor
        return extractor

    def _mcd_for(self, sample_rate: int) -> MelCepstralDistortion:
        # Returns the MCD metric for a sample rate, building it on first
        # use; MCD binds its analysis grid to the rate at construction.
        cached_metric: MelCepstralDistortion | None = self._mcd_metrics.get(sample_rate)
        if cached_metric is not None:
            return cached_metric
        metric: MelCepstralDistortion = self._build_required(
            "mcd",
            MelCepstralDistortion,
            sample_rate
        )
        self._mcd_metrics[sample_rate] = metric
        return metric

    def _build_optional[MetricType](
        self,
        name: MetricName,
        expected_type: type[MetricType],
        device: str = "cpu"
    ) -> MetricType | None:
        # Builds a metric only when its name is selected; unselected metrics
        # stay None and are never constructed.
        if name not in self._selection.names:
            return None
        return self._build_required(name, expected_type, device=device)

    def _build_required[MetricType](
        self,
        name: MetricName,
        expected_type: type[MetricType],
        sample_rate: int | None = None,
        device: str = "cpu"
    ) -> MetricType:
        # Builds a metric through the registry and proves its runtime type
        # matches the caller's expectation.
        metric: object = self._registry.build(name, sample_rate, device)
        return self._require_instance(metric, expected_type)

    def _require_instance[ValueType](
        self,
        value: object | None,
        expected_type: type[ValueType]
    ) -> ValueType:
        # Narrows an optional slot to its expected type, failing loudly on
        # a metric that was never built for this selection.
        if not isinstance(value, expected_type):
            actual_type: str = type(value).__name__ if value is not None else "None"
            raise TypeError(f"Expected {expected_type.__name__}, got {actual_type}")
        return value

    def _require_metric_model(self, module: Module) -> MetricModel:
        # Verifies the evaluated module structurally satisfies the metric
        # model contract before any measurement uses it.
        network: object = getattr(module, "network", None)
        metric_mel_protocol: object = getattr(module, "metric_mel_protocol", None)
        if not isinstance(network, torch.nn.Module) or not isinstance(
            metric_mel_protocol,
            MelConfig
        ):
            raise TypeError(
                f"Module {type(module).__name__} does not satisfy the metric model contract"
            )
        return cast(MetricModel, module)

    def _extract_reference_batch(self, batch: Batch) -> torch.Tensor:
        # Reads the reference waveforms from the collated batch and
        # normalizes accepted shapes to [batch, time].
        if not isinstance(batch, dict):
            raise TypeError(f"Expected mapping batch, got {type(batch).__name__}")
        waveform: Batch | None = batch.get("waveform")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("batch['waveform'] must be a Tensor")
        normalized: torch.Tensor = waveform
        if normalized.ndim == 1:
            normalized: torch.Tensor = normalized.unsqueeze(0)
        if normalized.ndim == 3 and normalized.shape[1] == 1:
            normalized: torch.Tensor = normalized.squeeze(1)
        if normalized.ndim != 2:
            raise ValueError(f"Expected waveform shape [batch, time], got {tuple(normalized.shape)}")
        return normalized

    def _extract_candidate_batch(self, outputs: StepOutput) -> torch.Tensor:
        # Reads the synthesized waveforms from the test-step output under
        # the required key and normalizes accepted shapes to [batch, time].
        candidate: torch.Tensor | float | None = outputs.get("synthesized_waveform")
        if not isinstance(candidate, torch.Tensor):
            raise TypeError("test_step output must contain Tensor key 'synthesized_waveform'")
        normalized: torch.Tensor = candidate
        if normalized.ndim == 1:
            normalized: torch.Tensor = normalized.unsqueeze(0)
        if normalized.ndim == 3 and normalized.shape[1] == 1:
            normalized: torch.Tensor = normalized.squeeze(1)
        if normalized.ndim != 2:
            raise ValueError(
                f"Expected synthesized waveform shape [batch, time], got {tuple(normalized.shape)}"
            )
        return normalized

    def _extract_lengths(self, batch: Batch, reference: torch.Tensor) -> tuple[int, ...]:
        # Reads the true per-utterance lengths, accepting tensor, sequence,
        # or scalar forms; without lengths the padded width is assumed.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected mapping batch, got {type(batch).__name__}")
        raw_lengths: Batch | None = batch.get("waveform_length")
        if raw_lengths is None:
            return tuple(int(reference.shape[-1]) for _ in range(reference.shape[0]))
        if isinstance(raw_lengths, torch.Tensor):
            lengths: tuple[int, ...] = tuple(int(value) for value in raw_lengths.flatten().tolist())
        elif isinstance(raw_lengths, tuple | list):
            lengths: tuple[int, ...] = tuple(int(value) for value in raw_lengths)
        elif isinstance(raw_lengths, int):
            lengths: tuple[int, ...] = (raw_lengths,)
        else:
            raise TypeError(f"Unsupported waveform_length type: {type(raw_lengths).__name__}")
        if len(lengths) != reference.shape[0]:
            raise ValueError(
                f"waveform_length count {len(lengths)} does not match batch size {reference.shape[0]}"
            )
        return lengths

    def _extract_sample_rate(self, batch: Batch) -> int:
        # Reads the batch sample rate in scalar or tensor form.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected mapping batch, got {type(batch).__name__}")
        raw_sample_rate: Batch | None = batch.get("sample_rate")
        if isinstance(raw_sample_rate, int):
            return raw_sample_rate
        if isinstance(raw_sample_rate, torch.Tensor):
            return int(raw_sample_rate.flatten()[0].item())
        raise TypeError(f"Unsupported sample_rate type: {type(raw_sample_rate).__name__}")

    def _log_metrics(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        # Publishes a metric group through the experiment logger against the
        # trainer's global step.
        if metrics and trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)
