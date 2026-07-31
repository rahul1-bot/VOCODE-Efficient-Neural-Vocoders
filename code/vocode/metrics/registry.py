# This module:
# 1. Defines the closed metric vocabulary of the study, its partition into
#    static and waveform families, and the mapping from metric names onto
#    the output column names of the result schema
# 2. Validates metric selections and constructs metric instances by name
#
# Design decisions:
# - The registry is the single authority on which metrics exist; an unknown
#   or duplicated name fails selection validation before any run starts
# - Output names differ from metric names where the column must state its
#   unit or definition (for example f0 reports f0_rmse_cents), keeping the
#   CSV self-describing
# - The rtf name is registered but not buildable here, because real-time
#   factor is a prediction-stage callback that requires the run's timing
#   configuration rather than a stateless metric object
#
# Author: Rahul Sawhney

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, field_validator

from vocode.metrics.f0 import F0Rmse
from vocode.metrics.las import LogAmplitudeSpectrumRmse, LogAmplitudeSpectrumRmseConfig
from vocode.metrics.macs import MacsProfiler, MacsProfilerConfig
from vocode.metrics.mcd import MelCepstralDistortion, MelCepstralDistortionConfig
from vocode.metrics.mel import MelError
from vocode.metrics.parameters import ParameterCount
from vocode.metrics.periodicity import PeriodicityRmse
from vocode.metrics.pesq import Pesq, PesqConfig
from vocode.metrics.size import ModelSize
from vocode.metrics.stft import MultiResolutionStftError, MultiResolutionStftErrorConfig
from vocode.metrics.stoi import Stoi, StoiConfig
from vocode.metrics.utmos import UtmosPredictor, UtmosPredictorConfig
from vocode.metrics.voicing import VoicingF1

__all__: list[str] = ["MetricName", "MetricRegistry", "MetricSelection"]

type MetricName = str


class MetricSelection(BaseModel):
    # Frozen, registry-validated set of metric names for one run; the
    # default covers the lightweight quality and complexity panel.
    #
    # The record is an ordered tuple rather than a set, and validation
    # preserves the caller's order exactly. That order is load-bearing
    # downstream: the metric sequence evaluates its per-utterance metrics
    # in it and lays out the per-utterance table's columns in it. It does
    # not affect the reduced panel, which is a mapping keyed by output name
    # and therefore identical however the selection was ordered.
    #
    # Fields:
    #     names: Ordered metric names to compute for this run. Every name
    #         must belong to the registry vocabulary and appear at most
    #         once, and the selection must be non-empty; a run measuring
    #         nothing fails before it starts rather than producing an empty
    #         panel. Default: the lightweight panel
    #         ``("pesq", "stoi", "mel", "rtf", "parameters", "size")``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    names: tuple[MetricName, ...] = (
        "pesq",
        "stoi",
        "mel",
        "rtf",
        "parameters",
        "size"
    )

    @field_validator("names")
    @classmethod
    def validate_names(cls, names: tuple[MetricName, ...]) -> tuple[MetricName, ...]:
        # Delegates to the registry so selection validity and registry
        # membership can never disagree.
        return MetricRegistry().validate(names)


class MetricRegistry:
    # The metric vocabulary authority: the full name set, its family
    # partition, output-name mapping, and by-name construction.
    #
    # The vocabulary is a closed, ordered tuple of fourteen names, and the
    # declared order is part of the contract rather than an accident of
    # editing. It groups the ten waveform metrics first, then the
    # callback-only rtf name, then the three static complexity metrics, so
    # reading the tuple states the order in which a run's work is
    # organized: per-utterance measurement before whole-model measurement.
    #
    # The two family tuples partition the vocabulary around rtf: every name
    # is either a waveform metric, a static metric, or rtf, and no name is
    # in both families. rtf belongs to neither because real-time factor is
    # measured by a prediction-stage callback rather than computed from a
    # signal pair or a network, which is also why it is the one registered
    # name that refuses construction here. Each family lists its own
    # members in vocabulary order, so a caller may iterate a family and a
    # caller may iterate the vocabulary and the two agree.
    #
    # Composition by name is the whole point of the registry: a selection
    # is a tuple of names, this class alone decides whether those names
    # exist, this class alone builds their instances, and this class alone
    # maps them onto result columns. No other component may extend the
    # vocabulary, which is what makes an unknown or duplicated metric a
    # failure before a run starts instead of a missing column afterwards.
    # Instances hold no state, so a caller constructs one wherever the
    # vocabulary is needed.
    names: ClassVar[tuple[MetricName, ...]] = (
        "pesq",
        "stoi",
        "mel",
        "stft",
        "mcd",
        "las",
        "f0",
        "periodicity",
        "voicing",
        "utmos",
        "rtf",
        "macs",
        "parameters",
        "size"
    )
    static_names: ClassVar[tuple[MetricName, ...]] = ("macs", "parameters", "size")
    waveform_names: ClassVar[tuple[MetricName, ...]] = (
        "pesq",
        "stoi",
        "mel",
        "stft",
        "mcd",
        "las",
        "f0",
        "periodicity",
        "voicing",
        "utmos"
    )

    def validate(self, names: tuple[MetricName, ...]) -> tuple[MetricName, ...]:
        # Accepts a selection only when it is non-empty, known, and free of
        # duplicates, naming the offenders otherwise.
        #
        # Args:
        #     names: The selection to check, in the caller's order.
        #
        # Returns:
        #     The same tuple unchanged. Validation neither sorts nor
        #     deduplicates nor rewrites the selection, so the caller's
        #     order survives into the run and remains observable in the
        #     per-utterance table's column layout.
        #
        # Raises:
        #     ValueError: If the selection is empty, since a run must
        #         measure something; if any name is outside the
        #         vocabulary, in which case the message names the offenders
        #         and lists the available vocabulary; or if any name
        #         repeats, since a duplicate would compute one metric twice
        #         and give it two chances to fail.
        if not names:
            raise ValueError("At least one metric name is required")
        unknown_names: tuple[str, ...] = tuple(name for name in names if name not in self.names)
        if unknown_names:
            raise ValueError(f"Unknown metrics: {unknown_names}; available={self.names}")
        if len(set(names)) != len(names):
            raise ValueError(f"Metric names must be unique, got {names}")
        return names

    def output_name(self, name: MetricName) -> str:
        # Maps a metric name onto its result-schema column name; names
        # without a special mapping pass through unchanged.
        #
        # A metric name is chosen for brevity at the call site, while a
        # result column must state the unit or definition of the number it
        # holds, because the exported table is read without the code beside
        # it. The mapping is injective over the vocabulary, so two metrics
        # can never collide into one column and overwrite each other.
        #
        # Args:
        #     name: The metric name to map. Membership is not checked here;
        #         this is a pure rename and validate is the membership
        #         authority, so an unregistered name passes through as
        #         itself.
        #
        # Returns:
        #     The column name under which this metric's value appears in
        #     the result schema, the per-utterance table, and the published
        #     metric payload.
        match name:
            case "mel":
                return "mel_error"
            case "stft":
                return "multi_resolution_stft_error"
            case "las":
                return "las_rmse"
            case "f0":
                return "f0_rmse_cents"
            case "periodicity":
                return "periodicity_rmse"
            case "voicing":
                return "vuv_f1"
            case "utmos":
                return "utmos_strong"
            case "rtf":
                return "real_time_factor"
            case "macs":
                return "macs_per_second_audio"
            case "parameters":
                return "parameter_count"
            case "size":
                return "model_size_megabytes"
            case _:
                return name

    def build(
        self,
        name: MetricName,
        sample_rate: int | None = None,
        device: str = "cpu"
    ) -> object:
        # Constructs one metric instance by name. Only MCD consumes the
        # sample rate (defaulting to the LJSpeech rate) and only UTMOS
        # consumes the device; rtf refuses construction here by design.
        #
        # Every call returns a freshly built instance, never a shared one,
        # because the pitch-family metrics accumulate state across a pass
        # and a shared instance would leak one run's frames into the next.
        # The two optional arguments exist because exactly two metrics are
        # not fully self-describing: mel-cepstral distortion binds its
        # analysis grid to the run's sample rate at construction, and UTMOS
        # binds an execution device. Every other metric ignores both, which
        # is why a rate passed alongside a rate-independent name has no
        # effect on the instance built.
        #
        # Args:
        #     name: The metric name to construct, expected to have passed
        #         validate already.
        #     sample_rate: Run sample rate in hertz, consumed by
        #         mel-cepstral distortion alone; ``None`` falls back to the
        #         corpus rate. Default: ``None``.
        #     device: Torch device string, consumed by the UTMOS predictor
        #         alone. Default: ``"cpu"``.
        #
        # Returns:
        #     The constructed metric, typed loosely because the vocabulary
        #     spans unrelated metric shapes; the caller narrows the result
        #     to the type it expects.
        #
        # Raises:
        #     ValueError: If the name is rtf, which is a prediction-stage
        #         callback needing the run's timing configuration and so
        #         cannot be produced as a stateless metric object; or if
        #         the name is outside the vocabulary, since construction
        #         must not invent a metric the registry does not declare.
        match name:
            case "pesq":
                return Pesq(PesqConfig())
            case "stoi":
                return Stoi(StoiConfig())
            case "mel":
                return MelError()
            case "stft":
                return MultiResolutionStftError(MultiResolutionStftErrorConfig())
            case "mcd":
                return MelCepstralDistortion(
                    MelCepstralDistortionConfig(sample_rate=sample_rate or 22050)
                )
            case "las":
                return LogAmplitudeSpectrumRmse(LogAmplitudeSpectrumRmseConfig())
            case "f0":
                return F0Rmse()
            case "periodicity":
                return PeriodicityRmse()
            case "voicing":
                return VoicingF1()
            case "utmos":
                return UtmosPredictor(UtmosPredictorConfig(device=device))
            case "macs":
                return MacsProfiler(MacsProfilerConfig())
            case "parameters":
                return ParameterCount()
            case "size":
                return ModelSize()
            case "rtf":
                raise ValueError("RTF is a prediction-stage callback and requires its run configuration")
            case _:
                raise ValueError(f"Unknown metric: {name}")
