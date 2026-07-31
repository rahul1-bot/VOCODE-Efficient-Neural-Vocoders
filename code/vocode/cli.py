# This module:
# 1. Implements the VOCODE command-line interface: one atomic command per
#    evidence lane and stage, converting command-line input into one
#    validated ExperimentConfiguration and dispatching the matching
#    runner inside a RunTracker lifecycle
# 2. Resolves configuration from three layered sources: the optional
#    YAML file, dotted-path override flags, and command-line arguments,
#    all funneled through pydantic validation before anything executes
#
# Design decisions:
# - Every command runs exactly one (architecture, seed, stage) cell;
#   fleet orchestration lives outside this process, so a crashed cell
#   can never take sibling cells with it
# - The evidence category selects the lane runner (Study 1 reproduction
#   and hybrid training, Study 2 recovery training and optimized
#   evaluation, published reference evaluation) through one exhaustive
#   dispatch
# - Validation and configuration errors surface as argparse errors, so
#   misuse produces usage guidance rather than a traceback
#
# Author: Rahul Sawhney

import argparse
import sys
from pathlib import Path
from typing import ClassVar, Literal, assert_never, get_args

import yaml
from loguru import logger as log
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveFloat, PositiveInt, ValidationError

from vocode.configs.layout import (
    DatasetSplitName,
    EvidenceCategory,
    ExperimentArtifactLayout,
    ExperimentStage,
    HardwareName,
    PrecisionName,
)
from vocode.configs.run import BatchLimit, ExperimentConfiguration
from vocode.data.ljspeech_datamodule import LJSpeechDataConfig
from vocode.loggers.tracker import RunTracker
from vocode.metrics.registry import MetricRegistry, MetricSelection
from vocode.metrics.rtf import RealTimeFactorConfig
from vocode.models.vocoder import ArchitectureName
from vocode.optimization.recovery import OptimizationRecoveryRunner
from vocode.optimization.registry import OptimizationVariantName
from vocode.trainers.optimized import OptimizedVariantEvaluator
from vocode.trainers.published import PublishedWeightsEvaluator
from vocode.trainers.reproduction import ReproductionTrainingRunner

__all__: list[str] = ["VocodeCli"]

type VocodeCliCommandName = Literal[
    "train-reproduction",
    "validation-reproduction",
    "test-reproduction",
    "train-hybrid",
    "test-hybrid",
    "test-published",
    "evaluate-optimized-variant",
    "recover-optimized-variant"
]


class VocodeCliArchitectureSet:
    # The closed architecture-name choices offered by the command line,
    # mirroring the registry vocabulary.
    names: ClassVar[tuple[ArchitectureName, ...]] = (
        "hifigan_v1",
        "hifigan_v2",
        "hifigan_v3",
        "melgan",
        "vocos",
        "bigvgan",
        "apnet2",
        "freev",
        "hiftnet",
        "lpcnet",
        "rndvoc",
        "vocosformer",
        "rfwave"
    )


class VocodeCliFileConfiguration(BaseModel):
    # Frozen shape of the optional YAML configuration file; every field is
    # optional because the command line can supply or override any of
    # them. Each field mirrors one resolved run or data setting, and
    # ``None`` means the file layer expressed no opinion, which is what
    # lets the resolver fall through to the declared default. The record
    # is closed, so a misspelled setting fails loudly at validation
    # instead of being ignored and silently leaving the default in place.
    # Unlike the records this feeds, it is not strict: values arriving
    # from YAML as strings are coerced at this boundary, which is how a
    # path written as plain text becomes a Path.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    experiment_name: str | None = None
    hypothesis: str | None = None
    interpretation_notes: str | None = None
    artifact_root: Path | None = None
    dataset_root: Path | None = None
    published_weights_root: Path | None = None
    project_checkpoint_path: Path | None = None
    hiftnet_f0_checkpoint_path: Path | None = None
    hardware_name: HardwareName | None = None
    precision_name: PrecisionName | None = None
    accelerator: str | None = None
    train_epoch_count: PositiveInt | None = None
    training_batch_size: PositiveInt | None = None
    validation_batch_size: PositiveInt | None = None
    test_batch_size: PositiveInt | None = None
    prediction_batch_size: PositiveInt | None = None
    num_workers: int | None = None
    persistent_workers: bool | None = None
    prefetch_factor: PositiveInt | None = None
    pin_memory: bool | None = None
    test_split_size: PositiveInt | None = None
    validation_split_size: PositiveInt | None = None
    partition_strategy: Literal["ordered_identifier_holdout"] | None = None
    max_test_utterances: PositiveInt | None = None
    training_segment_size: PositiveInt | None = None
    peak_normalization_enabled: bool | None = None
    peak_normalization_value: PositiveFloat | None = None
    rtf_warmup_iterations: NonNegativeInt | None = None
    rtf_repetition_count: PositiveInt | None = None
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    limit_predict_batches: int | float | None = None
    runtime_profiling_enabled: bool | None = None
    runtime_profile_interval_steps: PositiveInt | None = None
    metrics: tuple[str, ...] | None = None


class VocodeCliRequest(BaseModel):
    # Frozen record of one parsed command invocation before configuration
    # resolution. It is the command-line layer expressed as data: the
    # subcommand's own bindings (evidence category, stage, default split)
    # sit beside the flags the caller supplied. Every optional field is
    # ``None`` exactly when its flag was absent, which is what separates
    # "not supplied" from "supplied with a value that equals the default"
    # and therefore what gives the command line its precedence over the
    # file configuration.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True
    )
    command_name: VocodeCliCommandName
    evidence_category: EvidenceCategory
    stage: ExperimentStage
    dataset_split_name: DatasetSplitName
    config_path: Path | None
    override_values: list[str]
    architecture_name: ArchitectureName
    optimization_variant_name: str | None
    optimization_hypothesis_id: str | None
    code_commit_hash: str | None
    seed: int
    run_id: str
    experiment_name: str | None
    hypothesis: str | None
    interpretation_notes: str | None
    artifact_root: Path | None
    dataset_root: Path | None
    published_weights_root: Path | None
    project_checkpoint_path: Path | None
    hiftnet_f0_checkpoint_path: Path | None
    hardware_name: HardwareName | None
    precision_name: PrecisionName | None
    accelerator: str | None
    train_epoch_count: int | None
    training_batch_size: int | None
    validation_batch_size: int | None
    test_batch_size: int | None
    prediction_batch_size: int | None
    num_workers: int | None
    persistent_workers: bool | None
    prefetch_factor: int | None
    pin_memory: bool | None
    test_split_size: int | None
    validation_split_size: int | None
    max_test_utterances: int | None
    training_segment_size: int | None
    peak_normalization_enabled: bool | None
    peak_normalization_value: float | None
    rtf_warmup_iterations: int | None
    rtf_repetition_count: int | None
    limit_train_batches: BatchLimit
    limit_val_batches: BatchLimit
    limit_test_batches: BatchLimit
    limit_predict_batches: BatchLimit
    runtime_profiling_enabled: bool | None
    runtime_profile_interval_steps: int | None
    metrics: tuple[str, ...] | None

    @classmethod
    def from_namespace(cls, parsed_arguments: argparse.Namespace) -> VocodeCliRequest:
        # Converts the argparse namespace into the validated request record. The two
        # accumulating flags are copied rather than referenced, so the frozen record owns its
        # override list and metric tuple and cannot be edited afterwards through the mutable
        # lists argparse built.
        return cls(
            command_name=parsed_arguments.command_name,
            evidence_category=parsed_arguments.evidence_category,
            stage=parsed_arguments.stage,
            dataset_split_name=parsed_arguments.dataset_split_name,
            config_path=parsed_arguments.config_path,
            override_values=list(parsed_arguments.override_values),
            architecture_name=parsed_arguments.architecture_name,
            optimization_variant_name=parsed_arguments.optimization_variant_name,
            optimization_hypothesis_id=parsed_arguments.optimization_hypothesis_id,
            code_commit_hash=parsed_arguments.code_commit_hash,
            seed=parsed_arguments.seed,
            run_id=parsed_arguments.run_id,
            experiment_name=parsed_arguments.experiment_name,
            hypothesis=parsed_arguments.hypothesis,
            interpretation_notes=parsed_arguments.interpretation_notes,
            artifact_root=parsed_arguments.artifact_root,
            dataset_root=parsed_arguments.dataset_root,
            published_weights_root=parsed_arguments.published_weights_root,
            project_checkpoint_path=parsed_arguments.project_checkpoint_path,
            hiftnet_f0_checkpoint_path=parsed_arguments.hiftnet_f0_checkpoint_path,
            hardware_name=parsed_arguments.hardware_name,
            precision_name=parsed_arguments.precision_name,
            accelerator=parsed_arguments.accelerator,
            train_epoch_count=parsed_arguments.train_epoch_count,
            training_batch_size=parsed_arguments.training_batch_size,
            validation_batch_size=parsed_arguments.validation_batch_size,
            test_batch_size=parsed_arguments.test_batch_size,
            prediction_batch_size=parsed_arguments.prediction_batch_size,
            num_workers=parsed_arguments.num_workers,
            persistent_workers=parsed_arguments.persistent_workers,
            prefetch_factor=parsed_arguments.prefetch_factor,
            pin_memory=parsed_arguments.pin_memory,
            test_split_size=parsed_arguments.test_split_size,
            validation_split_size=parsed_arguments.validation_split_size,
            max_test_utterances=parsed_arguments.max_test_utterances,
            training_segment_size=parsed_arguments.training_segment_size,
            peak_normalization_enabled=parsed_arguments.peak_normalization_enabled,
            peak_normalization_value=parsed_arguments.peak_normalization_value,
            rtf_warmup_iterations=parsed_arguments.rtf_warmup_iterations,
            rtf_repetition_count=parsed_arguments.rtf_repetition_count,
            limit_train_batches=parsed_arguments.limit_train_batches,
            limit_val_batches=parsed_arguments.limit_val_batches,
            limit_test_batches=parsed_arguments.limit_test_batches,
            limit_predict_batches=parsed_arguments.limit_predict_batches,
            runtime_profiling_enabled=parsed_arguments.runtime_profiling_enabled,
            runtime_profile_interval_steps=parsed_arguments.runtime_profile_interval_steps,
            metrics=(
                tuple(parsed_arguments.metrics)
                if parsed_arguments.metrics is not None
                else None
            )
        )


class VocodeCliYamlLoader:
    # Loader for the optional YAML configuration file.
    def load(self, config_path: Path | None) -> dict[str, object]:
        # Reads and validates the YAML file into the file-configuration
        # record; a missing path yields the empty configuration. An empty
        # document is likewise the empty configuration, since a file that
        # declares nothing and no file at all are the same statement. A
        # path that was named but does not exist is a different matter and
        # is not absorbed, because silently running with defaults after a
        # mistyped path would record a run nobody asked for.
        #
        # Raises:
        #     FileNotFoundError: If a path is given and no file is there.
        #     ValueError: If the document is not a mapping or carries a
        #         non-string key, neither of which can name a setting.
        if config_path is None:
            return {}
        loaded_document: object = yaml.safe_load(config_path.read_text())
        if loaded_document is None:
            return {}
        if not isinstance(loaded_document, dict):
            raise ValueError(f"Configuration file must contain a mapping: {config_path}")
        configuration: dict[str, object] = {}
        for key, value in loaded_document.items():
            if not isinstance(key, str):
                raise ValueError(f"Configuration key must be a string in {config_path}")
            configuration[key] = value
        return configuration


class VocodeCliOverrideParser:
    # Parser applying dotted-path override flags onto the file
    # configuration before validation.
    def apply(self, configuration: dict[str, object], override_values: list[str]) -> dict[str, object]:
        # Applies every dotted-path override onto a copy of the file configuration, in the
        # order the flags were given, so a later override of one key wins over an earlier one.
        # The incoming mapping is never mutated, which keeps the loaded document available
        # unchanged for anything that needs to distinguish file values from overridden ones.
        #
        # Returns:
        #     The merged mapping, still unvalidated: overrides are typed
        #     here but checked against the file record afterwards, so an
        #     override naming an unknown setting fails validation rather
        #     than being silently accepted.
        resolved_configuration: dict[str, object] = dict(configuration)
        for override_value in override_values:
            key, value = self._parse_override(override_value)
            resolved_configuration[key] = value
        return resolved_configuration

    def _parse_override(self, override_value: str) -> tuple[str, object]:
        # Converts override text into a typed value before validation.
        if "=" not in override_value:
            raise ValueError(f"Override must use key=value syntax: {override_value}")
        key, raw_value = override_value.split("=", 1)
        normalized_key: str = key.strip()
        if not normalized_key:
            raise ValueError(f"Override key cannot be empty: {override_value}")
        parsed_value: object = yaml.safe_load(raw_value)
        return normalized_key, parsed_value


class VocodeCliConfigurationFactory:
    # Factory resolving one ExperimentConfiguration from the layered
    # sources: file values, override flags, command-line arguments, and
    # the artifact layout derived from them.
    def __init__(self) -> None:
        # Binds the collaborators used during configuration resolution.
        self._yaml_loader: VocodeCliYamlLoader = VocodeCliYamlLoader()
        self._override_parser: VocodeCliOverrideParser = VocodeCliOverrideParser()

    def build(self, request: VocodeCliRequest) -> ExperimentConfiguration:
        # Resolves the complete ExperimentConfiguration for one request
        # from the layered sources. Resolution runs in a fixed order: read
        # the YAML document, apply the override flags onto it, validate the
        # merged mapping as the file record, build the artifact layout,
        # data, timing, and metric collaborators from it, check the
        # requirements the specific command imposes, and only then
        # construct the experiment record, whose own validators enforce the
        # stage-to-split binding and the optimized-variant contract.
        #
        # Returns:
        #     The validated run description. Nothing it names is read here:
        #     the corpus, checkpoints, and weights directories remain
        #     unopened values, because this stage resolves settings rather
        #     than loading artifacts.
        #
        # Raises:
        #     ValueError: If the document is not a mapping, an override is
        #         malformed, a resolved number leaves its domain, the
        #         corpus root was supplied by neither source, or the
        #         command requires a checkpoint that is absent.
        #     ValidationError: If the merged file mapping or the resulting
        #         experiment record fails validation. The caller converts
        #         both into argparse usage errors.
        raw_configuration: dict[str, object] = self._yaml_loader.load(request.config_path)
        overridden_configuration: dict[str, object] = self._override_parser.apply(
            raw_configuration,
            request.override_values
        )
        file_configuration: VocodeCliFileConfiguration = VocodeCliFileConfiguration.model_validate(
            overridden_configuration
        )
        artifact_layout: ExperimentArtifactLayout = self._build_artifact_layout(request, file_configuration)
        data_configuration: LJSpeechDataConfig = self._build_data_configuration(request, file_configuration)
        rtf_configuration: RealTimeFactorConfig = self._build_rtf_configuration(request, file_configuration)
        metric_selection: MetricSelection = self._build_metric_selection(request, file_configuration)
        self._validate_stage_requirements(request, file_configuration)
        return ExperimentConfiguration(
            experiment_name=self._resolve_str(request.experiment_name, file_configuration.experiment_name, "vocode_checkpoint01"),
            run_id=request.run_id,
            hypothesis=self._resolve_str(request.hypothesis, file_configuration.hypothesis, "Controlled VOCODE reproduction run."),
            interpretation_notes=self._resolve_str(request.interpretation_notes, file_configuration.interpretation_notes, ""),
            evidence_category=request.evidence_category,
            stage=request.stage,
            dataset_split_name=request.dataset_split_name,
            architecture_name=request.architecture_name,
            seed=request.seed,
            artifact_layout=artifact_layout,
            published_weights_root=self._resolve_path(
                request.published_weights_root,
                file_configuration.published_weights_root,
                Path("published_weights")
            ),
            project_checkpoint_path=self._resolve_optional_path(
                request.project_checkpoint_path,
                file_configuration.project_checkpoint_path
            ),
            hiftnet_f0_checkpoint_path=self._resolve_optional_path(
                request.hiftnet_f0_checkpoint_path,
                file_configuration.hiftnet_f0_checkpoint_path
            ),
            optimization_variant_name=request.optimization_variant_name,
            optimization_hypothesis_id=request.optimization_hypothesis_id,
            code_commit_hash=request.code_commit_hash,
            data_configuration=data_configuration,
            metric_selection=metric_selection,
            real_time_factor_configuration=rtf_configuration,
            train_epoch_count=self._resolve_positive_int(
                request.train_epoch_count,
                file_configuration.train_epoch_count,
                1
            ),
            limit_train_batches=self._resolve_optional_batch_limit(
                request.limit_train_batches,
                file_configuration.limit_train_batches
            ),
            limit_val_batches=self._resolve_optional_batch_limit(
                request.limit_val_batches,
                file_configuration.limit_val_batches
            ),
            limit_test_batches=self._resolve_optional_batch_limit(
                request.limit_test_batches,
                file_configuration.limit_test_batches
            ),
            limit_predict_batches=self._resolve_optional_batch_limit(
                request.limit_predict_batches,
                file_configuration.limit_predict_batches
            ),
            accelerator=self._resolve_str(request.accelerator, file_configuration.accelerator, "auto"),
            runtime_profiling_enabled=self._resolve_bool(
                request.runtime_profiling_enabled,
                file_configuration.runtime_profiling_enabled,
                True
            ),
            runtime_profile_interval_steps=self._resolve_positive_int(
                request.runtime_profile_interval_steps,
                file_configuration.runtime_profile_interval_steps,
                50
            )
        )

    def _build_artifact_layout(
        self,
        request: VocodeCliRequest,
        file_configuration: VocodeCliFileConfiguration
    ) -> ExperimentArtifactLayout:
        # Builds this collaborator from the resolved configuration values.
        return ExperimentArtifactLayout(
            artifact_root=self._resolve_path(
                request.artifact_root,
                file_configuration.artifact_root,
                Path("experiment_artifacts")
            ),
            evidence_category=request.evidence_category,
            architecture_name=request.architecture_name,
            hardware_name=self._resolve_hardware_name(request.hardware_name, file_configuration.hardware_name),
            precision_name=self._resolve_precision_name(request.precision_name, file_configuration.precision_name),
            seed=request.seed,
            run_id=request.run_id,
            variant_name=(
                request.optimization_variant_name
                if request.evidence_category == "project_optimized_variants"
                else None
            )
        )

    def _build_data_configuration(
        self,
        request: VocodeCliRequest,
        file_configuration: VocodeCliFileConfiguration
    ) -> LJSpeechDataConfig:
        # Builds this collaborator from the resolved configuration values. The corpus root is
        # demanded first because it is the one setting with no default, then the architecture
        # selects a block of data defaults, and only then are the layered sources applied on
        # top of them.
        #
        # The match below encodes each architecture's published data recipe: the evaluation
        # batch size, the training crop length, whether waveforms are peak-normalized, the
        # rate the recipe operates at, and the random gain range it trains under. The
        # unlisted arm is a deliberate generic fallback rather than an error, so an
        # architecture can be registered and exercised before its recipe is transcribed. The
        # first three of these are defaults and remain overridable from the command line or
        # the file; the resample rate and the gain range are taken from the architecture
        # alone and consult neither source, so a run cannot contradict the sample rate its
        # reference recipe and mel protocol are defined at.
        dataset_root: Path | None = self._resolve_optional_path(
            request.dataset_root,
            file_configuration.dataset_root
        )
        if dataset_root is None:
            raise ValueError("dataset_root is required through --dataset-root or --config.")
        default_validation_batch_size: int
        default_training_segment_size: int | None
        default_peak_normalization: bool
        default_resample_rate: int | None
        default_training_random_peak_gain_range_db: tuple[float, float] | None
        match request.architecture_name:
            case "hifigan_v1" | "hifigan_v2" | "hifigan_v3":
                default_validation_batch_size: int = 1
                default_training_segment_size: int | None = 8192
                default_peak_normalization: bool = True
                default_resample_rate: int | None = None
                default_training_random_peak_gain_range_db: tuple[float, float] | None = None
            case "melgan":
                default_validation_batch_size: int = 1
                default_training_segment_size: int | None = 16000
                default_peak_normalization: bool = False
                default_resample_rate: int | None = None
                default_training_random_peak_gain_range_db: tuple[float, float] | None = None
            case "vocos":
                default_validation_batch_size: int = 1
                default_training_segment_size: int | None = 16384
                default_peak_normalization: bool = False
                default_resample_rate: int | None = 24000
                default_training_random_peak_gain_range_db: tuple[float, float] | None = (-6.0, -1.0)
            case "vocosformer":
                default_validation_batch_size: int = 1
                default_training_segment_size: int | None = 16384
                default_peak_normalization: bool = False
                default_resample_rate: int | None = 24000
                default_training_random_peak_gain_range_db: tuple[float, float] | None = (-6.0, -1.0)
            case "rfwave":
                default_validation_batch_size: int = 16
                default_training_segment_size: int | None = 32512
                default_peak_normalization: bool = False
                default_resample_rate: int | None = 24000
                default_training_random_peak_gain_range_db: tuple[float, float] | None = None
            case "bigvgan":
                default_validation_batch_size: int = 1
                default_training_segment_size: int | None = 8192
                default_peak_normalization: bool = False
                default_resample_rate: int | None = 24000
                default_training_random_peak_gain_range_db: tuple[float, float] | None = None
            case "apnet2":
                default_validation_batch_size: int = 1
                default_training_segment_size: int | None = 8192
                default_peak_normalization: bool = False
                default_resample_rate: int | None = None
                default_training_random_peak_gain_range_db: tuple[float, float] | None = None
            case "freev":
                default_validation_batch_size: int = 1
                default_training_segment_size: int | None = 8192
                default_peak_normalization: bool = False
                default_resample_rate: int | None = None
                default_training_random_peak_gain_range_db: tuple[float, float] | None = None
            case "rndvoc":
                default_validation_batch_size: int = 1
                default_training_segment_size: int | None = 16384
                default_peak_normalization: bool = False
                default_resample_rate: int | None = None
                default_training_random_peak_gain_range_db: tuple[float, float] | None = None
            case "lpcnet":
                default_validation_batch_size: int = 1
                default_training_segment_size: int | None = 2400
                default_peak_normalization: bool = False
                default_resample_rate: int | None = 16000
                default_training_random_peak_gain_range_db: tuple[float, float] | None = None
            case _:
                default_validation_batch_size: int = 16
                default_training_segment_size: int | None = None
                default_peak_normalization: bool = False
                default_resample_rate: int | None = None
                default_training_random_peak_gain_range_db: tuple[float, float] | None = None
        return LJSpeechDataConfig(
            dataset_root=dataset_root,
            seed=request.seed,
            training_batch_size=self._resolve_positive_int(
                request.training_batch_size,
                file_configuration.training_batch_size,
                16
            ),
            validation_batch_size=self._resolve_positive_int(
                request.validation_batch_size,
                file_configuration.validation_batch_size,
                default_validation_batch_size
            ),
            test_batch_size=self._resolve_positive_int(
                request.test_batch_size,
                file_configuration.test_batch_size,
                16
            ),
            prediction_batch_size=self._resolve_positive_int(
                request.prediction_batch_size,
                file_configuration.prediction_batch_size,
                1
            ),
            num_workers=self._resolve_non_negative_int(
                request.num_workers,
                file_configuration.num_workers,
                0
            ),
            persistent_workers=self._resolve_bool(
                request.persistent_workers,
                file_configuration.persistent_workers,
                False
            ),
            prefetch_factor=self._resolve_optional_positive_int(
                request.prefetch_factor,
                file_configuration.prefetch_factor
            ),
            pin_memory=self._resolve_bool(
                request.pin_memory,
                file_configuration.pin_memory,
                False
            ),
            test_split_size=self._resolve_positive_int(
                request.test_split_size,
                file_configuration.test_split_size,
                525
            ),
            validation_split_size=self._resolve_positive_int(
                request.validation_split_size,
                file_configuration.validation_split_size,
                100
            ),
            partition_strategy=(
                file_configuration.partition_strategy
                if file_configuration.partition_strategy is not None
                else "ordered_identifier_holdout"
            ),
            max_test_utterances=self._resolve_optional_positive_int(
                request.max_test_utterances,
                file_configuration.max_test_utterances
            ),
            training_segment_size=self._resolve_optional_positive_int(
                request.training_segment_size,
                file_configuration.training_segment_size,
                default_training_segment_size
            ),
            peak_normalization_enabled=self._resolve_bool(
                request.peak_normalization_enabled,
                file_configuration.peak_normalization_enabled,
                default_peak_normalization
            ),
            peak_normalization_value=self._resolve_positive_float(
                request.peak_normalization_value,
                file_configuration.peak_normalization_value,
                0.95
            ),
            resample_rate=default_resample_rate,
            training_random_peak_gain_range_db=default_training_random_peak_gain_range_db
        )

    def _build_rtf_configuration(
        self,
        request: VocodeCliRequest,
        file_configuration: VocodeCliFileConfiguration
    ) -> RealTimeFactorConfig:
        # Builds this collaborator from the resolved configuration values.
        # The repetition count is an executed protocol input: the monitor runs exactly
        # this many measured repetitions per timed batch after the warm-up batches.
        return RealTimeFactorConfig(
            warmup_iterations=self._resolve_non_negative_int(
                request.rtf_warmup_iterations,
                file_configuration.rtf_warmup_iterations,
                5
            ),
            timed_repetitions=self._resolve_positive_int(
                request.rtf_repetition_count,
                file_configuration.rtf_repetition_count,
                3
            )
        )

    def _build_metric_selection(
        self,
        request: VocodeCliRequest,
        file_configuration: VocodeCliFileConfiguration
    ) -> MetricSelection:
        # Resolves the metric panel. A selection supplied through either source replaces the
        # default panel outright rather than extending it, so a run measures exactly the
        # metrics it names; only when neither source names any metric does the registry's
        # default panel apply. The names themselves are validated by MetricSelection against
        # the metric registry, so an unregistered name cannot reach a runner.
        metric_names: tuple[str, ...] | None = (
            request.metrics if request.metrics is not None else file_configuration.metrics
        )
        if metric_names is None:
            return MetricSelection()
        return MetricSelection(names=metric_names)

    def _validate_stage_requirements(
        self,
        request: VocodeCliRequest,
        file_configuration: VocodeCliFileConfiguration
    ) -> None:
        # Validates runtime requirements before dispatching the experiment stage. These are the
        # requirements the command imposes rather than the record does: evaluating a trained
        # reproduction has nothing to load without its checkpoint, whereas the training command
        # produces that checkpoint and must not demand one. The check runs before the
        # experiment record is constructed, so the failure names the missing flag instead of
        # surfacing later as a load error inside the runner.
        project_checkpoint_path: Path | None = self._resolve_optional_path(
            request.project_checkpoint_path,
            file_configuration.project_checkpoint_path
        )
        if request.command_name in ("validation-reproduction", "test-reproduction") and project_checkpoint_path is None:
            raise ValueError(
                "project_checkpoint_path is required for validation-reproduction and test-reproduction."
            )

    def _resolve_str(self, request_value: str | None, file_value: str | None, default_value: str) -> str:
        # Resolves this value through the source precedence: command-line
        # input first, then the file configuration, then the default.
        if request_value is not None:
            return request_value
        if file_value is not None:
            return file_value
        return default_value

    def _resolve_int(self, request_value: int | None, file_value: int | None, default_value: int) -> int:
        # Resolves this value through the source precedence: command-line
        # input first, then the file configuration, then the default.
        if request_value is not None:
            return request_value
        if file_value is not None:
            return file_value
        return default_value

    def _resolve_non_negative_int(
        self,
        request_value: int | None,
        file_value: int | None,
        default_value: int
    ) -> int:
        # Resolves this value through the source precedence: command-line
        # input first, then the file configuration, then the default.
        resolved_value: int = self._resolve_int(request_value, file_value, default_value)
        if resolved_value < 0:
            raise ValueError(f"Expected a non-negative integer, got {resolved_value}")
        return resolved_value

    def _resolve_positive_int(
        self,
        request_value: int | None,
        file_value: int | None,
        default_value: int
    ) -> int:
        # Resolves this value through the source precedence: command-line
        # input first, then the file configuration, then the default.
        resolved_value: int = self._resolve_int(request_value, file_value, default_value)
        if resolved_value < 1:
            raise ValueError(f"Expected a positive integer, got {resolved_value}")
        return resolved_value

    def _resolve_optional_positive_int(
        self,
        request_value: int | None,
        file_value: int | None,
        default_value: int | None = None
    ) -> int | None:
        # Resolves this value through the source precedence: command-line
        # input first, then the file configuration, then the default.
        resolved_value: int | None = request_value if request_value is not None else file_value
        if resolved_value is None:
            resolved_value: int | None = default_value
        if resolved_value is None:
            return None
        if resolved_value < 1:
            raise ValueError(f"Expected a positive integer, got {resolved_value}")
        return resolved_value

    def _resolve_positive_float(
        self,
        request_value: float | None,
        file_value: float | None,
        default_value: float
    ) -> float:
        # Resolves this value through the source precedence: command-line
        # input first, then the file configuration, then the default.
        resolved_value: float
        if request_value is not None:
            resolved_value: float = request_value
        elif file_value is not None:
            resolved_value: float = file_value
        else:
            resolved_value: float = default_value
        if resolved_value <= 0.0:
            raise ValueError(f"Expected a positive float, got {resolved_value}")
        return resolved_value

    def _resolve_optional_batch_limit(
        self,
        request_value: BatchLimit,
        file_value: BatchLimit
    ) -> BatchLimit:
        # Resolves execution limits without changing the train/validation/test split contract.
        resolved_value: BatchLimit = request_value if request_value is not None else file_value
        if resolved_value is None:
            return None
        if isinstance(resolved_value, bool):
            raise ValueError("batch limits must be int, float, or None; bool is invalid")
        if isinstance(resolved_value, int):
            if resolved_value < 0:
                raise ValueError(f"integer batch limits must be >= 0, got {resolved_value}")
            return resolved_value
        if resolved_value <= 0.0 or resolved_value > 1.0:
            raise ValueError(f"float batch limits must be in (0.0, 1.0], got {resolved_value}")
        return resolved_value

    def _resolve_bool(self, request_value: bool | None, file_value: bool | None, default_value: bool) -> bool:
        # Resolves this value through the source precedence: command-line
        # input first, then the file configuration, then the default.
        if request_value is not None:
            return request_value
        if file_value is not None:
            return file_value
        return default_value

    def _resolve_path(self, request_value: Path | None, file_value: Path | None, default_value: Path) -> Path:
        # Resolves this value through the source precedence: command-line
        # input first, then the file configuration, then the default.
        if request_value is not None:
            return request_value
        if file_value is not None:
            return file_value
        return default_value

    def _resolve_optional_path(self, request_value: Path | None, file_value: Path | None) -> Path | None:
        # Resolves this value through the source precedence: command-line
        # input first, then the file configuration, then the default.
        if request_value is not None:
            return request_value
        return file_value

    def _resolve_hardware_name(
        self,
        request_value: HardwareName | None,
        file_value: HardwareName | None
    ) -> HardwareName:
        # Resolves this value through the source precedence: command-line
        # input first, then the file configuration, then the default.
        if request_value is not None:
            return request_value
        if file_value is not None:
            return file_value
        return "b200"

    def _resolve_precision_name(
        self,
        request_value: PrecisionName | None,
        file_value: PrecisionName | None
    ) -> PrecisionName:
        # Resolves this value through the source precedence: command-line
        # input first, then the file configuration, then the default.
        if request_value is not None:
            return request_value
        if file_value is not None:
            return file_value
        return "fp32"


class VocodeCli:
    # The command-line entry point: parses one atomic command, resolves
    # its configuration, and executes the matching lane runner inside the
    # run-tracking lifecycle. One process runs exactly one
    # (architecture, seed, stage) cell, so a crashed cell can never take
    # sibling cells with it; running several cells is a matter of
    # launching several commands.
    #
    # Example::
    #
    #     # Evaluate a published checkpoint on the held-out test split:
    #     python -m vocode.cli test-published \
    #         --model hifigan_v1 \
    #         --seed 0 \
    #         --run-id hifigan_v1_seed0_test \
    #         --dataset-root data/LJSpeech-1.1
    #
    #     # Measure an optimized variant. This lane additionally requires
    #     # the variant name, the hypothesis identifier, the commit, the
    #     # base checkpoint, and an explicit accelerator agreeing with the
    #     # declared hardware lane:
    #     python -m vocode.cli evaluate-optimized-variant \
    #         --model vocos \
    #         --seed 42 \
    #         --run-id vocos_int8_dynamic_seed42_test \
    #         --dataset-root data/LJSpeech-1.1 \
    #         --optimization-variant int8_dynamic \
    #         --hypothesis-id h_int8_dynamic \
    #         --code-commit-hash 0f1e2d3c4b5a \
    #         --project-checkpoint-path checkpoints/best.ckpt \
    #         --hardware-name cpu \
    #         --accelerator cpu
    def __init__(self) -> None:
        # Builds the argument parser and the configuration factory.
        self._argument_parser: argparse.ArgumentParser = self._build_argument_parser()
        self._configuration_factory: VocodeCliConfigurationFactory = VocodeCliConfigurationFactory()

    def run(self, command_line_arguments: list[str] | None = None) -> None:
        # Executes one command end to end: parse, resolve, then run the
        # lane runner inside RunTracker so the capsule records its own
        # lifecycle; validation errors become argparse usage errors.
        #
        # Args:
        #     command_line_arguments: Argument vector to execute; ``None``
        #         reads the process arguments through argparse itself.
        #         Default: ``None``.
        #
        # Raises:
        #     SystemExit: Raised through argparse for every malformed
        #         invocation. Both parse failures and the configuration
        #         and validation errors caught here are converted into
        #         usage errors, so misuse produces usage guidance naming
        #         the offending setting rather than a traceback.
        #
        # Note:
        #     Only ValidationError and ValueError are converted. A failure
        #     raised inside the lane runner propagates untouched, so a run
        #     that broke while executing is never disguised as a usage
        #     mistake.
        try:
            parsed_arguments: argparse.Namespace = self._argument_parser.parse_args(command_line_arguments)
            request: VocodeCliRequest = VocodeCliRequest.from_namespace(parsed_arguments)
            configuration: ExperimentConfiguration = self._configuration_factory.build(request)
            log.info(
                f"Starting VOCODE command={request.command_name} "
                f"architecture={configuration.architecture_name} seed={configuration.seed} "
                f"run_id={configuration.run_id}"
            )
            with RunTracker(configuration) as run_tracker:
                completed_rows: int = self._execute_lane_runner(configuration)
                run_tracker.record_completed_rows(completed_rows)
            log.info("VOCODE command completed")
        except (ValidationError, ValueError) as caught:
            self._argument_parser.error(str(caught))

    def _execute_lane_runner(self, configuration: ExperimentConfiguration) -> int:
        # Dispatches the evidence-category lane runner and returns its written row count.
        # The evidence category alone selects the runner, except in the optimized-variants
        # lane, where the stage separates recovery training from variant evaluation. The
        # dispatch is exhaustive over the category vocabulary and asserts that exhaustiveness
        # statically, so adding a category without a runner fails type checking rather than
        # falling through at run time.
        #
        # Returns:
        #     The number of summary rows the lane runner wrote, which the
        #     run tracker records as the run's produced evidence.
        match configuration.evidence_category:
            case "project_trained_reproduction" | "project_hybrid_variants":
                return ReproductionTrainingRunner(configuration).run()
            case "project_optimized_variants":
                if configuration.stage == "train":
                    return OptimizationRecoveryRunner(configuration).run()
                return OptimizedVariantEvaluator(configuration).run()
            case "published_checkpoint_evaluation":
                return PublishedWeightsEvaluator(configuration).run()
            case unreachable:
                assert_never(unreachable)

    def _build_argument_parser(self) -> argparse.ArgumentParser:
        # Builds the argument parser with one subcommand per evidence lane
        # and stage, each carrying the shared argument set.
        parser: argparse.ArgumentParser = argparse.ArgumentParser(
            prog="vocode",
            description="VOCODE atomic reproduction and author-weight evaluation CLI"
        )
        subparsers: argparse._SubParsersAction[argparse.ArgumentParser] = parser.add_subparsers(dest="command_name", required=True)
        self._add_atomic_command(
            subparsers,
            command_name="train-reproduction",
            evidence_category="project_trained_reproduction",
            stage="train",
            default_split="train"
        )
        self._add_atomic_command(
            subparsers,
            command_name="validation-reproduction",
            evidence_category="project_trained_reproduction",
            stage="validation",
            default_split="validation"
        )
        self._add_atomic_command(
            subparsers,
            command_name="test-reproduction",
            evidence_category="project_trained_reproduction",
            stage="test",
            default_split="test"
        )
        self._add_atomic_command(
            subparsers,
            command_name="train-hybrid",
            evidence_category="project_hybrid_variants",
            stage="train",
            default_split="train"
        )
        self._add_atomic_command(
            subparsers,
            command_name="test-hybrid",
            evidence_category="project_hybrid_variants",
            stage="test",
            default_split="test"
        )
        self._add_atomic_command(
            subparsers,
            command_name="test-published",
            evidence_category="published_checkpoint_evaluation",
            stage="test",
            default_split="test"
        )
        self._add_atomic_command(
            subparsers,
            command_name="evaluate-optimized-variant",
            evidence_category="project_optimized_variants",
            stage="test",
            default_split="test"
        )
        self._add_atomic_command(
            subparsers,
            command_name="recover-optimized-variant",
            evidence_category="project_optimized_variants",
            stage="train",
            default_split="train"
        )
        return parser

    def _add_atomic_command(
        self,
        subparsers: argparse._SubParsersAction,
        command_name: VocodeCliCommandName,
        evidence_category: EvidenceCategory,
        stage: ExperimentStage,
        default_split: DatasetSplitName
    ) -> None:
        # Registers one atomic subcommand bound to its evidence category,
        # stage, and default split.
        parser: argparse.ArgumentParser = subparsers.add_parser(command_name)
        parser.set_defaults(
            evidence_category=evidence_category,
            stage=stage,
            dataset_split_name=default_split
        )
        self._add_common_arguments(parser)

    def _add_common_arguments(self, parser: argparse.ArgumentParser) -> None:
        # Adds the argument set shared by every atomic command. Only three flags are required,
        # because they are the identity of the executed cell; every other flag defaults to
        # absent so the resolution layer can fall back to the file configuration and then to
        # the declared default. The flags below name the configuration field each one resolves
        # into, so the command line and the persisted resolved configuration can be read
        # against each other.
        #
        # Args:
        #     --config: Path of the optional YAML file, the lowest-priority
        #         configuration source.
        #     --override: Repeatable ``key=value`` pair applied onto the
        #         file configuration before it is validated. It addresses
        #         file settings only, so an unknown key fails validation.
        #     --model: Architecture under measurement, restricted to the
        #         registry vocabulary. It resolves into architecture_name
        #         and additionally selects the architecture-specific data
        #         defaults. Required.
        #     --seed: Run seed; resolves into seed and into the data
        #         configuration's shuffle seed. Required.
        #     --run-id: Run identifier and capsule leaf directory. Required.
        #     --split: Dataset partition to evaluate. It is suppressed when
        #         absent so the subcommand's own default survives, and it
        #         must agree with the subcommand's stage.
        #     --optimization-variant: Deployment transformation named by
        #         the optimization registry. It becomes a directory segment
        #         only in the optimized-variants lane.
        #     --hypothesis-id: Optimization hypothesis identifier, required
        #         by the optimized-variants lane.
        #     --code-commit-hash: Commit the run executes from, required by
        #         the optimized-variants lane.
        #     --experiment-name, --hypothesis, --interpretation-notes:
        #         Narrative fields carried into the summary row.
        #     --artifact-root: Root of the artifact tree, resolving into
        #         the artifact layout.
        #     --dataset-root: Corpus location. It has no default because it
        #         is machine specific, so resolution fails when neither
        #         this flag nor the file supplies it.
        #     --published-weights-root: Local root for published author
        #         checkpoints.
        #     --project-checkpoint-path: Retained Project Checkpoint to
        #         load; required by the reproduction evaluation commands.
        #     --hiftnet-f0-checkpoint-path: Separate F0-predictor
        #         checkpoint used by the HiFTNet architecture.
        #     --hardware-name, --precision-name: Closed lane labels
        #         resolving into the artifact layout and therefore into the
        #         capsule directory.
        #     --accelerator: Device selector handed to the trainer.
        #     --train-epoch-count: Epoch ceiling of the training stages.
        #     --training-batch-size, --validation-batch-size,
        #         --test-batch-size, --prediction-batch-size: Per-stage
        #         batch sizes of the data configuration.
        #     --num-workers, --persistent-workers, --prefetch-factor,
        #         --pin-memory: Dataloader worker policy. The last three
        #         take effect only when workers are requested.
        #     --test-split-size, --validation-split-size: Sizes of the two
        #         holdout blocks of the ordered-identifier partition.
        #     --max-test-utterances: Cap applied to the evaluation
        #         partition after the split, used by bounded verification
        #         runs.
        #     --training-segment-size: Training crop length, overriding the
        #         architecture default.
        #     --peak-normalization-enabled, --peak-normalization-value:
        #         Load-time peak normalization policy and its target.
        #     --rtf-warmup-iterations, --rtf-repetition-count: Timing
        #         protocol inputs of the warm real-time factor measurement,
        #         resolving into the excluded warm-up count and the
        #         synchronized repetitions averaged per timed utterance.
        #     --metric: Repeatable metric name validated against the metric
        #         registry. Supplying any metric replaces the default panel
        #         rather than extending it.
        #     --limit-train-batches, --limit-val-batches,
        #         --limit-test-batches, --limit-predict-batches: Execution
        #         limits accepting an integer batch count or a fraction in
        #         (0.0, 1.0]. They restrict execution only and never change
        #         split membership.
        #     --runtime-profiling-enabled, --runtime-profile-interval-steps:
        #         Runtime profiler switch and sampling interval.
        #
        # Note:
        #     Two data settings are deliberately unreachable from this
        #     surface: the resample rate and the training random peak gain
        #     range are fixed by the architecture's published recipe and
        #     are accepted from neither the command line nor the file, so a
        #     reference recipe's sample rate cannot be contradicted per
        #     run. The partition strategy has no flag either and is
        #     reachable through the file configuration alone.
        parser.add_argument("--config", dest="config_path", type=Path, default=None)
        parser.add_argument("--override", dest="override_values", action="append", default=[])
        parser.add_argument("--model", dest="architecture_name", choices=VocodeCliArchitectureSet.names, required=True)
        parser.add_argument(
            "--optimization-variant",
            dest="optimization_variant_name",
            choices=get_args(OptimizationVariantName.__value__),
            default=None
        )
        parser.add_argument(
            "--hypothesis-id",
            dest="optimization_hypothesis_id",
            type=str,
            default=None
        )
        parser.add_argument("--code-commit-hash", type=str, default=None)
        parser.add_argument("--seed", type=int, required=True)
        parser.add_argument("--run-id", type=str, required=True)
        parser.add_argument(
            "--split",
            dest="dataset_split_name",
            choices=("train", "validation", "test"),
            default=argparse.SUPPRESS
        )
        parser.add_argument("--experiment-name", type=str, default=None)
        parser.add_argument("--hypothesis", type=str, default=None)
        parser.add_argument("--interpretation-notes", type=str, default=None)
        parser.add_argument("--artifact-root", type=Path, default=None)
        parser.add_argument("--dataset-root", type=Path, default=None)
        parser.add_argument("--published-weights-root", type=Path, default=None)
        parser.add_argument("--project-checkpoint-path", type=Path, default=None)
        parser.add_argument("--hiftnet-f0-checkpoint-path", type=Path, default=None)
        parser.add_argument(
            "--hardware-name",
            choices=("b200", "h100", "a100_80gb", "l40s", "cuda", "mps", "m3_max", "cpu"),
            default=None
        )
        parser.add_argument("--precision-name", choices=("fp32", "fp16", "bf16"), default=None)
        parser.add_argument("--accelerator", type=str, default=None)
        parser.add_argument("--train-epoch-count", type=int, default=None)
        parser.add_argument("--training-batch-size", type=int, default=None)
        parser.add_argument("--validation-batch-size", type=int, default=None)
        parser.add_argument("--test-batch-size", type=int, default=None)
        parser.add_argument("--prediction-batch-size", type=int, default=None)
        parser.add_argument("--num-workers", type=int, default=None)
        parser.add_argument(
            "--persistent-workers",
            type=self._parse_boolean_text,
            default=None
        )
        parser.add_argument("--prefetch-factor", type=int, default=None)
        parser.add_argument(
            "--pin-memory",
            type=self._parse_boolean_text,
            default=None
        )
        parser.add_argument("--test-split-size", type=int, default=None)
        parser.add_argument("--validation-split-size", type=int, default=None)
        parser.add_argument("--max-test-utterances", type=int, default=None)
        parser.add_argument("--training-segment-size", type=int, default=None)
        parser.add_argument(
            "--peak-normalization-enabled",
            type=self._parse_boolean_text,
            default=None
        )
        parser.add_argument("--peak-normalization-value", type=float, default=None)
        parser.add_argument("--rtf-warmup-iterations", type=int, default=None)
        parser.add_argument("--rtf-repetition-count", type=int, default=None)
        parser.add_argument(
            "--metric",
            dest="metrics",
            action="append",
            choices=MetricRegistry.names,
            default=None
        )
        parser.add_argument("--limit-train-batches", type=self._parse_batch_limit, default=None)
        parser.add_argument("--limit-val-batches", type=self._parse_batch_limit, default=None)
        parser.add_argument("--limit-test-batches", type=self._parse_batch_limit, default=None)
        parser.add_argument("--limit-predict-batches", type=self._parse_batch_limit, default=None)
        parser.add_argument(
            "--runtime-profiling-enabled",
            type=self._parse_boolean_text,
            default=None
        )
        parser.add_argument("--runtime-profile-interval-steps", type=int, default=None)

    def _parse_boolean_text(self, value: str) -> bool:
        # Argument converter for the boolean flags, so they reach the namespace as typed bools
        # rather than as text. Only the two spellings are accepted, in any letter case;
        # numeric and colloquial forms are refused rather than guessed at, because silently
        # reading "0" as false would let a typo change a recorded run setting.
        #
        # Args:
        #     value: Raw flag text as argparse received it.
        #
        # Raises:
        #     argparse.ArgumentTypeError: If the text is neither spelling,
        #         which argparse turns into a usage error at parse time,
        #         before any configuration is resolved.
        normalized_value: str = value.lower()
        match normalized_value:
            case "true":
                return True
            case "false":
                return False
            case _:
                raise argparse.ArgumentTypeError(f"Expected true or false, got {value}")

    def _parse_batch_limit(self, value: str) -> int | float:
        # Parses a trainer batch limit while keeping it distinct from dataset split selection.
        # The text is typed through the YAML scalar rules, then checked against the limit
        # domain, so the distinction between an absolute count and a fraction of the loader
        # survives into the configuration instead of collapsing to one numeric type.
        #
        # Args:
        #     value: Raw flag text as argparse received it.
        #
        # Returns:
        #     An integer batch count of zero or more, or a float fraction
        #     in (0.0, 1.0]. Zero is admitted as a count, which is how an
        #     empty pass is expressed; the zero fraction is excluded so the
        #     two spellings cannot both mean the same thing.
        #
        # Raises:
        #     argparse.ArgumentTypeError: If the text is not numeric, is a
        #         boolean, is a negative count, or is a fraction outside
        #         the unit interval. A boolean is rejected explicitly
        #         because it is an integer subtype that would otherwise
        #         pass as a count of zero or one.
        parsed_value: object = yaml.safe_load(value)
        if isinstance(parsed_value, bool) or not isinstance(parsed_value, (int, float)):
            raise argparse.ArgumentTypeError(
                f"Expected integer >= 0 or float in (0.0, 1.0], got {value}"
            )
        if isinstance(parsed_value, int):
            if parsed_value < 0:
                raise argparse.ArgumentTypeError(f"Expected integer >= 0, got {value}")
            return parsed_value
        if parsed_value <= 0.0 or parsed_value > 1.0:
            raise argparse.ArgumentTypeError(f"Expected float in (0.0, 1.0], got {value}")
        return parsed_value


if __name__ == "__main__":
    VocodeCli().run(sys.argv[1:])
