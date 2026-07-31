# This module:
# 1. Defines ExperimentConfiguration, the single frozen record that fully
#    describes one experiment run: identity, hypothesis, stage, data, metric
#    selection, batch limits, device, and artifact layout
# 2. Validates the record at construction: batch-limit domains, the
#    stage-to-split binding, and the provenance and device requirements of
#    optimized-variant runs
# 3. Exposes the artifact paths of the run as properties delegating to the
#    artifact layout, so runners never compose paths themselves
#
# Design decisions:
# - The record is frozen, strict, and closed to extra fields, so experiment
#   settings cannot drift or accumulate unvalidated keys between runs
# - The stage must equal the dataset split name; evaluating one split under
#   another split's label would silently mislabel the evidence
# - Optimized-variant runs must declare their variant, base checkpoint,
#   hypothesis identifier, code commit, and an explicit accelerator that
#   matches the declared hardware lane, because those fields carry the
#   scientific identity of the capsule
#
# Author: Rahul Sawhney

from pathlib import Path
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, PositiveInt, field_validator, model_validator

from vocode.configs.layout import DatasetSplitName, EvidenceCategory, ExperimentArtifactLayout, ExperimentStage
from vocode.data.ljspeech_datamodule import LJSpeechDataConfig
from vocode.metrics.registry import MetricSelection
from vocode.metrics.rtf import RealTimeFactorConfig
from vocode.models.vocoder import ArchitectureName

__all__: list[str] = ["BatchLimit", "ExperimentConfiguration"]

type BatchLimit = int | float | None


class ExperimentConfiguration(BaseModel):
    # The complete validated description of one experiment run. Every runner
    # receives exactly one of these records and derives all behavior from it.
    #
    # Fields:
    #     experiment_name: Experiment label carried into the summary row.
    #     run_id: Run identifier; also the leaf capsule directory through
    #         the artifact layout.
    #     hypothesis: Claim the run is designed to test. It is logged when
    #         the run starts and written into the summary row, so the
    #         stated intent is recorded before any result exists.
    #     interpretation_notes: Analyst-facing note column carried into the
    #         summary row.
    #     evidence_category: Evidence lane of the run. It selects both the
    #         artifact tree and the lane runner the command line dispatches.
    #     stage: Stage executed by the run. Validation requires it to equal
    #         the dataset split name.
    #     dataset_split_name: Corpus partition the run measures, bound to
    #         the stage so evidence cannot carry another split's label.
    #     architecture_name: Architecture of the Project-Trained
    #         Configuration under measurement.
    #     seed: Run seed. Every runner seeds global randomness with it
    #         before building anything, and the command line propagates it
    #         into the data configuration, where it seeds the dataloader
    #         shuffle generator.
    #     artifact_layout: Identity record that computes every artifact
    #         path of this run; the path properties below delegate to it.
    #     published_weights_root: Local root under which published author
    #         checkpoints are resolved and, when absent, downloaded.
    #     project_checkpoint_path: Retained Project Checkpoint to load.
    #         This is the state every measurement of the configuration
    #         inherits and the state each deployment transformation is
    #         applied to. Required by the reproduction evaluation
    #         commands and by optimized-variant runs. Default: ``None``.
    #     hiftnet_f0_checkpoint_path: Separate F0-predictor checkpoint the
    #         HiFTNet architecture requires in addition to its generator
    #         weights. Default: ``None``.
    #     optimization_variant_name: Deployment transformation under
    #         measurement, whose paired same-lane control is the untransformed
    #         run of the same configuration. Mandatory for the
    #         optimized-variants lane. Default: ``None``.
    #     optimization_hypothesis_id: Identifier binding this capsule to the
    #         optimization hypothesis it tests; mandatory for the
    #         optimized-variants lane. Default: ``None``.
    #     code_commit_hash: Commit the run executed from; mandatory for the
    #         optimized-variants lane. Default: ``None``.
    #     data_configuration: Frozen data-pipeline record supplying the
    #         corpus root, split sizes, loader policy, and preprocessing.
    #     metric_selection: Registry-validated metric panel computed by the
    #         run. Default: the lightweight quality and complexity panel.
    #     real_time_factor_configuration: Timing protocol of the warm
    #         real-time factor measurement: the warm-up batches excluded
    #         from timing and the synchronized repetitions averaged per
    #         timed utterance. Real-time factors are comparable only
    #         within one hardware lane, never across lanes.
    #     train_epoch_count: Maximum epochs for the training stages; passed
    #         to the trainer as its epoch ceiling. Default: ``1``.
    #     limit_train_batches: Execution limit for the training loader. An
    #         integer is an absolute batch count, a float is a fraction of
    #         the loader in (0.0, 1.0], and ``None`` is unlimited. Limits
    #         restrict execution only and never alter split membership.
    #         Default: ``None``.
    #     limit_val_batches: Execution limit for the validation loader,
    #         with the same domain. Default: ``None``.
    #     limit_test_batches: Execution limit for the test loader, with the
    #         same domain. Default: ``None``.
    #     limit_predict_batches: Execution limit for the prediction loader,
    #         with the same domain. Default: ``None``.
    #     accelerator: Device selector handed to the trainer. The
    #         optimized-variants lane forbids ``"auto"`` because a resolved
    #         device could contradict the declared hardware lane.
    #         Default: ``"auto"``.
    #     runtime_profiling_enabled: Whether the runtime profiler callback
    #         is attached to the run. Default: ``True``.
    #     runtime_profile_interval_steps: Step interval between runtime
    #         profile samples when profiling is enabled. Default: ``50``.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True
    )
    experiment_name: str
    run_id: str
    hypothesis: str
    interpretation_notes: str
    evidence_category: EvidenceCategory
    stage: ExperimentStage
    dataset_split_name: DatasetSplitName
    architecture_name: ArchitectureName
    seed: int
    artifact_layout: ExperimentArtifactLayout
    published_weights_root: Path
    project_checkpoint_path: Path | None = None
    hiftnet_f0_checkpoint_path: Path | None = None
    optimization_variant_name: str | None = None
    optimization_hypothesis_id: str | None = None
    code_commit_hash: str | None = None
    data_configuration: LJSpeechDataConfig
    metric_selection: MetricSelection = MetricSelection()
    real_time_factor_configuration: RealTimeFactorConfig
    train_epoch_count: PositiveInt = 1
    limit_train_batches: BatchLimit = None
    limit_val_batches: BatchLimit = None
    limit_test_batches: BatchLimit = None
    limit_predict_batches: BatchLimit = None
    accelerator: str = "auto"
    runtime_profiling_enabled: bool = True
    runtime_profile_interval_steps: PositiveInt = 50

    @field_validator(
        "limit_train_batches",
        "limit_val_batches",
        "limit_test_batches",
        "limit_predict_batches"
    )
    @classmethod
    def validate_batch_limit(cls, value: BatchLimit) -> BatchLimit:
        # Enforces the batch-limit domain: an integer is an absolute batch
        # count, a float is a fraction of the loader in (0.0, 1.0], and bool
        # is rejected explicitly because it is an int subtype that would
        # otherwise pass as a count of zero or one.
        if value is None:
            return value
        if isinstance(value, bool):
            raise ValueError("batch limits must be int, float, or None; bool is invalid")
        if isinstance(value, int):
            if value < 0:
                raise ValueError(f"integer batch limits must be >= 0, got {value}")
            return value
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"float batch limits must be in (0.0, 1.0], got {value}")
        return value

    @model_validator(mode="after")
    def validate_stage_split_binding(self) -> Self:
        # Requires the stage and dataset split to agree, so a test row can
        # never be produced from validation data under a test label.
        if self.stage != self.dataset_split_name:
            raise ValueError(
                f"stage={self.stage!r} requires dataset_split_name={self.stage!r}, "
                f"got {self.dataset_split_name!r}."
            )
        return self

    @model_validator(mode="after")
    def validate_optimized_variant_binding(self) -> Self:
        # Optimized-variant capsules must bind their scientific identity and their
        # execution device before anything is built, so a run cannot start with a
        # generic experiment label, a null checkpoint, or a device that contradicts
        # the declared hardware lane.
        if self.evidence_category != "project_optimized_variants":
            return self
        missing_fields: list[str] = [
            field_name
            for field_name, field_value in (
                ("optimization_variant_name", self.optimization_variant_name),
                ("project_checkpoint_path", self.project_checkpoint_path),
                ("optimization_hypothesis_id", self.optimization_hypothesis_id),
                ("code_commit_hash", self.code_commit_hash)
            )
            if field_value is None
        ]
        if missing_fields:
            raise ValueError(
                f"project_optimized_variants runs require explicit provenance; "
                f"missing fields: {missing_fields}."
            )
        if self.accelerator == "auto":
            raise ValueError(
                "project_optimized_variants runs must declare an explicit accelerator; "
                "'auto' would allow the resolved device to contradict the hardware lane."
            )
        hardware_name: str = self.artifact_layout.hardware_name
        if hardware_name == "cpu" and self.accelerator != "cpu":
            raise ValueError(
                f"hardware_name='cpu' requires accelerator='cpu', got {self.accelerator!r}."
            )
        if hardware_name in ("b200", "h100", "a100_80gb", "l40s") and self.accelerator not in ("cuda", "gpu"):
            raise ValueError(
                f"hardware_name={hardware_name!r} requires accelerator='cuda', "
                f"got {self.accelerator!r}."
            )
        if hardware_name in ("m3_max", "mps") and self.accelerator not in ("mps", "cpu"):
            raise ValueError(
                f"hardware_name={hardware_name!r} requires accelerator 'mps' or 'cpu', "
                f"got {self.accelerator!r}."
            )
        return self

    @property
    def artifact_root(self) -> Path:
        # Returns the root directory for all lightweight experiment evidence.
        return self.artifact_layout.artifact_root

    @property
    def run_directory(self) -> Path:
        # Returns the canonical directory for this experiment run.
        return self.artifact_layout.run_directory

    @property
    def checkpoints_directory(self) -> Path:
        # Returns the canonical checkpoint directory for this experiment run.
        return self.artifact_layout.checkpoints_directory

    @property
    def summary_csv_path(self) -> Path:
        # Returns the canonical experiment summary CSV path.
        return self.artifact_layout.summary_csv_path

    @property
    def resolved_configuration_path(self) -> Path:
        # Returns the path used to persist the resolved run configuration.
        return self.artifact_layout.resolved_configuration_path

    @property
    def hyperparameters_path(self) -> Path:
        # Returns the path used to persist run hyperparameters.
        return self.artifact_layout.hyperparameters_path

    @property
    def run_manifest_path(self) -> Path:
        # Returns the path used to persist the run manifest.
        return self.artifact_layout.run_manifest_path

    @property
    def metrics_path(self) -> Path:
        # Returns the path used to persist the metric snapshot.
        return self.artifact_layout.metrics_path
