# This module:
# 1. Defines the closed label domains of the experiment tree: evidence
#    category, stage, dataset split, hardware lane, and numeric precision
# 2. Defines ExperimentArtifactLayout, the frozen record that computes every
#    artifact path of a run capsule from its identity fields, so no runner or
#    logger ever composes a path by hand
#
# Design decisions:
# - The capsule tree is category/dataset/hardware-precision/architecture
#   [/variant]/runs/run-id; encoding identity in the path makes every
#   artifact attributable without opening it
# - The optimized-variants root encodes hardware only, because numeric
#   precision is the experimental variable there and lives in the variant
#   name, while every other category pins precision in the directory label
# - The optimized-variants category writes the versioned experiments_v2.csv
#   summary surface so historical first-schema summary files remain frozen
#   evidence
# - The variant path segment exists exactly for the optimized-variants
#   category; both its absence there and its presence elsewhere are
#   construction errors
#
# Author: Rahul Sawhney

from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from vocode.models.vocoder import ArchitectureName

__all__: list[str] = [
    "DatasetSplitName",
    "EvidenceCategory",
    "ExperimentArtifactLayout",
    "ExperimentStage",
    "HardwareName",
    "PrecisionName"
]

type EvidenceCategory = Literal[
    "published_checkpoint_evaluation",
    "project_trained_reproduction",
    "project_optimized_variants",
    "project_hybrid_variants"
]
type ExperimentStage = Literal["train", "validation", "test"]
type DatasetSplitName = Literal["train", "validation", "test"]
type HardwareName = Literal["b200", "h100", "a100_80gb", "l40s", "cuda", "mps", "m3_max", "cpu"]
type PrecisionName = Literal["fp32", "fp16", "bf16"]


class ExperimentArtifactLayout(BaseModel):
    # Frozen identity record of one run capsule and the single source of its
    # artifact paths. Every directory and file location below the artifact
    # root derives from these fields.
    #
    # Fields:
    #     artifact_root: Directory beneath which every evidence-category
    #         tree is written. All other paths are composed relative to it.
    #     evidence_category: Evidence lane this capsule belongs to. It
    #         selects the top-level directory, whether a variant segment is
    #         required, whether the directory label carries precision, and
    #         which summary-CSV schema the lane writes.
    #     architecture_name: Vocoder architecture under measurement; the
    #         directory segment immediately below the lane context.
    #     hardware_name: Hardware lane the run executes on. Datacenter
    #         NVIDIA lanes gain a vendor prefix in the directory label.
    #     precision_name: Numeric precision of the run. It is pinned in the
    #         directory label for every lane except the optimized-variants
    #         lane, where precision is the experimental variable and is
    #         carried by the variant name instead.
    #     seed: Run seed recorded for provenance. It deliberately takes no
    #         part in path composition, so two seeds of one run identifier
    #         address the same capsule directory.
    #     run_id: Run identifier; the leaf directory beneath ``runs`` and
    #         the only field that separates one capsule from its siblings.
    #     dataset_name: Corpus label pinned to the studied corpus.
    #         Default: ``"ljspeech"``.
    #     variant_name: Optimization-variant directory segment. Required
    #         exactly for the ``project_optimized_variants`` category and
    #         rejected for every other category. Default: ``None``.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True
    )
    artifact_root: Path
    evidence_category: EvidenceCategory
    architecture_name: ArchitectureName
    hardware_name: HardwareName
    precision_name: PrecisionName
    seed: int
    run_id: str
    dataset_name: Literal["ljspeech"] = "ljspeech"
    variant_name: str | None = None

    @model_validator(mode="after")
    def validate_variant_binding(self) -> Self:
        # The variant segment exists exactly for the optimized-variants evidence root.
        if self.evidence_category == "project_optimized_variants" and self.variant_name is None:
            raise ValueError("variant_name is required for the project_optimized_variants evidence category.")
        if self.evidence_category != "project_optimized_variants" and self.variant_name is not None:
            raise ValueError("variant_name is only valid for the project_optimized_variants evidence category.")
        return self

    def create_run_directories(self) -> None:
        # Creates the artifact directories required for one experiment run:
        # the run capsule itself, its checkpoint, log, and metric
        # subdirectories, and the lane-level summary directory. Creation is
        # idempotent, so re-running a command against an existing capsule
        # succeeds. It prepares directories only and fabricates no artifact
        # file. The seed-record directory is deliberately left out and is
        # created by the component that writes seed records, so an empty
        # seed directory never suggests a recorded seed that does not exist.
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.checkpoints_directory.mkdir(parents=True, exist_ok=True)
        self.logs_directory.mkdir(parents=True, exist_ok=True)
        self.metrics_directory.mkdir(parents=True, exist_ok=True)
        self.summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def evaluation_context(self) -> str:
        # Returns the canonical dataset-hardware-precision context label.
        return f"{self.dataset_name}_{self.hardware_precision_directory}"

    @property
    def hardware_precision_directory(self) -> str:
        # Returns the canonical hardware-precision directory label used by experiment artifacts.
        # The optimized-variants root encodes hardware only, because numeric precision is the
        # experimental variable there and lives in the variant name by the checkpoint statement.
        if self.evidence_category == "project_optimized_variants":
            if self.hardware_name == "cpu":
                return "modal_cpu8"
            if self.hardware_name in ("b200", "h100", "a100_80gb", "l40s"):
                return f"nvidia_{self.hardware_name}"
            return f"{self.hardware_name}"
        if self.hardware_name in ("b200", "h100", "a100_80gb", "l40s"):
            return f"nvidia_{self.hardware_name}_{self.precision_name}"
        return f"{self.hardware_name}_{self.precision_name}"

    @property
    def context_directory(self) -> Path:
        # Returns the evidence-category directory for the current evaluation context.
        return (
            self.artifact_root
            / self.evidence_category
            / self.dataset_name
            / self.hardware_precision_directory
        )

    @property
    def model_directory(self) -> Path:
        # Returns the architecture-specific artifact directory inside the evaluation context.
        # Optimized-variant capsules add the variant segment between architecture and runs.
        if self.evidence_category == "project_optimized_variants" and self.variant_name is not None:
            return self.context_directory / self.architecture_name / self.variant_name
        return self.context_directory / self.architecture_name

    @property
    def seed_directory(self) -> Path:
        # Returns the seed-record directory for this run.
        return self.run_directory / "seed_records"

    @property
    def run_directory(self) -> Path:
        # Returns the canonical directory for this experiment run.
        return self.model_directory / "runs" / self.run_id

    @property
    def checkpoints_directory(self) -> Path:
        # Returns the canonical checkpoint directory for this experiment run.
        return self.run_directory / "checkpoints"

    @property
    def logs_directory(self) -> Path:
        # Returns the directory used for textual run logs.
        return self.run_directory / "logs"

    @property
    def metrics_directory(self) -> Path:
        # Returns the directory used for metric artifacts.
        return self.run_directory / "metrics"

    @property
    def summary_csv_path(self) -> Path:
        # Returns the canonical experiment summary CSV path. The optimized-variants
        # category writes the versioned second-schema surface so historical summary
        # files keep their frozen first-schema evidence untouched.
        if self.evidence_category == "project_optimized_variants":
            return self.context_directory / "summary" / "experiments_v2.csv"
        return self.context_directory / "summary" / "experiments.csv"

    @property
    def resolved_configuration_path(self) -> Path:
        # Returns the path used to persist the resolved run configuration.
        return self.run_directory / "resolved_config.yaml"

    @property
    def hyperparameters_path(self) -> Path:
        # Returns the path used to persist run hyperparameters.
        return self.run_directory / "hyperparameters.yaml"

    @property
    def run_manifest_path(self) -> Path:
        # Returns the path used to persist the run manifest.
        return self.run_directory / "run_manifest.yaml"

    @property
    def metrics_path(self) -> Path:
        # Returns the path used to persist the metric snapshot.
        return self.metrics_directory / "metrics.json"

    @property
    def execution_log_path(self) -> Path:
        # Returns the canonical execution-log path for this run capsule.
        return self.logs_directory / "execution.log"
