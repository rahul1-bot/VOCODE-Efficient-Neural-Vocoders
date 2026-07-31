# This module:
# 1. Dispatches VOCODE training and evaluation capsules to Modal: Study 1
#    reproduction training, Study 2 recovery training and optimized-variant
#    evaluation, across the registered hardware lanes
# 2. Owns the launch path (parsing, validated frozen configuration,
#    detached spawn) and the container path (data preparation, import
#    configuration, atomic CLI invocation, artifact commit)
#
# Design decisions:
# - The container executes the same atomic VOCODE CLI a local run would,
#   so cloud execution adds transport and resources, never behavior
# - Hardware lanes apply at invocation time through resource profiles on
#   one shared worker class; the CPU profile carries no GPU key because
#   an option can be set but never unset
# - Long full-scale runs resume across Modal's per-attempt time cap
#   through the durable-checkpoint resolver, while smoke reruns are kept
#   isolated from automatic checkpoint reuse
# - The launching repository commit is recorded into the configuration so
#   container-side recipes carry code provenance
#
# Author: Rahul Sawhney

import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Literal, Self

from application import app, image, volume
from loguru import logger as log
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, field_validator, model_validator
from runtime import LJSpeechCorpusPreparer, ModalPythonPathConfigurator

import modal

__all__: list[str] = [
    "ModalCliRegistrar",
    "ModalHardwareProfile",
    "ModalHardwareProfileResolver",
    "ModalTrainingCliArgumentBuilder",
    "ModalTrainingCommandLineParser",
    "ModalTrainingConfig",
    "ModalTrainingEntrypoint",
    "ModalTrainingLaunchApplication",
    "ModalTrainingResumeCheckpointResolver",
    "ModalTrainingWorker",
    "app",
    "image",
    "volume"
]

type ModalArchitectureName = Literal[
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
]
type ModalDatasetSplitName = Literal["train", "validation", "test"]
type ModalExperimentStage = Literal["train", "validation", "test"]
type ModalBatchLimit = int | float | None
type ModalGpuName = Literal["b200", "h100", "a100_80gb", "l40s", "cpu"]
type ModalPrecisionName = Literal["fp32", "fp16", "bf16"]


class ModalTrainingConfig(BaseModel):
    # Configuration for a project-trained reproduction job on Modal.
    # It describes one model, seed, training stage, artifact root, and checkpoint path boundary.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    available_metrics: ClassVar[tuple[str, ...]] = (
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
    experiment_name: str
    run_id: str
    hypothesis: str
    interpretation_notes: str
    architecture_name: ModalArchitectureName
    seed: int
    artifact_root: str = "/data/experiment_artifacts"
    stage: ModalExperimentStage = "train"
    dataset_split_name: ModalDatasetSplitName
    hardware_name: ModalGpuName = "b200"
    precision_name: ModalPrecisionName = "fp32"
    dispatch_mode: Literal["spawned", "attached"] = "attached"
    evidence_category: Literal["project_trained_reproduction", "project_hybrid_variants"] = "project_trained_reproduction"
    optimization_variant: str | None = None
    optimization_hypothesis_id: str | None = None
    code_commit_hash: str | None = None
    project_checkpoint_path: str | None = None
    train_epoch_count: PositiveInt = 200
    hiftnet_f0_checkpoint_path: str | None = None
    max_test_utterances: PositiveInt | None = None
    training_batch_size: PositiveInt = 16
    training_segment_size: PositiveInt | None = None
    validation_batch_size: PositiveInt = 1
    test_batch_size: PositiveInt = 16
    prediction_batch_size: PositiveInt = 1
    num_workers: NonNegativeInt = 4
    persistent_workers: bool = True
    prefetch_factor: PositiveInt = 4
    pin_memory: bool = True
    rtf_warmup_iterations: NonNegativeInt = 5
    rtf_repetition_count: PositiveInt = 3
    limit_train_batches: ModalBatchLimit = None
    limit_val_batches: ModalBatchLimit = None
    limit_test_batches: ModalBatchLimit = None
    limit_predict_batches: ModalBatchLimit = None
    runtime_profiling_enabled: bool = True
    runtime_profile_interval_steps: PositiveInt = 50
    metrics: tuple[str, ...] = ("pesq", "stoi", "mel", "rtf", "parameters", "size")

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, metrics: tuple[str, ...]) -> tuple[str, ...]:
        if not metrics:
            raise ValueError("At least one metric is required")
        unknown_metrics: tuple[str, ...] = tuple(
            metric_name
            for metric_name in metrics
            if metric_name not in cls.available_metrics
        )
        if unknown_metrics:
            raise ValueError(f"Unknown metrics: {unknown_metrics}")
        if len(set(metrics)) != len(metrics):
            raise ValueError(f"Metrics must be unique, got {metrics}")
        return metrics

    @field_validator(
        "limit_train_batches",
        "limit_val_batches",
        "limit_test_batches",
        "limit_predict_batches"
    )
    @classmethod
    def validate_batch_limit(cls, value: ModalBatchLimit) -> ModalBatchLimit:
        # Keeps Modal preflight controls separate from dataset split identity.
        if value is None:
            return None
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
    def validate_stage_requirements(self) -> Self:
        # Validation and test stages need a project-generated checkpoint to load.
        if self.stage in ("validation", "test") and self.project_checkpoint_path is None:
            raise ValueError(
                "project_checkpoint_path is required for validation and test Modal reproduction stages."
            )
        if (
            self.stage == "train"
            and self.optimization_variant is not None
            and self.project_checkpoint_path is None
        ):
            raise ValueError(
                "project_checkpoint_path is required for optimized-variant recovery training."
            )
        if self.stage != self.dataset_split_name:
            raise ValueError(
                f"stage={self.stage!r} requires dataset_split_name={self.stage!r}, "
                f"got {self.dataset_split_name!r}."
            )
        if self.evidence_category == "project_hybrid_variants" and self.stage == "validation":
            raise ValueError(
                "project_hybrid_variants exposes train and test stages only."
            )
        if (
            self.stage == "test"
            and "rtf" in self.metrics
            and self.max_test_utterances is not None
            and self.max_test_utterances <= self.rtf_warmup_iterations
        ):
            raise ValueError(
                "max_test_utterances must exceed rtf_warmup_iterations when RTF is selected."
            )
        artifact_root: Path = Path(self.artifact_root)
        artifact_parts: tuple[str, ...] = artifact_root.parts
        if (
            not artifact_root.is_absolute()
            or ".." in artifact_parts
            or not artifact_root.is_relative_to("/data")
        ):
            raise ValueError(
                f"artifact_root must be an absolute path under /data, got {self.artifact_root!r}."
            )
        return self


class ModalHardwareProfile(BaseModel):
    # Immutable resource profile applied to the shared worker through Cls.with_options.
    # The CPU lane carries no GPU key at all because with_options can set but never unset one.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    gpu: str | None
    cpu: float
    memory: int

    def to_options(self) -> dict[str, object]:
        # Returns the with_options keyword mapping, omitting the GPU key on CPU lanes.
        if self.gpu is None:
            return {"cpu": self.cpu, "memory": self.memory}
        return {"gpu": self.gpu, "cpu": self.cpu, "memory": self.memory}


class ModalHardwareProfileResolver:
    # Maps the hardware lane onto the invocation-time resource profile for the shared worker.
    def resolve(self, hardware_name: ModalGpuName) -> ModalHardwareProfile:
        # Resolves the fleet-standard resource profile for one hardware lane.
        match hardware_name:
            case "b200":
                return ModalHardwareProfile(gpu="B200", cpu=16.0, memory=65536)
            case "h100":
                return ModalHardwareProfile(gpu="H100", cpu=16.0, memory=65536)
            case "a100_80gb":
                return ModalHardwareProfile(gpu="A100-80GB", cpu=16.0, memory=65536)
            case "l40s":
                return ModalHardwareProfile(gpu="L40S", cpu=8.0, memory=32768)
            case "cpu":
                return ModalHardwareProfile(gpu=None, cpu=8.0, memory=16384)


class ModalTrainingCliArgumentBuilder:
    # Argument builder for project-trained reproduction commands inside Modal.
    # It converts validated training configuration into an explicit atomic CLI invocation.
    def __init__(self, configuration: ModalTrainingConfig) -> None:
        # Binds the collaborators this component uses.
        self._configuration: ModalTrainingConfig = configuration

    def build(self) -> list[str]:
        # Builds the explicit command argument list for the in-container VOCODE CLI invocation.
        cli_arguments: list[str] = [
            self._build_command_name(),
            "--model",
            self._configuration.architecture_name,
            "--seed",
            str(self._configuration.seed),
            "--run-id",
            self._configuration.run_id,
            "--split",
            self._configuration.dataset_split_name,
            "--experiment-name",
            self._configuration.experiment_name,
            "--hypothesis",
            self._configuration.hypothesis,
            "--interpretation-notes",
            self._configuration.interpretation_notes,
            "--artifact-root",
            self._configuration.artifact_root,
            "--dataset-root",
            "/data/datasets/LJSpeech-1.1",
            "--published-weights-root",
            "/data/pretrained",
            "--hardware-name",
            self._configuration.hardware_name,
            "--precision-name",
            self._configuration.precision_name,
            "--accelerator",
            "cpu" if self._configuration.hardware_name == "cpu" else "cuda",
            "--train-epoch-count",
            str(self._configuration.train_epoch_count),
            "--training-batch-size",
            str(self._configuration.training_batch_size),
            "--validation-batch-size",
            str(self._configuration.validation_batch_size),
            "--test-batch-size",
            str(self._configuration.test_batch_size),
            "--prediction-batch-size",
            str(self._configuration.prediction_batch_size),
            "--num-workers",
            str(self._configuration.num_workers),
            "--persistent-workers",
            self._format_boolean(self._configuration.persistent_workers),
            "--prefetch-factor",
            str(self._configuration.prefetch_factor),
            "--pin-memory",
            self._format_boolean(self._configuration.pin_memory),
            "--rtf-warmup-iterations",
            str(self._configuration.rtf_warmup_iterations),
            "--rtf-repetition-count",
            str(self._configuration.rtf_repetition_count),
            "--runtime-profiling-enabled",
            self._format_boolean(self._configuration.runtime_profiling_enabled),
            "--runtime-profile-interval-steps",
            str(self._configuration.runtime_profile_interval_steps)
        ]
        if self._configuration.optimization_variant is not None:
            cli_arguments.extend([
                "--optimization-variant",
                self._configuration.optimization_variant
            ])
        if self._configuration.optimization_hypothesis_id is not None:
            cli_arguments.extend([
                "--hypothesis-id",
                self._configuration.optimization_hypothesis_id
            ])
        if self._configuration.code_commit_hash is not None:
            cli_arguments.extend([
                "--code-commit-hash",
                self._configuration.code_commit_hash
            ])
        if self._configuration.project_checkpoint_path is not None:
            cli_arguments.extend([
                "--project-checkpoint-path",
                self._configuration.project_checkpoint_path
            ])
        if self._configuration.hiftnet_f0_checkpoint_path is not None:
            cli_arguments.extend([
                "--hiftnet-f0-checkpoint-path",
                self._configuration.hiftnet_f0_checkpoint_path
            ])
        if self._configuration.max_test_utterances is not None:
            cli_arguments.extend([
                "--max-test-utterances",
                str(self._configuration.max_test_utterances)
            ])
        if self._configuration.training_segment_size is not None:
            cli_arguments.extend([
                "--training-segment-size",
                str(self._configuration.training_segment_size)
            ])
        self._extend_optional_batch_limit(
            cli_arguments,
            "--limit-train-batches",
            self._configuration.limit_train_batches
        )
        self._extend_optional_batch_limit(
            cli_arguments,
            "--limit-val-batches",
            self._configuration.limit_val_batches
        )
        self._extend_optional_batch_limit(
            cli_arguments,
            "--limit-test-batches",
            self._configuration.limit_test_batches
        )
        self._extend_optional_batch_limit(
            cli_arguments,
            "--limit-predict-batches",
            self._configuration.limit_predict_batches
        )
        metric_name: str
        for metric_name in self._configuration.metrics:
            cli_arguments.extend(["--metric", metric_name])
        return cli_arguments

    def _format_boolean(self, value: bool) -> str:
        # Formats boolean values as lowercase CLI text for in-container commands.
        if value:
            return "true"
        return "false"

    def _extend_optional_batch_limit(
        self,
        cli_arguments: list[str],
        flag_name: str,
        value: ModalBatchLimit
    ) -> None:
        # Adds an explicit trainer limit only when the Modal job requested one.
        if value is not None:
            cli_arguments.extend([flag_name, str(value)])

    def _build_command_name(self) -> str:
        # Maps the Modal stage selector onto the VOCODE atomic CLI command.
        if self._configuration.optimization_variant is not None:
            if self._configuration.stage == "train":
                return "recover-optimized-variant"
            return "evaluate-optimized-variant"
        if self._configuration.evidence_category == "project_hybrid_variants":
            match self._configuration.stage:
                case "train":
                    return "train-hybrid"
                case "validation":
                    raise ValueError("The hybrid evidence category exposes no standalone validation command.")
                case "test":
                    return "test-hybrid"
        match self._configuration.stage:
            case "train":
                return "train-reproduction"
            case "validation":
                return "validation-reproduction"
            case "test":
                return "test-reproduction"


class ModalTrainingResumeCheckpointResolver:
    # Resolver for continuing long-running full training jobs across Modal execution attempts.
    # Modal caps one function attempt at 24 hours, so long full-scale runs must be resumable.
    def resolve(self, configuration: ModalTrainingConfig) -> ModalTrainingConfig:
        # Adds an explicit checkpoint path when a full training run is being retried.
        if configuration.stage != "train":
            return configuration
        if configuration.project_checkpoint_path is not None:
            return configuration
        if not self._is_full_training_run(configuration):
            return configuration

        checkpoint_path: Path | None = self._resolve_checkpoint_path(configuration)
        if checkpoint_path is None:
            return configuration

        log.info(f"Resuming Modal training from project checkpoint: {checkpoint_path}")
        return configuration.model_copy(
            update={"project_checkpoint_path": str(checkpoint_path)}
        )

    def _is_full_training_run(self, configuration: ModalTrainingConfig) -> bool:
        # Keeps smoke-test reruns isolated from automatic full-run checkpoint reuse.
        return (
            configuration.limit_train_batches is None
            and configuration.limit_val_batches is None
            and configuration.limit_test_batches is None
            and configuration.limit_predict_batches is None
        )

    def _resolve_checkpoint_path(self, configuration: ModalTrainingConfig) -> Path | None:
        # Finds the durable training checkpoint for the configured run id.
        checkpoint_directory: Path = (
            Path(configuration.artifact_root)
            / configuration.evidence_category
            / "ljspeech"
            / self._hardware_precision_directory(configuration)
            / configuration.architecture_name
            / "runs"
            / configuration.run_id
            / "checkpoints"
        )
        last_checkpoint_path: Path = checkpoint_directory / "last.ckpt"
        if self._is_checkpoint_file(last_checkpoint_path):
            return last_checkpoint_path

        checkpoint_paths: list[Path] = [
            checkpoint_path
            for checkpoint_path in checkpoint_directory.glob("checkpoint-*.ckpt")
            if self._is_checkpoint_file(checkpoint_path)
        ]
        if not checkpoint_paths:
            return None
        return max(
            checkpoint_paths,
            key=lambda checkpoint_path: checkpoint_path.stat().st_mtime_ns
        )

    def _hardware_precision_directory(self, configuration: ModalTrainingConfig) -> str:
        # Mirrors the VOCODE artifact-layout hardware-precision directory rule for Modal GPU jobs.
        if configuration.hardware_name == "cpu":
            return f"cpu_{configuration.precision_name}"
        return f"nvidia_{configuration.hardware_name}_{configuration.precision_name}"

    def _is_checkpoint_file(self, checkpoint_path: Path) -> bool:
        # Validates that a candidate resume checkpoint exists and is non-empty.
        return (
            checkpoint_path.exists()
            and checkpoint_path.is_file()
            and checkpoint_path.stat().st_size > 0
        )


class ModalTrainingEntrypoint:
    # Modal entrypoint for project-trained reproduction jobs.
    # It prepares cloud data, configures imports, launches training or evaluation, and commits artifacts.
    def __init__(self) -> None:
        # Binds the collaborators this component uses.
        self._corpus_preparer: LJSpeechCorpusPreparer = LJSpeechCorpusPreparer()
        self._python_path_configurator: ModalPythonPathConfigurator = ModalPythonPathConfigurator()
        self._resume_checkpoint_resolver: ModalTrainingResumeCheckpointResolver = (
            ModalTrainingResumeCheckpointResolver()
        )

    def run_remote(self, configuration_payload: dict[str, object]) -> None:
        # Runs the configured VOCODE command inside the Modal container.
        self._python_path_configurator.configure_workspace_paths()
        os.chdir("/workspace")
        from vocode.cli import VocodeCli
        validated_training_config: ModalTrainingConfig = ModalTrainingConfig.model_validate(
            configuration_payload
        )
        self._corpus_preparer.ensure(Path("/data/datasets/LJSpeech-1.1"))
        resolved_training_config: ModalTrainingConfig = self._resume_checkpoint_resolver.resolve(
            validated_training_config
        )
        argument_builder: ModalTrainingCliArgumentBuilder = ModalTrainingCliArgumentBuilder(
            resolved_training_config
        )
        cli_arguments: list[str] = argument_builder.build()
        log.info(f"Dispatching atomic project-trained reproduction command: {cli_arguments}")
        VocodeCli().run(cli_arguments)
        volume.commit()

    def dispatch_remote(self, training_config: ModalTrainingConfig) -> None:
        # Dispatches the shared worker with the lane's resource profile applied at invocation.
        configuration_payload: dict[str, object] = training_config.model_dump()
        profile_resolver: ModalHardwareProfileResolver = ModalHardwareProfileResolver()
        profile: ModalHardwareProfile = profile_resolver.resolve(training_config.hardware_name)
        worker: ModalTrainingWorker = ModalTrainingWorker.with_options(**profile.to_options())()
        log.info(
            f"Dispatching train_{training_config.hardware_name} "
            f"architecture={training_config.architecture_name} "
            f"precision={training_config.precision_name} "
            f"seed={training_config.seed} run_id={training_config.run_id}"
        )
        if training_config.dispatch_mode == "spawned":
            function_call: modal.FunctionCall[None] = worker.execute.spawn(configuration_payload)
            log.info(
                f"Spawned train_{training_config.hardware_name} "
                f"function_call_id={function_call.object_id} "
                f"dashboard_url={function_call.get_dashboard_url()}"
            )
            return
        worker.execute.remote(configuration_payload)


class ModalTrainingCommandLineParser:
    # Converts raw modal-run arguments into the validated frozen training configuration.
    # Every categorical flag mirrors its Literal type through argparse choices, and the
    # frozen Pydantic model is the second validation wall behind the parser.
    def __init__(self) -> None:
        # Binds the collaborators this component uses.
        self._parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="launch_training")
        self._parser.add_argument(
            "--experiment-name",
            type=str,
            default="vocode_checkpoint01_training_reproduction"
        )
        self._parser.add_argument("--run-id", type=str, default=None)
        self._parser.add_argument(
            "--artifact-root",
            type=str,
            default="/data/experiment_artifacts"
        )
        self._parser.add_argument(
            "--hypothesis",
            type=str,
            default=(
                "Project-trained efficient vocoder reproduces the selected published architecture "
                "under the VOCODE LJSpeech training protocol."
            )
        )
        self._parser.add_argument(
            "--interpretation-notes",
            type=str,
            default="Checkpoint 01 project-trained reproduction run"
        )
        self._parser.add_argument(
            "--architecture-name",
            choices=(
                "hifigan_v1", "hifigan_v2", "hifigan_v3", "melgan", "vocos", "bigvgan",
                "apnet2", "freev", "hiftnet", "lpcnet", "rndvoc", "vocosformer", "rfwave"
            ),
            default="hifigan_v1"
        )
        self._parser.add_argument("--seed", type=int, default=42)
        self._parser.add_argument("--stage", choices=("train", "validation", "test"), default="train")
        self._parser.add_argument(
            "--dataset-split-name",
            choices=("train", "validation", "test"),
            default="train"
        )
        self._parser.add_argument(
            "--hardware-name",
            choices=("b200", "h100", "a100_80gb", "l40s", "cpu"),
            default="b200"
        )
        self._parser.add_argument("--precision-name", choices=("fp32", "fp16", "bf16"), default="fp32")
        self._parser.add_argument(
            "--optimization-variant",
            choices=(
                "baseline_cpu", "baseline_b200", "torch_compile", "torch_compile_overhead", "int8_dynamic",
                "int8_weight_only", "int4_weight_only", "fp16_weights", "onnx_fp32",
                "onnx_int8_static", "pruned_30", "pruned_50", "pruned_70",
                "pruned_50_recovered", "pruned_50_recovered_half", "dense_continued",
                "ode_steps_8", "ode_steps_4", "ode_steps_2"
            ),
            default=None
        )
        self._parser.add_argument("--project-checkpoint-path", type=str, default=None)
        self._parser.add_argument("--train-epoch-count", type=int, default=200)
        self._parser.add_argument("--hiftnet-f0-checkpoint-path", type=str, default=None)
        self._parser.add_argument("--max-test-utterances", type=int, default=None)
        self._parser.add_argument("--training-batch-size", type=int, default=16)
        self._parser.add_argument("--training-segment-size", type=int, default=None)
        self._parser.add_argument("--validation-batch-size", type=int, default=1)
        self._parser.add_argument("--test-batch-size", type=int, default=16)
        self._parser.add_argument("--prediction-batch-size", type=int, default=1)
        self._parser.add_argument("--num-workers", type=int, default=4)
        self._parser.add_argument(
            "--persistent-workers",
            action=argparse.BooleanOptionalAction,
            default=True
        )
        self._parser.add_argument("--prefetch-factor", type=int, default=4)
        self._parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
        self._parser.add_argument("--rtf-warmup-iterations", type=int, default=5)
        self._parser.add_argument("--rtf-repetition-count", type=int, default=3)
        self._parser.add_argument("--limit-train-batches", type=self._parse_batch_limit, default=None)
        self._parser.add_argument("--limit-val-batches", type=self._parse_batch_limit, default=None)
        self._parser.add_argument("--limit-test-batches", type=self._parse_batch_limit, default=None)
        self._parser.add_argument("--limit-predict-batches", type=self._parse_batch_limit, default=None)
        self._parser.add_argument(
            "--runtime-profiling-enabled",
            action=argparse.BooleanOptionalAction,
            default=True
        )
        self._parser.add_argument("--runtime-profile-interval-steps", type=int, default=50)
        self._parser.add_argument(
            "--metric",
            dest="metrics",
            action="append",
            choices=ModalTrainingConfig.available_metrics,
            default=None
        )
        self._parser.add_argument(
            "--spawn-remote",
            dest="dispatch_mode",
            action="store_const",
            const="spawned",
            default="attached"
        )
        self._parser.add_argument(
            "--evidence-category",
            choices=("project_trained_reproduction", "project_hybrid_variants"),
            default="project_trained_reproduction"
        )
        self._parser.add_argument("--hypothesis-id", dest="optimization_hypothesis_id", type=str, default=None)

    def _resolve_code_commit_hash(self) -> str | None:
        # Records the launching repository commit so container-side recipes carry code provenance.
        try:
            completed_hash: subprocess.CompletedProcess[str] = subprocess.run(
                ["git", "rev-parse", "--short=12", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False
            )
            completed_status: subprocess.CompletedProcess[str] = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal", "--", "code"],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed_hash.returncode != 0 or completed_status.returncode != 0:
            return None
        resolved_hash: str = completed_hash.stdout.strip()
        if not resolved_hash:
            return None
        if completed_status.stdout.strip():
            return f"{resolved_hash}-dirty"
        return resolved_hash

    def parse(self, arguments: list[str]) -> ModalTrainingConfig:
        # Parses raw launch arguments and returns the validated frozen configuration.
        namespace: argparse.Namespace = self._parser.parse_args(arguments)
        selected_metrics: tuple[str, ...] = (
            tuple(namespace.metrics)
            if namespace.metrics is not None
            else ("pesq", "stoi", "mel", "rtf", "parameters", "size")
        )
        return ModalTrainingConfig(
            experiment_name=namespace.experiment_name,
            run_id=self._resolve_run_id(namespace.run_id),
            hypothesis=namespace.hypothesis,
            interpretation_notes=namespace.interpretation_notes,
            architecture_name=namespace.architecture_name,
            seed=namespace.seed,
            artifact_root=namespace.artifact_root,
            stage=namespace.stage,
            dataset_split_name=namespace.dataset_split_name,
            hardware_name=namespace.hardware_name,
            precision_name=namespace.precision_name,
            dispatch_mode=namespace.dispatch_mode,
            evidence_category=namespace.evidence_category,
            optimization_variant=namespace.optimization_variant,
            optimization_hypothesis_id=namespace.optimization_hypothesis_id,
            code_commit_hash=self._resolve_code_commit_hash(),
            project_checkpoint_path=namespace.project_checkpoint_path,
            train_epoch_count=namespace.train_epoch_count,
            hiftnet_f0_checkpoint_path=namespace.hiftnet_f0_checkpoint_path,
            max_test_utterances=namespace.max_test_utterances,
            training_batch_size=namespace.training_batch_size,
            training_segment_size=namespace.training_segment_size,
            validation_batch_size=namespace.validation_batch_size,
            test_batch_size=namespace.test_batch_size,
            prediction_batch_size=namespace.prediction_batch_size,
            num_workers=namespace.num_workers,
            persistent_workers=namespace.persistent_workers,
            prefetch_factor=namespace.prefetch_factor,
            pin_memory=namespace.pin_memory,
            rtf_warmup_iterations=namespace.rtf_warmup_iterations,
            rtf_repetition_count=namespace.rtf_repetition_count,
            limit_train_batches=namespace.limit_train_batches,
            limit_val_batches=namespace.limit_val_batches,
            limit_test_batches=namespace.limit_test_batches,
            limit_predict_batches=namespace.limit_predict_batches,
            runtime_profiling_enabled=namespace.runtime_profiling_enabled,
            runtime_profile_interval_steps=namespace.runtime_profile_interval_steps,
            metrics=selected_metrics
        )

    def _resolve_run_id(self, run_id: str | None) -> str:
        # Resolves an explicit or generated run identifier for artifact naming.
        if run_id:
            return run_id
        return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")

    def _parse_batch_limit(self, value: str) -> int | float:
        # Parses explicit trainer limits from launch-argument text.
        if "." in value:
            parsed_float: float = float(value)
            if parsed_float <= 0.0 or parsed_float > 1.0:
                raise argparse.ArgumentTypeError(f"float batch limits must be in (0.0, 1.0], got {value}")
            return parsed_float
        parsed_int: int = int(value)
        if parsed_int < 0:
            raise argparse.ArgumentTypeError(f"integer batch limits must be >= 0, got {value}")
        return parsed_int


class ModalTrainingLaunchApplication:
    # Owns the local launch path: parse, validate, and dispatch to the shared worker.
    def __init__(self) -> None:
        # Binds the collaborators this component uses.
        self._parser: ModalTrainingCommandLineParser = ModalTrainingCommandLineParser()

    def launch_training(self, *arglist: str) -> None:
        # Launches one project-trained reproduction job from the local machine.
        configuration: ModalTrainingConfig = self._parser.parse(list(arglist))
        entrypoint: ModalTrainingEntrypoint = ModalTrainingEntrypoint()
        entrypoint.dispatch_remote(configuration)


class ModalCliRegistrar:
    # Registers class-owned local entrypoints on the Modal application at import time.
    def register(self, modal_application: modal.App) -> None:
        # Registers every launch surface this module exposes to modal run.
        application: ModalTrainingLaunchApplication = ModalTrainingLaunchApplication()
        modal_application.local_entrypoint(name="launch_training")(application.launch_training)


@app.cls(
    image=image,
    volumes={"/data": volume},
    cpu=8.0,
    memory=16384,
    timeout=86400,
    single_use_containers=True
)
class ModalTrainingWorker:
    # Shared cloud worker for every hardware lane; with_options selects the lane at dispatch.
    # The base declaration is CPU-only so GPU lanes are always explicit invocation decisions.
    @modal.method()
    def execute(self, configuration_payload: dict[str, object]) -> None:
        # Runs one project-trained reproduction command on the selected lane.
        entrypoint: ModalTrainingEntrypoint = ModalTrainingEntrypoint()
        entrypoint.run_remote(configuration_payload)


registrar: ModalCliRegistrar = ModalCliRegistrar()
registrar.register(app)
