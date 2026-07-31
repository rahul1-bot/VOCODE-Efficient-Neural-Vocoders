# This module:
# 1. Dispatches published-checkpoint evaluation capsules to the fixed B200
#    evaluator lane: local launch parsing, validated frozen configuration,
#    in-container CLI invocation, and artifact commit back to the volume
#
# Design decisions:
# - The container executes the same atomic VOCODE CLI a local run would,
#   so cloud evaluation adds transport, never behavior
# - Every categorical flag mirrors its Literal type through argparse
#   choices, with the frozen Pydantic model as the second validation wall
#
# Author: Rahul Sawhney

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Literal

from application import app, image, volume
from loguru import logger as log
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, field_validator
from runtime import LJSpeechCorpusPreparer, ModalPythonPathConfigurator

import modal

__all__: list[str] = [
    "ModalPublishedCheckpointCliArgumentBuilder",
    "ModalPublishedCheckpointCliRegistrar",
    "ModalPublishedCheckpointCommandLineParser",
    "ModalPublishedCheckpointConfig",
    "ModalPublishedCheckpointEntrypoint",
    "ModalPublishedCheckpointLaunchApplication",
    "ModalPublishedCheckpointWorker",
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
    "hiftnet"
]
type ModalDatasetSplitName = Literal["test"]


class ModalPublishedCheckpointConfig(BaseModel):
    # Configuration for a published-checkpoint evaluation job on Modal.
    # It describes the model, seed, split, artifact root, and cloud storage paths for one atomic run.
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
    dataset_split_name: ModalDatasetSplitName
    dispatch_mode: Literal["spawned", "attached"] = "attached"
    max_test_utterances: PositiveInt | None = None
    test_batch_size: PositiveInt = 16
    prediction_batch_size: PositiveInt = 1
    num_workers: NonNegativeInt = 4
    persistent_workers: bool = True
    prefetch_factor: PositiveInt = 4
    pin_memory: bool = True
    rtf_warmup_iterations: PositiveInt = 5
    rtf_repetition_count: PositiveInt = 1
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

    @field_validator("artifact_root")
    @classmethod
    def validate_artifact_root(cls, artifact_root: str) -> str:
        resolved_root: Path = Path(artifact_root)
        resolved_parts: tuple[str, ...] = resolved_root.parts
        if (
            not resolved_root.is_absolute()
            or ".." in resolved_parts
            or not resolved_root.is_relative_to("/data")
        ):
            raise ValueError(
                f"artifact_root must be an absolute path under /data, got {artifact_root!r}."
            )
        return artifact_root


class ModalPublishedCheckpointCliArgumentBuilder:
    # Argument builder for invoking the local VOCODE CLI inside a Modal container.
    # It serializes validated cloud configuration into explicit command-line flags.
    def __init__(self, configuration: ModalPublishedCheckpointConfig) -> None:
        # Binds the collaborators this component uses.
        self._configuration: ModalPublishedCheckpointConfig = configuration

    def build(self) -> list[str]:
        # Builds the explicit command argument list for the in-container VOCODE CLI invocation.
        cli_arguments: list[str] = [
            "test-published",
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
            "b200",
            "--precision-name",
            "fp32",
            "--accelerator",
            "cuda",
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
            str(self._configuration.rtf_repetition_count)
        ]
        if self._configuration.max_test_utterances is not None:
            cli_arguments.extend([
                "--max-test-utterances",
                str(self._configuration.max_test_utterances)
            ])
        metric_name: str
        for metric_name in self._configuration.metrics:
            cli_arguments.extend(["--metric", metric_name])
        return cli_arguments

    def _format_boolean(self, value: bool) -> str:
        # Formats boolean values as lowercase CLI text for in-container commands.
        if value:
            return "true"
        return "false"


class ModalPublishedCheckpointEntrypoint:
    # Modal entrypoint for published-checkpoint evaluation jobs.
    # It prepares data, configures imports, launches the evaluator, and commits volume artifacts.
    def __init__(self) -> None:
        # Binds the collaborators this component uses.
        self._corpus_preparer: LJSpeechCorpusPreparer = LJSpeechCorpusPreparer()
        self._python_path_configurator: ModalPythonPathConfigurator = ModalPythonPathConfigurator()

    def run_remote(self, configuration_payload: dict[str, object]) -> None:
        # Runs the configured VOCODE command inside the Modal container.
        self._python_path_configurator.configure_workspace_paths()
        os.chdir("/workspace")
        from vocode.cli import VocodeCli
        evaluation_config: ModalPublishedCheckpointConfig = ModalPublishedCheckpointConfig.model_validate(
            configuration_payload
        )
        self._corpus_preparer.ensure(Path("/data/datasets/LJSpeech-1.1"))
        argument_builder: ModalPublishedCheckpointCliArgumentBuilder = ModalPublishedCheckpointCliArgumentBuilder(
            evaluation_config
        )
        cli_arguments: list[str] = argument_builder.build()
        log.info(f"Dispatching atomic published-checkpoint evaluation command: {cli_arguments}")
        VocodeCli().run(cli_arguments)
        volume.commit()

    def dispatch_remote(self, evaluation_config: ModalPublishedCheckpointConfig) -> None:
        # Dispatches the shared worker with a serialized configuration payload.
        configuration_payload: dict[str, object] = evaluation_config.model_dump()
        worker: ModalPublishedCheckpointWorker = ModalPublishedCheckpointWorker()
        log.info(
            f"Dispatching evaluate_published_b200 architecture={evaluation_config.architecture_name} "
            f"seed={evaluation_config.seed} run_id={evaluation_config.run_id}"
        )
        if evaluation_config.dispatch_mode == "spawned":
            function_call: modal.FunctionCall[None] = worker.execute.spawn(configuration_payload)
            log.info(
                f"Spawned evaluate_published_b200 "
                f"function_call_id={function_call.object_id} "
                f"dashboard_url={function_call.get_dashboard_url()}"
            )
            return
        worker.execute.remote(configuration_payload)


class ModalPublishedCheckpointCommandLineParser:
    # Converts raw modal-run arguments into the validated frozen evaluation configuration.
    # Every categorical flag mirrors its Literal type through argparse choices, and the
    # frozen Pydantic model is the second validation wall behind the parser.
    def __init__(self) -> None:
        # Binds the collaborators this component uses.
        self._parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="launch_published_evaluation")
        self._parser.add_argument(
            "--experiment-name",
            type=str,
            default="vocode_checkpoint01_published_checkpoint_evaluation"
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
                "Published checkpoint evaluation reproduces the selected released-weight metric row "
                "under the VOCODE LJSpeech evaluator."
            )
        )
        self._parser.add_argument(
            "--interpretation-notes",
            type=str,
            default="Checkpoint 01 published-checkpoint evaluation run"
        )
        self._parser.add_argument(
            "--architecture-name",
            choices=(
                "hifigan_v1", "hifigan_v2", "hifigan_v3", "melgan", "vocos",
                "bigvgan", "apnet2", "freev", "hiftnet"
            ),
            default="hifigan_v1"
        )
        self._parser.add_argument("--seed", type=int, default=42)
        self._parser.add_argument("--dataset-split-name", choices=("test",), default="test")
        self._parser.add_argument("--max-test-utterances", type=int, default=None)
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
        self._parser.add_argument("--rtf-repetition-count", type=int, default=1)
        self._parser.add_argument(
            "--metric",
            dest="metrics",
            action="append",
            choices=ModalPublishedCheckpointConfig.available_metrics,
            default=None
        )
        self._parser.add_argument(
            "--spawn-remote",
            dest="dispatch_mode",
            action="store_const",
            const="spawned",
            default="attached"
        )

    def parse(self, arguments: list[str]) -> ModalPublishedCheckpointConfig:
        # Parses raw launch arguments and returns the validated frozen configuration.
        namespace: argparse.Namespace = self._parser.parse_args(arguments)
        selected_metrics: tuple[str, ...] = (
            tuple(namespace.metrics)
            if namespace.metrics is not None
            else ("pesq", "stoi", "mel", "rtf", "parameters", "size")
        )
        return ModalPublishedCheckpointConfig(
            experiment_name=namespace.experiment_name,
            run_id=self._resolve_run_id(namespace.run_id),
            hypothesis=namespace.hypothesis,
            interpretation_notes=namespace.interpretation_notes,
            architecture_name=namespace.architecture_name,
            seed=namespace.seed,
            artifact_root=namespace.artifact_root,
            dataset_split_name=namespace.dataset_split_name,
            dispatch_mode=namespace.dispatch_mode,
            max_test_utterances=namespace.max_test_utterances,
            test_batch_size=namespace.test_batch_size,
            prediction_batch_size=namespace.prediction_batch_size,
            num_workers=namespace.num_workers,
            persistent_workers=namespace.persistent_workers,
            prefetch_factor=namespace.prefetch_factor,
            pin_memory=namespace.pin_memory,
            rtf_warmup_iterations=namespace.rtf_warmup_iterations,
            rtf_repetition_count=namespace.rtf_repetition_count,
            metrics=selected_metrics
        )

    def _resolve_run_id(self, run_id: str | None) -> str:
        # Resolves an explicit or generated run identifier for artifact naming.
        if run_id:
            return run_id
        return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")


class ModalPublishedCheckpointLaunchApplication:
    # Owns the local launch paths for full and bounded published-checkpoint evaluations.
    def __init__(self) -> None:
        # Binds the collaborators this component uses.
        self._parser: ModalPublishedCheckpointCommandLineParser = ModalPublishedCheckpointCommandLineParser()

    def launch_published_evaluation(self, *arglist: str) -> None:
        # Launches one published-checkpoint evaluation job from the local machine.
        configuration: ModalPublishedCheckpointConfig = self._parser.parse(list(arglist))
        entrypoint: ModalPublishedCheckpointEntrypoint = ModalPublishedCheckpointEntrypoint()
        entrypoint.dispatch_remote(configuration)

    def verify(self, *arglist: str) -> None:
        # Runs a small published-checkpoint verification job for Modal integration checks.
        bounded_defaults: list[str] = [
            "--experiment-name",
            "vocode_checkpoint01_published_checkpoint_limited_test",
            "--hypothesis",
            "Published checkpoint evaluation reproduces a metric row on a bounded test-set preflight pass.",
            "--interpretation-notes",
            "Limited test-set published-checkpoint evaluation",
            "--max-test-utterances",
            "50"
        ]
        configuration: ModalPublishedCheckpointConfig = self._parser.parse(bounded_defaults + list(arglist))
        entrypoint: ModalPublishedCheckpointEntrypoint = ModalPublishedCheckpointEntrypoint()
        entrypoint.dispatch_remote(configuration)


class ModalPublishedCheckpointCliRegistrar:
    # Registers class-owned local entrypoints on the Modal application at import time.
    def register(self, modal_application: modal.App) -> None:
        # Registers every launch surface this module exposes to modal run.
        application: ModalPublishedCheckpointLaunchApplication = ModalPublishedCheckpointLaunchApplication()
        modal_application.local_entrypoint(name="launch_published_evaluation")(application.launch_published_evaluation)
        modal_application.local_entrypoint(name="verify")(application.verify)


@app.cls(
    image=image,
    volumes={"/data": volume},
    gpu="B200",
    cpu=16.0,
    memory=65536,
    timeout=7200,
    single_use_containers=True
)
class ModalPublishedCheckpointWorker:
    # Cloud worker for published-checkpoint evaluation on the fixed B200 evaluator lane.
    @modal.method()
    def execute(self, configuration_payload: dict[str, object]) -> None:
        # Runs the published-checkpoint evaluator on a Modal B200 worker.
        entrypoint: ModalPublishedCheckpointEntrypoint = ModalPublishedCheckpointEntrypoint()
        entrypoint.run_remote(configuration_payload)


registrar: ModalPublishedCheckpointCliRegistrar = ModalPublishedCheckpointCliRegistrar()
registrar.register(app)
