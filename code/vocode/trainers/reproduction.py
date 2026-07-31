# This module:
# 1. Runs one project-trained reproduction cell (architecture, seed, stage)
#    end to end: module construction through the model registry, harness
#    dispatch, and durable evidence writing. This is the Study 1 evidence
#    lane, which trains the twelve-model cohort from random initialization
#    and produces the baselines that Study 2 interventions and the Study 3
#    comparison consume
# 2. Encodes the per-family training policy that keeps each reference recipe
#    honest: callback selection, validation cadence, and checkpoint cadence
#    differ between the adversarial family and the autoregressive and flow
#    architectures
# 3. Writes the complete evidence set for the cell: checkpoint manifest,
#    experiment CSV rows, hyperparameter dump, and a metrics snapshot JSON
#
# Harness contract (syntheticmind):
# - Each stage receives a fresh syntheticmind Trainer; the harness owns the
#   loops, checkpoint resume with mid-epoch replay, precision handling, and
#   callback scheduling, while this runner owns what enters the Trainer
# - Step-cadence architectures configure validation through the harness
#   val_check_interval with global_step scope and suppress the epoch-end
#   validation pass; epoch-cadence architectures keep the harness defaults
# - Artifact precision labels map onto the harness precision domain: fp32 to
#   32-true, fp16 to 16-mixed, bf16 to bf16-mixed
# - MetricSequence and RealTimeFactorMonitor are vocode-side Callback
#   subclasses injected into the harness test and prediction loops; the
#   ExperimentLogger implements the harness logger interface, so every
#   metric the harness reduces lands in the experiment CSV evidence
# - Checkpoints are read through the harness load_checkpoint boundary and
#   restored from the model_state_dict entry written by ModelCheckpoint
#
# Design decisions:
# - The registry status gate refuses architectures that are not
#   training-level ready, so a cell can never silently train a partial
#   implementation
# - Seeding happens before module construction because parameter
#   initialization must be part of the seeded region for a reproduction claim
# - EarlyStopping and EMA are withheld from the adversarial, autoregressive,
#   and flow families because they are foreign to those reference recipes;
#   applying them would be a hidden recipe deviation
# - The metrics snapshot is written with allow_nan=False so a corrupted
#   metric fails the run loudly instead of entering the evidence
#
# Author: Rahul Sawhney

import json
from pathlib import Path
from typing import Literal

from loguru import logger as log

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.early_stopping import EarlyStopping
from syntheticmind.callbacks.ema import EMACallback
from syntheticmind.callbacks.model_checkpoint import ModelCheckpoint
from syntheticmind.callbacks.nan_inf_guard import NanInfGuard
from syntheticmind.callbacks.runtime_profiler import RuntimeProfiler
from syntheticmind.core.module import Module
from syntheticmind.core.trainer import Trainer
from syntheticmind.utilities.checkpoint import load_checkpoint
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.seed import SeedManager
from syntheticmind.utilities.types import CheckpointDict, CheckpointValue, HyperparameterDict, Precision

from vocode.configs.run import ExperimentConfiguration
from vocode.data.ljspeech_datamodule import LJSpeechDataModule
from vocode.loggers.checkpoint import CheckpointEvidence, CheckpointLogger, CheckpointManifest
from vocode.loggers.experiment import ExperimentLogger
from vocode.metrics.rtf import RealTimeFactorMonitor
from vocode.metrics.sequence import MetricSequence
from vocode.models.registry import (
    ArchitectureImplementationRecord,
    ArchitectureModuleSpec,
    ModelRegistry,
    ModuleBuildOptions,
)
from vocode.models.vocoder import ArchitectureName

__all__: list[str] = ["ReproductionTrainingRunner"]


class ReproductionTrainingRunner:
    # Runner for one project-trained reproduction cell. It resolves the
    # architecture through the model registry, dispatches the configured stage
    # to the harness, and writes the checkpoint, CSV, hyperparameter, and
    # metric evidence that the cell contributes to the study.
    #
    # In the report's vocabulary, one cell of this runner fits one
    # Project-Trained Configuration from random initialization under its
    # registered budget, and the terminal durable state the checkpoint gate
    # admits is that configuration's Retained Project Checkpoint, which every
    # later measurement in the study inherits.
    #
    # Integration: this runner owns what enters the harness; the harness owns
    # the loops. Its contract with the harness has four parts.
    #
    # Trainer construction: every stage receives its own freshly built Trainer
    # and none is reused across stages. Training builds one Trainer over the
    # configured epoch budget; validation, prediction, and test each build a
    # single-epoch Trainer. All four receive the run's accelerator, the
    # experiment logger, and the batch limit belonging to their own stage, so a
    # limit intended for one stage can never bound another.
    #
    # Callback policy: the policy is chosen by architecture family and never by
    # convenience. Step-cadence families receive the monitored checkpoint, the
    # optional runtime profiler, and the non-finite guard only; the epoch-cadence
    # family additionally receives early stopping on the validation loss and
    # exponential weight averaging. Early stopping and weight averaging are
    # withheld from the adversarial, autoregressive, and flow families because
    # they are foreign to those reference recipes, and adding them would be a
    # hidden recipe deviation rather than an improvement. The monitored
    # checkpoint always leads the callback list, and the evaluation stages
    # additionally place the timing monitor first in prediction and the metric
    # panel first in test, so each brackets the calls it measures.
    #
    # Precision mapping: the artifact layout's precision label and the harness
    # precision domain are separate vocabularies, and the runner maps between
    # them explicitly for the training Trainer; the evaluation Trainers take the
    # harness default.
    #
    # Checkpoint boundaries: the boundary is the monitored validation loss with
    # the last state also retained and an on-exception save, so an interrupted
    # run still leaves a loadable artifact. Step-cadence families additionally
    # checkpoint every five thousand optimizer steps, except under a truncated
    # epoch where a step cadence would never fire and the runner falls back to
    # the epoch boundary.
    #
    # Responsibility rule: one runner instance handles exactly one cell. It owns
    # its own model registry and its own row counter, and it holds no state that
    # another runner could observe, so several cells may be run in one process
    # without their evidence interfering.
    def __init__(self, configuration: ExperimentConfiguration) -> None:
        # Binds the validated experiment configuration, opens the model
        # registry, and zeroes the written-row counter. Construction performs no
        # validation of its own beyond what the configuration already carries,
        # creates no artifact directory, and writes no row.
        #
        # Args:
        #     configuration: The validated run description naming the cell's
        #         architecture, stage, seed, evidence category, artifact
        #         layout, and batch limits.
        self._configuration: ExperimentConfiguration = configuration
        self._registry: ModelRegistry = ModelRegistry()
        self._row_count: int = 0

    def run(self) -> int:
        # Executes the configured stage for the cell. The registry status gate
        # fires first, seeding precedes module construction so parameter
        # initialization is reproducible, and the stage dispatch is followed
        # by the hyperparameter dump, the CSV flush, and the metrics snapshot
        # in that order, so evidence exists only for stages that completed.
        # The test stage runs prediction before test, because the timing
        # measurement belongs to the synthesis pass and the objective metric
        # panel belongs to the test pass.
        #
        # Raises:
        #     MisconfigurationError: If the architecture is not training-level
        #         ready, if the stage is outside the three supported ones, or
        #         if an evaluation stage cites no loadable project checkpoint.
        #
        # Returns:
        #     The number of experiment result rows this runner has written,
        #     which is one after a completed stage.
        architecture_name: ArchitectureName = self._configuration.architecture_name
        record: ArchitectureImplementationRecord = self._registry.get(architecture_name)
        if record.status != "training_level_ready":
            raise MisconfigurationError(
                f"Architecture {architecture_name} is not training-level ready. "
                f"status={record.status}; blocker={record.blocker}"
            )
        SeedManager.seed_everything(self._configuration.seed)
        module_spec: ArchitectureModuleSpec = self._build_module_spec(architecture_name)
        datamodule: LJSpeechDataModule = LJSpeechDataModule(self._configuration.data_configuration)
        experiment_logger: ExperimentLogger = self._build_experiment_logger(module_spec)
        match self._configuration.stage:
            case "train":
                self._run_training(module_spec, datamodule, experiment_logger)
            case "validation":
                self._load_project_checkpoint(module_spec.module)
                self._dispatch_validation(module_spec.module, datamodule, experiment_logger)
            case "test":
                self._load_project_checkpoint(module_spec.module)
                self._dispatch_prediction(module_spec.module, datamodule, experiment_logger)
                self._dispatch_test(module_spec.module, datamodule, experiment_logger)
            case _:
                raise MisconfigurationError(f"Unsupported reproduction stage={self._configuration.stage}")
        experiment_logger.log_hyperparams(self._build_hyperparameter_dump(module_spec))
        experiment_logger.flush_to_experiments_csv()
        self._row_count: int = self._row_count + 1
        self._write_metrics_snapshot(experiment_logger)
        return self._row_count

    @property
    def row_count(self) -> int:
        # Returns the number of experiment result rows written by this runner.
        return self._row_count

    def _run_training(
        self,
        module_spec: ArchitectureModuleSpec,
        datamodule: LJSpeechDataModule,
        experiment_logger: ExperimentLogger
    ) -> None:
        # Assembles the training Trainer and runs fit. Step-cadence
        # architectures checkpoint every five thousand optimizer steps and
        # validate through the harness global-step interval with the epoch-end
        # pass suppressed; epoch-cadence architectures checkpoint on the
        # monitored validation loss each epoch. Bounded functional verification
        # runs carrying a train-batch limit fall back to epoch cadence because a
        # step cadence would never fire within the truncated epoch. After fit,
        # the checkpoint manifest records what ModelCheckpoint actually kept.
        #
        # Args:
        #     module_spec: The registry-built architecture, its variant name,
        #         and its configuration dump.
        #     datamodule: The corpus the fit and its validation draw from.
        #     experiment_logger: The logger every reduced metric reaches.
        #
        # Note:
        #     The configured project checkpoint enters fit as the resume path
        #     rather than as a weight load, so the harness restores optimizer,
        #     scheduler, and loop state alongside the weights and continues
        #     mid-epoch where it left off.
        use_step_cadence: bool = (
            self._uses_step_cadence_policy(module_spec.architecture_name)
            and self._configuration.limit_train_batches is None
        )
        model_checkpoint_step_interval: int | None = (
            5000 if use_step_cadence else None
        )
        model_checkpoint: ModelCheckpoint = ModelCheckpoint(
            dirpath=self._configuration.checkpoints_directory,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            every_n_train_steps=model_checkpoint_step_interval,
            save_on_exception=True
        )
        training_callbacks: list[Callback] = self._build_training_callbacks(
            module_spec,
            model_checkpoint
        )
        validation_step_interval: int | float = (
            self._resolve_validation_step_interval(module_spec.architecture_name)
            if use_step_cadence
            else 1.0
        )
        validation_step_scope: Literal["epoch", "global_step"] = (
            "global_step" if use_step_cadence else "epoch"
        )
        validate_at_epoch_end: bool = not use_step_cadence
        training_trainer: Trainer = Trainer(
            max_epochs=self._configuration.train_epoch_count,
            callbacks=training_callbacks,
            logger=experiment_logger,
            accelerator=self._configuration.accelerator,
            precision=self._resolve_trainer_precision(),
            val_check_interval=validation_step_interval,
            val_check_interval_scope=validation_step_scope,
            validate_at_epoch_end=validate_at_epoch_end,
            log_every_n_steps=5 if self._is_gan_vocoder(module_spec.architecture_name) else 10,
            limit_train_batches=self._configuration.limit_train_batches,
            limit_val_batches=self._configuration.limit_val_batches
        )
        training_trainer.fit(
            module_spec.module,
            datamodule,
            ckpt_path=self._configuration.project_checkpoint_path
        )
        project_checkpoint_manifest: CheckpointManifest = CheckpointLogger().write(
            model_checkpoint=model_checkpoint,
            run_directory=self._configuration.run_directory,
            evidence_label=self._resolve_evidence_label(),
            architecture_name=module_spec.architecture_name,
            variant_name=module_spec.variant_name,
            seed=self._configuration.seed,
            unique_id=self._build_unique_id(module_spec),
            experiment_name=self._configuration.experiment_name,
            run_id=self._configuration.run_id
        )
        log.info(
            f"Project-trained reproduction checkpoint written for "
            f"{module_spec.architecture_name} seed={self._configuration.seed}; "
            f"manifest={project_checkpoint_manifest.checkpoint_directory}"
        )

    def _build_training_callbacks(
        self,
        module_spec: ArchitectureModuleSpec,
        model_checkpoint: ModelCheckpoint
    ) -> list[Callback]:
        # Builds the callback policy for project-trained reproduction without hiding recipe deviations.
        # Both branches place the monitored checkpoint first and the non-finite
        # guard last, so the checkpoint observes every metric the loops publish
        # and the guard is the final word on whether a step's values may enter
        # the evidence.
        #
        # Args:
        #     module_spec: The cell's architecture, which selects the family
        #         policy.
        #     model_checkpoint: The monitored checkpoint callback already
        #         built with this run's cadence and boundaries.
        #
        # Returns:
        #     The ordered callback list the training Trainer is built with.
        if self._uses_step_cadence_policy(module_spec.architecture_name):
            return [
                model_checkpoint,
                *self._build_runtime_profiler_callbacks(),
                NanInfGuard()
            ]
        return [
            model_checkpoint,
            EarlyStopping(monitor="val_loss", mode="min", patience=10),
            EMACallback(decay=0.999),
            *self._build_runtime_profiler_callbacks(),
            NanInfGuard()
        ]

    def _is_gan_vocoder(self, architecture_name: str) -> bool:
        # Adversarial vocoders form the GAN family for the step-cadence policy and the
        # dense loss-component logging cadence, because adversarial losses are non-monotonic
        # and early stopping or weight averaging would distort long-horizon GAN dynamics.
        return architecture_name in (
            "hifigan_v1", "hifigan_v2", "hifigan_v3", 
            "melgan", "vocos", "bigvgan", "apnet2", 
            "freev", "rndvoc", "vocosformer"
        )

    def _uses_step_cadence_policy(self, architecture_name: str) -> bool:
        # LPCNet and RFWave join the adversarial vocoders on the lean step-cadence policy:
        # their short epochs would trigger hundreds of validation passes, EMA weight averaging
        # is foreign to both reference recipes (and would corrupt LPCNet's sparsification
        # masks), and both references train a fixed budget without early stopping.
        return self._is_gan_vocoder(architecture_name) or architecture_name in ("lpcnet", "rfwave")

    def _resolve_validation_step_interval(self, architecture_name: str) -> int:
        # LPCNet and RFWave validate on the checkpoint cadence (chunked full-utterance
        # validation and flow-composite validation respectively cost real wall clock);
        # adversarial vocoders keep the fleet's 1000-step cadence.
        return 5000 if architecture_name in ("lpcnet", "rfwave") else 1000

    def _resolve_trainer_precision(self) -> Precision:
        # Maps the artifact precision label onto the harness training-precision contract.
        # The two vocabularies are deliberately kept separate: the artifact label
        # names the capsule's precision lane and the harness literal names an
        # execution mode, so the mapping is stated once here rather than being
        # assumed equal anywhere.
        #
        # Returns:
        #     The harness precision literal the training Trainer is built
        #     with; the match is exhaustive over the closed label domain.
        match self._configuration.artifact_layout.precision_name:
            case "fp32":
                return "32-true"
            case "fp16":
                return "16-mixed"
            case "bf16":
                return "bf16-mixed"

    def _build_runtime_profiler_callbacks(self) -> list[Callback]:
        # Builds optional harness-level runtime telemetry callbacks for the current run.
        if not self._configuration.runtime_profiling_enabled:
            return []
        return [
            RuntimeProfiler(
                profile_every_n_steps=self._configuration.runtime_profile_interval_steps
            )
        ]

    def _load_project_checkpoint(self, module: Module) -> None:
        # Restores the module weights for the evaluation stages from the
        # model_state_dict entry of a harness checkpoint. The path is
        # mandatory here because validation and test of a project-trained
        # model without its trained weights would be meaningless evidence.
        # This is a weight load rather than a resume: only the model state is
        # restored, since evaluation needs no optimizer or loop state.
        #
        # Args:
        #     module: The freshly constructed module the weights are loaded
        #         into.
        #
        # Raises:
        #     MisconfigurationError: If no project checkpoint path is
        #         configured, or if the checkpoint carries no model state
        #         dictionary under the harness key.
        checkpoint_path: Path | None = self._configuration.project_checkpoint_path
        if checkpoint_path is None:
            raise MisconfigurationError(
                "project_checkpoint_path is required for validation and test reproduction stages."
            )
        checkpoint: CheckpointDict = load_checkpoint(checkpoint_path)
        state_dict_candidate: CheckpointValue | None = checkpoint.get("model_state_dict")
        if not isinstance(state_dict_candidate, dict):
            raise MisconfigurationError(
                f"Checkpoint {checkpoint_path} does not contain a model_state_dict mapping."
            )
        module.load_state_dict(state_dict_candidate)

    def _build_module_spec(self, architecture_name: ArchitectureName) -> ArchitectureModuleSpec:
        # Delegates module construction to the single-source model registry,
        # forwarding the HiFTNet F0-extractor checkpoint path as the only
        # architecture-specific build option.
        build_options: ModuleBuildOptions = ModuleBuildOptions(
            hiftnet_f0_checkpoint_path=self._configuration.hiftnet_f0_checkpoint_path
        )
        return self._registry.build_module_spec(architecture_name, build_options)

    def _build_experiment_logger(self, module_spec: ArchitectureModuleSpec) -> ExperimentLogger:
        # Constructs the harness-facing experiment logger bound to this cell's
        # identity columns, so every metric row it flushes is attributable to
        # the architecture, variant, seed, and run without a join.
        return ExperimentLogger(
            run_directory=self._configuration.run_directory,
            summary_csv_path=self._configuration.summary_csv_path,
            hyperparameters_path=self._configuration.hyperparameters_path,
            architecture_name=module_spec.architecture_name,
            variant_name=self._build_variant_name(module_spec),
            seed=self._configuration.seed,
            unique_id=self._build_unique_id(module_spec),
            dataset_name="ljspeech",
            hyperparameters_summary=self._summarize_hyperparameters(module_spec),
            interpretation_notes=self._configuration.interpretation_notes
        )

    def _build_unique_id(self, module_spec: ArchitectureModuleSpec) -> str:
        # Composes the row identifier from architecture, evidence category,
        # stage, seed, and run identifier, which together make every CSV row
        # unique across the study.
        return (
            f"{module_spec.architecture_name}_"
            f"{self._configuration.evidence_category}_"
            f"{self._configuration.stage}_"
            f"seed{self._configuration.seed}_"
            f"{self._configuration.run_id}"
        )

    def _build_variant_name(self, module_spec: ArchitectureModuleSpec) -> str:
        # Prefixes the registry variant with the evidence family so
        # project-trained rows are distinguishable from published-weight rows
        # of the same architecture in the shared CSV.
        return f"project_trained_{module_spec.variant_name}"

    def _resolve_evidence_label(self) -> CheckpointEvidence:
        # Maps the training-capable evidence categories onto the checkpoint manifest label domain.
        match self._configuration.evidence_category:
            case "project_hybrid_variants":
                return "project_hybrid_variants"
            case _:
                return "project_trained_reproduction"

    def _summarize_hyperparameters(self, module_spec: ArchitectureModuleSpec) -> str:
        # Summarizes the core run hyperparameters written to experiment logs.
        return (
            f"architecture={module_spec.architecture_name};variant={module_spec.variant_name};"
            f"seed={self._configuration.seed};stage={self._configuration.stage};"
            f"dataset_split={self._configuration.dataset_split_name};"
            f"epochs={self._configuration.train_epoch_count};"
            f"limit_train_batches={self._configuration.limit_train_batches};"
            f"limit_val_batches={self._configuration.limit_val_batches};"
            f"limit_test_batches={self._configuration.limit_test_batches};"
            f"limit_predict_batches={self._configuration.limit_predict_batches};"
            f"metrics={','.join(self._configuration.metric_selection.names)};"
            f"runtime_profiling_enabled={self._configuration.runtime_profiling_enabled};"
            f"runtime_profile_interval_steps={self._configuration.runtime_profile_interval_steps}"
        )

    def _build_hyperparameter_dump(
        self,
        module_spec: ArchitectureModuleSpec
    ) -> HyperparameterDict:
        # Assembles the full run-provenance record written next to the run:
        # identity, hypothesis, data and model configuration dumps, metric
        # selection, and every batch limit, sufficient to reconstruct the
        # exact invocation from the artifact alone.
        return {
            "experiment_name": self._configuration.experiment_name,
            "run_id": self._configuration.run_id,
            "evidence_category": self._configuration.evidence_category,
            "stage": self._configuration.stage,
            "dataset_split_name": self._configuration.dataset_split_name,
            "architecture_name": module_spec.architecture_name,
            "variant_name": self._build_variant_name(module_spec),
            "seed": self._configuration.seed,
            "hypothesis": self._configuration.hypothesis,
            "interpretation_notes": self._configuration.interpretation_notes,
            "run_directory": str(self._configuration.run_directory),
            "project_checkpoint_path": (
                str(self._configuration.project_checkpoint_path)
                if self._configuration.project_checkpoint_path is not None
                else None
            ),
            "data_configuration": self._configuration.data_configuration.model_dump(mode="json"),
            "model_configuration": module_spec.configuration_dump,
            "real_time_factor_configuration": self._configuration.real_time_factor_configuration.model_dump(mode="json"),
            "metrics": list(self._configuration.metric_selection.names),
            "train_epoch_count": self._configuration.train_epoch_count,
            "limit_train_batches": self._configuration.limit_train_batches,
            "limit_val_batches": self._configuration.limit_val_batches,
            "limit_test_batches": self._configuration.limit_test_batches,
            "limit_predict_batches": self._configuration.limit_predict_batches,
            "runtime_profiling_enabled": self._configuration.runtime_profiling_enabled,
            "runtime_profile_interval_steps": self._configuration.runtime_profile_interval_steps
        }

    def _write_metrics_snapshot(self, experiment_logger: ExperimentLogger) -> None:
        # Serializes the logger's reduced metric buffer to the per-run
        # metrics.json. Keys are sorted for diff stability and allow_nan is
        # disabled so a non-finite metric aborts the run instead of entering
        # the evidence silently.
        metrics_snapshot: dict[str, float | int] = experiment_logger.metric_buffer
        self._configuration.metrics_path.write_text(
            json.dumps(metrics_snapshot, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8"
        )

    def _dispatch_validation(
        self,
        module: Module,
        datamodule: LJSpeechDataModule,
        experiment_logger: ExperimentLogger
    ) -> None:
        # Runs one harness validation pass over the configured split; the
        # module's own validation-step logging supplies the metric rows. No
        # metric callback is attached here, because this stage reports what the
        # model itself logs rather than the objective panel.
        #
        # Args:
        #     module: The weight-loaded module to evaluate.
        #     datamodule: The corpus the validation split is drawn from.
        #     experiment_logger: The logger the reduced metrics reach.
        validation_trainer: Trainer = Trainer(
            max_epochs=1,
            callbacks=self._build_runtime_profiler_callbacks(),
            logger=experiment_logger,
            accelerator=self._configuration.accelerator,
            limit_val_batches=self._configuration.limit_val_batches
        )
        validation_trainer.validate(module, datamodule)

    def _dispatch_prediction(
        self,
        module: Module,
        datamodule: LJSpeechDataModule,
        experiment_logger: ExperimentLogger
    ) -> None:
        # Runs the harness prediction loop for synthesis-time measurement.
        # When the metric selection requests rtf, the real-time-factor monitor
        # is inserted ahead of the profiler so its batch hooks bracket the
        # synthesis calls it times. The monitor carries the run's declared
        # timing protocol, so the warm-up count and the number of measured
        # repetitions per timed batch are run configuration rather than
        # anything this method decides.
        #
        # Lane semantics: the figure this produces is the pre-transformation
        # record of the configuration on the run's own lane. Timings are
        # compared only within a lane, and the reported study declares the
        # pre-transformation records non-comparable with deployment-lane
        # real-time factors, which are measured over repeated warm synthesis
        # calls of a transformed artifact.
        prediction_callbacks: list[Callback] = self._build_runtime_profiler_callbacks()
        if "rtf" in self._configuration.metric_selection.names:
            rtf_callback: RealTimeFactorMonitor = RealTimeFactorMonitor(
                configuration=self._configuration.real_time_factor_configuration
            )
            prediction_callbacks.insert(0, rtf_callback)
        prediction_trainer: Trainer = Trainer(
            max_epochs=1,
            callbacks=prediction_callbacks,
            logger=experiment_logger,
            accelerator=self._configuration.accelerator,
            limit_predict_batches=self._configuration.limit_predict_batches
        )
        prediction_trainer.predict(module, datamodule)

    def _dispatch_test(
        self,
        module: Module,
        datamodule: LJSpeechDataModule,
        experiment_logger: ExperimentLogger
    ) -> None:
        # Runs the harness test loop with the MetricSequence callback, which
        # computes the configured objective metric panel over the test split
        # and logs the reduced values through the experiment logger. The panel
        # evaluates complete utterances and truncates reference and candidate to
        # their common true length before scoring, so the published values are
        # true-length values and padding never reaches a metric.
        test_trainer: Trainer = Trainer(
            max_epochs=1,
            callbacks=[
                MetricSequence(self._configuration.metric_selection),
                *self._build_runtime_profiler_callbacks()
            ],
            logger=experiment_logger,
            accelerator=self._configuration.accelerator,
            limit_test_batches=self._configuration.limit_test_batches
        )
        test_trainer.test(module, datamodule)
