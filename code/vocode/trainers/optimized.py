# This module:
# 1. Evaluates one optimized-variant capsule: a project-trained Study 1
#    checkpoint transformed by a registered optimization technique and
#    measured through the prediction and test stages. This is the Study 2
#    evidence lane, whose intervention matrix the Study 3 comparison
#    consumes
# 2. Guards measurement validity before dispatch: hardware-lane registration,
#    summary-CSV schema preflight, and a transformed-execution assertion
#    proving the sampler synthesizes with the transformed network
# 3. Writes the evidence set for the capsule: the variant-level and run-level
#    optimization recipes, the hyperparameter dump with the base-checkpoint
#    SHA-256, peak-memory metrics, the experiment CSV row, and the metrics
#    snapshot JSON
#
# Harness contract (syntheticmind):
# - Prediction and test each receive a fresh single-pass Trainer; the module
#   entering them is the transformed module in evaluation mode, so the
#   harness loops run unchanged over optimized variants
# - Every registered technique maps a harness Module onto a harness Module,
#   which is the property that keeps this evaluator technique-agnostic
# - The base checkpoint is read through the harness load_checkpoint boundary
#   and restored from its model_state_dict entry before transformation
# - RuntimeProfiler, RealTimeFactorMonitor, and MetricSequence enter the
#   Trainer as ordinary harness callbacks
#
# Design decisions:
# - Only the test stage is supported because optimized variants are
#   measurement subjects, not training subjects; training a transformed
#   module would be a different experiment
# - The declared hardware lane is validated against the variant registry
#   because the study's ratios hold the measurement lane fixed for
#   denominator continuity
# - The base-checkpoint SHA-256 enters both recipes so any evaluated variant
#   can be traced to the exact checkpoint bytes it transformed
# - Peak-memory counters are reset before module construction so the report
#   covers construction, transformation, and both evaluation stages of this
#   run only
#
# Author: Rahul Sawhney

import hashlib
import json
import platform
import resource
from pathlib import Path
from typing import ClassVar, cast

import torch
import yaml
from loguru import logger as log
from pydantic import BaseModel, ConfigDict

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.runtime_profiler import RuntimeProfiler
from syntheticmind.core.module import Module
from syntheticmind.core.trainer import Trainer
from syntheticmind.utilities.checkpoint import load_checkpoint
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.seed import SeedManager
from syntheticmind.utilities.types import CheckpointDict, CheckpointValue, HyperparameterDict

from vocode.configs.run import ExperimentConfiguration
from vocode.data.ljspeech_datamodule import LJSpeechDataModule
from vocode.loggers.experiment import ExperimentLogger
from vocode.loggers.writer import ExperimentResultWriter
from vocode.metrics.rtf import RealTimeFactorMonitor
from vocode.metrics.sequence import MetricSequence
from vocode.models.registry import ModelRegistry, ModuleBuildOptions
from vocode.models.vocoder import ArchitectureName
from vocode.optimization.deployment import OnnxRuntimeDeployment
from vocode.optimization.registry import OptimizationVariantName, OptimizationVariantRecord, OptimizationVariantRegistry

__all__: list[str] = ["OptimizedVariantEvaluator"]


class OptimizedVariantModuleSpec(BaseModel):
    # Frozen binding of one evaluation subject: the registry record, the
    # transformed module, and the identity of the checkpoint it came from.
    # Freezing the spec prevents any post-construction substitution between
    # what was transformed and what gets measured.
    #
    # Fields:
    #     record: The registry record naming the variant, its base
    #         architecture, the technique instance that was applied, and the
    #         lane-interpretation note; the technique is read again after
    #         application so its recipe dump carries what the transformation
    #         measured.
    #     module: The transformed harness Module in evaluation mode, which is
    #         the object both stages measure; arbitrary types are permitted on
    #         this model so the live module can be bound directly.
    #     base_checkpoint_path: The checkpoint whose weights the technique
    #         transformed.
    #     base_checkpoint_sha256: Content hash of that checkpoint, computed
    #         from the file itself, so an evaluated variant is traceable to
    #         the exact bytes it started from.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True
    )
    record: OptimizationVariantRecord
    module: Module
    base_checkpoint_path: Path
    base_checkpoint_sha256: str


class OptimizedVariantEvaluator:
    # Evaluator for one optimized-variant capsule. It loads the project
    # checkpoint, hashes it, applies the registered technique, proves the
    # transformed network is the one that synthesizes, and dispatches the
    # prediction and test stages through fresh harness Trainers.
    #
    # In the report's vocabulary, the checkpoint this evaluator loads is the
    # base configuration's Retained Project Checkpoint, and the object it
    # measures is one deployment artifact: one transformation applied to that
    # checkpoint on one requested hardware lane. The identity transformation on
    # the same lane is that artifact's paired same-lane control, which is why
    # the two baseline lanes are registered variants rather than an absence of
    # one.
    #
    # Integration: this evaluator owns what enters the harness; the harness owns
    # the loops. Its contract with the harness has four parts.
    #
    # Trainer construction: prediction and test each receive their own
    # single-epoch Trainer carrying the run's accelerator, the experiment
    # logger, and that stage's own batch limit. The module handed to both is the
    # already-transformed module in evaluation mode, which is what lets the
    # harness loops run unchanged over every registered technique; no training
    # Trainer exists on this path, because an optimized variant is a measurement
    # subject rather than a training subject.
    #
    # Callback policy: prediction places the real-time-factor monitor ahead of
    # the optional runtime profiler so its batch hooks bracket the synthesis
    # calls it times, and omits the monitor when the metric panel does not
    # request the timing metric; test places the objective metric panel first
    # and additionally asks it for the per-utterance table, which the other two
    # evidence paths do not, because an intervention's effect is a distribution
    # rather than a single mean.
    #
    # Precision mapping: no artifact precision label is mapped onto a harness
    # precision mode here. The variant's own numeric precision is a property of
    # the applied technique, so the declared precision label travels into the run
    # recipe as evidence rather than into a Trainer as a setting.
    #
    # Checkpoint boundaries: none. This path writes no checkpoint and no
    # manifest; it reads one. Its artifacts are the two-scope optimization
    # recipe, the hyperparameter dump carrying the base checkpoint and its
    # content hash, the peak-memory metrics, the experiment row, and the metrics
    # snapshot.
    #
    # Responsibility rule: one evaluator instance carries exactly one
    # optimization state. It opens its own variant registry, resolves exactly one
    # variant name from its own configuration, binds that variant's technique
    # into a frozen spec that cannot be substituted afterwards, and keeps its own
    # row counter. Because a technique instance records what its transformation
    # measured, sharing one across capsules would let one capsule's recipe
    # describe another's transformation; keeping the state per evaluator is what
    # makes several capsules in one process safe.
    def __init__(self, configuration: ExperimentConfiguration) -> None:
        # Binds the validated experiment configuration, opens the
        # optimization-variant registry, and zeroes the written-row counter.
        # Construction resolves no variant, applies no technique, creates no
        # artifact directory, and writes no row.
        #
        # Args:
        #     configuration: The validated run description naming the
        #         architecture, the optimization variant, the base
        #         checkpoint, the hardware lane, the seed, and the batch
        #         limits.
        self._configuration: ExperimentConfiguration = configuration
        self._registry: OptimizationVariantRegistry = OptimizationVariantRegistry()
        self._row_count: int = 0

    def run(self) -> int:
        # Executes the capsule in guard-then-measure order: stage and
        # hardware-lane validation, CSV schema preflight, seeding, peak-memory
        # reset, module construction and transformation, the
        # transformed-execution assertion, recipe and hyperparameter writes,
        # then prediction and test dispatch followed by the memory metrics,
        # the CSV flush, and the metrics snapshot. Every guard precedes every
        # write, so a refused capsule leaves neither a run directory nor a
        # summary CSV behind, and the schema preflight fires before measurement
        # rather than after it, so a schema mismatch costs a guard instead of a
        # completed evaluation.
        #
        # Raises:
        #     MisconfigurationError: If the run declares a stage other than
        #         test, states no optimization variant, declares a hardware
        #         lane the variant is not registered for, cites no base
        #         checkpoint, or cites one carrying no model state dictionary.
        #     RuntimeError: If the transformed-execution assertion finds the
        #         sampler resolving a network other than the module's current
        #         one.
        #
        # Returns:
        #     The number of experiment result rows this evaluator has
        #     written, which is one after a completed capsule.
        if self._configuration.stage != "test":
            raise MisconfigurationError(
                "Optimized-variant evaluation supports only stage='test'."
            )
        variant_name: OptimizationVariantName = self._resolve_variant_name()
        self._validate_hardware_lane(variant_name)
        ExperimentResultWriter(self._configuration.summary_csv_path).preflight_schema()
        SeedManager.seed_everything(self._configuration.seed)
        self._reset_peak_memory_statistics()
        module_spec: OptimizedVariantModuleSpec = self._build_module_spec(
            self._configuration.architecture_name,
            variant_name
        )
        self._assert_transformed_network_execution(module_spec.module)
        datamodule: LJSpeechDataModule = LJSpeechDataModule(self._configuration.data_configuration)
        experiment_logger: ExperimentLogger = self._build_experiment_logger(module_spec)
        self._write_optimization_recipe(module_spec)
        experiment_logger.log_hyperparams(self._build_hyperparameter_dump(module_spec))
        self._dispatch_prediction(module_spec.module, datamodule, experiment_logger)
        self._dispatch_test(module_spec.module, datamodule, experiment_logger)
        self._log_peak_memory(experiment_logger)
        experiment_logger.flush_to_experiments_csv()
        self._row_count: int = self._row_count + 1
        self._write_metrics_snapshot(experiment_logger)
        log.info(
            f"Optimized-variant row {self._row_count} written for "
            f"{module_spec.record.base_architecture} variant={module_spec.record.variant_name} "
            f"seed={self._configuration.seed}"
        )
        return self._row_count

    @property
    def row_count(self) -> int:
        # Returns the number of experiment result rows written by this runner.
        return self._row_count

    def _resolve_variant_name(self) -> OptimizationVariantName:
        # Requires an explicit variant name; there is no default variant
        # because an unstated technique would make the capsule's identity
        # ambiguous.
        variant_name: str | None = self._configuration.optimization_variant_name
        if variant_name is None:
            raise MisconfigurationError(
                "optimization_variant_name is required for optimized-variant evaluation."
            )
        return cast(OptimizationVariantName, variant_name)

    def _build_module_spec(
        self,
        architecture_name: ArchitectureName,
        variant_name: OptimizationVariantName
    ) -> OptimizedVariantModuleSpec:
        # Produces the evaluation subject: builds the baseline module, loads
        # and hashes the project checkpoint, binds the ONNX deployment
        # technique to the run configuration where applicable, applies the
        # technique, and freezes the result in evaluation mode. Hashing
        # happens against the checkpoint file itself so the recipe records
        # the bytes that were actually restored. The registry lookup comes
        # first, so an unsupported cell is refused before a module is built or
        # a checkpoint is read. The deployment technique is the one member of
        # the technique family needing run-level context, and it receives it
        # through an explicit bind rather than through a widened apply
        # signature that every other technique would have to accept.
        #
        # Args:
        #     architecture_name: The architecture whose trained checkpoint is
        #         transformed.
        #     variant_name: The registered variant to construct and apply.
        #
        # Raises:
        #     MisconfigurationError: If the cell is unsupported, if no base
        #         checkpoint is configured, or if the checkpoint carries no
        #         model state dictionary.
        #
        # Returns:
        #     The frozen evaluation subject binding the transformed module to
        #     its registry record and its base checkpoint identity.
        record: OptimizationVariantRecord = self._registry.get(architecture_name, variant_name)
        module: Module = self._build_base_module(architecture_name)
        base_checkpoint_path: Path = self._resolve_base_checkpoint_path()
        self._load_base_checkpoint(module, base_checkpoint_path)
        base_checkpoint_sha256: str = self._compute_sha256(base_checkpoint_path)
        if isinstance(record.technique, OnnxRuntimeDeployment):
            record.technique.bind(self._configuration)
        transformed_module: Module = record.technique.apply(module)
        transformed_module.eval()
        return OptimizedVariantModuleSpec(
            record=record,
            module=transformed_module,
            base_checkpoint_path=base_checkpoint_path,
            base_checkpoint_sha256=base_checkpoint_sha256
        )

    def _validate_hardware_lane(self, variant_name: OptimizationVariantName) -> None:
        # Rejects a run whose declared hardware lane is not registered for the variant.
        # A speed ratio is only meaningful when its numerator and its
        # denominator were measured on the same hardware, so the lane is fixed
        # per variant and a run declaring another one is refused rather than
        # measured and later reconciled.
        #
        # Args:
        #     variant_name: The resolved variant whose registered lanes the
        #         run's declared lane is checked against.
        #
        # Raises:
        #     MisconfigurationError: If the declared lane is absent from the
        #         variant's registered lanes, naming both.
        registered_lanes: tuple[str, ...] = self._registry.supported_hardware(variant_name)
        declared_lane: str = self._configuration.artifact_layout.hardware_name
        if declared_lane not in registered_lanes:
            raise MisconfigurationError(
                f"Variant {variant_name} is registered for hardware lanes "
                f"{registered_lanes}, got {declared_lane!r}; measurement lanes are fixed "
                f"for denominator continuity."
            )

    def _assert_transformed_network_execution(self, module: Module) -> None:
        # Proves that any provider-backed sampler resolves the module's current network,
        # so no retained pre-transformation object can participate in synthesis.
        # Techniques that replace the network object rather than mutating it in
        # place would otherwise leave a sampler holding the original, and the
        # capsule would then time one model while scoring another. A module
        # carrying no sampler, or a sampler with no provider seam, has nothing to
        # diverge from and is admitted unconditionally.
        #
        # Args:
        #     module: The transformed module about to be measured.
        #
        # Raises:
        #     RuntimeError: If the sampler's provider resolves a network
        #         object that is not the module's current network.
        sampler: object = getattr(module, "_sampler", None)
        network_provider: object = getattr(sampler, "_network_provider", None)
        if callable(network_provider) and network_provider() is not module.network:
            raise RuntimeError(
                "Transformed-execution assertion failed: the sampler resolves a network "
                "object different from module.network, so measurements would describe "
                "two different models."
            )

    def _reset_peak_memory_statistics(self) -> None:
        # Resets accelerator peak-memory counters so the capsule reports this run only.
        # The reset happens before module construction, so the reported peak
        # brackets construction, transformation, and both evaluation stages. It
        # is a no-op without an accelerator, since the host counter the run also
        # reports is a process-lifetime maximum the operating system owns and
        # cannot be reset.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _log_peak_memory(self, experiment_logger: ExperimentLogger) -> None:
        # Records peak host and accelerator memory. The ru_maxrss unit
        # differs by platform: Darwin reports bytes while Linux reports
        # kilobytes, hence the conditional scaling before the megabyte
        # conversion. Accelerator peak memory is reported only when CUDA
        # is present.
        #
        # Args:
        #     experiment_logger: The logger the peak-memory metrics are
        #         published to; they are logged at step zero because they are
        #         run-level facts rather than points on a curve.
        peak_metrics: dict[str, float] = {}
        maximum_resident_bytes: float = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if platform.system() != "Darwin":
            maximum_resident_bytes: float = maximum_resident_bytes * 1024.0
        peak_metrics["peak_host_memory_megabytes"] = maximum_resident_bytes / (1024.0 * 1024.0)
        if torch.cuda.is_available():
            peak_metrics["peak_accelerator_memory_megabytes"] = (
                float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            )
        experiment_logger.log_metrics(peak_metrics, step=0)

    def _build_base_module(self, architecture_name: ArchitectureName) -> Module:
        # Builds the baseline module through the single-source model registry.
        model_registry: ModelRegistry = ModelRegistry()
        return model_registry.build_module_spec(architecture_name, ModuleBuildOptions()).module

    def _resolve_base_checkpoint_path(self) -> Path:
        # Requires the project checkpoint path; an optimized variant without
        # a stated base checkpoint would have no defined identity.
        checkpoint_path: Path | None = self._configuration.project_checkpoint_path
        if checkpoint_path is None:
            raise MisconfigurationError(
                "project_checkpoint_path is required for optimized-variant evaluation."
            )
        return checkpoint_path

    def _load_base_checkpoint(self, module: Module, checkpoint_path: Path) -> None:
        # Restores the baseline weights from the model_state_dict entry of
        # the harness checkpoint before any transformation is applied. Loading
        # must precede the technique because several techniques read or rewrite
        # the weights themselves, and a transformation applied to an
        # uninitialized network would measure nothing the study claims.
        #
        # Args:
        #     module: The freshly built baseline module.
        #     checkpoint_path: The harness checkpoint whose model state is
        #         restored.
        #
        # Raises:
        #     MisconfigurationError: If the checkpoint carries no model state
        #         dictionary under the harness key.
        checkpoint: CheckpointDict = load_checkpoint(checkpoint_path)
        state_dict_candidate: CheckpointValue | None = checkpoint.get("model_state_dict")
        if not isinstance(state_dict_candidate, dict):
            raise MisconfigurationError(
                f"Checkpoint {checkpoint_path} does not contain a model_state_dict mapping."
            )
        module.load_state_dict(state_dict_candidate)

    def _compute_sha256(self, file_path: Path) -> str:
        # Computes the checkpoint content hash recorded into the optimization recipe.
        digest: hashlib._Hash = hashlib.sha256()
        with file_path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_optimization_recipe(self, module_spec: OptimizedVariantModuleSpec) -> None:
        # Writes the recipe at two scopes: a variant-level recipe holding only
        # run-invariant facts (technique configuration, base-checkpoint
        # identity), and a run-level recipe that adds this capsule's lane,
        # precision, seed, hypothesis identifier, and code commit. The run
        # recipe references the variant recipe path so the two remain
        # joinable from the artifact tree alone. The split exists because the
        # variant recipe is shared by every run of that variant: admitting a
        # per-run field into it would make it describe one run and misdescribe
        # every other. The technique dump is read after the transformation has
        # been applied, so it carries what the transformation measured and not
        # only what it was configured with.
        #
        # Args:
        #     module_spec: The frozen evaluation subject supplying the
        #         technique configuration and the base checkpoint identity
        #         both scopes record.
        variant_recipe: dict[str, object] = {
            "recipe_scope": "variant",
            "variant_name": module_spec.record.variant_name,
            "base_architecture": module_spec.record.base_architecture,
            "base_checkpoint_path": str(module_spec.base_checkpoint_path),
            "base_checkpoint_sha256": module_spec.base_checkpoint_sha256,
            "technique_configuration": module_spec.record.technique.configuration_dump(),
            "measurement_note": module_spec.record.measurement_note
        }
        run_recipe: dict[str, object] = {
            **variant_recipe,
            "recipe_scope": "run",
            "variant_recipe_path": str(self._variant_optimization_recipe_path()),
            "hardware_name": self._configuration.artifact_layout.hardware_name,
            "precision_name": self._configuration.artifact_layout.precision_name,
            "run_id": self._configuration.run_id,
            "seed": self._configuration.seed,
            "stage": self._configuration.stage,
            "dataset_split_name": self._configuration.dataset_split_name,
            "hypothesis_id": self._configuration.optimization_hypothesis_id,
            "code_commit_hash": self._configuration.code_commit_hash
        }
        self._write_yaml(self._variant_optimization_recipe_path(), variant_recipe)
        self._write_yaml(self._run_optimization_recipe_path(), run_recipe)

    def _variant_optimization_recipe_path(self) -> Path:
        # Returns the variant-level recipe path, which must not contain per-run fields.
        # It sits two levels above the run capsule, which is the variant's own
        # directory, so every seed and every repetition of that variant writes
        # and overwrites the identical run-invariant document.
        return self._configuration.run_directory.parents[1] / "optimization_recipe.yaml"

    def _run_optimization_recipe_path(self) -> Path:
        # Returns the immutable recipe path for the current run capsule.
        return self._configuration.run_directory / "optimization_recipe.yaml"

    def _write_yaml(self, path: Path, payload: dict[str, object]) -> None:
        # Writes a YAML artifact atomically enough for one-run-per-process execution.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
            encoding="utf-8"
        )

    def _build_experiment_logger(self, module_spec: OptimizedVariantModuleSpec) -> ExperimentLogger:
        # Constructs the harness-facing experiment logger with the optimized
        # variant-name prefix, keeping optimized rows distinguishable from
        # project-trained and published rows in the shared CSV.
        return ExperimentLogger(
            run_directory=self._configuration.run_directory,
            summary_csv_path=self._configuration.summary_csv_path,
            hyperparameters_path=self._configuration.hyperparameters_path,
            architecture_name=module_spec.record.base_architecture,
            variant_name=f"optimized_{module_spec.record.variant_name}",
            seed=self._configuration.seed,
            unique_id=self._build_unique_id(module_spec),
            dataset_name="ljspeech",
            hyperparameters_summary=self._summarize_hyperparameters(module_spec),
            interpretation_notes=self._configuration.interpretation_notes
        )

    def _build_unique_id(self, module_spec: OptimizedVariantModuleSpec) -> str:
        # Composes the row identifier from architecture, variant, evidence
        # category, stage, seed, and run identifier, which together make
        # every optimized-variant CSV row unique across the study.
        return (
            f"{module_spec.record.base_architecture}_"
            f"{module_spec.record.variant_name}_"
            f"{self._configuration.evidence_category}_"
            f"{self._configuration.stage}_"
            f"seed{self._configuration.seed}_"
            f"{self._configuration.run_id}"
        )

    def _summarize_hyperparameters(self, module_spec: OptimizedVariantModuleSpec) -> str:
        # Summarizes the core run hyperparameters written to experiment logs.
        return (
            f"architecture={module_spec.record.base_architecture};"
            f"variant={module_spec.record.variant_name};"
            f"stage={self._configuration.stage};"
            f"dataset_split={self._configuration.dataset_split_name};"
            f"base_checkpoint={module_spec.base_checkpoint_path};"
            f"seed={self._configuration.seed};"
            f"limit_test_batches={self._configuration.limit_test_batches};"
            f"limit_predict_batches={self._configuration.limit_predict_batches};"
            f"metrics={','.join(self._configuration.metric_selection.names)};"
            f"runtime_profiling_enabled={self._configuration.runtime_profiling_enabled};"
            f"runtime_profile_interval_steps={self._configuration.runtime_profile_interval_steps}"
        )

    def _build_hyperparameter_dump(
        self,
        module_spec: OptimizedVariantModuleSpec
    ) -> HyperparameterDict:
        # Assembles the full run-provenance record: identity, technique
        # configuration, base-checkpoint path and SHA-256, data and
        # measurement configuration dumps, and every batch limit, sufficient
        # to reconstruct the exact invocation from the artifact alone.
        return {
            "experiment_name": self._configuration.experiment_name,
            "run_id": self._configuration.run_id,
            "evidence_category": self._configuration.evidence_category,
            "stage": self._configuration.stage,
            "dataset_split_name": self._configuration.dataset_split_name,
            "architecture_name": module_spec.record.base_architecture,
            "variant_name": f"optimized_{module_spec.record.variant_name}",
            "optimization_variant_name": module_spec.record.variant_name,
            "base_checkpoint_path": str(module_spec.base_checkpoint_path),
            "base_checkpoint_sha256": module_spec.base_checkpoint_sha256,
            "technique_configuration": module_spec.record.technique.configuration_dump(),
            "seed": self._configuration.seed,
            "hypothesis": self._configuration.hypothesis,
            "interpretation_notes": self._configuration.interpretation_notes,
            "run_directory": str(self._configuration.run_directory),
            "data_configuration": self._configuration.data_configuration.model_dump(mode="json"),
            "real_time_factor_configuration": self._configuration.real_time_factor_configuration.model_dump(mode="json"),
            "metrics": list(self._configuration.metric_selection.names),
            "limit_train_batches": self._configuration.limit_train_batches,
            "limit_val_batches": self._configuration.limit_val_batches,
            "limit_test_batches": self._configuration.limit_test_batches,
            "limit_predict_batches": self._configuration.limit_predict_batches,
            "runtime_profiling_enabled": self._configuration.runtime_profiling_enabled,
            "runtime_profile_interval_steps": self._configuration.runtime_profile_interval_steps
        }

    def _build_runtime_profiler_callbacks(self) -> list[Callback]:
        # Builds optional harness-level runtime telemetry callbacks for the current run.
        if not self._configuration.runtime_profiling_enabled:
            return []
        return [
            RuntimeProfiler(
                profile_every_n_steps=self._configuration.runtime_profile_interval_steps
            )
        ]

    def _write_metrics_snapshot(self, experiment_logger: ExperimentLogger) -> None:
        # Serializes the logger's reduced metric buffer to the per-run
        # metrics.json with sorted keys for diff stability; allow_nan is
        # disabled so a non-finite metric aborts the run instead of entering
        # the evidence silently.
        metrics_snapshot: dict[str, float | int] = experiment_logger.metric_buffer
        self._configuration.metrics_path.write_text(
            json.dumps(metrics_snapshot, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8"
        )

    def _dispatch_prediction(
        self,
        module: Module,
        datamodule: LJSpeechDataModule,
        experiment_logger: ExperimentLogger
    ) -> None:
        # Runs the harness prediction loop for synthesis-time measurement.
        # When the metric selection requests rtf, the real-time-factor
        # monitor is inserted ahead of the profiler so its batch hooks
        # bracket the synthesis calls it times. The monitor carries the run's
        # declared timing protocol, so the warm-up count and the number of
        # measured repetitions per timed batch are run configuration rather
        # than anything this method decides.
        #
        # Lane semantics: the figure this produces is the deployment-lane
        # estimand, measured over repeated warm synthesis calls after the
        # warm-up batches. The reported study declares it non-comparable with
        # the single-pass pre-transformation accelerator timings and with the
        # retained comparative processor-lane record, so a speedup is formed
        # only against the same-lane control of the same artifact, and never
        # across lanes or against a differently timed baseline record.
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
        # computes the configured objective metric panel and additionally
        # writes the per-utterance CSV into the capsule's metrics directory
        # for distribution-level analysis. The panel evaluates complete
        # utterances and truncates reference and candidate to their common true
        # length before scoring, so the recorded values are true-length values
        # and padding never reaches a metric. The per-utterance table is
        # requested on this path alone, because an intervention's quality
        # effect is a distribution over utterances: the study pairs an artifact
        # with its same-lane control utterance by utterance, and a single
        # reduced mean would neither support that pairing nor show whether a
        # transformation degrades a few utterances severely or all of them
        # slightly.
        #
        # Args:
        #     module: The transformed module in evaluation mode.
        #     datamodule: The corpus the test split is drawn from.
        #     experiment_logger: The logger the reduced panel reaches.
        test_trainer: Trainer = Trainer(
            max_epochs=1,
            callbacks=[
                MetricSequence(
                    self._configuration.metric_selection,
                    per_utterance_csv_path=(
                        self._configuration.artifact_layout.metrics_directory
                        / "metrics_per_utterance.csv"
                    )
                ),
                *self._build_runtime_profiler_callbacks()
            ],
            logger=experiment_logger,
            accelerator=self._configuration.accelerator,
            limit_test_batches=self._configuration.limit_test_batches
        )
        test_trainer.test(module, datamodule)
