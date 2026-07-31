# This module:
# 1. Runs the Study 2 recovery-training arms: pruned-and-recovered
#    fine-tuning (full and half budget) and the matched dense-continuation
#    causal control, all starting from the SHA-verified Study 1 baseline
# 2. Writes the recovery evidence: the baked recovered checkpoint with its
#    hash, the recovery recipe with mask paths and sparsity trajectory,
#    the checkpoint manifest, and the experiment CSV row
#
# Harness contract (syntheticmind):
# - Fine-tuning runs through an ordinary Trainer.fit with ModelCheckpoint
#   and device telemetry; the pruning masks are held by the torch pruning
#   hooks across the whole fit, so the optimizer trains under the mask
#   exactly as the loops execute it
# - The masked model is validated once before recovery through a bounded
#   Trainer.validate pass, giving the recovery curve its step-zero point
#
# Design decisions:
# - Masks are baked only after fine-tuning completes, so the persisted
#   recovered checkpoint carries literal zeros and no hook dependency
# - The dense-continuation arm fine-tunes the unmodified baseline under
#   the identical budget and learning-rate scale, which is what makes it
#   the causal control for recovery claims
# - Recovery uses a tenth of the baseline learning rate through the
#   registry's build option, matching fine-tuning practice rather than
#   restarting the training recipe
# - Both checkpoint hashes and the full mask inventory enter the recovery
#   recipe, so the artifact lineage is verifiable from the capsule alone
#
# Author: Rahul Sawhney

import hashlib
import json
from pathlib import Path
from typing import Literal, cast

import torch
import yaml
from loguru import logger as log
from torch import nn

from syntheticmind.callbacks.device_stats_monitor import DeviceStatsMonitor
from syntheticmind.callbacks.model_checkpoint import ModelCheckpoint
from syntheticmind.core.module import Module
from syntheticmind.core.trainer import Trainer
from syntheticmind.utilities.checkpoint import load_checkpoint
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.seed import SeedManager
from syntheticmind.utilities.types import CheckpointDict, CheckpointValue

from vocode.configs.run import ExperimentConfiguration
from vocode.data.ljspeech_datamodule import LJSpeechDataModule
from vocode.loggers.checkpoint import CheckpointLogger, CheckpointManifest
from vocode.loggers.experiment import ExperimentLogger
from vocode.models.registry import ModelRegistry, ModuleBuildOptions
from vocode.optimization.pruning import (
    StructuredMagnitudePruning,
    StructuredPruningConfig,
    UnstructuredMagnitudePruning,
    UnstructuredPruningConfig,
)
from vocode.optimization.registry import PruningArmName, TechniqueResolutionTable

__all__: list[str] = ["OptimizationRecoveryRunner"]

type RecoveryVariantName = Literal["pruned_50_recovered", "pruned_50_recovered_half", "dense_continued"]


class OptimizationRecoveryRunner:
    # Experiment runner coordinating one controlled VOCODE recovery-training path.
    # The pruned arms prune the SHA-verified baseline, fine-tune with masks held by the
    # pruning hooks, bake the zeros, and write the recovered checkpoint capsule. The
    # dense-continuation arm fine-tunes the unmodified baseline under the identical
    # budget, giving recovery claims their matched causal control. The half-budget arm
    # is the recovery-budget ablation and receives its budget from the run configuration.
    #
    # Integration: this runner owns what enters the harness while the harness owns
    # the loops. It constructs up to two Trainers per run: a bounded
    # single-epoch validation Trainer that measures the masked model before any
    # recovery, giving the recovery curve its step-zero point, and the recovery
    # fit Trainer carrying a monitored ModelCheckpoint and device telemetry. The
    # masks are deliberately left unbaked across the whole fit, because the
    # pruning reparametrization is what holds the zeros in place while the
    # optimizer runs; baking happens once afterwards, so the persisted checkpoint
    # carries literal zeros and no hook dependency. One runner instance owns
    # exactly one recovery arm, resolved at construction from the configuration
    # and immutable thereafter, and it owns its own arm implementations and row
    # counter, so two runners in one process never share recovery state.
    #
    # Executed status: the reported study admitted no dense-continuation group,
    # so its recovery rows describe one retained recovered checkpoint and
    # support descriptive rather than causal claims. The dense-continuation path
    # here is the control the recovery design calls for, not an executed control
    # the reported evidence rests on.
    def __init__(self, configuration: ExperimentConfiguration) -> None:
        # Binds the configuration, resolves the recovery variant and the
        # architecture's pruning arm, and prepares both arm implementations.
        # Both resolutions happen here rather than inside run, so a
        # misconfigured cell fails at construction and never reaches a corpus
        # or an artifact directory.
        #
        # Args:
        #     configuration: The validated run description, which must
        #         declare one of the three registered recovery variants and
        #         an architecture holding a declared technique resolution.
        #
        # Raises:
        #     MisconfigurationError: If the declared optimization variant is
        #         absent or is not a recovery arm, or if the architecture has
        #         no declared row in the technique resolution table.
        self._configuration: ExperimentConfiguration = configuration
        self._recovery_variant: RecoveryVariantName = self._resolve_recovery_variant()
        self._pruning_arm: PruningArmName = TechniqueResolutionTable().get(
            configuration.architecture_name
        ).pruning_arm
        self._structured_pruning: StructuredMagnitudePruning = StructuredMagnitudePruning(StructuredPruningConfig())
        self._unstructured_pruning: UnstructuredMagnitudePruning = UnstructuredMagnitudePruning(UnstructuredPruningConfig())
        self._row_count: int = 0

    def run(self) -> int:
        # Executes the recovery arm: seeded module construction with the
        # reduced learning rate, SHA-verified baseline restoration, mask
        # application with the step-zero masked validation (pruned arms
        # only), the bounded recovery fit, mask baking, and the full
        # evidence write-out. The stage guard fires first, before seeding and
        # before any artifact directory exists, so a refused run leaves nothing
        # behind. Seeding precedes module construction because parameter
        # initialization must sit inside the seeded region even though the
        # weights are about to be overwritten from the baseline.
        #
        # Raises:
        #     MisconfigurationError: If the run declares a stage other than
        #         training, if no base checkpoint path is stated, or if that
        #         checkpoint carries no model state dictionary.
        #
        # Returns:
        #     The number of experiment result rows this runner has written,
        #     which is one after a completed arm.
        if self._configuration.stage != "train":
            raise MisconfigurationError("Optimization recovery only supports stage='train'.")
        SeedManager.seed_everything(self._configuration.seed)
        model_registry: ModelRegistry = ModelRegistry()
        module: Module = model_registry.build_module_spec(
            self._configuration.architecture_name,
            ModuleBuildOptions(learning_rate_scale=0.1)
        ).module
        base_checkpoint_path: Path = self._resolve_base_checkpoint_path()
        self._load_base_state(module, base_checkpoint_path)
        base_checkpoint_sha256: str = self._compute_sha256(base_checkpoint_path)
        datamodule: LJSpeechDataModule = LJSpeechDataModule(self._configuration.data_configuration)
        experiment_logger: ExperimentLogger = self._build_experiment_logger()
        network: nn.Module = getattr(module, "network")
        pruned_paths: list[str] = []
        sparsity_at_mask: float = 0.0
        coverage: float = 0.0
        # The pruned arms mask here and leave the masks unbaked; the dense
        # continuation arm masks nothing, which is exactly what makes the two
        # comparable under one identical budget.
        if self._is_pruned_variant():
            pruned_paths: list[str] = self._apply_arm_masks(network)
            sparsity_at_mask: float = self._measure_arm_sparsity(network)
            coverage: float = self._measure_arm_coverage(network)
            log.info(
                f"{self._pruning_arm} masks applied to {len(pruned_paths)} weight tensors; "
                f"sparsity={sparsity_at_mask:.4f} coverage={coverage:.4f}"
            )
            self._run_unrecovered_validation(module, datamodule, experiment_logger)
        else:
            log.info(
                "Dense continuation arm: no masks applied; fine-tuning the unmodified "
                "baseline under the matched recovery budget."
            )
        recovered_manifest: CheckpointManifest = self._run_recovery_fit(
            module,
            datamodule,
            experiment_logger
        )
        # Baking happens only after the fit has finished, so the sparsity
        # measured next describes literal zeros in the weights about to be
        # persisted rather than a mask that would not survive serialization.
        if self._is_pruned_variant():
            self._bake_arm_masks(network)
        final_sparsity: float = self._measure_arm_sparsity(network)
        recovered_checkpoint_path: Path = self._resolve_recovered_checkpoint_path(recovered_manifest)
        self._save_baked_state(module, recovered_checkpoint_path)
        recovered_sha256: str = self._compute_sha256(recovered_checkpoint_path)
        self._write_recovery_recipe(
            base_checkpoint_path=base_checkpoint_path,
            base_checkpoint_sha256=base_checkpoint_sha256,
            recovered_checkpoint_path=recovered_checkpoint_path,
            recovered_sha256=recovered_sha256,
            pruned_paths=pruned_paths,
            sparsity_at_mask=sparsity_at_mask,
            final_sparsity=final_sparsity,
            coverage=coverage
        )
        experiment_logger.log_hyperparams(self._build_hyperparameter_dump(base_checkpoint_path))
        experiment_logger.flush_to_experiments_csv()
        self._row_count: int = self._row_count + 1
        self._write_metrics_snapshot(experiment_logger)
        log.info(
            f"{self._recovery_variant} training finished: final arm sparsity "
            f"{final_sparsity:.4f}, checkpoint {recovered_checkpoint_path}"
        )
        return self._row_count

    def _resolve_recovery_variant(self) -> RecoveryVariantName:
        # Requires one of the three registered recovery variants; anything
        # else cannot be trained by this runner.
        variant_name: str | None = self._configuration.optimization_variant_name
        if variant_name not in ("pruned_50_recovered", "pruned_50_recovered_half", "dense_continued"):
            raise MisconfigurationError(
                f"Optimization recovery supports pruned_50_recovered, "
                f"pruned_50_recovered_half, and dense_continued; got {variant_name!r}."
            )
        return cast(RecoveryVariantName, variant_name)

    def _is_pruned_variant(self) -> bool:
        # Reports whether the resolved recovery variant carries pruning masks.
        return self._recovery_variant in ("pruned_50_recovered", "pruned_50_recovered_half")

    def _apply_arm_masks(self, network: nn.Module) -> list[str]:
        # Applies the resolved arm's masks and returns the pruned parameter paths.
        match self._pruning_arm:
            case "linear_structured":
                return self._structured_pruning.apply_masks(network)
            case "global_unstructured":
                return self._unstructured_pruning.apply_masks(network)

    def _bake_arm_masks(self, network: nn.Module) -> None:
        # Bakes the resolved arm's masks into literal zero weights.
        match self._pruning_arm:
            case "linear_structured":
                self._structured_pruning.bake_masks(network)
            case "global_unstructured":
                self._unstructured_pruning.bake_masks(network)

    def _measure_arm_sparsity(self, network: nn.Module) -> float:
        # Measures sparsity over the resolved arm's covered scope.
        match self._pruning_arm:
            case "linear_structured":
                return self._structured_pruning.measured_sparsity(network)
            case "global_unstructured":
                return self._unstructured_pruning.measured_sparsity(network)

    def _measure_arm_coverage(self, network: nn.Module) -> float:
        # Measures the parameter fraction the resolved arm's scope covers.
        match self._pruning_arm:
            case "linear_structured":
                return self._structured_pruning.covered_parameter_fraction(network)
            case "global_unstructured":
                return self._unstructured_pruning.covered_parameter_fraction(network)

    def _arm_configuration_dump(self) -> dict[str, object]:
        # Returns the resolved arm's transformation configuration for durable records.
        match self._pruning_arm:
            case "linear_structured":
                return self._structured_pruning.configuration_dump()
            case "global_unstructured":
                return self._unstructured_pruning.configuration_dump()

    def _resolve_base_checkpoint_path(self) -> Path:
        # Requires the baseline checkpoint path; recovery without a stated
        # starting checkpoint would have no defined lineage.
        checkpoint_path: Path | None = self._configuration.project_checkpoint_path
        if checkpoint_path is None:
            raise MisconfigurationError(
                "project_checkpoint_path is required for optimization recovery."
            )
        return checkpoint_path

    def _load_base_state(self, module: Module, checkpoint_path: Path) -> None:
        # Restores the baseline weights from the model_state_dict entry of
        # the harness checkpoint before any masking.
        checkpoint: CheckpointDict = load_checkpoint(checkpoint_path)
        state_dict_candidate: CheckpointValue | None = checkpoint.get("model_state_dict")
        if not isinstance(state_dict_candidate, dict):
            raise MisconfigurationError(
                f"Checkpoint {checkpoint_path} does not contain a model_state_dict mapping."
            )
        module.load_state_dict(state_dict_candidate)

    def _run_unrecovered_validation(
        self,
        module: Module,
        datamodule: LJSpeechDataModule,
        experiment_logger: ExperimentLogger
    ) -> None:
        # Measures the masked model before any recovery so the curve has its step-zero point.
        # The pass runs on its own single-epoch Trainer with the progress bar
        # suppressed and the harness pre-fit validation steps disabled, under
        # the run's validation batch limit, and it publishes through the same
        # experiment logger, so the unrecovered point and the recovery curve
        # share one metric stream.
        #
        # Args:
        #     module: The masked module, with its pruning reparametrizations
        #         still live.
        #     datamodule: The corpus the validation split is drawn from.
        #     experiment_logger: The logger the step-zero metrics reach.
        validation_trainer: Trainer = Trainer(
            max_epochs=1,
            logger=experiment_logger,
            accelerator=self._configuration.accelerator,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            limit_val_batches=self._configuration.limit_val_batches
        )
        validation_trainer.validate(module, datamodule)
        log.info("Unrecovered masked validation recorded before recovery fine-tuning.")

    def _run_recovery_fit(
        self,
        module: Module,
        datamodule: LJSpeechDataModule,
        experiment_logger: ExperimentLogger
    ) -> CheckpointManifest:
        # Executes the bounded fine-tuning; pruned arms keep masks held by the pruning hooks.
        # The checkpoint boundary is the monitored validation loss with the last
        # epoch also retained and an on-exception save, so an interrupted
        # recovery still leaves a loadable artifact. Both recovery arms receive
        # the identical Trainer configuration, epoch budget, and precision,
        # which is what makes the dense continuation a matched control rather
        # than merely an untreated run.
        #
        # Args:
        #     module: The module to fine-tune, masked or dense according to
        #         the arm.
        #     datamodule: The corpus the fit and its validation draw from.
        #     experiment_logger: The logger the recovery metrics reach.
        #
        # Returns:
        #     The checkpoint manifest recording what the monitored checkpoint
        #     callback actually kept, whose directory receives the baked
        #     state written afterwards.
        model_checkpoint: ModelCheckpoint = ModelCheckpoint(
            dirpath=self._configuration.checkpoints_directory,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            save_on_exception=True
        )
        recovery_trainer: Trainer = Trainer(
            max_epochs=self._configuration.train_epoch_count,
            callbacks=[model_checkpoint, DeviceStatsMonitor()],
            logger=experiment_logger,
            accelerator=self._configuration.accelerator,
            precision=self._resolve_trainer_precision(),
            log_every_n_steps=5,
            num_sanity_val_steps=0,
            limit_train_batches=self._configuration.limit_train_batches,
            limit_val_batches=self._configuration.limit_val_batches
        )
        recovery_trainer.fit(module, datamodule)
        return CheckpointLogger().write(
            model_checkpoint=model_checkpoint,
            run_directory=self._configuration.run_directory,
            evidence_label="project_optimization_checkpoint",
            architecture_name=self._configuration.architecture_name,
            variant_name=self._recovery_variant,
            seed=self._configuration.seed,
            unique_id=self._build_unique_id(),
            experiment_name=self._configuration.experiment_name,
            run_id=self._configuration.run_id
        )

    def _resolve_trainer_precision(self) -> str:
        # Resolves the harness precision literal for the recovery trainer.
        # The artifact layout's precision label and the harness precision domain
        # are different vocabularies, so the mapping is explicit; anything
        # outside the two mixed-precision labels resolves to full precision.
        #
        # Returns:
        #     The harness precision literal the recovery Trainer is built
        #     with.
        match self._configuration.artifact_layout.precision_name:
            case "bf16":
                return "bf16-mixed"
            case "fp16":
                return "16-mixed"
            case _:
                return "32-true"

    def _resolve_recovered_checkpoint_path(self, manifest: CheckpointManifest) -> Path:
        # Resolves where the trained weights are written inside the capsule.
        # The two arms write under different filenames inside the same
        # checkpoint directory, so a recovered artifact and a dense continuation
        # can never be mistaken for one another on disk.
        #
        # Args:
        #     manifest: The manifest written after the fit, whose checkpoint
        #         directory anchors the path.
        #
        # Returns:
        #     The file the post-recovery state is persisted to.
        checkpoint_directory: Path = Path(manifest.checkpoint_directory)
        if self._is_pruned_variant():
            return checkpoint_directory / "recovered_baked.ckpt"
        return checkpoint_directory / "continued_dense.ckpt"

    def _save_baked_state(self, module: Module, checkpoint_path: Path) -> None:
        # Persists the post-recovery state under the model_state_dict key,
        # matching the layout every evaluator loads. Because the masks were
        # baked first, the written tensors contain literal zeros and no
        # pruning-hook entries, which is what lets the artifact load into an
        # unmodified network at evaluation time.
        #
        # Args:
        #     module: The baked module whose state dictionary is written.
        #     checkpoint_path: Destination file, whose parent directory is
        #         created if absent.
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": module.state_dict()}, checkpoint_path)

    def _compute_sha256(self, file_path: Path) -> str:
        # Computes the integrity hash recorded beside the artifact. The file is
        # read in fixed-size chunks so a checkpoint of any size is hashed
        # without being held in memory.
        #
        # Args:
        #     file_path: The artifact whose bytes are digested.
        #
        # Returns:
        #     The hexadecimal digest recorded into the recovery recipe.
        digest: hashlib._Hash = hashlib.sha256()
        with file_path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_recovery_recipe(
        self,
        base_checkpoint_path: Path,
        base_checkpoint_sha256: str,
        recovered_checkpoint_path: Path,
        recovered_sha256: str,
        pruned_paths: list[str],
        sparsity_at_mask: float,
        final_sparsity: float,
        coverage: float
    ) -> None:
        # Writes the recovery recipe: artifact lineage (both checkpoint
        # hashes), the arm configuration, the mask inventory, the sparsity
        # trajectory, and the run identity. Recording both hashes is what makes
        # the artifact lineage verifiable from the capsule alone: a reader can
        # confirm which baseline bytes were fine-tuned and which recovered bytes
        # resulted, without consulting any external record.
        #
        # Args:
        #     base_checkpoint_path: The baseline the arm started from.
        #     base_checkpoint_sha256: Content hash of that baseline.
        #     recovered_checkpoint_path: The baked artifact this arm wrote.
        #     recovered_sha256: Content hash of that artifact.
        #     pruned_paths: The mask inventory, empty for the dense
        #         continuation arm.
        #     sparsity_at_mask: Sparsity measured immediately after masking
        #         and before any recovery, zero for the dense arm.
        #     final_sparsity: Sparsity measured after the fit and after
        #         baking.
        #     coverage: Parameter fraction the arm's scope covers, zero for
        #         the dense arm.
        recipe: dict[str, object] = {
            "variant_name": self._recovery_variant,
            "base_architecture": self._configuration.architecture_name,
            "base_checkpoint_path": str(base_checkpoint_path),
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "recovered_checkpoint_path": str(recovered_checkpoint_path),
            "recovered_checkpoint_sha256": recovered_sha256,
            "pruning_arm": self._pruning_arm,
            "masks_applied": self._is_pruned_variant(),
            "technique_configuration": self._arm_configuration_dump(),
            "pruned_module_paths": pruned_paths,
            "sparsity_at_mask": sparsity_at_mask,
            "final_sparsity": final_sparsity,
            "covered_parameter_fraction": coverage,
            "recovery_epoch_count": self._configuration.train_epoch_count,
            "recovery_learning_rate_scale": 0.1,
            "hardware_name": self._configuration.artifact_layout.hardware_name,
            "run_id": self._configuration.run_id,
            "seed": self._configuration.seed,
            "hypothesis_id": self._configuration.optimization_hypothesis_id,
            "code_commit_hash": self._configuration.code_commit_hash
        }
        recipe_path: Path = self._configuration.run_directory / "recovery_recipe.yaml"
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        recipe_path.write_text(
            yaml.safe_dump(recipe, sort_keys=False, default_flow_style=False),
            encoding="utf-8"
        )

    def _build_experiment_logger(self) -> ExperimentLogger:
        # Constructs the harness-facing experiment logger under the recovery
        # variant label, keeping recovery-training rows distinguishable from
        # evaluation rows.
        return ExperimentLogger(
            run_directory=self._configuration.run_directory,
            summary_csv_path=self._configuration.summary_csv_path,
            hyperparameters_path=self._configuration.hyperparameters_path,
            architecture_name=self._configuration.architecture_name,
            variant_name=f"optimized_{self._recovery_variant}_recovery",
            seed=self._configuration.seed,
            unique_id=self._build_unique_id(),
            dataset_name="ljspeech",
            hyperparameters_summary=self._summarize_hyperparameters(),
            interpretation_notes=self._configuration.interpretation_notes
        )

    def _build_unique_id(self) -> str:
        # Composes the row identifier from architecture, recovery variant,
        # seed, and run identifier.
        return (
            f"{self._configuration.architecture_name}_"
            f"{self._recovery_variant}_recovery_"
            f"seed{self._configuration.seed}_"
            f"{self._configuration.run_id}"
        )

    def _summarize_hyperparameters(self) -> str:
        # Summarizes the core run hyperparameters written to experiment logs.
        return (
            f"architecture={self._configuration.architecture_name};"
            f"variant={self._recovery_variant};"
            f"stage=recovery_train;"
            f"recovery_epoch_count={self._configuration.train_epoch_count};"
            f"recovery_learning_rate_scale=0.1;"
            f"masks_applied={self._is_pruned_variant()};"
            f"pruning_arm={self._pruning_arm};"
            f"pruning={json.dumps(self._arm_configuration_dump(), sort_keys=True)};"
            f"seed={self._configuration.seed}"
        )

    def _build_hyperparameter_dump(self, base_checkpoint_path: Path) -> dict[str, object]:
        # Assembles the run-provenance record including the arm
        # configuration and recovery budget, sufficient to reconstruct the
        # exact invocation from the artifact alone.
        return {
            "experiment_name": self._configuration.experiment_name,
            "run_id": self._configuration.run_id,
            "evidence_category": self._configuration.evidence_category,
            "stage": self._configuration.stage,
            "architecture_name": self._configuration.architecture_name,
            "variant_name": f"optimized_{self._recovery_variant}_recovery",
            "optimization_variant_name": self._recovery_variant,
            "base_checkpoint_path": str(base_checkpoint_path),
            "pruning_arm": self._pruning_arm,
            "masks_applied": self._is_pruned_variant(),
            "technique_configuration": self._arm_configuration_dump(),
            "recovery_epoch_count": self._configuration.train_epoch_count,
            "seed": self._configuration.seed,
            "hypothesis": self._configuration.hypothesis,
            "interpretation_notes": self._configuration.interpretation_notes,
            "run_directory": str(self._configuration.run_directory),
            "data_configuration": self._configuration.data_configuration.model_dump(mode="json"),
            "metrics": list(self._configuration.metric_selection.names),
            "limit_train_batches": self._configuration.limit_train_batches,
            "limit_val_batches": self._configuration.limit_val_batches
        }

    def _write_metrics_snapshot(self, experiment_logger: ExperimentLogger) -> None:
        # Serializes the logger's reduced metric buffer to the per-run
        # metrics.json with sorted keys and allow_nan disabled.
        metrics_snapshot: dict[str, float | int] = experiment_logger.metric_buffer
        self._configuration.metrics_path.write_text(
            json.dumps(metrics_snapshot, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8"
        )
