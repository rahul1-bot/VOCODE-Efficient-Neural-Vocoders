# This module:
# 1. Evaluates one published-weights cell: an author-released checkpoint
#    loaded into the project's implementation of the architecture and
#    measured through the prediction and test stages. These rows are the
#    explicitly labelled reference anchors of the study contract; they
#    never satisfy the Study 1 project-trained reproduction requirement
# 2. Records the full retrieval provenance of the weights (author source,
#    retrieval URI, and identity fields) into the hyperparameter dump, so a
#    published-weights row is auditable back to the exact release it used
#
# Harness contract (syntheticmind):
# - The registry-built module is an ordinary harness Module; the published
#   weights are loaded onto it through the architecture's weight adapter
#   before either stage is dispatched
# - Prediction and test each receive a fresh single-pass Trainer, with
#   RealTimeFactorMonitor, MetricSequence, and RuntimeProfiler entering as
#   ordinary harness callbacks
#
# Design decisions:
# - Only the test stage is supported because published weights are an
#   external reference point; training on top of them would contaminate the
#   comparison against project-trained reproductions
# - Rows carry the published_ variant-name prefix so they remain
#   distinguishable from project-trained and optimized rows in the shared
#   CSV
#
# Author: Rahul Sawhney

import json
from typing import ClassVar

from loguru import logger as log
from pydantic import BaseModel, ConfigDict

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.runtime_profiler import RuntimeProfiler
from syntheticmind.core.module import Module
from syntheticmind.core.trainer import Trainer
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.seed import SeedManager
from syntheticmind.utilities.types import HyperparameterDict

from vocode.configs.run import ExperimentConfiguration
from vocode.data.ljspeech_datamodule import LJSpeechDataModule
from vocode.loggers.experiment import ExperimentLogger
from vocode.metrics.rtf import RealTimeFactorMonitor
from vocode.metrics.sequence import MetricSequence
from vocode.models.registry import ArchitectureModuleSpec, ModelRegistry, ModuleBuildOptions
from vocode.models.vocoder import ArchitectureName, PublishedWeightProvenance, PublishedWeights

__all__: list[str] = ["PublishedWeightsEvaluator"]


class PublishedWeightsModuleSpec(BaseModel):
    # Frozen binding of one evaluation subject: the loaded module, the
    # provenance record of the weights it carries, and the architecture
    # configuration dump recorded alongside the results. Freezing the spec is
    # what guarantees that the release named in the evidence is the release
    # loaded into the measured module, because the binding cannot be
    # substituted after construction.
    #
    # Fields:
    #     provenance: The release-identity record of the loaded weights,
    #         carrying the author source, the retrieval source, the expected
    #         content hash, and the mapping the adapter applied.
    #     module: The harness Module the author weights were loaded into,
    #         which is the object both stages measure; arbitrary types are
    #         permitted on this model so the live module can be bound
    #         directly.
    #     configuration_dump: The architecture's own configuration record,
    #         copied into the hyperparameter dump so the row states which
    #         implementation the released weights were measured in.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True
    )
    provenance: PublishedWeightProvenance
    module: Module
    configuration_dump: dict[str, object]


class PublishedWeightsEvaluator:
    # Evaluator for one published-weights cell. It builds the architecture
    # through the model registry, loads the author-released weights through
    # the registered adapter, and dispatches the prediction and test stages
    # through fresh harness Trainers.
    #
    # Integration: this evaluator owns what enters the harness; the harness owns
    # the loops. Its contract with the harness has four parts.
    #
    # Trainer construction: prediction and test each receive their own
    # single-epoch Trainer carrying the run's accelerator, the experiment
    # logger, and that stage's own batch limit. No training Trainer exists on
    # this path at all, because the stage gate admits only the test stage.
    #
    # Callback policy: prediction places the real-time-factor monitor ahead of
    # the optional runtime profiler so its batch hooks bracket the synthesis
    # calls it times, and omits the monitor entirely when the metric panel does
    # not request the timing metric; test places the objective metric panel
    # first. Neither stage carries a checkpoint, an early-stopping, or a weight-
    # averaging callback, because none of those has meaning without training.
    #
    # Precision mapping: no artifact precision label is mapped on this path.
    # Both stages take the harness default precision, since a published release
    # is measured as the author published it rather than under a chosen
    # training-precision lane.
    #
    # Checkpoint boundaries: none. This path writes no checkpoint and no
    # manifest, because the weights it measures already exist as an external
    # release; its artifacts are the experiment row, the hyperparameter dump
    # carrying the full retrieval provenance, and the metrics snapshot.
    #
    # Responsibility rule: one evaluator instance handles exactly one cell. It
    # owns its own model registry and its own row counter and shares no state
    # with any other evaluator, so several published rows may be produced in one
    # process without their evidence interfering.
    def __init__(self, configuration: ExperimentConfiguration) -> None:
        # Binds the validated experiment configuration, opens the model
        # registry, and zeroes the written-row counter. Construction creates no
        # artifact directory, retrieves no weights, and writes no row.
        #
        # Args:
        #     configuration: The validated run description naming the cell's
        #         architecture, seed, artifact layout, published-weights
        #         root, and batch limits.
        self._configuration: ExperimentConfiguration = configuration
        self._registry: ModelRegistry = ModelRegistry()
        self._row_count: int = 0

    def run(self) -> int:
        # Executes the cell: stage gate, seeding, module construction with
        # published-weight loading, then prediction and test dispatch
        # followed by the hyperparameter dump, the CSV flush, and the
        # metrics snapshot. The stage gate fires before seeding, before any
        # weight retrieval, and before any artifact directory exists, so a
        # refused run leaves nothing behind. Prediction precedes test because
        # the timing measurement belongs to the synthesis pass.
        #
        # Raises:
        #     MisconfigurationError: If the run declares any stage other than
        #         test, since training or validating on an author release
        #         would contaminate the reference anchor.
        #
        # Returns:
        #     The number of experiment result rows this evaluator has
        #     written, which is one after a completed cell.
        if self._configuration.stage != "test":
            raise MisconfigurationError(
                "Published-weight evaluation supports only stage='test'."
            )
        architecture_name: ArchitectureName = self._configuration.architecture_name
        SeedManager.seed_everything(self._configuration.seed)
        module_spec: PublishedWeightsModuleSpec = self._prepare_module_spec(architecture_name)
        datamodule: LJSpeechDataModule = LJSpeechDataModule(self._configuration.data_configuration)
        experiment_logger: ExperimentLogger = self._build_experiment_logger(module_spec)
        self._dispatch_prediction(module_spec.module, datamodule, experiment_logger)
        self._dispatch_test(module_spec.module, datamodule, experiment_logger)
        experiment_logger.log_hyperparams(self._build_hyperparameter_dump(module_spec))
        experiment_logger.flush_to_experiments_csv()
        self._row_count: int = self._row_count + 1
        self._write_metrics_snapshot(experiment_logger)
        log.info(
            f"Published weights row {self._row_count} written for "
            f"{module_spec.provenance.architecture_name} seed={self._configuration.seed}"
        )
        return self._row_count

    @property
    def row_count(self) -> int:
        # Returns the number of experiment result rows written by this
        # evaluator.
        return self._row_count

    def _prepare_module_spec(
        self,
        architecture_name: ArchitectureName
    ) -> PublishedWeightsModuleSpec:
        # Builds the architecture module, then loads the registered published
        # weights onto it from the local weights root; the adapter verifies
        # the release identity and mapping, and its provenance record is
        # frozen into the spec for evidence writing. The weights are read from
        # a local root rather than fetched, so a run measures a retained binary
        # whose hash the adapter checks against the registered expectation.
        #
        # Args:
        #     architecture_name: The architecture whose registered release is
        #         evaluated.
        #
        # Returns:
        #     The frozen evaluation subject binding the loaded module, its
        #     release provenance, and the architecture configuration dump.
        build_options: ModuleBuildOptions = ModuleBuildOptions()
        architecture_module_spec: ArchitectureModuleSpec = self._registry.build_module_spec(
            architecture_name,
            build_options
        )
        published_weights: PublishedWeights = self._registry.build_published_weights(
            architecture_name
        )
        published_weights.load(
            architecture_module_spec.module,
            self._configuration.published_weights_root
        )
        provenance: PublishedWeightProvenance = published_weights.provenance
        return PublishedWeightsModuleSpec(
            provenance=provenance,
            module=architecture_module_spec.module,
            configuration_dump=dict(architecture_module_spec.configuration_dump)
        )

    def _build_experiment_logger(
        self,
        module_spec: PublishedWeightsModuleSpec
    ) -> ExperimentLogger:
        # Constructs the harness-facing experiment logger with the published_
        # variant-name prefix and this cell's identity columns.
        return ExperimentLogger(
            run_directory=self._configuration.run_directory,
            summary_csv_path=self._configuration.summary_csv_path,
            hyperparameters_path=self._configuration.hyperparameters_path,
            architecture_name=module_spec.provenance.architecture_name,
            variant_name=f"published_{module_spec.provenance.variant_name}",
            seed=self._configuration.seed,
            unique_id=self._build_unique_id(module_spec),
            dataset_name="ljspeech",
            hyperparameters_summary=self._summarize_hyperparameters(module_spec),
            interpretation_notes=self._configuration.interpretation_notes
        )

    def _build_unique_id(self, module_spec: PublishedWeightsModuleSpec) -> str:
        # Composes the row identifier from architecture, evidence category,
        # stage, seed, and run identifier.
        return (
            f"{module_spec.provenance.architecture_name}_"
            f"{self._configuration.evidence_category}_"
            f"{self._configuration.stage}_"
            f"seed{self._configuration.seed}_"
            f"{self._configuration.run_id}"
        )

    def _summarize_hyperparameters(self, module_spec: PublishedWeightsModuleSpec) -> str:
        # Summarizes the core run facts, including both the author source and
        # the retrieval source of the weights, for the CSV summary column.
        return (
            f"architecture={module_spec.provenance.architecture_name};"
            f"variant={module_spec.provenance.variant_name};"
            f"stage={self._configuration.stage};"
            f"dataset_split={self._configuration.dataset_split_name};"
            f"author_source={module_spec.provenance.author_source_uri};"
            f"retrieval_source={module_spec.provenance.retrieval_uri};"
            f"seed={self._configuration.seed};"
            f"limit_test_batches={self._configuration.limit_test_batches};"
            f"limit_predict_batches={self._configuration.limit_predict_batches};"
            f"metrics={','.join(self._configuration.metric_selection.names)};"
            f"runtime_profiling_enabled={self._configuration.runtime_profiling_enabled};"
            f"runtime_profile_interval_steps={self._configuration.runtime_profile_interval_steps}"
        )

    def _build_hyperparameter_dump(
        self,
        module_spec: PublishedWeightsModuleSpec
    ) -> HyperparameterDict:
        # Assembles the full run-provenance record, including the complete
        # published-weight provenance dump, sufficient to reconstruct the
        # exact invocation and weight source from the artifact alone.
        return {
            "experiment_name": self._configuration.experiment_name,
            "run_id": self._configuration.run_id,
            "evidence_category": self._configuration.evidence_category,
            "stage": self._configuration.stage,
            "dataset_split_name": self._configuration.dataset_split_name,
            "architecture_name": module_spec.provenance.architecture_name,
            "variant_name": module_spec.provenance.variant_name,
            "author_source_uri": module_spec.provenance.author_source_uri,
            "retrieval_uri": module_spec.provenance.retrieval_uri,
            "seed": self._configuration.seed,
            "hypothesis": self._configuration.hypothesis,
            "interpretation_notes": self._configuration.interpretation_notes,
            "run_directory": str(self._configuration.run_directory),
            "data_configuration": self._configuration.data_configuration.model_dump(mode="json"),
            "model_configuration": dict(module_spec.configuration_dump),
            "real_time_factor_configuration": self._configuration.real_time_factor_configuration.model_dump(
                mode="json"
            ),
            "metrics": list(self._configuration.metric_selection.names),
            "limit_train_batches": self._configuration.limit_train_batches,
            "limit_val_batches": self._configuration.limit_val_batches,
            "limit_test_batches": self._configuration.limit_test_batches,
            "limit_predict_batches": self._configuration.limit_predict_batches,
            "runtime_profiling_enabled": self._configuration.runtime_profiling_enabled,
            "runtime_profile_interval_steps": self._configuration.runtime_profile_interval_steps,
            "published_weight_provenance": module_spec.provenance.model_dump(mode="json")
        }

    def _build_runtime_profiler_callbacks(self) -> list[Callback]:
        # Builds the optional harness runtime-telemetry callback when
        # profiling is enabled for the run.
        if not self._configuration.runtime_profiling_enabled:
            return []
        callbacks: list[Callback] = [
            RuntimeProfiler(
                profile_every_n_steps=self._configuration.runtime_profile_interval_steps
            )
        ]
        return callbacks

    def _write_metrics_snapshot(self, experiment_logger: ExperimentLogger) -> None:
        # Serializes the logger's reduced metric buffer to the per-run
        # metrics.json with sorted keys; allow_nan is disabled so a
        # non-finite metric aborts the run instead of entering the evidence.
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
        # Runs the harness prediction loop for synthesis-time measurement;
        # when the metric selection requests rtf, the real-time-factor
        # monitor is inserted ahead of the profiler so its batch hooks
        # bracket the synthesis calls it times.
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
        # computes the configured objective metric panel over the test split.
        # The panel evaluates complete utterances and truncates reference and
        # candidate to their common true length before scoring, so a published
        # row carries true-length values measured under the identical protocol
        # as the project-trained rows it anchors.
        test_callbacks: list[Callback] = [
            MetricSequence(self._configuration.metric_selection),
            *self._build_runtime_profiler_callbacks()
        ]
        test_trainer: Trainer = Trainer(
            max_epochs=1,
            callbacks=test_callbacks,
            logger=experiment_logger,
            accelerator=self._configuration.accelerator,
            limit_test_batches=self._configuration.limit_test_batches
        )
        test_trainer.test(module, datamodule)
