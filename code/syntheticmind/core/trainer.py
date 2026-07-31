# This module:
# 1. Provides the Trainer, the harness's central orchestrator: it validates run
#    configuration, resolves the accelerator, strategy, and device, constructs
#    and connects the execution loops, and drives the fit, validate, test, and
#    predict entry points
# 2. Builds optimizers and schedulers from the module's
#    OptimizationConfiguration, including total-step resolution for
#    step-interval schedules
# 3. Manages distributed-sampler wrapping, sanity validation, checkpoint
#    restoration, exception handling, and stage teardown
#
# Design decisions:
# - Every entry point follows one skeleton: bind the module and datamodule,
#   prepare data on rank zero behind a barrier, set up per-process state,
#   resolve and bind the device, execute the stage loop inside a try block, and
#   perform teardown in a finally block, so end hooks, component teardown, and
#   logger finalization execute even when the stage fails
# - prepare_data runs only on rank zero with a barrier afterward, so single-copy
#   work such as dataset downloads cannot race across processes
# - The automatic-optimization contract is deliberately restrictive: multiple
#   optimizers require manual optimization, scheduler definitions must be
#   validated configurations so the loop can own stepping intervals, and manual
#   optimization excludes gradient accumulation and automatic clipping; each
#   violation raises a configuration error carrying corrective instructions
# - Prediction wraps its dataloader with an unrepeated distributed sampler that
#   partitions the dataset without padding shards to equal length, because
#   padding duplicates would corrupt prediction outputs, whereas in training
#   they merely repeat work
# - Step-interval scheduler horizons are computed before the first epoch from
#   the limited batch count and the accumulation factor, so schedules span the
#   true optimizer-step budget rather than the raw batch count
# - Sanity validation executes a bounded validation pass before training and
#   then discards its epoch metrics, resets the module's logging buffer, and
#   clears the callback metrics, so sanity results can never leak into real
#   monitoring or checkpoint decisions
# - Stateful callbacks must expose unique state keys, validated before fitting,
#   so two instances of one callback class cannot overwrite each other's
#   checkpoint state
#
# Author: Rahul Sawhney

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import torch
from loguru import logger as log
from torch.utils.data import DistributedSampler, IterableDataset

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.accelerators.cpu import CPUAccelerator
from syntheticmind.accelerators.cuda import CUDAAccelerator
from syntheticmind.accelerators.mps import MPSAccelerator
from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.module import Module
from syntheticmind.core.optimizer import (
    OptimizationConfiguration,
    OptimizerConfig,
    OptimizerDefinition,
    SchedulerConfig,
    SchedulerDefinition,
)
from syntheticmind.loggers.logger import Logger
from syntheticmind.loops.evaluation_loop import EvaluationLoop
from syntheticmind.loops.fit_loop import FitLoop
from syntheticmind.loops.prediction_loop import PredictionLoop
from syntheticmind.loops.training_epoch_loop import TrainingEpochLoop
from syntheticmind.state.checkpoint_state import CheckpointState
from syntheticmind.state.trainer_state import TrainerState
from syntheticmind.strategies.ddp import DDPStrategy
from syntheticmind.strategies.single_device import SingleDeviceStrategy
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.utilities.checkpoint import load_checkpoint
from syntheticmind.utilities.distributed import is_rank_zero
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.optimizers import build_optimizer
from syntheticmind.utilities.schedulers import build_scheduler
from syntheticmind.utilities.seed import SeedManager
from syntheticmind.utilities.types import (
    CheckpointDict,
    ClipAlgorithm,
    HyperparameterDict,
    ModelOutput,
    Precision,
    RunningStage,
)

__all__: list[str] = ["Trainer"]


class _UnrepeatedDistributedSampler(DistributedSampler):
    # Distributed sampler that assigns each rank a strided slice of the dataset
    # without padding shards to equal length. The standard DistributedSampler
    # repeats leading samples so every rank yields the same count, which is
    # acceptable during training but would emit duplicate predictions; this
    # sampler accepts unequal shard sizes instead.
    def __iter__(self) -> Iterator[int]:
        # Produces this rank's index sequence: the optionally shuffled full
        # index list is strided by rank so the union across ranks covers every
        # sample exactly once. Shuffling seeds from the base seed plus the
        # epoch, matching the parent sampler's epoch-dependent permutation.
        if self.shuffle:
            g: torch.Generator = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices: list[int] = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices: list[int] = list(range(len(self.dataset)))
        indices: list[int] = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        # Length of this rank's shard under the strided partition, which may
        # differ across ranks by one when the dataset size is not divisible by
        # the replica count.
        return max(0, math.ceil((len(self.dataset) - self.rank) / self.num_replicas))


class Trainer:
    # Central orchestrator of the harness. Construction validates the run
    # configuration, seeds the random sources when requested, resolves the
    # accelerator and strategy, and builds the four connected loops; the
    # fit, validate, test, and predict entry points then execute complete
    # stages against a module and datamodule pair. Collaborator state bound
    # during a stage (dataloaders, optimizers, schedulers, metrics) lives on
    # the trainer so loops and callbacks share one authoritative view.
    #
    # Integration: a client constructs one Trainer per run and passes its
    # syntheticmind.core.module.Module subclass together with a
    # syntheticmind.core.datamodule.DataModule to an entry point.
    #
    # Example::
    #
    #     from syntheticmind.callbacks.early_stopping import EarlyStopping
    #     from syntheticmind.callbacks.model_checkpoint import ModelCheckpoint
    #     from syntheticmind.core.trainer import Trainer
    #
    #     model: AutoEncoder = AutoEncoder()
    #     datamodule: SpeechDataModule = SpeechDataModule(corpus_root)
    #     trainer: Trainer = Trainer(
    #         max_epochs=100,
    #         callbacks=[
    #             ModelCheckpoint(monitor="val_loss", mode="min"),
    #             EarlyStopping(monitor="val_loss", mode="min", patience=10)
    #         ],
    #         precision="16-mixed",
    #         seed=42
    #     )
    #     trainer.fit(model, datamodule)
    #     test_metrics: dict[str, float] = trainer.test(model, datamodule)
    def __init__(
        self,
        max_epochs: int = 100,
        accelerator: str | Accelerator = "auto",
        strategy: str | Strategy = "auto",
        devices: int = 1,
        callbacks: list[Callback] | None = None,
        logger: Logger | None = None,
        accumulate_grad_batches: int = 1,
        val_check_interval: int | float = 1.0,
        val_check_interval_scope: Literal["epoch", "global_step"] = "epoch",
        validate_at_epoch_end: bool = True,
        precision: Precision = "32-true",
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: ClipAlgorithm = "norm",
        log_every_n_steps: int = 10,
        enable_progress_bar: bool = True,
        deterministic: bool = False,
        num_sanity_val_steps: int = 2,
        limit_train_batches: int | float | None = None,
        limit_val_batches: int | float | None = None,
        limit_test_batches: int | float | None = None,
        limit_predict_batches: int | float | None = None,
        use_distributed_sampler: bool = True,
        sync_batchnorm: bool = False,
        resume_from_checkpoint: str | Path | None = None,
        seed: int | None = None,
        reload_dataloaders_every_n_epochs: int = 0
    ) -> None:
        # Validates the configuration surface, optionally seeds every random
        # source, resolves the accelerator and strategy, installs the default
        # progress bar unless one is already present, constructs the four
        # loops with their limits, connects them to this trainer, and prepares
        # the empty collaborator slots that the entry points populate per run.
        #
        # Args:
        #     max_epochs: Number of complete training epochs the fit loop
        #         executes before stopping. Also enters the total-step
        #         estimate used to materialize step-interval schedulers.
        #         Default: ``100``.
        #     accelerator: Compute backend. Accepts ``"cpu"``, ``"cuda"``,
        #         ``"mps"``, ``"auto"``, or a constructed Accelerator
        #         instance; ``"auto"`` prefers CUDA, then MPS, then CPU by
        #         availability. Default: ``"auto"``.
        #     strategy: Execution strategy. Accepts ``"single_device"``,
        #         ``"ddp"``, ``"auto"``, or a constructed Strategy instance;
        #         ``"auto"`` selects distributed data parallel only for
        #         multi-device CUDA configurations and single-device
        #         execution otherwise. Default: ``"auto"``.
        #     devices: Declared device count, consulted by ``"auto"``
        #         strategy resolution; values above one on CUDA select
        #         distributed data parallel. The distributed world size and
        #         per-process device binding come from the launcher
        #         environment, not from this value. Default: ``1``.
        #     callbacks: Callbacks invoked at every hook point in list
        #         order. The list is extended with the default progress bar
        #         when enable_progress_bar is true and none is present.
        #         Default: ``None``.
        #     logger: Experiment logger receiving step-cadence metrics,
        #         reduced epoch metrics, and hyperparameters; ``None``
        #         disables experiment logging. Default: ``None``.
        #     accumulate_grad_batches: Number of consecutive batches whose
        #         gradients accumulate before one optimizer step; under
        #         distributed data parallel, gradient synchronization is
        #         suppressed on non-boundary batches. Must be at least one.
        #         Default: ``1``.
        #     val_check_interval: Mid-epoch validation cadence. An integer
        #         counts optimizer steps under the ``"global_step"`` scope
        #         and batches within the epoch under the ``"epoch"`` scope;
        #         a float in (0.0, 1.0] selects evenly spaced fractional
        #         positions of the epoch, with ``1.0`` meaning no mid-epoch
        #         validation. Default: ``1.0``.
        #     val_check_interval_scope: Unit for integer interval values:
        #         ``"epoch"`` counts batches within each epoch and excludes
        #         the final batch position, ``"global_step"`` counts
        #         optimizer steps across epoch boundaries and requires an
        #         integer interval. Default: ``"epoch"``.
        #     validate_at_epoch_end: Whether the fit loop schedules the
        #         epoch-boundary validation pass. Step-cadence training
        #         disables this so validation runs strictly on the step
        #         interval. Default: ``True``.
        #     precision: Numeric execution mode: ``"32-true"`` for full
        #         float32, ``"16-mixed"`` for float16 autocast with gradient
        #         scaling, ``"bf16-mixed"`` for bfloat16 autocast without
        #         scaling. Default: ``"32-true"``.
        #     gradient_clip_val: Clipping threshold applied to gradients
        #         before each optimizer step, after unscaling under
        #         ``"16-mixed"``; ``None`` disables clipping.
        #         Default: ``None``.
        #     gradient_clip_algorithm: Clipping method: ``"norm"`` rescales
        #         by total gradient norm, ``"value"`` clamps each component.
        #         Default: ``"norm"``.
        #     log_every_n_steps: Global-step cadence at which step-routed
        #         training metrics are emitted to the logger. Must be at
        #         least one. Default: ``10``.
        #     enable_progress_bar: Whether the default progress bar callback
        #         is appended when the callback list does not already
        #         contain one. Default: ``True``.
        #     deterministic: Forwarded to the seeding call when a seed is
        #         provided: enforces deterministic algorithm selection and
        #         disables cuDNN benchmark autotuning. Recorded on the
        #         trainer for collaborators to consult. Default: ``False``.
        #     num_sanity_val_steps: Number of validation batches run before
        #         training starts to surface evaluation defects early;
        #         every trace of the pass is discarded afterwards. Zero
        #         disables the sanity pass. Default: ``2``.
        #     limit_train_batches: Ceiling on training batches per epoch.
        #         An integer is an absolute count, a float is a fraction of
        #         the dataloader length (at least one batch), ``None`` runs
        #         the full loader. Fractions below one require a sized
        #         loader. Default: ``None``.
        #     limit_val_batches: Ceiling on validation batches per pass,
        #         with the same integer, float, and ``None`` semantics.
        #         Default: ``None``.
        #     limit_test_batches: Ceiling on test batches per pass, with
        #         the same semantics. Default: ``None``.
        #     limit_predict_batches: Ceiling on prediction batches per
        #         pass, with the same semantics. Default: ``None``.
        #     use_distributed_sampler: Whether distributed runs wrap sized
        #         map-style dataloaders with DistributedSampler, shuffled
        #         for training and unshuffled for evaluation; prediction
        #         uses the unrepeated sampler so no rank emits padded
        #         duplicate outputs. Iterable datasets are never wrapped.
        #         Default: ``True``.
        #     sync_batchnorm: Whether batch-normalization layers are
        #         converted to synchronized batch normalization before
        #         distributed training. Default: ``False``.
        #     resume_from_checkpoint: Default checkpoint path restored at
        #         the start of fit; the ckpt_path argument of fit overrides
        #         it per call. Default: ``None``.
        #     seed: When provided, seeds Python, NumPy, and torch random
        #         sources at construction through the seed manager.
        #         Default: ``None``.
        #     reload_dataloaders_every_n_epochs: Rebuild cadence for the
        #         training and validation dataloaders; epochs whose index
        #         is a positive multiple of this value request fresh
        #         loaders from the datamodule. Zero keeps the initial
        #         loaders for the whole run. Default: ``0``.
        #
        # Raises:
        #     MisconfigurationError: If accumulate_grad_batches is below
        #         one, an integer val_check_interval is below one, a float
        #         val_check_interval is outside (0.0, 1.0], the interval
        #         scope is invalid, the ``"global_step"`` scope receives a
        #         non-integer interval, or log_every_n_steps is below one.
        #     ValueError: If the accelerator or strategy name is unknown.
        self.max_epochs: int = max_epochs
        self.devices: int = devices
        self.precision: Precision = precision
        self.gradient_clip_val: float | None = gradient_clip_val
        self.gradient_clip_algorithm: ClipAlgorithm = gradient_clip_algorithm
        self.log_every_n_steps: int = log_every_n_steps
        self.num_sanity_val_steps: int = num_sanity_val_steps
        self.use_distributed_sampler: bool = use_distributed_sampler
        self.sync_batchnorm: bool = sync_batchnorm
        self.deterministic: bool = deterministic
        self.resume_from_checkpoint: Path | None = (
            Path(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )

        if accumulate_grad_batches < 1:
            raise MisconfigurationError("accumulate_grad_batches must be >= 1")
        if isinstance(val_check_interval, int) and val_check_interval < 1:
            raise MisconfigurationError("val_check_interval (int) must be >= 1")
        if isinstance(val_check_interval, float) and not (0.0 < val_check_interval <= 1.0):
            raise MisconfigurationError("val_check_interval (float) must be in (0.0, 1.0]")
        if val_check_interval_scope not in {"epoch", "global_step"}:
            raise MisconfigurationError("val_check_interval_scope must be 'epoch' or 'global_step'")
        if val_check_interval_scope == "global_step" and not isinstance(val_check_interval, int):
            raise MisconfigurationError("global_step validation intervals require integer val_check_interval")
        if log_every_n_steps < 1:
            raise MisconfigurationError("log_every_n_steps must be >= 1")

        if seed is not None:
            SeedManager.seed_everything(seed, deterministic=deterministic)

        self.accelerator: Accelerator = self._resolve_accelerator(accelerator)
        self.strategy: Strategy = self._resolve_strategy(strategy, self.accelerator)

        self.callbacks: list[Callback] = callbacks if callbacks is not None else []
        if enable_progress_bar:
            from syntheticmind.callbacks.progress_bar import ProgressBarCallback

            if not any(isinstance(c, ProgressBarCallback) for c in self.callbacks):
                self.callbacks.append(ProgressBarCallback())

        self.logger: Logger | None = logger
        self.state: TrainerState = TrainerState()
        self._sanity_checking: bool = False
        self._is_mid_epoch_validation: bool = False

        self.fit_loop: FitLoop = FitLoop(max_epochs=max_epochs)
        self.fit_loop.epoch_loop: TrainingEpochLoop | None = TrainingEpochLoop(
            accumulate_grad_batches=accumulate_grad_batches,
            val_check_interval=val_check_interval,
            val_check_interval_scope=val_check_interval_scope,
            validate_at_epoch_end=validate_at_epoch_end,
            precision=precision,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            log_every_n_steps=log_every_n_steps,
            limit_train_batches=limit_train_batches
        )
        self.validate_loop: EvaluationLoop = EvaluationLoop(
            stage="validate", limit_batches=limit_val_batches
        )
        self.test_loop: EvaluationLoop = EvaluationLoop(
            stage="test", limit_batches=limit_test_batches
        )
        self.predict_loop: PredictionLoop = PredictionLoop(
            limit_batches=limit_predict_batches
        )

        self._connect_loops()

        self.module: Module | None = None
        self.datamodule: DataModule | None = None
        self.optimizers: list[torch.optim.Optimizer] = []
        self.schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
        self.scheduler_configs: list[SchedulerConfig | None] = []
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.scheduler_config: SchedulerConfig | None = None
        self.train_dataloader: torch.utils.data.DataLoader | None = None
        self.val_dataloader: torch.utils.data.DataLoader | None = None
        self.test_dataloader: torch.utils.data.DataLoader | None = None
        self.predict_dataloader: torch.utils.data.DataLoader | None = None
        self._total_train_batches: int | None = None
        self._total_val_batches: int | None = None
        self._total_test_batches: int | None = None
        self.reload_dataloaders_every_n_epochs: int = reload_dataloaders_every_n_epochs
        self._callback_metrics: dict[str, float] = {}
        self._dataloader_fetch_time_seconds: dict[RunningStage, float] = {}
        self._dataloader_iterator_creation_time_seconds: dict[RunningStage, float] = {}

    def fit(
        self,
        module: Module,
        datamodule: DataModule,
        ckpt_path: str | Path | None = None
    ) -> None:
        # Executes a complete training run. The sequence is: reset run-local
        # state, bind the module and datamodule, prepare and set up the data,
        # wrap dataloaders for distributed execution, resolve and bind the
        # device, build optimizers and schedulers from the module's
        # optimization configuration under the automatic-optimization
        # contract, optionally restore a checkpoint, fire the fit-start and
        # train-start hook pairs around an optional sanity validation, run the
        # fit loop, and unwind through the end hooks, teardown, and logger
        # finalization in the finally block. An explicit ckpt_path overrides
        # the constructor-level resume path.
        #
        # Args:
        #     module: The model to train. The trainer attaches itself to
        #         the module, materializes its configure_optimizers
        #         declaration into live optimizers and schedulers, and
        #         drives its step methods through the loops.
        #     datamodule: Provider of the training and validation
        #         dataloaders through prepare_data, setup, and the
        #         dataloader hooks; each hook must return a single
        #         DataLoader.
        #     ckpt_path: Checkpoint to restore before training starts;
        #         overrides the constructor-level resume path for this
        #         call. ``None`` falls back to that path or a fresh start.
        #         Default: ``None``.
        # Clear every run-local surface before anything executes, so a second
        # fit call on the same trainer inherits no stop request, cached
        # callback metrics, progress counters, or mid-epoch resume markers
        # from an earlier run.
        self.state.set_stage("fit")
        self.state.reset_on_run()
        self._callback_metrics.clear()
        self.fit_loop.should_stop: bool = False
        if self.fit_loop.epoch_loop is not None:
            self.fit_loop.epoch_loop.reset_on_run()
        # Bind the run participants and zero the module's progress mirrors.
        # The trainer back-reference is what powers module.log, the optimizer
        # and scheduler accessors, and gradient clipping inside the step
        # methods.
        self.module: Module | None = module
        self.datamodule: DataModule | None = datamodule
        module._trainer: Trainer | None = self
        module._current_epoch: int = 0
        module._global_step: int = 0

        # An explicit per-call checkpoint path overrides the constructor-level
        # resume path; with both absent the run starts fresh.
        resume_path: Path | None = (
            Path(ckpt_path) if ckpt_path is not None else self.resume_from_checkpoint
        )

        try:
            # Initialize the strategy's process context, run single-copy data
            # preparation on global rank zero only, and hold every rank at the
            # barrier so no process enters setup against partially prepared
            # data.
            self.strategy.setup_environment()

            if is_rank_zero():
                datamodule.prepare_data()
            self.strategy.barrier()

            datamodule.setup("fit")

            # Retrieve the fit-stage dataloaders. The harness supports exactly
            # one loader per stage, so collection returns are rejected here
            # rather than failing obscurely inside the loops.
            self.train_dataloader: torch.utils.data.DataLoader | None = datamodule.train_dataloader()

            if isinstance(self.train_dataloader, (list, tuple, dict)):
                raise MisconfigurationError(
                    "Multiple train dataloaders are not supported. train_dataloader() must return a single DataLoader."
                )

            self.val_dataloader: torch.utils.data.DataLoader | None = datamodule.val_dataloader()

            if isinstance(self.val_dataloader, (list, tuple, dict)):
                raise MisconfigurationError(
                    "Multiple val dataloaders are not supported. val_dataloader() must return a single DataLoader."
                )

            # Under a distributed strategy, map-style datasets are re-wrapped
            # with rank-aware samplers so every process sees a disjoint shard;
            # iterable datasets own their sharding and pass through with a
            # warning.
            if self.use_distributed_sampler and self.strategy.is_distributed:
                if not isinstance(self.train_dataloader.dataset, IterableDataset):
                    self.train_dataloader: torch.utils.data.DataLoader | None = self._wrap_with_distributed_sampler(self.train_dataloader, shuffle=True)
                else:
                    log.warning("Skipping DistributedSampler for train dataloader: IterableDataset detected.")
                if self.val_dataloader is not None:
                    if not isinstance(self.val_dataloader.dataset, IterableDataset):
                        self.val_dataloader: torch.utils.data.DataLoader | None = self._wrap_with_distributed_sampler(self.val_dataloader, shuffle=False)
                    else:
                        log.warning("Skipping DistributedSampler for val dataloader: IterableDataset detected.")

            # Record loader lengths where they are knowable; unsized iterable
            # loaders leave the totals as None and downstream cadence math
            # degrades to its unsized behavior.
            self._total_train_batches: int | None = self._safe_len(self.train_dataloader)
            if self.val_dataloader is not None:
                self._total_val_batches: int | None = self._safe_len(self.val_dataloader)

            # Advisory only: single-process loading feeding a non-CPU
            # accelerator is a common throughput mistake worth one warning,
            # never an error.
            if is_rank_zero():
                if hasattr(self.train_dataloader, 'num_workers') and self.train_dataloader.num_workers == 0:
                    if not isinstance(self.accelerator, CPUAccelerator):
                        log.warning("num_workers=0 with GPU accelerator. Consider increasing for better throughput.")

            # Resolve the concrete device, hand it to the accelerator, and let
            # the strategy place or wrap the module on it. BatchNorm
            # conversion precedes strategy setup so a distributed wrapper is
            # built around the converted layers, and the module's device
            # mirror is stamped once placement is final.
            device: torch.device = self._resolve_device()
            self.accelerator.setup_device(device)

            if self.sync_batchnorm and self.strategy.is_distributed:
                module: Module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)

            self.strategy.setup(module, device)
            module._device: torch.device = device

            # Materialize the module's declarative optimization contract. The
            # declaration is normalized to parallel definition lists first, so
            # single-optimizer and multi-optimizer modules flow through one
            # build path.
            optimization_configuration: OptimizationConfiguration = module.configure_optimizers()
            optimizer_config_value: OptimizerDefinition | list[OptimizerDefinition] = (
                optimization_configuration.optimizer
            )
            scheduler_config_value: SchedulerDefinition | list[SchedulerDefinition] | None = (
                optimization_configuration.scheduler
            )
            if isinstance(optimizer_config_value, list):
                optimizer_definitions: list[OptimizerDefinition] = optimizer_config_value
            else:
                optimizer_definitions: list[OptimizerDefinition] = [optimizer_config_value]
            if scheduler_config_value is None:
                scheduler_definitions: list[SchedulerDefinition] = []
            elif isinstance(scheduler_config_value, list):
                scheduler_definitions: list[SchedulerDefinition] = scheduler_config_value
            else:
                scheduler_definitions: list[SchedulerDefinition] = [scheduler_config_value]

            # OptimizerConfig records are compiled into live torch.optim
            # objects against the module's parameters; pre-built
            # torch.optim.Optimizer instances pass through untouched.
            self.optimizers: list[torch.optim.Optimizer] = [
                build_optimizer(optimizer_definition, module)
                if isinstance(optimizer_definition, OptimizerConfig)
                else optimizer_definition
                for optimizer_definition in optimizer_definitions
            ]
            self.scheduler_configs: list[SchedulerConfig | None] = [
                scheduler_definition
                for scheduler_definition in scheduler_definitions
                if isinstance(scheduler_definition, SchedulerConfig)
            ]

            # Contract guards, enforced before any hook can observe the run:
            # multiple optimizers demand manual optimization; automatic
            # optimization demands SchedulerConfig records so the loop owns
            # stepping cadence; manual optimization excludes trainer-managed
            # accumulation and clipping because the module owns its backward
            # path.
            if len(self.optimizers) > 1 and module.automatic_optimization:
                raise MisconfigurationError(
                    "Training with multiple optimizers requires manual optimization. "
                    "Set module.automatic_optimization = False before fitting."
                )
            if module.automatic_optimization and any(
                not isinstance(scheduler_definition, SchedulerConfig)
                for scheduler_definition in scheduler_definitions
            ):
                raise MisconfigurationError(
                    "Automatic optimization requires SchedulerConfig definitions so the loop can manage "
                    "scheduler stepping intervals. Provide SchedulerConfig objects or switch to "
                    "automatic_optimization = False and step schedulers manually."
                )
            if not module.automatic_optimization:
                assert self.fit_loop.epoch_loop is not None
                if self.fit_loop.epoch_loop.accumulate_grad_batches != 1:
                    raise MisconfigurationError(
                        "Manual optimization does not support accumulate_grad_batches != 1. "
                        "Set accumulate_grad_batches = 1 and manage accumulation inside training_step if needed."
                    )
                if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
                    raise MisconfigurationError(
                        "Manual optimization does not support automatic gradient clipping. "
                        "Call module.clip_gradients(...) explicitly inside training_step if needed."
                    )

            # The first optimizer doubles as the primary alias consumed by the
            # automatic-optimization path; the scheduler surfaces start empty
            # and are filled only when definitions exist.
            self.optimizer: torch.optim.Optimizer | None = self.optimizers[0] if self.optimizers else None
            self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
            self.scheduler_config: SchedulerConfig | None = None
            self.schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []

            # Schedulers pair positionally with optimizers, one definition per
            # optimizer. SchedulerConfig records are compiled against the
            # resolved total-step horizon; pre-built scheduler instances pass
            # through with no managed config, leaving their stepping to the
            # module.
            if scheduler_definitions:
                if len(scheduler_definitions) != len(self.optimizers):
                    raise MisconfigurationError(
                        "The number of scheduler definitions must match the number of optimizers."
                    )
                config_scheduler_count: int = sum(
                    1 for scheduler_definition in scheduler_definitions
                    if isinstance(scheduler_definition, SchedulerConfig)
                )
                total_scheduler_steps: int | None = None
                if config_scheduler_count > 0:
                    total_scheduler_steps: int | None = self._resolve_total_scheduler_steps(
                        scheduler_definitions=scheduler_definitions
                    )

                built_schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
                built_scheduler_configs: list[SchedulerConfig | None] = []
                for scheduler_definition, optimizer in zip(scheduler_definitions, self.optimizers, strict=True):
                    if isinstance(scheduler_definition, SchedulerConfig):
                        assert total_scheduler_steps is not None
                        built_schedulers.append(
                            build_scheduler(scheduler_definition, optimizer, total_scheduler_steps)
                        )
                        built_scheduler_configs.append(scheduler_definition)
                    else:
                        built_schedulers.append(scheduler_definition)
                        built_scheduler_configs.append(None)

                self.schedulers: list[torch.optim.lr_scheduler.LRScheduler] = built_schedulers
                self.scheduler_configs: list[SchedulerConfig | None] = built_scheduler_configs
                self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = self.schedulers[0]
                self.scheduler_config: SchedulerConfig | None = (
                    self.scheduler_configs[0] if self.scheduler_configs else None
                )

            # Checkpoint restoration happens after optimizers and schedulers
            # exist, because their serialized state must land on live objects,
            # and before any hook fires, so every hook observes the resumed
            # run.
            if resume_path is not None:
                self._restore_checkpoint(resume_path, module)

            # Duplicate stateful-callback keys are rejected before callback
            # setup so two instances cannot silently overwrite each other's
            # checkpoint state later.
            self._validate_callback_state_keys()

            for callback in self.callbacks:
                callback.setup(self, module, "fit")

            # The run configuration is announced to the logger from rank zero
            # only, keeping multi-process runs single-writer.
            if self.logger is not None and is_rank_zero():
                config_dict: HyperparameterDict = {
                    "max_epochs": self.max_epochs,
                    "precision": self.precision,
                    "accelerator": type(self.accelerator).__name__,
                    "strategy": type(self.strategy).__name__
                }
                self.logger.log_hyperparams(config_dict)

            # Start hooks fire callbacks-first, then the module. The optional
            # sanity pass runs inside the fit stage but before training
            # starts, surfacing validation-path defects before the first
            # training epoch is paid for; it discards every trace of itself.
            for callback in self.callbacks:
                callback.on_fit_start(self, module)
            module.on_fit_start()

            if self.val_dataloader is not None and self.num_sanity_val_steps > 0:
                self._sanity_check_validation(module)

            # Enter the training running-stage and hand control to the fit
            # loop, which owns epochs, optimization, cadence-driven
            # validation, and stop requests from here.
            self.state.set_running_stage("training")
            module._running_stage: str | None = "training"

            for callback in self.callbacks:
                callback.on_train_start(self, module)
            module.on_train_start()

            self.fit_loop.run()

            # End hooks mirror the start order reversed: the module first,
            # then the callbacks.
            module.on_train_end()
            for callback in self.callbacks:
                callback.on_train_end(self, module)
        # Interrupts and failures funnel through one handler that marks the
        # trainer state interrupted, offers the exception to the callback,
        # datamodule, and strategy hooks, and re-raises.
        except KeyboardInterrupt as exc:
            self._handle_exception(module, datamodule, exc, "Training")
        except Exception as exception:
            self._handle_exception(module, datamodule, exception, "Training")
        finally:
            # Unwind unconditionally, success or failure: end-of-fit hooks,
            # component teardown, and logger finalization all execute so a
            # failed run still releases its resources and flushes its records.
            module.on_fit_end()
            for callback in self.callbacks:
                callback.on_fit_end(self, module)

            for callback in self.callbacks:
                callback.teardown(self, module, "fit")
            datamodule.teardown("fit")
            self.strategy.teardown()

            if self.logger is not None:
                self.logger.finalize()

            # The module's stage markers are cleared and the epoch loop's
            # resume markers reset, so the module and loops leave fit inert
            # rather than carrying live-run state into later stages.
            module._running_stage: str | None = None
            module._current_fx_name: str | None = None
            self.state.finish()
            if self.fit_loop.epoch_loop is not None:
                self.fit_loop.epoch_loop.reset_on_run()

    def validate(
        self,
        module: Module,
        datamodule: DataModule
    ) -> dict[str, float]:
        # Executes a standalone validation stage and returns its reduced epoch
        # metrics. The module's training-mode flag is captured before the run
        # and restored in the finally block, so evaluating a model mid-workflow
        # does not permanently switch it out of training mode.
        #
        # Args:
        #     module: The model to evaluate; its validation steps run under
        #         inference mode.
        #     datamodule: Provider of the validation dataloader through
        #         prepare_data, setup, and the dataloader hook.
        #
        # Returns:
        #     The reduced epoch metrics of the validation pass, keyed by
        #     metric name.
        self.state.set_stage("validate")
        self._callback_metrics.clear()
        self.module: Module | None = module
        self.datamodule: DataModule | None = datamodule
        module._trainer: Trainer | None = self

        was_training: bool = module.training
        metrics: dict[str, float] = {}
        try:
            self.strategy.setup_environment()

            if is_rank_zero():
                datamodule.prepare_data()
            self.strategy.barrier()

            datamodule.setup("validate")
            self.val_dataloader: torch.utils.data.DataLoader | None = datamodule.val_dataloader()

            if isinstance(self.val_dataloader, (list, tuple, dict)):
                raise MisconfigurationError(
                    "Multiple val dataloaders are not supported. val_dataloader() must return a single DataLoader."
                )

            if self.use_distributed_sampler and self.strategy.is_distributed:
                if not isinstance(self.val_dataloader.dataset, IterableDataset):
                    self.val_dataloader: torch.utils.data.DataLoader | None = self._wrap_with_distributed_sampler(self.val_dataloader, shuffle=False)
                else:
                    log.warning("Skipping DistributedSampler for val dataloader: IterableDataset detected.")

            self._total_val_batches: int | None = self._safe_len(self.val_dataloader) if self.val_dataloader is not None else None

            device: torch.device = self._resolve_device()
            self.accelerator.setup_device(device)
            self.strategy.setup(module, device)
            module._device: torch.device = device

            for callback in self.callbacks:
                callback.setup(self, module, "validate")

            metrics: dict[str, float] = self.validate_loop.run()
        except KeyboardInterrupt as exc:
            self._handle_exception(module, datamodule, exc, "Validation")
        except Exception as exception:
            self._handle_exception(module, datamodule, exception, "Validation")
        finally:
            for callback in self.callbacks:
                callback.teardown(self, module, "validate")
            datamodule.teardown("validate")
            self.strategy.teardown()
            if self.logger is not None:
                self.logger.finalize()
            module._running_stage: str | None = None
            module._current_fx_name: str | None = None
            self.state.finish()
            module.train(was_training)

        return metrics

    def test(
        self,
        module: Module,
        datamodule: DataModule
    ) -> dict[str, float]:
        # Executes a standalone test stage and returns its reduced epoch
        # metrics, following the same setup, execution, and teardown skeleton
        # as validate with the test dataloader and test loop.
        #
        # Args:
        #     module: The model to evaluate; its test steps run under
        #         inference mode.
        #     datamodule: Provider of the test dataloader through
        #         prepare_data, setup, and the dataloader hook.
        #
        # Returns:
        #     The reduced epoch metrics of the test pass, keyed by metric
        #     name.
        self.state.set_stage("test")
        self._callback_metrics.clear()
        self.module: Module | None = module
        self.datamodule: DataModule | None = datamodule
        module._trainer: Trainer | None = self

        was_training: bool = module.training
        metrics: dict[str, float] = {}
        try:
            self.strategy.setup_environment()

            if is_rank_zero():
                datamodule.prepare_data()
            self.strategy.barrier()

            datamodule.setup("test")
            self.test_dataloader: torch.utils.data.DataLoader | None = datamodule.test_dataloader()

            if isinstance(self.test_dataloader, (list, tuple, dict)):
                raise MisconfigurationError(
                    "Multiple test dataloaders are not supported. test_dataloader() must return a single DataLoader."
                )

            if self.use_distributed_sampler and self.strategy.is_distributed:
                if not isinstance(self.test_dataloader.dataset, IterableDataset):
                    self.test_dataloader: torch.utils.data.DataLoader | None = self._wrap_with_distributed_sampler(self.test_dataloader, shuffle=False)
                else:
                    log.warning("Skipping DistributedSampler for test dataloader: IterableDataset detected.")

            self._total_test_batches: int | None = self._safe_len(self.test_dataloader) if self.test_dataloader is not None else None

            device: torch.device = self._resolve_device()
            self.accelerator.setup_device(device)
            self.strategy.setup(module, device)
            module._device: torch.device = device

            for callback in self.callbacks:
                callback.setup(self, module, "test")

            metrics: dict[str, float] = self.test_loop.run()
        except KeyboardInterrupt as exc:
            self._handle_exception(module, datamodule, exc, "Testing")
        except Exception as exception:
            self._handle_exception(module, datamodule, exception, "Testing")
        finally:
            for callback in self.callbacks:
                callback.teardown(self, module, "test")
            datamodule.teardown("test")
            self.strategy.teardown()
            if self.logger is not None:
                self.logger.finalize()
            module._running_stage: str | None = None
            module._current_fx_name: str | None = None
            self.state.finish()
            module.train(was_training)

        return metrics

    def predict(
        self,
        module: Module,
        datamodule: DataModule,
        return_predictions: bool = True
    ) -> list[ModelOutput]:
        # Executes a standalone prediction stage and returns the collected
        # step outputs. Distributed execution uses the unrepeated sampler so
        # no rank emits padded duplicate predictions, and the retention flag
        # is forwarded to the prediction loop before it runs.
        #
        # Args:
        #     module: The model to run inference with; its predict steps
        #         run under inference mode and metric logging is rejected
        #         by contract.
        #     datamodule: Provider of the prediction dataloader through
        #         prepare_data, setup, and the dataloader hook.
        #     return_predictions: Whether step outputs are retained and
        #         returned. Disable for measurement-only passes where
        #         retention would hold every output in memory for the whole
        #         stage. Default: ``True``.
        #
        # Returns:
        #     The prediction-step outputs in batch order, or an empty list
        #     when retention is disabled.
        self.state.set_stage("predict")
        self._callback_metrics.clear()
        self.module: Module | None = module
        self.datamodule: DataModule | None = datamodule
        module._trainer: Trainer | None = self

        was_training: bool = module.training
        predictions: list[ModelOutput] = []
        try:
            self.strategy.setup_environment()

            if is_rank_zero():
                datamodule.prepare_data()
            self.strategy.barrier()

            datamodule.setup("predict")
            self.predict_dataloader: torch.utils.data.DataLoader | None = datamodule.predict_dataloader()

            if isinstance(self.predict_dataloader, (list, tuple, dict)):
                raise MisconfigurationError(
                    "Multiple predict dataloaders are not supported. predict_dataloader() must return a single DataLoader."
                )

            if self.use_distributed_sampler and self.strategy.is_distributed:
                if not isinstance(self.predict_dataloader.dataset, IterableDataset):
                    self.predict_dataloader: torch.utils.data.DataLoader | None = self._wrap_with_unrepeated_sampler(self.predict_dataloader)
                else:
                    log.warning("Skipping DistributedSampler for predict dataloader: IterableDataset detected.")

            device: torch.device = self._resolve_device()
            self.accelerator.setup_device(device)
            self.strategy.setup(module, device)
            module._device: torch.device = device

            for callback in self.callbacks:
                callback.setup(self, module, "predict")

            self.predict_loop.return_predictions: bool = return_predictions
            predictions: list[ModelOutput] = self.predict_loop.run()
        except KeyboardInterrupt as exc:
            self._handle_exception(module, datamodule, exc, "Prediction")
        except Exception as exception:
            self._handle_exception(module, datamodule, exception, "Prediction")
        finally:
            for callback in self.callbacks:
                callback.teardown(self, module, "predict")
            datamodule.teardown("predict")
            self.strategy.teardown()
            if self.logger is not None:
                self.logger.finalize()
            module._running_stage: str | None = None
            module._current_fx_name: str | None = None
            self.state.finish()
            module.train(was_training)

        return predictions

    def _safe_len(
        self,
        dataloader: torch.utils.data.DataLoader | None
    ) -> int | None:
        # Returns the dataloader length, or None both for an absent dataloader
        # and for iterable datasets that define no length, so callers can
        # treat unknown totals uniformly.
        if dataloader is None:
            return None
        try:
            return len(dataloader)
        except TypeError:
            return None

    def _record_dataloader_fetch_time(self, running_stage: RunningStage, duration_seconds: float) -> None:
        # Records the most recent blocking dataloader fetch duration for the
        # stage; consumed by runtime diagnostics such as the profiler callback.
        self._dataloader_fetch_time_seconds[running_stage] = max(0.0, duration_seconds)

    def _get_dataloader_fetch_time(self, running_stage: RunningStage) -> float | None:
        # Returns the most recent recorded fetch duration for the stage, or
        # None when no fetch has been recorded yet.
        return self._dataloader_fetch_time_seconds.get(running_stage)

    def _record_dataloader_iterator_creation_time(
        self, running_stage: RunningStage, duration_seconds: float
    ) -> None:
        # Records how long dataloader iterator construction took for the
        # stage. Iterator creation spawns worker processes, so this duration
        # exposes worker startup cost separately from per-batch fetch time.
        self._dataloader_iterator_creation_time_seconds[running_stage] = max(0.0, duration_seconds)

    def _consume_dataloader_iterator_creation_time(self, running_stage: RunningStage) -> float | None:
        # Returns and clears the recorded iterator-construction duration for
        # the stage. Consumption is destructive so the one-time startup cost
        # cannot be attributed to more than one measurement window.
        return self._dataloader_iterator_creation_time_seconds.pop(running_stage, None)

    def _connect_loops(self) -> None:
        # Attaches this trainer to every loop, giving each loop access to the
        # shared collaborators (strategy, state, callbacks, dataloaders,
        # metrics) through its trainer property.
        self.fit_loop.trainer: Trainer = self
        assert self.fit_loop.epoch_loop is not None
        self.fit_loop.epoch_loop.trainer: Trainer = self
        self.validate_loop.trainer: Trainer = self
        self.test_loop.trainer: Trainer = self
        self.predict_loop.trainer: Trainer = self

    def _resolve_accelerator(self, accelerator: str | Accelerator) -> Accelerator:
        # Resolves the accelerator argument: a live instance passes through
        # unchanged, a backend name constructs the matching accelerator, and
        # "auto" prefers CUDA, then MPS, then CPU by availability.
        if isinstance(accelerator, Accelerator):
            return accelerator
        match accelerator:
            case "cuda":
                return CUDAAccelerator()
            case "mps":
                return MPSAccelerator()
            case "cpu":
                return CPUAccelerator()
            case "auto":
                if CUDAAccelerator.is_available():
                    return CUDAAccelerator()
                if MPSAccelerator.is_available():
                    return MPSAccelerator()
                return CPUAccelerator()
            case _:
                raise ValueError(f"Unknown accelerator: {accelerator}")

    def _resolve_strategy(self, strategy: str | Strategy, accelerator: Accelerator) -> Strategy:
        # Resolves the strategy argument: a live instance passes through
        # unchanged, a strategy name constructs the matching strategy, and
        # "auto" selects distributed data parallel only for multi-device CUDA
        # configurations, defaulting to single-device execution otherwise.
        if isinstance(strategy, Strategy):
            return strategy
        match strategy:
            case "ddp":
                return DDPStrategy()
            case "single_device":
                return SingleDeviceStrategy()
            case "auto":
                if self.devices > 1 and isinstance(accelerator, CUDAAccelerator):
                    return DDPStrategy()
                return SingleDeviceStrategy()
            case _:
                raise ValueError(f"Unknown strategy: {strategy}")

    def _resolve_device(self) -> torch.device:
        # Resolves the concrete device for this process. Distributed CUDA
        # execution derives the device index from the launcher-provided
        # LOCAL_RANK so each process binds its own GPU; single-process CUDA
        # uses device zero, and the remaining backends have one device each.
        accelerator_name: str = type(self.accelerator).__name__
        match accelerator_name:
            case "CUDAAccelerator":
                if self.strategy.is_distributed:
                    local_rank: int = int(__import__("os").environ.get("LOCAL_RANK", "0"))
                    return torch.device("cuda", local_rank)
                return torch.device("cuda", 0)
            case "MPSAccelerator":
                return torch.device("mps")
            case _:
                return torch.device("cpu")

    def _wrap_with_distributed_sampler(
        self,
        dataloader: torch.utils.data.DataLoader,
        shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        # Rebuilds the dataloader around a DistributedSampler while carrying
        # over every construction parameter the DataLoader exposes, so the
        # wrapped loader differs from the original only in its sampling.
        sampler: DistributedSampler = DistributedSampler(
            dataloader.dataset, shuffle=shuffle
        )
        return torch.utils.data.DataLoader(
            dataset=dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=sampler,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            collate_fn=dataloader.collate_fn,
            drop_last=dataloader.drop_last,
            timeout=dataloader.timeout,
            worker_init_fn=dataloader.worker_init_fn,
            persistent_workers=dataloader.persistent_workers,
            prefetch_factor=dataloader.prefetch_factor,
            generator=getattr(dataloader, "generator", None),
            multiprocessing_context=getattr(dataloader, "multiprocessing_context", None)
        )

    def _wrap_with_unrepeated_sampler(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> torch.utils.data.DataLoader:
        # Rebuilds the prediction dataloader around the unrepeated sampler,
        # which partitions without padding so no rank emits duplicate
        # predictions; all other construction parameters carry over unchanged.
        sampler: _UnrepeatedDistributedSampler = _UnrepeatedDistributedSampler(
            dataloader.dataset, shuffle=False
        )
        return torch.utils.data.DataLoader(
            dataset=dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=sampler,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            collate_fn=dataloader.collate_fn,
            drop_last=dataloader.drop_last,
            timeout=dataloader.timeout,
            worker_init_fn=dataloader.worker_init_fn,
            persistent_workers=dataloader.persistent_workers,
            prefetch_factor=dataloader.prefetch_factor,
            generator=getattr(dataloader, "generator", None),
            multiprocessing_context=getattr(dataloader, "multiprocessing_context", None)
        )

    def _sanity_check_validation(self, module: Module) -> None:
        # Runs a bounded validation pass before training to surface evaluation
        # defects immediately instead of after the first training epoch. The
        # validate loop's batch limit is temporarily replaced and restored in
        # a finally block, and every trace of the sanity pass (epoch metrics,
        # the module's logging buffer, callback metrics) is discarded so
        # sanity results cannot influence monitoring or checkpointing.
        log.info(f"Running {self.num_sanity_val_steps} sanity validation batches")

        for callback in self.callbacks:
            callback.on_sanity_check_start(self, module)

        self.state.set_running_stage("sanity_checking")
        module._running_stage: str | None = "sanity_checking"
        self._sanity_checking: bool = True

        original_limit: int | float | None = self.validate_loop.limit_batches
        self.validate_loop.limit_batches: int | float | None = self.num_sanity_val_steps
        try:
            self.validate_loop.run()
        finally:
            self._sanity_checking: bool = False
            self.validate_loop.limit_batches: int | float | None = original_limit

        self.validate_loop.epoch_metrics: dict[str, float] = {}
        module._reset_logged_metrics()
        self._callback_metrics.clear()

        for callback in self.callbacks:
            callback.on_sanity_check_end(self, module)

        module.on_validation_model_train()
        module._running_stage: str | None = "training"

    def _restore_checkpoint(self, path: Path, module: Module) -> None:
        # Restores a full checkpoint onto the live objects: model, optimizer,
        # and scheduler state through CheckpointState, the progress counters
        # into the trainer state and module mirrors, and the mid-epoch resume
        # markers (completed batches and epoch-start random state) into the
        # training epoch loop. The module and callback load hooks fire last so
        # they observe the fully restored payload.
        log.info(f"Restoring checkpoint from: {path}")
        checkpoint: CheckpointDict = load_checkpoint(path, map_location=self.strategy.root_device)
        assert self.optimizer is not None
        epoch: int
        global_step: int
        completed_batches: int
        epoch_rng_state: CheckpointDict | None
        epoch, global_step, completed_batches, epoch_rng_state = CheckpointState.restore(
            checkpoint=checkpoint,
            model=module,
            optimizers=self.optimizers,
            scheduler=self.scheduler,
            schedulers=self.schedulers,
            callbacks=self.callbacks,
            datamodule=self.datamodule
        )
        self.state.load_state_dict({"current_epoch": epoch, "global_step": global_step})
        module._current_epoch: int = epoch
        module._global_step: int = global_step
        if self.fit_loop.epoch_loop is not None:
            if completed_batches > 0:
                self.fit_loop.epoch_loop._completed_batches: int = completed_batches
            if epoch_rng_state is not None:
                self.fit_loop.epoch_loop._resume_rng_state: CheckpointDict | None = epoch_rng_state
        if hasattr(module, "on_load_checkpoint"):
            module.on_load_checkpoint(checkpoint)
        for callback in self.callbacks:
            callback.on_load_checkpoint(self, module, checkpoint)
        log.info(f"Resumed from epoch {epoch}, global step {global_step}")

    def _handle_exception(
        self,
        module: Module,
        datamodule: DataModule,
        exception: BaseException,
        stage: str
    ) -> None:
        # Central exception path for every entry point: marks the trainer
        # state interrupted, offers the exception to the callback, datamodule,
        # and strategy exception hooks so they can persist state or release
        # resources, and then re-raises. KeyboardInterrupt re-raises through
        # the bare raise so its traceback is preserved unchanged.
        self.state.interrupt()
        for callback in self.callbacks:
            callback.on_exception(self, module, exception)
        datamodule.on_exception(exception)
        self.strategy.on_exception(exception)
        if isinstance(exception, KeyboardInterrupt):
            log.warning(f"{stage} interrupted by user")
            raise
        else:
            raise exception

    def _validate_callback_state_keys(self) -> None:
        # Enforces state-key uniqueness across stateful callbacks before
        # fitting. Only callbacks that override the state methods participate,
        # so stateless callbacks of the same class may coexist freely, while
        # two stateful instances sharing a key would silently overwrite each
        # other inside checkpoints and are rejected instead.
        seen: dict[str, str] = {}
        for cb in self.callbacks:
            if type(cb).state_dict is not Callback.state_dict or type(cb).load_state_dict is not Callback.load_state_dict:
                key: str = cb.state_key
                if key in seen:
                    raise MisconfigurationError(
                        f"Duplicate callback state_key {key!r} between "
                        f"{seen[key]} and {cb!r}. "
                        f"Each stateful callback must have a unique state_key."
                    )
                seen[key] = repr(cb)

    def _resolve_total_scheduler_steps(
        self,
        scheduler_definitions: list[SchedulerDefinition]
    ) -> int:
        # Computes the horizon passed to the scheduler builder. When any
        # configured schedule advances per step, the horizon is the number of
        # optimizer steps across the whole run: the limited per-epoch batch
        # count divided by the accumulation factor, times the epoch budget. An
        # unknown batch total with a step-interval schedule is a configuration
        # error because the schedule cannot be sized. Epoch-interval schedules
        # use the epoch budget directly.
        config_schedulers: list[SchedulerConfig] = [
            scheduler_definition
            for scheduler_definition in scheduler_definitions
            if isinstance(scheduler_definition, SchedulerConfig)
        ]
        if not config_schedulers:
            raise MisconfigurationError(
                "Expected at least one scheduler configuration when resolving scheduler steps."
            )
        if any(scheduler_config.interval == "step" for scheduler_config in config_schedulers):
            assert self.fit_loop.epoch_loop is not None
            limited_batches: int | None = self.fit_loop.epoch_loop._resolve_limit_batches(
                self._total_train_batches
            )
            if limited_batches is None:
                raise MisconfigurationError(
                    "Cannot determine total training steps for the learning rate scheduler. "
                    "With iterable datasets, set limit_train_batches to an integer value."
                )
            accum: int = self.fit_loop.epoch_loop.accumulate_grad_batches
            stepping_per_epoch: int = max(1, math.ceil(limited_batches / accum))
            return stepping_per_epoch * self.max_epochs
        return self.max_epochs

    @property
    def current_epoch(self) -> int:
        # Zero-based epoch counter mirrored from the trainer state.
        return self.state.current_epoch

    @property
    def global_step(self) -> int:
        # Count of optimizer steps executed, mirrored from the trainer state.
        return self.state.global_step

    @property
    def callback_metrics(self) -> dict[str, float]:
        # Copy of the metric mapping monitored by callbacks; returned as a new
        # dictionary so callers cannot mutate the trainer's internal state.
        return dict(self._callback_metrics)

    @property
    def logged_metrics(self) -> dict[str, float]:
        # Reduced epoch metrics from the most recent validation pass, exposed
        # for inspection after validate() or a fit-scheduled validation.
        metrics: dict[str, float] = {}
        metrics.update(self.validate_loop.epoch_metrics)
        return metrics

    @property
    def interrupted(self) -> bool:
        # Whether the most recent run terminated through an interruption.
        return self.state.status == "interrupted"

    @property
    def estimated_stepping_batches(self) -> int | float:
        # Estimated total optimizer steps for the configured run, derived from
        # the limited batch count and the accumulation factor. Returns
        # infinity when the batch total is unknown, which callers such as
        # one-cycle schedule sizing must handle explicitly.
        if self.fit_loop.epoch_loop is None:
            return float("inf")
        limited_batches: int | None = self.fit_loop.epoch_loop._resolve_limit_batches(
            self._total_train_batches
        )
        if limited_batches is None:
            return float("inf")
        accum: int = self.fit_loop.epoch_loop.accumulate_grad_batches
        stepping_per_epoch: int = max(1, math.ceil(limited_batches / accum))
        return stepping_per_epoch * self.max_epochs

    def __repr__(self) -> str:
        # Compact configuration line for logs and interactive debugging.
        return (
            f"Trainer("
            f"max_epochs={self.max_epochs}, "
            f"accelerator={type(self.accelerator).__name__}, "
            f"strategy={type(self.strategy).__name__}, "
            f"devices={self.devices}, "
            f"precision={self.precision!r}, "
            f"callbacks={[type(callback).__name__ for callback in self.callbacks]})"
        )
