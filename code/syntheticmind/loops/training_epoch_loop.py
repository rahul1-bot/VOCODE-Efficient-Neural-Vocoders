# This module:
# 1. Executes one training epoch: batch fetching with timing instrumentation, the
#    batch-transfer hook chain, step execution under the configured precision
#    policy, gradient accumulation, optimizer and scheduler stepping, and metric
#    dispatch at step and epoch granularity
# 2. Implements deterministic mid-epoch resume by restoring the epoch-start
#    random-number state recorded in checkpoints and replaying already-completed
#    batches without executing them
# 3. Schedules mid-epoch validation against batch-count, epoch-fraction, or
#    global-step intervals
# 4. Owns the mixed-precision GradScaler lifecycle and both gradient-clipping
#    entry points
#
# Design decisions:
# - For each batch-transfer hook, the datamodule implementation takes precedence
#   over the module implementation. Precedence is decided per hook by comparing
#   the datamodule's bound attribute against the DataModule base attribute, so a
#   datamodule that overrides a hook wins for that hook only, and the module
#   remains the default for the rest
# - During accumulation batches under DistributedDataParallel, the step and
#   backward run inside model.no_sync(), so inter-rank gradient reduction occurs
#   only on the optimizer-step batch rather than on every micro-batch
# - The loss is divided by accumulate_grad_batches before backward, so the
#   accumulated gradient matches the gradient of the equivalent large batch
# - The final limited batch forces a non-accumulating step, and a tail flush
#   after the loop performs one more optimizer step when the epoch ends inside
#   an accumulation window, so no accumulated gradients are silently discarded
# - Step metrics are withheld during accumulation batches and emitted from rank
#   zero every log_every_n_steps. Epoch metrics accumulate through
#   MetricAccumulator under each metric's declared reduction and batch-size
#   weighting, update the trainer's callback metrics on every rank, and reach
#   the logger only from rank zero
# - Mid-epoch resume first restores the epoch-start random-number state, then
#   consumes and discards the already-completed prefix from a fresh dataloader
#   iterator. The dataloader therefore reproduces the exact original batch
#   sequence, and execution continues at the first incomplete batch
# - The GradScaler exists only for 16-mixed precision on CUDA devices, and
#   gradients are unscaled before clipping so thresholds apply to true
#   gradient magnitudes rather than scaled ones
#
# Author: Rahul Sawhney

import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.module import LoggedMetric, Module
from syntheticmind.core.optimizer import SchedulerConfig
from syntheticmind.loggers.logger import Logger
from syntheticmind.loops.loop import Loop
from syntheticmind.state.trainer_state import TrainerState
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.utilities.distributed import DistributedUtils
from syntheticmind.utilities.metric_accumulator import MetricAccumulator
from syntheticmind.utilities.types import Batch, CheckpointDict, StepOutput

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["TrainingEpochLoop"]


class TrainingEpochLoop(Loop):
    # Loop that executes one training epoch each time run() is invoked. The fit
    # loop constructs it once, configures it from the trainer arguments, and
    # calls it once per epoch. The loop coordinates the strategy (step
    # execution), the module and callbacks (hook delivery), the optimizer and
    # scheduler (stepping cadence), and the logger (metric emission), while
    # keeping enough internal state to support deterministic mid-epoch resume.
    def __init__(
        self,
        accumulate_grad_batches: int = 1,
        val_check_interval: int | float = 1.0,
        val_check_interval_scope: Literal["epoch", "global_step"] = "epoch",
        validate_at_epoch_end: bool = True,
        precision: str = "32-true",
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str = "norm",
        log_every_n_steps: int = 10,
        limit_train_batches: int | float | None = None
    ) -> None:
        # Stores the epoch-execution configuration supplied by the trainer and
        # initializes the resume-tracking state: the count of completed batches
        # inside the current epoch, the flag marking whether the batch loop is
        # active, and the random-number-state slots used by mid-epoch resume.
        super().__init__()
        self.accumulate_grad_batches: int = accumulate_grad_batches
        self.val_check_interval: int | float = val_check_interval
        self.val_check_interval_scope: Literal["epoch", "global_step"] = val_check_interval_scope
        self.validate_at_epoch_end: bool = validate_at_epoch_end
        self.precision: str = precision
        self.gradient_clip_val: float | None = gradient_clip_val
        self.gradient_clip_algorithm: str = gradient_clip_algorithm
        self.log_every_n_steps: int = log_every_n_steps
        self.limit_train_batches: int | float | None = limit_train_batches
        self.validation_ran_this_epoch: bool = False
        self._grad_scaler: GradScaler | None = None
        self._completed_batches: int = 0
        self._has_entered_batch_loop: bool = False
        self._epoch_start_rng_state: CheckpointDict | None = None
        self._resume_rng_state: CheckpointDict | None = None

    def reset_on_run(self) -> None:
        # Clears the mid-epoch resume markers at the start of a new run so a
        # fresh run never replays batch positions recorded by a previous one.
        self._completed_batches: int = 0
        self._has_entered_batch_loop: bool = False

    def run(self) -> None:
        # Executes one complete training epoch. The sequence per batch is:
        # fetch with timing instrumentation, on_train_batch_start hooks, the
        # batch-transfer hook chain, the training step under the precision
        # context, backward and optimizer handling according to the
        # accumulation schedule, on_train_batch_end hooks, step-metric
        # dispatch, and finally the mid-epoch validation check and the
        # cooperative stop check. After the batch loop, a tail optimizer step
        # flushes any accumulated gradients and epoch metrics are dispatched.
        self.validation_ran_this_epoch: bool = False

        from syntheticmind.state.checkpoint_state import CheckpointState

        if self._resume_rng_state is not None:
            CheckpointState._restore_rng_states(self._resume_rng_state)
            self._resume_rng_state: CheckpointDict | None = None

        self._epoch_start_rng_state: CheckpointDict | None = CheckpointState._collect_rng_states()

        trainer: Trainer = self.trainer
        module: Module | None = trainer.module
        assert module is not None
        strategy: Strategy = trainer.strategy
        state: TrainerState = trainer.state
        callbacks: list[Callback] = trainer.callbacks
        optimizers: list[torch.optim.Optimizer] = trainer.optimizers
        if not optimizers:
            raise RuntimeError("Trainer has no optimizers configured for training.")
        optimizer: torch.optim.Optimizer | None = trainer.optimizer
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = trainer.scheduler
        scheduler_config: SchedulerConfig | None = trainer.scheduler_config
        train_dataloader: DataLoader | None = trainer.train_dataloader  # type: ignore[type-arg]
        assert train_dataloader is not None
        log_fn: Logger | None = trainer.logger
        device: torch.device = strategy.root_device
        model: nn.Module | None = strategy.model
        assert model is not None

        if self.precision == "16-mixed" and device.type == "cuda" and self._grad_scaler is None:
            self._grad_scaler: GradScaler | None = GradScaler("cuda")

        total_batches: int | None = self._safe_len(train_dataloader)
        max_batches: int | None = self._resolve_limit_batches(total_batches)
        epoch_accum: dict[str, MetricAccumulator] = {}
        epoch_logger_names: set[str] = set()
        skip_batches: int = self._completed_batches
        tail_is_accumulating: bool = False

        self._has_entered_batch_loop: bool = True
        dataloader_iterator_start_time: float = time.perf_counter()
        train_iterator: Iterator[Batch] = iter(train_dataloader)
        trainer._record_dataloader_iterator_creation_time(
            "training",
            time.perf_counter() - dataloader_iterator_start_time
        )
        batch_idx: int = -1
        while True:
            dataloader_fetch_start_time: float = time.perf_counter()
            try:
                batch: Batch = next(train_iterator)
            except StopIteration:
                break
            trainer._record_dataloader_fetch_time(
                "training",
                time.perf_counter() - dataloader_fetch_start_time
            )
            batch_idx += 1
            if max_batches is not None and batch_idx >= max_batches:
                break
            # Mid-epoch resume: the already-completed prefix is consumed and
            # discarded, so the dataloader reproduces the original batch
            # sequence and execution continues at the first incomplete batch.
            if batch_idx < skip_batches:
                continue

            for callback in callbacks:
                callback.on_train_batch_start(trainer, module, batch, batch_idx)
            module.on_train_batch_start(batch, batch_idx)

            dm: DataModule | None = trainer.datamodule
            if dm is not None and type(dm).on_before_batch_transfer is not DataModule.on_before_batch_transfer:
                batch: Batch = dm.on_before_batch_transfer(batch, dataloader_idx=0)
            else:
                batch: Batch = module.on_before_batch_transfer(batch, dataloader_idx=0)
            if dm is not None and type(dm).transfer_batch_to_device is not DataModule.transfer_batch_to_device:
                batch: Batch = dm.transfer_batch_to_device(batch, device, dataloader_idx=0)
            else:
                batch: Batch = module.transfer_batch_to_device(batch, device, dataloader_idx=0)
            if dm is not None and type(dm).on_after_batch_transfer is not DataModule.on_after_batch_transfer:
                batch: Batch = dm.on_after_batch_transfer(batch, dataloader_idx=0)
            else:
                batch: Batch = module.on_after_batch_transfer(batch, dataloader_idx=0)

            # Automatic path: the loss is divided by the accumulation factor
            # so the accumulated gradient matches the equivalent large batch,
            # DDP gradient sync is suppressed on accumulation batches, and
            # the optimizer steps only on window boundaries, with the final
            # limited batch forcing a non-accumulating step.
            if module.automatic_optimization:
                assert optimizer is not None
                is_accumulating: bool = (batch_idx + 1) % self.accumulate_grad_batches != 0
                if max_batches is not None and (batch_idx + 1) >= max_batches:
                    is_accumulating: bool = False

                no_sync_ctx: AbstractContextManager[None] = nullcontext()
                if isinstance(model, DistributedDataParallel) and is_accumulating:
                    no_sync_ctx: AbstractContextManager[None] = model.no_sync()

                with no_sync_ctx:
                    outputs: StepOutput = self._run_training_step(strategy, module, model, batch, batch_idx)
                    loss_raw: torch.Tensor | float = outputs["loss"]
                    loss: torch.Tensor = (
                        loss_raw if isinstance(loss_raw, torch.Tensor) else torch.tensor(loss_raw, device=device)
                    )
                    scaled_loss: torch.Tensor = loss / self.accumulate_grad_batches
                    self.manual_backward(trainer, module, scaled_loss)

                if not is_accumulating:
                    self._do_optimizer_step(
                        trainer, module, optimizer, scheduler, scheduler_config, state
                    )
            else:
                outputs: StepOutput = self._run_training_step(strategy, module, model, batch, batch_idx)
                is_accumulating: bool = False
                self._increment_global_step(state, module)

            for callback in callbacks:
                callback.on_train_batch_end(trainer, module, outputs, batch, batch_idx)
            module.on_train_batch_end(outputs, batch, batch_idx)

            self._dispatch_step_metrics(module, state, log_fn, epoch_accum, epoch_logger_names, is_accumulating)

            self._completed_batches: int = batch_idx + 1
            tail_is_accumulating: bool = is_accumulating

            val_dataloader: DataLoader | None = trainer.val_dataloader  # type: ignore[type-arg]
            if (
                val_dataloader is not None
                and max_batches is not None
                and self._should_validate_mid_epoch(batch_idx, max_batches, state)
            ):
                if module.automatic_optimization and not is_accumulating:
                    module.zero_grad(set_to_none=True)
                trainer._is_mid_epoch_validation: bool = True
                trainer.validate_loop.run()
                trainer._is_mid_epoch_validation: bool = False
                self.validation_ran_this_epoch: bool = True
                module.on_validation_model_train()
                module._running_stage: str | None = "training"

            if state.should_stop:
                break

        # Tail flush: when the epoch ends inside an accumulation window, one
        # more optimizer step runs so accumulated gradients are never
        # silently discarded.
        if module.automatic_optimization and tail_is_accumulating:
            self._do_optimizer_step(
                trainer, module, optimizer, scheduler, scheduler_config, state
            )

        self._has_entered_batch_loop: bool = False
        self._completed_batches: int = 0
        self._dispatch_epoch_metrics(epoch_accum, state, log_fn, epoch_logger_names)

    def _make_autocast_context(self, device: torch.device) -> autocast | nullcontext[None]:
        # Returns the autocast context matching the configured precision policy:
        # float16 autocast for 16-mixed, bfloat16 autocast for bf16-mixed, and a
        # null context for full precision.
        if self.precision == "16-mixed":
            return autocast(device.type, dtype=torch.float16)
        if self.precision == "bf16-mixed":
            return autocast(device.type, dtype=torch.bfloat16)
        return nullcontext()

    def _run_training_step(
        self,
        strategy: Strategy,
        module: Module,
        model: nn.Module,
        batch: Batch,
        batch_idx: int
    ) -> StepOutput:
        # Executes the training step through the strategy inside the precision
        # context. The module's current-hook marker is set for the duration of
        # the step so Module.log can resolve its logging rule, and it is cleared
        # in a finally block so a raising step cannot leave a stale marker.
        with self._make_autocast_context(self.trainer.strategy.root_device):
            module._current_fx_name: str | None = "training_step"
            try:
                outputs: StepOutput = strategy.training_step(model, batch, batch_idx)
            finally:
                module._current_fx_name: str | None = None
        return outputs

    @staticmethod
    def _safe_len(dataloader: DataLoader) -> int | None:  # type: ignore[type-arg]
        # Returns the dataloader length, or None for iterable datasets that do
        # not define one, so limit resolution can distinguish the two cases.
        try:
            return len(dataloader)
        except TypeError:
            return None

    def _resolve_limit_batches(self, total: int | None) -> int | None:
        # Resolves the effective batch ceiling for this epoch. With a known
        # total, a float limit selects that fraction of the epoch (at least one
        # batch) and an integer limit is capped at the total. Without a known
        # total, an integer limit is used directly, a fractional float limit is
        # rejected as a configuration error, and no limit means unbounded.
        if total is None:
            if isinstance(self.limit_train_batches, int):
                return self.limit_train_batches
            if isinstance(self.limit_train_batches, float) and self.limit_train_batches != 1.0:
                from syntheticmind.utilities.exceptions import MisconfigurationError
                raise MisconfigurationError(
                    f"limit_train_batches={self.limit_train_batches} (float) is not supported "
                    f"with iterable datasets that have no length. Use an integer limit or set to 1.0."
                )
            return None
        if self.limit_train_batches is None:
            return total
        if isinstance(self.limit_train_batches, float):
            return max(1, int(total * self.limit_train_batches))
        return min(int(self.limit_train_batches), total)

    def _clip_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        # Applies the configured clipping algorithm to every parameter in the
        # optimizer that currently holds a gradient. Parameters without
        # gradients are excluded so norm computation reflects only live
        # gradients.
        params: list[torch.nn.Parameter] = [
            p for g in optimizer.param_groups for p in g["params"] if p.grad is not None
        ]
        if not params:
            return
        if self.gradient_clip_algorithm == "norm":
            torch.nn.utils.clip_grad_norm_(params, self.gradient_clip_val)  # type: ignore[arg-type]
        elif self.gradient_clip_algorithm == "value":
            torch.nn.utils.clip_grad_value_(params, self.gradient_clip_val)  # type: ignore[arg-type]

    def clip_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None
    ) -> None:
        # Public clipping entry point for manual-optimization modules. Explicit
        # arguments override the configured values for this invocation only:
        # the configured values are swapped in, restored in a finally block,
        # and gradients are unscaled first when a GradScaler is active so the
        # threshold applies to true magnitudes. A non-positive resolved value
        # disables clipping for the invocation.
        resolved_gradient_clip_val: float | None = (
            self.gradient_clip_val if gradient_clip_val is None else gradient_clip_val
        )
        if resolved_gradient_clip_val is None or resolved_gradient_clip_val <= 0:
            return
        resolved_gradient_clip_algorithm: str = (
            self.gradient_clip_algorithm
            if gradient_clip_algorithm is None
            else gradient_clip_algorithm
        )
        if resolved_gradient_clip_algorithm not in {"norm", "value"}:
            from syntheticmind.utilities.exceptions import MisconfigurationError

            raise MisconfigurationError(
                "gradient_clip_algorithm must be 'norm' or 'value'."
            )
        original_gradient_clip_val: float | None = self.gradient_clip_val
        original_gradient_clip_algorithm: str = self.gradient_clip_algorithm
        self.gradient_clip_val: float | None = resolved_gradient_clip_val
        self.gradient_clip_algorithm: str = resolved_gradient_clip_algorithm
        try:
            if self._grad_scaler is not None:
                self._grad_scaler.unscale_(optimizer)
            self._clip_gradients(optimizer)
        finally:
            self.gradient_clip_val: float | None = original_gradient_clip_val
            self.gradient_clip_algorithm: str = original_gradient_clip_algorithm

    def manual_backward(
        self,
        trainer: Trainer,
        module: Module,
        loss: torch.Tensor
    ) -> None:
        # Runs the backward pass surrounded by the backward hook pair. Callback
        # hooks fire before the corresponding module hooks. With an active
        # GradScaler the loss is scaled before backward; otherwise backward is
        # delegated to the strategy so distributed synchronization applies.
        callbacks: list[Callback] = trainer.callbacks
        for callback in callbacks:
            callback.on_before_backward(trainer, module, loss)
        module.on_before_backward(loss)

        if self._grad_scaler is not None:
            self._grad_scaler.scale(loss).backward()
        else:
            trainer.strategy.backward(loss)

        for callback in callbacks:
            callback.on_after_backward(trainer, module)
        module.on_after_backward()

    def optimizer_step(
        self,
        trainer: Trainer,
        module: Module,
        optimizer: torch.optim.Optimizer
    ) -> None:
        # Public optimizer-step entry point for manual-optimization modules;
        # delegates to the shared stepping helper so hook order and scaler
        # handling match the automatic path.
        self._optimizer_step_only(trainer, module, optimizer)

    def optimizer_zero_grad(
        self,
        trainer: Trainer,
        module: Module,
        optimizer: torch.optim.Optimizer
    ) -> None:
        # Public gradient-clearing entry point for manual-optimization modules;
        # delegates to the shared helper so the zero-grad hooks always fire.
        self._optimizer_zero_grad(trainer, module, optimizer)

    def _do_optimizer_step(
        self,
        trainer: Trainer,
        module: Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        scheduler_config: SchedulerConfig | None,
        state: TrainerState
    ) -> None:
        # Executes the full step sequence for the automatic-optimization path:
        # optimizer step, step-interval scheduler advance, gradient clearing,
        # and the global-step increment. Plateau schedulers are excluded here
        # because they advance on monitored metrics at epoch boundaries.
        self._optimizer_step_only(trainer, module, optimizer)
        if scheduler is not None and scheduler_config is not None:
            if scheduler_config.interval == "step" and scheduler_config.name != "reduce_on_plateau":
                scheduler.step()
        self._optimizer_zero_grad(trainer, module, optimizer)
        self._increment_global_step(state, module)

    def _optimizer_step_only(
        self,
        trainer: Trainer,
        module: Module,
        optimizer: torch.optim.Optimizer
    ) -> None:
        # Applies the optimizer step preceded by its hooks and by gradient
        # clipping. When a GradScaler is active, gradients are unscaled before
        # clipping and the step runs through the scaler, which skips the update
        # on overflow and adjusts the scale afterward.
        callbacks: list[Callback] = trainer.callbacks
        for callback in callbacks:
            callback.on_before_optimizer_step(trainer, module, optimizer)
        module.on_before_optimizer_step(optimizer)

        if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
            if self._grad_scaler is not None:
                self._grad_scaler.unscale_(optimizer)
            self._clip_gradients(optimizer)

        if self._grad_scaler is not None:
            self._grad_scaler.step(optimizer)
            self._grad_scaler.update()
        else:
            optimizer.step()

    def _optimizer_zero_grad(
        self,
        trainer: Trainer,
        module: Module,
        optimizer: torch.optim.Optimizer
    ) -> None:
        # Clears gradients preceded by the zero-grad hooks, so callbacks and
        # the module observe gradients in their final post-step state before
        # they are released.
        callbacks: list[Callback] = trainer.callbacks
        for callback in callbacks:
            callback.on_before_zero_grad(trainer, module, optimizer)
        module.on_before_zero_grad(optimizer)
        optimizer.zero_grad()

    @staticmethod
    def _increment_global_step(state: TrainerState, module: Module) -> None:
        # Advances the trainer-owned global-step counter and mirrors the new
        # value onto the module so Module.global_step stays consistent with
        # trainer state between checkpoints.
        state.increment_global_step()
        module._global_step: int = state.global_step

    def _should_validate_mid_epoch(
        self,
        batch_idx: int,
        total_batches: int,
        state: TrainerState
    ) -> bool:
        # Decides whether validation runs after the current batch. An integer
        # interval counts either optimizer steps (global_step scope) or batches
        # within the epoch (epoch scope); a float below one selects evenly
        # spaced positions as a fraction of the epoch. The final batch position
        # is excluded in batch-scoped modes because epoch-end validation is
        # scheduled separately by the fit loop.
        if isinstance(self.val_check_interval, int):
            if self.val_check_interval_scope == "global_step":
                return state.global_step > 0 and state.global_step % self.val_check_interval == 0
            return (batch_idx + 1) % self.val_check_interval == 0 and (batch_idx + 1) < total_batches
        if self.val_check_interval >= 1.0:
            return False
        check_every: int = max(1, int(total_batches * self.val_check_interval))
        return (batch_idx + 1) % check_every == 0 and (batch_idx + 1) < total_batches

    def _dispatch_step_metrics(
        self,
        module: Module,
        state: TrainerState,
        log_fn: Logger | None,
        epoch_accum: dict[str, MetricAccumulator],
        epoch_logger_names: set[str],
        is_accumulating: bool = False
    ) -> None:
        # Drains the module's logged-metric buffer after each batch. Metrics
        # marked for distributed synchronization are mean-reduced across ranks
        # before use. Step-scoped logger metrics are emitted from rank zero at
        # the configured cadence, and never during accumulation batches so a
        # partially accumulated step is not reported. Epoch-scoped metrics feed
        # per-name accumulators under each metric's declared reduction, with
        # logger visibility tracked separately from callback visibility.
        logged: dict[str, LoggedMetric] = module.logged_metrics
        if not logged:
            return

        step_metrics: dict[str, float] = {}
        for name, metric in logged.items():
            value: torch.Tensor | float = metric.value
            if metric.sync_distributed and DistributedUtils.is_distributed():
                if not isinstance(value, torch.Tensor):
                    value: torch.Tensor | float = torch.tensor(value, device=self.trainer.strategy.root_device)
                value: torch.Tensor | float = DistributedUtils.all_reduce_mean(value)
            if isinstance(value, torch.Tensor):
                value: torch.Tensor | float = value.item()

            if metric.on_step and metric.logger:
                step_metrics[name] = value
            if metric.on_epoch:
                if name not in epoch_accum:
                    epoch_accum[name] = MetricAccumulator(reduction=metric.reduction)
                epoch_accum[name].update(value, batch_size=metric.batch_size)
                if metric.logger:
                    epoch_logger_names.add(name)

        if step_metrics and log_fn is not None and not is_accumulating and state.global_step % self.log_every_n_steps == 0:
            if DistributedUtils.is_rank_zero():
                log_fn.log_metrics(step_metrics, step=state.global_step)

        module._reset_logged_metrics()

    def _dispatch_epoch_metrics(
        self,
        epoch_accum: dict[str, MetricAccumulator],
        state: TrainerState,
        log_fn: Logger | None,
        epoch_logger_names: set[str]
    ) -> None:
        # Reduces every epoch accumulator at the epoch boundary. The reduced
        # values update the trainer's callback metrics on every rank, because
        # monitoring callbacks such as early stopping must see them everywhere.
        # Logger emission is restricted to rank zero and to the metrics whose
        # logging configuration requested logger visibility.
        if not epoch_accum:
            return
        all_reduced: dict[str, float] = {}
        for name in epoch_accum:
            all_reduced[name] = epoch_accum[name].compute()
        self.trainer._callback_metrics.update(all_reduced)
        if log_fn is None:
            return
        if not DistributedUtils.is_rank_zero():
            return
        logger_reduced: dict[str, float] = {
            k: v for k, v in all_reduced.items() if k in epoch_logger_names
        }
        if logger_reduced:
            log_fn.log_metrics(logger_reduced, step=state.global_step)
