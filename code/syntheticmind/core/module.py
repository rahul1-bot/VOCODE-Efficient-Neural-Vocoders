# This module:
# 1. Defines Module, the base class every harness model subclasses: the step and
#    optimizer-configuration contract, the buffered metric-logging interface,
#    device and dtype mirrors, the manual-optimization surface, and the
#    trainer-attached accessors
# 2. Defines MetricLoggingConfiguration, the frozen per-call options accepted by
#    Module.log
# 3. Defines LoggedMetric, the buffered record pairing one logged value with the
#    routing metadata the loops use to dispatch it
#
# Design decisions:
# - Module composes torch.nn.Module with the model, data, and checkpoint hook
#   families through multiple inheritance, so one subclass receives the entire
#   lifecycle surface without wrapper objects
# - log() validates its call context through the hook-name marker the loops
#   maintain around every hook and step, resolved against the central
#   metric-logging contract; an illegal call site fails immediately with the
#   offending hook named, and unset step and epoch flags resolve to per-hook
#   defaults instead of one global default
# - Logged tensor values are detached at logging time, so the metric buffer can
#   never retain autograd graphs across batches
# - The device and dtype overrides (to, cuda, cpu, type, float, half) refresh
#   the cached mirrors by introspecting parameters first, buffers second, and
#   the call arguments last, so parameter-free modules still report accurate
#   placement and precision
# - The manual-optimization methods delegate to the trainer's epoch loop rather
#   than reimplementing stepping, so hook order, gradient-scaler handling, and
#   clipping behave identically under manual and automatic optimization
# - toggle_optimizer records the requires_grad state of every parameter before
#   restricting training to one optimizer's parameter set, and
#   untoggle_optimizer restores the recorded state exactly, so alternating
#   multi-optimizer schemes cannot corrupt gradient flags
#
# Author: Rahul Sawhney

import builtins
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar, override

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from syntheticmind.core.hooks import DataHooks, MetricLoggingContract, ModelHooks
from syntheticmind.core.optimizer import OptimizationConfiguration
from syntheticmind.core.saving import CheckpointHooks
from syntheticmind.loggers.logger import Logger
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.types import Batch, ModelOutput, Reduction, StepOutput

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer
    from syntheticmind.loops.training_epoch_loop import TrainingEpochLoop

__all__: list[str] = ["Module", "MetricLoggingConfiguration", "LoggedMetric"]


class MetricLoggingConfiguration(BaseModel):
    # Frozen per-call options for Module.log.
    #
    # Fields:
    #     progress_bar: Whether the metric appears in the training progress
    #         bar postfix. Default: ``False``.
    #     logger: Whether the metric is forwarded to the experiment logger.
    #         Default: ``True``.
    #     on_step: Explicit per-step emission routing; ``None`` defers to
    #         the per-hook default of the metric-logging contract.
    #         Default: ``None``.
    #     on_epoch: Explicit epoch-reduction routing; ``None`` defers to
    #         the per-hook default of the metric-logging contract.
    #         Default: ``None``.
    #     reduction: Epoch reduction applied to the accumulated step
    #         values: ``"mean"``, ``"sum"``, ``"min"``, or ``"max"``.
    #         Default: ``"mean"``.
    #     sync_distributed: Whether the reduced value is synchronized
    #         across ranks before publication. Default: ``False``.
    #     batch_size: Weight applied to this step's value in the epoch
    #         mean; ``None`` weights every step equally. Default: ``None``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    progress_bar: bool = False
    logger: bool = True
    on_step: bool | None = None
    on_epoch: bool | None = None
    reduction: Reduction = "mean"
    sync_distributed: bool = False
    batch_size: int | None = None


class Module(nn.Module, ModelHooks, DataHooks, CheckpointHooks):
    # Base class for every model the harness trains or evaluates. Subclasses
    # implement forward, the step methods, and configure_optimizers; the
    # trainer attaches itself before execution, after which the metric-logging
    # buffer, the epoch and step mirrors, and the manual-optimization surface
    # become operational. The device and dtype mirrors track the module's true
    # placement across every torch movement method.
    #
    # Integration: a minimal client model subclasses
    # syntheticmind.core.module.Module (not torch.nn.Module directly; this
    # class already inherits it) and implements forward, training_step,
    # validation_step, and configure_optimizers; everything else has a
    # working default.
    #
    # Example::
    #
    #     import torch
    #     from torch import nn
    #     from torch.nn import functional as F
    #
    #     from syntheticmind.core.module import Module
    #     from syntheticmind.core.optimizer import (
    #         OptimizationConfiguration,
    #         OptimizerConfig
    #     )
    #     from syntheticmind.utilities.types import Batch, StepOutput
    #
    #     class AutoEncoder(Module):
    #         def __init__(self) -> None:
    #             super().__init__()
    #             self.encoder: nn.Linear = nn.Linear(784, 64)
    #             self.decoder: nn.Linear = nn.Linear(64, 784)
    #
    #         def forward(self, features: torch.Tensor) -> torch.Tensor:
    #             return self.decoder(self.encoder(features))
    #
    #         def training_step(self, batch: Batch, batch_idx: int) -> StepOutput:
    #             features, _ = batch
    #             reconstruction: torch.Tensor = self(features)
    #             loss: torch.Tensor = F.mse_loss(reconstruction, features)
    #             self.log("train_loss", loss)
    #             return {"loss": loss}
    #
    #         def validation_step(self, batch: Batch, batch_idx: int) -> StepOutput:
    #             features, _ = batch
    #             loss: torch.Tensor = F.mse_loss(self(features), features)
    #             self.log("val_loss", loss)
    #             return {"loss": loss}
    #
    #         def configure_optimizers(self) -> OptimizationConfiguration:
    #             return OptimizationConfiguration(
    #                 optimizer=OptimizerConfig(name="adamw", lr=1e-3)
    #             )
    #
    # Models that drive several optimizers (for example adversarial training)
    # disable automatic optimization in their constructor and step manually
    # through the syntheticmind.core.module.Module manual-optimization
    # surface. The declarations returned by configure_optimizers are
    # syntheticmind configuration records; the live objects handed back by
    # self.optimizers() at run time are plain torch.optim.Optimizer
    # instances.
    #
    # Example::
    #
    #     class Gan(Module):
    #         def __init__(self) -> None:
    #             super().__init__()
    #             self.automatic_optimization = False
    #             ...
    #
    #         def training_step(self, batch: Batch, batch_idx: int) -> StepOutput:
    #             optimizer_pair: list[torch.optim.Optimizer] = self.optimizers()
    #             generator_opt: torch.optim.Optimizer = optimizer_pair[0]
    #             discriminator_opt: torch.optim.Optimizer = optimizer_pair[1]
    #
    #             discriminator_loss: torch.Tensor = ...
    #             self.optimizer_zero_grad(discriminator_opt)
    #             self.manual_backward(discriminator_loss)
    #             self.optimizer_step(discriminator_opt)
    #
    #             generator_loss: torch.Tensor = ...
    #             self.optimizer_zero_grad(generator_opt)
    #             self.manual_backward(generator_loss)
    #             self.optimizer_step(generator_opt)
    #             return {"loss": generator_loss.detach()}
    def __init__(self) -> None:
        # Starts detached from any trainer with empty logging state, CPU
        # float32 mirrors, automatic optimization enabled, and no recorded
        # requires_grad snapshot. The trainer reference, epoch and step
        # mirrors, and hook-name marker are maintained by the trainer and
        # loops during execution.
        super().__init__()
        self._trainer: Trainer | None = None
        self._example_input_array: torch.Tensor | tuple[object, ...] | dict[str, object] | None = None
        self._automatic_optimization: bool = True
        self._strict_loading: bool | None = None
        self._current_epoch: int = 0
        self._global_step: int = 0
        self._logged_metrics: dict[str, LoggedMetric] = {}
        self._running_stage: str | None = None
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32
        self._current_fx_name: str | None = None
        self._param_requires_grad_state: dict[torch.nn.Parameter, bool] = {}

    @override
    def forward(self, *args: object, **kwargs: object) -> ModelOutput:
        # The model's tensor computation. Subclasses must override; the
        # default predict step and any direct invocation route through here.
        raise NotImplementedError

    def training_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Operates on a single batch from the training dataloader. This is
        # where the client computes its training loss and logs training
        # metrics through self.log.
        #
        # Args:
        #     batch: The output of the training dataloader after the
        #         batch-transfer hook chain has placed it on the run device.
        #     batch_idx: The index of this batch within the current epoch.
        #
        # Return:
        #     A mapping (syntheticmind.utilities.types.StepOutput) that must
        #     contain the key ``"loss"`` holding a scalar tensor with an
        #     attached graph. Under automatic optimization the training
        #     epoch loop backpropagates exactly this tensor; additional keys
        #     pass through to the batch-end callbacks.
        #
        # Note:
        #     Under manual optimization (automatic_optimization = False) the
        #     loop performs no backward and no stepping; the step body
        #     drives self.manual_backward, self.optimizer_step, and
        #     self.optimizer_zero_grad itself, as shown in the class-level
        #     example. The returned ``"loss"`` is then reporting-only and
        #     should be detached.
        raise NotImplementedError

    def validation_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Operates on a single batch from the validation dataloader,
        # computing the quantities the client monitors (for example the
        # validation loss watched by EarlyStopping and ModelCheckpoint).
        #
        # Args:
        #     batch: The output of the validation dataloader after the
        #         batch-transfer hook chain.
        #     batch_idx: The index of this batch within the validation pass.
        #
        # Return:
        #     A mapping of scalar tensors; values logged through self.log
        #     inside this step are epoch-reduced by the evaluation loop and
        #     published to the progress bar, logger, and callback metrics.
        #
        # Note:
        #     When this step runs, the module has been switched to
        #     evaluation mode and the pass executes under
        #     torch.inference_mode, so no gradients exist. After a
        #     validation pass that runs inside fit, the module returns to
        #     training mode automatically.
        raise NotImplementedError

    def test_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Operates on a single batch from the test dataloader.
        #
        # Args:
        #     batch: The output of the test dataloader after the
        #         batch-transfer hook chain.
        #     batch_idx: The index of this batch within the test pass.
        #
        # Return:
        #     A mapping of scalar tensors, with the same logging and
        #     reduction behavior as validation_step.
        #
        # Note:
        #     The default implementation delegates to validation_step so
        #     shared evaluation logic is written once; override when
        #     test-time behavior differs. The evaluation-mode and
        #     inference-mode conditions of validation_step apply here
        #     equally.
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch: Batch, batch_idx: int) -> ModelOutput:
        # Operates on a single batch during Trainer.predict, producing the
        # model's inference output.
        #
        # Args:
        #     batch: The output of the prediction dataloader after the
        #         batch-transfer hook chain.
        #     batch_idx: The index of this batch within the prediction pass.
        #
        # Return:
        #     The prediction for this batch. Trainer.predict collects these
        #     returns in batch order (unless retention is disabled through
        #     its return_predictions argument).
        #
        # Note:
        #     The default implementation forwards the whole batch through
        #     forward; override to unpack inputs or add postprocessing.
        #     Metric logging is rejected in this step by the logging
        #     contract, so results travel exclusively through the return
        #     value.
        #
        # Example::
        #
        #     class Classifier(Module):
        #         def predict_step(self, batch: Batch, batch_idx: int) -> ModelOutput:
        #             features, _ = batch
        #             return self(features)
        return self.forward(batch)

    def configure_optimizers(self) -> OptimizationConfiguration:
        # Declares the run's optimizers and schedulers. The trainer calls
        # this once at the start of fit and materializes the declaration
        # into live torch objects before the first epoch.
        #
        # Return:
        #     A syntheticmind.core.optimizer.OptimizationConfiguration
        #     whose optimizer field is one definition or an ordered list,
        #     and whose scheduler field is None, one definition, or a list
        #     paired with the optimizer list by position. Each definition is
        #     either a validated configuration (OptimizerConfig,
        #     SchedulerConfig) or an already-constructed torch object.
        #
        # Note:
        #     Under automatic optimization the loop calls backward, the
        #     optimizer step, and scheduler stepping itself; a
        #     SchedulerConfig's interval field selects whether its schedule
        #     advances per optimizer step or per epoch, and the plateau
        #     family additionally requires a monitored metric name.
        #     Multiple optimizers require manual optimization; the trainer
        #     validates this pairing before training starts.
        #
        # Example::
        #
        #     # Single optimizer with a step-interval schedule:
        #     def configure_optimizers(self) -> OptimizationConfiguration:
        #         return OptimizationConfiguration(
        #             optimizer=OptimizerConfig(name="adamw", lr=2e-4),
        #             scheduler=SchedulerConfig(name="cosine", interval="step")
        #         )
        #
        #     # Adversarial pair (requires automatic_optimization = False),
        #     # schedulers paired by position:
        #     def configure_optimizers(self) -> OptimizationConfiguration:
        #         return OptimizationConfiguration(
        #             optimizer=[
        #                 OptimizerConfig(name="adamw", lr=2e-4),
        #                 OptimizerConfig(name="adamw", lr=2e-4)
        #             ],
        #             scheduler=[
        #                 SchedulerConfig(name="exponential", interval="epoch", gamma=0.999),
        #                 SchedulerConfig(name="exponential", interval="epoch", gamma=0.999)
        #             ]
        #         )
        #
        #     # Plateau schedule conditioned on a monitored metric:
        #     def configure_optimizers(self) -> OptimizationConfiguration:
        #         return OptimizationConfiguration(
        #             optimizer=OptimizerConfig(name="adam", lr=1e-3),
        #             scheduler=SchedulerConfig(
        #                 name="reduce_on_plateau",
        #                 interval="epoch",
        #                 monitor="val_loss"
        #             )
        #         )
        raise NotImplementedError

    def log(
        self,
        name: str,
        value: torch.Tensor | builtins.float,
        configuration: MetricLoggingConfiguration | None = None
    ) -> None:
        # Records one named metric into the module-level buffer that the loops
        # drain after every batch. The call is validated against the execution
        # context: the module must be trainer-attached inside a hook or step,
        # logging is rejected during prediction, and the step and epoch flags
        # are resolved through the metric-logging contract for the hook named
        # by the current marker. Tensor values are detached before buffering
        # so no autograd graph survives past the step.
        #
        # Args:
        #     name: Metric key under which the value reaches the progress
        #         bar, the experiment logger, and the trainer's callback
        #         metrics; step emissions and epoch reductions share this
        #         key.
        #     value: Scalar metric value as a tensor or float; tensors are
        #         detached before buffering.
        #     configuration: Per-call routing options; ``None`` applies the
        #         MetricLoggingConfiguration defaults, whose unset step and
        #         epoch flags resolve through the metric-logging contract
        #         for the hook this call executes in. Default: ``None``.
        #
        # Raises:
        #     MisconfigurationError: If the module is not attached to a
        #         trainer, the call happens outside a trainer-managed hook
        #         or step, or the current stage is prediction.
        #
        # Example::
        #
        #     from syntheticmind.core.module import MetricLoggingConfiguration
        #
        #     def training_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        #         loss: torch.Tensor = ...
        #         self.log("train_loss", loss)
        #         self.log(
        #             "grad_norm",
        #             grad_norm,
        #             MetricLoggingConfiguration(progress_bar=True, on_step=True)
        #         )
        #         return {"loss": loss}
        if self._trainer is None or self._running_stage is None:
            raise MisconfigurationError(
                "self.log() can only be called from within a hook or step method "
                "during trainer.fit(), trainer.validate(), or trainer.test()."
            )
        if self._running_stage == "predicting":
            raise MisconfigurationError(
                "self.log() is not supported in predict_step or predict hooks. "
                "Use the return value of predict_step() instead."
            )
        resolved_configuration: MetricLoggingConfiguration = (
            configuration if configuration is not None else MetricLoggingConfiguration()
        )
        current_fx_name: str | None = self._current_fx_name
        if current_fx_name is None:
            raise MisconfigurationError(
                "self.log() must be called inside a trainer-managed hook or step method."
            )
        resolved_on_step: bool
        resolved_on_epoch: bool
        resolved_on_step, resolved_on_epoch = MetricLoggingContract.resolve(
            hook_name=current_fx_name,
            on_step=resolved_configuration.on_step,
            on_epoch=resolved_configuration.on_epoch
        )

        resolved_value: torch.Tensor | builtins.float = value
        if isinstance(resolved_value, torch.Tensor):
            resolved_value: torch.Tensor | builtins.float = resolved_value.detach()

        metric: LoggedMetric = LoggedMetric(
            value=resolved_value,
            progress_bar=resolved_configuration.progress_bar,
            logger=resolved_configuration.logger,
            on_step=resolved_on_step,
            on_epoch=resolved_on_epoch,
            reduction=resolved_configuration.reduction,
            sync_distributed=resolved_configuration.sync_distributed,
            batch_size=resolved_configuration.batch_size
        )
        self._logged_metrics[name] = metric

    def log_dict(
        self,
        dictionary: dict[str, torch.Tensor | builtins.float],
        configuration: MetricLoggingConfiguration | None = None
    ) -> None:
        # Records a group of named metrics by applying log() to every entry
        # under one shared configuration.
        for name, value in dictionary.items():
            self.log(
                name=name,
                value=value,
                configuration=configuration
            )

    @override
    def to(self, *args: object, **kwargs: object) -> Module:
        # Standard torch device and dtype movement with mirror maintenance.
        # After the move, the cached device and dtype are refreshed from the
        # first parameter, falling back to the first buffer, and finally to
        # the call arguments themselves for parameter-free modules.
        result: Module = super().to(*args, **kwargs)
        param: torch.nn.Parameter | None = next(iter(result.parameters()), None)
        if param is not None:
            result._device: torch.device = param.device
            result._dtype: torch.dtype = param.dtype
        else:
            buf: torch.Tensor | None = next(iter(result.buffers()), None)
            if buf is not None:
                result._device: torch.device = buf.device
                result._dtype: torch.dtype = buf.dtype
            else:
                if args and isinstance(args[0], (torch.device, str)):
                    result._device: torch.device = torch.device(args[0])
                elif "device" in kwargs:
                    result._device: torch.device = torch.device(kwargs["device"])
                if args and isinstance(args[0], torch.dtype):
                    result._dtype: torch.dtype = args[0]
                elif "dtype" in kwargs and isinstance(kwargs["dtype"], torch.dtype):
                    result._dtype: torch.dtype = kwargs["dtype"]
        return result

    @override
    def on_validation_model_eval(self) -> None:
        # Switches the module to evaluation mode when a validation pass
        # begins; the evaluation loop invokes this before its first batch.
        self.eval()

    @override
    def on_validation_model_train(self) -> None:
        # Returns the module to training mode after a validation pass that ran
        # inside fit, so training resumes with the correct mode.
        self.train()

    @override
    def on_test_model_eval(self) -> None:
        # Switches the module to evaluation mode when a test pass begins.
        self.eval()

    @override
    def on_test_model_train(self) -> None:
        # Returns the module to training mode after a test pass when training
        # continues afterwards.
        self.train()

    @override
    def on_predict_model_eval(self) -> None:
        # Switches the module to evaluation mode when a prediction pass
        # begins.
        self.eval()

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        # Manual-optimization entry point that executes one optimizer step
        # through the trainer's epoch loop, so the step hooks, gradient
        # clipping, and scaler handling match the automatic path exactly.
        #
        # Args:
        #     optimizer: The torch.optim.Optimizer to step, as returned by
        #         self.optimizers().
        trainer: Trainer = self.trainer
        epoch_loop: TrainingEpochLoop | None = trainer.fit_loop.epoch_loop
        if epoch_loop is None:
            raise RuntimeError("Trainer fit loop is not initialized.")
        epoch_loop.optimizer_step(trainer, self, optimizer)

    def optimizer_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        # Manual-optimization entry point that clears gradients through the
        # trainer's epoch loop, preserving the zero-grad hook order.
        trainer: Trainer = self.trainer
        epoch_loop: TrainingEpochLoop | None = trainer.fit_loop.epoch_loop
        if epoch_loop is None:
            raise RuntimeError("Trainer fit loop is not initialized.")
        epoch_loop.optimizer_zero_grad(trainer, self, optimizer)

    def manual_backward(self, loss: torch.Tensor) -> None:
        # Manual-optimization entry point for the backward pass. Refused under
        # automatic optimization, where the loop owns backward; otherwise the
        # epoch loop runs backward with the backward hook pair and any active
        # gradient scaler applied.
        #
        # Args:
        #     loss: The torch.Tensor to differentiate; it must carry an
        #         attached graph. Calling this instead of loss.backward()
        #         keeps mixed-precision gradient scaling and the backward
        #         hooks consistent with the automatic path.
        #
        # Example::
        #
        #     def training_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        #         optimizer: torch.optim.Optimizer = self.optimizers()
        #         loss: torch.Tensor = ...
        #         self.optimizer_zero_grad(optimizer)
        #         self.manual_backward(loss)
        #         self.optimizer_step(optimizer)
        #         return {"loss": loss.detach()}
        if self.automatic_optimization:
            raise MisconfigurationError(
                "manual_backward() requires automatic_optimization = False."
            )
        trainer: Trainer = self.trainer
        epoch_loop: TrainingEpochLoop | None = trainer.fit_loop.epoch_loop
        if epoch_loop is None:
            raise RuntimeError("Trainer fit loop is not initialized.")
        epoch_loop.manual_backward(trainer, self, loss)

    def clip_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None
    ) -> None:
        # Manual-optimization entry point for gradient clipping, delegated to
        # the epoch loop so unscaling under mixed precision happens before the
        # threshold is applied.
        trainer: Trainer = self.trainer
        epoch_loop: TrainingEpochLoop | None = trainer.fit_loop.epoch_loop
        if epoch_loop is None:
            raise RuntimeError("Trainer fit loop is not initialized.")
        epoch_loop.clip_gradients(
            optimizer=optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm
        )

    def toggle_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        # Restricts training to one optimizer's parameters during manual
        # optimization. The requires_grad state of every parameter is recorded
        # first, then only the parameters owned by the given optimizer keep
        # gradient tracking enabled, so alternating optimization schemes never
        # accumulate gradients into inactive parameters.
        if self.automatic_optimization:
            raise MisconfigurationError(
                "toggle_optimizer() requires automatic_optimization = False."
            )
        self._param_requires_grad_state: dict[torch.nn.Parameter, bool] = {
            parameter: parameter.requires_grad for parameter in self.parameters()
        }
        optimizer_parameters: set[torch.nn.Parameter] = {
            parameter
            for parameter_group in optimizer.param_groups
            for parameter in parameter_group["params"]
        }
        for parameter in self.parameters():
            parameter.requires_grad: bool = parameter in optimizer_parameters

    def untoggle_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        # Restores the exact requires_grad state recorded by toggle_optimizer
        # and clears the snapshot. The optimizer argument exists for call-site
        # symmetry with toggle_optimizer and is not consulted.
        del optimizer
        for parameter, requires_grad in self._param_requires_grad_state.items():
            parameter.requires_grad: bool = requires_grad
        self._param_requires_grad_state.clear()

    @contextmanager
    def toggled_optimizer(self, optimizer: torch.optim.Optimizer) -> Iterator[torch.optim.Optimizer]:
        # Context manager form of the toggle pair: parameters are restricted
        # on entry and restored in the finally block, so an exception inside
        # the block cannot leave gradient flags in the toggled state.
        #
        # Example::
        #
        #     def training_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        #         optimizer_pair: list[torch.optim.Optimizer] = self.optimizers()
        #         discriminator_opt: torch.optim.Optimizer = optimizer_pair[1]
        #         with self.toggled_optimizer(discriminator_opt):
        #             discriminator_loss: torch.Tensor = ...
        #             self.optimizer_zero_grad(discriminator_opt)
        #             self.manual_backward(discriminator_loss)
        #             self.optimizer_step(discriminator_opt)
        #         ...
        self.toggle_optimizer(optimizer)
        try:
            yield optimizer
        finally:
            self.untoggle_optimizer(optimizer)

    def optimizers(self) -> torch.optim.Optimizer | list[torch.optim.Optimizer] | None:
        # Returns the trainer-materialized optimizers for manual optimization.
        #
        # Return:
        #     ``None`` when the module is detached or no optimizer exists,
        #     the single optimizer unwrapped when exactly one is configured,
        #     or the full list in configure_optimizers order when several
        #     are, so a two-optimizer model can unpack directly:
        #     ``generator_opt, discriminator_opt = self.optimizers()``.
        if self._trainer is None:
            return None
        if not self._trainer.optimizers:
            return None
        if len(self._trainer.optimizers) == 1:
            return self._trainer.optimizers[0]
        return self._trainer.optimizers

    def lr_schedulers(
        self
    ) -> torch.optim.lr_scheduler.LRScheduler | list[torch.optim.lr_scheduler.LRScheduler] | None:
        # Returns the trainer-materialized schedulers with the same unwrapping
        # convention as optimizers().
        if self._trainer is None:
            return None
        if not self._trainer.schedulers:
            return None
        if len(self._trainer.schedulers) == 1:
            return self._trainer.schedulers[0]
        return self._trainer.schedulers

    def freeze(self) -> None:
        # Disables gradient tracking for every parameter and switches to
        # evaluation mode, the standard preparation for a frozen component.
        #
        # Example::
        #
        #     model: AutoEncoder = AutoEncoder()
        #     model.freeze()
        for param in self.parameters():
            param.requires_grad: bool = False
        self.eval()

    def unfreeze(self) -> None:
        # Re-enables gradient tracking for every parameter and returns to
        # training mode, reversing freeze().
        #
        # Example::
        #
        #     model.unfreeze()
        for param in self.parameters():
            param.requires_grad: bool = True
        self.train()

    @override
    def cuda(self, device: int | torch.device | None = None) -> Module:
        # CUDA movement with mirror maintenance: the cached device is
        # refreshed from parameters, then buffers, then the requested index,
        # defaulting to device zero for parameter-free modules.
        result: Module = super().cuda(device)
        param: torch.nn.Parameter | None = next(iter(result.parameters()), None)
        if param is not None:
            result._device: torch.device = param.device
        else:
            buf: torch.Tensor | None = next(iter(result.buffers()), None)
            if buf is not None:
                result._device: torch.device = buf.device
            else:
                cuda_index: int = device if isinstance(device, int) else 0
                result._device: torch.device = torch.device("cuda", cuda_index)
        return result

    @override
    def cpu(self) -> Module:
        # CPU movement with mirror maintenance; the target device is known
        # unconditionally, so the mirror is set directly.
        result: Module = super().cpu()
        result._device: torch.device = torch.device("cpu")
        return result

    @override
    def type(self, dst_type: str | torch.dtype) -> Module:
        # Dtype casting with mirror maintenance: the cached dtype is refreshed
        # from parameters, then buffers, then the requested dtype when it was
        # given as a torch.dtype.
        result: Module = super().type(dst_type)
        param: torch.nn.Parameter | None = next(iter(result.parameters()), None)
        if param is not None:
            result._dtype: torch.dtype = param.dtype
        else:
            buf: torch.Tensor | None = next(iter(result.buffers()), None)
            if buf is not None:
                result._dtype: torch.dtype = buf.dtype
            elif isinstance(dst_type, torch.dtype):
                result._dtype: torch.dtype = dst_type
        return result

    @override
    def float(self) -> Module:
        # Casts to float32 and records the resulting dtype in the mirror.
        result: Module = super().float()
        result._dtype: torch.dtype = torch.float32
        return result

    @override
    def half(self) -> Module:
        # Casts to float16 and records the resulting dtype in the mirror.
        result: Module = super().half()
        result._dtype: torch.dtype = torch.float16
        return result

    @property
    def trainer(self) -> Trainer:
        # The attached trainer. Raises when the module has never been passed
        # to a trainer entry point, because trainer-dependent functionality
        # cannot operate detached.
        if self._trainer is None:
            raise RuntimeError("Module is not attached to a Trainer. Call trainer.fit(module, ...) first")
        return self._trainer

    @property
    def example_input_array(self) -> torch.Tensor | tuple[object, ...] | dict[str, object] | None:
        # Optional example input registered by the model for tooling that
        # needs a representative forward invocation.
        return self._example_input_array

    @example_input_array.setter
    def example_input_array(
        self,
        example_input_array: torch.Tensor | tuple[object, ...] | dict[str, object] | None
    ) -> None:
        # Registers the example input used by tooling that traces or profiles
        # the forward pass.
        self._example_input_array: torch.Tensor | tuple[object, ...] | dict[str, object] | None = example_input_array

    @property
    def automatic_optimization(self) -> bool:
        # Whether the training epoch loop owns backward, stepping, and
        # gradient clearing. Disabled by models that drive optimization
        # manually inside training_step.
        return self._automatic_optimization

    @automatic_optimization.setter
    def automatic_optimization(self, automatic_optimization: bool) -> None:
        # Selects between loop-owned and model-owned optimization. Must be set
        # before fitting because the trainer validates its contract against
        # this flag.
        self._automatic_optimization: bool = automatic_optimization

    @property
    def strict_loading(self) -> bool:
        # Whether state-dict loading should require exact key agreement. The
        # unset state reads as strict, so relaxed loading is always an
        # explicit opt-out.
        return self._strict_loading in (None, True)

    @strict_loading.setter
    def strict_loading(self, strict_loading: bool) -> None:
        # Records the explicit strictness choice for state-dict loading.
        self._strict_loading: bool | None = strict_loading

    @property
    def device(self) -> torch.device:
        # The module's current device, read from the first parameter, then the
        # first buffer, then the cached mirror for parameter-free modules.
        param: torch.nn.Parameter | None = next(iter(self.parameters()), None)
        if param is not None:
            return param.device
        buf: torch.Tensor | None = next(iter(self.buffers()), None)
        if buf is not None:
            return buf.device
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        # The module's current dtype, resolved with the same parameter, buffer,
        # and cached-mirror precedence as the device property.
        param: torch.nn.Parameter | None = next(iter(self.parameters()), None)
        if param is not None:
            return param.dtype
        buf: torch.Tensor | None = next(iter(self.buffers()), None)
        if buf is not None:
            return buf.dtype
        return self._dtype

    @property
    def logger(self) -> Logger | None:
        # The trainer's experiment logger, or None when detached or when the
        # trainer runs without one.
        if self._trainer is None:
            return None
        return self._trainer.logger

    @property
    def current_epoch(self) -> int:
        # Epoch counter mirrored from trainer state by the fit loop.
        return self._current_epoch

    @property
    def global_step(self) -> int:
        # Optimizer-step counter mirrored from trainer state by the epoch
        # loop.
        return self._global_step

    @property
    def global_rank(self) -> int:
        # Global process rank, resolved through the distributed utilities so
        # single-process execution reads as rank zero.
        from syntheticmind.utilities.distributed import get_rank
        return get_rank()

    @property
    def local_rank(self) -> int:
        # Node-local process rank, resolved through the distributed utilities
        # so single-process execution reads as rank zero.
        from syntheticmind.utilities.distributed import get_local_rank
        return get_local_rank()

    @property
    def logged_metrics(self) -> dict[str, LoggedMetric]:
        # Copy of the current logging buffer; returned as a new dictionary so
        # loop-side draining and model-side inspection cannot interfere.
        return dict(self._logged_metrics)

    def _reset_logged_metrics(self) -> None:
        # Clears the logging buffer; invoked by the loops after each dispatch
        # window so values are never dispatched twice.
        self._logged_metrics.clear()


class LoggedMetric:
    # Immutable-by-convention record of one logged value together with its
    # routing metadata: progress-bar and logger visibility, step and epoch
    # dispatch flags, the epoch reduction mode, distributed synchronization,
    # and the batch size for weighted epoch means. Instances live only inside
    # the module's logging buffer between dispatch windows; __slots__ keeps
    # the per-batch allocation cost minimal.
    __slots__: tuple[str, ...] = (
        "value",
        "progress_bar",
        "logger",
        "on_step",
        "on_epoch",
        "reduction",
        "sync_distributed",
        "batch_size"
    )

    def __init__(
        self,
        value: torch.Tensor | float,
        progress_bar: bool,
        logger: bool,
        on_step: bool,
        on_epoch: bool,
        reduction: Reduction,
        sync_distributed: bool,
        batch_size: int | None
    ) -> None:
        # Stores the already-detached value and the fully resolved routing
        # flags; resolution against per-hook defaults happens in Module.log
        # before construction.
        self.value: torch.Tensor | float = value
        self.progress_bar: bool = progress_bar
        self.logger: bool = logger
        self.on_step: bool = on_step
        self.on_epoch: bool = on_epoch
        self.reduction: Reduction = reduction
        self.sync_distributed: bool = sync_distributed
        self.batch_size: int | None = batch_size
