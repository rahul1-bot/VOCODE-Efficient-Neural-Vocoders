# This module:
# 1. Defines ModelHooks, the complete lifecycle vocabulary the trainer and
#    loops invoke on the module across the fit, validation, test, and
#    prediction stages
# 2. Defines DataHooks, the batch-transfer trio invoked around device
#    placement for every batch
# 3. Defines the metric-logging contract: the per-hook rules that resolve and
#    validate the step and epoch routing of every Module.log call
#
# Design decisions:
# - Every model hook defaults to a no-op so subclasses override only the
#   events they need; the loops invoke the hooks unconditionally and rely on
#   the defaults being free of side effects
# - The logging contract partitions the hook vocabulary into four families:
#   training-step-adjacent hooks default to per-step logging, evaluation-step
#   hooks default to per-epoch logging, epoch-boundary hooks are locked to
#   per-epoch logging, and a blocked family (prediction, checkpointing, batch
#   transfer, model-mode switches, and configuration hooks) rejects logging
#   outright
# - Contract resolution fails closed: a hook name outside every registered
#   family raises rather than silently accepting the call, so a newly added
#   hook must be classified before logging works inside it
# - The rules are frozen value objects held in class-level tables, making the
#   full logging policy auditable in one place rather than scattered through
#   the loops
#
# Author: Rahul Sawhney

from typing import ClassVar

import torch
from pydantic import BaseModel, ConfigDict

from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.types import Batch, ModelOutput, StepOutput

__all__: list[str] = ["DataHooks", "ModelHooks", "MetricLoggingRule", "MetricLoggingContract"]


class MetricLoggingRule(BaseModel):
    # Frozen routing rule for one hook family: the admissible values for the
    # step and epoch flags and the defaults applied when a log call leaves
    # them unset.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    allowed_on_step: tuple[bool, ...]
    allowed_on_epoch: tuple[bool, ...]
    default_on_step: bool
    default_on_epoch: bool

    def resolve(self, hook_name: str, on_step: bool | None, on_epoch: bool | None) -> tuple[bool, bool]:
        # Applies the family defaults to unset flags, then validates the
        # resolved pair against the admissible values, raising a
        # configuration error that names the offending hook so the illegal
        # call site is identifiable from the message alone.
        resolved_on_step: bool = self.default_on_step if on_step is None else on_step
        resolved_on_epoch: bool = self.default_on_epoch if on_epoch is None else on_epoch
        if resolved_on_step not in self.allowed_on_step:
            raise MisconfigurationError(
                f"self.log(on_step={resolved_on_step}) is invalid inside {hook_name}."
            )
        if resolved_on_epoch not in self.allowed_on_epoch:
            raise MisconfigurationError(
                f"self.log(on_epoch={resolved_on_epoch}) is invalid inside {hook_name}."
            )
        return resolved_on_step, resolved_on_epoch


class MetricLoggingContract:
    # Central classification of every hook name into its logging family,
    # consulted by Module.log through the hook-name marker the loops maintain.
    # Training-step-adjacent hooks log per step by default, evaluation-step
    # hooks log per epoch by default, epoch-boundary hooks accept only
    # per-epoch logging, and the blocked family rejects logging entirely.
    _training_step_rule: ClassVar[MetricLoggingRule] = MetricLoggingRule(
        allowed_on_step=(False, True),
        allowed_on_epoch=(False, True),
        default_on_step=True,
        default_on_epoch=False
    )
    _evaluation_step_rule: ClassVar[MetricLoggingRule] = MetricLoggingRule(
        allowed_on_step=(False, True),
        allowed_on_epoch=(False, True),
        default_on_step=False,
        default_on_epoch=True
    )
    _epoch_rule: ClassVar[MetricLoggingRule] = MetricLoggingRule(
        allowed_on_step=(False,),
        allowed_on_epoch=(True,),
        default_on_step=False,
        default_on_epoch=True
    )
    _training_step_hook_names: ClassVar[tuple[str, ...]] = (
        "training_step",
        "on_before_backward",
        "on_after_backward",
        "on_before_optimizer_step",
        "on_before_zero_grad",
        "on_train_batch_start",
        "on_train_batch_end"
    )
    _evaluation_step_hook_names: ClassVar[tuple[str, ...]] = (
        "validation_step",
        "test_step",
        "on_validation_batch_start",
        "on_validation_batch_end",
        "on_test_batch_start",
        "on_test_batch_end"
    )
    _epoch_hook_names: ClassVar[tuple[str, ...]] = (
        "on_train_start",
        "on_validation_start",
        "on_test_start",
        "on_train_epoch_start",
        "on_train_epoch_end",
        "on_validation_epoch_start",
        "on_validation_epoch_end",
        "on_test_epoch_start",
        "on_test_epoch_end"
    )
    _blocked_hook_names: ClassVar[tuple[str, ...]] = (
        "on_fit_start",
        "on_fit_end",
        "on_train_end",
        "on_validation_end",
        "on_test_end",
        "on_predict_start",
        "on_predict_end",
        "on_predict_epoch_start",
        "on_predict_epoch_end",
        "on_predict_batch_start",
        "on_predict_batch_end",
        "on_before_batch_transfer",
        "transfer_batch_to_device",
        "on_after_batch_transfer",
        "on_validation_model_eval",
        "on_validation_model_train",
        "on_test_model_eval",
        "on_test_model_train",
        "on_predict_model_eval",
        "on_save_checkpoint",
        "on_load_checkpoint",
        "predict_step",
        "configure_optimizers"
    )

    @classmethod
    def resolve(cls, hook_name: str, on_step: bool | None, on_epoch: bool | None) -> tuple[bool, bool]:
        # Routes the hook name to its family rule and returns the resolved
        # step and epoch flags. Blocked hooks and unregistered hook names both
        # raise, so logging is impossible from any context the contract does
        # not explicitly permit.
        if hook_name in cls._training_step_hook_names:
            return cls._training_step_rule.resolve(hook_name=hook_name, on_step=on_step, on_epoch=on_epoch)
        if hook_name in cls._evaluation_step_hook_names:
            return cls._evaluation_step_rule.resolve(hook_name=hook_name, on_step=on_step, on_epoch=on_epoch)
        if hook_name in cls._epoch_hook_names:
            return cls._epoch_rule.resolve(hook_name=hook_name, on_step=on_step, on_epoch=on_epoch)
        if hook_name in cls._blocked_hook_names:
            raise MisconfigurationError(f"self.log() cannot be used inside {hook_name}.")
        raise MisconfigurationError(f"self.log() is not registered for trainer hook {hook_name}.")


class DataHooks:
    # Batch-transfer hook trio invoked by every loop around device placement.
    # The loops give a datamodule override of any of these hooks precedence
    # over the module-side implementation, decided per hook, so batch
    # representation logic can live with either the data or the model.
    def on_before_batch_transfer(self, batch: Batch, dataloader_idx: int = 0) -> Batch:
        # Invoked on the host-side batch after the batch-start hooks and
        # before device placement; the returned batch is what gets moved.
        return batch

    def transfer_batch_to_device(self, batch: Batch, device: torch.device, dataloader_idx: int = 0) -> Batch:
        # Places the batch on the target device. The default delegates to the
        # shared recursive transfer helper; override for batches whose
        # placement the generic traversal cannot express.
        from syntheticmind.utilities.data_transfer import move_data_to_device

        return move_data_to_device(batch, device)

    def on_after_batch_transfer(self, batch: Batch, dataloader_idx: int = 0) -> Batch:
        # Invoked on the device-resident batch immediately before the step;
        # the returned batch is what the step receives.
        return batch


class ModelHooks:
    # Complete model-side lifecycle vocabulary. The trainer and loops invoke
    # these methods at fixed points; every default is a no-op, and the
    # corresponding callback hooks fire alongside them so both extension
    # surfaces observe the same events.
    def on_fit_start(self) -> None:
        # Invoked once by trainer.fit after setup and checkpoint restoration,
        # before the sanity validation and the first epoch.
        pass

    def on_fit_end(self) -> None:
        # Invoked once in trainer.fit teardown, including after failures, as
        # the fit stage unwinds.
        pass

    def on_train_start(self) -> None:
        # Invoked by trainer.fit after the sanity validation, immediately
        # before the epoch sequence begins.
        pass

    def on_train_end(self) -> None:
        # Invoked by trainer.fit immediately after the epoch sequence
        # completes normally.
        pass

    def on_validation_start(self) -> None:
        # Invoked by the evaluation loop at the beginning of every validation
        # pass, standalone or scheduled within fit.
        pass

    def on_validation_end(self) -> None:
        # Invoked by the evaluation loop after validation epoch-end handling
        # completes.
        pass

    def on_test_start(self) -> None:
        # Invoked by the evaluation loop at the beginning of a test pass.
        pass

    def on_test_end(self) -> None:
        # Invoked by the evaluation loop after test epoch-end handling
        # completes.
        pass

    def on_predict_start(self) -> None:
        # Invoked by the prediction loop before the first prediction batch.
        pass

    def on_predict_end(self) -> None:
        # Invoked by the prediction loop after the last prediction batch and
        # the prediction epoch-end hooks.
        pass

    def on_train_epoch_start(self) -> None:
        # Invoked by the fit loop at the start of each training epoch, before
        # the batch loop begins.
        pass

    def on_train_epoch_end(self) -> None:
        # Invoked by the fit loop after the batch loop finishes; metrics
        # logged here are flushed before the callback epoch-end hooks fire.
        pass

    def on_validation_epoch_start(self) -> None:
        # Invoked by the evaluation loop before the first validation batch of
        # a pass.
        pass

    def on_validation_epoch_end(self) -> None:
        # Invoked by the evaluation loop after the last validation batch;
        # metrics logged here join the epoch metrics the monitoring callbacks
        # receive.
        pass

    def on_test_epoch_start(self) -> None:
        # Invoked by the evaluation loop before the first test batch of a
        # pass.
        pass

    def on_test_epoch_end(self) -> None:
        # Invoked by the evaluation loop after the last test batch; metrics
        # logged here join the epoch metrics the monitoring callbacks receive.
        pass

    def on_predict_epoch_start(self) -> None:
        # Invoked by the prediction loop before the first prediction batch.
        pass

    def on_predict_epoch_end(self) -> None:
        # Invoked by the prediction loop after the last prediction batch.
        pass

    def on_validation_model_eval(self) -> None:
        # Invoked when a validation pass begins so the module can switch to
        # evaluation mode; Module overrides this with an eval() call.
        pass

    def on_validation_model_train(self) -> None:
        # Invoked when training resumes after a validation pass inside fit;
        # Module overrides this with a train() call.
        pass

    def on_test_model_eval(self) -> None:
        # Invoked when a test pass begins so the module can switch to
        # evaluation mode.
        pass

    def on_test_model_train(self) -> None:
        # Invoked when training resumes after a test pass.
        pass

    def on_predict_model_eval(self) -> None:
        # Invoked when a prediction pass begins so the module can switch to
        # evaluation mode.
        pass

    def on_train_batch_start(self, batch: Batch, batch_idx: int) -> None:
        # Invoked by the training epoch loop before batch transfer and the
        # training step for each batch.
        pass

    def on_train_batch_end(self, outputs: StepOutput, batch: Batch, batch_idx: int) -> None:
        # Invoked by the training epoch loop after the training step and the
        # optimizer handling for the batch, with the step outputs attached.
        pass

    def on_validation_batch_start(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Invoked by the evaluation loop before batch transfer and the
        # validation step for each batch.
        pass

    def on_validation_batch_end(self, outputs: StepOutput, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Invoked by the evaluation loop after the validation step for the
        # batch, with the step outputs attached.
        pass

    def on_test_batch_start(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Invoked by the evaluation loop before batch transfer and the test
        # step for each batch.
        pass

    def on_test_batch_end(self, outputs: StepOutput, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Invoked by the evaluation loop after the test step for the batch,
        # with the step outputs attached.
        pass

    def on_predict_batch_start(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Invoked by the prediction loop before batch transfer and the predict
        # step for each batch.
        pass

    def on_predict_batch_end(self, outputs: ModelOutput, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Invoked by the prediction loop after the predict step for the batch,
        # with the step outputs attached.
        pass

    def on_before_backward(self, loss: torch.Tensor) -> None:
        # Invoked immediately before the backward pass with the loss about to
        # be backpropagated, after the corresponding callback hooks.
        pass

    def on_after_backward(self) -> None:
        # Invoked immediately after the backward pass, while gradients are
        # populated and before any optimizer handling.
        pass

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        # Invoked immediately before the optimizer step, ahead of gradient
        # clipping, so implementations observe unclipped gradients.
        pass

    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        # Invoked immediately before gradients are cleared after an optimizer
        # step, the last point at which the stepped gradients are readable.
        pass
