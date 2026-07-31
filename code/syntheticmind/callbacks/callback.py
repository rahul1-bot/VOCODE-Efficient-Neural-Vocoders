# This module:
# 1. Defines the Callback base class: the trainer-side lifecycle vocabulary
#    across setup, fit, validation, test, prediction, sanity checking,
#    optimization, checkpointing, and exception handling
# 2. Defines the callback state-persistence contract: the state_key identity,
#    the state_dict and load_state_dict pair, and the helper for building
#    parameterized state keys
#
# Design decisions:
# - Every hook receives the trainer and module explicitly, so callbacks read
#   shared state (metrics, dataloaders, strategy) through the trainer instead
#   of holding private references that could go stale between runs
# - Every hook defaults to a no-op; the loops invoke callbacks unconditionally
#   and concrete callbacks override only the events they observe
# - At start-style events the callback hooks fire before the module hook of
#   the same name, and at end-style epoch and stage events the module hook
#   fires first, so callbacks at an epoch end observe metric state the module
#   hook has already contributed; on_validation_epoch_end receives the reduced
#   epoch metrics with the module's epoch-end contributions already merged
# - The default state_key is the class name, which is sufficient for
#   single-instance callbacks; _generate_state_key builds parameter-qualified
#   keys so several instances of one class can checkpoint side by side, and
#   the trainer rejects duplicate keys among stateful callbacks before fitting
#
# Author: Rahul Sawhney

from typing import TYPE_CHECKING

import torch

from syntheticmind.core.module import Module
from syntheticmind.utilities.types import (
    Batch,
    CheckpointDict,
    ModelOutput,
    StateDict,
    StateScalar,
    StepOutput,
    TrainerStage,
)

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["Callback"]


class Callback:
    # Base class for trainer-side lifecycle extensions. Concrete callbacks
    # override the hooks they need; monitoring callbacks read reduced metrics
    # from the trainer's callback-metric mapping and the epoch-end metric
    # arguments, and stateful callbacks additionally implement the state
    # persistence pair so their progress survives checkpoint round trips.
    def setup(self, trainer: Trainer, module: Module, stage: TrainerStage) -> None:
        # Invoked by every trainer entry point after the strategy, device, and
        # dataloaders are prepared, before the stage's hooks begin.
        pass

    def teardown(self, trainer: Trainer, module: Module, stage: TrainerStage) -> None:
        # Invoked in every entry point's teardown path, including after
        # failures, so acquired resources can be released unconditionally.
        pass

    def on_fit_start(self, trainer: Trainer, module: Module) -> None:
        # Invoked once per fit before the sanity validation and the first
        # epoch, ahead of the module's own on_fit_start.
        pass

    def on_fit_end(self, trainer: Trainer, module: Module) -> None:
        # Invoked once as the fit stage unwinds, after the module's
        # on_fit_end, including when the run terminated with a failure.
        pass

    def on_train_start(self, trainer: Trainer, module: Module) -> None:
        # Invoked after the sanity validation, immediately before the epoch
        # sequence begins.
        pass

    def on_train_end(self, trainer: Trainer, module: Module) -> None:
        # Invoked after the epoch sequence completes normally, following the
        # module's on_train_end.
        pass

    def on_train_epoch_start(self, trainer: Trainer, module: Module) -> None:
        # Invoked at the start of each training epoch, before the module hook
        # and the batch loop.
        pass

    def on_train_epoch_end(self, trainer: Trainer, module: Module) -> None:
        # Invoked at the end of each training epoch, after the module's
        # epoch-end hook and its metric flush.
        pass

    def on_train_batch_start(
        self, trainer: Trainer, module: Module, batch: Batch, batch_idx: int
    ) -> None:
        # Invoked before batch transfer and the training step for each
        # training batch.
        pass

    def on_train_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int
    ) -> None:
        # Invoked after the training step and optimizer handling for the
        # batch, with the step outputs attached.
        pass

    def on_before_backward(self, trainer: Trainer, module: Module, loss: torch.Tensor) -> None:
        # Invoked immediately before the backward pass with the loss about to
        # be backpropagated.
        pass

    def on_after_backward(self, trainer: Trainer, module: Module) -> None:
        # Invoked immediately after the backward pass while gradients are
        # populated and untouched by the optimizer.
        pass

    def on_before_optimizer_step(
        self, trainer: Trainer, module: Module, optimizer: torch.optim.Optimizer
    ) -> None:
        # Invoked immediately before the optimizer step, ahead of gradient
        # clipping, so implementations observe unclipped gradients.
        pass

    def on_before_zero_grad(
        self, trainer: Trainer, module: Module, optimizer: torch.optim.Optimizer
    ) -> None:
        # Invoked immediately before gradients are cleared after a step, the
        # last point at which the stepped gradients are readable.
        pass

    def on_validation_start(self, trainer: Trainer, module: Module) -> None:
        # Invoked at the beginning of every validation pass, standalone or
        # scheduled inside fit, before the module hook.
        pass

    def on_validation_end(self, trainer: Trainer, module: Module) -> None:
        # Invoked after validation epoch-end handling completes, following the
        # module's on_validation_end.
        pass

    def on_validation_epoch_start(self, trainer: Trainer, module: Module) -> None:
        # Invoked before the first validation batch of a pass.
        pass

    def on_validation_epoch_end(
        self, trainer: Trainer, module: Module, metrics: dict[str, float]
    ) -> None:
        # Invoked after the validation pass with the reduced epoch metrics,
        # including values the module's epoch-end hook contributed. Monitoring
        # callbacks such as early stopping and checkpointing act here.
        pass

    def on_validation_batch_start(
        self, trainer: Trainer, module: Module, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Invoked before batch transfer and the validation step for each
        # validation batch.
        pass

    def on_validation_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Invoked after the validation step for the batch, with the step
        # outputs attached.
        pass

    def on_test_start(self, trainer: Trainer, module: Module) -> None:
        # Invoked at the beginning of a test pass, before the module hook.
        pass

    def on_test_end(self, trainer: Trainer, module: Module) -> None:
        # Invoked after test epoch-end handling completes, following the
        # module's on_test_end.
        pass

    def on_test_epoch_start(self, trainer: Trainer, module: Module) -> None:
        # Invoked before the first test batch of a pass.
        pass

    def on_test_epoch_end(
        self, trainer: Trainer, module: Module, metrics: dict[str, float]
    ) -> None:
        # Invoked after the test pass with the reduced epoch metrics,
        # including values the module's epoch-end hook contributed.
        pass

    def on_test_batch_start(
        self, trainer: Trainer, module: Module, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Invoked before batch transfer and the test step for each test batch.
        pass

    def on_test_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Invoked after the test step for the batch, with the step outputs
        # attached.
        pass

    def on_predict_start(self, trainer: Trainer, module: Module) -> None:
        # Invoked before the first prediction batch, before the module hook.
        pass

    def on_predict_end(self, trainer: Trainer, module: Module) -> None:
        # Invoked after the prediction pass completes, following the module's
        # on_predict_end.
        pass

    def on_sanity_check_start(self, trainer: Trainer, module: Module) -> None:
        # Invoked before the bounded pre-training sanity validation begins.
        pass

    def on_sanity_check_end(self, trainer: Trainer, module: Module) -> None:
        # Invoked after the sanity validation finishes and its metric traces
        # have been discarded.
        pass

    def on_predict_epoch_start(self, trainer: Trainer, module: Module) -> None:
        # Invoked before the first prediction batch of the pass.
        pass

    def on_predict_epoch_end(self, trainer: Trainer, module: Module) -> None:
        # Invoked after the last prediction batch of the pass.
        pass

    def on_predict_batch_start(
        self, trainer: Trainer, module: Module, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Invoked before batch transfer and the predict step for each
        # prediction batch.
        pass

    def on_predict_batch_end(
        self, trainer: Trainer, module: Module, outputs: ModelOutput, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Invoked after the predict step for the batch, with the step outputs
        # attached.
        pass

    def on_exception(self, trainer: Trainer, module: Module, exception: BaseException) -> None:
        # Invoked from the trainer's central exception path before the
        # exception is re-raised, so the callback can persist state or release
        # resources while the run is still inspectable.
        pass

    def on_save_checkpoint(
        self, trainer: Trainer, module: Module, checkpoint: CheckpointDict
    ) -> None:
        # Invoked while a checkpoint payload is being assembled; the callback
        # may insert additional entries into the payload in place.
        pass

    def on_load_checkpoint(
        self, trainer: Trainer, module: Module, checkpoint: CheckpointDict
    ) -> None:
        # Invoked after a checkpoint payload has been restored, so the
        # callback can read back entries it contributed during saving.
        pass

    @classmethod
    def _generate_state_key(cls, **kwargs: StateScalar) -> str:
        # Builds a parameter-qualified state key of the form
        # ClassName{key=value, ...} with the pairs sorted for determinism, so
        # multiple differently configured instances of one callback class can
        # persist state side by side without collision.
        pairs: list[str] = [f"{k}={v!r}" for k, v in sorted(kwargs.items())]
        return f"{cls.__name__}{{{', '.join(pairs)}}}"

    @property
    def state_key(self) -> str:
        # Identity under which this callback's state is stored in checkpoint
        # payloads. Defaults to the class name; stateful callbacks that can
        # exist in multiples override this with a parameter-qualified key.
        return type(self).__name__

    def state_dict(self) -> StateDict:
        # Serializes callback state into checkpoints. The default contributes
        # nothing; stateful callbacks override this together with
        # load_state_dict.
        state_dict: StateDict = {}
        return state_dict

    def load_state_dict(self, state_dict: StateDict) -> None:
        # Restores callback state recorded by state_dict during resumption.
        pass

    def __repr__(self) -> str:
        # Compact identity line for logs and interactive debugging.
        return f"{type(self).__name__}()"
