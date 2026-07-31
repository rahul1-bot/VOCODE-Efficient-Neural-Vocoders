# This module:
# 1. Implements the strategy contract for multi-process data-parallel execution
#    on top of torch.distributed and DistributedDataParallel
# 2. Validates the launch environment, initializes and destroys the process
#    group, and wraps the model for gradient-synchronized training
# 3. Overrides the collective operations (barrier, broadcast, boolean decision
#    reduction, all-gather) with real torch.distributed implementations
#
# Design decisions:
# - setup_environment validates its preconditions before touching the process
#   group: the NCCL backend is refused when CUDA is unavailable, and the four
#   launcher-provided environment variables (RANK, WORLD_SIZE, MASTER_ADDR,
#   MASTER_PORT) are required so a missing torchrun launch fails with a precise
#   configuration error instead of a hang inside init_process_group
# - The training step executes through the DistributedDataParallel wrapper
#   rather than the unwrapped module, because gradient bucketing hooks fire only
#   when the wrapper's own forward path runs. The strategy temporarily redirects
#   the wrapped module's forward to its training_step, invokes the wrapper, and
#   restores the original forward in a finally block, guaranteeing restoration
#   even when the step raises
# - Evaluation, test, and prediction steps deliberately bypass the wrapper and
#   call the unwrapped module directly, because no gradients exist in those
#   stages and wrapper synchronization would add cost without effect
# - reduce_boolean_decision reduces with a logical-OR convention, implemented as
#   an integer sum compared against zero, so any single rank requesting an
#   action (such as early stopping) commits every rank to it and no rank
#   continues alone
# - teardown destroys the process group so consecutive runs inside one process
#   can re-initialize cleanly
#
# Author: Rahul Sawhney

import os
from collections.abc import Callable
from typing import override

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from syntheticmind.core.module import Module
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.utilities.types import Batch, ModelOutput, StepOutput

__all__: list[str] = ["DDPStrategy"]


class DDPStrategy(Strategy):
    # Distributed data-parallel strategy. Each participating process owns one
    # device and one replica; this class manages the process-group lifecycle,
    # the DistributedDataParallel wrapping, and the collective operations the
    # trainer and callbacks invoke through the Strategy interface.
    def __init__(self, backend: str = "nccl") -> None:
        # Records the requested communication backend. The local rank is
        # resolved later, during environment setup, from the launcher-provided
        # environment variables.
        super().__init__()
        self._backend: str = backend
        self._local_rank: int = 0

    @override
    def setup_environment(self) -> None:
        # Initializes the process group exactly once per process. The method
        # first validates that the backend is usable and that the launcher
        # environment is complete, so misconfiguration surfaces as an immediate
        # MisconfigurationError with corrective guidance rather than a
        # rendezvous hang inside torch.distributed.
        if not dist.is_initialized():
            from syntheticmind.utilities.exceptions import MisconfigurationError

            if self._backend == "nccl" and not torch.cuda.is_available():
                raise MisconfigurationError(
                    "DDPStrategy with backend='nccl' requires CUDA, but CUDA is not available. "
                    "Use DDPStrategy(backend='gloo') for CPU distributed training, "
                    "or ensure CUDA is properly installed."
                )

            required_env_vars: list[str] = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
            missing_env_vars: list[str] = [
                var for var in required_env_vars if var not in os.environ
            ]
            if missing_env_vars:
                raise MisconfigurationError(
                    f"DDPStrategy requires environment variables {missing_env_vars}, "
                    f"but they are not set. Launch with: "
                    f"torchrun --nproc_per_node=<NUM_GPUS> your_script.py"
                )

            self._local_rank: int = int(os.environ.get("LOCAL_RANK", "0"))
            dist.init_process_group(backend=self._backend)

    @override
    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:
        # Places the replica on this process's device and wraps it in
        # DistributedDataParallel. The device index is passed to the wrapper
        # only for CUDA devices, because the CPU-based gloo path expects no
        # device_ids argument.
        self._device: torch.device = device
        placed_model: nn.Module = model.to(device)
        self._model: nn.Module | None = DistributedDataParallel(
            placed_model,
            device_ids=[device.index] if device.type == "cuda" else None
        )
        return self._model

    @override
    def training_step(
        self, model: nn.Module, batch: Batch, batch_idx: int
    ) -> StepOutput:
        # Executes the training step through the DistributedDataParallel
        # forward path so gradient-bucketing hooks participate in the step.
        # The wrapped module's forward attribute is temporarily redirected to a
        # closure that restores the original forward and invokes training_step;
        # calling the wrapper then routes through that closure while the hooks
        # remain armed. The finally block guarantees the original forward is
        # restored even when the step raises.
        if isinstance(model, DistributedDataParallel):
            unwrapped: Module = model.module  # type: ignore[assignment]
            original_forward: Callable[..., object] = unwrapped.forward

            def redirected_forward(batch: Batch, batch_idx: int) -> StepOutput:
                # Restores the module's original forward before delegating, so
                # any forward call made inside training_step reaches the true
                # model computation rather than this redirection closure.
                unwrapped.forward: Callable[..., object] = original_forward  # type: ignore[assignment]
                return unwrapped.training_step(batch, batch_idx)

            unwrapped.forward: Callable[..., object] = redirected_forward  # type: ignore[assignment]
            try:
                output: StepOutput = model(batch, batch_idx)
            finally:
                unwrapped.forward: Callable[..., object] = original_forward  # type: ignore[assignment]
            return output
        if not isinstance(model, Module):
            raise TypeError(f"Expected Module, got {type(model).__name__}")
        return model.training_step(batch, batch_idx)

    @override
    def validation_step(
        self, model: nn.Module, batch: Batch, batch_idx: int
    ) -> StepOutput:
        # Executes the validation step on the unwrapped module. Gradient
        # synchronization is irrelevant without a backward pass, so the
        # DistributedDataParallel forward path is intentionally bypassed.
        unwrapped: nn.Module = model.module if isinstance(model, DistributedDataParallel) else model
        if not isinstance(unwrapped, Module):
            raise TypeError(f"Expected Module, got {type(unwrapped).__name__}")
        return unwrapped.validation_step(batch, batch_idx)

    @override
    def test_step(
        self, model: nn.Module, batch: Batch, batch_idx: int
    ) -> StepOutput:
        # Executes the test step on the unwrapped module, mirroring the
        # validation-step rationale for bypassing the wrapper.
        unwrapped: nn.Module = model.module if isinstance(model, DistributedDataParallel) else model
        if not isinstance(unwrapped, Module):
            raise TypeError(f"Expected Module, got {type(unwrapped).__name__}")
        return unwrapped.test_step(batch, batch_idx)

    @override
    def predict_step(
        self, model: nn.Module, batch: Batch, batch_idx: int
    ) -> ModelOutput:
        # Executes the prediction step on the unwrapped module, mirroring the
        # validation-step rationale for bypassing the wrapper.
        unwrapped: nn.Module = model.module if isinstance(model, DistributedDataParallel) else model
        if not isinstance(unwrapped, Module):
            raise TypeError(f"Expected Module, got {type(unwrapped).__name__}")
        return unwrapped.predict_step(batch, batch_idx)

    @override
    def backward(self, loss: torch.Tensor) -> None:
        # Executes the standard autograd backward pass. Gradient averaging
        # across ranks is performed by the DistributedDataParallel hooks armed
        # during the training-step forward, not by this method.
        loss.backward()

    @override
    def barrier(self) -> None:
        # Blocks until every rank reaches this synchronization point. The
        # initialization guard keeps the method callable before setup.
        if dist.is_initialized():
            dist.barrier()

    @override
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        # Broadcasts the tensor from the source rank to all ranks in place and
        # returns it, so call sites can use the result directly.
        if dist.is_initialized():
            dist.broadcast(tensor, src=src)
        return tensor

    @property
    @override
    def is_distributed(self) -> bool:
        # This strategy always coordinates multiple processes.
        return True

    @property
    def local_rank(self) -> int:
        # Node-local rank resolved from the launcher environment during
        # environment setup. Zero before setup and in single-node launches
        # without an explicit LOCAL_RANK.
        return self._local_rank

    @property
    def world_size(self) -> int:
        # Number of participating processes, or one before the process group
        # is initialized.
        if dist.is_initialized():
            return dist.get_world_size()
        return 1

    @override
    def reduce_boolean_decision(self, decision: bool) -> bool:
        # Agrees on a boolean control decision across ranks with logical-OR
        # semantics. The local decision is encoded as an integer, summed across
        # ranks, and compared against zero, so a request raised by any single
        # rank commits every rank to the same outcome.
        if not dist.is_initialized():
            return decision
        decision_tensor: torch.Tensor = torch.tensor(int(decision), device=self._device)
        dist.all_reduce(decision_tensor, op=dist.ReduceOp.SUM)
        return bool(decision_tensor.item() > 0)

    @override
    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        # Gathers the tensor from every rank into a new leading dimension. A
        # zero-initialized buffer is allocated per rank and filled by the
        # collective, and the buffers are stacked so the caller receives one
        # tensor of shape (world_size, *tensor.shape).
        if not dist.is_initialized():
            return tensor
        gathered: list[torch.Tensor] = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        return torch.stack(gathered)

    @override
    def teardown(self) -> None:
        # Destroys the process group so a subsequent run inside the same
        # process can initialize a fresh group without conflict.
        if dist.is_initialized():
            dist.destroy_process_group()
