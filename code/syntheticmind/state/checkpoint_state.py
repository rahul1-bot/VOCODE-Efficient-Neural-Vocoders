# This module:
# 1. Assembles the complete checkpoint payload: model weights, per-optimizer and
#    per-scheduler state, random-number streams, callback state keyed by state_key,
#    optional datamodule state, mid-epoch resume markers, and environment metadata
# 2. Restores that payload back onto live objects while tolerating legacy
#    single-optimizer and single-scheduler layouts
# 3. Collects and restores the Python, NumPy, torch, and CUDA random streams
#
# Design decisions:
# - The payload writes both the list form (optimizer_state_dicts,
#   scheduler_state_dicts) and the first entry under the legacy singular keys, so
#   older readers and single-optimizer tooling keep working against new files
# - restore() pairs optimizers and schedulers with saved states through
#   zip(strict=False) because a resumed configuration may carry a different count
#   than the checkpoint; unmatched objects simply start fresh
# - A checkpoint without optimizer state is tolerated with a warning rather than
#   an error, because exception-time checkpoints can capture a partially
#   constructed run whose weights are still worth restoring
# - completed_batches and epoch_rng_state enter the payload only when mid-epoch
#   resume is actually in play, keeping epoch-boundary checkpoints minimal
# - torch and CUDA RNG tensors are moved to CPU before restoration because the
#   set_rng_state APIs require CPU byte tensors regardless of the producing device
# - Callback state is keyed by each callback's state_key so two instances of the
#   same class with different configurations never overwrite each other
#
# Author: Rahul Sawhney

import random

import numpy as np
import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.datamodule import DataModule
from syntheticmind.utilities.types import CheckpointDict, CheckpointValue, StateDict

__all__: list[str] = ["CheckpointState"]


class CheckpointState:
    # Stateless assembler and restorer for checkpoint payloads, exposed as
    # classmethods because the payload schema, not instance state, is the thing
    # being managed. build() and restore() are exact inverses over that schema.

    @classmethod
    def build(
        cls,
        model: torch.nn.Module,
        optimizers: list[torch.optim.Optimizer],
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        schedulers: list[torch.optim.lr_scheduler.LRScheduler] | None,
        epoch: int,
        global_step: int,
        callbacks: list[Callback],
        datamodule: DataModule | None = None,
        completed_batches: int = 0,
        epoch_rng_state: CheckpointDict | None = None,
        strategy_name: str = "SingleDeviceStrategy",
        world_size: int = 1
    ) -> CheckpointDict:
        # Assembles the payload from live objects. Progress counters and model
        # weights always travel; optimizer state travels in list form plus the
        # legacy first-entry key; RNG streams are captured at call time so a
        # restored run continues the same random sequences.
        checkpoint: CheckpointDict = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizers[0].state_dict(),
            "optimizer_state_dicts": [optimizer.state_dict() for optimizer in optimizers],
            "rng_state": cls._collect_rng_states()
        }

        if completed_batches > 0:
            checkpoint["completed_batches"] = completed_batches

        if epoch_rng_state is not None:
            checkpoint["epoch_rng_state"] = epoch_rng_state

        scheduler_list: list[torch.optim.lr_scheduler.LRScheduler] = []
        if schedulers is not None:
            scheduler_list: list[torch.optim.lr_scheduler.LRScheduler] = schedulers
        elif scheduler is not None:
            scheduler_list: list[torch.optim.lr_scheduler.LRScheduler] = [scheduler]
        if scheduler_list:
            checkpoint["scheduler_state_dicts"] = [
                current_scheduler.state_dict() for current_scheduler in scheduler_list
            ]
            checkpoint["scheduler_state_dict"] = scheduler_list[0].state_dict()

        callback_states: dict[str, StateDict] = {}
        for callback in callbacks:
            state: StateDict = callback.state_dict()
            if state:
                callback_states[callback.state_key] = state
        checkpoint["callback_states"] = callback_states

        if datamodule is not None:
            datamodule_state: StateDict = datamodule.state_dict()
            if datamodule_state:
                checkpoint["datamodule_state_dict"] = datamodule_state

        checkpoint["metadata"] = {
            "torch_version": torch.__version__,
            "strategy": strategy_name,
            "world_size": world_size
        }

        return checkpoint

    @classmethod
    def restore(
        cls,
        checkpoint: CheckpointDict,
        model: torch.nn.Module,
        optimizers: list[torch.optim.Optimizer],
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        schedulers: list[torch.optim.lr_scheduler.LRScheduler] | None,
        callbacks: list[Callback],
        datamodule: DataModule | None = None
    ) -> tuple[int, int, int, CheckpointDict | None]:
        # Rehydrates live objects from the payload and returns the resume
        # coordinates (epoch, global step, completed batches, epoch RNG state).
        # The list-form optimizer and scheduler keys are preferred; the legacy
        # singular keys are the fallback for older checkpoints.
        model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore[arg-type]
        optimizer_state_dicts_raw: CheckpointValue | None = checkpoint.get("optimizer_state_dicts")
        if isinstance(optimizer_state_dicts_raw, list):
            for optimizer, optimizer_state_dict in zip(optimizers, optimizer_state_dicts_raw, strict=False):
                optimizer.load_state_dict(optimizer_state_dict)  # type: ignore[arg-type]
        elif "optimizer_state_dict" in checkpoint and optimizers:
            optimizers[0].load_state_dict(checkpoint["optimizer_state_dict"])  # type: ignore[arg-type]
        else:
            from loguru import logger as restore_log
            restore_log.warning(
                "Checkpoint missing optimizer_state_dict (partial/exception checkpoint); "
                "optimizer will start fresh."
            )

        scheduler_state_dicts_raw: CheckpointValue | None = checkpoint.get("scheduler_state_dicts")
        scheduler_list: list[torch.optim.lr_scheduler.LRScheduler] = []
        if schedulers is not None:
            scheduler_list: list[torch.optim.lr_scheduler.LRScheduler] = schedulers
        elif scheduler is not None:
            scheduler_list: list[torch.optim.lr_scheduler.LRScheduler] = [scheduler]
        if isinstance(scheduler_state_dicts_raw, list):
            for current_scheduler, scheduler_state_dict in zip(
                scheduler_list,
                scheduler_state_dicts_raw,
                strict=False
            ):
                current_scheduler.load_state_dict(scheduler_state_dict)  # type: ignore[arg-type]
        elif scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  # type: ignore[arg-type]

        rng_state: CheckpointValue | None = checkpoint.get("rng_state")
        if isinstance(rng_state, dict):
            cls._restore_rng_states(rng_state)

        callback_states_raw: CheckpointValue | None = checkpoint.get("callback_states")
        callback_states: dict[str, StateDict] = (
            callback_states_raw if isinstance(callback_states_raw, dict) else {}
        )
        for callback in callbacks:
            key: str = callback.state_key
            if key in callback_states:
                callback.load_state_dict(callback_states[key])

        datamodule_state_raw: CheckpointValue | None = checkpoint.get("datamodule_state_dict")
        if datamodule is not None and isinstance(datamodule_state_raw, dict):
            datamodule.load_state_dict(datamodule_state_raw)

        metadata_raw: CheckpointValue | None = checkpoint.get("metadata")
        metadata: CheckpointDict = metadata_raw if isinstance(metadata_raw, dict) else {}
        if metadata:
            from loguru import logger as log
            log.info(f"Checkpoint metadata: torch={metadata.get('torch_version')}")

        epoch_value: CheckpointValue | None = checkpoint.get("epoch", 0)
        global_step_value: CheckpointValue | None = checkpoint.get("global_step", 0)
        completed_batches_value: CheckpointValue | None = checkpoint.get("completed_batches", 0)
        epoch: int = int(epoch_value if epoch_value is not None else 0)
        global_step: int = int(global_step_value if global_step_value is not None else 0)
        completed_batches: int = int(completed_batches_value if completed_batches_value is not None else 0)
        epoch_rng_state_raw: CheckpointValue | None = checkpoint.get("epoch_rng_state")
        epoch_rng_state: CheckpointDict | None = (
            epoch_rng_state_raw if isinstance(epoch_rng_state_raw, dict) else None
        )
        return epoch, global_step, completed_batches, epoch_rng_state

    @classmethod
    def _collect_rng_states(cls) -> CheckpointDict:
        # Captures every random stream the harness seeds: Python, NumPy, torch
        # CPU, and one state per visible CUDA device when CUDA is present.
        states: CheckpointDict = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state()
        }
        if torch.cuda.is_available():
            states["cuda"] = torch.cuda.get_rng_state_all()
        return states

    @classmethod
    def _restore_rng_states(cls, rng_state: CheckpointDict) -> None:
        # Restores each stream that is present, skipping absent keys so payloads
        # from CPU-only producers restore cleanly on CUDA hosts and vice versa.
        # Tensor states are moved to CPU first because the torch restore APIs
        # accept only CPU byte tensors.
        if "python" in rng_state:
            random.setstate(rng_state["python"])  # type: ignore[arg-type]
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])  # type: ignore[arg-type]
        if "torch" in rng_state:
            torch_rng: CheckpointValue = rng_state["torch"]
            if isinstance(torch_rng, torch.Tensor):
                torch_rng: CheckpointValue = torch_rng.cpu()
            torch.random.set_rng_state(torch_rng)  # type: ignore[arg-type]
        if "cuda" in rng_state and torch.cuda.is_available():
            cuda_states: CheckpointValue = rng_state["cuda"]
            if isinstance(cuda_states, list):
                cuda_states: CheckpointValue = [s.cpu() if isinstance(s, torch.Tensor) else s for s in cuda_states]
            torch.cuda.set_rng_state_all(cuda_states)  # type: ignore[arg-type]
