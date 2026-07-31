# This module:
# 1. Seeds every random-number source the harness touches: Python random, NumPy,
#    torch CPU generators, and all visible CUDA devices
# 2. Optionally switches PyTorch into deterministic-algorithms mode together with the
#    matching cuDNN and cuBLAS settings
# 3. Provides the dataloader worker initializer that reseeds Python and NumPy inside
#    each worker process
#
# Design decisions:
# - torch.use_deterministic_algorithms is called with warn_only=deterministic so an
#   operation without a deterministic implementation degrades to a warning instead of
#   aborting the run
# - CUBLAS_WORKSPACE_CONFIG is set via setdefault before any cuBLAS call because
#   deterministic cuBLAS requires it, while an explicit value exported by the caller
#   is left untouched
# - cudnn.deterministic and cudnn.benchmark are flipped as a pair: benchmark
#   autotuning selects kernels nondeterministically, so it is enabled only when
#   determinism is off and disabled when determinism is requested
# - PYTHONHASHSEED is pinned so hash-order-dependent iteration cannot vary between
#   otherwise identical runs
#
# Author: Rahul Sawhney

import os
import random

import numpy as np
import torch

__all__: list[str] = ["SeedManager"]


class SeedManager:
    # Process-wide seeding entry points, grouped as classmethods so call sites read
    # as SeedManager.seed_everything(...) without constructing an instance. One call
    # covers the main process; worker_init_fn covers forked dataloader workers.

    @classmethod
    def seed_everything(cls, seed: int, deterministic: bool = False) -> None:
        # Seeds Python random, NumPy, torch CPU, and all CUDA generators from one
        # value and pins PYTHONHASHSEED. When deterministic is requested, also
        # enforces deterministic algorithm selection and disables cuDNN benchmark
        # autotuning; otherwise benchmark mode stays on for throughput.
        #
        # Args:
        #     seed: Value applied to every random source and pinned into
        #         PYTHONHASHSEED.
        #     deterministic: When true, pins the cuBLAS workspace, enforces
        #         deterministic algorithm selection, and disables cuDNN
        #         benchmark autotuning, trading throughput for bitwise
        #         repeatability. Default: ``False``.
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(deterministic, warn_only=deterministic)
        os.environ["PYTHONHASHSEED"] = str(seed)

        if deterministic:
            torch.backends.cudnn.deterministic: bool = True
            torch.backends.cudnn.benchmark: bool = False
        else:
            torch.backends.cudnn.deterministic: bool = False
            torch.backends.cudnn.benchmark: bool = True

    @classmethod
    def worker_init_fn(cls, worker_id: int) -> None:
        # Dataloader worker initializer. PyTorch already assigns each worker a
        # distinct initial seed; this reseeds Python random and NumPy from it,
        # reduced modulo 2**32 because NumPy accepts only 32-bit seeds. Without
        # this, every worker would share one NumPy augmentation stream. The
        # worker_id argument is required by the DataLoader contract but the seed
        # derivation does not need it.
        del cls, worker_id
        worker_seed: int = int(torch.initial_seed() % (2 ** 32))
        random.seed(worker_seed)
        np.random.seed(worker_seed)
