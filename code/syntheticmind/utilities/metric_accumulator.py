# This module:
# 1. Accumulates scalar metric observations across steps and reduces them to a
#    single value per logging window
# 2. Supports mean, sum, min, and max reductions, with the mean optionally
#    weighted by per-step batch size
# 3. Provides reset semantics so one instance serves consecutive epochs or
#    evaluation windows without reallocation
#
# Design decisions:
# - The mean prefers batch-size weighting whenever any weights were supplied,
#   because averaging per-batch means directly would let a smaller final batch
#   bias the window mean
# - compute() on an empty window returns 0.0 instead of raising, so summary
#   emission never fails on a metric that logged nothing in the current window
# - Min and max trackers start at plus and minus infinity so the first
#   observation always establishes the initial extreme
# - State lives in plain Python floats; tensors are detached and converted at
#   the logging boundary, keeping the accumulator free of autograd references
# - __slots__ is declared because one accumulator exists per logged metric name
#   and long runs hold many instances simultaneously
#
# Author: Rahul Sawhney

from typing import Literal

__all__: list[str] = ["MetricAccumulator"]

type ReductionMode = Literal["mean", "sum", "min", "max"]


class MetricAccumulator:
    # Streaming reducer for one named metric. Callers push observations with
    # update() during a window, read the reduced value with compute() at the
    # window boundary, and clear with reset() before the next window. The
    # reduction mode is fixed at construction so a metric cannot silently
    # change semantics mid-run.
    __slots__: tuple[str, ...] = ("_reduction", "_sum", "_count", "_min", "_max", "_weighted_sum", "_total_weight")

    def __init__(self, reduction: ReductionMode = "mean") -> None:
        # Fixes the reduction mode and zeroes every accumulator: plain sum and
        # count for unweighted means, running extremes for min and max, and the
        # weighted pair used when callers report batch sizes.
        self._reduction: ReductionMode = reduction
        self._sum: float = 0.0
        self._count: int = 0
        self._min: float = float("inf")
        self._max: float = float("-inf")
        self._weighted_sum: float = 0.0
        self._total_weight: float = 0.0

    def update(self, value: float, batch_size: int | None = None) -> None:
        # Adds one observation: always advances the unweighted sum, count, and
        # extremes; additionally advances the weighted pair when a batch size is
        # given, which arms batch-size-weighted averaging in compute().
        self._sum += value
        self._count += 1
        if value < self._min:
            self._min: float = value
        if value > self._max:
            self._max: float = value
        if batch_size is not None:
            self._weighted_sum += value * batch_size
            self._total_weight += batch_size

    def compute(self) -> float:
        # Reduces the window: sum, min, and max read their accumulators
        # directly; the mean divides the weighted sum by total weight when any
        # weights were recorded and otherwise falls back to the unweighted mean
        # over update calls. An empty window reduces to 0.0.
        if self._count == 0:
            return 0.0
        match self._reduction:
            case "sum":
                return self._sum
            case "min":
                return self._min
            case "max":
                return self._max
            case _:
                if self._total_weight > 0:
                    return self._weighted_sum / self._total_weight
                return self._sum / self._count

    def reset(self) -> None:
        # Returns every accumulator to its initial state so the instance can be
        # reused for the next window.
        self._sum: float = 0.0
        self._count: int = 0
        self._min: float = float("inf")
        self._max: float = float("-inf")
        self._weighted_sum: float = 0.0
        self._total_weight: float = 0.0

    @property
    def reduction(self) -> ReductionMode:
        # Reduction mode fixed at construction.
        return self._reduction

    @property
    def count(self) -> int:
        # Number of observations recorded in the current window.
        return self._count
