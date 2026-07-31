# This module:
# 1. Implements the reference LPCNet sparsification schedule over the
#    first GRU's recurrent weights, and the recurrent weight clipper that
#    keeps quantization-friendly magnitudes
#
# Design decisions:
# - Sparsity ramps along the reference schedule against the global step,
#   zeroing the smallest block magnitudes while preserving the diagonal,
#   matching the deployment geometry of the reference
# - Pruning is structured into fixed rectangular blocks rather than applied
#   per weight, because the deployment target gains nothing from scattered
#   zeros: only whole blocks can be skipped by the inference kernel, so the
#   block shape is a property of that kernel and not a tuning knob
# - The diagonal is exempt from pruning. Those weights carry each hidden
#   unit's own recurrence, so removing them would sever a unit's memory of
#   itself rather than merely thinning its connections to others
# - Density anneals cubically rather than linearly, so most of the schedule
#   is spent at high density and the aggressive pruning happens late, giving
#   the network the longest possible period to reorganize before its
#   capacity is actually removed
# - Masking is destructive rather than a retained mask multiplied at use
#   time: the pruned weights are written to zero in place, so no mask tensor
#   has to be checkpointed and the pruned state is visible in the weights
#   themselves. The cost is that the schedule must reapply after every
#   optimizer step, since the update regrows whatever it pruned
#
# Author: Rahul Sawhney

from typing import ClassVar, cast

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from torch import nn

__all__: list[str] = ["LpcnetSparsifier", "LpcnetSparsifierConfig", "LpcnetWeightClipper"]


class LpcnetSparsifierConfig(BaseModel):
    # Frozen sparsification schedule settings of the reference recipe.
    # Explicit validated fields prevent experiment settings from drifting
    # between runs.
    #
    # Fields:
    #     schedule_start_step: Global step below which no pruning occurs,
    #         giving the network an unconstrained warm-up in which to
    #         establish which connections matter before any are removed.
    #         Default: ``2000``.
    #     schedule_end_step: Global step at and beyond which the final
    #         densities apply and pruning is reapplied on every call rather
    #         than only on interval boundaries. Default: ``20000``.
    #     application_interval: Steps between mask applications while the
    #         schedule is ramping. Pruning intermittently rather than every
    #         step lets the optimizer partially recover between
    #         applications, which is what allows surviving weights to absorb
    #         the removed capacity. Default: ``400``.
    #     reset_gate_density: Final fraction of blocks retained in the reset
    #         gate. Default: ``0.05``.
    #     update_gate_density: Final fraction retained in the update gate.
    #         Default: ``0.05``.
    #     state_gate_density: Final fraction retained in the candidate-state
    #         gate, four times the gate densities because that gate carries
    #         the layer's actual content while the other two only modulate
    #         it. Default: ``0.2``.
    #     block_rows: Row extent of one pruning block. The hidden width must
    #         be divisible by this value. Default: ``4``.
    #     block_columns: Column extent of one pruning block. The hidden width
    #         must be divisible by this value. Default: ``8``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    schedule_start_step: PositiveInt = 2000
    schedule_end_step: PositiveInt = 20000
    application_interval: PositiveInt = 400
    reset_gate_density: PositiveFloat = 0.05
    update_gate_density: PositiveFloat = 0.05
    state_gate_density: PositiveFloat = 0.2
    block_rows: PositiveInt = 4
    block_columns: PositiveInt = 8


class LpcnetSparsifier:
    # Training-time component applying the reference block-sparsity schedule
    # to a recurrent weight matrix: cubic density annealing over rectangular
    # magnitude blocks, with the diagonal always retained.
    #
    # The component is stateless apart from its configuration. It derives
    # everything it needs from the global step it is handed, so it holds no
    # counter of its own, requires no checkpointing, and behaves identically
    # whether a run is fresh or resumed.
    #
    # Integration: the three gates of a recurrent layer are pruned
    # independently, because they are stacked into one weight matrix but serve
    # different roles and are given different final densities. The caller
    # invokes this after every optimizer step; the schedule itself decides
    # whether that particular step does any work.
    def __init__(self, configuration: LpcnetSparsifierConfig) -> None:
        # Binds the frozen schedule settings.
        self._configuration: LpcnetSparsifierConfig = configuration

    @torch.no_grad()
    def apply(self, recurrent_gru: nn.GRU, global_step: int) -> None:
        # Applies the scheduled block-sparse mask to the recurrent weight
        # matrix in place, one gate at a time.
        #
        # Three step regimes exist. Before the schedule starts, the method
        # returns immediately and the layer trains dense. During the ramp it
        # acts only on interval boundaries, so the optimizer has room to
        # redistribute between applications. From the end step onward the
        # interval test is bypassed entirely and the final densities are
        # enforced on every call, which is what keeps the layer at its target
        # sparsity for the remainder of training rather than letting the
        # optimizer refill pruned blocks between boundaries.
        #
        # The weight is modified through an in-place copy into a slice view,
        # so the parameter object and any optimizer state keyed to it remain
        # valid. Gradients are disabled because this is a direct edit of the
        # weights and must not enter the graph.
        #
        # Args:
        #     recurrent_gru: The layer whose recurrent weights are pruned. Its
        #         hidden width must divide evenly by both block dimensions.
        #     global_step: Optimizer-step counter driving the schedule.
        if global_step <= self._configuration.schedule_start_step:
            return
        is_final: bool = global_step >= self._configuration.schedule_end_step
        is_interval: bool = (
            (global_step - self._configuration.schedule_start_step)
            % self._configuration.application_interval == 0
        )
        if not is_final and not is_interval:
            return
        # The three gates occupy consecutive row bands of one stacked matrix,
        # so each is addressed as a slice and pruned to its own density.
        recurrent_weight: torch.Tensor = cast(torch.Tensor, recurrent_gru.weight_hh_l0)
        hidden_size: int = recurrent_gru.hidden_size
        gate_densities: tuple[float, float, float] = (
            self._configuration.reset_gate_density,
            self._configuration.update_gate_density,
            self._configuration.state_gate_density
        )
        for gate_index, final_density in enumerate(gate_densities):
            density: float = self._scheduled_density(final_density, global_step)
            gate_slice: torch.Tensor = recurrent_weight[gate_index * hidden_size: (gate_index + 1) * hidden_size, :]
            masked: torch.Tensor = self._block_sparse_mask(gate_slice, density)
            gate_slice.copy_(masked)

    def _scheduled_density(self, final_density: float, global_step: int) -> float:
        # Computes the annealed density for one gate at the given step.
        #
        # The curve runs from fully dense at the schedule start to the target
        # density at the end, and its shape is cubic in the remaining
        # progress. That exponent places almost all of the pruning in the
        # final portion of the schedule: at the halfway point the retained
        # fraction is still far above the target, so the network experiences
        # gentle pressure for most of the ramp and its capacity is removed
        # only once it has had time to consolidate.
        #
        # Args:
        #     final_density: Target retained fraction for this gate.
        #     global_step: Optimizer-step counter.
        #
        # Returns:
        #     The retained fraction to enforce at this step, which equals the
        #     target once the schedule has ended.
        if global_step >= self._configuration.schedule_end_step:
            return final_density
        progress_remaining: float = 1.0 - (
            (global_step - self._configuration.schedule_start_step)
            / (self._configuration.schedule_end_step - self._configuration.schedule_start_step)
        )
        return 1.0 - (1.0 - final_density) * (1.0 - progress_remaining ** 3)

    def _block_sparse_mask(self, gate_weight: torch.Tensor, density: float) -> torch.Tensor:
        # Builds and applies the block magnitude mask for one gate while
        # retaining the diagonal.
        #
        # Blocks are ranked by summed squared magnitude, so a block survives
        # on its total energy rather than on any single large weight; this is
        # what makes the criterion structural, since the inference kernel can
        # only skip a block as a unit. The diagonal is removed before ranking
        # so that a block containing diagonal weights is judged on its
        # off-diagonal content alone and cannot be kept merely because the
        # self-recurrence passing through it is large.
        #
        # Args:
        #     gate_weight: One gate's square weight slice.
        #     density: Fraction of blocks to retain.
        #
        # Returns:
        #     The masked weights. Diagonal entries always survive, so the
        #     realized density slightly exceeds the requested one.
        hidden_size: int = gate_weight.shape[0]
        block_rows: int = self._configuration.block_rows
        block_columns: int = self._configuration.block_columns
        off_diagonal: torch.Tensor = gate_weight - torch.diag(torch.diag(gate_weight))
        blocks: torch.Tensor = off_diagonal.reshape(
            hidden_size // block_rows,
            block_rows,
            hidden_size // block_columns,
            block_columns
        )
        block_energy: torch.Tensor = blocks.pow(2).sum(dim=(1, 3))
        # The cut is located by sorting the energies and indexing at the
        # position the requested density implies, then clamped into range so
        # that a density of zero or one still selects a valid entry rather
        # than indexing off either end.
        sorted_energy: torch.Tensor = block_energy.reshape(-1).sort().values
        threshold_index: int = round(block_energy.numel() * (1.0 - density))
        threshold_index: int = min(max(threshold_index, 0), block_energy.numel() - 1)
        threshold: torch.Tensor = sorted_energy[threshold_index]
        block_mask: torch.Tensor = (block_energy >= threshold).to(gate_weight.dtype)
        expanded_mask: torch.Tensor = block_mask.repeat_interleave(block_rows, dim=0).repeat_interleave(
            block_columns,
            dim=1
        )
        # The block decision is expanded back to weight resolution and the
        # diagonal is added in, then capped at one so a diagonal entry inside
        # a surviving block is not counted twice and scaled up.
        diagonal_mask: torch.Tensor = torch.eye(
            hidden_size,
            device=gate_weight.device,
            dtype=gate_weight.dtype
        )
        combined_mask: torch.Tensor = torch.minimum(
            torch.ones_like(expanded_mask),
            expanded_mask + diagonal_mask
        )
        return gate_weight * combined_mask

    @property
    def configuration(self) -> LpcnetSparsifierConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration


class LpcnetWeightClipper:
    # Training-time component applying the reference pairwise weight
    # constraint. The reference deployment evaluates these matrices with an
    # integer kernel that processes adjacent column pairs together and
    # accumulates at limited width, so what must stay bounded is the summed
    # magnitude of each pair rather than any individual weight. Enforcing that
    # bound during training rather than at export means the network converges
    # to weights the quantized path can represent, instead of being distorted
    # by a clip applied afterwards.
    def __init__(self, clip_value: float = 0.992) -> None:
        # Binds the pairwise magnitude bound.
        #
        # Args:
        #     clip_value: Maximum summed magnitude of each adjacent column
        #         pair. Default: ``0.992``, the reference value, chosen
        #         below unity to leave headroom in the fixed-point
        #         accumulator.
        self._clip_value: float = clip_value

    @torch.no_grad()
    def apply(self, weight: torch.Tensor) -> None:
        # Rescales the weight matrix in place so no adjacent column pair
        # exceeds the bound.
        #
        # Each pair's summed magnitude is computed once and applied to both
        # of its columns, so a violating pair is scaled down proportionally
        # and its two weights keep their relative sizes; hard clipping each
        # weight independently would distort that ratio instead. Taking the
        # maximum against the bound makes the operation a no-op for pairs
        # already inside it, so conforming weights pass through unchanged
        # rather than being scaled up to the limit.
        #
        # Args:
        #     weight: The matrix to constrain, modified in place. Its column
        #         count must be even for the pairing to be exact.
        paired_magnitude: torch.Tensor = torch.abs(weight[:, 1::2]) + torch.abs(weight[:, 0::2])
        repeated: torch.Tensor = paired_magnitude.repeat_interleave(2, dim=1)
        constrained: torch.Tensor = self._clip_value * weight / torch.maximum(
            torch.full_like(repeated, self._clip_value),
            repeated
        )
        weight.copy_(constrained)
