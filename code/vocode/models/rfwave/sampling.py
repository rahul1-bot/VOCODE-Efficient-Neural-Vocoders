# This module:
# 1. Implements RFWave synthesis: Euler integration of the learned
#    velocity field from noise to the band spectra over the configured
#    step count, followed by the inverse spectral reconstruction to the
#    waveform
#
# Integration semantics:
# - Training defines a velocity field along a straight path from noise to
#   data. Synthesis follows that field from time zero to time one, and the
#   Euler rule advances the state by the predicted velocity times the step
#   size at each of a fixed number of uniform steps
# - The published baseline is ten steps, which is the cost figure this
#   family's efficiency claim rests on: each step is one full forward pass,
#   so synthesis costs ten passes where a single-pass vocoder costs one.
#   Reducing the count shortens synthesis monotonically but sub-proportionally,
#   because the endpoint construction and the final reconstruction are paid
#   once regardless of how many steps run
# - Ten steps suffice only because the path is straight. On a straight path
#   the exact solution is one linear step, and any residual error comes from
#   the learned field deviating from that ideal rather than from curvature
#   the integrator must track; a curved formulation would need far more
#
# Design decisions:
# - The step count is the sampler's single knob and the seam the Study 2
#   solver-step reduction exchanges, holding weights constant while the
#   integration schedule varies. Reducing it is therefore a pure inference
#   intervention: it changes cost and quality without retraining, which is
#   what makes the trade-off measurable on one Retained Project Checkpoint
#   and places this family's step variants in their own intervention family
#   alongside compilation, reduced precision, quantization, exported
#   runtimes, and masked pruning with recovery
# - The sampler holds no reference to a flow and receives one per call, so a
#   single sampler can serve several models and a step-count variant is a
#   new sampler rather than a mutated one
# - Integration runs the full band set jointly at each step, matching
#   training-time conditioning
# - Each predicted velocity is projected onto the consistent-spectrum set
#   before it is applied, so inconsistency cannot accumulate across steps
#
# Author: Rahul Sawhney

import torch

from vocode.models.rfwave.flow import RfwaveRectifiedFlow

__all__: list[str] = ["RfwaveOdeSampler"]


class RfwaveOdeSampler:
    # Euler ordinary-differential-equation sampler over the learned rectified flow.
    # Follows the reference joint-parallel sampling: all bands advance together from the
    # noise endpoint along the predicted velocity field in a fixed number of steps.
    def __init__(self, step_count: int = 10) -> None:
        # Binds the integration step count, the sampler's only state.
        #
        # Args:
        #     step_count: Number of uniform Euler steps taken from time zero
        #         to time one. Each step costs one forward pass through the
        #         backbone, so this value is very nearly a multiplier on
        #         synthesis time. Default: ``10``, the published baseline.
        self._step_count: int = step_count

    @torch.no_grad()
    def synthesize(self, flow: RfwaveRectifiedFlow, mel: torch.Tensor) -> torch.Tensor:
        # Integrates the learned velocity field from the noise endpoint to the
        # data endpoint and reconstructs the waveform from the final state.
        #
        # The conditioning is expanded exactly as training expands it, with
        # the mel repeated per band and the band index cycling fastest, so the
        # backbone sees the same row layout at inference that it was trained
        # under. Gradients are disabled throughout, because the loop would
        # otherwise retain a graph across every step.
        #
        # Args:
        #     flow: The flow to integrate, supplied per call rather than held,
        #         so one sampler can serve several models.
        #     mel: Conditioning mel of shape ``[batch, channels, frames]``.
        #
        # Returns:
        #     The synthesized waveform, de-equalized and reconstructed from
        #     the final state.
        state: torch.Tensor = flow.noise_endpoint(mel)
        expanded_mel: torch.Tensor = torch.repeat_interleave(mel, flow.band_count, 0)
        band_index: torch.Tensor = torch.tile(
            torch.arange(flow.band_count, device=mel.device),
            (mel.size(0),)
        )
        # The grid carries one more point than there are steps, so each step
        # reads its own start time and the following point supplies the step
        # size; the spacing is uniform, but computing it from the grid keeps
        # the loop correct if the grid were ever made non-uniform.
        time_grid: torch.Tensor = torch.linspace(0.0, 1.0, self._step_count + 1, device=mel.device)
        for step in range(self._step_count):
            step_size: torch.Tensor = time_grid[step + 1] - time_grid[step]
            # Every row is evaluated at the same time, because integration
            # advances all samples and all bands in lockstep.
            time_values: torch.Tensor = torch.full(
                (state.size(0),),
                float(time_grid[step]),
                device=mel.device
            )
            velocity: torch.Tensor = flow.predict_velocity(state, time_values, expanded_mel, band_index)
            # The predicted velocity is projected before it is applied, so the
            # state never drifts off the set of realizable spectra.
            velocity: torch.Tensor = flow.project_to_consistent_spectrum(velocity)
            state: torch.Tensor = state + velocity * step_size
        return flow.waveform_from_joint(state)

    @property
    def step_count(self) -> int:
        # Returns the number of Euler integration steps.
        return self._step_count
