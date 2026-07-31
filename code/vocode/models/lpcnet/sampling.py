# This module:
# 1. Implements autoregressive LPCNet synthesis: frame-conditioned
#    sample-by-sample excitation sampling with temperature and threshold
#    controls, reconstructing the waveform through the linear-prediction
#    filter
#
# Design decisions:
# - The sampler resolves the network through a provider callable at every
#   synthesis call, so a transformed network (for example dynamic INT8) is
#   the object that actually synthesizes; a retained stale reference would
#   invalidate intervention measurements
# - Sampling follows the reference's pitch-adaptive temperature and
#   probability-floor scheme
# - The loop is genuinely sequential over samples and cannot be batched
#   along time, because each step's inputs are functions of the sample
#   produced immediately before it. The batch axis is the only parallelism
#   available, which is why
#   the recurrent states and the sample history are carried as batched
#   tensors rather than per-utterance scalars
# - Both distribution adjustments push toward periodicity rather than
#   novelty: sharpening concentrates the distribution on strongly voiced
#   frames, and the floor removes the long tail of near-zero levels whose
#   accumulated mass would otherwise inject audible noise
#
# Author: Rahul Sawhney

from collections.abc import Callable
from typing import ClassVar

import torch
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat

from vocode.models.lpcnet.network import LpcnetNetwork

__all__: list[str] = ["LpcnetSampler", "LpcnetSamplerConfig"]


class LpcnetSamplerConfig(BaseModel):
    # Frozen sampling settings of the reference synthesis scheme. Explicit
    # validated fields prevent experiment settings from drifting between runs.
    #
    # Fields:
    #     correlation_sharpening_scale: Slope of the affine map from pitch
    #         correlation to the sharpening amount. Default: ``1.5``.
    #     correlation_sharpening_offset: Intercept of that map. Together with
    #         the scale it sets the correlation below which no sharpening is
    #         applied, since the amount is floored at zero; at the reference
    #         values that threshold is one third, so only clearly periodic
    #         frames are sharpened. Default: ``-0.5``.
    #     probability_floor: Constant subtracted from every level's
    #         probability before renormalization. Subtracting rather than
    #         thresholding means the floor removes the same mass from every
    #         level, so levels below it are eliminated outright while the
    #         relative ordering of the survivors is preserved. Zero disables
    #         the mechanism. Default: ``0.002``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    correlation_sharpening_scale: PositiveFloat = 1.5
    correlation_sharpening_offset: float = -0.5
    probability_floor: NonNegativeFloat = 0.002


class LpcnetSampler:
    # Inference component running the batched autoregressive LPCNet synthesis loop.
    # The sampling protocol follows the reference inference path: pitch-correlation
    # sharpened probabilities with a constant floor subtraction before sampling.
    # The network is resolved through the provider at every synthesis call so that
    # quantized or otherwise replaced networks are the networks actually executed.
    def __init__(
        self,
        network_provider: Callable[[], LpcnetNetwork],
        configuration: LpcnetSamplerConfig
    ) -> None:
        # Stores the provider and settings. The network is deliberately not
        # stored: holding the callable instead of its result is what keeps
        # this component correct across a network replacement.
        #
        # Args:
        #     network_provider: Callable returning the network to execute,
        #         invoked once per synthesis call rather than once here.
        #     configuration: Frozen sampling settings.
        self._network_provider: Callable[[], LpcnetNetwork] = network_provider
        self._configuration: LpcnetSamplerConfig = configuration

    @torch.no_grad()
    def synthesize(
        self,
        conditioning: torch.Tensor,
        pitch_index: torch.Tensor,
        pitch_correlation: torch.Tensor,
        lpc_coefficients: torch.Tensor
    ) -> torch.Tensor:
        # Generates int16-scaled waveforms autoregressively from frame-rate
        # features. The conditioning encoder runs once over all frames before
        # the loop begins, so the only work inside the sequential path is the
        # per-sample core.
        #
        # The loop is nested: the outer level walks frames, fixing the
        # conditioning vector, the prediction coefficients, and the sharpening
        # exponent for the samples it governs, and the inner level walks the
        # samples of that frame. Gradients are disabled throughout because
        # this path exists only for inference; retaining a graph across
        # hundreds of thousands of sequential steps would exhaust memory.
        #
        # Args:
        #     conditioning: Frame-rate analysis features.
        #     pitch_index: Frame-rate pitch-period indices.
        #     pitch_correlation: Per-frame pitch correlation in ``[-1, 1]``,
        #         which selects how sharply each frame is sampled.
        #     lpc_coefficients: Frame-rate prediction coefficients. Their last
        #         axis defines the filter order and therefore the length of
        #         the sample history carried across steps.
        #
        # Returns:
        #     The synthesized waveform of shape ``[batch, frames * frame_size]``
        #     on the int16 scale, clamped to that range at every step.
        network: LpcnetNetwork = self._network_provider()
        condition_vectors: torch.Tensor = network.encode_conditioning(conditioning, pitch_index)
        batch_size: int = condition_vectors.shape[0]
        frame_count: int = condition_vectors.shape[1]
        frame_size: int = network.frame_size
        lpc_order: int = lpc_coefficients.shape[-1]
        device: torch.device = condition_vectors.device
        dtype: torch.dtype = condition_vectors.dtype
        # Initial state. The sample history starts silent, and the previous
        # excitation starts at the mu-law centre index, which is the companded
        # representation of zero rather than an arbitrary midpoint; both
        # encode the assumption that the utterance begins from silence. The
        # recurrent states start at zero, matching how training begins each
        # sequence.
        sample_history: torch.Tensor = torch.zeros(batch_size, lpc_order, device=device, dtype=dtype)
        previous_excitation: torch.Tensor = torch.full((batch_size, 1), 128.0, device=device, dtype=dtype)
        first_gru_state: torch.Tensor = torch.zeros(
            1,
            batch_size,
            network.first_gru.hidden_size,
            device=device,
            dtype=dtype
        )
        second_gru_state: torch.Tensor = torch.zeros(
            1,
            batch_size,
            network.second_gru.hidden_size,
            device=device,
            dtype=dtype
        )
        synthesized_frames: list[torch.Tensor] = []
        for frame_index in range(frame_count):
            frame_condition: torch.Tensor = condition_vectors[:, frame_index: frame_index + 1, :]
            frame_coefficients: torch.Tensor = lpc_coefficients[:, frame_index, :]
            frame_sharpening: torch.Tensor = self._sharpening_exponent(pitch_correlation[:, frame_index])
            frame_samples: list[torch.Tensor] = []
            for _ in range(frame_size):
                # The prediction reproduces what the training path computes in
                # bulk, one sample at a time. The history is reversed so its
                # most recent entry meets the first coefficient, matching the
                # tap ordering of the batched form, and the sign is negated
                # under the same inverse-filter convention.
                prediction: torch.Tensor = -(frame_coefficients * sample_history.flip(-1)).sum(dim=-1, keepdim=True)
                previous_sample: torch.Tensor = sample_history[:, -1:]
                mulaw_inputs: torch.Tensor = torch.stack(
                    [
                        network.mulaw.linear_to_mulaw(previous_sample),
                        network.mulaw.linear_to_mulaw(prediction),
                        previous_excitation
                    ],
                    dim=-1
                )
                embedded: torch.Tensor = network.signal_embedding(mulaw_inputs)
                flat_embedded: torch.Tensor = embedded.reshape(batch_size, 1, -1)
                first_input: torch.Tensor = torch.cat([flat_embedded, frame_condition], dim=-1)
                first_output: torch.Tensor
                first_output, first_gru_state = network.first_gru(first_input, first_gru_state)
                second_input: torch.Tensor = torch.cat([first_output, frame_condition], dim=-1)
                second_output: torch.Tensor
                second_output, second_gru_state = network.second_gru(second_input, second_gru_state)
                node_outputs: torch.Tensor = network.dual_fully_connected(second_output)
                probabilities: torch.Tensor = network.expand_tree_probabilities(node_outputs).squeeze(1)
                excitation_index: torch.Tensor = self._sample_excitation(probabilities, frame_sharpening)
                excitation_value: torch.Tensor = network.mulaw.mulaw_to_linear(
                    excitation_index.to(dtype).unsqueeze(-1)
                )
                # The sample is the prediction plus the decoded excitation,
                # clamped to the representable range so a divergent step
                # cannot propagate an unbounded value into the history and
                # destabilize every sample that follows.
                new_sample: torch.Tensor = (prediction + excitation_value).clamp(-32767.0, 32767.0)
                frame_samples.append(new_sample)
                # State advance: the history slides forward by one, and the
                # excitation is carried in its index form because that is the
                # domain the embedding expects at the next step.
                sample_history: torch.Tensor = torch.cat([sample_history[:, 1:], new_sample], dim=-1)
                previous_excitation: torch.Tensor = excitation_index.to(dtype).unsqueeze(-1)
            synthesized_frames.append(torch.cat(frame_samples, dim=-1))
        return torch.cat(synthesized_frames, dim=-1)

    def _sharpening_exponent(self, pitch_correlation: torch.Tensor) -> torch.Tensor:
        # Computes the pitch-adaptive sharpening exponent for one frame of
        # each batch row. The affine map is floored at zero before being added
        # to one, so the exponent never falls below unity: a weakly periodic
        # frame is sampled from the network's distribution unchanged, and only
        # a strongly periodic one is sharpened. Sharpening is applied as an
        # exponent, so raising it concentrates mass on the already-likely
        # levels and suppresses the tail.
        #
        # The direction is deliberate. Voiced speech is close to
        # deterministic given the pitch, and sampling it too freely produces
        # audible roughness, whereas unvoiced speech is genuinely noisy and
        # must not be sharpened at all.
        #
        # Returns:
        #     Exponents of shape ``[batch]``, each at least one.
        sharpening: torch.Tensor = (
            self._configuration.correlation_sharpening_scale * pitch_correlation
            + self._configuration.correlation_sharpening_offset
        )
        return 1.0 + sharpening.clamp_min(0.0)

    def _sample_excitation(self, probabilities: torch.Tensor, sharpening: torch.Tensor) -> torch.Tensor:
        # Draws one mu-law excitation index per batch row from the sharpened
        # and floored distribution.
        #
        # Three steps run in order. Sharpening raises each level to the
        # frame's exponent, which reweights without reordering. The floor is
        # then subtracted and negatives clipped away, which eliminates the
        # long tail of barely-possible levels whose combined mass would
        # otherwise be drawn often enough to be audible. Renormalization
        # restores a valid distribution.
        #
        # The uniform fallback covers the case where the floor removed
        # everything, which happens when the network's output is diffuse
        # enough that no level exceeds the floor. Falling back to a uniform
        # draw keeps synthesis running instead of failing on a degenerate
        # distribution; the divisor is additionally floored to guard the
        # normalization itself against an underflowed total.
        #
        # Returns:
        #     Sampled indices of shape ``[batch]``.
        sharpened: torch.Tensor = probabilities.pow(sharpening.unsqueeze(-1))
        floored: torch.Tensor = (sharpened - self._configuration.probability_floor).clamp_min(0.0)
        totals: torch.Tensor = floored.sum(dim=-1, keepdim=True)
        uniform_fallback: torch.Tensor = torch.full_like(floored, 1.0 / floored.shape[-1])
        normalized: torch.Tensor = torch.where(totals > 0.0, floored / totals.clamp_min(1e-12), uniform_fallback)
        return torch.multinomial(normalized, num_samples=1).squeeze(-1)
