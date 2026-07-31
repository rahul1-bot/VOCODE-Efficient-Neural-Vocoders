# This module:
# 1. Accumulates the fundamental-frequency root-mean-square error in cents
#    between reference and candidate audio across a whole evaluation pass
#
# Design decisions:
# - Only frames voiced in both signals contribute, because pitch is
#   undefined on unvoiced frames and disagreement about voicing itself is
#   measured separately by the voicing F1 metric
# - The error is expressed in cents (1200 times the base-two log ratio), a
#   pitch-perception scale independent of absolute frequency
# - The metric is stateful across the pass so every voiced frame weighs
#   equally; computing with zero jointly-voiced frames is an error
#
# Author: Rahul Sawhney

import torch

from vocode.metrics.pitch import PitchFeatures

__all__: list[str] = ["F0Rmse"]


class F0Rmse:
    # Pass-level F0 RMSE accumulator in cents over shared pitch features.
    #
    # The metric is stateful by design. One utterance pair enters through
    # update and the score is produced once by compute at the end of the
    # pass, so every jointly voiced frame carries equal weight no matter
    # how long the utterance it came from was; averaging per-utterance
    # scores instead would over-weight short utterances.
    #
    # Integration: the metric sequence owns this lifecycle. It calls reset
    # at test start, update with the reference and candidate PitchFeatures
    # of each evaluated pair, and compute once at test end. This class
    # never performs pitch extraction; it reads the bundles the sequence
    # extracted once and shares across the whole pitch family, which is why
    # its input type is the feature record rather than a waveform.
    def __init__(self) -> None:
        # Zeroes the squared-error and voiced-frame accumulators.
        self._squared_total: float = 0.0
        self._voiced_frame_count: int = 0

    def reset(self) -> None:
        # Clears the accumulators at the start of a pass.
        self._squared_total: float = 0.0
        self._voiced_frame_count: int = 0

    def update(self, reference: PitchFeatures, candidate: PitchFeatures) -> None:
        # Adds one utterance pair: squared cent differences over the frames
        # voiced in both signals enter the running totals; pairs without a
        # jointly voiced frame contribute nothing.
        #
        # Tracks of unequal length truncate to the shorter one rather than
        # comparing against frames that do not exist. The joint voicing
        # mask is applied before the logarithm, so an unvoiced frame
        # carrying zero hertz can never reach it and drive the score to
        # negative infinity.
        #
        # Args:
        #     reference: Pitch features of the true signal, supplying one
        #         half of the joint voicing mask.
        #     candidate: Pitch features of the synthesized signal. Its
        #         track is compared only where both signals call the frame
        #         voiced, because pitch is undefined on unvoiced frames and
        #         disagreement about voicing itself is the voicing F1
        #         metric's subject rather than this one's.
        frame_count: int = min(reference.pitch.shape[-1], candidate.pitch.shape[-1])
        both_voiced: torch.Tensor = (
            reference.voicing[:frame_count] & candidate.voicing[:frame_count]
        )
        voiced_count: int = int(both_voiced.sum().item())
        if voiced_count < 1:
            return
        reference_pitch: torch.Tensor = reference.pitch[:frame_count][both_voiced]
        candidate_pitch: torch.Tensor = candidate.pitch[:frame_count][both_voiced]
        cents_difference: torch.Tensor = 1200.0 * (
            torch.log2(reference_pitch) - torch.log2(candidate_pitch)
        )
        self._squared_total: float = (
            self._squared_total + float(cents_difference.pow(2).sum().item())
        )
        self._voiced_frame_count: int = self._voiced_frame_count + voiced_count

    def compute(self) -> float:
        # Finalizes the pass-level RMSE in cents; undefined without jointly
        # voiced frames.
        #
        # Returns:
        #     The root mean square of the accumulated cent differences, on
        #     which lower is better. Cents are a pitch-perception scale of
        #     twelve hundred to the octave and therefore independent of
        #     absolute frequency, which keeps the value comparable across
        #     speakers and registers.
        #
        # Raises:
        #     ValueError: If no frame was voiced in both signals anywhere
        #         in the pass. The quantity is undefined there, and
        #         reporting zero would claim perfect pitch agreement on
        #         evidence that does not exist.
        if self._voiced_frame_count < 1:
            raise ValueError("F0 RMSE is undefined because no frames are voiced in both signals")
        return float((self._squared_total / self._voiced_frame_count) ** 0.5)
