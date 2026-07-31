# This module:
# 1. Accumulates the root-mean-square error between the frame-level
#    periodicity curves of reference and candidate audio across a whole
#    evaluation pass
#
# Design decisions:
# - Periodicity is compared on every common frame, voiced or not, because
#   the curve itself encodes the voicing confidence being judged
# - The metric is stateful (update per utterance, compute at pass end) so
#   the RMSE weights every frame equally across the pass instead of
#   averaging per-utterance values of unequal length
# - Computing with zero observed frames is an error, never a zero
#
# Author: Rahul Sawhney

import torch

from vocode.metrics.pitch import PitchFeatures

__all__: list[str] = ["PeriodicityRmse"]


class PeriodicityRmse:
    # Pass-level periodicity RMSE accumulator over shared pitch features.
    #
    # The metric is stateful by design. One utterance pair enters through
    # update and the score is produced once by compute at the end of the
    # pass, so every observed frame carries equal weight no matter how long
    # the utterance it came from was; averaging per-utterance scores
    # instead would over-weight short utterances.
    #
    # Unlike the F0 metric, no voicing mask is applied: every frame the two
    # signals share is compared, because the periodicity curve is itself
    # the voicing confidence under judgement and masking it by a decision
    # derived from it would be circular.
    #
    # Integration: the metric sequence owns this lifecycle. It calls reset
    # at test start, update with the reference and candidate PitchFeatures
    # of each evaluated pair, and compute once at test end. This class
    # never performs pitch extraction; it reads the bundles the sequence
    # extracted once and shares across the whole pitch family.
    def __init__(self) -> None:
        # Zeroes the squared-error and frame accumulators.
        self._squared_total: float = 0.0
        self._frame_count: int = 0

    def reset(self) -> None:
        # Clears the accumulators at the start of a pass.
        self._squared_total: float = 0.0
        self._frame_count: int = 0

    def update(self, reference: PitchFeatures, candidate: PitchFeatures) -> None:
        # Adds one utterance pair: squared periodicity differences over the
        # common frame range enter the running totals.
        #
        # Curves of unequal length truncate to the shorter one rather than
        # comparing against frames that do not exist. A pair carrying no
        # frames at all contributes nothing and leaves the pass undefined
        # rather than scoring it zero.
        #
        # Args:
        #     reference: Pitch features of the true signal, read for its
        #         periodicity curve alone.
        #     candidate: Pitch features of the synthesized signal, likewise
        #         read for its periodicity curve alone. The voicing masks
        #         of both records are deliberately ignored here.
        frame_count: int = min(reference.periodicity.shape[-1], candidate.periodicity.shape[-1])
        difference: torch.Tensor = (
            reference.periodicity[:frame_count] - candidate.periodicity[:frame_count]
        )
        self._squared_total: float = (
            self._squared_total + float(difference.pow(2).sum().item())
        )
        self._frame_count: int = self._frame_count + frame_count

    def compute(self) -> float:
        # Finalizes the pass-level RMSE; undefined without observed frames.
        #
        # Returns:
        #     The root mean square of the accumulated periodicity
        #     differences, on which lower is better. It is read on whatever
        #     scale the extractor's confidence curve occupies rather than
        #     on a normalized one, so it is comparable only between runs
        #     that shared an extraction protocol.
        #
        # Raises:
        #     ValueError: If the pass observed no frame at all. The
        #         quantity is undefined there, and reporting zero would
        #         claim perfect periodicity agreement on evidence that does
        #         not exist.
        if self._frame_count < 1:
            raise ValueError("Periodicity RMSE is undefined because no frames were extracted")
        return float((self._squared_total / self._frame_count) ** 0.5)
