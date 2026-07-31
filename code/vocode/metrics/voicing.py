# This module:
# 1. Accumulates the voiced/unvoiced classification F1 score of candidate
#    audio against reference voicing decisions across a whole evaluation
#    pass
#
# Design decisions:
# - The reference signal's voicing decisions are the ground truth; the
#   candidate's decisions are the predictions, so the score measures how
#   faithfully synthesis preserves the voicing structure
# - Counts accumulate over the pass and the F1 is computed once, weighting
#   every frame equally regardless of utterance length
# - Computing with no observed voiced frame in either role is an error
#
# Author: Rahul Sawhney

import torch

from vocode.metrics.pitch import PitchFeatures

__all__: list[str] = ["VoicingF1"]


class VoicingF1:
    # Pass-level voicing F1 accumulator over shared pitch features.
    #
    # The metric is stateful by design. One utterance pair enters through
    # update and the score is produced once by compute at the end of the
    # pass, so every frame carries equal weight no matter how long the
    # utterance it came from was; averaging per-utterance scores instead
    # would over-weight short utterances.
    #
    # The measurement is a binary classification: the reference signal's
    # voicing decisions are the ground truth and the candidate's are the
    # predictions, which makes the score a statement about how faithfully
    # synthesis preserves voicing structure rather than a symmetric
    # agreement rate. Only three of the four confusion cells are
    # accumulated, because the F1 score excludes true negatives by
    # construction: frames both signals call unvoiced cannot change it.
    #
    # Integration: the metric sequence owns this lifecycle. It calls reset
    # at test start, update with the reference and candidate PitchFeatures
    # of each evaluated pair, and compute once at test end. This class
    # never performs pitch extraction; it reads the bundles the sequence
    # extracted once and shares across the whole pitch family.
    def __init__(self) -> None:
        # Zeroes the confusion counts.
        self._true_positive_count: int = 0
        self._false_positive_count: int = 0
        self._false_negative_count: int = 0

    def reset(self) -> None:
        # Clears the confusion counts at the start of a pass.
        self._true_positive_count: int = 0
        self._false_positive_count: int = 0
        self._false_negative_count: int = 0

    def update(self, reference: PitchFeatures, candidate: PitchFeatures) -> None:
        # Adds one utterance pair's confusion counts over the common frame
        # range, with the reference as truth and the candidate as
        # prediction.
        #
        # Decision sequences of unequal length truncate to the shorter one,
        # so frames past the end of one signal are dropped rather than
        # counted as disagreements against material that does not exist.
        #
        # Args:
        #     reference: Pitch features of the true signal, whose voicing
        #         mask is the ground truth.
        #     candidate: Pitch features of the synthesized signal, whose
        #         voicing mask is the prediction being scored.
        frame_count: int = min(reference.voicing.shape[-1], candidate.voicing.shape[-1])
        reference_voicing: torch.Tensor = reference.voicing[:frame_count]
        candidate_voicing: torch.Tensor = candidate.voicing[:frame_count]
        true_positive_count: int = int((reference_voicing & candidate_voicing).sum().item())
        false_positive_count: int = int((~reference_voicing & candidate_voicing).sum().item())
        false_negative_count: int = int((reference_voicing & ~candidate_voicing).sum().item())
        self._true_positive_count: int = self._true_positive_count + true_positive_count
        self._false_positive_count: int = self._false_positive_count + false_positive_count
        self._false_negative_count: int = self._false_negative_count + false_negative_count

    def compute(self) -> float:
        # Finalizes the pass-level F1 from the accumulated confusion counts.
        #
        # Returns:
        #     The harmonic mean of precision and recall over the whole
        #     pass, computed directly from the three accumulated counts, on
        #     which higher is better. It reaches one when the two signals
        #     agree on every voiced frame and zero when they share no
        #     voiced frame at all.
        #
        # Raises:
        #     ValueError: If neither signal marked a single frame voiced
        #         anywhere in the pass, which leaves the score with no
        #         denominator. That is undefined rather than zero, since
        #         zero would report total voicing disagreement where in
        #         fact the two signals agreed on silence throughout.
        denominator: int = (
            2 * self._true_positive_count
            + self._false_positive_count
            + self._false_negative_count
        )
        if denominator < 1:
            raise ValueError("Voicing F1 is undefined because no voiced frames were observed")
        return float(2.0 * self._true_positive_count / denominator)
