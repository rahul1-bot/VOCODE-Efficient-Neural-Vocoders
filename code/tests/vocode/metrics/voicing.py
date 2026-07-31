# This module:
# 1. Verifies the pass-level voicing F1 accumulator over fabricated
#    pitch-feature records: the confusion arithmetic, the exclusion of true
#    negatives, ragged-length truncation, frame-weighted accumulation across
#    utterances, and the undefined-result error path
#
# Design decisions:
# - Pitch features are fabricated directly as PitchFeatures records instead
#   of extracted, because the accumulator consumes only that record type and
#   real extraction would load pretrained CREPE weights for zero added
#   coverage of this module
# - Voicing patterns are chosen so the true-positive, false-positive, and
#   false-negative counts are readable off the fixture, making the expected
#   F1 exact rational arithmetic rather than a pinned floating-point sample
# - Frame weighting is checked with utterances of deliberately unequal
#   agreement, because a per-utterance average agrees with the frame-weighted
#   result on uniform inputs and would hide the defect
#
# Author: Rahul Sawhney

import unittest

import torch

from vocode.metrics.pitch import PitchFeatures
from vocode.metrics.voicing import VoicingF1


class VoicingDecisionFeatureBuilder:
    # Builds fabricated pitch-feature records carrying explicit voicing
    # decisions for the accumulator under test.
    def __init__(self, filler_pitch_hertz: float, filler_periodicity: float) -> None:
        # Binds the pitch and periodicity values used to fill the fields the
        # voicing metric never reads.
        self._filler_pitch_hertz: float = filler_pitch_hertz
        self._filler_periodicity: float = filler_periodicity

    def build(self, voicing_flags: list[bool]) -> PitchFeatures:
        # Assembles one record from an explicit sequence of voicing
        # decisions.
        frame_count: int = len(voicing_flags)
        return PitchFeatures(
            pitch=torch.full((frame_count,), self._filler_pitch_hertz, dtype=torch.float32),
            periodicity=torch.full((frame_count,), self._filler_periodicity, dtype=torch.float32),
            voicing=torch.tensor(voicing_flags, dtype=torch.bool)
        )


class VoicingF1ConfusionArithmeticTest(unittest.TestCase):
    # Verifies that the score follows from the accumulated true-positive,
    # false-positive, and false-negative counts.
    def setUp(self) -> None:
        # Prepares a fresh accumulator and a record builder.
        self._metric: VoicingF1 = VoicingF1()
        self._builder: VoicingDecisionFeatureBuilder = VoicingDecisionFeatureBuilder(220.0, 0.5)

    def test_perfect_agreement_scores_one(self) -> None:
        # Identical voicing decisions leave no false positive or negative.
        voicing_flags: list[bool] = [True, False, True, True]
        self._metric.update(self._builder.build(voicing_flags), self._builder.build(voicing_flags))
        measured: float = self._metric.compute()
        self.assertAlmostEqual(
            measured,
            1.0,
            places=6,
            msg=f"Identical voicing decisions must score one, measured {measured}"
        )

    def test_known_confusion_counts_produce_the_harmonic_score(self) -> None:
        # Two true positives against one false positive and one false
        # negative score four sixths.
        reference: PitchFeatures = self._builder.build([True, True, False, True])
        candidate: PitchFeatures = self._builder.build([True, True, True, False])
        self._metric.update(reference, candidate)
        measured: float = self._metric.compute()
        expected: float = 4.0 / 6.0
        self.assertAlmostEqual(
            measured,
            expected,
            places=6,
            msg=f"Expected {expected} from two hits, one miss, and one false alarm, measured {measured}"
        )

    def test_candidate_missing_every_voiced_frame_scores_zero(self) -> None:
        # Predicting silence over voiced reference material yields no true
        # positive at all.
        reference: PitchFeatures = self._builder.build([True, True, True])
        candidate: PitchFeatures = self._builder.build([False, False, False])
        self._metric.update(reference, candidate)
        measured: float = self._metric.compute()
        self.assertAlmostEqual(
            measured,
            0.0,
            places=6,
            msg=f"A candidate voicing nothing must score zero, measured {measured}"
        )

    def test_candidate_voicing_only_unvoiced_frames_scores_zero(self) -> None:
        # Predicting voice over unvoiced reference material also yields no
        # true positive.
        reference: PitchFeatures = self._builder.build([False, False, False])
        candidate: PitchFeatures = self._builder.build([True, True, True])
        self._metric.update(reference, candidate)
        measured: float = self._metric.compute()
        self.assertAlmostEqual(
            measured,
            0.0,
            places=6,
            msg=f"A candidate voicing only silence must score zero, measured {measured}"
        )

    def test_true_negatives_are_excluded_from_the_score(self) -> None:
        # Appending frames both signals call unvoiced changes no confusion
        # count and therefore no score.
        short_metric: VoicingF1 = VoicingF1()
        short_metric.update(
            self._builder.build([True, True, False, True]),
            self._builder.build([True, True, True, False])
        )
        self._metric.update(
            self._builder.build([True, True, False, True, False, False]),
            self._builder.build([True, True, True, False, False, False])
        )
        short_score: float = short_metric.compute()
        padded_score: float = self._metric.compute()
        self.assertAlmostEqual(
            short_score,
            padded_score,
            places=6,
            msg=f"Agreed unvoiced frames changed the score from {short_score} to {padded_score}"
        )


class VoicingF1FrameCoverageTest(unittest.TestCase):
    # Verifies which frames enter the confusion counts across ragged and
    # empty inputs.
    def setUp(self) -> None:
        # Prepares a fresh accumulator and a record builder.
        self._metric: VoicingF1 = VoicingF1()
        self._builder: VoicingDecisionFeatureBuilder = VoicingDecisionFeatureBuilder(220.0, 0.5)

    def test_ragged_decisions_truncate_to_the_common_frame_count(self) -> None:
        # Frames beyond the shorter sequence are dropped rather than counted
        # as disagreements.
        reference: PitchFeatures = self._builder.build([True, True, False, False])
        candidate: PitchFeatures = self._builder.build([True, True])
        self._metric.update(reference, candidate)
        measured: float = self._metric.compute()
        self.assertAlmostEqual(
            measured,
            1.0,
            places=6,
            msg=f"Trailing reference frames must be ignored, measured {measured}"
        )

    def test_empty_features_contribute_no_counts(self) -> None:
        # A pair carrying no frames leaves the pass undefined rather than
        # scoring zero.
        self._metric.update(self._builder.build([]), self._builder.build([]))
        with self.assertRaisesRegex(ValueError, "undefined"):
            self._metric.compute()


class VoicingF1AccumulationLifecycleTest(unittest.TestCase):
    # Verifies pass-level accumulation, the reset contract, and the
    # undefined-result error path.
    def setUp(self) -> None:
        # Prepares a fresh accumulator and a record builder.
        self._metric: VoicingF1 = VoicingF1()
        self._builder: VoicingDecisionFeatureBuilder = VoicingDecisionFeatureBuilder(220.0, 0.5)

    def test_accumulation_weights_frames_rather_than_utterances(self) -> None:
        # A perfect four-frame utterance followed by a fully missed two-frame
        # utterance scores the frame-weighted value, not the mean of the two.
        perfect_flags: list[bool] = [True, True, True, True]
        self._metric.update(self._builder.build(perfect_flags), self._builder.build(perfect_flags))
        self._metric.update(
            self._builder.build([True, True]),
            self._builder.build([False, False])
        )
        measured: float = self._metric.compute()
        expected: float = 8.0 / 10.0
        self.assertAlmostEqual(
            measured,
            expected,
            places=6,
            msg=f"Expected the frame-weighted {expected}, measured {measured}"
        )
        self.assertNotAlmostEqual(
            measured,
            0.5,
            places=2,
            msg=f"The score must not be the per-utterance mean, measured {measured}"
        )

    def test_compute_raises_when_no_voiced_frame_is_observed(self) -> None:
        # With every frame unvoiced in both signals the score has no
        # denominator and is undefined rather than zero.
        unvoiced_flags: list[bool] = [False, False, False]
        self._metric.update(self._builder.build(unvoiced_flags), self._builder.build(unvoiced_flags))
        with self.assertRaisesRegex(ValueError, "undefined"):
            self._metric.compute()

    def test_compute_raises_before_any_update(self) -> None:
        # The pass-level score is undefined rather than zero without counts.
        with self.assertRaisesRegex(ValueError, "undefined"):
            self._metric.compute()

    def test_reset_restores_the_undefined_state(self) -> None:
        # Clearing the counts returns the metric to its undefined starting
        # condition.
        voiced_flags: list[bool] = [True, True]
        self._metric.update(self._builder.build(voiced_flags), self._builder.build(voiced_flags))
        self._metric.reset()
        with self.assertRaisesRegex(ValueError, "undefined"):
            self._metric.compute()

    def test_reset_discards_previously_accumulated_counts(self) -> None:
        # Counts observed before a reset cannot influence the score computed
        # after it.
        self._metric.update(
            self._builder.build([True, True, True, True]),
            self._builder.build([False, False, False, False])
        )
        self._metric.reset()
        agreeing_flags: list[bool] = [True, True]
        self._metric.update(self._builder.build(agreeing_flags), self._builder.build(agreeing_flags))
        measured: float = self._metric.compute()
        self.assertAlmostEqual(
            measured,
            1.0,
            places=6,
            msg=f"Counts before the reset must be discarded, measured {measured}"
        )


if __name__ == "__main__":
    unittest.main()
