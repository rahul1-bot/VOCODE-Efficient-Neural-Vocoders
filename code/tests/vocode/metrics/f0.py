# This module:
# 1. Verifies the pass-level F0 root-mean-square-error accumulator over
#    fabricated pitch-feature records: the cent-scale error arithmetic, the
#    jointly-voiced frame mask, ragged-length truncation, frame-weighted
#    accumulation across utterances, and the undefined-result error path
#
# Design decisions:
# - Pitch features are fabricated directly as PitchFeatures records instead
#   of extracted, because the accumulator consumes only that record type and
#   real extraction would load pretrained CREPE weights for zero added
#   coverage of this module
# - Candidate tracks are built by shifting a reference track a known number
#   of cents, so the expected root-mean-square error is exact closed-form
#   arithmetic rather than a pinned floating-point sample
# - Frame weighting is checked with utterances of deliberately unequal
#   length, because a per-utterance average agrees with the frame-weighted
#   result on equal-length inputs and would hide the defect
#
# Author: Rahul Sawhney

import unittest

import torch

from vocode.metrics.f0 import F0Rmse
from vocode.metrics.pitch import PitchFeatures


class PitchTrackFeatureBuilder:
    # Builds fabricated pitch-feature records carrying explicit pitch tracks
    # and voicing decisions for the accumulator under test.
    def __init__(self, filler_periodicity: float) -> None:
        # Binds the periodicity value used to fill the field the F0 metric
        # never reads.
        self._filler_periodicity: float = filler_periodicity

    def build(self, pitch_values: list[float], voicing_flags: list[bool]) -> PitchFeatures:
        # Assembles one record from an explicit pitch track and its voicing
        # decisions.
        frame_count: int = len(pitch_values)
        return PitchFeatures(
            pitch=torch.tensor(pitch_values, dtype=torch.float32),
            periodicity=torch.full((frame_count,), self._filler_periodicity, dtype=torch.float32),
            voicing=torch.tensor(voicing_flags, dtype=torch.bool)
        )

    def build_shifted(
        self,
        pitch_values: list[float],
        cent_offset: float,
        voicing_flags: list[bool]
    ) -> PitchFeatures:
        # Assembles a record whose pitch track sits an exact number of cents
        # above the supplied track.
        ratio: float = 2.0 ** (cent_offset / 1200.0)
        shifted_values: list[float] = [value * ratio for value in pitch_values]
        return self.build(shifted_values, voicing_flags)


class F0RmseCentErrorTest(unittest.TestCase):
    # Verifies that the accumulated error reproduces the cent-scale distance
    # between two pitch tracks.
    def setUp(self) -> None:
        # Prepares a fresh accumulator, a record builder, and a voiced track.
        self._metric: F0Rmse = F0Rmse()
        self._builder: PitchTrackFeatureBuilder = PitchTrackFeatureBuilder(0.5)
        self._pitch_values: list[float] = [220.0, 246.94, 261.63, 293.66]
        self._voicing_flags: list[bool] = [True, True, True, True]

    def test_identical_tracks_yield_zero_error(self) -> None:
        # Two identical pitch tracks are separated by zero cents.
        reference: PitchFeatures = self._builder.build(self._pitch_values, self._voicing_flags)
        candidate: PitchFeatures = self._builder.build(self._pitch_values, self._voicing_flags)
        self._metric.update(reference, candidate)
        measured: float = self._metric.compute()
        self.assertAlmostEqual(
            measured,
            0.0,
            places=4,
            msg=f"Identical pitch tracks must score zero cents, measured {measured}"
        )

    def test_uniform_cent_shift_equals_the_shift_magnitude(self) -> None:
        # A track shifted by a constant number of cents scores that constant.
        cent_offset: float = 50.0
        reference: PitchFeatures = self._builder.build(self._pitch_values, self._voicing_flags)
        candidate: PitchFeatures = self._builder.build_shifted(
            self._pitch_values, cent_offset, self._voicing_flags
        )
        self._metric.update(reference, candidate)
        measured: float = self._metric.compute()
        self.assertAlmostEqual(
            measured,
            cent_offset,
            places=2,
            msg=f"A {cent_offset} cent shift must score {cent_offset}, measured {measured}"
        )

    def test_octave_shift_measures_twelve_hundred_cents(self) -> None:
        # Doubling every pitch value is exactly one octave, or 1200 cents.
        doubled_values: list[float] = [value * 2.0 for value in self._pitch_values]
        reference: PitchFeatures = self._builder.build(self._pitch_values, self._voicing_flags)
        candidate: PitchFeatures = self._builder.build(doubled_values, self._voicing_flags)
        self._metric.update(reference, candidate)
        measured: float = self._metric.compute()
        self.assertAlmostEqual(
            measured,
            1200.0,
            places=2,
            msg=f"An octave shift must score 1200 cents, measured {measured}"
        )

    def test_error_is_symmetric_under_argument_swap(self) -> None:
        # Squaring the cent difference makes the score independent of which
        # signal is passed as the reference.
        reference: PitchFeatures = self._builder.build(self._pitch_values, self._voicing_flags)
        candidate: PitchFeatures = self._builder.build_shifted(
            self._pitch_values, 30.0, self._voicing_flags
        )
        self._metric.update(reference, candidate)
        forward_error: float = self._metric.compute()
        self._metric.reset()
        self._metric.update(candidate, reference)
        swapped_error: float = self._metric.compute()
        self.assertAlmostEqual(
            forward_error,
            swapped_error,
            places=4,
            msg=f"Swapping arguments changed the score from {forward_error} to {swapped_error}"
        )


class F0RmseVoicingMaskTest(unittest.TestCase):
    # Verifies that only frames voiced in both signals reach the error
    # accumulator.
    def setUp(self) -> None:
        # Prepares a fresh accumulator and a record builder.
        self._metric: F0Rmse = F0Rmse()
        self._builder: PitchTrackFeatureBuilder = PitchTrackFeatureBuilder(0.5)

    def test_only_jointly_voiced_frames_contribute(self) -> None:
        # Frames voiced in one signal alone are excluded however far apart
        # their pitch values sit.
        shift_ratio: float = 2.0 ** (50.0 / 1200.0)
        reference: PitchFeatures = self._builder.build(
            [100.0, 220.0, 100.0], [True, True, False]
        )
        candidate: PitchFeatures = self._builder.build(
            [400.0, 220.0 * shift_ratio, 400.0], [False, True, True]
        )
        self._metric.update(reference, candidate)
        measured: float = self._metric.compute()
        self.assertAlmostEqual(
            measured,
            50.0,
            places=2,
            msg=f"Only the jointly voiced frame may contribute, measured {measured}"
        )

    def test_unvoiced_frames_may_carry_zero_pitch_without_poisoning_the_result(self) -> None:
        # Masking happens before the logarithm, so a zero-hertz unvoiced
        # frame cannot drive the score to negative infinity.
        reference: PitchFeatures = self._builder.build([0.0, 220.0], [False, True])
        candidate: PitchFeatures = self._builder.build([0.0, 220.0], [False, True])
        self._metric.update(reference, candidate)
        measured: float = self._metric.compute()
        self.assertTrue(
            torch.isfinite(torch.tensor(measured)).item(),
            msg=f"Unvoiced zero-hertz frames must not reach the logarithm, measured {measured}"
        )
        self.assertAlmostEqual(
            measured,
            0.0,
            places=4,
            msg=f"Only the voiced matching frame may contribute, measured {measured}"
        )

    def test_pair_without_a_jointly_voiced_frame_contributes_nothing(self) -> None:
        # An utterance pair with disjoint voicing leaves the accumulators
        # untouched.
        disjoint_reference: PitchFeatures = self._builder.build([110.0, 880.0], [True, False])
        disjoint_candidate: PitchFeatures = self._builder.build([880.0, 110.0], [False, True])
        self._metric.update(disjoint_reference, disjoint_candidate)
        scored_reference: PitchFeatures = self._builder.build([220.0], [True])
        scored_candidate: PitchFeatures = self._builder.build_shifted([220.0], 100.0, [True])
        self._metric.update(scored_reference, scored_candidate)
        measured: float = self._metric.compute()
        self.assertAlmostEqual(
            measured,
            100.0,
            places=2,
            msg=f"The disjoint pair must not perturb the score, measured {measured}"
        )

    def test_ragged_tracks_truncate_to_the_common_frame_count(self) -> None:
        # Frames beyond the shorter track are dropped rather than compared
        # against missing material.
        reference: PitchFeatures = self._builder.build(
            [220.0, 220.0, 50.0, 50.0], [True, True, True, True]
        )
        candidate: PitchFeatures = self._builder.build([220.0, 220.0], [True, True])
        self._metric.update(reference, candidate)
        measured: float = self._metric.compute()
        self.assertAlmostEqual(
            measured,
            0.0,
            places=4,
            msg=f"Trailing reference frames must be ignored, measured {measured}"
        )


class F0RmseAccumulationLifecycleTest(unittest.TestCase):
    # Verifies pass-level accumulation, the reset contract, and the
    # undefined-result error path.
    def setUp(self) -> None:
        # Prepares a fresh accumulator and a record builder.
        self._metric: F0Rmse = F0Rmse()
        self._builder: PitchTrackFeatureBuilder = PitchTrackFeatureBuilder(0.5)

    def test_accumulation_weights_frames_rather_than_utterances(self) -> None:
        # Three frames at 100 cents followed by one frame at zero cents score
        # the frame-weighted value, not the mean of the two utterances.
        long_reference: PitchFeatures = self._builder.build(
            [220.0, 220.0, 220.0], [True, True, True]
        )
        long_candidate: PitchFeatures = self._builder.build_shifted(
            [220.0, 220.0, 220.0], 100.0, [True, True, True]
        )
        short_reference: PitchFeatures = self._builder.build([330.0], [True])
        short_candidate: PitchFeatures = self._builder.build([330.0], [True])
        self._metric.update(long_reference, long_candidate)
        self._metric.update(short_reference, short_candidate)
        measured: float = self._metric.compute()
        frame_weighted_expectation: float = (3.0 * 100.0 ** 2 / 4.0) ** 0.5
        self.assertAlmostEqual(
            measured,
            frame_weighted_expectation,
            places=2,
            msg=f"Expected the frame-weighted {frame_weighted_expectation}, measured {measured}"
        )
        self.assertNotAlmostEqual(
            measured,
            50.0,
            places=1,
            msg=f"The score must not be the per-utterance mean, measured {measured}"
        )

    def test_compute_raises_when_no_frame_is_jointly_voiced(self) -> None:
        # The pass-level score is undefined rather than zero without a single
        # jointly voiced frame.
        with self.assertRaisesRegex(ValueError, "undefined"):
            self._metric.compute()

    def test_reset_restores_the_undefined_state(self) -> None:
        # Clearing the accumulators returns the metric to its undefined
        # starting condition.
        reference: PitchFeatures = self._builder.build([220.0], [True])
        candidate: PitchFeatures = self._builder.build([220.0], [True])
        self._metric.update(reference, candidate)
        self._metric.reset()
        with self.assertRaisesRegex(ValueError, "undefined"):
            self._metric.compute()

    def test_reset_discards_previously_accumulated_frames(self) -> None:
        # Frames observed before a reset cannot influence the score computed
        # after it.
        stale_reference: PitchFeatures = self._builder.build([220.0], [True])
        stale_candidate: PitchFeatures = self._builder.build_shifted([220.0], 400.0, [True])
        self._metric.update(stale_reference, stale_candidate)
        self._metric.reset()
        fresh_reference: PitchFeatures = self._builder.build([220.0], [True])
        fresh_candidate: PitchFeatures = self._builder.build([220.0], [True])
        self._metric.update(fresh_reference, fresh_candidate)
        measured: float = self._metric.compute()
        self.assertAlmostEqual(
            measured,
            0.0,
            places=4,
            msg=f"Frames before the reset must be discarded, measured {measured}"
        )


if __name__ == "__main__":
    unittest.main()
