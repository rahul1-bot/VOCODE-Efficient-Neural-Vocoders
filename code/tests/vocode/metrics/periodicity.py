# This module:
# 1. Verifies the pass-level periodicity root-mean-square-error accumulator
#    over fabricated pitch-feature records: the error arithmetic, the
#    voicing-independent frame coverage, ragged-length truncation,
#    frame-weighted accumulation, and the undefined-result error path
#
# Design decisions:
# - Pitch features are fabricated directly as PitchFeatures records instead
#   of extracted, because the accumulator consumes only that record type and
#   real extraction would load pretrained CREPE weights for zero added
#   coverage of this module
# - Candidate curves are offset from the reference curve by an exact
#   constant, so the expected root-mean-square error is that constant and
#   no floating-point sample has to be pinned
# - Voicing flags are varied independently of the periodicity curves,
#   because the documented contract is that every common frame counts
#   whether or not it is voiced
#
# Author: Rahul Sawhney

import unittest

import torch

from vocode.metrics.periodicity import PeriodicityRmse
from vocode.metrics.pitch import PitchFeatures


class PeriodicityCurveFeatureBuilder:
    # Builds fabricated pitch-feature records carrying explicit periodicity
    # curves and voicing decisions for the accumulator under test.
    def __init__(self, filler_pitch_hertz: float) -> None:
        # Binds the pitch value used to fill the field the periodicity metric
        # never reads.
        self._filler_pitch_hertz: float = filler_pitch_hertz

    def build(
        self,
        periodicity_values: list[float],
        voicing_flags: list[bool]
    ) -> PitchFeatures:
        # Assembles one record from an explicit periodicity curve and its
        # voicing decisions.
        frame_count: int = len(periodicity_values)
        return PitchFeatures(
            pitch=torch.full((frame_count,), self._filler_pitch_hertz, dtype=torch.float32),
            periodicity=torch.tensor(periodicity_values, dtype=torch.float32),
            voicing=torch.tensor(voicing_flags, dtype=torch.bool)
        )

    def build_voiced(self, periodicity_values: list[float]) -> PitchFeatures:
        # Assembles one record whose frames are all marked voiced.
        voicing_flags: list[bool] = [True] * len(periodicity_values)
        return self.build(periodicity_values, voicing_flags)

    def build_offset(self, periodicity_values: list[float], offset: float) -> PitchFeatures:
        # Assembles a fully voiced record whose curve sits an exact constant
        # below the supplied curve.
        shifted_values: list[float] = [value - offset for value in periodicity_values]
        return self.build_voiced(shifted_values)


class PeriodicityRmseErrorTest(unittest.TestCase):
    # Verifies that the accumulated error reproduces the distance between two
    # periodicity curves.
    def setUp(self) -> None:
        # Prepares a fresh accumulator, a record builder, and a curve.
        self._metric: PeriodicityRmse = PeriodicityRmse()
        self._builder: PeriodicityCurveFeatureBuilder = PeriodicityCurveFeatureBuilder(220.0)
        self._periodicity_values: list[float] = [0.9, 0.8, 0.7, 0.6]

    def test_identical_curves_yield_zero_error(self) -> None:
        # Two identical periodicity curves are separated by zero.
        reference: PitchFeatures = self._builder.build_voiced(self._periodicity_values)
        candidate: PitchFeatures = self._builder.build_voiced(self._periodicity_values)
        self._metric.update(reference, candidate)
        measured: float = self._metric.compute()
        self.assertAlmostEqual(
            measured,
            0.0,
            places=5,
            msg=f"Identical periodicity curves must score zero, measured {measured}"
        )

    def test_constant_offset_equals_the_offset_magnitude(self) -> None:
        # A curve offset by a constant scores that constant.
        offset: float = 0.25
        reference: PitchFeatures = self._builder.build_voiced(self._periodicity_values)
        candidate: PitchFeatures = self._builder.build_offset(self._periodicity_values, offset)
        self._metric.update(reference, candidate)
        measured: float = self._metric.compute()
        self.assertAlmostEqual(
            measured,
            offset,
            places=5,
            msg=f"A constant offset of {offset} must score {offset}, measured {measured}"
        )

    def test_error_is_symmetric_under_argument_swap(self) -> None:
        # Squaring the difference makes the score independent of which signal
        # is passed as the reference.
        reference: PitchFeatures = self._builder.build_voiced(self._periodicity_values)
        candidate: PitchFeatures = self._builder.build_offset(self._periodicity_values, 0.1)
        self._metric.update(reference, candidate)
        forward_error: float = self._metric.compute()
        self._metric.reset()
        self._metric.update(candidate, reference)
        swapped_error: float = self._metric.compute()
        self.assertAlmostEqual(
            forward_error,
            swapped_error,
            places=5,
            msg=f"Swapping arguments changed the score from {forward_error} to {swapped_error}"
        )


class PeriodicityRmseFrameCoverageTest(unittest.TestCase):
    # Verifies which frames enter the accumulator: every common frame
    # regardless of voicing, and nothing beyond the shorter curve.
    def setUp(self) -> None:
        # Prepares a fresh accumulator and a record builder.
        self._metric: PeriodicityRmse = PeriodicityRmse()
        self._builder: PeriodicityCurveFeatureBuilder = PeriodicityCurveFeatureBuilder(220.0)

    def test_unvoiced_frames_still_contribute_to_the_error(self) -> None:
        # The curve itself encodes the voicing confidence being judged, so
        # unvoiced frames are compared like any other.
        periodicity_values: list[float] = [0.4, 0.4, 0.4, 0.4]
        unvoiced_flags: list[bool] = [False, False, False, False]
        reference: PitchFeatures = self._builder.build(periodicity_values, unvoiced_flags)
        candidate: PitchFeatures = self._builder.build(
            [0.1, 0.1, 0.1, 0.1], unvoiced_flags
        )
        self._metric.update(reference, candidate)
        measured: float = self._metric.compute()
        self.assertAlmostEqual(
            measured,
            0.3,
            places=5,
            msg=f"Unvoiced frames must still be compared, measured {measured}"
        )

    def test_voicing_flags_do_not_change_the_score(self) -> None:
        # The same curves score identically whether their frames are marked
        # voiced or unvoiced.
        reference_values: list[float] = [0.9, 0.5]
        candidate_values: list[float] = [0.7, 0.2]
        voiced_metric: PeriodicityRmse = PeriodicityRmse()
        voiced_metric.update(
            self._builder.build(reference_values, [True, True]),
            self._builder.build(candidate_values, [True, True])
        )
        self._metric.update(
            self._builder.build(reference_values, [False, False]),
            self._builder.build(candidate_values, [False, False])
        )
        voiced_error: float = voiced_metric.compute()
        unvoiced_error: float = self._metric.compute()
        self.assertAlmostEqual(
            voiced_error,
            unvoiced_error,
            places=6,
            msg=f"Voicing changed the score from {voiced_error} to {unvoiced_error}"
        )

    def test_ragged_curves_truncate_to_the_common_frame_count(self) -> None:
        # Frames beyond the shorter curve are dropped rather than compared
        # against missing material.
        reference: PitchFeatures = self._builder.build_voiced([0.5, 0.5, 0.0, 0.0])
        candidate: PitchFeatures = self._builder.build_voiced([0.5, 0.5])
        self._metric.update(reference, candidate)
        measured: float = self._metric.compute()
        self.assertAlmostEqual(
            measured,
            0.0,
            places=5,
            msg=f"Trailing reference frames must be ignored, measured {measured}"
        )

    def test_empty_features_contribute_no_frames(self) -> None:
        # A pair carrying no frames leaves the pass undefined rather than
        # scoring zero.
        empty_reference: PitchFeatures = self._builder.build_voiced([])
        empty_candidate: PitchFeatures = self._builder.build_voiced([])
        self._metric.update(empty_reference, empty_candidate)
        with self.assertRaisesRegex(ValueError, "undefined"):
            self._metric.compute()


class PeriodicityRmseAccumulationLifecycleTest(unittest.TestCase):
    # Verifies pass-level accumulation, the reset contract, and the
    # undefined-result error path.
    def setUp(self) -> None:
        # Prepares a fresh accumulator and a record builder.
        self._metric: PeriodicityRmse = PeriodicityRmse()
        self._builder: PeriodicityCurveFeatureBuilder = PeriodicityCurveFeatureBuilder(220.0)

    def test_accumulation_weights_frames_rather_than_utterances(self) -> None:
        # Three frames offset by 0.2 followed by one exact frame score the
        # frame-weighted value, not the mean of the two utterances.
        long_reference: PitchFeatures = self._builder.build_voiced([0.9, 0.9, 0.9])
        long_candidate: PitchFeatures = self._builder.build_offset([0.9, 0.9, 0.9], 0.2)
        short_reference: PitchFeatures = self._builder.build_voiced([0.4])
        short_candidate: PitchFeatures = self._builder.build_voiced([0.4])
        self._metric.update(long_reference, long_candidate)
        self._metric.update(short_reference, short_candidate)
        measured: float = self._metric.compute()
        frame_weighted_expectation: float = (3.0 * 0.2 ** 2 / 4.0) ** 0.5
        self.assertAlmostEqual(
            measured,
            frame_weighted_expectation,
            places=5,
            msg=f"Expected the frame-weighted {frame_weighted_expectation}, measured {measured}"
        )
        self.assertNotAlmostEqual(
            measured,
            0.1,
            places=2,
            msg=f"The score must not be the per-utterance mean, measured {measured}"
        )

    def test_compute_raises_before_any_frame_is_observed(self) -> None:
        # The pass-level score is undefined rather than zero without frames.
        with self.assertRaisesRegex(ValueError, "undefined"):
            self._metric.compute()

    def test_reset_restores_the_undefined_state(self) -> None:
        # Clearing the accumulators returns the metric to its undefined
        # starting condition.
        self._metric.update(
            self._builder.build_voiced([0.5]),
            self._builder.build_voiced([0.5])
        )
        self._metric.reset()
        with self.assertRaisesRegex(ValueError, "undefined"):
            self._metric.compute()

    def test_reset_discards_previously_accumulated_frames(self) -> None:
        # Frames observed before a reset cannot influence the score computed
        # after it.
        self._metric.update(
            self._builder.build_voiced([1.0, 1.0]),
            self._builder.build_voiced([0.0, 0.0])
        )
        self._metric.reset()
        self._metric.update(
            self._builder.build_voiced([0.6]),
            self._builder.build_voiced([0.6])
        )
        measured: float = self._metric.compute()
        self.assertAlmostEqual(
            measured,
            0.0,
            places=5,
            msg=f"Frames before the reset must be discarded, measured {measured}"
        )


if __name__ == "__main__":
    unittest.main()
