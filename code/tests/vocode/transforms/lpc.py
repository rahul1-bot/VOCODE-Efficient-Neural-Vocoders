# This module:
# 1. Verifies the LpcnetFeatureConfig analysis protocol: the 16 kHz reference
#    defaults and the frozen, strict, extra-forbidding record semantics
# 2. Verifies the reference 8-bit mu-law companding protocol: the index of
#    silence, full-scale saturation, sign symmetry, monotonicity, and exact
#    round-trip recovery of int16-scaled samples
# 3. Verifies the frame analysis: precomputed band weights, the orthonormal DCT
#    matrix and analysis window, feature shapes and dtypes, accepted and
#    rejected waveform ranks, autocorrelation pitch search on a signal of known
#    period, numerical stability on silence, and batch independence
#
# Design decisions:
# - Pitch is asserted against a synthetic waveform whose period is known by
#   construction; the detected lag must be that period or its octave, because a
#   normalized autocorrelation peaks equally at every integer multiple
# - Feature values are bounded (finiteness, correlation range, index range) and
#   cross-checked for internal consistency rather than pinned, since no
#   reference feature file ships with this repository
# - The DCT matrix and the mu-law round trip are asserted exactly, because both
#   are closed-form mathematics with no floating-point ambiguity beyond epsilon
# - Waveforms stay at or below 3200 samples, which is twenty analysis frames and
#   keeps the whole file inside a second on CPU
# - Batch independence is asserted directly, because the frame, pitch, and band
#   projections are batched tensor operations where an indexing error would
#   silently mix utterances
#
# Author: Rahul Sawhney

import math
import unittest

import torch
from pydantic import ValidationError

from vocode.transforms.lpc import LpcnetFeatureConfig, LpcnetFeatureExtractor, LpcnetFeatures, LpcnetMuLaw


class PeriodicWaveformBuilder:
    # Builds deterministic periodic, broadband, and silent analysis material.
    def __init__(self, sample_count: int) -> None:
        # Binds the waveform length and the fixed seed backing the broadband draw.
        self._sample_count: int = sample_count
        self._noise_seed: int = 20260730

    def periodic(self, period_samples: int, amplitude: float) -> torch.Tensor:
        # Produces a waveform whose fundamental period is exact by construction.
        positions: torch.Tensor = torch.arange(self._sample_count, dtype=torch.float32)
        return amplitude * torch.sin(2.0 * torch.pi * positions / period_samples)

    def broadband(self, amplitude: float) -> torch.Tensor:
        # Draws a reproducible gaussian sequence with no dominant period.
        torch.manual_seed(self._noise_seed)
        return amplitude * torch.randn(self._sample_count)

    def silence(self) -> torch.Tensor:
        # Produces an all-zero waveform that must not destabilize the analysis.
        return torch.zeros(self._sample_count)


class Int16SampleBuilder:
    # Builds int16-scaled sample sets for the mu-law companding checks.
    def __init__(self) -> None:
        # Binds the full-scale bound of the int16 sample domain.
        self._full_scale: float = 32768.0

    def full_scale(self) -> float:
        # Returns the positive full-scale bound of the sample domain.
        return self._full_scale

    def sweep(self, step_count: int) -> torch.Tensor:
        # Produces an ascending sweep across the whole int16 sample domain.
        return torch.linspace(-self._full_scale, self._full_scale, step_count)

    def graded_magnitudes(self) -> torch.Tensor:
        # Produces representative magnitudes below the companding saturation point.
        return torch.tensor([-30000.0, -8000.0, -1000.0, -1.0, 0.0, 1.0, 1000.0, 8000.0, 30000.0])


class LpcnetAnalysisProtocolTest(unittest.TestCase):
    # Verifies the frozen LPCNet analysis protocol and its validation surface.
    def setUp(self) -> None:
        # Builds the default record and its field mapping for the rejection checks.
        self._configuration: LpcnetFeatureConfig = LpcnetFeatureConfig()
        self._fields: dict[str, object] = self._configuration.model_dump()

    def test_defaults_follow_the_sixteen_kilohertz_reference_recipe(self) -> None:
        # The shipped defaults are the published 16 kHz LPCNet analysis constants.
        self.assertEqual(self._configuration.sample_rate, 16000)
        self.assertEqual(self._configuration.frame_size, 160)
        self.assertEqual(self._configuration.window_size, 320)
        self.assertEqual(self._configuration.band_count, 18)
        self.assertEqual(self._configuration.lpc_order, 16)
        self.assertEqual(self._configuration.minimum_pitch_lag, 32)
        self.assertEqual(self._configuration.maximum_pitch_lag, 255)
        self.assertEqual(self._configuration.preemphasis_coefficient, 0.85)

    def test_analysis_window_spans_two_hops(self) -> None:
        # Ten-millisecond frames under a twenty-millisecond window overlap by half.
        self.assertEqual(self._configuration.window_size, 2 * self._configuration.frame_size)
        self.assertEqual(
            self._configuration.frame_size * 100,
            self._configuration.sample_rate,
            msg="The reference recipe analyses one hundred frames per second"
        )

    def test_pitch_search_range_stays_inside_the_index_domain(self) -> None:
        # Pitch indices are stored in one byte, so the search may not exceed 255.
        self.assertLess(self._configuration.minimum_pitch_lag, self._configuration.maximum_pitch_lag)
        self.assertLessEqual(self._configuration.maximum_pitch_lag, 255)

    def test_protocol_rejects_field_mutation(self) -> None:
        # A bound analysis protocol cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.frame_size: int = 320

    def test_protocol_rejects_unknown_field(self) -> None:
        # A forbidden extra turns a mistyped analysis field into a failure.
        with self.assertRaises(ValidationError):
            LpcnetFeatureConfig.model_validate({**self._fields, "frame_length": 160})

    def test_protocol_rejects_float_where_an_integer_field_is_declared(self) -> None:
        # Strict validation refuses a fractional frame size or sample rate.
        with self.assertRaises(ValidationError):
            LpcnetFeatureConfig.model_validate({**self._fields, "sample_rate": 16000.0})
        with self.assertRaises(ValidationError):
            LpcnetFeatureConfig.model_validate({**self._fields, "frame_size": 160.5})

    def test_protocol_rejects_non_positive_sizes(self) -> None:
        # Zero-length frames or bandless analysis cannot express a measurement.
        with self.assertRaises(ValidationError):
            LpcnetFeatureConfig.model_validate({**self._fields, "frame_size": 0})
        with self.assertRaises(ValidationError):
            LpcnetFeatureConfig.model_validate({**self._fields, "band_count": -18})


class LpcnetMuLawCompandingTest(unittest.TestCase):
    # Verifies the reference 8-bit mu-law companding and expansion protocol.
    def setUp(self) -> None:
        # Binds the transform and the int16 sample builder.
        self._companding: LpcnetMuLaw = LpcnetMuLaw()
        self._samples: Int16SampleBuilder = Int16SampleBuilder()

    def test_silence_maps_to_the_centre_of_the_index_domain(self) -> None:
        # A zero sample is the midpoint of the unsigned 0-255 index range.
        index: torch.Tensor = self._companding.linear_to_mulaw(torch.zeros(1))
        self.assertAlmostEqual(float(index.item()), 128.0, places=6)

    def test_full_scale_samples_saturate_at_the_index_bounds(self) -> None:
        # Positive and negative full scale reach the ends of the index domain.
        full_scale: float = self._samples.full_scale()
        bounds: torch.Tensor = self._companding.linear_to_mulaw(
            torch.tensor([-full_scale, full_scale])
        )
        self.assertAlmostEqual(float(bounds[0].item()), 0.0, places=5)
        self.assertAlmostEqual(float(bounds[1].item()), 255.0, places=5)

    def test_out_of_range_samples_are_clamped_into_the_index_domain(self) -> None:
        # The protocol clamps rather than wrapping when input exceeds full scale.
        indices: torch.Tensor = self._companding.linear_to_mulaw(
            torch.tensor([-1.0e6, 1.0e6])
        )
        self.assertGreaterEqual(float(indices.min()), 0.0)
        self.assertLessEqual(float(indices.max()), 255.0)

    def test_companding_is_symmetric_about_the_silence_index(self) -> None:
        # Opposite samples sit at mirrored distances from the centre index.
        magnitudes: torch.Tensor = torch.tensor([1.0, 1000.0, 8000.0, 20000.0])
        positive: torch.Tensor = self._companding.linear_to_mulaw(magnitudes)
        negative: torch.Tensor = self._companding.linear_to_mulaw(-magnitudes)
        self.assertTrue(
            torch.allclose(positive + negative, torch.full_like(positive, 256.0), atol=1e-3),
            msg="Mirrored samples must sum to twice the silence index"
        )

    def test_companding_never_decreases_across_the_sample_domain(self) -> None:
        # The mapping is monotone, saturating only at the index bounds.
        indices: torch.Tensor = self._companding.linear_to_mulaw(self._samples.sweep(101))
        self.assertTrue(
            bool((indices.diff() >= 0.0).all()),
            msg="Mu-law companding must be non-decreasing in the linear sample"
        )
        self.assertGreaterEqual(float(indices.min()), 0.0)
        self.assertLessEqual(float(indices.max()), 255.0)

    def test_companding_strictly_increases_below_saturation(self) -> None:
        # Away from the clamped extremes distinct samples take distinct indices.
        indices: torch.Tensor = self._companding.linear_to_mulaw(self._samples.graded_magnitudes())
        self.assertTrue(
            bool((indices.diff() > 0.0).all()),
            msg="Distinct sub-saturation samples must receive distinct mu-law indices"
        )

    def test_expansion_recovers_the_original_samples(self) -> None:
        # Companding followed by expansion is the identity up to float epsilon. The absolute
        # tolerance below is expressed on the int16 scale, where full scale is 32768, so
        # admitting five hundredths of a sample is a far tighter statement than it looks and
        # leaves no room for a mismatched scale constant between the two directions. The
        # magnitudes are drawn below the saturation point, because a saturated sample is
        # clamped by design and could not be recovered by any inverse.
        linear: torch.Tensor = self._samples.graded_magnitudes()
        recovered: torch.Tensor = self._companding.mulaw_to_linear(
            self._companding.linear_to_mulaw(linear)
        )
        self.assertTrue(
            torch.allclose(recovered, linear, rtol=1e-5, atol=0.05),
            msg=f"Round trip drifted by {float((recovered - linear).abs().max())} samples"
        )

    def test_expansion_of_the_centre_index_returns_silence(self) -> None:
        # The midpoint index expands back to a zero-amplitude sample.
        recovered: torch.Tensor = self._companding.mulaw_to_linear(torch.tensor([128.0]))
        self.assertAlmostEqual(float(recovered.item()), 0.0, places=6)


class LpcnetAnalysisMachineryTest(unittest.TestCase):
    # Verifies the precomputed band weights, DCT matrix, and window buffers.
    def setUp(self) -> None:
        # Builds the extractor under the default 16 kHz analysis protocol.
        self._configuration: LpcnetFeatureConfig = LpcnetFeatureConfig()
        self._extractor: LpcnetFeatureExtractor = LpcnetFeatureExtractor(self._configuration)

    def test_analysis_buffers_are_registered(self) -> None:
        # The extractor owns its band weights, DCT matrix, and analysis window.
        buffer_names: set[str] = {name for name, _ in self._extractor.named_buffers(recurse=False)}
        self.assertEqual(buffer_names, {"_band_weights", "_dct_matrix", "_analysis_window"})

    def test_derived_buffers_stay_out_of_the_state_dictionary(self) -> None:
        # Every buffer is derivable from the protocol, so checkpoints omit them.
        self.assertEqual(
            list(self._extractor.state_dict().keys()),
            [],
            msg="Analysis buffers must be non-persistent and stay out of checkpoints"
        )

    def test_band_weights_interpolate_the_fft_grid_without_negative_gain(self) -> None:
        # Triangular weights map the half-spectrum onto the Bark-frequency bands.
        band_weights: torch.Tensor = self._extractor.get_buffer("_band_weights")
        self.assertEqual(
            band_weights.shape,
            (self._configuration.band_count, self._configuration.window_size // 2 + 1)
        )
        self.assertTrue(bool((band_weights >= 0.0).all()), msg="Band weights must be non-negative")
        self.assertLessEqual(float(band_weights.max()), 1.0 + 1e-6)
        self.assertAlmostEqual(
            float(band_weights[0, 0].item()),
            1.0,
            places=6,
            msg="The lowest band must take the direct-current bin at unit weight"
        )

    def test_dct_matrix_is_orthonormal(self) -> None:
        # The cepstral projection preserves energy, so its matrix is orthonormal.
        dct_matrix: torch.Tensor = self._extractor.get_buffer("_dct_matrix")
        identity: torch.Tensor = torch.eye(self._configuration.band_count)
        deviation: float = float((dct_matrix @ dct_matrix.transpose(0, 1) - identity).abs().max())
        self.assertLess(deviation, 1e-5, msg=f"DCT matrix deviated from orthonormality by {deviation}")

    def test_analysis_window_is_a_periodic_hann_window(self) -> None:
        # A periodic Hann window of length N sums to exactly half of N.
        window: torch.Tensor = self._extractor.get_buffer("_analysis_window")
        self.assertEqual(window.shape, (self._configuration.window_size,))
        self.assertAlmostEqual(
            float(window.sum().item()),
            self._configuration.window_size / 2.0,
            places=3
        )

    def test_band_count_mismatch_is_rejected_at_construction(self) -> None:
        # The band edge table is fixed, so a differing band count cannot be honoured.
        with self.assertRaises(ValueError):
            LpcnetFeatureExtractor(LpcnetFeatureConfig(band_count=17))

    def test_configuration_property_returns_the_injected_protocol(self) -> None:
        # The extractor exposes exactly the protocol it was constructed with.
        self.assertIs(self._extractor.configuration, self._configuration)


class LpcnetFeatureShapeTest(unittest.TestCase):
    # Verifies feature shapes, dtypes, and the accepted waveform ranks.
    def setUp(self) -> None:
        # Builds the extractor and a ten-frame broadband waveform.
        self._configuration: LpcnetFeatureConfig = LpcnetFeatureConfig()
        self._extractor: LpcnetFeatureExtractor = LpcnetFeatureExtractor(self._configuration)
        self._builder: PeriodicWaveformBuilder = PeriodicWaveformBuilder(1600)
        self._waveform: torch.Tensor = self._builder.broadband(0.1)
        self._frame_count: int = 1600 // self._configuration.frame_size

    def test_extraction_produces_one_feature_row_per_frame(self) -> None:
        # Front padding by one hop yields exactly time / frame_size frames.
        features: LpcnetFeatures = self._extractor.extract(self._waveform.unsqueeze(0))
        self.assertEqual(
            features.conditioning.shape,
            (1, self._frame_count, self._configuration.band_count + 2)
        )
        self.assertEqual(features.pitch_index.shape, (1, self._frame_count))
        self.assertEqual(
            features.lpc_coefficients.shape,
            (1, self._frame_count, self._configuration.lpc_order)
        )

    def test_conditioning_appends_pitch_and_correlation_to_the_cepstra(self) -> None:
        # The feature row is the band cepstra followed by the two pitch columns.
        features: LpcnetFeatures = self._extractor.extract(self._waveform.unsqueeze(0))
        self.assertEqual(
            features.conditioning.shape[-1] - self._configuration.band_count,
            2,
            msg="The conditioning row must carry exactly two pitch columns"
        )

    def test_pitch_indices_are_long_integers_inside_the_byte_domain(self) -> None:
        # Pitch indices address a byte-wide embedding table in the decoder.
        features: LpcnetFeatures = self._extractor.extract(self._waveform.unsqueeze(0))
        self.assertEqual(features.pitch_index.dtype, torch.long)
        self.assertGreaterEqual(int(features.pitch_index.min()), 0)
        self.assertLessEqual(int(features.pitch_index.max()), 255)

    def test_features_are_finite(self) -> None:
        # Numerical floors in the band, pitch, and Levinson stages keep values finite.
        features: LpcnetFeatures = self._extractor.extract(self._waveform.unsqueeze(0))
        self.assertTrue(torch.isfinite(features.conditioning).all(), msg="Conditioning must be finite")
        self.assertTrue(
            torch.isfinite(features.lpc_coefficients).all(),
            msg="LPC coefficients must be finite"
        )

    def test_unbatched_waveforms_are_promoted_to_a_batch(self) -> None:
        # A bare [time] buffer analyses as a single-utterance batch.
        features: LpcnetFeatures = self._extractor.extract(self._waveform)
        self.assertEqual(features.conditioning.shape[0], 1)
        self.assertEqual(features.conditioning.shape[1], self._frame_count)

    def test_single_channel_batched_waveforms_are_accepted(self) -> None:
        # A [batch, 1, time] buffer loses its channel axis before framing.
        batched: torch.Tensor = self._waveform.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1)
        features: LpcnetFeatures = self._extractor.extract(batched)
        self.assertEqual(features.conditioning.shape[0], 2)

    def test_multichannel_waveforms_are_rejected(self) -> None:
        # Multi-channel material has no defined single-track analysis protocol.
        with self.assertRaises(ValueError):
            self._extractor.extract(torch.zeros(2, 3, 1600))

    def test_module_call_matches_the_extract_method(self) -> None:
        # The module-call form composes like any torch transform without diverging.
        called: LpcnetFeatures = self._extractor(self._waveform.unsqueeze(0))
        extracted: LpcnetFeatures = self._extractor.extract(self._waveform.unsqueeze(0))
        self.assertTrue(torch.equal(called.conditioning, extracted.conditioning))
        self.assertTrue(torch.equal(called.pitch_index, extracted.pitch_index))
        self.assertTrue(torch.equal(called.lpc_coefficients, extracted.lpc_coefficients))

    def test_feature_bundle_is_frozen(self) -> None:
        # A produced feature bundle cannot be mutated after analysis.
        features: LpcnetFeatures = self._extractor.extract(self._waveform.unsqueeze(0))
        with self.assertRaises(ValidationError):
            features.conditioning: torch.Tensor = torch.zeros(1)


class LpcnetPitchSearchTest(unittest.TestCase):
    # Verifies the autocorrelation pitch search and its feature encoding.
    def setUp(self) -> None:
        # Builds the extractor and a waveform whose period is known by construction.
        self._configuration: LpcnetFeatureConfig = LpcnetFeatureConfig()
        self._extractor: LpcnetFeatureExtractor = LpcnetFeatureExtractor(self._configuration)
        self._builder: PeriodicWaveformBuilder = PeriodicWaveformBuilder(3200)
        self._period_samples: int = 100

    def test_detected_lag_matches_the_true_period_or_its_octave(self) -> None:
        # A normalized autocorrelation peaks at every integer multiple of the period.
        # Admitting the octave is therefore correctness rather than leniency: a pure sine
        # correlates exactly as well at twice its period, so demanding the fundamental would
        # assert a tie-break the algorithm does not claim to make. The period of one hundred
        # samples is chosen so that both it and its double fall inside the search range,
        # which is what makes the admissible set meaningful rather than vacuous.
        features: LpcnetFeatures = self._extractor.extract(
            self._builder.periodic(self._period_samples, 0.5).unsqueeze(0)
        )
        detected_lags: set[int] = {int(value) for value in features.pitch_index.flatten().tolist()}
        admissible_lags: set[int] = {self._period_samples, 2 * self._period_samples}
        self.assertTrue(
            detected_lags.issubset(admissible_lags),
            msg=f"Detected lags {sorted(detected_lags)} are not multiples of the true period"
        )

    def test_detected_lag_stays_inside_the_configured_search_range(self) -> None:
        # The search never reports a lag outside its own bounds.
        features: LpcnetFeatures = self._extractor.extract(
            self._builder.periodic(self._period_samples, 0.5).unsqueeze(0)
        )
        self.assertGreaterEqual(int(features.pitch_index.min()), self._configuration.minimum_pitch_lag)
        self.assertLessEqual(int(features.pitch_index.max()), self._configuration.maximum_pitch_lag)

    def test_correlation_column_stays_inside_the_normalized_range(self) -> None:
        # A normalized correlation is bounded by minus one and one.
        features: LpcnetFeatures = self._extractor.extract(
            self._builder.periodic(self._period_samples, 0.5).unsqueeze(0)
        )
        correlations: torch.Tensor = features.conditioning[..., -1]
        self.assertGreaterEqual(float(correlations.min()), -1.0)
        self.assertLessEqual(float(correlations.max()), 1.0)

    def test_periodic_material_reaches_a_high_correlation(self) -> None:
        # A perfectly periodic waveform must correlate strongly at its period.
        features: LpcnetFeatures = self._extractor.extract(
            self._builder.periodic(self._period_samples, 0.5).unsqueeze(0)
        )
        correlations: torch.Tensor = features.conditioning[..., -1]
        self.assertGreater(
            float(correlations.max()),
            0.9,
            msg="A purely periodic waveform must produce a near-unit pitch correlation"
        )

    def test_pitch_column_encodes_the_detected_lag(self) -> None:
        # The conditioning row carries the lag offset by one hundred over fifty.
        features: LpcnetFeatures = self._extractor.extract(
            self._builder.periodic(self._period_samples, 0.5).unsqueeze(0)
        )
        expected_column: torch.Tensor = (features.pitch_index.to(torch.float32) - 100.0) / 50.0
        self.assertTrue(
            torch.allclose(features.conditioning[..., -2], expected_column, atol=1e-6),
            msg="The pitch column must encode the detected lag consistently"
        )

    def test_silence_analyses_without_producing_undefined_values(self) -> None:
        # Energy floors keep silent frames finite and pin the lag to the search floor.
        features: LpcnetFeatures = self._extractor.extract(self._builder.silence().unsqueeze(0))
        self.assertTrue(torch.isfinite(features.conditioning).all())
        self.assertTrue(torch.isfinite(features.lpc_coefficients).all())
        self.assertTrue(
            bool((features.pitch_index == self._configuration.minimum_pitch_lag).all()),
            msg="Silence has no correlation peak, so the search must report its floor"
        )

    def test_silent_band_energies_reach_the_configured_floor(self) -> None:
        # Clamped band energies compress to the decimal logarithm of the floor.
        features: LpcnetFeatures = self._extractor.extract(self._builder.silence().unsqueeze(0))
        cepstra: torch.Tensor = features.conditioning[..., : self._configuration.band_count]
        dct_matrix: torch.Tensor = self._extractor.get_buffer("_dct_matrix")
        floor_energies: torch.Tensor = torch.full((self._configuration.band_count,), math.log10(1e-2))
        expected_cepstra: torch.Tensor = dct_matrix @ floor_energies
        self.assertTrue(
            torch.allclose(cepstra[0, -1], expected_cepstra, atol=1e-5),
            msg="Silent frames must project the clamped energy floor through the DCT"
        )


class LpcnetBatchIndependenceTest(unittest.TestCase):
    # Verifies that batched analysis never mixes utterances and stays deterministic.
    def setUp(self) -> None:
        # Builds the extractor and two distinct ten-frame waveforms.
        self._extractor: LpcnetFeatureExtractor = LpcnetFeatureExtractor(LpcnetFeatureConfig())
        self._builder: PeriodicWaveformBuilder = PeriodicWaveformBuilder(1600)
        self._first: torch.Tensor = self._builder.periodic(120, 0.4)
        self._second: torch.Tensor = self._builder.broadband(0.1)

    def test_batched_rows_match_single_utterance_analysis(self) -> None:
        # Framing, pitch indexing, and band projection are per-utterance operations.
        batched: LpcnetFeatures = self._extractor.extract(
            torch.stack([self._first, self._second])
        )
        second_alone: LpcnetFeatures = self._extractor.extract(self._second.unsqueeze(0))
        self.assertTrue(
            torch.allclose(batched.conditioning[1:2], second_alone.conditioning, atol=1e-6),
            msg="Batched conditioning must equal single-utterance conditioning"
        )
        self.assertTrue(
            torch.allclose(batched.lpc_coefficients[1:2], second_alone.lpc_coefficients, atol=1e-6),
            msg="Batched LPC coefficients must equal single-utterance coefficients"
        )
        self.assertTrue(torch.equal(batched.pitch_index[1:2], second_alone.pitch_index))

    def test_repeated_analysis_returns_identical_features(self) -> None:
        # Feature extraction is a deterministic function of the waveform.
        first_pass: LpcnetFeatures = self._extractor.extract(self._first.unsqueeze(0))
        second_pass: LpcnetFeatures = self._extractor.extract(self._first.unsqueeze(0))
        self.assertTrue(torch.equal(first_pass.conditioning, second_pass.conditioning))
        self.assertTrue(torch.equal(first_pass.lpc_coefficients, second_pass.lpc_coefficients))


if __name__ == "__main__":
    unittest.main()
