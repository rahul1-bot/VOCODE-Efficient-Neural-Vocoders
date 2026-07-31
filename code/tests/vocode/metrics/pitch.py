# This module:
# 1. Verifies the frozen pitch extraction configuration record and its
#    validation rules, and the frozen pitch-feature bundle record
# 2. Verifies the extractor logic that surrounds CREPE inference: settings
#    forwarding, waveform normalization, silence gating of the periodicity
#    curve, hysteresis-derived voicing, and the ungated pitch track
#
# Design decisions:
# - The CREPE prediction entry point is replaced by a fabricated frame-level
#   responder for the duration of each extraction test, so the surrounding
#   project logic runs end to end while the pretrained pitch backbone is
#   never fetched, downloaded, or instantiated
# - Everything else in the extraction path is the real implementation: the
#   torchcrepe silence and hysteresis operators are pure signal processing
#   carrying no learned weights, so they are exercised rather than stubbed
# - Silence gating is driven to both extremes through the configured
#   threshold rather than through frame-level loudness values, because the
#   loudness floor of digital silence is an implementation detail of the
#   third-party weighting curve and pinning it would be brittle
# - The fabricated pitch track is a ramp rather than a constant, because
#   hysteresis whitens the track by its own standard deviation and a
#   constant track makes that step degenerate
# - Design decision boundary: real CREPE inference is never invoked, so the
#   numerical fidelity of the pitch backbone itself is out of scope here and
#   belongs to an evaluation run with retained author weights
#
# Author: Rahul Sawhney

import unittest
from collections.abc import Callable
from typing import ClassVar

import torch
import torchcrepe
from pydantic import BaseModel, ConfigDict, ValidationError

from vocode.metrics.pitch import PitchConfig, PitchExtractor, PitchFeatures


class RecordedCrepeRequest(BaseModel):
    # Frozen record of the arguments the extractor forwarded to the CREPE
    # prediction entry point.
    #
    # Capturing the request as a validated record rather than a loose
    # mapping is what lets the settings-forwarding tests assert that each
    # configured value reached the backbone instead of a library default.
    #
    # Fields:
    #     audio_shape: Shape of the waveform as the backbone received it,
    #         proving the extractor reshaped into a batch of one.
    #     audio_dtype: Dtype as received, proving the cast to single
    #         precision happened before inference.
    #     sample_rate: Rate forwarded from the configuration.
    #     hop_length: Frame advance forwarded from the configuration.
    #     fmin: Lower search bound forwarded from the configuration.
    #     fmax: Upper search bound forwarded from the configuration.
    #     model_capacity: Backbone capacity forwarded from the
    #         configuration under the library's own argument name.
    #     batch_size: Inference batch size forwarded from the
    #         configuration.
    #     device: Execution device forwarded from the configuration.
    #     return_periodicity: Whether the periodicity curve was requested
    #         alongside the pitch track, which the shared extraction
    #         requires unconditionally.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True
    )
    audio_shape: tuple[int, ...]
    audio_dtype: torch.dtype
    sample_rate: int
    hop_length: int
    fmin: float
    fmax: float
    model_capacity: str
    batch_size: int
    device: str
    return_periodicity: bool


class SyntheticToneBuilder:
    # Builds deterministic tone waveforms standing in for recorded speech.
    def __init__(self, sample_rate: int) -> None:
        # Binds the rate the generated tone is sampled at.
        self._sample_rate: int = sample_rate

    def build(self, sample_count: int, frequency_hertz: float, amplitude: float) -> torch.Tensor:
        # Generates a single-channel sine of the requested length.
        time_index: torch.Tensor = torch.arange(sample_count, dtype=torch.float32)
        phase: torch.Tensor = 2.0 * torch.pi * frequency_hertz * time_index / self._sample_rate
        return amplitude * torch.sin(phase)


class CrepePredictionSubstitution:
    # Replaces the CREPE prediction entry point with a fabricated frame-level
    # response so the extraction path runs without pretrained pitch weights.
    #
    # The substitution is deliberately narrow. Only the prediction entry
    # point is redirected; the silence gate, the hysteresis thresholding,
    # the reshaping, and the casting all remain the real implementation, so
    # what these tests exercise is the project's own extraction logic
    # rather than a mock of it. The responder additionally records the
    # request it received, which turns settings forwarding into an
    # observable property instead of an assumed one.
    #
    # Integration: install and restore bracket every test that scores, and
    # restore must run in tearDown even when the test fails, because the
    # entry point is patched on the imported module and would otherwise
    # leak into unrelated tests in the same process.
    def __init__(
        self,
        base_pitch_hertz: float,
        pitch_step_hertz: float,
        voiced_frame_ratio: float,
        voiced_periodicity: float,
        unvoiced_periodicity: float
    ) -> None:
        # Binds the fabrication parameters and captures the entry point that
        # will be restored afterwards.
        self._base_pitch_hertz: float = base_pitch_hertz
        self._pitch_step_hertz: float = pitch_step_hertz
        self._voiced_frame_ratio: float = voiced_frame_ratio
        self._voiced_periodicity: float = voiced_periodicity
        self._unvoiced_periodicity: float = unvoiced_periodicity
        self._original_predict: Callable[..., object] = torchcrepe.predict
        self._call_count: int = 0
        self._voiced_frame_count: int = 0
        self._recorded_request: RecordedCrepeRequest | None = None
        self._produced_pitch: torch.Tensor = torch.zeros(0, dtype=torch.float32)
        self._produced_periodicity: torch.Tensor = torch.zeros(0, dtype=torch.float32)

    def install(self) -> None:
        # Redirects the CREPE entry point at the fabricated responder.
        torchcrepe.predict = self._respond

    def restore(self) -> None:
        # Reinstates the real CREPE entry point.
        torchcrepe.predict = self._original_predict

    @property
    def call_count(self) -> int:
        # Returns how often the extractor asked CREPE for a prediction.
        return self._call_count

    @property
    def voiced_frame_count(self) -> int:
        # Returns how many leading frames were fabricated as confidently
        # periodic.
        return self._voiced_frame_count

    @property
    def recorded_request(self) -> RecordedCrepeRequest:
        # Returns the arguments of the most recent prediction request.
        if self._recorded_request is None:
            raise ValueError("No prediction was requested, so no arguments were recorded")
        return self._recorded_request

    @property
    def produced_pitch(self) -> torch.Tensor:
        # Returns a copy of the pitch track handed back to the extractor.
        return self._produced_pitch.clone()

    @property
    def produced_periodicity(self) -> torch.Tensor:
        # Returns a copy of the periodicity curve handed back to the
        # extractor.
        return self._produced_periodicity.clone()

    def _respond(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        hop_length: int,
        fmin: float,
        fmax: float,
        model: str,
        batch_size: int,
        device: str,
        return_periodicity: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Records the request and fabricates a ramped pitch track alongside a
        # periodicity curve split into a confident and an unconfident region.
        #
        # The frame count is derived from the audio length and hop so the
        # fabricated curves have the geometry the real backbone would have
        # produced. The pitch track ramps rather than holding constant
        # because hysteresis whitens the track by its own standard
        # deviation, which a constant track would make degenerate. The
        # periodicity curve is split into a confident leading region and an
        # unconfident tail so the voicing decision derived from it has a
        # known, assertable boundary.
        self._call_count: int = self._call_count + 1
        frame_count: int = audio.shape[-1] // hop_length + 1
        voiced_frame_count: int = int(frame_count * self._voiced_frame_ratio)
        frame_index: torch.Tensor = torch.arange(frame_count, dtype=torch.float32)
        pitch: torch.Tensor = self._base_pitch_hertz + self._pitch_step_hertz * frame_index
        periodicity: torch.Tensor = torch.cat([
            torch.full((voiced_frame_count,), self._voiced_periodicity, dtype=torch.float32),
            torch.full(
                (frame_count - voiced_frame_count,),
                self._unvoiced_periodicity,
                dtype=torch.float32
            )
        ])
        self._voiced_frame_count: int = voiced_frame_count
        self._produced_pitch: torch.Tensor = pitch.clone()
        self._produced_periodicity: torch.Tensor = periodicity.clone()
        self._recorded_request: RecordedCrepeRequest | None = RecordedCrepeRequest(
            audio_shape=tuple(audio.shape),
            audio_dtype=audio.dtype,
            sample_rate=sample_rate,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            model_capacity=model,
            batch_size=batch_size,
            device=device,
            return_periodicity=return_periodicity
        )
        return pitch.reshape(1, -1), periodicity.reshape(1, -1)


class PitchConfigurationDefaultsTest(unittest.TestCase):
    # Verifies the declared default extraction settings.
    def setUp(self) -> None:
        # Builds the configuration from its defaults.
        self._configuration: PitchConfig = PitchConfig()

    def test_default_audio_grid_matches_the_study_protocol(self) -> None:
        # The default grid is the corpus rate with a 256-sample hop.
        self.assertEqual(self._configuration.sample_rate, 22050)
        self.assertEqual(self._configuration.hop_length, 256)

    def test_default_search_range_spans_the_speech_band(self) -> None:
        # The default pitch search runs from 50 to 550 hertz.
        self.assertAlmostEqual(self._configuration.fmin_hertz, 50.0, places=6)
        self.assertAlmostEqual(self._configuration.fmax_hertz, 550.0, places=6)

    def test_default_inference_settings_are_full_capacity_on_processor(self) -> None:
        # The default evaluator is the full CREPE capacity batched on the
        # processor.
        self.assertEqual(self._configuration.model_capacity, "full")
        self.assertEqual(self._configuration.batch_size, 512)
        self.assertEqual(self._configuration.device, "cpu")

    def test_default_silence_gate_sits_at_minus_sixty_decibels(self) -> None:
        # The default silence gate is 60 decibels below full scale.
        self.assertEqual(self._configuration.silence_threshold_dbfs, -60)


class PitchConfigurationValidationTest(unittest.TestCase):
    # Verifies that the configuration record rejects malformed settings and
    # refuses mutation.
    def test_configuration_is_immutable(self) -> None:
        # A frozen record cannot be rebound after construction.
        configuration: PitchConfig = PitchConfig()
        with self.assertRaises(ValidationError):
            configuration.sample_rate = 8000

    def test_unknown_settings_are_rejected(self) -> None:
        # Extra fields are forbidden so a misspelt setting cannot be ignored.
        with self.assertRaises(ValidationError):
            PitchConfig(sampling_rate=22050)

    def test_non_positive_sample_rate_is_rejected(self) -> None:
        # A rate of zero describes no audio grid at all.
        with self.assertRaises(ValidationError):
            PitchConfig(sample_rate=0)

    def test_non_positive_hop_length_is_rejected(self) -> None:
        # A hop of zero would place every frame at the same offset.
        with self.assertRaises(ValidationError):
            PitchConfig(hop_length=0)

    def test_non_positive_search_bound_is_rejected(self) -> None:
        # A lower search bound of zero hertz is not a pitch.
        with self.assertRaises(ValidationError):
            PitchConfig(fmin_hertz=0.0)

    def test_fractional_sample_rate_is_rejected(self) -> None:
        # Strict validation refuses a float where an integer rate is
        # declared.
        with self.assertRaises(ValidationError):
            PitchConfig(sample_rate=22050.5)

    def test_string_sample_rate_is_rejected(self) -> None:
        # Strict validation refuses to coerce a numeric string.
        with self.assertRaises(ValidationError):
            PitchConfig(sample_rate="22050")

    def test_unknown_model_capacity_is_rejected(self) -> None:
        # The capacity vocabulary is closed to the two published sizes.
        with self.assertRaises(ValidationError):
            PitchConfig(model_capacity="medium")

    def test_published_model_capacities_are_accepted(self) -> None:
        # Both published CREPE capacities are valid settings.
        tiny_configuration: PitchConfig = PitchConfig(model_capacity="tiny")
        full_configuration: PitchConfig = PitchConfig(model_capacity="full")
        self.assertEqual(tiny_configuration.model_capacity, "tiny")
        self.assertEqual(full_configuration.model_capacity, "full")

    def test_negative_silence_threshold_is_accepted(self) -> None:
        # The silence gate is a level below full scale and is therefore
        # negative.
        configuration: PitchConfig = PitchConfig(silence_threshold_dbfs=-90)
        self.assertEqual(configuration.silence_threshold_dbfs, -90)

    def test_fractional_silence_threshold_is_rejected(self) -> None:
        # Strict validation refuses a float where an integer level is
        # declared.
        with self.assertRaises(ValidationError):
            PitchConfig(silence_threshold_dbfs=-60.5)


class PitchFeaturesRecordTest(unittest.TestCase):
    # Verifies the frozen feature bundle exchanged between the extractor and
    # the pitch-family metrics.
    def setUp(self) -> None:
        # Builds one well-formed feature bundle.
        self._features: PitchFeatures = PitchFeatures(
            pitch=torch.tensor([220.0, 225.0], dtype=torch.float32),
            periodicity=torch.tensor([0.9, 0.2], dtype=torch.float32),
            voicing=torch.tensor([True, False], dtype=torch.bool)
        )

    def test_tensor_fields_are_retained_unchanged(self) -> None:
        # The record carries torch tensors through without conversion.
        self.assertIsInstance(self._features.pitch, torch.Tensor)
        self.assertIsInstance(self._features.periodicity, torch.Tensor)
        self.assertEqual(self._features.voicing.dtype, torch.bool)

    def test_record_is_immutable(self) -> None:
        # A frozen bundle cannot be rebound after construction.
        with self.assertRaises(ValidationError):
            self._features.pitch = torch.zeros(2, dtype=torch.float32)

    def test_missing_field_is_rejected(self) -> None:
        # All three curves are mandatory, because every consumer reads at
        # least one of them.
        with self.assertRaises(ValidationError):
            PitchFeatures(
                pitch=torch.zeros(2, dtype=torch.float32),
                periodicity=torch.zeros(2, dtype=torch.float32)
            )

    def test_unknown_field_is_rejected(self) -> None:
        # Extra fields are forbidden so a renamed curve cannot pass silently.
        with self.assertRaises(ValidationError):
            PitchFeatures(
                pitch=torch.zeros(2, dtype=torch.float32),
                periodicity=torch.zeros(2, dtype=torch.float32),
                voicing=torch.zeros(2, dtype=torch.bool),
                confidence=torch.zeros(2, dtype=torch.float32)
            )

    def test_non_tensor_field_is_rejected(self) -> None:
        # Strict validation refuses a plain list where a tensor is declared.
        with self.assertRaises(ValidationError):
            PitchFeatures(
                pitch=[220.0, 225.0],
                periodicity=torch.zeros(2, dtype=torch.float32),
                voicing=torch.zeros(2, dtype=torch.bool)
            )


class PitchExtractorConstructionTest(unittest.TestCase):
    # Verifies that constructing the extractor binds its settings without
    # reaching for the pitch backbone.
    def setUp(self) -> None:
        # Installs the fabricated prediction responder before construction.
        self._substitution: CrepePredictionSubstitution = CrepePredictionSubstitution(
            200.0, 5.0, 0.5, 0.9, 0.02
        )
        self._substitution.install()
        self._configuration: PitchConfig = PitchConfig(sample_rate=16000, model_capacity="tiny")

    def tearDown(self) -> None:
        # Reinstates the real prediction entry point.
        self._substitution.restore()

    def test_configuration_is_exposed_unchanged(self) -> None:
        # The extractor hands back the exact record it was constructed with.
        extractor: PitchExtractor = PitchExtractor(self._configuration)
        self.assertIs(extractor.configuration, self._configuration)

    def test_construction_does_not_request_a_prediction(self) -> None:
        # Building the threshold operators must not trigger pitch inference.
        PitchExtractor(self._configuration)
        self.assertEqual(
            self._substitution.call_count,
            0,
            msg="Constructing the extractor must not invoke the pitch backbone"
        )


class PitchExtractorFeatureDerivationTest(unittest.TestCase):
    # Verifies how the extractor shapes, gates, and thresholds the fabricated
    # prediction into the shared feature bundle.
    #
    # The silence gate is driven to its two extremes through the configured
    # threshold rather than through frame loudness values: a gate at minus
    # one hundred decibels sits below every frame of the test tone and must
    # leave the curve untouched, while a gate at zero decibels sits above
    # every frame and must zero it entirely. Choosing the extremes this way
    # avoids pinning the loudness floor of the third-party weighting curve,
    # which is an implementation detail that could shift between releases.
    def setUp(self) -> None:
        # Installs the fabricated responder and prepares a loud test tone.
        self._sample_rate: int = 16000
        self._hop_length: int = 256
        self._tone_builder: SyntheticToneBuilder = SyntheticToneBuilder(self._sample_rate)
        self._waveform: torch.Tensor = self._tone_builder.build(4096, 1000.0, 0.5)
        self._substitution: CrepePredictionSubstitution = CrepePredictionSubstitution(
            200.0, 5.0, 0.5, 0.9, 0.02
        )
        self._substitution.install()

    def tearDown(self) -> None:
        # Reinstates the real prediction entry point.
        self._substitution.restore()

    def test_features_are_flattened_to_one_dimension(self) -> None:
        # All three curves leave the extractor as parallel one-dimensional
        # frame sequences.
        features: PitchFeatures = self._build_extractor(-100).extract(self._waveform)
        self.assertEqual(features.pitch.dim(), 1)
        self.assertEqual(features.periodicity.dim(), 1)
        self.assertEqual(features.voicing.dim(), 1)
        self.assertEqual(features.periodicity.shape, features.pitch.shape)
        self.assertEqual(features.voicing.shape, features.pitch.shape)

    def test_pitch_track_is_returned_ungated(self) -> None:
        # The raw pitch track reaches the consumer untouched by the voicing
        # decision.
        features: PitchFeatures = self._build_extractor(-100).extract(self._waveform)
        self.assertTrue(
            torch.equal(features.pitch, self._substitution.produced_pitch),
            msg="The predicted pitch track must be returned without modification"
        )

    def test_permissive_silence_gate_preserves_the_periodicity_curve(self) -> None:
        # A gate below the loudness of every frame leaves the curve intact.
        features: PitchFeatures = self._build_extractor(-100).extract(self._waveform)
        self.assertTrue(
            torch.equal(features.periodicity, self._substitution.produced_periodicity),
            msg="An unreachable silence gate must not alter the periodicity curve"
        )

    def test_voicing_follows_the_periodicity_split(self) -> None:
        # Frames fabricated as confidently periodic survive hysteresis
        # thresholding and the unconfident tail does not.
        features: PitchFeatures = self._build_extractor(-100).extract(self._waveform)
        frame_count: int = int(features.voicing.shape[0])
        voiced_frame_count: int = self._substitution.voiced_frame_count
        expected_voicing: torch.Tensor = torch.cat([
            torch.ones(voiced_frame_count, dtype=torch.bool),
            torch.zeros(frame_count - voiced_frame_count, dtype=torch.bool)
        ])
        self.assertTrue(
            torch.equal(features.voicing, expected_voicing),
            msg=f"Expected {voiced_frame_count} leading voiced frames, got {features.voicing.tolist()}"
        )

    def test_voicing_is_reported_as_boolean_decisions(self) -> None:
        # Voicing is a decision mask rather than a confidence curve.
        features: PitchFeatures = self._build_extractor(-100).extract(self._waveform)
        self.assertEqual(features.voicing.dtype, torch.bool)

    def test_silence_gate_zeroes_the_periodicity_curve(self) -> None:
        # A gate above the loudness of every frame silences the whole curve.
        features: PitchFeatures = self._build_extractor(0).extract(self._waveform)
        self.assertTrue(
            torch.all(features.periodicity == 0.0).item(),
            msg=f"Every frame must be gated to zero, got {features.periodicity.tolist()}"
        )

    def test_gated_periodicity_removes_every_voicing_decision(self) -> None:
        # Digital silence cannot register as voiced material.
        features: PitchFeatures = self._build_extractor(0).extract(self._waveform)
        self.assertFalse(
            bool(features.voicing.any().item()),
            msg=f"Gated audio must be fully unvoiced, got {features.voicing.tolist()}"
        )

    def test_pitch_survives_full_gating_because_it_is_never_masked(self) -> None:
        # Gating drives voicing to nothing yet leaves the pitch track intact,
        # keeping pitch and voicing separable for the consumers.
        features: PitchFeatures = self._build_extractor(0).extract(self._waveform)
        self.assertTrue(
            torch.equal(features.pitch, self._substitution.produced_pitch),
            msg="Full silence gating must not modify the returned pitch track"
        )
        self.assertFalse(
            bool(torch.isnan(features.pitch).any().item()),
            msg="The returned pitch track must not carry hysteresis unvoiced markers"
        )

    def test_configuration_is_forwarded_to_the_prediction_request(self) -> None:
        # Every extraction setting reaches the backbone rather than a default.
        extractor: PitchExtractor = self._build_extractor(-100)
        extractor.extract(self._waveform)
        request: RecordedCrepeRequest = self._substitution.recorded_request
        self.assertEqual(request.sample_rate, self._sample_rate)
        self.assertEqual(request.hop_length, self._hop_length)
        self.assertAlmostEqual(request.fmin, extractor.configuration.fmin_hertz, places=6)
        self.assertAlmostEqual(request.fmax, extractor.configuration.fmax_hertz, places=6)
        self.assertEqual(request.model_capacity, "tiny")
        self.assertEqual(request.batch_size, extractor.configuration.batch_size)
        self.assertEqual(request.device, "cpu")

    def test_periodicity_is_always_requested_from_the_backbone(self) -> None:
        # The shared extraction needs the periodicity curve, not the pitch
        # track alone.
        self._build_extractor(-100).extract(self._waveform)
        request: RecordedCrepeRequest = self._substitution.recorded_request
        self.assertTrue(
            request.return_periodicity,
            msg="The extractor must request the periodicity curve alongside the pitch track"
        )

    def test_waveform_is_reshaped_and_cast_before_prediction(self) -> None:
        # A flat double-precision waveform reaches the backbone as a
        # single-precision batch of one.
        double_waveform: torch.Tensor = self._waveform.double()
        self._build_extractor(-100).extract(double_waveform)
        request: RecordedCrepeRequest = self._substitution.recorded_request
        self.assertEqual(request.audio_shape, (1, int(self._waveform.shape[0])))
        self.assertEqual(request.audio_dtype, torch.float32)

    def test_extraction_requests_one_prediction_per_utterance(self) -> None:
        # One CREPE call serves all three pitch-family metrics.
        self._build_extractor(-100).extract(self._waveform)
        self.assertEqual(
            self._substitution.call_count,
            1,
            msg="Extraction must invoke the pitch backbone exactly once"
        )

    def _build_extractor(self, silence_threshold_dbfs: int) -> PitchExtractor:
        # Builds an extractor on the test audio grid with the requested
        # silence gate.
        configuration: PitchConfig = PitchConfig(
            sample_rate=self._sample_rate,
            hop_length=self._hop_length,
            model_capacity="tiny",
            silence_threshold_dbfs=silence_threshold_dbfs
        )
        return PitchExtractor(configuration)


if __name__ == "__main__":
    unittest.main()
