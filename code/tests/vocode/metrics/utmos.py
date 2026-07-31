# This module:
# 1. Verifies the frozen UTMOS evaluator-identity record and its validation
#    rules, including the pinned hub release the study scores against
# 2. Verifies the predictor logic that surrounds the hub model: lazy loading,
#    single-load caching, evaluation mode, waveform normalization, the
#    resampling decision, and gradient-free scoring
#
# Design decisions:
# - The torch.hub entry point is replaced by a locally constructed stub
#   module for the duration of each scoring test, so the surrounding project
#   logic runs end to end while the pinned UTMOS predictor weights are never
#   downloaded, fetched from cache, or instantiated
# - Design decision boundary: the perceptual fidelity of the UTMOS strong
#   model is therefore out of scope here; the returned score is asserted as
#   plumbing, not as a mean-opinion-score value, and the numerical behaviour
#   of the real predictor belongs to an evaluation run with the pinned
#   release available
# - The pinned repository tag and model name are asserted literally, because
#   they are the reproducibility anchor that makes scores comparable across
#   evaluation dates and a silent bump would invalidate published numbers
# - The resampling decision is observed through the sample count the stub
#   receives, which is the only externally visible consequence of the branch
#
# Author: Rahul Sawhney

import unittest
from collections.abc import Callable
from typing import ClassVar, override

import torch
from pydantic import BaseModel, ConfigDict, ValidationError

from vocode.metrics.utmos import UtmosPredictor, UtmosPredictorConfig


class RecordedScoringRequest(BaseModel):
    # Frozen record of one waveform the predictor handed to the hub model.
    #
    # The record captures execution state alongside the tensor, because
    # the gradient-tracking and evaluation-mode contracts are observable
    # only from inside the call being made.
    #
    # Fields:
    #     waveform_shape: Shape as the model received it, which is how the
    #         flattening and the resampling decision are both observed;
    #         resampling changes the sample count and nothing else visible.
    #     waveform_dtype: Dtype as received, proving the cast to single
    #         precision preceded scoring.
    #     sample_rate: Rate reported to the model, which must be the
    #         operating rate rather than the source rate.
    #     is_gradient_enabled: Whether autograd was recording during the
    #         call, which must be false for evaluation.
    #     is_training: Whether the module was in training mode during the
    #         call, which must be false so no training-time behaviour such
    #         as dropout perturbs a score.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True
    )
    waveform_shape: tuple[int, ...]
    waveform_dtype: torch.dtype
    sample_rate: int
    is_gradient_enabled: bool
    is_training: bool


class StubMeanOpinionScoreModel(torch.nn.Module):
    # Stands in for the pinned hub predictor, recording how it was called and
    # returning a fixed score.
    #
    # The returned score is a constant in the real predictor's batched
    # output layout, which is enough to verify that the caller reduces that
    # layout to one scalar. Nothing here models perceptual behaviour, and
    # the tests correspondingly assert plumbing rather than score values.
    def __init__(self, score_value: float) -> None:
        # Binds the score to return and prepares the request log.
        super().__init__()
        self._score_value: float = score_value
        self._recordings: list[RecordedScoringRequest] = []

    @override
    def forward(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # Records the received waveform and execution state, then returns the
        # fixed score in the batched layout the real predictor uses.
        self._recordings.append(
            RecordedScoringRequest(
                waveform_shape=tuple(waveform.shape),
                waveform_dtype=waveform.dtype,
                sample_rate=sample_rate,
                is_gradient_enabled=torch.is_grad_enabled(),
                is_training=self.training
            )
        )
        return torch.full((1, 1), self._score_value, dtype=torch.float32)

    @property
    def recordings(self) -> list[RecordedScoringRequest]:
        # Returns a copy of every scoring request observed so far.
        return list(self._recordings)


class HubLoadSubstitution:
    # Replaces the torch.hub entry point with a locally constructed stub so
    # no pretrained predictor is ever retrieved.
    #
    # Redirecting the loader rather than the predictor keeps the lazy-load
    # contract observable: the substitution counts retrievals, so the tests
    # can prove that construction retrieves nothing and that repeated
    # scoring retrieves exactly once. It also records the identity
    # requested, which is how the pinned release is asserted without any
    # network access.
    #
    # Integration: install and restore bracket every test that scores, and
    # restore must run in tearDown even when the test fails, because the
    # entry point is patched on the shared torch.hub module and would
    # otherwise leak into unrelated tests in the same process.
    def __init__(self, model: StubMeanOpinionScoreModel) -> None:
        # Binds the stub to hand back and captures the entry point that will
        # be restored afterwards.
        self._model: StubMeanOpinionScoreModel = model
        self._original_load: Callable[..., object] = torch.hub.load
        self._recorded_repositories: list[str] = []
        self._recorded_model_names: list[str] = []
        self._recorded_trust_flags: list[bool] = []

    def install(self) -> None:
        # Redirects the hub entry point at the local stub.
        torch.hub.load = self._respond

    def restore(self) -> None:
        # Reinstates the real hub entry point.
        torch.hub.load = self._original_load

    @property
    def load_count(self) -> int:
        # Returns how often a model was requested from the hub.
        return len(self._recorded_repositories)

    @property
    def recorded_repositories(self) -> list[str]:
        # Returns a copy of every repository tag requested.
        return list(self._recorded_repositories)

    @property
    def recorded_model_names(self) -> list[str]:
        # Returns a copy of every model name requested.
        return list(self._recorded_model_names)

    @property
    def recorded_trust_flags(self) -> list[bool]:
        # Returns a copy of every repository-trust decision passed.
        return list(self._recorded_trust_flags)

    def _respond(
        self,
        repository: str,
        model_name: str,
        trust_repo: bool
    ) -> StubMeanOpinionScoreModel:
        # Records the requested identity and returns the local stub.
        self._recorded_repositories.append(repository)
        self._recorded_model_names.append(model_name)
        self._recorded_trust_flags.append(trust_repo)
        return self._model


class UtmosPredictorConfigurationDefaultsTest(unittest.TestCase):
    # Verifies the declared default evaluator identity.
    def setUp(self) -> None:
        # Builds the configuration from its defaults.
        self._configuration: UtmosPredictorConfig = UtmosPredictorConfig()

    def test_default_hub_identity_is_pinned_to_an_exact_release(self) -> None:
        # The evaluator is pinned to a tagged release and a named model so
        # scores stay comparable across evaluation dates.
        self.assertEqual(self._configuration.hub_repository, "tarepan/SpeechMOS:v1.2.0")
        self.assertEqual(self._configuration.hub_model_name, "utmos22_strong")

    def test_default_operating_rate_is_sixteen_kilohertz(self) -> None:
        # The predictor operates at its published rate.
        self.assertEqual(self._configuration.target_sample_rate, 16000)

    def test_default_device_is_the_processor(self) -> None:
        # Scoring defaults to the processor so evaluation is reproducible on
        # any machine.
        self.assertEqual(self._configuration.device, "cpu")


class UtmosPredictorConfigurationValidationTest(unittest.TestCase):
    # Verifies that the evaluator-identity record rejects malformed settings
    # and refuses mutation.
    def test_configuration_is_immutable(self) -> None:
        # A frozen identity cannot be rebound after construction.
        configuration: UtmosPredictorConfig = UtmosPredictorConfig()
        with self.assertRaises(ValidationError):
            configuration.hub_repository = "tarepan/SpeechMOS:main"

    def test_unknown_settings_are_rejected(self) -> None:
        # Extra fields are forbidden so a misspelt setting cannot be ignored.
        with self.assertRaises(ValidationError):
            UtmosPredictorConfig(hub_revision="v1.2.0")

    def test_non_positive_target_sample_rate_is_rejected(self) -> None:
        # A rate of zero describes no audio grid at all.
        with self.assertRaises(ValidationError):
            UtmosPredictorConfig(target_sample_rate=0)

    def test_fractional_target_sample_rate_is_rejected(self) -> None:
        # Strict validation refuses a float where an integer rate is
        # declared.
        with self.assertRaises(ValidationError):
            UtmosPredictorConfig(target_sample_rate=16000.5)

    def test_non_string_hub_repository_is_rejected(self) -> None:
        # Strict validation refuses a non-textual repository tag.
        with self.assertRaises(ValidationError):
            UtmosPredictorConfig(hub_repository=1)


class UtmosPredictorLazyLoadingTest(unittest.TestCase):
    # Verifies when the pinned hub model is requested, how often, and under
    # which identity.
    def setUp(self) -> None:
        # Installs the local stub in place of the hub entry point.
        self._model: StubMeanOpinionScoreModel = StubMeanOpinionScoreModel(3.75)
        self._substitution: HubLoadSubstitution = HubLoadSubstitution(self._model)
        self._substitution.install()
        self._waveform: torch.Tensor = torch.zeros(4096, dtype=torch.float32)

    def tearDown(self) -> None:
        # Reinstates the real hub entry point.
        self._substitution.restore()

    def test_construction_does_not_load_the_hub_model(self) -> None:
        # Building the metric is inexpensive; the model slot stays empty
        # until a score is actually requested.
        UtmosPredictor(UtmosPredictorConfig())
        self.assertEqual(
            self._substitution.load_count,
            0,
            msg="Constructing the predictor must not retrieve the hub model"
        )

    def test_first_scoring_call_loads_the_pinned_release(self) -> None:
        # The identity reaching the hub is the pinned repository tag and
        # model name.
        predictor: UtmosPredictor = UtmosPredictor(UtmosPredictorConfig())
        predictor(self._waveform, 16000)
        self.assertEqual(self._substitution.recorded_repositories, ["tarepan/SpeechMOS:v1.2.0"])
        self.assertEqual(self._substitution.recorded_model_names, ["utmos22_strong"])

    def test_repeated_scoring_reuses_the_cached_model(self) -> None:
        # The model is retrieved once and held for the remainder of the pass.
        predictor: UtmosPredictor = UtmosPredictor(UtmosPredictorConfig())
        predictor(self._waveform, 16000)
        predictor(self._waveform, 16000)
        self.assertEqual(
            self._substitution.load_count,
            1,
            msg=f"Expected one retrieval across two scores, saw {self._substitution.load_count}"
        )
        self.assertEqual(len(self._model.recordings), 2)

    def test_configured_identity_overrides_the_default_release(self) -> None:
        # The evaluator identity is driven by the configuration record rather
        # than hardcoded at the call site.
        configuration: UtmosPredictorConfig = UtmosPredictorConfig(
            hub_repository="tarepan/SpeechMOS:v1.0.0",
            hub_model_name="utmos22_base"
        )
        predictor: UtmosPredictor = UtmosPredictor(configuration)
        predictor(self._waveform, 16000)
        self.assertEqual(self._substitution.recorded_repositories, ["tarepan/SpeechMOS:v1.0.0"])
        self.assertEqual(self._substitution.recorded_model_names, ["utmos22_base"])

    def test_pinned_repository_is_loaded_without_an_interactive_prompt(self) -> None:
        # Evaluation runs unattended, so the pinned repository is trusted
        # explicitly rather than confirmed at a prompt.
        predictor: UtmosPredictor = UtmosPredictor(UtmosPredictorConfig())
        predictor(self._waveform, 16000)
        self.assertEqual(self._substitution.recorded_trust_flags, [True])

    def test_loaded_model_is_switched_to_evaluation_mode(self) -> None:
        # Scoring must not run the predictor with training-time behaviour.
        predictor: UtmosPredictor = UtmosPredictor(UtmosPredictorConfig())
        predictor(self._waveform, 16000)
        self.assertFalse(
            self._model.recordings[0].is_training,
            msg="The hub model must be switched to evaluation mode before scoring"
        )


class UtmosPredictorScoringTest(unittest.TestCase):
    # Verifies how one candidate waveform is normalized, resampled, and
    # reduced to a scalar score.
    #
    # The resampling branch is observed through the sample count the stub
    # receives, which is its only externally visible consequence. The
    # not-resampled case deliberately uses a length no resampler would
    # preserve by coincidence, so an accidental round trip through the
    # resampler could not pass unnoticed.
    def setUp(self) -> None:
        # Installs the local stub and prepares a predictor at the default
        # operating rate.
        self._score_value: float = 3.75
        self._model: StubMeanOpinionScoreModel = StubMeanOpinionScoreModel(self._score_value)
        self._substitution: HubLoadSubstitution = HubLoadSubstitution(self._model)
        self._substitution.install()
        self._configuration: UtmosPredictorConfig = UtmosPredictorConfig()
        self._predictor: UtmosPredictor = UtmosPredictor(self._configuration)
        self._sample_count: int = 4096

    def tearDown(self) -> None:
        # Reinstates the real hub entry point.
        self._substitution.restore()

    def test_score_is_returned_as_a_python_float(self) -> None:
        # The batched predictor output is reduced to one scalar per
        # utterance.
        score: float = self._predictor(torch.zeros(self._sample_count), 16000)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(
            score,
            self._score_value,
            places=6,
            msg=f"Expected the predicted {self._score_value}, got {score}"
        )

    def test_waveform_is_flattened_to_the_batch_layout(self) -> None:
        # A flat waveform reaches the predictor as a batch of one.
        self._predictor(torch.zeros(self._sample_count), 16000)
        self.assertEqual(self._model.recordings[0].waveform_shape, (1, self._sample_count))

    def test_multi_channel_waveform_is_flattened_to_the_batch_layout(self) -> None:
        # A shaped waveform is reduced to the same batched layout rather than
        # forwarded with its original dimensions.
        self._predictor(torch.zeros(1, self._sample_count), 16000)
        self.assertEqual(self._model.recordings[0].waveform_shape, (1, self._sample_count))

    def test_waveform_is_cast_to_single_precision(self) -> None:
        # Double-precision audio is narrowed before scoring.
        self._predictor(torch.zeros(self._sample_count, dtype=torch.float64), 16000)
        self.assertEqual(self._model.recordings[0].waveform_dtype, torch.float32)

    def test_matching_sample_rate_is_not_resampled(self) -> None:
        # Audio already at the operating rate keeps its exact sample count,
        # including lengths no resampler would preserve by chance.
        source_sample_count: int = 3001
        self._predictor(torch.zeros(source_sample_count), 16000)
        self.assertEqual(self._model.recordings[0].waveform_shape, (1, source_sample_count))

    def test_lower_sample_rate_is_resampled_to_the_operating_rate(self) -> None:
        # Halving the source rate doubles the sample count on the way to the
        # predictor.
        source_sample_count: int = 2048
        self._predictor(torch.zeros(source_sample_count), 8000)
        self.assertEqual(self._model.recordings[0].waveform_shape, (1, source_sample_count * 2))

    def test_operating_rate_is_reported_to_the_predictor(self) -> None:
        # The predictor is told the rate it actually receives, not the source
        # rate.
        self._predictor(torch.zeros(2048), 8000)
        self.assertEqual(self._model.recordings[0].sample_rate, 16000)

    def test_scoring_runs_without_gradient_tracking(self) -> None:
        # Evaluation builds no autograd graph.
        self._predictor(torch.zeros(self._sample_count), 16000)
        self.assertFalse(
            self._model.recordings[0].is_gradient_enabled,
            msg="Scoring must run with gradient tracking disabled"
        )

    def test_configuration_is_exposed_unchanged(self) -> None:
        # The predictor hands back the exact record it was constructed with.
        self.assertIs(self._predictor.configuration, self._configuration)


if __name__ == "__main__":
    unittest.main()
