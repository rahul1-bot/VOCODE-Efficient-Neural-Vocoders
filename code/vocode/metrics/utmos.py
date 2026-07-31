# This module:
# 1. Predicts the UTMOS strong-model mean-opinion score of candidate audio
#    through the pinned SpeechMOS torch.hub release
#
# Design decisions:
# - The evaluator identity is pinned to an exact hub repository tag and
#   model name, because an unpinned perceptual predictor would make scores
#   incomparable across evaluation dates
# - UTMOS is a no-reference predictor: it scores the candidate alone, so
#   it complements the reference-based metrics rather than replacing them
# - Audio is resampled to the predictor's sixteen-kilohertz operating rate
#   whenever the source rate differs
# - The hub model loads lazily on first use, so constructing the metric is
#   inexpensive and network access happens only when scoring actually runs
#
# Author: Rahul Sawhney

from typing import ClassVar, cast

import torch
import torchaudio
from pydantic import BaseModel, ConfigDict, PositiveInt

__all__: list[str] = ["UtmosPredictor", "UtmosPredictorConfig"]


class UtmosPredictorConfig(BaseModel):
    # Frozen evaluator identity: hub repository tag, model name, operating
    # sample rate, and execution device.
    #
    # The first two fields together are the reproducibility anchor of every
    # score this metric produces. A perceptual predictor is itself a
    # trained model, so a silently moving revision would change published
    # numbers without changing any code; pinning an exact release tag is
    # what makes scores comparable across evaluation dates.
    #
    # Fields:
    #     hub_repository: Torch hub repository and release tag the
    #         predictor is loaded from, pinned rather than tracking a
    #         branch. Default: ``"tarepan/SpeechMOS:v1.2.0"``.
    #     hub_model_name: Entry point within that repository, naming the
    #         strong UTMOS variant.
    #         Default: ``"utmos22_strong"``.
    #     target_sample_rate: Rate in hertz the candidate is resampled to
    #         before scoring, and the rate reported to the predictor, so
    #         the predictor is always told the rate it actually receives.
    #         Default: ``16000``.
    #     device: Torch device string the loaded predictor is moved to and
    #         scored on. Default: ``"cpu"``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    hub_repository: str = "tarepan/SpeechMOS:v1.2.0"
    hub_model_name: str = "utmos22_strong"
    target_sample_rate: PositiveInt = 16000
    device: str = "cpu"


class UtmosPredictor:
    # No-reference mean-opinion-score predictor with a lazily loaded,
    # identity-pinned hub model.
    def __init__(self, configuration: UtmosPredictorConfig) -> None:
        # Binds the evaluator identity; the model slot stays empty until
        # the first scoring call.
        self._configuration: UtmosPredictorConfig = configuration
        self._model: torch.nn.Module | None = None

    def __call__(self, candidate: torch.Tensor, source_sample_rate: int) -> float:
        # Scores one candidate waveform: flattens to the predictor's batch
        # layout, resamples to the operating rate when needed, and
        # evaluates on the configured device without gradient tracking.
        #
        # No reference signal is taken, because UTMOS judges the candidate
        # on its own. The first call also loads the pinned hub model, so
        # the first score of a pass is materially slower than the rest.
        #
        # Args:
        #     candidate: The synthesized signal to score, in any shape; it
        #         is flattened into a batch of one before scoring.
        #     source_sample_rate: Rate the candidate is currently sampled
        #         at. Resampling is skipped when it already equals the
        #         configured operating rate.
        #
        # Returns:
        #     The predicted mean-opinion score, on which higher is better,
        #     reduced from the predictor's batched output to one scalar.
        model: torch.nn.Module = self._require_model()
        flat_candidate: torch.Tensor = candidate.reshape(1, -1).float()
        target_rate: int = self._configuration.target_sample_rate
        if source_sample_rate != target_rate:
            flat_candidate: torch.Tensor = torchaudio.functional.resample(
                flat_candidate, orig_freq=source_sample_rate, new_freq=target_rate
            )
        with torch.no_grad():
            score: torch.Tensor = model(flat_candidate.to(self._configuration.device), target_rate)
        return float(score.reshape(-1)[0].item())

    @property
    def configuration(self) -> UtmosPredictorConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    def _require_model(self) -> torch.nn.Module:
        # Loads the pinned hub model on first use, switches it to
        # evaluation mode on the configured device, and caches it for the
        # remainder of the pass.
        if self._model is None:
            loaded_model: torch.nn.Module = cast(
                torch.nn.Module,
                torch.hub.load(
                    self._configuration.hub_repository,
                    self._configuration.hub_model_name,
                    trust_repo=True
                )
            )
            resolved_model: torch.nn.Module = loaded_model.eval().to(self._configuration.device)
            self._model: torch.nn.Module | None = resolved_model
            return resolved_model
        return self._model
