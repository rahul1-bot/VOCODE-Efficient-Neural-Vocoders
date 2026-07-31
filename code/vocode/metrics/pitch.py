# This module:
# 1. Extracts the pitch-family features shared by the F0, periodicity, and
#    voicing metrics through the torchcrepe CREPE implementation: the pitch
#    track, the silence-gated periodicity curve, and the hysteresis-derived
#    voicing decisions
#
# Design decisions:
# - One extraction serves all three pitch-family metrics, because CREPE
#   inference is the expensive step and the metrics differ only in how they
#   consume its outputs
# - Periodicity is silence-gated before thresholding so digital silence
#   cannot register as voiced material
# - Voicing decisions come from torchcrepe's hysteresis thresholding of the
#   gated periodicity, which suppresses single-frame voicing flicker
# - The raw pitch track is returned ungated; consumers apply the voicing
#   mask themselves, keeping pitch and voicing decisions separable
#
# Author: Rahul Sawhney

import warnings
from typing import ClassVar, Literal, cast

import torch
import torchcrepe
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

__all__: list[str] = ["PitchConfig", "PitchExtractor", "PitchFeatures"]


class PitchConfig(BaseModel):
    # Frozen extraction settings: audio grid, pitch search range, CREPE
    # capacity, silence gate level, inference batch size, and device.
    #
    # Fields:
    #     sample_rate: Rate in hertz at which the analysed waveform is
    #         sampled; it is forwarded to the backbone so the frame grid
    #         matches the audio. Default: ``22050``, the corpus rate.
    #     hop_length: Samples advanced between consecutive analysis frames,
    #         which fixes the frame rate shared by all three returned
    #         curves. Default: ``256``.
    #     fmin_hertz: Lower bound of the pitch search; no fundamental below
    #         this frequency is reported. Default: ``50.0``.
    #     fmax_hertz: Upper bound of the pitch search. Default: ``550.0``.
    #     model_capacity: CREPE backbone size, restricted to the two
    #         published capacities. Default: ``"full"``.
    #     silence_threshold_dbfs: Loudness level below full scale under
    #         which a frame's periodicity is gated to zero, so digital
    #         silence cannot register as voiced material. Stated as a
    #         negative level. Default: ``-60``.
    #     batch_size: Frames submitted per backbone inference batch.
    #         Default: ``512``.
    #     device: Torch device string the backbone inference executes on.
    #         Default: ``"cpu"``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    sample_rate: PositiveInt = 22050
    hop_length: PositiveInt = 256
    fmin_hertz: PositiveFloat = 50.0
    fmax_hertz: PositiveFloat = 550.0
    model_capacity: Literal["tiny", "full"] = "full"
    silence_threshold_dbfs: int = -60
    batch_size: PositiveInt = 512
    device: str = "cpu"


class PitchFeatures(BaseModel):
    # Frozen per-utterance feature bundle: the frame-level pitch track in
    # hertz, the silence-gated periodicity curve, and the boolean voicing
    # decisions, all flattened to one dimension.
    #
    # The three curves are parallel: entry i of each describes the same
    # analysis frame, so a consumer may index them against one another
    # without realignment. All three are mandatory because the pitch-family
    # metrics between them read every one.
    #
    # Fields:
    #     pitch: Frame-level fundamental frequency in hertz exactly as the
    #         backbone predicted it, carrying no voicing mask and no unvoiced
    #         markers, so pitch and voicing stay separable for consumers.
    #     periodicity: Frame-level periodicity confidence after silence
    #         gating; frames gated as silent read zero.
    #     voicing: Boolean per-frame voicing decisions derived by hysteresis
    #         thresholding of the gated periodicity, which suppresses
    #         single-frame voicing flicker.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True
    )
    pitch: torch.Tensor
    periodicity: torch.Tensor
    voicing: torch.Tensor


class PitchExtractor:
    # CREPE-based extractor producing the shared pitch-family features. One
    # call to extract runs the pitch backbone exactly once and returns all
    # three curves the pitch-family metrics consume, because backbone
    # inference dominates the cost of the family while the three metrics
    # differ only in which curve they read. The instance is stateless
    # between calls: it holds the settings and the two threshold operators,
    # accumulates nothing, and is therefore safe to reuse across every
    # utterance of a pass.
    #
    # Integration: the metric sequence owns the extractor and performs the
    # extraction itself, calling extract once for the reference signal and
    # once for the candidate, then handing both bundles unchanged to
    # F0Rmse, PeriodicityRmse, and VoicingF1 in turn. Those three metrics
    # never invoke the extractor. One utterance pair therefore costs exactly
    # two backbone invocations no matter how many pitch-family metrics the
    # run selected, and all three metrics of a pass observe identical pitch,
    # periodicity, and voicing evidence.
    def __init__(self, configuration: PitchConfig) -> None:
        # Binds the settings and constructs the torchcrepe silence and
        # hysteresis threshold operators. Construction touches no pretrained
        # weights; the backbone is reached only when extract runs.
        self._configuration: PitchConfig = configuration
        self._hysteresis: torchcrepe.threshold.Hysteresis = torchcrepe.threshold.Hysteresis()
        self._silence: torchcrepe.threshold.Silence = torchcrepe.threshold.Silence(
            configuration.silence_threshold_dbfs
        )

    def extract(self, waveform: torch.Tensor) -> PitchFeatures:
        # Runs CREPE prediction with periodicity, gates the periodicity on
        # silence, derives voicing through hysteresis thresholding (whose
        # all-NaN warning on unvoiced audio is expected and suppressed), and
        # returns the flattened feature bundle.
        #
        # Args:
        #     waveform: One utterance of audio in any shape; it is reshaped
        #         into a single-channel batch of one, cast to float32, and
        #         moved to the CPU before inference, so the backbone always
        #         receives the layout it requires regardless of how the
        #         caller held the signal.
        #
        # Returns:
        #     A PitchFeatures bundle whose three curves share one frame
        #     grid. Hysteresis thresholding writes unvoiced markers into its
        #     own copy of the pitch track; that copy is read as the voicing
        #     mask and then discarded, so the returned pitch track is the
        #     backbone's raw prediction and carries no markers of its own.
        # Prediction: one backbone invocation returns the pitch track and
        # the ungated periodicity curve together.
        audio: torch.Tensor = waveform.reshape(1, -1).float().cpu()
        prediction: tuple[torch.Tensor, torch.Tensor] = cast(
            tuple[torch.Tensor, torch.Tensor],
            torchcrepe.predict(
                audio,
                sample_rate=self._configuration.sample_rate,
                hop_length=self._configuration.hop_length,
                fmin=self._configuration.fmin_hertz,
                fmax=self._configuration.fmax_hertz,
                model=self._configuration.model_capacity,
                batch_size=self._configuration.batch_size,
                device=self._configuration.device,
                return_periodicity=True
            )
        )
        pitch: torch.Tensor = prediction[0].detach().cpu()
        periodicity: torch.Tensor = prediction[1].detach().cpu()
        # Gating: frames quieter than the configured level lose their
        # periodicity before any voicing decision is taken.
        gated_periodicity: torch.Tensor = self._silence(
            periodicity.clone(),
            audio,
            self._configuration.sample_rate,
            self._configuration.hop_length
        )
        # Voicing: hysteresis marks unvoiced frames of a throwaway copy of
        # the track, and the surviving frames become the boolean mask.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            thresholded_pitch: torch.Tensor = cast(
                torch.Tensor,
                self._hysteresis(pitch.clone(), gated_periodicity)
            )
        voicing: torch.Tensor = ~torch.isnan(thresholded_pitch)
        return PitchFeatures(
            pitch=pitch.reshape(-1),
            periodicity=gated_periodicity.reshape(-1),
            voicing=voicing.reshape(-1)
        )

    @property
    def configuration(self) -> PitchConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration
