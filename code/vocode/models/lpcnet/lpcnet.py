# This module:
# 1. Implements the LPCNet vocoder for the Study 1 reproduction cohort: a
#    sample-level autoregressive model that predicts the mu-law excitation
#    of a linear-prediction filter, with the reference sparsification
#    schedule on the first recurrent core
# 2. Defines LpcnetConfig, the frozen record fixing this family's
#    Project-Trained Configuration: network dimensions, analysis geometry,
#    training noise, and the sparsifier, sampler, and metric protocols
#
# Study contract:
# - This family completes its Study 1 quality executions but contributes no
#   admitted Study 2 group, its deployment attempts being recorded in the
#   exclusion register as incomplete. The sparsifier and the weight clipper
#   are consequently reference training machinery whose deployment payoff,
#   the optimized sparse integer inference path, this project never executes
# - Its recorded timing is a single-execution partial observation and is
#   excluded from the strict speed frontier, so it is not comparable with
#   the deployment-lane real-time factors of the other families
#
# Harness contract (syntheticmind):
# - Subclasses the harness Module with automatic optimization disabled;
#   training_step drives the single optimizer manually and then applies
#   the reference post-step protocol in order: inverse-time learning-rate
#   decay, the sparsification schedule against the global step, and the
#   recurrent weight clip
# - predict_step and test_step return the synthesized_waveform mapping
#   produced by autoregressive copy-synthesis from the reference
#   waveform's own features
#
# Rate structure:
# - The architecture runs at two rates. The conditioning encoder runs once
#   per analysis frame, and the autoregressive core runs once per sample,
#   with one frame covering frame_size samples; the frame-rate conditioning
#   vector is repeated across the samples it governs. This split is what
#   makes the design tractable: the expensive convolutional encoding is paid
#   at frame rate while the per-sample path stays small enough to run one
#   step at a time
# - Training and synthesis differ in how the sample-rate path is driven.
#   Training is teacher-forced and evaluates every sample in one pass,
#   because the true previous samples are known in advance. Synthesis has
#   no such advance knowledge and runs a genuine sequential loop, one
#   sample at a time, because each input depends on the sample produced
#   immediately before it
#
# Design decisions:
# - Training is teacher-forced on int16-scaled samples with the
#   excitation cross-entropy objective of the reference recipe
# - The sampler resolves the network through a provider callable at
#   call time, so a transformed network (for example dynamic INT8) is
#   the one that actually synthesizes
# - Sparsification masks are part of training state; foreign weight
#   averaging would corrupt them, which is why EMA is excluded for this
#   family at the runner level
# - The module conditions on LPC analysis features rather than on a mel
#   spectrogram, so it exposes no conditioning mel protocol and no
#   mel-to-waveform synthesis entry point; it therefore stands outside the
#   Vocoder structural protocol the mel-conditioned families satisfy, and
#   the measurement stack reaches it through predict_step instead. The mel
#   protocol it does expose is for measurement only
#
# Author: Rahul Sawhney

from contextlib import nullcontext
from typing import ClassVar, cast, override

import torch
import torchaudio
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat, PositiveInt

from syntheticmind.core.module import Module
from syntheticmind.core.optimizer import OptimizationConfiguration
from syntheticmind.utilities.types import Batch, ModelOutput, StepOutput

from vocode.data.ljspeech_dataset import LJSpeechBatch, LJSpeechBatchValue
from vocode.losses.lpcnet import LpcnetLoss, LpcnetLossConfig
from vocode.metrics.pesq import PesqConfig
from vocode.metrics.stoi import StoiConfig
from vocode.models.lpcnet.network import LpcnetNetwork, LpcnetNetworkOutput
from vocode.models.lpcnet.sampling import LpcnetSampler, LpcnetSamplerConfig
from vocode.models.lpcnet.sparsification import LpcnetSparsifier, LpcnetSparsifierConfig, LpcnetWeightClipper
from vocode.transforms.lpc import LpcnetFeatureConfig, LpcnetFeatureExtractor, LpcnetFeatures
from vocode.transforms.mel import MelConfig, MelSpectrogram

__all__: list[str] = ["Lpcnet", "LpcnetConfig"]


class LpcnetConfig(BaseModel):
    # Frozen LPCNet configuration: network dimensions, LPC analysis order and
    # frame geometry, training noise, and the sparsifier, sampler, and metric
    # protocols. Explicit validated fields prevent experiment settings from
    # drifting between runs.
    #
    # Fields:
    #     sample_rate: Waveform rate in hertz. This family runs at the
    #         reference telephony-band rate rather than the 22.05 kHz used by
    #         the mel-conditioned families, which is why its measurement mel
    #         protocol is a separate record.
    #     feature_dimension: Width of the per-frame analysis feature vector
    #         the extractor produces, excluding the pitch embedding the
    #         encoder concatenates onto it.
    #     condition_dimension: Width of the frame-rate conditioning vector
    #         the encoder emits and both recurrent layers consume.
    #     embedding_dimension: Width of each signal embedding. Three
    #         embedded inputs are concatenated per sample, so the sample-rate
    #         path receives three times this width plus the conditioning
    #         width.
    #     first_gru_dimension: Hidden width of the first recurrent layer.
    #         This layer is the sparsification target, and its width must be
    #         divisible by both block dimensions of the sparsifier for the
    #         block reshape to be exact.
    #     second_gru_dimension: Hidden width of the second recurrent layer,
    #         deliberately far narrower than the first because it feeds the
    #         output stage directly.
    #     lpc_order: Number of linear-prediction taps. The predictor consumes
    #         this many past samples, so synthesis must carry a sample
    #         history of exactly this length.
    #     frame_size: Samples covered by one analysis frame, and therefore
    #         the factor by which the conditioning vector is repeated to
    #         reach sample rate.
    #     preemphasis_coefficient: First-order preemphasis coefficient
    #         applied before analysis and inverted after synthesis.
    #     training_noise_standard_deviation: Standard deviation of the
    #         Gaussian noise added to the teacher-forced mu-law inputs during
    #         training. Nonzero values teach the model to tolerate its own
    #         sampling errors, which teacher forcing alone would never
    #         expose it to; the field is non-negative so noise injection can
    #         be disabled outright.
    #     learning_rate: Initial Adam step size, and the numerator of the
    #         inverse-time decay applied after every step. Default:
    #         ``0.001``.
    #     adam_beta_1: First Adam moment decay. Default: ``0.5``.
    #     adam_beta_2: Second Adam moment decay. Default: ``0.8``.
    #     learning_rate_step_decay: Per-step coefficient of the inverse-time
    #         decay. Zero disables decay, leaving the rate at its initial
    #         value. Default: ``5e-05``.
    #     recurrent_clip_value: Bound of the pairwise recurrent weight
    #         constraint applied after every step. Default: ``0.992``.
    #     feature_configuration: LPC analysis settings.
    #         Default: ``LpcnetFeatureConfig()``.
    #     sparsifier_configuration: Block-sparsity schedule settings.
    #         Default: ``LpcnetSparsifierConfig()``.
    #     sampler_configuration: Autoregressive sampling settings.
    #         Default: ``LpcnetSamplerConfig()``.
    #     loss_configuration: Objective settings.
    #         Default: ``LpcnetLossConfig()``.
    #     metric_mel_protocol: Mel protocol used for measurement only. This
    #         family conditions on LPC features, so there is no
    #         corresponding conditioning protocol.
    #     pesq_protocol: PESQ measurement protocol for this family.
    #     stoi_protocol: STOI measurement protocol for this family.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    sample_rate: PositiveInt
    feature_dimension: PositiveInt
    condition_dimension: PositiveInt
    embedding_dimension: PositiveInt
    first_gru_dimension: PositiveInt
    second_gru_dimension: PositiveInt
    lpc_order: PositiveInt
    frame_size: PositiveInt
    preemphasis_coefficient: PositiveFloat
    training_noise_standard_deviation: NonNegativeFloat
    learning_rate: PositiveFloat = 0.001
    adam_beta_1: PositiveFloat = 0.5
    adam_beta_2: PositiveFloat = 0.8
    learning_rate_step_decay: NonNegativeFloat = 5e-5
    recurrent_clip_value: PositiveFloat = 0.992
    feature_configuration: LpcnetFeatureConfig = LpcnetFeatureConfig()
    sparsifier_configuration: LpcnetSparsifierConfig = LpcnetSparsifierConfig()
    sampler_configuration: LpcnetSamplerConfig = LpcnetSamplerConfig()
    loss_configuration: LpcnetLossConfig = LpcnetLossConfig()
    metric_mel_protocol: MelConfig
    pesq_protocol: PesqConfig
    stoi_protocol: StoiConfig

    @classmethod
    def xiph_reference(cls) -> LpcnetConfig:
        # Builds the LPCNet configuration aligned with the xiph reference
        # training recipe. The frame geometry is the defining choice: one
        # hundred sixty samples per frame at sixteen kilohertz places the
        # conditioning encoder at one hundred frames per second, so each
        # encoder evaluation is amortized over one hundred sixty sample-rate
        # steps.
        #
        # Returns:
        #     The frozen Project-Trained Configuration of this architecture,
        #     the record a Study 1 training trajectory is fitted under.
        return cls(
            sample_rate=16000,
            feature_dimension=20,
            condition_dimension=128,
            embedding_dimension=128,
            first_gru_dimension=384,
            second_gru_dimension=16,
            lpc_order=16,
            frame_size=160,
            preemphasis_coefficient=0.85,
            training_noise_standard_deviation=0.3,
            metric_mel_protocol=MelConfig.lpcnet_metric_16khz(),
            pesq_protocol=PesqConfig(),
            stoi_protocol=StoiConfig()
        )


class Lpcnet(Module):
    # LPCNet vocoder module: a frame-rate conditioning network feeding a
    # sample-rate autoregressive core that predicts the mu-law excitation of a
    # linear-prediction filter, under a scheduled sparsification of the first
    # recurrent layer. The module owns the network, the LPC feature extractor,
    # the objective, the autoregressive sampler, the sparsifier, and the
    # weight clipper; the last three are training-time machinery with no role
    # at inference.
    #
    # Integration: this family is the package's one non-mel-conditioned
    # architecture. It derives its own LPC analysis features from the
    # reference waveform, so it accepts a waveform rather than a spectrogram
    # and performs copy synthesis rather than mel-to-waveform synthesis. It
    # consequently does not satisfy the Vocoder structural protocol, whose
    # conditioning-protocol and synthesize members it does not expose; the
    # measurement stack reaches it through predict_step, and the mel protocol
    # it publishes exists only so quality metrics have a defined analysis
    # basis.
    #
    # Integration: automatic optimization is disabled not because several
    # optimizers exist, as in the adversarial families, but because the
    # reference recipe requires work after the optimizer step that the loop
    # provides no hook for. Three operations must run in a fixed order once
    # the step has completed, and expressing them inside training_step is what
    # guarantees that order.
    def __init__(self, configuration: LpcnetConfig) -> None:
        # Disables automatic optimization for the reference post-step
        # protocol and constructs the network, feature extractor, loss,
        # provider-backed sampler, sparsifier, and weight clipper. The sampler
        # receives a bound method rather than the network object, which is
        # what keeps it pointed at whatever network the module currently
        # holds.
        super().__init__()
        self.automatic_optimization: bool = False
        self._configuration: LpcnetConfig = configuration
        self._int16_scale: float = 32767.0
        self.network: LpcnetNetwork = LpcnetNetwork(
            feature_dimension=configuration.feature_dimension,
            condition_dimension=configuration.condition_dimension,
            embedding_dimension=configuration.embedding_dimension,
            first_gru_dimension=configuration.first_gru_dimension,
            second_gru_dimension=configuration.second_gru_dimension,
            lpc_order=configuration.lpc_order,
            frame_size=configuration.frame_size,
            training_noise_standard_deviation=configuration.training_noise_standard_deviation
        )
        self._feature_extractor: LpcnetFeatureExtractor = LpcnetFeatureExtractor(
            configuration.feature_configuration
        )
        self._loss: LpcnetLoss = LpcnetLoss(configuration.loss_configuration)
        self._sampler: LpcnetSampler = LpcnetSampler(
            self._resolve_current_network,
            configuration.sampler_configuration
        )
        self._sparsifier: LpcnetSparsifier = LpcnetSparsifier(configuration.sparsifier_configuration)
        self._weight_clipper: LpcnetWeightClipper = LpcnetWeightClipper(configuration.recurrent_clip_value)
        self._metric_mel_spectrogram: MelSpectrogram = MelSpectrogram(configuration.metric_mel_protocol)

    @override
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Executes copy synthesis: the reference waveform is analyzed into LPC
        # features and those features drive autoregressive resynthesis. The
        # input is a waveform rather than a conditioning spectrogram because
        # this family owns its own analysis front end, so the round trip
        # through analysis and synthesis is the only meaningful forward
        # operation it has.
        #
        # Args:
        #     waveform: Reference audio of shape ``[time]``,
        #         ``[batch, time]``, or ``[batch, 1, time]``.
        #
        # Returns:
        #     The resynthesized waveform of shape ``[batch, samples]``, with
        #     preemphasis inverted so it is directly comparable with the
        #     reference. Its length is the usable frame count times the frame
        #     size, which is at most the reference length.
        return self._synthesize_from_reference(waveform)

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Executes one teacher-forced training step followed by the reference
        # post-step protocol. Teacher forcing lets the whole sample-rate
        # sequence be evaluated in a single pass, because every input the core
        # needs is derived from the known target rather than from its own
        # output; this is what makes training tractable for a model that must
        # otherwise run one sample at a time.
        #
        # The three operations after the optimizer step are ordered and the
        # order is load-bearing. The learning rate is recomputed first, from
        # the base rate and the current global step, so it is a function of
        # the step count rather than an accumulated product and cannot drift
        # if a step is repeated. Sparsification runs second, because it must
        # observe the weights the optimizer has produced and re-zero whatever
        # the update regrew inside pruned blocks. The pairwise clip runs last,
        # so its bound holds on the weights that actually persist; running it
        # before sparsification would leave the final values unconstrained.
        #
        # Args:
        #     batch: Collated LJSpeech mapping; only the ``"waveform"`` entry
        #         is read, and the LPC features are derived from it.
        #     batch_idx: Index of this batch within the epoch. Unused, since
        #         every batch follows the same protocol; the schedules are
        #         driven by the global step instead.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping or carries no waveform
        #         tensor.
        #     ValueError: If the waveform does not normalize to
        #         ``[batch, time]``.
        #     RuntimeError: If the trainer materialized anything other than
        #         exactly one optimizer.
        #
        # Returns:
        #     A mapping whose ``"loss"`` entry is the detached excitation
        #     cross entropy. It is reporting-only: backward and the step have
        #     already run here, and the post-step protocol has already
        #     modified the weights.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        optimizer: torch.optim.Optimizer = self._require_single_optimizer()
        target_samples, features = self._prepare_training_inputs(batch)
        network_output: LpcnetNetworkOutput = self.network(
            target_samples,
            features.conditioning,
            features.pitch_index,
            features.lpc_coefficients
        )
        training_loss: torch.Tensor = self._loss.compute_loss(
            target_samples=target_samples,
            prediction=network_output.prediction,
            node_outputs=network_output.node_outputs
        )
        self.optimizer_zero_grad(optimizer)
        self.manual_backward(training_loss)
        self.optimizer_step(optimizer)
        # Post-step protocol, in the order the reference recipe requires.
        self._apply_inverse_time_learning_rate(optimizer)
        self._sparsifier.apply(self.network.first_gru, self.global_step)
        self._apply_weight_constraints()
        # Both keys carry the same value: this objective has a single term, so
        # the generic monitoring key and the named term coincide.
        self.log("train_loss", training_loss.detach())
        self.log("train_excitation_cross_entropy", training_loss.detach())
        return {"loss": training_loss.detach()}

    @override
    def validation_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Scores the same teacher-forced excitation cross entropy used in
        # training, without any weight update. Validation stays teacher-forced
        # rather than autoregressive because a sequential sampling pass over
        # full utterances would dominate epoch time; the tradeoff is that this
        # metric measures one-step prediction quality and not the accumulated
        # drift that free-running synthesis exhibits.
        #
        # Full utterances are processed in bounded frame chunks because the
        # fused recurrent kernel rejects the hundred-thousand-step sequences a
        # whole utterance would produce.
        #
        # Args:
        #     batch: Collated LJSpeech mapping carrying a waveform tensor.
        #     batch_idx: Index within the validation pass. Unused.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping or carries no waveform
        #         tensor.
        #
        # Returns:
        #     A mapping whose ``"loss"`` entry is the sample-weighted mean
        #     cross entropy across chunks.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        target_samples, features = self._prepare_training_inputs(batch)
        validation_loss: torch.Tensor = self._chunked_teacher_forced_loss(target_samples, features)
        self.log("val_loss", validation_loss)
        self.log("val_excitation_cross_entropy", validation_loss)
        return {"loss": validation_loss}

    @override
    def predict_step(self, batch: Batch, batch_idx: int) -> ModelOutput:
        # Runs autoregressive copy synthesis from the reference waveform's own
        # features and returns the mapping the measurement stack consumes.
        # This is the only path that exercises free-running generation, so it
        # is where accumulated autoregressive drift becomes observable and,
        # being genuinely sequential, it is far slower per utterance than any
        # other step in this module.
        #
        # Args:
        #     batch: Collated LJSpeech mapping carrying a waveform tensor.
        #     batch_idx: Index within the prediction pass. Unused.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping or its ``"waveform"``
        #         entry is not a tensor.
        #
        # Returns:
        #     A mapping under the single key ``"synthesized_waveform"``.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        waveform: LJSpeechBatchValue | None = batch.get("waveform")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("batch['waveform'] must be Tensor")
        synthesized_waveform: torch.Tensor = self._synthesize_from_reference(waveform)
        return {"synthesized_waveform": synthesized_waveform}

    @override
    def test_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Delegates to predict_step and revalidates its output, so the test
        # pass and the prediction pass synthesize through one code path.
        #
        # Raises:
        #     TypeError: If predict_step returns something other than a
        #         mapping carrying a ``"synthesized_waveform"`` tensor.
        prediction: ModelOutput = self.predict_step(batch, batch_idx)
        if not isinstance(prediction, dict):
            raise TypeError(f"predict_step must return a mapping, got {type(prediction).__name__}")
        synthesized_waveform: ModelOutput | None = prediction.get("synthesized_waveform")
        if not isinstance(synthesized_waveform, torch.Tensor):
            raise TypeError("predict_step output must contain Tensor key synthesized_waveform")
        return {"synthesized_waveform": synthesized_waveform}

    @override
    def configure_optimizers(self) -> OptimizationConfiguration:
        # Declares the single Adam optimizer used by the reference recipe. No
        # scheduler is declared, because the learning rate is driven by the
        # inverse-time rule applied inside training_step rather than by a
        # torch schedule; a scheduler would fight that rule for ownership of
        # the parameter group's rate.
        #
        # Returns:
        #     An OptimizationConfiguration holding one already-constructed
        #     optimizer over the whole network.
        optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self._configuration.learning_rate,
            betas=(self._configuration.adam_beta_1, self._configuration.adam_beta_2)
        )
        return OptimizationConfiguration(optimizer=optimizer)

    @property
    def configuration(self) -> LpcnetConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    @property
    def metric_mel_protocol(self) -> MelConfig:
        # Returns the mel protocol used for measurement. Unlike the
        # mel-conditioned families, this property has no conditioning
        # counterpart: the architecture never consumes a mel, and this record
        # exists purely to give the quality metrics a defined analysis basis
        # at this family's sampling rate.
        return self._configuration.metric_mel_protocol

    def _chunked_teacher_forced_loss(
        self,
        target_samples: torch.Tensor,
        features: LpcnetFeatures,
        frames_per_chunk: int = 150
    ) -> torch.Tensor:
        # Computes the excitation cross entropy over bounded frame chunks with
        # sample-count weighting, so long utterances stay inside the fused
        # recurrent kernel's sequence limits.
        #
        # Weighting by sample count rather than averaging the per-chunk losses
        # is what makes the result equal the loss over the whole utterance: a
        # trailing chunk is generally shorter than the others, and an
        # unweighted mean would over-represent it. Chunks are independent
        # forward passes with zero-initialized recurrent state, so a chunk
        # boundary discards the context the previous chunk accumulated; the
        # chunk length is chosen large enough that this affects a small
        # fraction of samples.
        #
        # Args:
        #     target_samples: Teacher-forced targets of shape
        #         ``[batch, samples, 1]`` on the int16 scale.
        #     features: Frame-rate LPC analysis features covering those
        #         samples.
        #     frames_per_chunk: Frames evaluated per forward pass.
        #         Default: ``150``.
        #
        # Returns:
        #     The sample-weighted mean cross entropy as a scalar tensor. The
        #     divisor is floored at one, so an empty utterance yields zero
        #     rather than a division by zero.
        frame_size: int = self._configuration.frame_size
        frame_count: int = features.conditioning.shape[1]
        weighted_loss: torch.Tensor = torch.zeros((), device=target_samples.device)
        total_samples: int = 0
        for chunk_start in range(0, frame_count, frames_per_chunk):
            chunk_end: int = min(chunk_start + frames_per_chunk, frame_count)
            chunk_targets: torch.Tensor = target_samples[
                :,
                chunk_start * frame_size: chunk_end * frame_size
            ]
            network_output: LpcnetNetworkOutput = self.network(
                chunk_targets,
                features.conditioning[:, chunk_start:chunk_end],
                features.pitch_index[:, chunk_start:chunk_end],
                features.lpc_coefficients[:, chunk_start:chunk_end]
            )
            chunk_loss: torch.Tensor = self._loss.compute_loss(
                target_samples=chunk_targets,
                prediction=network_output.prediction,
                node_outputs=network_output.node_outputs
            )
            chunk_samples: int = chunk_targets.shape[1]
            weighted_loss: torch.Tensor = weighted_loss + chunk_loss * float(chunk_samples)
            total_samples += chunk_samples
        return weighted_loss / float(max(total_samples, 1))

    def _prepare_training_inputs(self, batch: LJSpeechBatch) -> tuple[torch.Tensor, LpcnetFeatures]:
        # Prepares the int16-scaled preemphasized targets and frame features
        # for one batch. Three conversions happen here and each is required by
        # the reference recipe: preemphasis flattens the spectral tilt so the
        # linear predictor has less to explain, the int16 rescaling puts
        # samples on the domain the mu-law companding constants assume, and
        # feature extraction runs under no-grad in full precision because it
        # is a fixed analysis rather than a learned stage.
        #
        # The targets are finally truncated to a whole number of frames, since
        # a partial trailing frame has no conditioning vector to govern it.
        #
        # Returns:
        #     The targets of shape ``[batch, samples, 1]`` and the frame-rate
        #     features covering exactly those samples.
        waveform: torch.Tensor = self._prepare_reference_waveform(batch)
        preemphasized: torch.Tensor = self._preemphasize(waveform) * self._int16_scale
        with torch.no_grad():
            with self._full_precision_context():
                features: LpcnetFeatures = self._feature_extractor.extract(preemphasized.float())
        usable_samples: int = features.conditioning.shape[1] * self._configuration.frame_size
        target_samples: torch.Tensor = preemphasized[:, :usable_samples].unsqueeze(-1)
        return target_samples, features

    def _resolve_current_network(self) -> LpcnetNetwork:
        # Reads the currently registered network. Passing this method to the
        # sampler instead of the network object is what makes the indirection
        # work: an intervention that replaces the attribute, such as dynamic
        # integer quantization, is observed on the next synthesis call,
        # whereas a captured reference would keep the sampler executing the
        # pre-intervention network and silently invalidate any measurement of
        # that intervention.
        return cast(LpcnetNetwork, self.network)

    def _synthesize_from_reference(self, waveform: torch.Tensor) -> torch.Tensor:
        # Runs feature extraction and the autoregressive sampler for copy
        # synthesis, then undoes the analysis-side scaling and preemphasis so
        # the output is comparable with the reference waveform.
        #
        # The pitch correlation the sampler needs is read from the last
        # channel of the conditioning features, which is where the extractor
        # places it; the sampler uses it to sharpen its sampling distribution
        # on strongly periodic frames.
        prepared: torch.Tensor = self._prepare_waveform_shape(waveform)
        preemphasized: torch.Tensor = self._preemphasize(prepared) * self._int16_scale
        with self._full_precision_context():
            features: LpcnetFeatures = self._feature_extractor.extract(preemphasized.float())
            pitch_correlation: torch.Tensor = features.conditioning[..., -1]
            synthesized: torch.Tensor = self._sampler.synthesize(
                conditioning=features.conditioning,
                pitch_index=features.pitch_index,
                pitch_correlation=pitch_correlation,
                lpc_coefficients=features.lpc_coefficients
            )
        return self._deemphasize(synthesized / self._int16_scale)

    def _preemphasize(self, waveform: torch.Tensor) -> torch.Tensor:
        # Applies the reference first-order preemphasis filter along the
        # sample axis, attenuating low frequencies so the linear predictor
        # faces a flatter spectrum and its residual is closer to white. The
        # shift is zero-filled at the first position rather than wrapped, so
        # the filter never mixes the end of an utterance into its beginning.
        shifted: torch.Tensor = torch.roll(waveform, shifts=1, dims=-1)
        shifted: torch.Tensor = torch.cat([torch.zeros_like(shifted[..., :1]), shifted[..., 1:]], dim=-1)
        return waveform - self._configuration.preemphasis_coefficient * shifted

    def _deemphasize(self, waveform: torch.Tensor) -> torch.Tensor:
        # Inverts preemphasis on the synthesized waveform, restoring the
        # spectral tilt analysis removed. This is a recursive filter rather
        # than the one-tap difference of its forward counterpart, which is why
        # it is expressed through the audio filtering primitive instead of a
        # shift and subtract. Clamping is disabled so the reconstruction is
        # not silently limited before the caller has rescaled it.
        denominator: torch.Tensor = torch.tensor(
            [1.0, -self._configuration.preemphasis_coefficient],
            device=waveform.device,
            dtype=waveform.dtype
        )
        numerator: torch.Tensor = torch.tensor([1.0, 0.0], device=waveform.device, dtype=waveform.dtype)
        return torchaudio.functional.lfilter(waveform, denominator, numerator, clamp=False)

    def _apply_inverse_time_learning_rate(self, optimizer: torch.optim.Optimizer) -> None:
        # Applies the reference per-step inverse-time learning-rate decay. The
        # rate is recomputed from the configured base and the current global
        # step rather than scaled from its previous value, so it is a pure
        # function of the step count: resuming from a checkpoint restores the
        # correct rate without any scheduler state, and a repeated call at the
        # same step is idempotent.
        decayed_rate: float = self._configuration.learning_rate / (
            1.0 + self._configuration.learning_rate_step_decay * float(self.global_step)
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = decayed_rate

    def _apply_weight_constraints(self) -> None:
        # Applies the reference pairwise clip constraint to three kernels: the
        # first layer's recurrent weights, and both the recurrent and input
        # weights of the second. The selection follows the reference
        # deployment, where these are the matrices the quantized inference
        # path loads at reduced precision and whose paired magnitudes must
        # therefore stay inside a representable range. The first layer's input
        # weights are excluded because that path is not quantized the same
        # way.
        self._weight_clipper.apply(cast(torch.Tensor, self.network.first_gru.weight_hh_l0))
        self._weight_clipper.apply(cast(torch.Tensor, self.network.second_gru.weight_hh_l0))
        self._weight_clipper.apply(cast(torch.Tensor, self.network.second_gru.weight_ih_l0))

    def _full_precision_context(self) -> torch.autocast | nullcontext[None]:
        # Keeps feature extraction and synthesis in float32 regardless of
        # trainer precision. The LPC analysis chain performs an
        # autocorrelation and a recursive coefficient solve, both of which
        # lose accuracy badly at reduced precision and can yield an unstable
        # predictor; the excitation would then be computed against a filter
        # that does not match the one synthesis uses. Only the two device
        # types that support autocast are wrapped, and any other device falls
        # back to a null context, which is correct because autocast is not
        # active there in the first place.
        device_type: str = self.device.type
        if device_type in ("cuda", "cpu"):
            return torch.autocast(device_type=device_type, enabled=False)
        return nullcontext()

    def _prepare_reference_waveform(self, batch: LJSpeechBatch) -> torch.Tensor:
        # Reads the reference waveform and normalizes it to [batch, time].
        waveform: LJSpeechBatchValue | None = batch.get("waveform")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError(f"batch['waveform'] must be Tensor, got {type(waveform).__name__}")
        return self._prepare_waveform_shape(waveform)

    def _prepare_waveform_shape(self, waveform: torch.Tensor) -> torch.Tensor:
        # Normalizes a waveform to the [batch, time] layout the preemphasis
        # filter and the feature extractor consume, promoting an unbatched
        # sequence and dropping a singleton channel axis.
        if waveform.ndim == 1:
            return waveform.unsqueeze(0)
        if waveform.ndim == 2:
            return waveform
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            return waveform.squeeze(1)
        raise ValueError(f"Expected waveform shape [time], [batch, time], or [batch, 1, time], got {tuple(waveform.shape)}")

    def _compute_full_precision_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        # Computes the metric mel extraction inside the float32 island.
        with self._full_precision_context():
            return self._metric_mel_spectrogram(waveform.float())

    def _align_waveform_pair(
        self,
        first_waveform: torch.Tensor,
        second_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Truncates both waveforms to their common length along the time axis.
        # Synthesis covers only whole frames, so a resynthesized utterance is
        # generally shorter than its reference by the discarded partial frame.
        minimum_samples: int = min(first_waveform.shape[-1], second_waveform.shape[-1])
        return first_waveform[..., :minimum_samples], second_waveform[..., :minimum_samples]

    def _require_single_optimizer(self) -> torch.optim.Optimizer:
        # Returns the single optimizer this recipe requires, accepting both
        # forms the harness may hand back: a bare optimizer when one is
        # configured, or a one-element list. Anything else means the module
        # was fitted under a multi-optimizer setup that the post-step protocol
        # has no defined behavior for, so it is refused rather than silently
        # applied to whichever optimizer happens to come first.
        optimizers: list[torch.optim.Optimizer] | torch.optim.Optimizer | None = self.optimizers()
        if isinstance(optimizers, torch.optim.Optimizer):
            return optimizers
        if isinstance(optimizers, list) and len(optimizers) == 1:
            return optimizers[0]
        raise RuntimeError("LPCNet training requires exactly one optimizer")
