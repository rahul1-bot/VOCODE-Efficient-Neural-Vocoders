# This module:
# 1. Implements the RNDVoC vocoder for the Study 1 reproduction cohort:
#    range-null-space decomposed mel inversion, where the range-space
#    component is computed analytically from the mel and the network
#    predicts only the null-space refinement through its staged decoder
# 2. Trains adversarially with per-epoch exponential schedulers following
#    the reference recipe
#
# Harness contract (syntheticmind):
# - Subclasses the harness Module with automatic optimization disabled;
#   training_step predicts the decomposed components once and drives
#   both optimizer updates manually; the epoch-end hook advances the
#   schedulers
# - Validation computes the spectral validation loss and logs val_loss;
#   predict_step and test_step return the synthesized_waveform mapping;
#   checkpoint hooks stamp and verify the configuration dump
#
# Update semantics:
# - The two optimizers alternate by batch parity rather than both stepping
#   every batch: even batches update the discriminators and odd batches the
#   generator. Each half therefore sees half as many updates per epoch as it
#   would under a both-every-batch schedule, and the returned loss differs
#   in meaning between the two cases, carrying the discriminator total on
#   even batches and the generator total on odd ones
# - Because only one half updates per batch, the generic train_loss key is
#   published on generator batches alone; a monitor reading it observes a
#   value every second batch rather than every batch
#
# Design decisions:
# - The range-null decomposition is the architecture's defining idea:
#   the analytically recoverable spectral content never consumes model
#   capacity, so the network spends parameters only on what the mel
#   projection destroyed
# - The conditioning protocol is preserved in float32 under a
#   disabled-autocast context
# - Two distinct mel protocols are carried: one for conditioning, whose
#   geometry the network's pseudo-inverse is derived from, and one for
#   reconstruction scoring and measurement. They are separate records
#   because the decomposition binds the first to the network's internal
#   linear algebra, so it cannot be varied for measurement convenience
#
# Author: Rahul Sawhney

from contextlib import nullcontext
from typing import ClassVar, override

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from syntheticmind.core.module import Module
from syntheticmind.core.optimizer import OptimizationConfiguration
from syntheticmind.utilities.types import Batch, ModelOutput, StepOutput

from vocode.data.ljspeech_dataset import LJSpeechBatch, LJSpeechBatchValue
from vocode.losses.apnet2 import Apnet2Spectrum, Apnet2SpectrumAnalyzer
from vocode.losses.rndvoc import RndvocLoss, RndvocLossConfig
from vocode.metrics.pesq import PesqConfig
from vocode.metrics.stoi import StoiConfig
from vocode.models.rndvoc.discriminator import RndvocMultiPeriodDiscriminator, RndvocMultiResolutionDiscriminator
from vocode.models.rndvoc.network import RndvocGeneratorOutput, RndvocNetwork
from vocode.transforms.mel import MelConfig, MelSpectrogram

__all__: list[str] = ["Rndvoc", "RndvocConfig"]


class RndvocConfig(BaseModel):
    # Frozen RNDVoC configuration: the analysis geometry the decomposition is
    # derived from, the null-space decoder's depth and widths, adversarial
    # hyperparameters, and the two mel protocols this family uses.
    #
    # Fields:
    #     sample_rate: Waveform rate in hertz, used to construct the mel
    #         filterbank the decomposition inverts.
    #     input_mel_channels: Mel band count. This is what makes the mel map
    #         rank-deficient and therefore what creates the null space the
    #         network exists to predict.
    #     n_fft: Transform size of the spectral representation, fixing the
    #         bin count the decomposition operates on.
    #     hop_size: Hop of that transform, and therefore the number of
    #         waveform samples one spectral frame reconstructs to.
    #     win_size: Window length of that transform.
    #     fmin: Lowest frequency covered by the mel filterbank.
    #     fmax: Highest frequency covered. Content above it lies wholly in
    #         the null space, since no filter responds to it.
    #     null_stage_count: Number of refinement stages in the null-space
    #         decoder, the architecture's depth parameter.
    #     repeat_count: Residual blocks inside each stage's temporal module.
    #     input_dimension: Feature width carried through the decoder.
    #     squeeze_dimension: Reduced width the dense band mixing runs in.
    #     hidden_dimension: Expanded width inside each temporal block's
    #         bottleneck.
    #     kernel_size: Temporal extent of the depthwise convolutions.
    #     learning_rate: Shared AdamW step size for both optimizers.
    #         Default: ``0.0002``.
    #     adam_beta_1: First AdamW moment decay. Default: ``0.8``.
    #     adam_beta_2: Second AdamW moment decay. Default: ``0.99``.
    #     learning_rate_decay: Per-epoch multiplicative decay of both
    #         schedules. Default: ``0.999``.
    #     gradient_clip_norm: Maximum gradient norm applied separately to
    #         each parameter set, permissive enough to act as a divergence
    #         guard rather than a shaping constraint. Default: ``1000.0``.
    #     loss_configuration: Weights of the composite objective.
    #         Default: ``RndvocLossConfig()``.
    #     mel_protocol: Conditioning protocol. Its geometry must agree with
    #         the analysis fields above, because the network derives its
    #         pseudo-inverse from the same filterbank.
    #     reconstruction_mel_protocol: Protocol used for the reconstruction
    #         term and republished as the measurement protocol.
    #     pesq_protocol: PESQ measurement protocol for this family.
    #     stoi_protocol: STOI measurement protocol for this family.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    sample_rate: PositiveInt
    input_mel_channels: PositiveInt
    n_fft: PositiveInt
    hop_size: PositiveInt
    win_size: PositiveInt
    fmin: float
    fmax: PositiveFloat
    null_stage_count: PositiveInt
    repeat_count: PositiveInt
    input_dimension: PositiveInt
    squeeze_dimension: PositiveInt
    hidden_dimension: PositiveInt
    kernel_size: PositiveInt
    learning_rate: PositiveFloat = 0.0002
    adam_beta_1: PositiveFloat = 0.8
    adam_beta_2: PositiveFloat = 0.99
    learning_rate_decay: PositiveFloat = 0.999
    gradient_clip_norm: PositiveFloat = 1000.0
    loss_configuration: RndvocLossConfig = RndvocLossConfig()
    mel_protocol: MelConfig
    reconstruction_mel_protocol: MelConfig
    pesq_protocol: PesqConfig
    stoi_protocol: StoiConfig

    @classmethod
    def andong_22k(cls) -> RndvocConfig:
        # Builds the RNDVoC configuration aligned with the Andong-Li 22.05 kHz
        # LJSpeech recipe. The analysis geometry is the same
        # eighty-band, two-hundred-fifty-six-hop protocol the mel-conditioned
        # GAN families in this package share, which is what makes their
        # results directly comparable; what differs is entirely how the mel is
        # inverted.
        #
        # Returns:
        #     The frozen Project-Trained Configuration of this architecture,
        #     the record a Study 1 training trajectory is fitted under.
        return cls(
            sample_rate=22050,
            input_mel_channels=80,
            n_fft=1024,
            hop_size=256,
            win_size=1024,
            fmin=0.0,
            fmax=8000.0,
            null_stage_count=6,
            repeat_count=2,
            input_dimension=256,
            squeeze_dimension=64,
            hidden_dimension=256,
            kernel_size=7,
            mel_protocol=MelConfig.rndvoc_andong(),
            reconstruction_mel_protocol=MelConfig.rndvoc_andong_reconstruction(),
            pesq_protocol=PesqConfig(),
            stoi_protocol=StoiConfig()
        )


class Rndvoc(Module):
    # RNDVoC vocoder module: mel inversion by range-null space decomposition,
    # trained adversarially against a period ensemble and a multi-resolution
    # ensemble under manual optimization. The module owns both mel transforms,
    # the spectrum analyzer the objective scores against, the generator
    # network, both discriminator ensembles, and the composite objective.
    #
    # Integration: what distinguishes this family from the other adversarial
    # vocoders is not its training loop but its generator. The range component
    # of the output spectrum is computed by pseudo-inverting the mel basis and
    # is therefore correct regardless of training state; the network and the
    # adversarial objective act only on the null-space component. A
    # consequence worth stating plainly is that adversarial pressure here
    # shapes strictly less of the output than in a family whose generator
    # produces the whole spectrum.
    #
    # Integration: the module satisfies the vocode.models.vocoder.Vocoder
    # structural protocol through its network property, its mel_protocol and
    # metric_mel_protocol properties, and synthesize. Unlike the other
    # families, those two protocols are different records: conditioning is
    # bound to the decomposition's linear algebra, while measurement uses the
    # reconstruction protocol.
    def __init__(self, configuration: RndvocConfig) -> None:
        # Disables automatic optimization, because the adversarial recipe
        # drives two optimizers from inside training_step and alternates
        # between them by batch parity. Constructs both mel transforms, the
        # spectrum analyzer, the generator, both discriminator ensembles, and
        # the objective; the mel filterbank the decomposition rests on is
        # built inside the network from the analysis fields.
        super().__init__()
        self.automatic_optimization: bool = False
        self._configuration: RndvocConfig = configuration
        self._mel_spectrogram: MelSpectrogram = MelSpectrogram(configuration.mel_protocol)
        self._reconstruction_mel_spectrogram: MelSpectrogram = MelSpectrogram(configuration.reconstruction_mel_protocol)
        self._spectrum_analyzer: Apnet2SpectrumAnalyzer = Apnet2SpectrumAnalyzer(
            n_fft=configuration.n_fft,
            hop_size=configuration.hop_size,
            win_size=configuration.win_size
        )
        self.network: RndvocNetwork = RndvocNetwork(
            sample_rate=configuration.sample_rate,
            num_mels=configuration.input_mel_channels,
            n_fft=configuration.n_fft,
            hop_size=configuration.hop_size,
            win_size=configuration.win_size,
            fmin=configuration.fmin,
            fmax=configuration.fmax,
            null_stage_count=configuration.null_stage_count,
            repeat_count=configuration.repeat_count,
            input_dimension=configuration.input_dimension,
            squeeze_dimension=configuration.squeeze_dimension,
            hidden_dimension=configuration.hidden_dimension,
            kernel_size=configuration.kernel_size
        )
        self._period_discriminator: RndvocMultiPeriodDiscriminator = RndvocMultiPeriodDiscriminator()
        self._resolution_discriminator: RndvocMultiResolutionDiscriminator = RndvocMultiResolutionDiscriminator()
        self._loss: RndvocLoss = RndvocLoss(configuration.loss_configuration)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Synthesizes a waveform batch from conditioning mels through the
        # generator network.
        return self.network(mel)

    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        # Generates waveform output from the architecture-specific acoustic
        # representation. This is the Vocoder protocol entry point used by the
        # measurement and profiling stacks; it delegates to forward so the
        # protocol surface and the module surface cannot diverge.
        return self.forward(mel)

    @override
    def on_train_epoch_end(self) -> None:
        # Advances both exponential schedules at the epoch boundary. As in the
        # other families that hand back pre-built torch schedulers, the
        # trainer files a null scheduler configuration for each and never
        # advances them, so this stepping is the model's responsibility. Both
        # the unwrapped and the list form of lr_schedulers() are handled.
        schedulers: list[torch.optim.lr_scheduler.LRScheduler] | torch.optim.lr_scheduler.LRScheduler | None = (
            self.lr_schedulers()
        )
        if isinstance(schedulers, list):
            for scheduler in schedulers:
                scheduler.step()
        elif schedulers is not None:
            schedulers.step()

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Executes one half of the adversarial recipe, selected by batch
        # parity: even batches update the discriminators and odd batches the
        # generator. This differs from the both-every-batch schedule the other
        # adversarial families in this package use, and the difference is
        # substantive rather than cosmetic. Each half receives half as many
        # updates per epoch, so the adversarial balance and the effective
        # learning-rate schedule are not comparable step-for-step with those
        # families.
        #
        # The generator runs forward once before the branch, so the same
        # prediction serves whichever half executes; on discriminator batches
        # that forward pass is used only for its detached synthesis and its
        # graph is discarded.
        #
        # Args:
        #     batch: Collated LJSpeech mapping; only the ``"waveform"`` entry
        #         is read, and the conditioning mel is derived from it.
        #     batch_idx: Index of this batch within the epoch. Unlike the
        #         other families, this argument is load-bearing: its parity
        #         selects which optimizer steps.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping or carries no waveform
        #         tensor.
        #     ValueError: If the waveform does not normalize to
        #         ``[batch, time]``.
        #     RuntimeError: If the trainer did not materialize exactly the two
        #         optimizers this recipe requires.
        #
        # Returns:
        #     A mapping whose ``"loss"`` entry is the detached total of
        #     whichever half ran, so its meaning alternates with batch parity.
        #     It is reporting-only, since backward and the step have already
        #     executed here.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        generator_optimizer: torch.optim.Optimizer
        discriminator_optimizer: torch.optim.Optimizer
        generator_optimizer, discriminator_optimizer = self._require_gan_optimizers()
        reference_waveform: torch.Tensor = self._prepare_reference_waveform(batch)
        reference_mel: torch.Tensor = self._prepare_reference_mel(reference_waveform)
        generator_output: RndvocGeneratorOutput = self.network.predict_components(reference_mel)
        real_waveform: torch.Tensor
        fake_waveform: torch.Tensor
        real_waveform, fake_waveform = self._align_waveform_pair(
            reference_waveform,
            generator_output.waveform
        )
        component_name: str
        component_value: float
        # Even batches: critics only. The generic train_loss key is
        # deliberately not published here, so a monitor watching it sees the
        # generator's progress rather than an alternating pair of
        # incomparable quantities.
        if batch_idx % 2 == 0:
            discriminator_loss: torch.Tensor
            discriminator_components: dict[str, float]
            discriminator_loss, discriminator_components = self._run_discriminator_training_step(
                discriminator_optimizer=discriminator_optimizer,
                real_waveform=real_waveform,
                fake_waveform=fake_waveform
            )
            self.log("train_discriminator_loss", discriminator_loss.detach())
            for component_name, component_value in discriminator_components.items():
                self.log(f"train_{component_name}", component_value)
            return {"loss": discriminator_loss.detach()}
        # Odd batches: generator only, scored against critics that were last
        # updated on the previous batch.
        generator_loss: torch.Tensor
        generator_components: dict[str, float]
        generator_loss, generator_components = self._run_generator_training_step(
            generator_optimizer=generator_optimizer,
            reference_waveform=reference_waveform,
            generator_output=generator_output,
            real_waveform=real_waveform,
            fake_waveform=fake_waveform
        )
        self.log("train_loss", generator_loss.detach())
        self.log("train_generator_loss", generator_loss.detach())
        for component_name, component_value in generator_components.items():
            self.log(f"train_{component_name}", component_value)
        return {"loss": generator_loss.detach()}

    @override
    def validation_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Scores the spectral validation objective over the aligned reference
        # and candidate, published as val_loss for checkpoint selection. The
        # adversarial terms are excluded, because critic scores drift as the
        # critics train and are not comparable across epochs; the spectral
        # terms are stable quality proxies.
        #
        # Two views of the candidate are supplied. The generator's own
        # predicted spectrum is scored directly, and the waveform it
        # reconstructs to is re-analyzed, so the objective can penalize both
        # the spectrum the network intended and the one that survives the
        # inverse transform. Those differ whenever the predicted magnitude and
        # phase are not consistent with any real signal.
        #
        # Args:
        #     batch: Collated LJSpeech mapping carrying a waveform tensor.
        #     batch_idx: Index within the validation pass. Unused, since no
        #         alternation applies outside training.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping or carries no waveform
        #         tensor.
        #
        # Returns:
        #     A mapping whose ``"loss"`` entry is the composite spectral
        #     validation total.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        reference_waveform: torch.Tensor = self._prepare_reference_waveform(batch)
        reference_mel: torch.Tensor = self._prepare_reference_mel(reference_waveform)
        generator_output: RndvocGeneratorOutput = self.network.predict_components(reference_mel)
        aligned_reference: torch.Tensor
        aligned_candidate: torch.Tensor
        aligned_reference, aligned_candidate = self._align_waveform_pair(
            reference_waveform,
            generator_output.waveform
        )
        reference_spectrum: Apnet2Spectrum = self._analyze_full_precision_spectrum(aligned_reference)
        final_spectrum: Apnet2Spectrum = self._analyze_full_precision_spectrum(aligned_candidate)
        reference_reconstruction_mel: torch.Tensor = self._prepare_reference_reconstruction_mel(aligned_reference)
        candidate_reconstruction_mel: torch.Tensor = self._prepare_network_mel(
            self._compute_full_precision_mel(self._reconstruction_mel_spectrogram, aligned_candidate)
        )
        validation_loss: torch.Tensor
        components: dict[str, float]
        validation_loss, components = self._loss.compute_validation_loss(
            reference_spectrum=reference_spectrum,
            candidate_log_amplitude=generator_output.log_amplitude,
            candidate_phase=generator_output.phase,
            candidate_real_spectrum=generator_output.real_spectrum,
            candidate_imaginary_spectrum=generator_output.imaginary_spectrum,
            final_spectrum=final_spectrum,
            reference_mel=reference_reconstruction_mel,
            candidate_mel=candidate_reconstruction_mel
        )
        self.log("val_loss", validation_loss)
        component_name: str
        component_value: float
        for component_name, component_value in components.items():
            self.log(f"val_{component_name}", component_value)
        return {"loss": validation_loss}

    @override
    def predict_step(self, batch: Batch, batch_idx: int) -> ModelOutput:
        # Synthesizes from the conditioning mel derived from the batch
        # waveform and returns the mapping the measurement stack consumes. No
        # channel axis is squeezed away here, unlike the families whose
        # inverse transform introduces one, because this network's
        # reconstruction is already ``[batch, samples]``.
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
        reference_mel: torch.Tensor = self._prepare_reference_mel(waveform)
        synthesized_waveform: torch.Tensor = self.network(reference_mel)
        return {"synthesized_waveform": synthesized_waveform}

    @override
    def test_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Delegates to predict_step and validates the output contract, so
        # test evaluation measures exactly the prediction path.
        prediction: ModelOutput = self.predict_step(batch, batch_idx)
        if not isinstance(prediction, dict):
            raise TypeError(f"predict_step must return a mapping, got {type(prediction).__name__}")
        synthesized_waveform: ModelOutput | None = prediction.get("synthesized_waveform")
        if not isinstance(synthesized_waveform, torch.Tensor):
            raise TypeError("predict_step output must contain Tensor key synthesized_waveform")
        return {"synthesized_waveform": synthesized_waveform}

    @override
    def configure_optimizers(self) -> OptimizationConfiguration:
        # Declares the adversarial optimizer pair in the order training_step
        # unpacks it: the generator optimizer first, the discriminator
        # optimizer second. The discriminator optimizer covers the
        # concatenated parameters of both ensembles, so one step updates both
        # critics together.
        #
        # Returns:
        #     An OptimizationConfiguration holding already-constructed torch
        #     objects rather than harness configuration records, which is why
        #     on_train_epoch_end performs the per-epoch decay itself.
        generator_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self._configuration.learning_rate,
            betas=(self._configuration.adam_beta_1, self._configuration.adam_beta_2)
        )
        discriminator_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            list(self._resolution_discriminator.parameters()) + list(self._period_discriminator.parameters()),
            lr=self._configuration.learning_rate,
            betas=(self._configuration.adam_beta_1, self._configuration.adam_beta_2)
        )
        generator_scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.ExponentialLR(
            generator_optimizer,
            gamma=self._configuration.learning_rate_decay
        )
        discriminator_scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.ExponentialLR(
            discriminator_optimizer,
            gamma=self._configuration.learning_rate_decay
        )
        return OptimizationConfiguration(
            optimizer=[generator_optimizer, discriminator_optimizer],
            scheduler=[generator_scheduler, discriminator_scheduler]
        )

    @property
    def mel_protocol(self) -> MelConfig:
        # Returns the mel-spectrogram protocol required by this architecture family.
        return self._configuration.mel_protocol

    @property
    def metric_mel_protocol(self) -> MelConfig:
        # Returns the reconstruction protocol for measurement. This family is
        # the case where the conditioning and measurement protocols genuinely
        # differ: the conditioning protocol is fixed by the filterbank the
        # decomposition inverts and cannot be varied for measurement, so a
        # second record carries the scoring geometry.
        return self._configuration.reconstruction_mel_protocol

    @property
    def configuration(self) -> RndvocConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    def _run_discriminator_training_step(
        self,
        discriminator_optimizer: torch.optim.Optimizer,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Discriminator update. Both ensembles score the reference against the
        # detached synthesis, so no gradient reaches the generator; the toggle
        # restricts requires_grad to the critic parameters for the duration of
        # the block and restores the prior flags on exit, even if the body
        # raises.
        #
        # Returns:
        #     The detached discriminator total and its scalar component
        #     breakdown.
        with self.toggled_optimizer(discriminator_optimizer):
            real_period_logits: list[torch.Tensor]
            fake_period_logits: list[torch.Tensor]
            real_period_logits, fake_period_logits, _, _ = self._period_discriminator(
                real_waveform,
                fake_waveform.detach()
            )
            real_resolution_logits: list[torch.Tensor]
            fake_resolution_logits: list[torch.Tensor]
            real_resolution_logits, fake_resolution_logits, _, _ = self._resolution_discriminator(
                real_waveform,
                fake_waveform.detach()
            )
            discriminator_loss: torch.Tensor
            components: dict[str, float]
            discriminator_loss, components = self._loss.compute_discriminator_loss(
                real_period_logits=real_period_logits,
                fake_period_logits=fake_period_logits,
                real_resolution_logits=real_resolution_logits,
                fake_resolution_logits=fake_resolution_logits
            )
            self.optimizer_zero_grad(discriminator_optimizer)
            self.manual_backward(discriminator_loss)
            torch.nn.utils.clip_grad_norm_(
                list(self._period_discriminator.parameters()) + list(self._resolution_discriminator.parameters()),
                self._configuration.gradient_clip_norm
            )
            self.optimizer_step(discriminator_optimizer)
        return discriminator_loss.detach(), components

    def _run_generator_training_step(
        self,
        generator_optimizer: torch.optim.Optimizer,
        reference_waveform: torch.Tensor,
        generator_output: RndvocGeneratorOutput,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Generator update. The ensembles are re-run on the graph-carrying
        # synthesis, which is required rather than wasteful, since the
        # discriminator branch of a previous batch consumed a detached copy.
        # Only the synthesis logits are taken from each ensemble, because the
        # generator objective scores how convincing its output is and never
        # needs the reference verdicts; the reference features are still
        # needed, for the feature-matching term. Clipping covers the whole
        # network, including the buffers of the decomposition, which carry no
        # gradient and are therefore unaffected.
        #
        # Returns:
        #     The detached generator total and its scalar component
        #     breakdown.
        with self.toggled_optimizer(generator_optimizer):
            reference_spectrum: Apnet2Spectrum = self._analyze_full_precision_spectrum(real_waveform)
            final_spectrum: Apnet2Spectrum = self._analyze_full_precision_spectrum(fake_waveform)
            reference_reconstruction_mel: torch.Tensor = self._prepare_reference_reconstruction_mel(real_waveform)
            candidate_reconstruction_mel: torch.Tensor = self._prepare_network_mel(
                self._compute_full_precision_mel(self._reconstruction_mel_spectrogram, fake_waveform)
            )
            fake_period_logits: list[torch.Tensor]
            real_period_features: list[list[torch.Tensor]]
            fake_period_features: list[list[torch.Tensor]]
            _, fake_period_logits, real_period_features, fake_period_features = self._period_discriminator(
                real_waveform,
                fake_waveform
            )
            fake_resolution_logits: list[torch.Tensor]
            real_resolution_features: list[list[torch.Tensor]]
            fake_resolution_features: list[list[torch.Tensor]]
            _, fake_resolution_logits, real_resolution_features, fake_resolution_features = (
                self._resolution_discriminator(real_waveform, fake_waveform)
            )
            generator_loss: torch.Tensor
            components: dict[str, float]
            generator_loss, components = self._loss.compute_generator_loss(
                reference_spectrum=reference_spectrum,
                candidate_log_amplitude=generator_output.log_amplitude,
                candidate_phase=generator_output.phase,
                candidate_real_spectrum=generator_output.real_spectrum,
                candidate_imaginary_spectrum=generator_output.imaginary_spectrum,
                final_spectrum=final_spectrum,
                reference_mel=reference_reconstruction_mel,
                candidate_mel=candidate_reconstruction_mel,
                fake_period_logits=fake_period_logits,
                fake_resolution_logits=fake_resolution_logits,
                real_period_features=real_period_features,
                fake_period_features=fake_period_features,
                real_resolution_features=real_resolution_features,
                fake_resolution_features=fake_resolution_features
            )
            self.optimizer_zero_grad(generator_optimizer)
            self.manual_backward(generator_loss)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self._configuration.gradient_clip_norm)
            self.optimizer_step(generator_optimizer)
        return generator_loss.detach(), components

    def _prepare_reference_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        # Produces the batched full-precision conditioning mel. This is the
        # protocol the network's pseudo-inverse was derived from, so it is the
        # only mel the decomposition is valid against.
        mel: torch.Tensor = self._compute_full_precision_mel(self._mel_spectrogram, waveform)
        return self._prepare_network_mel(mel)

    def _prepare_reference_reconstruction_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        # Produces the batched full-precision mel under the reconstruction
        # protocol, used for the objective's reconstruction term and for
        # measurement rather than for conditioning.
        mel: torch.Tensor = self._compute_full_precision_mel(self._reconstruction_mel_spectrogram, waveform)
        return self._prepare_network_mel(mel)

    def _compute_full_precision_mel(self, transform: MelSpectrogram, waveform: torch.Tensor) -> torch.Tensor:
        # Computes one mel extraction inside the float32 island regardless of trainer precision.
        with self._full_precision_context():
            return transform(self._prepare_waveform_for_mel(waveform.float()))

    def _analyze_full_precision_spectrum(self, waveform: torch.Tensor) -> Apnet2Spectrum:
        # Keeps the STFT target extraction in float32 so mixed precision cannot alter the spectra.
        with self._full_precision_context():
            return self._spectrum_analyzer.analyze(waveform.float())

    def _full_precision_context(self) -> torch.autocast | nullcontext[None]:
        # Keeps spectral extraction in float32 so mixed-precision training
        # cannot alter the protocol used by the reference recipe. Precision
        # matters more here than in a purely learned pipeline: the conditioning
        # mel is the input the pseudo-inverse is applied to, so rounding it
        # perturbs the analytic component the network is not permitted to
        # correct. Only the two device types that support autocast are
        # wrapped; any other device falls back to a null context, which is
        # correct because autocast is not active there.
        device_type: str = self.device.type
        if device_type in ("cuda", "cpu"):
            return torch.autocast(device_type=device_type, enabled=False)
        return nullcontext()

    def _prepare_reference_waveform(self, batch: LJSpeechBatch) -> torch.Tensor:
        # Reads the reference waveform and normalizes it to [batch, time],
        # promoting an unbatched sequence, dropping a singleton channel axis,
        # and rejecting anything that does not reduce to rank two.
        waveform: torch.Tensor = self._extract_waveform(batch)
        if waveform.ndim == 1:
            waveform: torch.Tensor = waveform.unsqueeze(0)
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform: torch.Tensor = waveform.squeeze(1)
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform shape [batch, time], got {tuple(waveform.shape)}")
        return waveform

    def _prepare_waveform_for_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        # Reduces a waveform to the [batch, time] layout the mel transforms
        # require, which is what lets the generator's own output be fed
        # straight back into mel analysis for the reconstruction term.
        if waveform.ndim == 1:
            return waveform.unsqueeze(0)
        if waveform.ndim == 2:
            return waveform
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            return waveform.squeeze(1)
        raise ValueError(f"Expected waveform shape [time], [batch, time], or [batch, 1, time], got {tuple(waveform.shape)}")

    def _prepare_network_mel(self, mel: torch.Tensor) -> torch.Tensor:
        # Normalizes mel layouts to the [batch, bands, frames] shape the
        # network consumes.
        if mel.ndim == 2:
            return mel.unsqueeze(0)
        if mel.ndim == 3:
            return mel
        if mel.ndim == 4 and mel.shape[1] == 1:
            return mel.squeeze(1)
        raise ValueError(f"Expected mel shape [channels, frames], [batch, channels, frames], or [batch, 1, channels, frames], got {tuple(mel.shape)}")

    def _align_waveform_pair(
        self,
        first_waveform: torch.Tensor,
        second_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Truncates both waveforms to their common length along the time axis.
        # The inverse transform reconstructs a sample count determined by the
        # frame count and hop, which need not equal the reference length, so
        # the critics and the spectral terms would otherwise be handed
        # mismatched inputs.
        minimum_samples: int = min(first_waveform.shape[-1], second_waveform.shape[-1])
        return first_waveform[..., :minimum_samples], second_waveform[..., :minimum_samples]

    def _require_gan_optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        # Returns the generator and discriminator optimizers in declaration
        # order, failing loudly when the pair is absent. The list form is
        # asserted explicitly because the harness unwraps a single-optimizer
        # configuration to a bare object; receiving anything else means the
        # module was fitted under a setup the alternating schedule cannot
        # execute.
        optimizers: list[torch.optim.Optimizer] | torch.optim.Optimizer = self.optimizers()
        if not isinstance(optimizers, list) or len(optimizers) != 2:
            raise RuntimeError("RNDVoC training requires generator and discriminator optimizers")
        return optimizers[0], optimizers[1]

    def _extract_waveform(self, batch: LJSpeechBatch) -> torch.Tensor:
        # Reads the waveform tensor from the collated batch mapping.
        waveform: LJSpeechBatchValue | None = batch.get("waveform")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError(f"batch['waveform'] must be Tensor, got {type(waveform).__name__}")
        return waveform
