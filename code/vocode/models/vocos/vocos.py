# This module:
# 1. Implements the Vocos vocoder for the Study 1 reproduction cohort,
#    following the charactr 24 kHz recipe: a frame-level ConvNeXt
#    backbone predicting Fourier coefficients that an inverse STFT head
#    converts to the waveform in one pass
# 2. Trains adversarially against the multi-period and multi-resolution
#    discriminators with cosine learning-rate schedules stepped per
#    training step
#
# Harness contract (syntheticmind):
# - Subclasses the harness Module with automatic optimization disabled;
#   training_step synthesizes once, drives the discriminator update on
#   the detached synthesis and the generator update on the shared
#   synthesis, and steps both cosine schedulers explicitly
# - Validation logs val_loss for checkpoint monitoring; predict_step and
#   test_step return the synthesized_waveform mapping; checkpoint hooks
#   stamp and verify the configuration dump
#
# Design decisions:
# - The batch waveform is resampled to the protocol's 24 kHz rate when
#   the corpus rate differs, because the published recipe operates at
#   24 kHz with 100 mel bands
# - Mel extraction stays in float32 under a disabled-autocast context;
#   generator gradients are norm-clipped per the reference recipe
#
# Author: Rahul Sawhney

from contextlib import nullcontext
from typing import ClassVar, override

import torch
import torchaudio
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from syntheticmind.core.module import Module
from syntheticmind.core.optimizer import OptimizationConfiguration, SchedulerConfig
from syntheticmind.utilities.types import Batch, ModelOutput, StepOutput

from vocode.data.ljspeech_dataset import LJSpeechBatch, LJSpeechBatchValue
from vocode.losses.vocos import VocosLoss, VocosLossConfig
from vocode.metrics.pesq import PesqConfig
from vocode.metrics.stoi import StoiConfig
from vocode.models.vocos.discriminator import VocosMultiPeriodDiscriminator, VocosMultiResolutionDiscriminator
from vocode.models.vocos.network import VocosNetwork, VocosNetworkConfig
from vocode.transforms.mel import MelConfig, MelSpectrogram

__all__: list[str] = ["Vocos", "VocosConfig"]


class VocosConfig(BaseModel):
    # Frozen Vocos configuration: ConvNeXt backbone topology, ISTFT head
    # grid, adversarial hyperparameters, and the family's mel and metric
    # protocols.
    #
    # Fields:
    #     input_channels: Mel band count the backbone's embedding
    #         convolution consumes. The published 24 kHz recipe conditions
    #         on a hundred bands rather than the eighty the HiFi-GAN
    #         family uses. Default: ``100``.
    #     hidden_dimension: Working width of the ConvNeXt stack, also the
    #         input width of the ISTFT head's projection.
    #         Default: ``512``.
    #     intermediate_dimension: Expanded width inside each ConvNeXt
    #         block's pointwise pair. Default: ``1536``.
    #     layer_count: Number of ConvNeXt blocks. It also sets the default
    #         layer-scale initialization, which is its reciprocal.
    #         Default: ``8``.
    #     n_fft: Transform size of the inverse-STFT head. The head
    #         projects to two more than this many channels, split evenly
    #         into magnitude and phase. Default: ``1024``.
    #     hop_length: Hop of the inverse-STFT head. Since the backbone
    #         never upsamples in time, this single value is the entire
    #         sample-rate expansion of the architecture.
    #         Default: ``256``.
    #     learning_rate: Shared AdamW step size for the generator and
    #         discriminator optimizers. Default: ``0.0005``.
    #     adam_beta_1: First AdamW moment decay for both optimizers.
    #         Default: ``0.8``.
    #     adam_beta_2: Second AdamW moment decay for both optimizers.
    #         Default: ``0.9``.
    #     gradient_clip_norm: Maximum gradient norm applied separately to
    #         the generator and to the discriminator parameter sets before
    #         each step. The reference value is deliberately permissive
    #         and acts as a divergence guard rather than a shaping
    #         constraint. Default: ``1000.0``.
    #     scheduler_total_steps: Horizon of the cosine schedules, in
    #         optimizer steps. Both schedules decay to their floor over
    #         exactly this many steps, so shortening a run without
    #         adjusting this value leaves the rate mid-descent.
    #         Default: ``249600``.
    #     loss_configuration: Weights of the mel reconstruction term and
    #         of the multi-resolution discriminator contribution.
    #         Default: ``VocosLossConfig()``.
    #     mel_protocol: The single mel protocol this family both
    #         conditions on and measures with.
    #         Default: ``MelConfig.vocos_charactr_mel_24khz()``.
    #     pesq_protocol: PESQ measurement protocol for this family.
    #         Default: ``PesqConfig()``.
    #     stoi_protocol: STOI measurement protocol for this family.
    #         Default: ``StoiConfig()``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    input_channels: PositiveInt = 100
    hidden_dimension: PositiveInt = 512
    intermediate_dimension: PositiveInt = 1536
    layer_count: PositiveInt = 8
    n_fft: PositiveInt = 1024
    hop_length: PositiveInt = 256
    learning_rate: PositiveFloat = 0.0005
    adam_beta_1: PositiveFloat = 0.8
    adam_beta_2: PositiveFloat = 0.9
    gradient_clip_norm: PositiveFloat = 1000.0
    scheduler_total_steps: PositiveInt = 249600
    loss_configuration: VocosLossConfig = VocosLossConfig()
    mel_protocol: MelConfig = MelConfig.vocos_charactr_mel_24khz()
    pesq_protocol: PesqConfig = PesqConfig()
    stoi_protocol: StoiConfig = StoiConfig()

    @classmethod
    def charactr_mel_24khz(cls) -> VocosConfig:
        # Builds the Vocos 24 kHz mel configuration aligned with the charactr reference model.
        # Every field of this family already defaults to its published
        # value, so the factory constructs the record unmodified; it
        # exists as the named entry point the registry routes through, so
        # the reference recipe is reached by name rather than by relying
        # on defaults at the call site.
        #
        # Returns:
        #     The frozen reference configuration for this architecture.
        return cls()


class Vocos(Module):
    # Vocos vocoder module: ConvNeXt frame backbone with an inverse-STFT
    # synthesis head, trained adversarially against the multi-period and
    # multi-resolution discriminators under manual optimization.
    #
    # Integration: the module satisfies the vocode.models.vocoder.Vocoder
    # synthesis contract through its network member, its two mel-protocol
    # properties, and synthesize; both protocol properties return the same
    # record, because this family conditions and measures on one grid.
    #
    # This class is also a subclassing surface: VocosFormer extends it and
    # replaces only the network member after construction, inheriting the
    # discriminators, loss, optimization declaration, steps, and metric
    # surface unchanged. Anything added here that a controlled comparison
    # must hold fixed therefore reaches that row automatically, and
    # anything specific to the convolutional backbone belongs in the
    # network module rather than here.
    #
    # Two behaviors distinguish this family from the HiFi-GAN recipe. The
    # cosine schedules advance per optimizer step rather than per epoch,
    # so this module steps them inside the training step instead of at
    # epoch end. Gradient norms are clipped inside each update by calling
    # torch directly rather than through the harness clipping surface,
    # which keeps each clip scoped to exactly the parameter set that
    # update owns.
    def __init__(self, configuration: VocosConfig) -> None:
        # Disables automatic optimization for the two-optimizer adversarial
        # recipe and constructs the mel transform, ConvNeXt-ISTFT network,
        # both discriminators, and the composite loss.
        #
        # Args:
        #     configuration: The frozen recipe record. The network
        #         topology is forwarded into a separate network
        #         configuration record, and the ISTFT padding mode is
        #         fixed to the centered grid here rather than exposed as a
        #         module-level field.
        #
        # Note:
        #     Both discriminator ensembles are constructed at their
        #     reference settings and take no configuration input, because
        #     the reference recipe holds them fixed.
        super().__init__()
        self.automatic_optimization: bool = False
        self._configuration: VocosConfig = configuration
        self._mel_spectrogram: MelSpectrogram = MelSpectrogram(configuration.mel_protocol)
        self.network: VocosNetwork = VocosNetwork(
            VocosNetworkConfig(
                input_channels=configuration.input_channels,
                hidden_dimension=configuration.hidden_dimension,
                intermediate_dimension=configuration.intermediate_dimension,
                layer_count=configuration.layer_count,
                n_fft=configuration.n_fft,
                hop_length=configuration.hop_length,
                padding="center"
            )
        )
        self._period_discriminator: VocosMultiPeriodDiscriminator = VocosMultiPeriodDiscriminator()
        self._resolution_discriminator: VocosMultiResolutionDiscriminator = VocosMultiResolutionDiscriminator()
        self._loss: VocosLoss = VocosLoss(configuration.loss_configuration)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Synthesizes a waveform batch from conditioning mels through the
        # generator network. Because the network member is what the
        # subclass replaces, routing forward through it rather than
        # through a captured reference is what makes the adaptation
        # inherit this method unchanged.
        return self.network(mel)

    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        # Generates waveform output from the architecture-specific acoustic representation.
        # This is the Vocoder protocol entry point the measurement stack
        # calls. Unlike the convolutional families it returns a
        # [batch, samples] waveform with no channel axis, because the
        # inverse-STFT head emits the waveform directly.
        return self.forward(mel)

    @property
    def mel_protocol(self) -> MelConfig:
        # Returns the mel-spectrogram protocol required by this architecture family.
        # Its sample rate is also the rate reference waveforms are
        # resampled to when the corpus was recorded at another rate.
        return self._configuration.mel_protocol

    @property
    def metric_mel_protocol(self) -> MelConfig:
        # Returns the protocol the mel-error metric extracts with, which
        # for this family is the conditioning protocol itself.
        return self._configuration.mel_protocol

    @property
    def configuration(self) -> VocosConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # One adversarial training step: single synthesis shared by both
        # updates (detached for the discriminator), followed by the explicit
        # per-step cosine scheduler advance; components are logged and the
        # detached generator loss returned for reporting.
        #
        # The updates run discriminator first and generator second. Both
        # read the same synthesis rather than each producing its own, so
        # the generator update sees exactly the material the discriminator
        # was trained against within this same step. Reference and
        # synthesis are cropped to
        # a common length before either ensemble sees them, because the
        # inverse-STFT head returns one hop less than the reference under
        # the centered grid.
        #
        # Args:
        #     batch: The collated LJSpeech mapping. Both its waveform and
        #         its sample rate are read, the latter to decide whether
        #         resampling to the protocol rate is required.
        #     batch_idx: Index of this batch within the epoch. It is not
        #         consulted.
        #
        # Returns:
        #     A mapping whose ``"loss"`` is the detached generator loss,
        #     reporting-only because both optimizers have already stepped
        #     and both schedules have already advanced.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping, its waveform entry
        #         is not a tensor, or its sample-rate entry is neither an
        #         integer nor a tensor.
        #     ValueError: If the waveform cannot be read as [batch, time].
        #     RuntimeError: If the module is not attached to a trainer
        #         holding the optimizer pair, or holding the two cosine
        #         schedules the step advances.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        generator_optimizer, discriminator_optimizer = self._require_gan_optimizers()
        reference_waveform: torch.Tensor = self._prepare_reference_waveform(batch)
        reference_mel: torch.Tensor = self._prepare_reference_mel(reference_waveform)
        synthesized_waveform: torch.Tensor = self.network(reference_mel)
        reference_waveform, synthesized_waveform = self._align_waveform_pair(reference_waveform, synthesized_waveform)
        discriminator_loss, discriminator_components = self._run_discriminator_training_step(
            discriminator_optimizer=discriminator_optimizer,
            real_waveform=reference_waveform,
            fake_waveform=synthesized_waveform
        )
        generator_loss, generator_components = self._run_generator_training_step(
            generator_optimizer=generator_optimizer,
            reference_mel=reference_mel,
            real_waveform=reference_waveform,
            fake_waveform=synthesized_waveform
        )
        self._step_cosine_schedulers()
        self.log("train_loss", generator_loss.detach())
        self.log("train_generator_loss", generator_loss.detach())
        self.log("train_discriminator_loss", discriminator_loss.detach())
        for name, value in generator_components.items():
            self.log(f"train_{name}", value)
        for name, value in discriminator_components.items():
            self.log(f"train_{name}", value)
        return {"loss": generator_loss.detach()}

    @override
    def validation_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Validates by mel L1 between reference and synthesis, logged as
        # val_loss for checkpoint monitoring. The synthesis is re-analyzed
        # under the same protocol that produced its conditioning mel, so
        # the comparison is a closed loop through the family's own grid.
        # The discriminators are not consulted: an adversarial score
        # tracks the current discriminator rather than synthesis quality
        # and cannot serve as a checkpoint-selection monitor.
        #
        # Args:
        #     batch: The collated LJSpeech mapping; both its waveform and
        #         its sample rate are read.
        #     batch_idx: Index of this batch within the validation pass.
        #         It is not consulted.
        #
        # Returns:
        #     A mapping whose ``"loss"`` is the validation loss, the same
        #     value published as ``val_loss``, alongside which every named
        #     component is published under a validation prefix.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping, its waveform entry
        #         is not a tensor, or its sample-rate entry is neither an
        #         integer nor a tensor.
        #     ValueError: If the waveform cannot be read as [batch, time],
        #         or the re-analyzed mel cannot be read as a batched
        #         [batch, bands, frames] tensor.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        reference_waveform: torch.Tensor = self._prepare_reference_waveform(batch)
        reference_mel: torch.Tensor = self._prepare_reference_mel(reference_waveform)
        synthesized_waveform: torch.Tensor = self.network(reference_mel)
        candidate_mel: torch.Tensor = self._prepare_network_mel(self._compute_full_precision_mel(synthesized_waveform))
        validation_loss, components = self._loss.compute_validation_loss(
            reference_mel=reference_mel,
            candidate_mel=candidate_mel
        )
        self.log("val_loss", validation_loss)
        for name, value in components.items():
            self.log(f"val_{name}", value)
        return {"loss": validation_loss}

    @override
    def predict_step(self, batch: Batch, batch_idx: int) -> ModelOutput:
        # Synthesizes from the batch waveform's conditioning mel and
        # returns the synthesized_waveform mapping the measurement stack
        # consumes. Conditioning is analysis-by-synthesis: the mel comes
        # from the reference waveform of the batch itself, after any
        # resampling to the protocol rate. The synthesis is returned at
        # the protocol rate rather than the corpus rate, so a resampled
        # batch yields a longer waveform than it carried in.
        #
        # Args:
        #     batch: The collated LJSpeech mapping; both its waveform and
        #         its sample rate are read.
        #     batch_idx: Index of this batch within the pass. It is not
        #         consulted.
        #
        # Returns:
        #     A mapping under the key ``"synthesized_waveform"`` holding a
        #     [batch, samples] tensor straight from the inverse-STFT head,
        #     with no channel axis to squeeze.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping, its waveform entry
        #         is not a tensor, or its sample-rate entry is neither an
        #         integer nor a tensor.
        #     ValueError: If the waveform cannot be read as [batch, time],
        #         which includes genuinely multichannel material.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        reference_waveform: torch.Tensor = self._prepare_reference_waveform(batch)
        reference_mel: torch.Tensor = self._prepare_reference_mel(reference_waveform)
        synthesized_waveform: torch.Tensor = self.network(reference_mel)
        return {"synthesized_waveform": synthesized_waveform}

    @override
    def test_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Delegates to predict_step and validates the output contract, so
        # test evaluation measures exactly the prediction path. The
        # returned mapping is rebuilt with the single verified key rather
        # than forwarded.
        #
        # Args:
        #     batch: The collated LJSpeech mapping, forwarded unchanged.
        #     batch_idx: Index of this batch, forwarded unchanged.
        #
        # Returns:
        #     A mapping holding exactly ``"synthesized_waveform"``.
        #
        # Raises:
        #     TypeError: If the prediction step returns a non-mapping, or
        #         its synthesized-waveform entry is absent or not a
        #         tensor, in addition to the guards the prediction step
        #         applies to the batch.
        prediction: ModelOutput = self.predict_step(batch, batch_idx)
        if not isinstance(prediction, dict):
            raise TypeError(f"predict_step must return a mapping, got {type(prediction).__name__}")
        synthesized_waveform: ModelOutput | None = prediction.get("synthesized_waveform")
        if not isinstance(synthesized_waveform, torch.Tensor):
            raise TypeError("predict_step output must contain Tensor key synthesized_waveform")
        return {"synthesized_waveform": synthesized_waveform}

    @override
    def configure_optimizers(self) -> OptimizationConfiguration:
        # Declares the optimizers and schedulers required by this architecture.
        # Declaration order is load-bearing, since the training step
        # unpacks the pair by position and the schedules are paired to the
        # optimizers by position as well: the generator optimizer owns
        # exactly the network parameters, and the discriminator optimizer
        # owns the period and resolution ensembles together.
        #
        # Returns:
        #     An OptimizationConfiguration holding the two live AdamW
        #     optimizers, generator first, each paired with a cosine
        #     schedule declared at step interval over the configured
        #     horizon and decaying to zero. The schedules are
        #     declarations; this module advances them itself inside the
        #     training step because it optimizes manually.
        generator_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self._configuration.learning_rate,
            betas=(self._configuration.adam_beta_1, self._configuration.adam_beta_2)
        )
        discriminator_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            list(self._period_discriminator.parameters()) + list(self._resolution_discriminator.parameters()),
            lr=self._configuration.learning_rate,
            betas=(self._configuration.adam_beta_1, self._configuration.adam_beta_2)
        )
        return OptimizationConfiguration(
            optimizer=[generator_optimizer, discriminator_optimizer],
            scheduler=[
                SchedulerConfig(
                    name="cosine",
                    interval="step",
                    t_max=self._configuration.scheduler_total_steps,
                    eta_min=0.0
                ),
                SchedulerConfig(
                    name="cosine",
                    interval="step",
                    t_max=self._configuration.scheduler_total_steps,
                    eta_min=0.0
                )
            ]
        )

    def _prepare_reference_waveform(self, batch: LJSpeechBatch) -> torch.Tensor:
        # Reads the reference waveform, normalizes it to [batch, time], and
        # resamples to the protocol rate when the corpus rate differs.
        # A single-sample waveform is promoted to a batch of one and a
        # degenerate single-channel axis is squeezed out, but genuinely
        # multichannel material is rejected rather than mixed down.
        # Resampling is what lets a 22.05 kHz corpus drive this family's
        # 24 kHz recipe without altering the published protocol; it
        # changes the sample count, which is why downstream length
        # arithmetic is expressed against the protocol rate rather than
        # the corpus rate.
        #
        # Args:
        #     batch: The collated LJSpeech mapping supplying both the
        #         waveform and the corpus sample rate.
        #
        # Returns:
        #     The reference waveform as [batch, time] at the protocol
        #     sample rate.
        #
        # Raises:
        #     TypeError: If the waveform or sample-rate entry is missing
        #         or carries an unusable type.
        #     ValueError: If the waveform cannot be read as [batch, time].
        waveform: torch.Tensor = self._extract_waveform(batch)
        sample_rate: int = self._extract_sample_rate(batch)
        if waveform.ndim == 1:
            waveform: torch.Tensor = waveform.unsqueeze(0)
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform: torch.Tensor = waveform.squeeze(1)
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform shape [batch, time], got {tuple(waveform.shape)}")
        target_sample_rate: int = self._configuration.mel_protocol.sample_rate
        if sample_rate != target_sample_rate:
            waveform: torch.Tensor = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=target_sample_rate)
        return waveform

    def _prepare_reference_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        # Produces the batched full-precision conditioning mel.
        # Composing extraction with layout normalization in one helper is
        # what keeps the conditioning mel and the re-analysis mel of the
        # validation loop derived through identical steps.
        return self._prepare_network_mel(self._compute_full_precision_mel(waveform))

    def _compute_full_precision_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        # Mel features condition the network and define loss targets, so they stay float32.
        # The explicit cast and the disabled-autocast window are both
        # applied: the cast fixes the input precision and the window stops
        # the transform's internal operations from being downcast.
        with self._full_precision_context():
            return self._mel_spectrogram(waveform.float())

    def _full_precision_context(self) -> torch.autocast | nullcontext[None]:
        # Keeps mel-spectrogram extraction in float32 so mixed-precision training cannot alter
        # the conditioning protocol validated by the published checkpoint.
        # torch.autocast accepts only device types it implements, so any
        # other placement falls back to a null context; the caller's
        # explicit float cast carries the precision guarantee even there.
        device_type: str = self.device.type
        if device_type in ("cuda", "cpu"):
            return torch.autocast(device_type=device_type, enabled=False)
        return nullcontext()

    def _step_cosine_schedulers(self) -> None:
        # Steps the paired generator and discriminator cosine schedulers once per optimizer step.
        # Stepping is explicit because manual optimization leaves schedule
        # advancement to the module, and it happens per batch rather than
        # per epoch because the reference recipe declares a step-interval
        # cosine decay. Both resulting rates are logged so the descent is
        # visible in the run record.
        #
        # Raises:
        #     RuntimeError: If the trainer does not hold exactly the two
        #         declared schedules, which would mean the optimization
        #         declaration and the module disagree.
        schedulers: torch.optim.lr_scheduler.LRScheduler | list[torch.optim.lr_scheduler.LRScheduler] | None = self.lr_schedulers()
        if not isinstance(schedulers, list) or len(schedulers) != 2:
            raise RuntimeError("Vocos training requires generator and discriminator schedulers")
        for scheduler in schedulers:
            scheduler.step()
        self.log("learning_rate_generator", float(schedulers[0].get_last_lr()[0]))
        self.log("learning_rate_discriminator", float(schedulers[1].get_last_lr()[0]))

    def _prepare_network_mel(self, mel: torch.Tensor) -> torch.Tensor:
        # Normalizes mel layouts to the [batch, bands, frames] shape the
        # network consumes. An unbatched mel gains a batch axis and a mel
        # carrying a degenerate channel axis has it squeezed out; anything
        # else is a layout error rather than something to reshape
        # silently.
        #
        # Raises:
        #     ValueError: If the mel carries a layout outside the three
        #         accepted forms. The message enumerates them.
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
        # Aligns generated and reference tensors before loss or metric computation.
        # Both tensors are cropped from the front to their common length.
        # Under the centered inverse-STFT grid the synthesis is one hop
        # shorter than the reference by construction, so this crop is the
        # expected path rather than an error branch; HiFi-GAN raises on
        # the same disagreement because its upsampling chain has no such
        # boundary deficit.
        #
        # Truncating a reference and a candidate to their common unpadded
        # length is the same rule the evaluation protocol applies when it
        # reports true-length quality. This helper serves the training and
        # validation paths of one module; the evaluation-side truncation
        # is performed independently by the measurement stack.
        #
        # Returns:
        #     The two waveforms trimmed to a common sample count, in the
        #     order they were given.
        minimum_samples: int = min(first_waveform.shape[-1], second_waveform.shape[-1])
        return first_waveform[..., :minimum_samples], second_waveform[..., :minimum_samples]

    def _run_discriminator_training_step(
        self,
        discriminator_optimizer: torch.optim.Optimizer,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Discriminator update: both ensembles judge real against the
        # detached synthesis, stepping only the discriminator under the
        # optimizer toggle. Gradients are norm-clipped over exactly the
        # two ensembles' parameters before the step, so the clip scope
        # matches the optimizer's own parameter set.
        #
        # Args:
        #     discriminator_optimizer: The optimizer owning both
        #         ensembles.
        #     real_waveform: The reference waveform as [batch, samples].
        #     fake_waveform: The synthesis in the same layout, detached
        #         here before either ensemble sees it.
        #
        # Returns:
        #     The detached discriminator loss paired with its named scalar
        #     components for logging.
        with self.toggled_optimizer(discriminator_optimizer):
            real_period_logits, fake_period_logits, _, _ = self._period_discriminator(
                real_waveform,
                fake_waveform.detach()
            )
            real_resolution_logits, fake_resolution_logits, _, _ = self._resolution_discriminator(
                real_waveform,
                fake_waveform.detach()
            )
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
        reference_mel: torch.Tensor,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Generator update: adversarial logits and feature maps from both
        # ensembles combine with the mel reconstruction term; generator
        # gradients are norm-clipped before the step under the optimizer
        # toggle.
        # The ensembles are re-run on the attached synthesis so gradients
        # reach the generator through them, and the real-side logits are
        # discarded immediately because only the real branch's feature
        # maps enter the objective. The reconstruction term compares the
        # conditioning mel against a fresh re-analysis of the synthesis
        # under the same protocol, so it closes the loop through the
        # family's own grid.
        #
        # Args:
        #     generator_optimizer: The optimizer owning the generator
        #         network parameters.
        #     reference_mel: The conditioning mel, which is also the
        #         target of the reconstruction term.
        #     real_waveform: The reference waveform as [batch, samples].
        #     fake_waveform: The synthesis in the same layout, kept
        #         attached so the update reaches the generator.
        #
        # Returns:
        #     The detached generator loss paired with its named scalar
        #     components for logging.
        with self.toggled_optimizer(generator_optimizer):
            real_period_logits, fake_period_logits, real_period_features, fake_period_features = (
                self._period_discriminator(real_waveform, fake_waveform)
            )
            del real_period_logits
            real_resolution_logits, fake_resolution_logits, real_resolution_features, fake_resolution_features = (
                self._resolution_discriminator(real_waveform, fake_waveform)
            )
            del real_resolution_logits
            candidate_mel: torch.Tensor = self._prepare_network_mel(self._compute_full_precision_mel(fake_waveform))
            generator_loss, components = self._loss.compute_generator_loss(
                reference_mel=reference_mel,
                candidate_mel=candidate_mel,
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

    def _require_gan_optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        # Returns the generator and discriminator optimizers in declaration
        # order, failing loudly when the pair is absent.
        #
        # Raises:
        #     RuntimeError: If the module is detached from a trainer, or
        #         the trainer holds a single optimizer rather than the
        #         declared pair. A detached module reads as no optimizers
        #         at all, so calling the training step outside a fit run
        #         fails here rather than partway through an update.
        optimizers: torch.optim.Optimizer | list[torch.optim.Optimizer] | None = self.optimizers()
        if not isinstance(optimizers, list) or len(optimizers) != 2:
            raise RuntimeError("Vocos training requires generator and discriminator optimizers")
        return optimizers[0], optimizers[1]

    def _extract_waveform(self, batch: LJSpeechBatch) -> torch.Tensor:
        # Reads the waveform tensor from the collated batch mapping.
        # A missing key and a non-tensor value fail identically, so a
        # batch carrying an audio path instead of decoded samples is
        # rejected here rather than at the first transform.
        #
        # Raises:
        #     TypeError: If the batch has no waveform entry or its value
        #         is not a tensor.
        waveform: LJSpeechBatchValue | None = batch.get("waveform")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError(f"batch['waveform'] must be Tensor, got {type(waveform).__name__}")
        return waveform

    def _extract_sample_rate(self, batch: LJSpeechBatch) -> int:
        # Reads the batch sample rate from the collated batch mapping.
        # Collation may turn a per-sample integer rate into a tensor, so
        # both forms are accepted; a tensor is read from its first element
        # because a batch is drawn from one corpus at one rate.
        #
        # Returns:
        #     The corpus sample rate as an integer, which the caller
        #     compares against the protocol rate to decide on resampling.
        #
        # Raises:
        #     TypeError: If the batch has no sample-rate entry or its
        #         value is neither an integer nor a tensor. The rate is
        #         mandatory rather than defaulted, because assuming it
        #         would silently skip a needed resample.
        sample_rate: LJSpeechBatchValue | None = batch.get("sample_rate")
        if isinstance(sample_rate, int):
            return sample_rate
        if isinstance(sample_rate, torch.Tensor):
            return int(sample_rate.flatten()[0].item())
        raise TypeError(f"batch['sample_rate'] must be int or Tensor, got {type(sample_rate).__name__}")
