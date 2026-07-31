# This module:
# 1. Implements the HiFTNet vocoder aligned with the yl4579 reference: a
#    neural-source-filter design in which an F0 extractor drives a
#    harmonic source signal that the inverse-STFT generator filters into
#    the waveform
# 2. Defines HiftnetConfig, the frozen record fixing generator topology,
#    F0-extractor provenance, adversarial hyperparameters, and the mel and
#    metric protocols this family is measured under
# 3. Remains the documented non-executed exclusion of the study contract:
#    no admissible project-trained checkpoint was produced, and the
#    architecture enters no Study 1 aggregate or Study 2 cell
#
# Harness contract (syntheticmind):
# - Subclasses the harness Module with automatic optimization disabled
#   under the same two-optimizer adversarial surface as the executed GAN
#   families; the implementation stays training-capable for future work
# - configure_optimizers returns already-constructed torch optimizers and
#   schedulers rather than harness SchedulerConfig records; the trainer
#   files a null scheduler configuration against each pre-built scheduler
#   and the fit loop advances only configured epoch-interval schedules, so
#   on_train_epoch_end steps the exponential decay itself
#
# Design decisions:
# - The F0 extractor is a JDC pitch network instantiated inside the
#   generator and initialized from the published pitch-model release when
#   HiftnetConfig.f0_checkpoint_path names an existing file; its
#   parameters remain inside HiftnetNetwork.parameters() and therefore
#   inside the generator optimizer, so the pitch model is fine-tuned
#   jointly with the filter network rather than held frozen
# - An unset or absent F0 checkpoint path is a silent no-op rather than an
#   error, so the architecture stays constructible and testable without
#   the external release; only the pitch initialization is then random
# - One generator forward pass serves both halves of the training step:
#   the discriminator update consumes its detached synthesis and the
#   generator update reuses the same graph-carrying tensors
#
# Author: Rahul Sawhney

from pathlib import Path
from typing import ClassVar, override

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from syntheticmind.core.module import Module
from syntheticmind.core.optimizer import OptimizationConfiguration
from syntheticmind.utilities.types import Batch, ModelOutput, StepOutput

from vocode.data.ljspeech_dataset import LJSpeechBatch, LJSpeechBatchValue
from vocode.losses.hiftnet import HiftnetLoss, HiftnetLossConfig
from vocode.metrics.pesq import PesqConfig
from vocode.metrics.stoi import StoiConfig
from vocode.models.hiftnet.discriminator import (
    HiftnetMultiPeriodDiscriminator,
    HiftnetMultiResolutionSpectrogramDiscriminator,
)
from vocode.models.hiftnet.network import HiftnetGeneratorOutput, HiftnetNetwork, HiftnetNetworkConfig
from vocode.transforms.mel import MelConfig, MelSpectrogram

__all__: list[str] = ["Hiftnet", "HiftnetConfig"]


class HiftnetConfig(BaseModel):
    # Frozen HiFTNet configuration: generator and source-module topology,
    # F0-extractor provenance, adversarial hyperparameters, and the family's
    # mel and metric protocols. Explicit validated fields prevent experiment
    # settings from drifting between runs; the model is frozen, forbids
    # unknown keys, and validates strictly, so a typo or a silently widened
    # type fails at construction rather than at training time.
    #
    # Fields:
    #     input_mel_channels: Number of mel bands in the conditioning
    #         representation, which must equal the band count produced by
    #         ``mel_protocol``.
    #     sampling_rate: Waveform sampling rate in hertz, forwarded to the
    #         harmonic source module so sine phases advance at the correct
    #         rate.
    #     upsample_rates: Per-stage temporal upsampling factors of the
    #         transposed-convolution stack. Their product multiplied by
    #         ``gen_istft_hop_size`` is the network's total upsampling and
    #         must equal the hop length of ``mel_protocol``.
    #     upsample_kernel_sizes: Transposed-convolution kernel widths paired
    #         positionally with ``upsample_rates``; the two tuples must have
    #         equal length.
    #     upsample_initial_channel: Channel count entering the first
    #         upsampling stage, halved at every subsequent stage.
    #     resblock_kernel_sizes: Kernel widths of the multi-receptive-field
    #         residual blocks applied after each upsampling stage.
    #     resblock_dilation_sizes: Dilation triples paired positionally with
    #         ``resblock_kernel_sizes``; the two tuples must have equal
    #         length.
    #     gen_istft_n_fft: Transform size of the inverse-STFT head. The head
    #         emits ``gen_istft_n_fft // 2 + 1`` magnitude channels and the
    #         same number of phase channels.
    #     gen_istft_hop_size: Hop length of the inverse-STFT head, the final
    #         upsampling factor of the synthesis chain.
    #     f0_checkpoint_path: Filesystem location of the published JDC
    #         pitch-model release. ``None`` or a path that does not exist
    #         leaves the pitch network randomly initialized without raising.
    #     learning_rate: Shared AdamW step size for the generator and
    #         discriminator optimizers. Default: ``0.0002``.
    #     adam_beta_1: First AdamW moment decay for both optimizers.
    #         Default: ``0.8``.
    #     adam_beta_2: Second AdamW moment decay for both optimizers.
    #         Default: ``0.99``.
    #     learning_rate_decay: Per-epoch multiplicative factor of the
    #         exponential schedules attached to both optimizers.
    #         Default: ``0.999``.
    #     gradient_clip_norm: Maximum gradient norm applied separately to
    #         the generator and discriminator parameter sets before each
    #         step. The reference value is deliberately permissive and acts
    #         as a divergence guard rather than a shaping constraint.
    #         Default: ``1000.0``.
    #     loss_configuration: Weights of the composite objective.
    #         Default: ``HiftnetLossConfig()``.
    #     mel_protocol: Mel protocol used both to build the generator's
    #         conditioning input and to score the mel reconstruction term.
    #     pesq_protocol: PESQ measurement protocol for this family.
    #     stoi_protocol: STOI measurement protocol for this family.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    input_mel_channels: PositiveInt
    sampling_rate: PositiveInt
    upsample_rates: tuple[PositiveInt, ...]
    upsample_kernel_sizes: tuple[PositiveInt, ...]
    upsample_initial_channel: PositiveInt
    resblock_kernel_sizes: tuple[PositiveInt, ...]
    resblock_dilation_sizes: tuple[tuple[PositiveInt, PositiveInt, PositiveInt], ...]
    gen_istft_n_fft: PositiveInt
    gen_istft_hop_size: PositiveInt
    f0_checkpoint_path: Path | None
    learning_rate: PositiveFloat = 0.0002
    adam_beta_1: PositiveFloat = 0.8
    adam_beta_2: PositiveFloat = 0.99
    learning_rate_decay: PositiveFloat = 0.999
    gradient_clip_norm: PositiveFloat = 1000.0
    loss_configuration: HiftnetLossConfig = HiftnetLossConfig()
    mel_protocol: MelConfig
    pesq_protocol: PesqConfig
    stoi_protocol: StoiConfig

    @classmethod
    def yl4579(cls, f0_checkpoint_path: Path | None = None) -> HiftnetConfig:
        # Builds the HiFTNet configuration aligned with the yl4579 reference
        # implementation. The two upsampling stages contribute a factor of
        # sixty-four and the inverse-STFT head the remaining factor of four,
        # so the chain reproduces the two-hundred-fifty-six-sample hop of the
        # HiFi-GAN mel protocol this family shares.
        #
        # Args:
        #     f0_checkpoint_path: Location of the published JDC pitch-model
        #         release. ``None`` leaves the pitch network randomly
        #         initialized, which is the correct setting for construction
        #         and shape tests that must not depend on an external
        #         download. Default: ``None``.
        #
        # Returns:
        #     The frozen configuration record of this architecture. Unlike the
        #     executed families, it backs no Project-Trained Configuration,
        #     because this architecture produced no admissible project-trained
        #     checkpoint and remains a documented exclusion.
        return cls(
            input_mel_channels=80,
            sampling_rate=22050,
            upsample_rates=(8, 8),
            upsample_kernel_sizes=(16, 16),
            upsample_initial_channel=512,
            resblock_kernel_sizes=(3, 7, 11),
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            gen_istft_n_fft=16,
            gen_istft_hop_size=4,
            f0_checkpoint_path=f0_checkpoint_path,
            mel_protocol=MelConfig.hiftnet_yl4579(),
            pesq_protocol=PesqConfig(),
            stoi_protocol=StoiConfig()
        )


class Hiftnet(Module):
    # HiFTNet vocoder module: an F0 extractor drives a harmonic source signal
    # that the inverse-STFT filter generator shapes into the waveform, trained
    # adversarially against a period ensemble and a multi-resolution
    # spectrogram ensemble under manual optimization. The module owns the mel
    # transform used for both conditioning and reconstruction scoring, the
    # generator network, both discriminator ensembles, and the composite
    # objective; the discriminators exist only during training and are absent
    # from the synthesis surface. Local training and published-checkpoint
    # loading stay separate: this class never reads a release, and the
    # HiftnetWeights adapter never constructs a module.
    #
    # Integration: the external F0 dependency is the boundary that
    # distinguishes this family from the other GAN vocoders. The pitch network
    # is constructed unconditionally inside HiftnetNetwork, but its published
    # parameters arrive only when HiftnetConfig.f0_checkpoint_path names an
    # existing file; an unset or missing path leaves the pitch network
    # randomly initialized and raises nothing. Construction, shape, and
    # gradient tests therefore run without any external download, while a
    # faithful reproduction run requires the release to be present. The pitch
    # parameters are not frozen: they belong to HiftnetNetwork.parameters()
    # and so enter the generator optimizer, meaning any run fine-tunes the
    # pitch model jointly and a checkpoint supplies initialization rather than
    # a fixed pitch oracle.
    #
    # Integration: the module satisfies the vocode.models.vocoder.Vocoder
    # structural protocol through its network property, its mel_protocol and
    # metric_mel_protocol properties, and synthesize, so the measurement and
    # profiling stacks consume it without inheritance coupling.
    def __init__(self, configuration: HiftnetConfig) -> None:
        # Disables automatic optimization, because the adversarial recipe
        # drives two optimizers from inside training_step and the harness
        # forbids loop-owned stepping in that regime. Constructs the mel
        # transform, the generator network including its F0 extractor, both
        # discriminator ensembles, and the composite objective. Any F0
        # checkpoint named by the configuration is read during network
        # construction, so this constructor performs filesystem access
        # exactly when that path exists.
        super().__init__()
        self.automatic_optimization: bool = False
        self._configuration: HiftnetConfig = configuration
        self._mel_spectrogram: MelSpectrogram = MelSpectrogram(configuration.mel_protocol)
        self.network: HiftnetNetwork = HiftnetNetwork(
            HiftnetNetworkConfig(
                input_mel_channels=configuration.input_mel_channels,
                sampling_rate=configuration.sampling_rate,
                upsample_rates=configuration.upsample_rates,
                upsample_kernel_sizes=configuration.upsample_kernel_sizes,
                upsample_initial_channel=configuration.upsample_initial_channel,
                resblock_kernel_sizes=configuration.resblock_kernel_sizes,
                resblock_dilation_sizes=configuration.resblock_dilation_sizes,
                gen_istft_n_fft=configuration.gen_istft_n_fft,
                gen_istft_hop_size=configuration.gen_istft_hop_size,
                f0_checkpoint_path=configuration.f0_checkpoint_path
            )
        )
        self._period_discriminator: HiftnetMultiPeriodDiscriminator = HiftnetMultiPeriodDiscriminator()
        self._spectrogram_discriminator: HiftnetMultiResolutionSpectrogramDiscriminator = (
            HiftnetMultiResolutionSpectrogramDiscriminator()
        )
        self._loss: HiftnetLoss = HiftnetLoss(configuration.loss_configuration)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Synthesizes a waveform batch from conditioning mels through the
        # F0-driven source-filter generator.
        #
        # Args:
        #     mel: Conditioning mel of shape ``[batch, channels, frames]``
        #         built under this family's mel protocol.
        #
        # Returns:
        #     The synthesized waveform of shape ``[batch, 1, samples]``, the
        #     channel axis introduced by the inverse-STFT head.
        return self.network(mel)

    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        # Generates waveform output from the architecture-specific acoustic
        # representation. This is the Vocoder protocol entry point used by the
        # measurement and profiling stacks; it delegates to forward so the
        # protocol surface and the module surface cannot diverge.
        return self.forward(mel)

    @property
    def mel_protocol(self) -> MelConfig:
        # Returns the mel-spectrogram protocol required by this architecture family.
        return self._configuration.mel_protocol

    @property
    def metric_mel_protocol(self) -> MelConfig:
        # Returns the mel protocol used for measurement. This family scores
        # under the same protocol it conditions on, so the two properties
        # deliberately return one record.
        return self._configuration.mel_protocol

    @property
    def configuration(self) -> HiftnetConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    @override
    def on_train_epoch_end(self) -> None:
        # Advances the exponential learning-rate decay for every optimizer at
        # the epoch boundary. This stepping is the model's responsibility
        # rather than the loop's: configure_optimizers hands back
        # already-constructed torch schedulers, the trainer records a null
        # scheduler configuration for each, and the fit loop advances only
        # schedules carrying a configured epoch interval. Both the unwrapped
        # single-scheduler form and the list form of lr_schedulers() are
        # handled so the method stays correct if the optimizer count changes.
        schedulers: torch.optim.lr_scheduler.LRScheduler | list[torch.optim.lr_scheduler.LRScheduler] | None = self.lr_schedulers()
        if isinstance(schedulers, list):
            for scheduler in schedulers:
                scheduler.step()
        elif schedulers is not None:
            schedulers.step()

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Executes one full adversarial step under manual optimization. The
        # generator runs forward exactly once and its output serves both
        # updates: the discriminator update consumes a detached copy so no
        # generator gradient is produced while the critics learn, and the
        # generator update then reuses the same graph-carrying synthesis so
        # the second half costs no additional forward pass. The discriminator
        # is updated first, so the generator is scored against critics that
        # have already seen this batch. Each update is fenced by the harness
        # optimizer toggle, which restricts requires_grad to the parameters
        # owned by the stepping optimizer and restores the previous flags on
        # exit, so neither half can accumulate gradients into the other's
        # parameters.
        #
        # Args:
        #     batch: Collated LJSpeech mapping; only the ``"waveform"`` entry
        #         is read, and the conditioning mel is derived from it rather
        #         than taken from the batch, so training conditions on
        #         exactly the protocol this module owns.
        #     batch_idx: Index of this batch within the epoch. Unused, because
        #         the update schedule is identical for every batch.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping, or if its
        #         ``"waveform"`` entry is absent or not a tensor.
        #     ValueError: If the reference waveform does not normalize to
        #         ``[batch, time]``.
        #     RuntimeError: If the trainer did not materialize exactly the two
        #         optimizers this recipe requires.
        #
        # Returns:
        #     A mapping whose ``"loss"`` entry is the detached generator loss.
        #     The value is reporting-only: both backward passes and both
        #     optimizer steps have already executed inside this method, so the
        #     loop must not differentiate it.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        # Reference preparation: the waveform is normalized to [batch, time]
        # and analyzed into the conditioning mel that drives synthesis.
        generator_optimizer, discriminator_optimizer = self._require_gan_optimizers()
        reference_waveform: torch.Tensor = self._prepare_reference_waveform(batch)
        reference_mel: torch.Tensor = self._prepare_reference_mel(reference_waveform)
        generator_output: HiftnetGeneratorOutput = self.network.predict_components(reference_mel)
        # Critic-input preparation: both waveforms are lifted to the
        # [batch, 1, time] layout the ensembles expect and truncated to their
        # common length, because the inverse-STFT head does not reproduce the
        # reference sample count exactly.
        real_discriminator_waveform: torch.Tensor = self._to_discriminator_waveform(reference_waveform)
        fake_discriminator_waveform: torch.Tensor = self._to_discriminator_waveform(generator_output.waveform)
        real_discriminator_waveform, fake_discriminator_waveform = self._align_waveform_pair(
            real_discriminator_waveform,
            fake_discriminator_waveform
        )
        # Critic update, then generator update; each returns its detached
        # total together with the scalar component breakdown.
        discriminator_loss, discriminator_components = self._run_discriminator_training_step(
            discriminator_optimizer=discriminator_optimizer,
            real_waveform=real_discriminator_waveform,
            fake_waveform=fake_discriminator_waveform
        )
        generator_loss, generator_components = self._run_generator_training_step(
            generator_optimizer=generator_optimizer,
            reference_mel=reference_mel,
            generator_output=generator_output,
            real_waveform=real_discriminator_waveform,
            fake_waveform=fake_discriminator_waveform
        )
        # Reporting: train_loss duplicates the generator total under the
        # generic key the monitoring stack watches across every family, and
        # each component is republished under a train_ prefix.
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
        # Scores the weighted mel reconstruction distance between the
        # reference and the synthesis, published as val_loss for checkpoint
        # selection and early stopping. The adversarial terms are deliberately
        # excluded: discriminator scores drift as the critics train and are
        # therefore not comparable across epochs, whereas the mel distance is
        # a stable monotone quality proxy. The synthesis is re-analyzed
        # through the same mel transform used for conditioning, so the
        # reference and candidate spectra are directly comparable.
        #
        # Args:
        #     batch: Collated LJSpeech mapping; only the ``"waveform"`` entry
        #         is read.
        #     batch_idx: Index of this batch within the validation pass.
        #         Unused.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping or carries no waveform
        #         tensor.
        #
        # Returns:
        #     A mapping whose ``"loss"`` entry is the weighted mel
        #     reconstruction distance.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        reference_waveform: torch.Tensor = self._prepare_reference_waveform(batch)
        reference_mel: torch.Tensor = self._prepare_reference_mel(reference_waveform)
        generator_output: HiftnetGeneratorOutput = self.network.predict_components(reference_mel)
        candidate_mel: torch.Tensor = self._prepare_network_mel(
            self._mel_spectrogram(self._prepare_waveform_for_mel(generator_output.waveform))
        )
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
        # Synthesizes from the conditioning mel derived from the batch
        # waveform and returns the mapping the measurement stack consumes.
        # The waveform tensor is passed to mel analysis without the
        # normalization applied during training, so this path requires the
        # dataloader to deliver an already-batched waveform. The channel axis
        # introduced by the inverse-STFT head is squeezed away, so the result
        # is ``[batch, samples]`` and aligns with the reference layout the
        # metric components compare against.
        #
        # Args:
        #     batch: Collated LJSpeech mapping carrying a ``"waveform"``
        #         tensor.
        #     batch_idx: Index of this batch within the prediction pass.
        #         Unused.
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
        synthesized_waveform: torch.Tensor = self.network(reference_mel).squeeze(1)
        return {"synthesized_waveform": synthesized_waveform}

    @override
    def test_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Delegates to predict_step and revalidates its output, so the test
        # pass and the prediction pass synthesize through one code path. The
        # re-checks are not redundant with those inside predict_step: they
        # assert the contract of the override itself, which a subclass could
        # otherwise break silently.
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
        # Declares the adversarial optimizer pair in the order training_step
        # unpacks it: the generator optimizer first, the discriminator
        # optimizer second. The generator optimizer receives the whole
        # network, which includes the F0 extractor, so pitch parameters are
        # trained jointly. The discriminator optimizer receives the
        # concatenated parameters of both ensembles, so one step updates the
        # period and spectrogram critics together.
        #
        # Returns:
        #     An OptimizationConfiguration holding already-constructed torch
        #     objects rather than harness configuration records. This is a
        #     deliberate choice with a consequence: the trainer files a null
        #     scheduler configuration against a pre-built scheduler and never
        #     advances it, so on_train_epoch_end performs the per-epoch
        #     exponential decay itself.
        generator_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self._configuration.learning_rate,
            betas=(self._configuration.adam_beta_1, self._configuration.adam_beta_2)
        )
        discriminator_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            list(self._period_discriminator.parameters()) + list(self._spectrogram_discriminator.parameters()),
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

    def _prepare_reference_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        # Analyzes a waveform into the conditioning mel, normalizing the
        # layout on both sides of the transform so the result is always
        # ``[batch, channels, frames]``.
        mel: torch.Tensor = self._mel_spectrogram(self._prepare_waveform_for_mel(waveform))
        return self._prepare_network_mel(mel)

    def _prepare_reference_waveform(self, batch: LJSpeechBatch) -> torch.Tensor:
        # Reads the reference waveform and normalizes it to [batch, time],
        # promoting an unbatched sequence and rejecting any higher rank.
        waveform: torch.Tensor = self._extract_waveform(batch)
        if waveform.ndim == 1:
            waveform: torch.Tensor = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform shape [batch, time], got {tuple(waveform.shape)}")
        return waveform

    def _prepare_waveform_for_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        # Reduces a waveform to the [batch, time] layout the mel transform
        # requires. An unbatched sequence gains a batch axis and a singleton
        # channel axis is dropped, which is what makes it safe to feed the
        # generator's own [batch, 1, samples] output straight back into mel
        # analysis for the reconstruction term.
        if waveform.ndim == 1:
            return waveform.unsqueeze(0)
        if waveform.ndim == 2:
            return waveform
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            return waveform.squeeze(1)
        raise ValueError(f"Expected waveform shape [time], [batch, time], or [batch, 1, time], got {tuple(waveform.shape)}")

    def _prepare_network_mel(self, mel: torch.Tensor) -> torch.Tensor:
        # Normalizes a mel to the [batch, channels, frames] layout the
        # generator consumes, promoting an unbatched spectrogram and dropping
        # a singleton channel axis that some transform paths introduce.
        if mel.ndim == 2:
            return mel.unsqueeze(0)
        if mel.ndim == 3:
            return mel
        if mel.ndim == 4 and mel.shape[1] == 1:
            return mel.squeeze(1)
        raise ValueError(f"Expected mel shape [channels, frames], [batch, channels, frames], or [batch, 1, channels, frames], got {tuple(mel.shape)}")

    def _to_discriminator_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        # Lifts a waveform to the [batch, 1, time] layout both discriminator
        # ensembles expect, adding whichever axes are missing and passing an
        # already-shaped tensor through untouched.
        if waveform.ndim == 1:
            return waveform.unsqueeze(0).unsqueeze(0)
        if waveform.ndim == 2:
            return waveform.unsqueeze(1)
        if waveform.ndim == 3:
            return waveform
        raise ValueError(f"Expected waveform with 1, 2, or 3 dimensions, got {tuple(waveform.shape)}")

    def _align_waveform_pair(
        self,
        first_waveform: torch.Tensor,
        second_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Truncates both waveforms to their common length along the time axis.
        # The inverse-STFT head reconstructs a sample count determined by the
        # frame count and hop, which need not equal the reference length, so
        # the critics would otherwise receive mismatched inputs. Truncation
        # rather than padding is used because appending zeros would present
        # the critics with silence that the generator never produced.
        minimum_samples: int = min(first_waveform.shape[-1], second_waveform.shape[-1])
        return first_waveform[..., :minimum_samples], second_waveform[..., :minimum_samples]

    def _run_discriminator_training_step(
        self,
        discriminator_optimizer: torch.optim.Optimizer,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Discriminator update. Both ensembles score the reference against the
        # detached synthesis, so no gradient reaches the generator; the toggle
        # additionally restricts requires_grad to the critic parameters for
        # the duration of the block and restores the prior flags on exit, even
        # if the body raises. Clipping is applied to the concatenated critic
        # parameters after backward and before the step, matching the order
        # the automatic path uses.
        #
        # Returns:
        #     The detached discriminator total and its scalar component
        #     breakdown, the latter already reduced to plain floats by the
        #     objective.
        with self.toggled_optimizer(discriminator_optimizer):
            real_period_logits, fake_period_logits, _, _ = self._period_discriminator(
                real_waveform,
                fake_waveform.detach()
            )
            real_spectrogram_logits, fake_spectrogram_logits, _, _ = self._spectrogram_discriminator(
                real_waveform,
                fake_waveform.detach()
            )
            discriminator_loss, components = self._loss.compute_discriminator_loss(
                real_period_logits=real_period_logits,
                fake_period_logits=fake_period_logits,
                real_spectrogram_logits=real_spectrogram_logits,
                fake_spectrogram_logits=fake_spectrogram_logits
            )
            self.optimizer_zero_grad(discriminator_optimizer)
            self.manual_backward(discriminator_loss)
            torch.nn.utils.clip_grad_norm_(
                list(self._period_discriminator.parameters()) + list(self._spectrogram_discriminator.parameters()),
                self._configuration.gradient_clip_norm
            )
            self.optimizer_step(discriminator_optimizer)
        return discriminator_loss.detach(), components

    def _run_generator_training_step(
        self,
        generator_optimizer: torch.optim.Optimizer,
        reference_mel: torch.Tensor,
        generator_output: HiftnetGeneratorOutput,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Generator update. The ensembles are re-run on the graph-carrying
        # synthesis, which is required rather than wasteful: the discriminator
        # pass above consumed a detached copy and so produced no path back to
        # the generator. Both critics are queried for logits and intermediate
        # features, the synthesis is re-analyzed into a mel for the
        # reconstruction term, and the composite objective combines the
        # adversarial, feature-matching, and reconstruction contributions.
        # Clipping covers the whole network, including the F0 extractor.
        #
        # Returns:
        #     The detached generator total and its scalar component
        #     breakdown.
        with self.toggled_optimizer(generator_optimizer):
            real_period_logits, fake_period_logits, real_period_features, fake_period_features = (
                self._period_discriminator(real_waveform, fake_waveform)
            )
            (
                real_spectrogram_logits,
                fake_spectrogram_logits,
                real_spectrogram_features,
                fake_spectrogram_features
            ) = self._spectrogram_discriminator(real_waveform, fake_waveform)
            candidate_mel: torch.Tensor = self._prepare_network_mel(
                self._mel_spectrogram(self._prepare_waveform_for_mel(generator_output.waveform))
            )
            generator_loss, components = self._loss.compute_generator_loss(
                reference_mel=reference_mel,
                candidate_mel=candidate_mel,
                real_period_logits=real_period_logits,
                fake_period_logits=fake_period_logits,
                real_spectrogram_logits=real_spectrogram_logits,
                fake_spectrogram_logits=fake_spectrogram_logits,
                real_period_features=real_period_features,
                fake_period_features=fake_period_features,
                real_spectrogram_features=real_spectrogram_features,
                fake_spectrogram_features=fake_spectrogram_features
            )
            self.optimizer_zero_grad(generator_optimizer)
            self.manual_backward(generator_loss)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self._configuration.gradient_clip_norm)
            self.optimizer_step(generator_optimizer)
        return generator_loss.detach(), components

    def _require_gan_optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        # Returns the generator and discriminator optimizers in declaration
        # order, failing loudly when the pair is absent. The list form is
        # asserted explicitly because the harness unwraps a single-optimizer
        # configuration to a bare object; receiving anything other than a
        # two-element list means the module was fitted under an optimization
        # setup this recipe cannot execute.
        optimizers: torch.optim.Optimizer | list[torch.optim.Optimizer] | None = self.optimizers()
        if not isinstance(optimizers, list) or len(optimizers) != 2:
            raise RuntimeError("HiFTNet training requires generator and discriminator optimizers")
        return optimizers[0], optimizers[1]

    def _extract_waveform(self, batch: LJSpeechBatch) -> torch.Tensor:
        # Reads the waveform tensor from the collated batch mapping.
        waveform: LJSpeechBatchValue | None = batch.get("waveform")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError(f"batch['waveform'] must be Tensor, got {type(waveform).__name__}")
        return waveform
