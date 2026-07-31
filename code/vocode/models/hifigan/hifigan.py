# This module:
# 1. Implements the HiFi-GAN vocoder family for the Study 1 reproduction
#    cohort: the V1, V2, and V3 reference configurations of the jik876
#    LJSpeech recipe plus the project-defined half-width V1 control
# 2. Trains adversarially against the multi-period and multi-scale
#    discriminators with the reference loss composition, and evaluates
#    through mel-conditioned waveform synthesis
#
# Harness contract (syntheticmind):
# - Subclasses the harness Module with automatic optimization disabled:
#   training_step drives the discriminator and generator optimizers
#   manually through the toggled-optimizer, manual-backward, and
#   optimizer-step surface, in the reference discriminator-first order
# - configure_optimizers declares paired AdamW optimizers with paired
#   per-epoch exponential schedulers, stepped explicitly at train-epoch
#   end; validation logs val_loss (mel L1) for checkpoint monitoring
# - predict_step and test_step return the synthesized_waveform mapping
#   consumed by the metric panel and timing callbacks
# - Checkpoint hooks stamp and verify the configuration dump, so a resume
#   or evaluation against a mismatched configuration fails loudly
#
# Design decisions:
# - Mel extraction runs under a disabled-autocast context so mixed
#   precision can never alter the validated conditioning and
#   reconstruction protocols
# - Conditioning and reconstruction use the two distinct HiFi-GAN mel
#   protocols of the reference recipe (band-limited versus full-band)
# - Waveform-length and mel-shape agreement are asserted rather than
#   silently cropped, because shape drift would indicate a protocol
#   violation upstream
#
# Author: Rahul Sawhney

from contextlib import nullcontext
from typing import ClassVar, Literal, override

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from syntheticmind.core.module import Module
from syntheticmind.core.optimizer import OptimizationConfiguration, SchedulerConfig
from syntheticmind.utilities.types import Batch, CheckpointDict, CheckpointValue, ModelOutput, StepOutput

from vocode.data.ljspeech_dataset import LJSpeechBatch, LJSpeechBatchValue
from vocode.losses.hifigan import HifiganLoss, HifiganLossConfig
from vocode.metrics.pesq import PesqConfig
from vocode.metrics.stoi import StoiConfig
from vocode.models.hifigan.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from vocode.models.hifigan.network import HifiganNetwork
from vocode.transforms.mel import MelConfig, MelSpectrogram

__all__: list[str] = ["Hifigan", "HifiganConfig"]


class HifiganConfig(BaseModel):
    # Frozen HiFi-GAN configuration: generator topology, adversarial
    # training hyperparameters, and the family's mel and metric protocols.
    # The named factories reproduce the reference V1, V2, and V3 recipes
    # and the project half-width control.
    #
    # Fields:
    #     input_mel_channels: Mel band count the generator's
    #         pre-convolution consumes; every reference recipe conditions
    #         on eighty bands.
    #     upsample_initial_channels: Channel width entering the first
    #         upsampling stage, halved at every subsequent stage. This is
    #         the only field separating V1, V2, and V3 by capacity.
    #     upsample_rates: Transposed-convolution stride of each upsampling
    #         stage. Their product is the total upsampling factor and
    #         equals the hop length of the conditioning protocol, so one
    #         mel frame becomes exactly one hop of waveform samples.
    #     upsample_kernel_sizes: Kernel width of each upsampling stage,
    #         paired positionally with ``upsample_rates``; the two tuples
    #         must have equal length or construction fails.
    #     resblock_kernel_sizes: Kernel widths of the parallel residual
    #         branches whose outputs the multi-receptive-field fusion
    #         averages after each upsampling stage.
    #     resblock_dilation_sizes: Dilation sets paired positionally with
    #         ``resblock_kernel_sizes``; the two tuples must have equal
    #         length, and each set must hold three dilations for
    #         ``resblock_kind`` ``"1"`` or two for ``"2"``.
    #     resblock_kind: Residual-block variant. ``"1"`` selects the
    #         paired dilated-then-refinement block of the V1 and V2
    #         recipes; ``"2"`` selects the lighter single-convolution
    #         block of V3.
    #     leaky_relu_slope: Negative slope of the per-stage activations
    #         and of the activations inside the residual blocks. The final
    #         activation before the post-convolution deliberately keeps
    #         the framework default slope, matching the reference
    #         generator.
    #     channel_multiplier: Uniform scale applied to the initial stage
    #         width, from which every later stage width is derived. The
    #         reference recipes use ``1.0``; the project half-width
    #         control uses ``0.5`` to halve capacity without touching the
    #         topology.
    #     conditioning_mel_protocol: Band-limited mel protocol the
    #         generator is conditioned on.
    #     reconstruction_mel_protocol: Full-band mel protocol the
    #         reconstruction loss term and the mel-error metric extract
    #         with. Keeping it distinct from the conditioning protocol is
    #         the reference behavior of this family.
    #     pesq_protocol: PESQ measurement protocol for this family.
    #     stoi_protocol: STOI measurement protocol for this family.
    #     learning_rate: Shared AdamW step size for the generator and
    #         discriminator optimizers. Default: ``0.0002``.
    #     adam_beta_1: First AdamW moment decay for both optimizers.
    #         Default: ``0.8``.
    #     adam_beta_2: Second AdamW moment decay for both optimizers.
    #         Default: ``0.99``.
    #     learning_rate_decay: Per-epoch multiplicative factor of the
    #         exponential schedules attached to both optimizers.
    #         Default: ``0.999``.
    #     loss_configuration: Component weights of the composite
    #         objective: mel reconstruction, feature matching, and the
    #         least-squares adversarial term.
    #         Default: ``HifiganLossConfig()``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    input_mel_channels: PositiveInt
    upsample_initial_channels: PositiveInt
    upsample_rates: tuple[int, ...]
    upsample_kernel_sizes: tuple[int, ...]
    resblock_kernel_sizes: tuple[int, ...]
    resblock_dilation_sizes: tuple[tuple[int, ...], ...]
    resblock_kind: Literal["1", "2"]
    leaky_relu_slope: PositiveFloat
    channel_multiplier: PositiveFloat
    conditioning_mel_protocol: MelConfig
    reconstruction_mel_protocol: MelConfig
    pesq_protocol: PesqConfig
    stoi_protocol: StoiConfig
    learning_rate: PositiveFloat = 0.0002
    adam_beta_1: PositiveFloat = 0.8
    adam_beta_2: PositiveFloat = 0.99
    learning_rate_decay: PositiveFloat = 0.999
    loss_configuration: HifiganLossConfig = HifiganLossConfig()

    @classmethod
    def v1(cls) -> HifiganConfig:
        # Builds the HiFi-GAN V1 configuration using the canonical LJSpeech protocol.
        # V1 is the full-width reference generator: four upsampling stages
        # multiplying to the two-hundred-fifty-six-sample hop, fused by
        # three parallel ResBlock1 branches per stage.
        #
        # Returns:
        #     The frozen reference configuration for this architecture.
        return cls(
            input_mel_channels=80,
            upsample_initial_channels=512,
            upsample_rates=(8, 8, 2, 2),
            upsample_kernel_sizes=(16, 16, 4, 4),
            resblock_kernel_sizes=(3, 7, 11),
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            resblock_kind="1",
            leaky_relu_slope=0.1,
            channel_multiplier=1.0,
            conditioning_mel_protocol=MelConfig.hifigan_conditioning(),
            reconstruction_mel_protocol=MelConfig.hifigan_reconstruction(),
            pesq_protocol=PesqConfig(),
            stoi_protocol=StoiConfig()
        )

    @classmethod
    def v2(cls) -> HifiganConfig:
        # Builds the HiFi-GAN V2 configuration using the canonical LJSpeech protocol.
        # V2 differs from V1 in generator width alone: the stage and
        # fusion topology, the residual-block variant, and every training
        # hyperparameter are identical, so the pair isolates capacity.
        #
        # Returns:
        #     The frozen reference configuration for this architecture.
        return cls(
            input_mel_channels=80,
            upsample_initial_channels=128,
            upsample_rates=(8, 8, 2, 2),
            upsample_kernel_sizes=(16, 16, 4, 4),
            resblock_kernel_sizes=(3, 7, 11),
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            resblock_kind="1",
            leaky_relu_slope=0.1,
            channel_multiplier=1.0,
            conditioning_mel_protocol=MelConfig.hifigan_conditioning(),
            reconstruction_mel_protocol=MelConfig.hifigan_reconstruction(),
            pesq_protocol=PesqConfig(),
            stoi_protocol=StoiConfig()
        )

    @classmethod
    def v3(cls) -> HifiganConfig:
        # Builds the HiFi-GAN V3 configuration using the canonical LJSpeech protocol.
        # V3 is the lightest reference recipe: it reaches the same
        # two-hundred-fifty-six-sample hop in three stages instead of
        # four and fuses with the two-dilation ResBlock2, which is why its
        # dilation sets carry two entries rather than three.
        #
        # Returns:
        #     The frozen reference configuration for this architecture.
        return cls(
            input_mel_channels=80,
            upsample_initial_channels=256,
            upsample_rates=(8, 8, 4),
            upsample_kernel_sizes=(16, 16, 8),
            resblock_kernel_sizes=(3, 5, 7),
            resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
            resblock_kind="2",
            leaky_relu_slope=0.1,
            channel_multiplier=1.0,
            conditioning_mel_protocol=MelConfig.hifigan_conditioning(),
            reconstruction_mel_protocol=MelConfig.hifigan_reconstruction(),
            pesq_protocol=PesqConfig(),
            stoi_protocol=StoiConfig()
        )

    @classmethod
    def half_width_v1(cls) -> HifiganConfig:
        # Builds the project-defined half-width HiFi-GAN V1 variant configuration.
        # This is a project control rather than a published recipe: it
        # reproduces V1 field for field and sets the channel multiplier to
        # one half, so the two configurations differ in exactly one value.
        #
        # The registry routes only the three published widths, so this
        # recipe is reachable solely by direct construction and no
        # evidence lane builds it. Half-width controls lie outside the
        # executed Study 1 cohort, which is the twelve Project-Trained
        # Configurations; this factory therefore carries no measured
        # result and supports no reported comparison.
        #
        # Returns:
        #     The frozen configuration of the project half-width control.
        return cls(
            input_mel_channels=80,
            upsample_initial_channels=512,
            upsample_rates=(8, 8, 2, 2),
            upsample_kernel_sizes=(16, 16, 4, 4),
            resblock_kernel_sizes=(3, 7, 11),
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            resblock_kind="1",
            leaky_relu_slope=0.1,
            channel_multiplier=0.5,
            conditioning_mel_protocol=MelConfig.hifigan_conditioning(),
            reconstruction_mel_protocol=MelConfig.hifigan_reconstruction(),
            pesq_protocol=PesqConfig(),
            stoi_protocol=StoiConfig()
        )


class Hifigan(Module):
    # HiFi-GAN vocoder module: transposed-convolution upsampling generator
    # with multi-receptive-field fusion, trained adversarially against the
    # multi-period and multi-scale discriminators under manual optimization.
    #
    # Integration: the module satisfies the vocode.models.vocoder.Vocoder
    # synthesis contract through its network member, its two mel-protocol
    # properties, and synthesize, so the metric panel and the complexity
    # profiler consume it without inheritance coupling. Only the generator
    # is exposed as network; the discriminators and the loss are private,
    # which is what keeps profiling measured on inference cost alone. The
    # discriminators exist solely for training and are not consulted on
    # any evaluation path.
    def __init__(self, configuration: HifiganConfig) -> None:
        # Disables automatic optimization for the two-optimizer adversarial
        # recipe and constructs the mel transforms, generator network,
        # both discriminators, and the composite loss.
        #
        # Args:
        #     configuration: The frozen recipe record. Every topology and
        #         hyperparameter value is read from it here, so the module
        #         holds no independent defaults and the configuration dump
        #         fully describes the constructed module.
        #
        # Note:
        #     Both discriminators are constructed at their reference
        #     settings and take no configuration input, because the
        #     reference recipe holds them fixed across V1, V2, and V3.
        super().__init__()
        self.automatic_optimization: bool = False
        self._configuration: HifiganConfig = configuration
        self._conditioning_mel_spectrogram: MelSpectrogram = MelSpectrogram(configuration.conditioning_mel_protocol)
        self._reconstruction_mel_spectrogram: MelSpectrogram = MelSpectrogram(configuration.reconstruction_mel_protocol)
        self.network: HifiganNetwork = HifiganNetwork(
            input_mel_channels=configuration.input_mel_channels,
            upsample_initial_channels=configuration.upsample_initial_channels,
            upsample_rates=configuration.upsample_rates,
            upsample_kernel_sizes=configuration.upsample_kernel_sizes,
            resblock_kernel_sizes=configuration.resblock_kernel_sizes,
            resblock_dilation_sizes=configuration.resblock_dilation_sizes,
            resblock_kind=configuration.resblock_kind,
            leaky_relu_slope=configuration.leaky_relu_slope,
            channel_multiplier=configuration.channel_multiplier
        )
        self._period_discriminator: MultiPeriodDiscriminator = MultiPeriodDiscriminator()
        self._scale_discriminator: MultiScaleDiscriminator = MultiScaleDiscriminator()
        self._loss: HifiganLoss = HifiganLoss(configuration.loss_configuration)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Synthesizes a waveform batch from conditioning mels through the
        # generator network.
        return self.network(mel)

    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        # Generates waveform output from the architecture-specific acoustic representation.
        # This is the Vocoder protocol entry point the measurement stack
        # calls; for this family it is exactly the generator forward, and
        # the returned waveform keeps the generator's [batch, 1, samples]
        # layout rather than the squeezed layout the prediction step
        # emits.
        return self.forward(mel)

    @property
    def mel_protocol(self) -> MelConfig:
        # Returns the mel-spectrogram protocol required by this architecture family.
        # For HiFi-GAN this is the band-limited conditioning protocol,
        # which differs from the protocol the metrics measure with.
        return self._configuration.conditioning_mel_protocol

    @property
    def metric_mel_protocol(self) -> MelConfig:
        # Returns the full-band protocol the mel-error metric extracts with.
        return self._configuration.reconstruction_mel_protocol

    @property
    def configuration(self) -> HifiganConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # One adversarial training step in the reference order: extract the
        # conditioning and reconstruction mels in full precision, synthesize
        # once, train the discriminators on the detached fake, then train
        # the generator through the full loss composition; all loss
        # components are logged and the detached generator loss is returned
        # for reporting.
        #
        # Args:
        #     batch: The collated LJSpeech mapping; only its waveform
        #         entry is read, because both mels are derived from the
        #         reference waveform rather than taken from the batch.
        #     batch_idx: Index of this batch within the epoch. It is not
        #         consulted: the step performs the same update sequence on
        #         every batch.
        #
        # Returns:
        #     A mapping whose ``"loss"`` is the detached generator loss.
        #     Under manual optimization the loop performs no backward on
        #     it, so the value is reporting-only; both optimizers have
        #     already stepped by the time this returns.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping, or its waveform
        #         entry is not a tensor.
        #     ValueError: If the waveform cannot be normalized to
        #         [batch, time], or the reference and synthesized
        #         waveforms disagree in length, or the two reconstruction
        #         mels disagree in shape.
        #     RuntimeError: If the module is not attached to a trainer
        #         holding the generator and discriminator optimizer pair.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        generator_optimizer, discriminator_optimizer = self._require_gan_optimizers()
        reference_waveform: torch.Tensor = self._prepare_reference_waveform(batch)
        # Both mels come from the reference waveform under one disabled-autocast
        # window, and each is promoted to a batch axis when the transform
        # returns an unbatched result.
        with self._full_precision_context():
            generator_input_mel: torch.Tensor = self._conditioning_mel_spectrogram(reference_waveform.float())
            if generator_input_mel.ndim == 2:
                generator_input_mel: torch.Tensor = generator_input_mel.unsqueeze(0)
            reference_reconstruction_mel: torch.Tensor = self._reconstruction_mel_spectrogram(
                reference_waveform.float()
            )
            if reference_reconstruction_mel.ndim == 2:
                reference_reconstruction_mel: torch.Tensor = reference_reconstruction_mel.unsqueeze(0)
        # One synthesis serves both updates; the discriminator update detaches
        # it internally, so the generator graph survives into its own update.
        synthesized_waveform: torch.Tensor = self.network(generator_input_mel)
        real_discriminator_waveform: torch.Tensor = self._to_discriminator_waveform(reference_waveform)
        fake_discriminator_waveform: torch.Tensor = self._to_discriminator_waveform(synthesized_waveform)
        real_discriminator_waveform, fake_discriminator_waveform = self._align_waveform_pair(
            real_discriminator_waveform,
            fake_discriminator_waveform
        )
        discriminator_loss, discriminator_components = self._run_discriminator_training_step(
            discriminator_optimizer=discriminator_optimizer,
            real_waveform=real_discriminator_waveform,
            fake_waveform=fake_discriminator_waveform
        )
        generator_loss, generator_components = self._run_generator_training_step(
            generator_optimizer=generator_optimizer,
            reference_reconstruction_mel=reference_reconstruction_mel,
            real_waveform=real_discriminator_waveform,
            fake_waveform=fake_discriminator_waveform
        )
        # train_loss mirrors the generator loss so the monitored series is
        # comparable across families whose objectives differ; every loss
        # component is additionally published under its own prefixed name.
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
        # Validates by full-band mel L1 between reference and synthesis,
        # logged as val_loss for checkpoint and stopping policies. The
        # discriminators are not consulted here: an adversarial score
        # tracks the current discriminator rather than synthesis quality,
        # so it cannot serve as a checkpoint-selection monitor. Mels are
        # extracted at the ambient precision of the evaluation pass rather
        # than through the full-precision window the training step uses.
        #
        # Args:
        #     batch: The collated LJSpeech mapping; only its waveform
        #         entry is read.
        #     batch_idx: Index of this batch within the validation pass.
        #         It is not consulted.
        #
        # Returns:
        #     A mapping whose ``"loss"`` is the full-band mel L1, the same
        #     value published as ``val_loss`` and ``val_mel_l1``.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping, or its waveform
        #         entry is not a tensor.
        #     ValueError: If the waveform cannot be normalized to
        #         [batch, time], or the reference and synthesized
        #         reconstruction mels disagree in shape, which would mean
        #         the protocols drifted apart upstream.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        reference_waveform: torch.Tensor = self._prepare_reference_waveform(batch)
        generator_input_mel: torch.Tensor = self._conditioning_mel_spectrogram(reference_waveform)
        if generator_input_mel.ndim == 2:
            generator_input_mel: torch.Tensor = generator_input_mel.unsqueeze(0)
        reference_reconstruction_mel: torch.Tensor = self._reconstruction_mel_spectrogram(reference_waveform)
        if reference_reconstruction_mel.ndim == 2:
            reference_reconstruction_mel: torch.Tensor = reference_reconstruction_mel.unsqueeze(0)
        synthesized_waveform: torch.Tensor = self.network(generator_input_mel)
        synthesized_reconstruction_mel: torch.Tensor = self._reconstruction_mel_spectrogram(
            self._prepare_waveform_for_mel(synthesized_waveform)
        )
        if synthesized_reconstruction_mel.ndim == 2:
            synthesized_reconstruction_mel: torch.Tensor = synthesized_reconstruction_mel.unsqueeze(0)
        self._assert_matching_mel_shape(
            reference_reconstruction_mel,
            synthesized_reconstruction_mel,
            "validation mel reconstruction"
        )
        validation_loss: torch.Tensor = torch.nn.functional.l1_loss(
            synthesized_reconstruction_mel,
            reference_reconstruction_mel
        )
        self.log("val_loss", validation_loss)
        self.log("val_mel_l1", validation_loss)
        return {"loss": validation_loss}

    @override
    def predict_step(self, batch: Batch, batch_idx: int) -> ModelOutput:
        # Synthesizes from the batch waveform's own conditioning mel and
        # returns the synthesized_waveform mapping the measurement stack
        # consumes. Conditioning is analysis-by-synthesis: the mel is
        # extracted from the reference waveform of the batch itself, so
        # the reference and the synthesis are directly comparable.
        #
        # Args:
        #     batch: The collated LJSpeech mapping; only its waveform
        #         entry is read. An unbatched waveform is promoted to a
        #         batch of one before extraction.
        #     batch_idx: Index of this batch within the pass. It is not
        #         consulted.
        #
        # Returns:
        #     A mapping under the key ``"synthesized_waveform"``. The
        #     generator's channel axis is squeezed out, so the value is
        #     [batch, samples] rather than the [batch, 1, samples] layout
        #     synthesize returns.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping, or its waveform
        #         entry is not a tensor.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        waveform: torch.Tensor = self._extract_waveform(batch)
        if waveform.ndim == 1:
            waveform: torch.Tensor = waveform.unsqueeze(0)
        mel: torch.Tensor = self._conditioning_mel_spectrogram(waveform)
        if mel.ndim == 2:
            mel: torch.Tensor = mel.unsqueeze(0)
        synthesized: torch.Tensor = self.network(mel)
        return {"synthesized_waveform": synthesized.squeeze(1) if synthesized.ndim == 3 else synthesized}

    @override
    def test_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Delegates to predict_step and validates the output contract, so
        # test evaluation measures exactly the prediction path. The
        # returned mapping is rebuilt with the single verified key rather
        # than forwarded, so no unchecked entry can reach the metric
        # panel.
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
        #         tensor, in addition to the guards predict_step applies
        #         to the batch itself.
        prediction: ModelOutput = self.predict_step(batch, batch_idx)
        if not isinstance(prediction, dict):
            raise TypeError(f"predict_step must return a mapping, got {type(prediction).__name__}")
        synthesized_waveform: ModelOutput | None = prediction.get("synthesized_waveform")
        if not isinstance(synthesized_waveform, torch.Tensor):
            raise TypeError("predict_step output must contain Tensor key synthesized_waveform")
        return {"synthesized_waveform": synthesized_waveform}

    @override
    def on_train_epoch_end(self) -> None:
        # Steps the paired generator and discriminator exponential schedulers once per epoch.
        # Stepping is explicit here because manual optimization leaves
        # schedule advancement to the module; both resulting rates are
        # logged so the decay is visible in the run record.
        #
        # Raises:
        #     RuntimeError: If the trainer does not hold exactly the two
        #         declared schedulers, which would mean the optimization
        #         declaration and the module disagree.
        schedulers: torch.optim.lr_scheduler.LRScheduler | list[torch.optim.lr_scheduler.LRScheduler] | None = self.lr_schedulers()
        if not isinstance(schedulers, list) or len(schedulers) != 2:
            raise RuntimeError("HiFi-GAN training requires generator and discriminator schedulers")
        for scheduler in schedulers:
            scheduler.step()
        self.log("learning_rate_generator", float(schedulers[0].get_last_lr()[0]))
        self.log("learning_rate_discriminator", float(schedulers[1].get_last_lr()[0]))

    @override
    def on_save_checkpoint(self, checkpoint: CheckpointDict) -> None:
        # Adds reproducibility metadata while tensor state remains owned by the shared checkpoint harness.
        # Only the JSON configuration dump is stamped; parameters,
        # optimizer state, and loop counters are written by the harness.
        #
        # Args:
        #     checkpoint: The checkpoint payload being assembled, mutated
        #         in place under the ``"hifigan_configuration"`` key.
        checkpoint["hifigan_configuration"] = self._configuration.model_dump(mode="json")

    @override
    def on_load_checkpoint(self, checkpoint: CheckpointDict) -> None:
        # Rejects incompatible checkpoint resumes before corrupted evidence can be written.
        # The hook is a compatibility gate, never a restore path: it
        # neither adopts the stored recipe nor backfills a stamp, so on
        # success the module keeps exactly the configuration it was
        # constructed with. A checkpoint written before the stamp existed
        # carries no such key and is accepted unchanged.
        #
        # Args:
        #     checkpoint: The checkpoint payload being loaded, inspected
        #         but not modified.
        #
        # Raises:
        #     ValueError: If the stamped configuration differs in any
        #         field from the current one. This is what stops a
        #         half-width control from resuming off a full-width run,
        #         and the reverse.
        checkpoint_configuration: CheckpointValue | None = checkpoint.get("hifigan_configuration")
        if checkpoint_configuration is None:
            return
        current_configuration: dict[str, object] = self._configuration.model_dump(mode="json")
        if checkpoint_configuration != current_configuration:
            raise ValueError("Checkpoint HiFi-GAN configuration does not match the current module configuration.")

    @override
    def configure_optimizers(self) -> OptimizationConfiguration:
        # Declares the optimizers and schedulers required by this architecture.
        # Both optimizers are constructed here rather than declared as
        # configuration records, because their parameter partition is a
        # property of this module: the generator optimizer owns exactly
        # the network parameters, and the discriminator optimizer owns the
        # period and scale ensembles together. Declaration order is
        # load-bearing, since the training step unpacks the pair by
        # position and the schedulers are paired to the optimizers by
        # position as well.
        #
        # Returns:
        #     An OptimizationConfiguration holding the two live AdamW
        #     optimizers, generator first, each paired with a per-epoch
        #     exponential schedule at the configured decay. The schedules
        #     are declarations; this module steps them itself at
        #     train-epoch end because it optimizes manually.
        generator_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self._configuration.learning_rate,
            betas=(self._configuration.adam_beta_1, self._configuration.adam_beta_2)
        )
        discriminator_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            list(self._period_discriminator.parameters()) + list(self._scale_discriminator.parameters()),
            lr=self._configuration.learning_rate,
            betas=(self._configuration.adam_beta_1, self._configuration.adam_beta_2)
        )
        return OptimizationConfiguration(
            optimizer=[generator_optimizer, discriminator_optimizer],
            scheduler=[
                SchedulerConfig(
                    name="exponential",
                    interval="epoch",
                    gamma=self._configuration.learning_rate_decay
                ),
                SchedulerConfig(
                    name="exponential",
                    interval="epoch",
                    gamma=self._configuration.learning_rate_decay
                )
            ]
        )

    def _run_discriminator_training_step(
        self,
        discriminator_optimizer: torch.optim.Optimizer,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Discriminator update: both discriminators judge the real waveform
        # against the detached synthesis, and the least-squares loss steps
        # only the discriminator parameters under the optimizer toggle.
        # Detaching the synthesis is what keeps this update from reaching
        # generator parameters; the toggle is the second, independent
        # guard, restoring every requires_grad flag on exit.
        #
        # Args:
        #     discriminator_optimizer: The optimizer owning both
        #         discriminator ensembles.
        #     real_waveform: The reference waveform in the
        #         [batch, 1, samples] discriminator layout.
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
            real_scale_logits, fake_scale_logits, _, _ = self._scale_discriminator(
                real_waveform,
                fake_waveform.detach()
            )
            discriminator_loss, components = self._loss.compute_discriminator_loss(
                real_period_logits=real_period_logits,
                fake_period_logits=fake_period_logits,
                real_scale_logits=real_scale_logits,
                fake_scale_logits=fake_scale_logits
            )
            self.optimizer_zero_grad(discriminator_optimizer)
            self.manual_backward(discriminator_loss)
            self.optimizer_step(discriminator_optimizer)
        return discriminator_loss.detach(), components

    def _run_generator_training_step(
        self,
        generator_optimizer: torch.optim.Optimizer,
        reference_reconstruction_mel: torch.Tensor,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Generator update: adversarial logits and feature maps from both
        # discriminators combine with the full-band mel reconstruction loss,
        # stepping only the generator parameters under the optimizer toggle.
        # The discriminators are re-run on the attached synthesis, because
        # the generator needs gradients flowing back through them; the
        # real-side logits are discarded immediately since only the
        # feature maps of the real branch enter the objective.
        #
        # Args:
        #     generator_optimizer: The optimizer owning the generator
        #         network parameters.
        #     reference_reconstruction_mel: Full-band mel of the reference
        #         waveform, the target of the reconstruction term.
        #     real_waveform: The reference waveform in discriminator
        #         layout.
        #     fake_waveform: The synthesis in discriminator layout, kept
        #         attached so the update reaches the generator.
        #
        # Returns:
        #     The detached generator loss paired with its named scalar
        #     components for logging.
        #
        # Raises:
        #     ValueError: If the reference and synthesized reconstruction
        #         mels disagree in shape.
        with self.toggled_optimizer(generator_optimizer):
            real_period_logits, fake_period_logits, real_period_features, fake_period_features = (
                self._period_discriminator(real_waveform, fake_waveform)
            )
            real_scale_logits, fake_scale_logits, real_scale_features, fake_scale_features = (
                self._scale_discriminator(real_waveform, fake_waveform)
            )
            del real_period_logits, real_scale_logits
            with self._full_precision_context():
                synthesized_reconstruction_mel: torch.Tensor = self._reconstruction_mel_spectrogram(
                    self._prepare_waveform_for_mel(fake_waveform).float()
                )
                if synthesized_reconstruction_mel.ndim == 2:
                    synthesized_reconstruction_mel: torch.Tensor = synthesized_reconstruction_mel.unsqueeze(0)
            self._assert_matching_mel_shape(
                reference_reconstruction_mel,
                synthesized_reconstruction_mel,
                "training generator mel reconstruction"
            )
            generator_loss, components = self._loss.compute_generator_loss(
                reference_mel=reference_reconstruction_mel,
                synthesized_mel=synthesized_reconstruction_mel,
                fake_period_logits=fake_period_logits,
                fake_scale_logits=fake_scale_logits,
                real_period_features=real_period_features,
                fake_period_features=fake_period_features,
                real_scale_features=real_scale_features,
                fake_scale_features=fake_scale_features
            )
            self.optimizer_zero_grad(generator_optimizer)
            self.manual_backward(generator_loss)
            self.optimizer_step(generator_optimizer)
        return generator_loss.detach(), components

    def _full_precision_context(self) -> torch.autocast | nullcontext[None]:
        # Keeps mel-spectrogram extraction in float32 so mixed-precision training cannot alter
        # the validated conditioning and reconstruction protocols.
        # torch.autocast accepts only device types it implements, so any
        # other placement falls back to a null context; the callers pair
        # this window with an explicit float cast, which carries the
        # precision guarantee even where autocast cannot be disabled.
        device_type: str = self.device.type
        if device_type in ("cuda", "cpu"):
            return torch.autocast(device_type=device_type, enabled=False)
        return nullcontext()

    def _require_gan_optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        # Returns the generator and discriminator optimizers in declaration
        # order, failing loudly when the pair is absent.
        #
        # Raises:
        #     RuntimeError: If the module is detached from a trainer, or
        #         the trainer holds a single optimizer rather than the
        #         declared pair. A detached module reads as no optimizers
        #         at all, which is why calling the training step outside a
        #         fit run fails here rather than partway through an
        #         update.
        optimizers: torch.optim.Optimizer | list[torch.optim.Optimizer] | None = self.optimizers()
        if not isinstance(optimizers, list) or len(optimizers) != 2:
            raise RuntimeError("HiFi-GAN training requires generator and discriminator optimizers")
        return optimizers[0], optimizers[1]

    def _prepare_reference_waveform(self, batch: LJSpeechBatch) -> torch.Tensor:
        # Reads the reference waveform and normalizes it to [batch, time].
        # A single-sample waveform is promoted to a batch of one; anything
        # that is still not two-dimensional is a batch-shape error rather
        # than something to reshape silently.
        #
        # Raises:
        #     TypeError: If the waveform entry is missing or not a tensor.
        #     ValueError: If the waveform cannot be read as [batch, time].
        waveform: torch.Tensor = self._extract_waveform(batch)
        if waveform.ndim == 1:
            waveform: torch.Tensor = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform shape [batch, time], got {tuple(waveform.shape)}")
        return waveform

    def _to_discriminator_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        # Normalizes a waveform to the [batch, 1, time] layout the
        # discriminators consume. A three-dimensional input is already in
        # that layout and passes through, which is how the generator's own
        # output reaches the ensembles unchanged.
        #
        # Raises:
        #     ValueError: If the waveform carries more than three
        #         dimensions, since no unambiguous channel axis exists.
        if waveform.ndim == 1:
            return waveform.unsqueeze(0).unsqueeze(0)
        if waveform.ndim == 2:
            return waveform.unsqueeze(1)
        if waveform.ndim == 3:
            return waveform
        raise ValueError(f"Expected waveform with 1, 2, or 3 dimensions, got {tuple(waveform.shape)}")

    def _to_mel_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        # Normalizes a waveform to the [batch, time] layout the mel
        # transforms consume, dropping the generator's channel axis.
        #
        # Raises:
        #     ValueError: If the waveform carries more than three
        #         dimensions.
        if waveform.ndim == 3:
            return waveform.squeeze(1)
        if waveform.ndim == 2:
            return waveform
        if waveform.ndim == 1:
            return waveform.unsqueeze(0)
        raise ValueError(f"Expected waveform with 1, 2, or 3 dimensions, got {tuple(waveform.shape)}")

    def _prepare_waveform_for_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        # Normalizes the layout and right-pads ultra-short syntheses to the
        # minimum length the reconstruction STFT accepts. Below one
        # transform size plus one sample the centered STFT cannot form a
        # frame at all, so padding here converts a hard transform failure
        # into a well-defined short analysis; material of ordinary
        # training length is returned untouched.
        mel_waveform: torch.Tensor = self._to_mel_waveform(waveform)
        minimum_samples: int = self._configuration.reconstruction_mel_protocol.n_fft + 1
        if mel_waveform.shape[-1] >= minimum_samples:
            return mel_waveform
        padding_amount: int = minimum_samples - mel_waveform.shape[-1]
        return torch.nn.functional.pad(mel_waveform, (0, padding_amount))

    def _align_waveform_pair(
        self,
        first_waveform: torch.Tensor,
        second_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Aligns generated and reference tensors before loss or metric computation.
        # For this family alignment is a verification rather than a
        # repair: equal lengths are returned unchanged and unequal lengths
        # raise, because the reference recipe upsamples exactly to the mel
        # hop and any drift would mean a protocol violation upstream. The
        # sibling families crop to the shorter tensor instead; HiFi-GAN
        # deliberately does not.
        #
        # Raises:
        #     ValueError: If the two waveforms differ in sample count. The
        #         message reports both lengths.
        if first_waveform.shape[-1] != second_waveform.shape[-1]:
            raise ValueError(
                f"HiFi-GAN strict training requires equal waveform lengths before discriminator loss: "
                f"reference={first_waveform.shape[-1]}, candidate={second_waveform.shape[-1]}"
            )
        return first_waveform, second_waveform

    def _assert_matching_mel_shape(
        self,
        reference_mel: torch.Tensor,
        candidate_mel: torch.Tensor,
        context: str
    ) -> None:
        # Preserves strict reproduction invariants by failing on mel protocol drift.
        # Band count and frame count must both agree, since the
        # reconstruction term is an elementwise L1 whose broadcast would
        # otherwise silently compare misaligned material.
        #
        # Args:
        #     reference_mel: Mel of the reference waveform.
        #     candidate_mel: Mel of the synthesized waveform.
        #     context: Short label naming the call site, carried into the
        #         failure message so training and validation drift are
        #         distinguishable.
        #
        # Raises:
        #     ValueError: If the two shapes differ. The message reports
        #         both shapes alongside the context label.
        if reference_mel.shape != candidate_mel.shape:
            raise ValueError(
                f"HiFi-GAN {context} requires equal mel shapes: "
                f"reference={tuple(reference_mel.shape)}, candidate={tuple(candidate_mel.shape)}"
            )

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
