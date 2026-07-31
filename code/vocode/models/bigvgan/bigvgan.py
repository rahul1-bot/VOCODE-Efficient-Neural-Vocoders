# This module:
# 1. Implements the BigVGAN-base vocoder for the Study 1 reproduction
#    cohort, following the NVIDIA 24 kHz 100-band recipe: a HiFi-GAN-style
#    upsampling generator whose residual stacks use anti-aliased periodic
#    (snake) activations
# 2. Trains adversarially against the discriminator ensembles with
#    per-step exponential schedulers and an initial discriminator freeze
#    window
# 3. Defines BigvganConfig, the frozen experiment record pinning the
#    topology, both discriminator ensembles, the optimization settings, and
#    the two mel protocols
#
# Harness contract (syntheticmind):
# - Subclasses the harness Module with automatic optimization disabled;
#   training_step synthesizes once, runs the discriminator update on the
#   detached synthesis and the generator update on the shared synthesis,
#   and advances the schedulers explicitly
# - Validation logs val_loss for checkpoint monitoring; predict_step and
#   test_step return the synthesized_waveform mapping the measurement stack
#   consumes
# - No checkpoint hook is overridden, so state capture is exactly the
#   harness default: the module's own state dictionary and nothing else
#
# Design decisions:
# - The snakebeta activation with logscale parameters follows the
#   published base configuration; the activation is part of the
#   architecture identity, not a tunable
# - The discriminator stays frozen for the configured initial steps so
#   the generator establishes a signal before adversarial pressure
#   begins, per the training policy
# - Mel extraction stays in float32 under a disabled-autocast context,
#   preserving the validated 24 kHz 100-band conditioning protocol
# - The schedulers are advanced inside training_step rather than at epoch
#   end, because the published decay factor is calibrated per optimizer
#   step and would barely move across whole epochs
#
# Author: Rahul Sawhney

from contextlib import nullcontext
from typing import ClassVar, Literal, override

import torch
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveFloat, PositiveInt

from syntheticmind.core.module import Module
from syntheticmind.core.optimizer import OptimizationConfiguration
from syntheticmind.utilities.types import Batch, ModelOutput, StepOutput

from vocode.data.ljspeech_dataset import LJSpeechBatch, LJSpeechBatchValue
from vocode.losses.bigvgan import BigvganLoss, BigvganLossConfig
from vocode.metrics.pesq import PesqConfig
from vocode.metrics.stoi import StoiConfig
from vocode.models.bigvgan.discriminator import BigvganMultiPeriodDiscriminator, BigvganMultiResolutionDiscriminator
from vocode.models.bigvgan.network import BigvganNetwork
from vocode.transforms.mel import MelConfig, MelSpectrogram

__all__: list[str] = ["Bigvgan", "BigvganConfig"]


class BigvganConfig(BaseModel):
    # Frozen experiment record of one BigVGAN run. Generator topology,
    # discriminator geometry, optimization, and both mel protocols are pinned
    # together, so a run is fully described by this record and no setting can
    # drift between construction and training.
    #
    # This record is the architecture recipe of the BigVGAN-base
    # Project-Trained Configuration. Fitting it from random initialization
    # under the registered budget produces the Retained Project Checkpoint, the
    # terminal durable state admitted by the checkpoint gate, and every
    # reported measurement of this configuration, including every deployment
    # transformation applied afterwards, is inherited from that one state.
    #
    # Fields:
    #     input_mel_channels: Band count of the conditioning mel and the
    #         input width of the generator's conditioning convolution.
    #     upsample_initial_channel: Width that convolution lifts to; each
    #         upsampling stage halves it.
    #     upsample_rates: Time expansion of each stage; their product is the
    #         samples synthesized per conditioning frame.
    #     upsample_kernel_sizes: Kernel of each stage's transposed
    #         convolution, paired with the rates by position.
    #     resblock_kernel_sizes: Kernel of each residual block within a
    #         stage; its length is the blocks per stage.
    #     resblock_dilation_sizes: Dilation tuple of each of those blocks,
    #         paired with the kernel sizes by position.
    #     resblock: Closed literal selecting the wide (``"1"``) or narrow
    #         (``"2"``) residual variant.
    #     activation: Closed literal selecting the periodic activation;
    #         architecture identity rather than a tunable.
    #     snake_logscale: Whether the activation stores its parameters as
    #         logarithms.
    #     use_bias_at_final: Whether the output projection carries a bias.
    #         Default: ``True``.
    #     use_tanh_at_final: Selects the output bounding between a hyperbolic
    #         tangent and a hard clamp. Default: ``True``.
    #     resolutions: The three ``(n_fft, hop_length, win_length)`` triples
    #         of the multi-resolution discriminator.
    #     mpd_reshapes: Sample strides the multi-period discriminator folds
    #         the waveform at, one sub-discriminator each.
    #     use_spectral_norm: Selects spectral over weight normalization
    #         across both discriminator ensembles.
    #     discriminator_channel_multiplier: Capacity scale applied to every
    #         convolution width in both ensembles.
    #     freeze_discriminator_steps: Optimizer steps during which the
    #         discriminators are neither updated nor consulted, so the
    #         generator trains as a pure mel regressor first. Zero disables
    #         the window. Default: ``0``.
    #     gradient_clip_norm: Global gradient-norm ceiling applied separately
    #         to the generator and to the discriminator ensembles.
    #         Default: ``1000.0``.
    #     learning_rate: Initial rate of both AdamW optimizers.
    #         Default: ``0.0001``.
    #     adam_beta_1: First AdamW moment coefficient. Default: ``0.8``.
    #     adam_beta_2: Second AdamW moment coefficient. Default: ``0.99``.
    #     learning_rate_decay: Gamma of both exponential schedules, applied
    #         once per optimizer step. Default: ``0.9999996``.
    #     mel_protocol: Conditioning mel protocol the generator is driven by.
    #     reconstruction_mel_protocol: Mel protocol the reconstruction loss
    #         and the validation error are measured under, and the protocol
    #         published to the measurement stack.
    #     pesq_protocol: PESQ measurement protocol recorded for this run.
    #     stoi_protocol: STOI measurement protocol recorded for this run.
    #     loss_configuration: Term weights of the BigVGAN loss composition.
    #         Default: ``BigvganLossConfig()``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    input_mel_channels: PositiveInt
    upsample_initial_channel: PositiveInt
    upsample_rates: tuple[int, ...]
    upsample_kernel_sizes: tuple[int, ...]
    resblock_kernel_sizes: tuple[int, ...]
    resblock_dilation_sizes: tuple[tuple[int, ...], ...]
    resblock: Literal["1", "2"]
    activation: Literal["snake", "snakebeta"]
    snake_logscale: bool
    use_bias_at_final: bool = True
    use_tanh_at_final: bool = True
    resolutions: tuple[tuple[int, int, int], ...]
    mpd_reshapes: tuple[int, ...]
    use_spectral_norm: bool
    discriminator_channel_multiplier: PositiveFloat
    freeze_discriminator_steps: NonNegativeInt = 0
    gradient_clip_norm: PositiveFloat = 1000.0
    learning_rate: PositiveFloat = 0.0001
    adam_beta_1: PositiveFloat = 0.8
    adam_beta_2: PositiveFloat = 0.99
    learning_rate_decay: PositiveFloat = 0.9999996
    mel_protocol: MelConfig
    reconstruction_mel_protocol: MelConfig
    pesq_protocol: PesqConfig
    stoi_protocol: StoiConfig
    loss_configuration: BigvganLossConfig = BigvganLossConfig()

    @classmethod
    def nvidia_base_24khz_100band(cls) -> BigvganConfig:
        # Builds the BigVGAN-base 24 kHz 100-band configuration aligned with NVIDIA reference settings.
        #
        # This is the only entry point that reproduces the published anchor:
        # the topology stated here is what the released generator was trained
        # at, so a strict author load succeeds against a network built from
        # this record and fails against any other. Fields not listed take their
        # declared defaults, which already match the reference, including a
        # freeze window of zero.
        return cls(
            input_mel_channels=100,
            upsample_initial_channel=512,
            upsample_rates=(8, 8, 2, 2),
            upsample_kernel_sizes=(16, 16, 4, 4),
            resblock_kernel_sizes=(3, 7, 11),
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            resblock="1",
            activation="snakebeta",
            snake_logscale=True,
            resolutions=((1024, 120, 600), (2048, 240, 1200), (512, 50, 240)),
            mpd_reshapes=(2, 3, 5, 7, 11),
            use_spectral_norm=False,
            discriminator_channel_multiplier=1.0,
            mel_protocol=MelConfig.bigvgan_nvidia_base_24khz_100band(),
            reconstruction_mel_protocol=MelConfig.bigvgan_nvidia_base_reconstruction_24khz_100band(),
            pesq_protocol=PesqConfig(),
            stoi_protocol=StoiConfig()
        )


class Bigvgan(Module):
    # The BigVGAN vocoder as a harness module. It owns the generator network,
    # both discriminator ensembles, the loss composition, and the two mel
    # transforms, and drives adversarial training itself because automatic
    # optimization cannot express a two-optimizer scheme.
    #
    # One training step synthesizes exactly once and spends that synthesis
    # twice. The discriminator update comes first and judges the reference
    # against the detached synthesis, so no gradient reaches the generator
    # through it. The generator update then re-runs both ensembles on the
    # attached synthesis, meaning the generator is scored by the discriminator
    # weights that update produced. Both updates run inside the harness
    # optimizer toggle, which restricts requires_grad to the stepping
    # optimizer's parameters, so neither can leave gradients in the other's
    # weights. Both schedules are advanced once per step at the end.
    #
    # The freeze window is the one place the step's shape changes. While the
    # global step is below the configured threshold, the discriminator update
    # returns immediately without touching its optimizer, and the generator is
    # trained on the weighted mel reconstruction term alone; the critics
    # therefore stay at their initialization and contribute nothing. At the
    # threshold both halves switch to the full adversarial composition in the
    # same step. The purpose is to keep a randomly initialized critic from
    # dominating a generator that has not yet learned to produce anything
    # coherent.
    #
    # Integration: the module is constructed through the model registry from a
    # BigvganConfig, and the published NVIDIA release is loaded onto its
    # network attribute by vocode.models.bigvgan.weights.BigvganWeights. It
    # satisfies the vocode.models.vocoder.Vocoder protocol through network,
    # mel_protocol, metric_mel_protocol, and synthesize, so the metric and
    # profiling components consume it without knowing its architecture.
    def __init__(self, configuration: BigvganConfig) -> None:
        # Builds the generator, both discriminator ensembles, the loss
        # composition, and the two mel transforms from one frozen record, and
        # disables automatic optimization before anything else so the trainer
        # validates this module against the manual-optimization contract.
        #
        # Args:
        #     configuration: Frozen experiment record; it is retained as-is and
        #         republished through the configuration property, so the run is
        #         reproducible from the module alone.
        super().__init__()
        self.automatic_optimization: bool = False
        self._configuration: BigvganConfig = configuration
        self._mel_spectrogram: MelSpectrogram = MelSpectrogram(configuration.mel_protocol)
        self._reconstruction_mel_spectrogram: MelSpectrogram = MelSpectrogram(configuration.reconstruction_mel_protocol)
        self.network: BigvganNetwork = BigvganNetwork(
            num_mels=configuration.input_mel_channels,
            upsample_initial_channel=configuration.upsample_initial_channel,
            upsample_rates=configuration.upsample_rates,
            upsample_kernel_sizes=configuration.upsample_kernel_sizes,
            resblock_kernel_sizes=configuration.resblock_kernel_sizes,
            resblock_dilation_sizes=configuration.resblock_dilation_sizes,
            resblock=configuration.resblock,
            activation=configuration.activation,
            snake_logscale=configuration.snake_logscale,
            use_bias_at_final=configuration.use_bias_at_final,
            use_tanh_at_final=configuration.use_tanh_at_final
        )
        self._period_discriminator: BigvganMultiPeriodDiscriminator = BigvganMultiPeriodDiscriminator(
            periods=configuration.mpd_reshapes,
            channel_multiplier=configuration.discriminator_channel_multiplier,
            use_spectral_norm=configuration.use_spectral_norm
        )
        self._resolution_discriminator: BigvganMultiResolutionDiscriminator = BigvganMultiResolutionDiscriminator(
            resolutions=configuration.resolutions,
            channel_multiplier=configuration.discriminator_channel_multiplier,
            use_spectral_norm=configuration.use_spectral_norm
        )
        self._loss: BigvganLoss = BigvganLoss(configuration.loss_configuration)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Synthesizes a waveform batch from conditioning mels through the
        # generator network.
        return self.network(mel)

    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        # Generates waveform output from the architecture-specific acoustic representation.
        # This is the Vocoder protocol's synthesis entry point and is kept
        # distinct from forward so the measurement stack depends on the
        # protocol rather than on nn.Module's call convention.
        return self.forward(mel)

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # One adversarial training step: single synthesis shared by both
        # updates (detached for the discriminator), followed by the explicit
        # scheduler advance; components are logged and the detached
        # generator loss returned for reporting.
        #
        # The waveform pair handed to the discriminators is truncated to the
        # shorter of the two, and the full-band reference mel of the
        # reconstruction term is extracted before either update so the two
        # halves share one target.
        #
        # Args:
        #     batch: Collated LJSpeech mapping; only the waveform entry is
        #         read, and the conditioning mel is derived from it rather
        #         than taken from the batch, so training and inference share
        #         one extraction path.
        #     batch_idx: Index of this batch within the epoch; unused, because
        #         the step's shape is governed by the global step and not by
        #         position within an epoch.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping, or its waveform entry
        #         is not a tensor.
        #     ValueError: If the waveform cannot be normalized to
        #         ``[batch, time]``.
        #     RuntimeError: If the trainer has not materialized both
        #         optimizers, or both schedulers.
        #
        # Returns:
        #     A mapping whose ``"loss"`` is the detached generator loss.
        #     Backward and stepping have already happened here, so the value is
        #     reporting-only and the harness performs no further optimization.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        generator_optimizer, discriminator_optimizer = self._require_gan_optimizers()
        reference_waveform: torch.Tensor = self._prepare_reference_waveform(batch)
        reference_mel: torch.Tensor = self._prepare_reference_mel(reference_waveform)
        synthesized_waveform: torch.Tensor = self.network(reference_mel)
        real_discriminator_waveform: torch.Tensor = self._to_discriminator_waveform(reference_waveform)
        fake_discriminator_waveform: torch.Tensor = self._to_discriminator_waveform(synthesized_waveform)
        real_discriminator_waveform, fake_discriminator_waveform = self._align_waveform_pair(
            real_discriminator_waveform,
            fake_discriminator_waveform
        )
        reference_reconstruction_mel: torch.Tensor = self._prepare_network_mel(
            self._compute_full_precision_mel(self._reconstruction_mel_spectrogram, reference_waveform)
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
        self._step_exponential_schedulers()
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
        # val_loss for checkpoint monitoring.
        #
        # The adversarial terms are deliberately excluded, so the monitored
        # quantity measures synthesis fidelity alone and does not move with the
        # state of the discriminators. Both mels are extracted under the
        # reconstruction protocol and compared over their common frame span,
        # since the synthesis need not yield exactly the reference length. The
        # same value is published twice: under val_loss for the monitoring
        # contract and under val_mel_l1 for the metric it actually is.
        #
        # Args:
        #     batch: Collated LJSpeech mapping carrying the reference waveform.
        #     batch_idx: Index of this batch within the validation pass;
        #         unused.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping, or its waveform entry
        #         is not a tensor.
        #     ValueError: If the waveform cannot be normalized to
        #         ``[batch, time]``.
        #
        # Returns:
        #     A mapping whose ``"loss"`` is the mel L1 that checkpointing and
        #     early stopping monitor.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        reference_waveform: torch.Tensor = self._prepare_reference_waveform(batch)
        reference_mel: torch.Tensor = self._prepare_reference_mel(reference_waveform)
        synthesized_waveform: torch.Tensor = self.network(reference_mel)
        reference_reconstruction_mel: torch.Tensor = self._prepare_network_mel(
            self._compute_full_precision_mel(self._reconstruction_mel_spectrogram, reference_waveform)
        )
        synthesized_reconstruction_mel: torch.Tensor = self._prepare_network_mel(
            self._compute_full_precision_mel(self._reconstruction_mel_spectrogram, synthesized_waveform)
        )
        aligned_frames: int = min(reference_reconstruction_mel.shape[-1], synthesized_reconstruction_mel.shape[-1])
        validation_loss: torch.Tensor = torch.nn.functional.l1_loss(
            synthesized_reconstruction_mel[..., :aligned_frames],
            reference_reconstruction_mel[..., :aligned_frames]
        )
        self.log("val_loss", validation_loss)
        self.log("val_mel_l1", validation_loss)
        return {"loss": validation_loss}

    @override
    def predict_step(self, batch: Batch, batch_idx: int) -> ModelOutput:
        # Synthesizes from the batch waveform's conditioning mel and
        # returns the synthesized_waveform mapping the measurement stack
        # consumes.
        #
        # The channel axis the generator emits is squeezed away here, because
        # the measurement components compare ``[batch, time]`` signals.
        #
        # Args:
        #     batch: Collated mapping carrying the reference waveform whose
        #         conditioning mel drives the synthesis.
        #     batch_idx: Index of this batch within the prediction pass;
        #         unused.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping, or its waveform entry
        #         is not a tensor.
        #
        # Returns:
        #     A mapping holding the synthesis under ``"synthesized_waveform"``
        #     shaped ``[batch, time]``.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        waveform: object = batch.get("waveform")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("batch['waveform'] must be Tensor")
        reference_mel: torch.Tensor = self._prepare_reference_mel(waveform)
        synthesized_waveform: torch.Tensor = self.network(reference_mel).squeeze(1)
        return {"synthesized_waveform": synthesized_waveform}

    @override
    def test_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Delegates to predict_step and validates the output contract, so
        # test evaluation measures exactly the prediction path.
        #
        # Raises:
        #     TypeError: If predict_step returns something other than a
        #         mapping carrying a tensor under ``"synthesized_waveform"``,
        #         which would mean the two paths had diverged.
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
        #
        # Two AdamW optimizers are returned in a fixed order that the training
        # step depends on: the generator's first, then one covering both
        # discriminator ensembles as a single parameter set. Each is paired
        # with an exponential schedule by position. The schedules are advanced
        # by the training step rather than the harness, once per optimizer
        # step.
        #
        # Returns:
        #     An OptimizationConfiguration carrying the two optimizers and the
        #     two schedules, already constructed as torch objects.
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
        # Returns the reconstruction protocol mel error is measured under,
        # which is a separate record from the conditioning protocol.
        return self._configuration.reconstruction_mel_protocol

    @property
    def configuration(self) -> BigvganConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    def _prepare_sample_rate(self, sample_rate_raw: object) -> int:
        # Normalizes a batch sample-rate entry to a plain integer, accepting
        # either the collated scalar tensor or an already-unwrapped integer. No
        # step in this module currently consults the batch sample rate, since
        # the conditioning protocol fixes it, so this helper has no call site
        # and is retained for parity with the other architecture wrappers.
        #
        # Raises:
        #     TypeError: If the entry is neither an integer nor a tensor.
        if isinstance(sample_rate_raw, int):
            return sample_rate_raw
        if isinstance(sample_rate_raw, torch.Tensor):
            return int(sample_rate_raw.item())
        raise TypeError("batch['sample_rate'] must be int or Tensor")

    def _prepare_reference_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        # Produces the batched full-precision conditioning mel.
        return self._prepare_network_mel(self._compute_full_precision_mel(self._mel_spectrogram, waveform))

    def _compute_full_precision_mel(self, transform: MelSpectrogram, waveform: torch.Tensor) -> torch.Tensor:
        # Mel features condition the network and define loss targets, so they stay float32.
        # The waveform is cast before the layout normalization, so the
        # transform never observes a reduced-precision input.
        with self._full_precision_context():
            return transform(self._prepare_waveform_for_mel(waveform.float()))

    def _full_precision_context(self) -> torch.autocast | nullcontext[None]:
        # Keeps mel-spectrogram extraction in float32 so mixed-precision training cannot alter
        # the conditioning protocol validated by the published checkpoint.
        # Autocast is only meaningful on the device types torch supports it
        # for; anywhere else a null context is returned, because entering
        # autocast for an unsupported device would raise rather than no-op.
        device_type: str = self.device.type
        if device_type in ("cuda", "cpu"):
            return torch.autocast(device_type=device_type, enabled=False)
        return nullcontext()

    def _step_exponential_schedulers(self) -> None:
        # Steps the paired generator and discriminator exponential schedulers once per optimizer step.
        #
        # Both schedules advance unconditionally, including during the
        # discriminator freeze window: the discriminator optimizer is not
        # stepped there, but its schedule still decays, so its learning rate on
        # the first adversarial step reflects every step taken since training
        # began. The resulting rates are logged so the decay is auditable from
        # the run record rather than inferred from the configured gamma.
        #
        # Raises:
        #     RuntimeError: If the trainer has not materialized exactly two
        #         schedulers, since the pairing with the two optimizers is
        #         positional and cannot be recovered otherwise.
        schedulers: torch.optim.lr_scheduler.LRScheduler | list[torch.optim.lr_scheduler.LRScheduler] | None = self.lr_schedulers()
        if not isinstance(schedulers, list) or len(schedulers) != 2:
            raise RuntimeError("BigVGAN training requires generator and discriminator schedulers")
        for scheduler in schedulers:
            scheduler.step()
        self.log("learning_rate_generator", float(schedulers[0].get_last_lr()[0]))
        self.log("learning_rate_discriminator", float(schedulers[1].get_last_lr()[0]))

    def _prepare_reference_waveform(self, batch: LJSpeechBatch) -> torch.Tensor:
        # Reads the reference waveform and normalizes it to [batch, time].
        #
        # Raises:
        #     ValueError: If the waveform carries more than two dimensions,
        #         which no collation of this dataset produces.
        waveform: torch.Tensor = self._extract_waveform(batch)
        if waveform.ndim == 1:
            waveform: torch.Tensor = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform shape [batch, time], got {tuple(waveform.shape)}")
        return waveform

    def _prepare_waveform_for_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        # Reduces any accepted waveform layout to the [batch, time] form the
        # mel transform consumes, so a synthesis carrying the generator's
        # channel axis and a reference without one take the same path.
        #
        # Raises:
        #     ValueError: If the layout is none of the three accepted forms.
        if waveform.ndim == 1:
            return waveform.unsqueeze(0)
        if waveform.ndim == 2:
            return waveform
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            return waveform.squeeze(1)
        raise ValueError(
            f"Expected waveform shape [time], [batch, time], or [batch, 1, time], got {tuple(waveform.shape)}"
        )

    def _prepare_network_mel(self, mel: torch.Tensor) -> torch.Tensor:
        # Normalizes mel layouts to the [batch, bands, frames] shape the
        # network consumes.
        #
        # Raises:
        #     ValueError: If the layout is none of the three accepted forms.
        if mel.ndim == 2:
            return mel.unsqueeze(0)
        if mel.ndim == 3:
            return mel
        if mel.ndim == 4 and mel.shape[1] == 1:
            return mel.squeeze(1)
        raise ValueError(
            "Expected mel shape [channels, frames], [batch, channels, frames], "
            f"or [batch, 1, channels, frames], got {tuple(mel.shape)}"
        )

    def _to_discriminator_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        # Adds whatever axes are missing to reach the [batch, channel, time]
        # layout both ensembles require, so the reference and the synthesis
        # arrive in identical shape.
        #
        # Raises:
        #     ValueError: If the waveform carries more than three dimensions.
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
        # Truncates both signals to their common length before they are
        # compared, since the synthesis length is the frame count times the
        # upsampling product and need not equal the reference length; an
        # unaligned pair would make the ensembles judge different spans.
        minimum_samples: int = min(first_waveform.shape[-1], second_waveform.shape[-1])
        return first_waveform[..., :minimum_samples], second_waveform[..., :minimum_samples]

    def _run_discriminator_training_step(
        self,
        discriminator_optimizer: torch.optim.Optimizer,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Discriminator update, skipped inside the configured freeze window
        # so adversarial pressure starts only after the generator has a
        # signal; afterwards both ensembles judge real against the detached
        # synthesis under the optimizer toggle.
        #
        # Inside the window the return is a zero loss and a zero component, and
        # nothing else happens: no forward through either ensemble, no
        # backward, and no optimizer step, so the critics remain exactly at
        # their initialization and cost nothing to keep frozen. The logged
        # discriminator loss is therefore genuinely zero during the window
        # rather than an untracked quantity.
        #
        # Outside the window, detaching the synthesis keeps this update from
        # reaching the generator, and the toggle additionally clears
        # requires_grad on every parameter the discriminator optimizer does not
        # own, so the generator cannot accumulate gradients even indirectly.
        #
        # Args:
        #     discriminator_optimizer: Optimizer covering both ensembles.
        #     real_waveform: Reference signal in discriminator layout.
        #     fake_waveform: Synthesis in the same layout; detached here rather
        #         than by the caller, because the caller still needs the
        #         attached tensor for the generator update.
        #
        # Returns:
        #     The detached discriminator loss and its per-term component
        #     mapping for logging.
        if int(self.global_step) < self._configuration.freeze_discriminator_steps:
            return torch.zeros((), device=real_waveform.device), {"discriminator_loss_total": 0.0}
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
        reference_reconstruction_mel: torch.Tensor,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Generator update: adversarial logits and feature maps from both
        # ensembles combine with the mel reconstruction term, stepping only
        # the generator under the optimizer toggle.
        #
        # The branch mirrors the discriminator half and tests the same
        # predicate, so the two switch together. Inside the freeze window the
        # objective is the weighted mel L1 alone, computed over the common
        # frame span and reported under the same two component names the loss
        # composition would emit, with the mel term divided back out of its
        # weight so the reported value is comparable across the boundary.
        # Outside the window both ensembles are re-run on the attached
        # synthesis, so the generator is scored against the discriminator
        # weights the preceding update produced; the real logits are
        # discarded because the generator objective only asks how convincing
        # its own output is, while the real feature maps are kept for the
        # feature-matching term.
        #
        # Args:
        #     generator_optimizer: Optimizer covering the generator network.
        #     reference_reconstruction_mel: Full-band reference mel of the
        #         reconstruction term.
        #     real_waveform: Reference signal in discriminator layout.
        #     fake_waveform: Attached synthesis in the same layout; it is also
        #         the tensor the candidate mel is extracted from, so the
        #         reconstruction gradient reaches the generator.
        #
        # Returns:
        #     The detached generator loss and its per-term component mapping
        #     for logging.
        with self.toggled_optimizer(generator_optimizer):
            synthesized_mel: torch.Tensor = self._prepare_network_mel(
                self._compute_full_precision_mel(self._reconstruction_mel_spectrogram, fake_waveform)
            )
            if int(self.global_step) < self._configuration.freeze_discriminator_steps:
                aligned_frames: int = min(reference_reconstruction_mel.shape[-1], synthesized_mel.shape[-1])
                generator_loss: torch.Tensor = torch.nn.functional.l1_loss(
                    synthesized_mel[..., :aligned_frames],
                    reference_reconstruction_mel[..., :aligned_frames]
                ) * self._configuration.loss_configuration.mel_reconstruction_weight
                components: dict[str, float] = {
                    "generator_loss_total": float(generator_loss.detach().item()),
                    "generator_loss_mel": float(
                        (generator_loss / self._configuration.loss_configuration.mel_reconstruction_weight)
                        .detach()
                        .item()
                    )
                }
            else:
                real_period_logits, fake_period_logits, real_period_features, fake_period_features = (
                    self._period_discriminator(real_waveform, fake_waveform)
                )
                real_resolution_logits, fake_resolution_logits, real_resolution_features, fake_resolution_features = (
                    self._resolution_discriminator(real_waveform, fake_waveform)
                )
                del real_period_logits, real_resolution_logits
                generator_loss, components = self._loss.compute_generator_loss(
                    reference_mel=reference_reconstruction_mel,
                    synthesized_mel=synthesized_mel,
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
        #     RuntimeError: If the module is detached from a trainer or the
        #         trainer materialized anything other than exactly two
        #         optimizers, since the adversarial step has no meaning
        #         otherwise.
        optimizers: torch.optim.Optimizer | list[torch.optim.Optimizer] | None = self.optimizers()
        if not isinstance(optimizers, list) or len(optimizers) != 2:
            raise RuntimeError("BigVGAN training requires generator and discriminator optimizers")
        return optimizers[0], optimizers[1]

    def _extract_waveform(self, batch: LJSpeechBatch) -> torch.Tensor:
        # Reads the waveform tensor from the collated batch mapping.
        #
        # Raises:
        #     TypeError: If the entry is absent or is not a tensor; the
        #         message names the offending type so a malformed collation is
        #         identifiable from the log alone.
        waveform: LJSpeechBatchValue | None = batch.get("waveform")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError(f"batch['waveform'] must be Tensor, got {type(waveform).__name__}")
        return waveform
