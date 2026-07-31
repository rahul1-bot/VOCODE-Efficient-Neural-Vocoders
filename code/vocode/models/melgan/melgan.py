# This module:
# 1. Implements the MelGAN vocoder for the Study 1 reproduction cohort,
#    aligned with the Seungwon Park reference recipe: a lightweight
#    transposed-convolution generator trained adversarially against the
#    MelGAN multi-scale discriminator ensemble
#
# Harness contract (syntheticmind):
# - Subclasses the harness Module with automatic optimization disabled;
#   training_step drives the discriminator and generator Adam optimizers
#   manually through the toggled-optimizer surface, with no scheduler
#   per the reference recipe
# - Validation logs the mel L1 as val_loss; predict_step and test_step
#   return the synthesized_waveform mapping the measurement stack
#   consumes; checkpoint hooks stamp and verify the configuration dump
#
# Design decisions:
# - The generator objective is adversarial hinge loss plus feature
#   matching, without a spectral reconstruction term, exactly as in the
#   reference; validation mel L1 exists for monitoring only
# - Mel extraction runs under a disabled-autocast context so mixed
#   precision cannot alter the conditioning protocol
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
from vocode.losses.melgan import MelganLoss, MelganLossConfig
from vocode.metrics.pesq import PesqConfig
from vocode.metrics.stoi import StoiConfig
from vocode.models.melgan.discriminator import MelganMultiScaleDiscriminator
from vocode.models.melgan.network import MelganNetwork
from vocode.transforms.mel import MelConfig, MelSpectrogram

__all__: list[str] = ["Melgan", "MelganConfig"]


class MelganConfig(BaseModel):
    # Frozen MelGAN configuration: generator topology, adversarial
    # hyperparameters, and the family's mel and metric protocols.
    #
    # Fields:
    #     input_mel_channels: Mel band count the generator's input
    #         convolution consumes.
    #     ngf: Base generator width. The input convolution widens to
    #         ``ngf`` doubled once per upsampling factor, and each stage
    #         halves that width again, so this one value scales the whole
    #         generator.
    #     upsample_factors: Transposed-convolution stride of each stage.
    #         Their product is the total upsampling factor and equals the
    #         hop length of the mel protocol.
    #     leaky_relu_slope: Negative slope used throughout the generator
    #         and forwarded into the discriminator ensemble, so both sides
    #         of the game share one activation shape.
    #     mel_protocol: The single mel protocol this family both
    #         conditions on and measures with; unlike HiFi-GAN it declares
    #         no separate reconstruction grid.
    #     pesq_protocol: PESQ measurement protocol for this family.
    #     stoi_protocol: STOI measurement protocol for this family.
    #     learning_rate: Shared Adam step size for the generator and
    #         discriminator optimizers. Default: ``0.0001``.
    #     adam_beta_1: First Adam moment decay for both optimizers. The
    #         reference value is markedly lower than the HiFi-GAN family
    #         uses. Default: ``0.5``.
    #     adam_beta_2: Second Adam moment decay for both optimizers.
    #         Default: ``0.9``.
    #     loss_configuration: Weight of the feature-matching term, which
    #         is this family's entire reconstruction signal.
    #         Default: ``MelganLossConfig()``.
    #
    # Note:
    #     There is no decay field because the reference recipe holds the
    #     learning rate constant for the whole run, which is why
    #     configure_optimizers declares no scheduler.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    input_mel_channels: PositiveInt
    ngf: PositiveInt
    upsample_factors: tuple[int, ...]
    leaky_relu_slope: float
    mel_protocol: MelConfig
    pesq_protocol: PesqConfig
    stoi_protocol: StoiConfig
    learning_rate: PositiveFloat = 0.0001
    adam_beta_1: PositiveFloat = 0.5
    adam_beta_2: PositiveFloat = 0.9
    loss_configuration: MelganLossConfig = MelganLossConfig()

    @classmethod
    def seungwon(cls) -> MelganConfig:
        # Builds the MelGAN configuration aligned with the Seungwon Park LJSpeech checkpoint.
        # This family publishes a single reference recipe rather than a
        # width ladder, and the topology it declares is the one the
        # released checkpoint was trained under, so the same record serves
        # both the project-trained and the published-weights lanes.
        #
        # Returns:
        #     The frozen reference configuration for this architecture.
        return cls(
            input_mel_channels=80,
            ngf=32,
            upsample_factors=(8, 8, 2, 2),
            leaky_relu_slope=0.2,
            mel_protocol=MelConfig.melgan_seungwon(),
            pesq_protocol=PesqConfig(),
            stoi_protocol=StoiConfig()
        )


class Melgan(Module):
    # MelGAN vocoder module: lightweight upsampling generator with residual
    # dilated stacks, trained adversarially against the multi-scale
    # discriminator under manual optimization.
    #
    # Integration: the module satisfies the vocode.models.vocoder.Vocoder
    # synthesis contract through its network member, its two mel-protocol
    # properties, and synthesize; both protocol properties return the same
    # record, because this family conditions and measures on one grid.
    #
    # The objective is adversarial hinge loss plus feature matching with
    # no spectral reconstruction term, which is what separates this family
    # from the HiFi-GAN and Vocos recipes. The mel L1 published during
    # validation is therefore a monitor rather than a training signal: it
    # exists so checkpoint selection has a quality-correlated series to
    # rank on, since adversarial sums do not track quality.
    def __init__(self, configuration: MelganConfig) -> None:
        # Disables automatic optimization for the two-optimizer adversarial
        # recipe and constructs the mel transform, generator, multi-scale
        # discriminator, and loss.
        #
        # Args:
        #     configuration: The frozen recipe record. Its activation
        #         slope reaches the discriminator ensemble as well as the
        #         generator, so both are built from this one record.
        super().__init__()
        self.automatic_optimization: bool = False
        self._configuration: MelganConfig = configuration
        self._mel_spectrogram: MelSpectrogram = MelSpectrogram(configuration.mel_protocol)
        self.network: MelganNetwork = MelganNetwork(
            input_mel_channels=configuration.input_mel_channels,
            ngf=configuration.ngf,
            upsample_factors=configuration.upsample_factors,
            leaky_relu_slope=configuration.leaky_relu_slope
        )
        self._discriminator: MelganMultiScaleDiscriminator = MelganMultiScaleDiscriminator(
            leaky_relu_slope=configuration.leaky_relu_slope
        )
        self._loss: MelganLoss = MelganLoss(configuration.loss_configuration)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Synthesizes a waveform batch from conditioning mels through the
        # generator network.
        return self.network(mel)

    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        # Generates waveform output from the architecture-specific acoustic representation.
        # This is the Vocoder protocol entry point the measurement stack
        # calls; the returned waveform keeps the generator's
        # [batch, 1, samples] layout rather than the squeezed layout the
        # prediction step emits.
        return self.forward(mel)

    @property
    def mel_protocol(self) -> MelConfig:
        # Returns the mel-spectrogram protocol required by this architecture family.
        # The generator additionally applies the upstream log-mel shift
        # and scale internally, so a mel extracted under this protocol is
        # fed to the network unmodified by the caller.
        return self._configuration.mel_protocol

    @property
    def metric_mel_protocol(self) -> MelConfig:
        # Returns the protocol the mel-error metric extracts with, which
        # for this family is the conditioning protocol itself.
        return self._configuration.mel_protocol

    @property
    def configuration(self) -> MelganConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # One adversarial training step: extract the conditioning mel in
        # full precision, train the discriminator on the detached synthesis,
        # then train the generator on adversarial and feature-matching
        # terms; components are logged and the detached generator loss
        # returned for reporting.
        #
        # The updates run generator first and discriminator second, the
        # reverse of the HiFi-GAN and Vocos ordering. Each update
        # synthesizes from the shared conditioning mel inside its own
        # optimizer toggle rather than sharing one synthesis, so the
        # discriminator judges a synthesis produced by the already-updated
        # generator.
        #
        # Args:
        #     batch: The collated LJSpeech mapping; only its waveform
        #         entry is read, since the conditioning mel is derived
        #         from that waveform.
        #     batch_idx: Index of this batch within the epoch. It is not
        #         consulted.
        #
        # Returns:
        #     A mapping whose ``"loss"`` is the detached generator loss,
        #     reporting-only because both optimizers have already stepped.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping, or its waveform
        #         entry is not a tensor.
        #     ValueError: If the waveform cannot be read as [batch, time].
        #     RuntimeError: If the module is not attached to a trainer
        #         holding the generator and discriminator optimizer pair.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        generator_optimizer, discriminator_optimizer = self._require_gan_optimizers()
        reference_waveform: torch.Tensor = self._prepare_reference_waveform(batch)
        with self._full_precision_context():
            reference_mel: torch.Tensor = self._mel_spectrogram(reference_waveform.float())
        if reference_mel.ndim == 2:
            reference_mel: torch.Tensor = reference_mel.unsqueeze(0)
        generator_loss, generator_components = self._run_generator_training_step(
            generator_optimizer=generator_optimizer,
            reference_mel=reference_mel,
            reference_waveform=reference_waveform
        )
        discriminator_loss, discriminator_components = self._run_discriminator_training_step(
            discriminator_optimizer=discriminator_optimizer,
            reference_mel=reference_mel,
            reference_waveform=reference_waveform
        )
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
        # val_loss for checkpoint monitoring. Both adversarial sums are
        # additionally computed and logged for visibility, but neither is
        # published as val_loss, because a hinge score reflects the
        # current discriminator rather than synthesis quality. Mel frames
        # are compared over the shorter of the two analyses, so a
        # boundary-length difference cannot fail the pass.
        #
        # Args:
        #     batch: The collated LJSpeech mapping; only its waveform
        #         entry is read.
        #     batch_idx: Index of this batch within the validation pass.
        #         It is not consulted.
        #
        # Returns:
        #     A mapping whose ``"loss"`` is the mel L1, the same value
        #     published as ``val_loss`` and ``val_mel_l1``.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping, or its waveform
        #         entry is not a tensor.
        #     ValueError: If the waveform cannot be read as [batch, time].
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        reference_waveform: torch.Tensor = self._prepare_reference_waveform(batch)
        reference_mel: torch.Tensor = self._mel_spectrogram(reference_waveform)
        if reference_mel.ndim == 2:
            reference_mel: torch.Tensor = reference_mel.unsqueeze(0)
        fake_waveform: torch.Tensor = self._to_discriminator_waveform(self.network(reference_mel))
        real_waveform: torch.Tensor = self._to_discriminator_waveform(reference_waveform)
        real_waveform, fake_waveform = self._align_waveform_pair(real_waveform, fake_waveform)
        synthesized_mel: torch.Tensor = self._mel_spectrogram(fake_waveform.squeeze(1))
        if synthesized_mel.ndim == 2:
            synthesized_mel: torch.Tensor = synthesized_mel.unsqueeze(0)
        aligned_frames: int = min(reference_mel.shape[-1], synthesized_mel.shape[-1])
        validation_mel_l1: torch.Tensor = torch.mean(
            torch.abs(reference_mel[..., :aligned_frames] - synthesized_mel[..., :aligned_frames])
        )
        fake_outputs: list[tuple[list[torch.Tensor], torch.Tensor]] = self._discriminator(fake_waveform)
        real_outputs: list[tuple[list[torch.Tensor], torch.Tensor]] = self._discriminator(real_waveform)
        generator_loss, _ = self._loss.compute_generator_loss(fake_outputs, real_outputs)
        discriminator_loss, _ = self._loss.compute_discriminator_loss(fake_outputs, real_outputs)
        # Mel-L1 is the checkpoint-selection monitor because adversarial sums do not track quality.
        self.log("val_loss", validation_mel_l1)
        self.log("val_mel_l1", validation_mel_l1)
        self.log("val_generator_loss", generator_loss)
        self.log("val_discriminator_loss", discriminator_loss)
        return {"loss": validation_mel_l1}

    @override
    def predict_step(self, batch: Batch, batch_idx: int) -> ModelOutput:
        # Synthesizes from the batch waveform's conditioning mel and
        # returns the synthesized_waveform mapping the measurement stack
        # consumes. Conditioning is analysis-by-synthesis: the mel comes
        # from the reference waveform of the batch itself. An unbatched
        # waveform needs no promotion here, because the mel transform then
        # yields a two-dimensional result that the batch-axis guard
        # promotes instead.
        #
        # Args:
        #     batch: The collated LJSpeech mapping; only its waveform
        #         entry is read.
        #     batch_idx: Index of this batch within the pass. It is not
        #         consulted.
        #
        # Returns:
        #     A mapping under the key ``"synthesized_waveform"`` holding a
        #     [batch, samples] tensor, the generator channel axis already
        #     squeezed out.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping, or its waveform
        #         entry is not a tensor.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        waveform: LJSpeechBatchValue | None = batch.get("waveform")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("batch['waveform'] must be Tensor")
        mel: torch.Tensor = self._mel_spectrogram(waveform)
        if mel.ndim == 2:
            mel: torch.Tensor = mel.unsqueeze(0)
        synthesized: torch.Tensor = self.network(mel).squeeze(1)
        return {"synthesized_waveform": synthesized}

    @override
    def test_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Delegates to the prediction step and validates its output
        # contract, so test evaluation measures exactly the prediction
        # path. The returned mapping is rebuilt with the single verified
        # key rather than forwarded.
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
        # unpacks the pair by position: the generator optimizer owns
        # exactly the network parameters and the discriminator optimizer
        # exactly the ensemble parameters.
        #
        # Returns:
        #     An OptimizationConfiguration holding the two live Adam
        #     optimizers, generator first, and no scheduler at all. The
        #     absent schedule is the reference recipe rather than an
        #     omission: MelGAN trains at a constant rate, which is why
        #     this module declares no epoch-end scheduler hook either.
        generator_optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self._configuration.learning_rate,
            betas=(self._configuration.adam_beta_1, self._configuration.adam_beta_2)
        )
        discriminator_optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self._discriminator.parameters(),
            lr=self._configuration.learning_rate,
            betas=(self._configuration.adam_beta_1, self._configuration.adam_beta_2)
        )
        return OptimizationConfiguration(optimizer=[generator_optimizer, discriminator_optimizer])

    def _run_generator_training_step(
        self,
        generator_optimizer: torch.optim.Optimizer,
        reference_mel: torch.Tensor,
        reference_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Generator update: fresh synthesis judged by the discriminator
        # ensemble, hinge-adversarial plus feature-matching loss, stepping
        # only the generator under the optimizer toggle.
        # The synthesis is kept attached so gradients reach the generator
        # through the ensemble, and the real branch is run as well because
        # feature matching needs the reference feature maps.
        #
        # Args:
        #     generator_optimizer: The optimizer owning the generator
        #         network parameters.
        #     reference_mel: Conditioning mel synthesis is driven from.
        #     reference_waveform: The reference waveform, judged alongside
        #         the synthesis to supply the feature-matching target.
        #
        # Returns:
        #     The detached generator loss paired with its named scalar
        #     components for logging.
        with self.toggled_optimizer(generator_optimizer):
            fake_waveform: torch.Tensor = self._to_discriminator_waveform(self.network(reference_mel))
            real_waveform: torch.Tensor = self._to_discriminator_waveform(reference_waveform)
            real_waveform, fake_waveform = self._align_waveform_pair(real_waveform, fake_waveform)
            fake_outputs: list[tuple[list[torch.Tensor], torch.Tensor]] = self._discriminator(fake_waveform)
            real_outputs: list[tuple[list[torch.Tensor], torch.Tensor]] = self._discriminator(real_waveform)
            generator_loss, components = self._loss.compute_generator_loss(fake_outputs, real_outputs)
            self.optimizer_zero_grad(generator_optimizer)
            self.manual_backward(generator_loss)
            self.optimizer_step(generator_optimizer)
        return generator_loss.detach(), components

    def _run_discriminator_training_step(
        self,
        discriminator_optimizer: torch.optim.Optimizer,
        reference_mel: torch.Tensor,
        reference_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Discriminator update: the ensemble judges real against detached
        # synthesis under the hinge objective, stepping only the
        # discriminator under the optimizer toggle.
        # This is a second forward through the generator rather than a
        # reuse of the generator update's synthesis, so the discriminator
        # is trained against the generator as it stands after its own step
        # in this batch. Detaching is what confines the update to the
        # ensemble; the toggle is the second, independent guard.
        #
        # Args:
        #     discriminator_optimizer: The optimizer owning the ensemble
        #         parameters.
        #     reference_mel: Conditioning mel the detached synthesis is
        #         produced from.
        #     reference_waveform: The reference waveform judged as real.
        #
        # Returns:
        #     The detached discriminator loss paired with its named scalar
        #     components for logging.
        with self.toggled_optimizer(discriminator_optimizer):
            fake_waveform: torch.Tensor = self._to_discriminator_waveform(self.network(reference_mel).detach())
            real_waveform: torch.Tensor = self._to_discriminator_waveform(reference_waveform)
            real_waveform, fake_waveform = self._align_waveform_pair(real_waveform, fake_waveform)
            fake_outputs: list[tuple[list[torch.Tensor], torch.Tensor]] = self._discriminator(fake_waveform)
            real_outputs: list[tuple[list[torch.Tensor], torch.Tensor]] = self._discriminator(real_waveform)
            discriminator_loss, components = self._loss.compute_discriminator_loss(fake_outputs, real_outputs)
            self.optimizer_zero_grad(discriminator_optimizer)
            self.manual_backward(discriminator_loss)
            self.optimizer_step(discriminator_optimizer)
        return discriminator_loss.detach(), components

    def _full_precision_context(self) -> torch.autocast | nullcontext[None]:
        # Keeps mel-spectrogram extraction in float32 so mixed-precision training cannot alter
        # the conditioning protocol validated by the published checkpoint.
        # torch.autocast accepts only device types it implements, so any
        # other placement falls back to a null context; the caller pairs
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
        #         at all, so calling the training step outside a fit run
        #         fails here rather than partway through an update.
        optimizers: torch.optim.Optimizer | list[torch.optim.Optimizer] | None = self.optimizers()
        if not isinstance(optimizers, list) or len(optimizers) != 2:
            raise RuntimeError("MelGAN training requires generator and discriminator optimizers")
        return optimizers[0], optimizers[1]

    def _prepare_reference_waveform(self, batch: LJSpeechBatch) -> torch.Tensor:
        # Reads the reference waveform and normalizes it to [batch, time].
        # A single-sample waveform is promoted to a batch of one; anything
        # still not two-dimensional is a batch-shape error rather than
        # something to reshape silently.
        #
        # Raises:
        #     TypeError: If the waveform entry is missing or not a tensor.
        #     ValueError: If the waveform cannot be read as [batch, time].
        waveform: LJSpeechBatchValue | None = batch.get("waveform")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("batch['waveform'] must be Tensor")
        if waveform.ndim == 1:
            waveform: LJSpeechBatchValue | None = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform shape [batch, time], got {tuple(waveform.shape)}")
        return waveform

    def _to_discriminator_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        # Normalizes a waveform to the [batch, 1, time] layout the
        # discriminator ensemble consumes. A three-dimensional input is
        # already in that layout and passes through, which is how the
        # generator's own output reaches the ensemble unchanged.
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

    def _align_waveform_pair(
        self,
        first_waveform: torch.Tensor,
        second_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Aligns generated and reference tensors before loss or metric computation.
        # Both tensors are cropped from the front to their common length,
        # because the generator's transposed-convolution chain can overrun
        # the reference by a few boundary samples. HiFi-GAN raises on the
        # same disagreement instead; this family repairs it, so a length
        # difference here is absorbed rather than reported.
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
