# This module:
# 1. Implements the FreeV vocoder for the Study 1 reproduction cohort,
#    following the official recipe: amplitude and phase frequency-domain
#    prediction seeded by a pseudo-inverse mel prior, so the amplitude
#    stream refines an analytic initial estimate instead of predicting
#    from scratch
# 2. Trains adversarially with the FreeV loss composition over the
#    predicted spectral components
# 3. Defines FreevConfig, the frozen experiment record pinning the topology,
#    the optimization settings, and the two mel protocols the reference
#    distinguishes between
#
# Harness contract (syntheticmind):
# - Subclasses the harness Module with automatic optimization disabled;
#   training_step predicts the spectral components once, drives both
#   optimizer updates manually, and the epoch-end hook advances the
#   exponential schedulers
# - Validation computes the spectral validation loss and logs val_loss for
#   checkpoint monitoring; predict_step and test_step return the
#   synthesized_waveform mapping the measurement stack consumes
# - No checkpoint hook is overridden, so state capture is exactly the
#   harness default: the module's own state dictionary and nothing else
#
# Design decisions:
# - The pseudo-inverse prior is the architecture's defining efficiency
#   idea: the amplitude head learns a residual refinement over an
#   analytically computed estimate, which shrinks the network
# - The center-true conditioning protocol of the reference is preserved
#   in float32 under a disabled-autocast context
# - The adversarial machinery is imported from APNet2 rather than
#   reimplemented: both discriminator ensembles and the STFT target analyzer
#   are the APNet2 components, so a FreeV-against-APNet2 comparison differs
#   only in the generator and its loss composition
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
from vocode.losses.freev import FreevLoss, FreevLossConfig
from vocode.metrics.pesq import PesqConfig
from vocode.metrics.stoi import StoiConfig
from vocode.models.apnet2.discriminator import Apnet2MultiPeriodDiscriminator, Apnet2MultiResolutionDiscriminator
from vocode.models.freev.network import FreevGeneratorOutput, FreevNetwork
from vocode.transforms.mel import MelConfig, MelSpectrogram

__all__: list[str] = ["Freev", "FreevConfig"]


class FreevConfig(BaseModel):
    # Frozen experiment record of one FreeV run. Every value the module and
    # its network need is pinned here, so a run is fully described by this
    # record and no setting can drift between construction and training.
    #
    # This record is the architecture recipe of the FreeV Project-Trained
    # Configuration. Fitting it from random initialization under the
    # registered budget produces the Retained Project Checkpoint, the terminal
    # durable state admitted by the checkpoint gate, and every reported
    # measurement of this configuration is inherited from that one state.
    #
    # Fields:
    #     input_mel_channels: Band count of the conditioning mel; it is also
    #         the column count of the pseudo-inverse projection.
    #     n_fft: Transform size of the synthesis STFT; the one-sided bin
    #         count is the width the amplitude refinement operates at.
    #     hop_size: Hop of the synthesis STFT, the samples each conditioning
    #         frame expands into.
    #     win_size: Window length of the synthesis STFT.
    #     sampling_rate: Rate the network reconstructs the mel basis against
    #         when it builds the analytic prior; it must agree with the
    #         conditioning protocol or the prior inverts the wrong basis.
    #     psp_channel: Working width of the phase stream.
    #     psp_input_conv_kernel_size: Kernel of the phase entry projection.
    #     psp_output_r_conv_kernel_size: Kernel of the real phase head.
    #     psp_output_i_conv_kernel_size: Kernel of the imaginary phase head.
    #     convnext_layer_count: Residual block depth of the phase stream.
    #         Default: ``8``.
    #     amplitude_refinement_layer_count: Residual block depth of the
    #         amplitude refinement, which is small precisely because the
    #         analytic prior does the bulk of the work. Default: ``1``.
    #     convnext_intermediate_dimension: Expanded width inside every
    #         residual block. Default: ``1536``.
    #     learning_rate: Initial rate of both AdamW optimizers.
    #         Default: ``0.0002``.
    #     adam_beta_1: First AdamW moment coefficient. Default: ``0.8``.
    #     adam_beta_2: Second AdamW moment coefficient. Default: ``0.99``.
    #     learning_rate_decay: Gamma of both exponential schedules, applied
    #         once per epoch by the epoch-end hook. Default: ``0.999``.
    #     gradient_clip_norm: Global gradient-norm ceiling applied separately
    #         to the generator and to the discriminator ensembles.
    #         Default: ``1000.0``.
    #     loss_configuration: Term weights of the FreeV loss composition.
    #         Default: ``FreevLossConfig()``.
    #     mel_protocol: Conditioning mel protocol; its band edges are what
    #         the network builds the analytic prior from.
    #     reconstruction_mel_protocol: Full-band mel protocol used for the
    #         reconstruction loss term and published as the metric protocol,
    #         so mel error is never measured on the band-limited view the
    #         network was conditioned with.
    #     pesq_protocol: PESQ measurement protocol recorded for this run.
    #     stoi_protocol: STOI measurement protocol recorded for this run.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    input_mel_channels: PositiveInt
    n_fft: PositiveInt
    hop_size: PositiveInt
    win_size: PositiveInt
    sampling_rate: PositiveInt
    psp_channel: PositiveInt
    psp_input_conv_kernel_size: PositiveInt
    psp_output_r_conv_kernel_size: PositiveInt
    psp_output_i_conv_kernel_size: PositiveInt
    convnext_layer_count: PositiveInt = 8
    amplitude_refinement_layer_count: PositiveInt = 1
    convnext_intermediate_dimension: PositiveInt = 1536
    learning_rate: PositiveFloat = 0.0002
    adam_beta_1: PositiveFloat = 0.8
    adam_beta_2: PositiveFloat = 0.99
    learning_rate_decay: PositiveFloat = 0.999
    gradient_clip_norm: PositiveFloat = 1000.0
    loss_configuration: FreevLossConfig = FreevLossConfig()
    mel_protocol: MelConfig
    reconstruction_mel_protocol: MelConfig
    pesq_protocol: PesqConfig
    stoi_protocol: StoiConfig

    @classmethod
    def official(cls) -> FreevConfig:
        # Builds the FreeV configuration aligned with the official reference implementation.
        #
        # This is the only entry point that reproduces the published anchor:
        # the topology stated here is what the released checkpoint was trained
        # at, so a strict author load succeeds against a network built from
        # this record and fails against any other. Fields not listed take their
        # declared defaults, which already match the reference. Note the
        # absence of any amplitude projection setting: the amplitude branch has
        # no learned entry or exit layer to configure.
        return cls(
            input_mel_channels=80,
            n_fft=1024,
            hop_size=256,
            win_size=1024,
            sampling_rate=22050,
            psp_channel=512,
            psp_input_conv_kernel_size=7,
            psp_output_r_conv_kernel_size=7,
            psp_output_i_conv_kernel_size=7,
            mel_protocol=MelConfig.freev_official(),
            reconstruction_mel_protocol=MelConfig.freev_official_reconstruction(),
            pesq_protocol=PesqConfig(),
            stoi_protocol=StoiConfig()
        )


class Freev(Module):
    # The FreeV vocoder as a harness module. It owns the generator network,
    # the loss composition, and the two mel transforms, and drives adversarial
    # training itself because automatic optimization cannot express a
    # two-optimizer scheme.
    #
    # Its adversarial machinery is deliberately not its own. Both discriminator
    # ensembles and the STFT target analyzer are the APNet2 components,
    # imported rather than reimplemented, so the two architectures are judged
    # by identical critics against identically extracted targets and any
    # measured difference between them is attributable to the generator and its
    # loss composition alone.
    #
    # One training step predicts the spectral components exactly once and
    # spends that single prediction twice. The discriminator update comes
    # first and judges the reference against the detached synthesis, so no
    # gradient reaches the generator through it. The generator update then
    # re-runs both ensembles on the attached synthesis, meaning the generator
    # is scored by the discriminator weights that update produced, and
    # combines the adversarial and feature-matching signals with the spectral
    # supervision terms. Each update runs inside the harness optimizer toggle,
    # which restricts requires_grad to the stepping optimizer's own
    # parameters, so neither update can leave gradients in the other's weights.
    #
    # Every spectral quantity that enters a loss is extracted inside a float32
    # island: the conditioning mel, the full-band reconstruction mel, and the
    # STFT targets are all computed with autocast disabled, because the
    # reference protocol is what the published checkpoint was trained under and
    # reduced precision would silently change it.
    #
    # Integration: the module is constructed through the model registry from a
    # FreevConfig, and the published release is loaded onto its network
    # attribute by vocode.models.freev.weights.FreevWeights. It satisfies the
    # vocode.models.vocoder.Vocoder protocol through network, mel_protocol,
    # metric_mel_protocol, and synthesize, so the metric and profiling
    # components consume it without knowing its architecture.
    def __init__(self, configuration: FreevConfig) -> None:
        # Builds the generator, the shared APNet2 discriminator ensembles, the
        # loss composition, and the two mel transforms from one frozen record,
        # and disables automatic optimization before anything else so the
        # trainer validates this module against the manual-optimization
        # contract.
        #
        # The network receives its band edges from the conditioning protocol
        # rather than from separate fields, which is what guarantees the
        # analytic prior inverts exactly the basis the network is fed.
        #
        # Args:
        #     configuration: Frozen experiment record; it is retained as-is and
        #         republished through the configuration property, so the run is
        #         reproducible from the module alone.
        super().__init__()
        self.automatic_optimization: bool = False
        self._configuration: FreevConfig = configuration
        self._mel_spectrogram: MelSpectrogram = MelSpectrogram(configuration.mel_protocol)
        self._reconstruction_mel_spectrogram: MelSpectrogram = MelSpectrogram(configuration.reconstruction_mel_protocol)
        self._spectrum_analyzer: Apnet2SpectrumAnalyzer = Apnet2SpectrumAnalyzer(
            n_fft=configuration.n_fft,
            hop_size=configuration.hop_size,
            win_size=configuration.win_size
        )
        self.network: FreevNetwork = FreevNetwork(
            num_mels=configuration.input_mel_channels,
            n_fft=configuration.n_fft,
            hop_size=configuration.hop_size,
            win_size=configuration.win_size,
            sampling_rate=configuration.sampling_rate,
            fmin=configuration.mel_protocol.fmin,
            fmax=configuration.mel_protocol.fmax,
            psp_channel=configuration.psp_channel,
            psp_input_conv_kernel_size=configuration.psp_input_conv_kernel_size,
            psp_output_r_conv_kernel_size=configuration.psp_output_r_conv_kernel_size,
            psp_output_i_conv_kernel_size=configuration.psp_output_i_conv_kernel_size,
            convnext_layer_count=configuration.convnext_layer_count,
            amplitude_refinement_layer_count=configuration.amplitude_refinement_layer_count,
            convnext_intermediate_dimension=configuration.convnext_intermediate_dimension
        )
        self._period_discriminator: Apnet2MultiPeriodDiscriminator = Apnet2MultiPeriodDiscriminator()
        self._resolution_discriminator: Apnet2MultiResolutionDiscriminator = Apnet2MultiResolutionDiscriminator()
        self._loss: FreevLoss = FreevLoss(configuration.loss_configuration)

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
    def on_train_epoch_end(self) -> None:
        # Advances both exponential schedules once per epoch, which is the
        # reference's decay cadence. The unwrapping mirrors the harness
        # convention that a single configured scheduler is returned bare while
        # several are returned as a list, and a detached module reports None,
        # in which case the hook is inert.
        schedulers: torch.optim.lr_scheduler.LRScheduler | list[torch.optim.lr_scheduler.LRScheduler] | None = self.lr_schedulers()
        if isinstance(schedulers, list):
            for scheduler in schedulers:
                scheduler.step()
        elif schedulers is not None:
            schedulers.step()

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # One adversarial training step over the predicted spectral
        # components: discriminator update on the detached synthesis, then
        # the generator update through the FreeV spectral and adversarial
        # composition; components are logged and the detached generator
        # loss returned.
        #
        # The reference targets are prepared before either update: the
        # band-limited conditioning mel, the full-band reconstruction mel, and
        # the STFT target spectrum, all in float32. The waveform pair handed to
        # the discriminators is truncated to the shorter of the two, because
        # the centered inverse STFT does not return exactly the input length.
        #
        # Args:
        #     batch: Collated LJSpeech mapping; only the waveform entry is
        #         read, and the conditioning mel is derived from it rather
        #         than taken from the batch, so training and inference share
        #         one extraction path.
        #     batch_idx: Index of this batch within the epoch; unused, since
        #         the step's behavior does not vary across an epoch.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping, or its waveform entry
        #         is not a tensor.
        #     ValueError: If the waveform cannot be normalized to
        #         ``[batch, time]``.
        #     RuntimeError: If the trainer has not materialized both
        #         optimizers.
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
        reference_reconstruction_mel: torch.Tensor = self._prepare_reference_reconstruction_mel(reference_waveform)
        reference_spectrum: Apnet2Spectrum = self._analyze_full_precision_spectrum(reference_waveform)
        generator_output: FreevGeneratorOutput = self.network.predict_components(reference_mel)
        real_discriminator_waveform: torch.Tensor = self._to_discriminator_waveform(reference_waveform)
        fake_discriminator_waveform: torch.Tensor = self._to_discriminator_waveform(generator_output.waveform)
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
            reference_spectrum=reference_spectrum,
            generator_output=generator_output,
            real_waveform=real_discriminator_waveform,
            fake_waveform=fake_discriminator_waveform
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
        # Validates through the spectral validation loss over the predicted
        # amplitude and phase components, logged as val_loss.
        #
        # The validation composition is the generator composition without its
        # adversarial and feature-matching terms, so the monitored quantity
        # measures synthesis quality alone and does not move with the state of
        # the discriminators.
        #
        # Args:
        #     batch: Collated LJSpeech mapping; the waveform entry supplies
        #         both the conditioning and the supervision targets.
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
        #     A mapping whose ``"loss"`` is the validation loss that
        #     checkpointing and early stopping monitor under the key
        #     ``val_loss``.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        reference_waveform: torch.Tensor = self._prepare_reference_waveform(batch)
        reference_mel: torch.Tensor = self._prepare_reference_mel(reference_waveform)
        reference_reconstruction_mel: torch.Tensor = self._prepare_reference_reconstruction_mel(reference_waveform)
        reference_spectrum: Apnet2Spectrum = self._analyze_full_precision_spectrum(reference_waveform)
        generator_output: FreevGeneratorOutput = self.network.predict_components(reference_mel)
        candidate_reconstruction_mel: torch.Tensor = self._prepare_network_mel(
            self._compute_full_precision_mel(self._reconstruction_mel_spectrogram, generator_output.waveform)
        )
        final_spectrum: Apnet2Spectrum = self._analyze_full_precision_spectrum(generator_output.waveform)
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
        for name, value in components.items():
            self.log(f"val_{name}", value)
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
        waveform: LJSpeechBatchValue | None = batch.get("waveform")
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
        # by the epoch-end hook rather than by the harness, which is why they
        # decay once per epoch and not once per optimizer step.
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
        # Returns the full-band reconstruction protocol mel error is measured
        # under, which is deliberately not the band-limited conditioning
        # protocol.
        return self._configuration.reconstruction_mel_protocol

    @property
    def configuration(self) -> FreevConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    def _prepare_reference_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        # Produces the batched full-precision conditioning mel.
        mel: torch.Tensor = self._compute_full_precision_mel(self._mel_spectrogram, waveform)
        return self._prepare_network_mel(mel)

    def _prepare_reference_reconstruction_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        # Produces the batched full-precision full-band mel that the
        # reconstruction term compares against, extracted under the
        # reconstruction protocol rather than the conditioning protocol.
        mel: torch.Tensor = self._compute_full_precision_mel(self._reconstruction_mel_spectrogram, waveform)
        return self._prepare_network_mel(mel)

    def _compute_full_precision_mel(self, transform: MelSpectrogram, waveform: torch.Tensor) -> torch.Tensor:
        # Computes one mel extraction inside the float32 island regardless of trainer precision.
        # The waveform is cast before the layout normalization so the transform
        # never observes a reduced-precision input.
        with self._full_precision_context():
            return transform(self._prepare_waveform_for_mel(waveform.float()))

    def _analyze_full_precision_spectrum(self, waveform: torch.Tensor) -> Apnet2Spectrum:
        # Keeps the STFT target extraction in float32 so mixed precision cannot alter the spectra.
        with self._full_precision_context():
            return self._spectrum_analyzer.analyze(waveform.float())

    def _full_precision_context(self) -> torch.autocast | nullcontext[None]:
        # Keeps spectral extraction in float32 so mixed-precision training cannot alter
        # the protocol validated by the published checkpoint.
        # Autocast is only meaningful on the device types torch supports it
        # for; anywhere else a null context is returned, because entering
        # autocast for an unsupported device would raise rather than no-op.
        device_type: str = self.device.type
        if device_type in ("cuda", "cpu"):
            return torch.autocast(device_type=device_type, enabled=False)
        return nullcontext()

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
        raise ValueError(f"Expected waveform shape [time], [batch, time], or [batch, 1, time], got {tuple(waveform.shape)}")

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
        raise ValueError(f"Expected mel shape [channels, frames], [batch, channels, frames], or [batch, 1, channels, frames], got {tuple(mel.shape)}")

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
        # compared. The centered inverse STFT returns
        # ``(frames - 1) * hop_size`` samples, which need not equal the
        # reference length, and an unaligned pair would make the ensembles
        # judge different spans.
        minimum_samples: int = min(first_waveform.shape[-1], second_waveform.shape[-1])
        return first_waveform[..., :minimum_samples], second_waveform[..., :minimum_samples]

    def _run_discriminator_training_step(
        self,
        discriminator_optimizer: torch.optim.Optimizer,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Discriminator update: the ensembles judge real against the
        # detached synthesis, stepping only the discriminator under the
        # optimizer toggle.
        #
        # Detaching the synthesis is what keeps this update from reaching the
        # generator, and the toggle additionally clears requires_grad on every
        # parameter the discriminator optimizer does not own, so the generator
        # cannot accumulate gradients even indirectly.
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
        reference_spectrum: Apnet2Spectrum,
        generator_output: FreevGeneratorOutput,
        real_waveform: torch.Tensor,
        fake_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Generator update: the spectral supervision terms combine with the
        # adversarial and feature-matching terms, stepping only the
        # generator under the optimizer toggle.
        #
        # Both ensembles are re-run on the attached synthesis, so the generator
        # is scored against the discriminator weights the preceding update
        # produced. The real logits are discarded because the generator
        # objective only asks how convincing its own output is; the real
        # feature maps are kept, since feature matching compares the two
        # stacks. The candidate mel and the candidate STFT spectrum are
        # extracted here rather than by the caller, because they must carry the
        # generator's graph for the reconstruction and consistency terms to
        # propagate.
        #
        # Args:
        #     generator_optimizer: Optimizer covering the generator network.
        #     reference_reconstruction_mel: Full-band reference mel of the
        #         reconstruction term.
        #     reference_spectrum: STFT targets extracted from the reference.
        #     generator_output: The single component prediction of this step.
        #     real_waveform: Reference signal in discriminator layout.
        #     fake_waveform: Attached synthesis in the same layout.
        #
        # Returns:
        #     The detached generator loss and its per-term component mapping
        #     for logging.
        with self.toggled_optimizer(generator_optimizer):
            real_period_logits, fake_period_logits, real_period_features, fake_period_features = (
                self._period_discriminator(real_waveform, fake_waveform)
            )
            real_resolution_logits, fake_resolution_logits, real_resolution_features, fake_resolution_features = (
                self._resolution_discriminator(real_waveform, fake_waveform)
            )
            del real_period_logits, real_resolution_logits
            candidate_reconstruction_mel: torch.Tensor = self._prepare_network_mel(
                self._compute_full_precision_mel(self._reconstruction_mel_spectrogram, generator_output.waveform)
            )
            final_spectrum: Apnet2Spectrum = self._analyze_full_precision_spectrum(generator_output.waveform)
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
            raise RuntimeError("FreeV training requires generator and discriminator optimizers")
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
