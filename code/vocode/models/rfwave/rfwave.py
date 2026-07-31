# This module:
# 1. Implements the RFWave vocoder for the Study 1 reproduction cohort: a
#    faithful reimplementation of the published multi-band rectified-flow
#    architecture, trained by flow matching and synthesized by Euler
#    integration of the learned velocity field
#
# Harness contract (syntheticmind):
# - Subclasses the harness Module with automatic optimization disabled;
#   training_step drives the single AdamW optimizer manually so the
#   reference gradient hygiene (finite-gradient skip and clip norm 5.0)
#   wraps the step exactly as published
# - Synthesis routes through the exchangeable ODE sampler collaborator,
#   which is the seam the Study 2 sampling-step reduction replaces
# - predict_step and test_step return the synthesized_waveform mapping;
#   checkpoint hooks stamp and verify the configuration dump
#
# Update semantics:
# - Single-phase: one optimizer step per batch, with no adversarial
#   component and therefore no phase alternation. This family is not a GAN
#   at all; its objective is a regression onto a velocity target plus two
#   auxiliary spectral terms
# - Training cost is independent of the sampling step count, because a
#   training step samples one point on the path rather than traversing it.
#   Synthesis cost, by contrast, scales directly with that step count, so
#   the two are decoupled and a step-count change is a pure inference
#   intervention requiring no retraining
#
# Design decisions:
# - Training supervises the velocity field on noised spectral states
#   (the rectified-flow objective) with auxiliary magnitude and overlap
#   terms weighted per the reference
# - The multi-band decomposition trains all bands jointly through the
#   shared backbone with band-index conditioning
# - A step producing non-finite gradients is skipped in its entirety rather
#   than clipped into range, matching the reference; the schedule advances
#   regardless, so a skipped step still consumes its place in the warmup and
#   decay curve
#
# Author: Rahul Sawhney

import math
from typing import ClassVar, override

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from syntheticmind.core.module import Module
from syntheticmind.core.optimizer import OptimizationConfiguration
from syntheticmind.utilities.types import Batch, ModelOutput, StepOutput

from vocode.losses.rfwave import RfwaveLoss, RfwaveLossConfig
from vocode.metrics.pesq import PesqConfig
from vocode.metrics.stoi import StoiConfig
from vocode.models.rfwave.flow import RfwaveRectifiedFlow, RfwaveTrainTuple
from vocode.models.rfwave.network import RfwaveNetworkConfig
from vocode.models.rfwave.sampling import RfwaveOdeSampler
from vocode.transforms.mel import MelConfig, MelSpectrogram

__all__: list[str] = ["Rfwave", "RfwaveConfig"]


class RfwaveConfig(BaseModel):
    # Frozen RFWave configuration: backbone and band topology, flow and
    # sampling settings, and the family's mel and metric protocols. Every
    # field carries the reference default, so the published recipe is
    # reproduced by constructing the record with no arguments at all.
    #
    # Fields:
    #     network: Backbone topology and band geometry.
    #         Default: ``RfwaveNetworkConfig()``.
    #     learning_rate: Peak AdamW step size, reached at the end of warmup
    #         and decayed by the cosine schedule thereafter.
    #         Default: ``0.0002``.
    #     adam_beta_1: First AdamW moment decay. Default: ``0.9``.
    #     adam_beta_2: Second AdamW moment decay. Default: ``0.999``.
    #     scheduler_warmup_steps: Steps over which the rate rises linearly
    #         from zero. Default: ``20000``.
    #     scheduler_total_steps: Total steps the cosine decay is planned
    #         against; training beyond it holds the rate at zero, since the
    #         schedule's progress is clamped. Default: ``125000``.
    #     gradient_clip_norm: Maximum gradient norm, applied only on steps
    #         whose gradients are finite. Default: ``5.0``.
    #     sampling_step_count: Euler steps used at synthesis. This is the
    #         only field that changes inference cost without changing the
    #         model, and the seam the sampling-step study exchanges.
    #         Default: ``10``.
    #     loss_configuration: Weights of the composite objective.
    #         Default: ``RfwaveLossConfig()``.
    #     mel_protocol: Conditioning protocol, one hundred unnormalized
    #         HTK-scale bands at 24 kHz. The rate differs from the 22.05 kHz
    #         families in this package, so mel error against this protocol is
    #         not comparable across sampling rates.
    #         Default: the 24 kHz reference protocol.
    #     pesq_protocol: PESQ measurement protocol. Default: ``PesqConfig()``.
    #     stoi_protocol: STOI measurement protocol. Default: ``StoiConfig()``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    network: RfwaveNetworkConfig = RfwaveNetworkConfig()
    learning_rate: PositiveFloat = 2e-4
    adam_beta_1: PositiveFloat = 0.9
    adam_beta_2: PositiveFloat = 0.999
    scheduler_warmup_steps: PositiveInt = 20000
    scheduler_total_steps: PositiveInt = 125000
    gradient_clip_norm: PositiveFloat = 5.0
    sampling_step_count: PositiveInt = 10
    loss_configuration: RfwaveLossConfig = RfwaveLossConfig()
    mel_protocol: MelConfig = MelConfig.vocos_charactr_mel_24khz()
    pesq_protocol: PesqConfig = PesqConfig()
    stoi_protocol: StoiConfig = StoiConfig()

    @classmethod
    def bfs18_24khz(cls) -> RfwaveConfig:
        # Builds the RFWave 24 kHz configuration aligned with the reference
        # repository. Every field already defaults to its reference value, so
        # this factory supplies no arguments; it exists so that call sites name
        # the recipe explicitly rather than relying on defaults implicitly, in
        # the same form the other families' factories take.
        #
        # Returns:
        #     The frozen Project-Trained Configuration of this architecture,
        #     the record a Study 1 training trajectory is fitted under.
        return cls()


class Rfwave(Module):
    # RFWave vocoder module: a multi-band rectified flow trained by flow
    # matching and synthesized by integrating the learned velocity field. The
    # module owns the mel transform, the flow (which in turn owns the backbone
    # and the equalizer), the objective, and the sampler.
    #
    # Integration: this is the only non-adversarial mel-conditioned family in
    # the package. It has no discriminators, its objective is a regression
    # rather than a game, and it drives a single optimizer. Automatic
    # optimization is nonetheless disabled, for a different reason than in the
    # adversarial families: the reference gradient hygiene inspects gradients
    # between backward and the step and skips the step entirely when they are
    # not finite, which the loop-owned path provides no hook for.
    #
    # Integration: synthesis routes through the sampler collaborator rather
    # than through inline integration, which is what makes the step count an
    # exchangeable seam. Replacing the sampler with one of a different step
    # count changes inference cost and quality on fixed weights, so the
    # trade-off can be measured across step counts from a single checkpoint.
    #
    # Integration: the module satisfies the vocode.models.vocoder.Vocoder
    # structural protocol through its network property, its mel_protocol and
    # metric_mel_protocol properties, and synthesize.
    def __init__(self, configuration: RfwaveConfig) -> None:
        # Disables automatic optimization for the reference gradient hygiene
        # and constructs the mel transform, flow network, rectified-flow
        # trainer, loss, and the exchangeable ODE sampler.
        super().__init__()
        self.automatic_optimization: bool = False
        self._configuration: RfwaveConfig = configuration
        self._mel_spectrogram: MelSpectrogram = MelSpectrogram(configuration.mel_protocol)
        self.network: RfwaveRectifiedFlow = RfwaveRectifiedFlow(configuration.network)
        self._sampler: RfwaveOdeSampler = RfwaveOdeSampler(step_count=configuration.sampling_step_count)
        self._loss: RfwaveLoss = RfwaveLoss(configuration.loss_configuration)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Synthesizes a waveform batch by integrating the learned velocity
        # field through the currently held sampler. This is the module's most
        # expensive operation by a wide margin: it costs one backbone forward
        # pass per integration step, so it is not comparable in cost to the
        # single-pass forward of any other family here.
        #
        # Args:
        #     mel: Conditioning mel of shape ``[batch, channels, frames]``
        #         under this family's 24 kHz protocol.
        #
        # Returns:
        #     The synthesized waveform of shape ``[batch, samples]``.
        return self._sampler.synthesize(self.network, mel)

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
    def configuration(self) -> RfwaveConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    def _compute_flow_losses(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        # Computes the three-term objective for one batch, from a single
        # forward pass through the backbone.
        #
        # The primary term is the flow-matching regression, measured on the
        # velocity error after time balancing and after transforming back to
        # the waveform domain. The two auxiliary terms address what that term
        # alone leaves under-constrained: the magnitude term compares the
        # destination spectra the predicted and true velocities imply, giving
        # the objective a direct handle on spectral magnitude, and the overlap
        # term penalizes neighboring bands for disagreeing about the bins they
        # share, which the placement step would otherwise silently resolve by
        # discarding one band's opinion.
        #
        # The conditioning mel is derived from the target waveform rather than
        # read from the batch, so training conditions on exactly the protocol
        # this module owns.
        #
        # Args:
        #     waveform: Target waveforms for this batch.
        #
        # Returns:
        #     The weighted total under the key ``"total"`` alongside the three
        #     unweighted terms, so the caller can log the breakdown without
        #     recomputing anything.
        mel: torch.Tensor = self._compute_full_precision_mel(waveform)
        train_tuple: RfwaveTrainTuple = self.network.build_train_tuple(mel, waveform)
        prediction: torch.Tensor = self.network.predict_velocity(
            train_tuple.noisy_state,
            train_tuple.time_values,
            train_tuple.expanded_mel,
            train_tuple.band_index
        )
        balanced_prediction: torch.Tensor
        balanced_target: torch.Tensor
        balanced_prediction, balanced_target = self._loss.time_balance(
            prediction,
            train_tuple.velocity_target
        )
        velocity_waveform: torch.Tensor = self.network.velocity_error_waveform(
            balanced_prediction,
            balanced_target
        )
        rectified_flow_loss: torch.Tensor = self._loss.waveform_velocity_loss(velocity_waveform)
        implied_prediction: torch.Tensor
        implied_target: torch.Tensor
        implied_prediction, implied_target = self.network.implied_endpoints(
            train_tuple.noisy_state,
            train_tuple.time_values,
            prediction,
            train_tuple.velocity_target
        )
        magnitude_loss: torch.Tensor = self._loss.magnitude_loss(implied_prediction, implied_target)
        real_bands: list[torch.Tensor]
        imaginary_bands: list[torch.Tensor]
        real_bands, imaginary_bands = self.network.split_band_lists(prediction)
        overlap_loss: torch.Tensor = self._loss.overlap_loss(
            real_bands,
            imaginary_bands,
            self.network.configuration.left_overlap,
            self.network.configuration.right_overlap
        )
        auxiliary_weight: float = self._loss.configuration.auxiliary_weight
        total_loss: torch.Tensor = rectified_flow_loss + auxiliary_weight * (magnitude_loss + overlap_loss)
        return {
            "total": total_loss,
            "rectified_flow": rectified_flow_loss,
            "magnitude": magnitude_loss,
            "overlap": overlap_loss
        }

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Runs one flow-matching step under the reference gradient hygiene.
        #
        # The hygiene is why this family disables automatic optimization
        # despite driving a single optimizer. Gradients are inspected between
        # backward and the step: if any is not finite the step is abandoned
        # and the gradients cleared, so a diverged batch contributes nothing
        # rather than contributing a clipped version of a meaningless
        # direction. Clipping applies only on the surviving path, which makes
        # it a shaping constraint on valid updates rather than a rescue for
        # invalid ones.
        #
        # The schedule advances unconditionally, including after a skipped
        # step. That is deliberate and follows the reference: the warmup and
        # decay curve is defined against elapsed steps, not against successful
        # ones, so skipping does not stall the schedule.
        #
        # Args:
        #     batch: Collated mapping carrying a ``"waveform"`` tensor.
        #     batch_idx: Index within the epoch. Unused, since every batch
        #         follows the same path.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping or carries no waveform
        #         tensor.
        #
        # Returns:
        #     A mapping whose ``"loss"`` entry is the detached weighted total.
        #     It is reporting-only, since backward and the step have already
        #     run here.
        waveform: torch.Tensor = self._extract_waveform(batch)
        optimizer: torch.optim.Optimizer = self.optimizers()
        scheduler: torch.optim.lr_scheduler.LRScheduler = self.lr_schedulers()
        optimizer.zero_grad()
        losses: dict[str, torch.Tensor] = self._compute_flow_losses(waveform)
        self.manual_backward(losses["total"])
        if self._gradients_are_finite():
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self._configuration.gradient_clip_norm)
            optimizer.step()
        else:
            optimizer.zero_grad()
        scheduler.step()
        self.log("train_loss", losses["total"].detach())
        self.log("train_rectified_flow_loss", losses["rectified_flow"].detach())
        self.log("train_magnitude_loss", losses["magnitude"].detach())
        self.log("train_overlap_loss", losses["overlap"].detach())
        return {"loss": losses["total"].detach()}

    def _gradients_are_finite(self) -> bool:
        # Reports whether every accumulated gradient is finite, returning at
        # the first violation so a healthy step pays the full scan but a
        # diverged one exits early. Parameters without gradients are skipped
        # rather than treated as failures, which is correct because a
        # parameter untouched by this batch has nothing to inspect.
        for parameter in self.network.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                return False
        return True

    @override
    def validation_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        # Scores the same three-term objective used in training, without any
        # weight update, and publishes the total as val_loss for checkpoint
        # selection.
        #
        # This measure is stochastic in a way the other families' validation
        # is not: it samples a fresh noise endpoint and a fresh path position
        # on every call, so two evaluations of the same batch differ. It is
        # therefore a noisy estimate averaged over the validation set rather
        # than a deterministic score, and it measures the objective rather
        # than synthesis quality, since no integration is performed here at
        # all.
        #
        # Args:
        #     batch: Collated mapping carrying a ``"waveform"`` tensor.
        #     batch_idx: Index within the validation pass. Unused.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping or carries no waveform
        #         tensor.
        #
        # Returns:
        #     A mapping whose ``"loss"`` entry is the detached weighted total.
        waveform: torch.Tensor = self._extract_waveform(batch)
        with torch.no_grad():
            losses: dict[str, torch.Tensor] = self._compute_flow_losses(waveform)
        validation_loss: torch.Tensor = losses["total"].detach()
        self.log("val_loss", validation_loss)
        self.log("val_rectified_flow_loss", losses["rectified_flow"].detach())
        self.log("val_magnitude_loss", losses["magnitude"].detach())
        self.log("val_overlap_loss", losses["overlap"].detach())
        return {"loss": validation_loss}

    @override
    def predict_step(self, batch: Batch, batch_idx: int) -> ModelOutput:
        # Synthesizes waveforms for the prediction stage and for real-time
        # factor measurement. This is the only step that runs the integration
        # loop, so it is where this family's inference cost is actually
        # incurred and where a change to the sampling step count becomes
        # visible in both quality and timing.
        #
        # Args:
        #     batch: Collated mapping carrying a ``"waveform"`` tensor, from
        #         which the conditioning mel is derived.
        #     batch_idx: Index within the prediction pass. Unused.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping or carries no waveform
        #         tensor.
        #
        # Returns:
        #     A mapping under the single key ``"synthesized_waveform"``.
        waveform: torch.Tensor = self._extract_waveform(batch)
        reference_mel: torch.Tensor = self._compute_full_precision_mel(waveform)
        synthesized_waveform: torch.Tensor = self._sampler.synthesize(self.network, reference_mel)
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
        # Declares the single optimizer and its cosine schedule with linear
        # warmup. The schedule is stepped by the training step rather than by
        # the loop, because a pre-built torch scheduler is filed with a null
        # configuration and never advanced automatically; the reference
        # advances it per optimizer step rather than per epoch, which is why
        # the stepping lives beside the optimizer step.
        #
        # Returns:
        #     An OptimizationConfiguration holding one already-constructed
        #     optimizer and one already-constructed schedule.
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self._configuration.learning_rate,
            betas=(self._configuration.adam_beta_1, self._configuration.adam_beta_2)
        )
        warmup_steps: int = self._configuration.scheduler_warmup_steps
        total_steps: int = self._configuration.scheduler_total_steps

        def cosine_warmup_factor(step: int) -> float:
            # Reproduces the reference schedule as a multiplier on the base
            # rate: a linear rise over the warmup interval, then a cosine
            # decay from full rate to zero across the remaining planned steps.
            # Progress past the planned total is clamped, so training beyond
            # the plan holds at zero rather than reversing as the cosine would
            # otherwise do. The denominators are floored at one so a
            # degenerate schedule cannot divide by zero.
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress: float = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

        scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=cosine_warmup_factor
        )
        return OptimizationConfiguration(optimizer=optimizer, scheduler=scheduler)

    def _extract_waveform(self, batch: Batch) -> torch.Tensor:
        # Reads and type-checks the waveform from the collated batch mapping.
        # Every step in this module routes through here, so the batch contract
        # is enforced in exactly one place.
        #
        # Raises:
        #     TypeError: If the batch is not a mapping or its ``"waveform"``
        #         entry is not a tensor.
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
        waveform: Batch | None = batch.get("waveform")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("batch['waveform'] must be Tensor")
        return waveform

    def _compute_full_precision_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        # Computes the conditioning mel in float32 regardless of the trainer's
        # precision, then casts the result back to the caller's dtype so the
        # rest of the graph runs at whatever precision is configured. Fixing
        # the analysis precision keeps the conditioning protocol identical
        # across precision settings, so a run at reduced precision conditions
        # on the same mel a full-precision run would.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            mel: torch.Tensor = self._mel_spectrogram(waveform.float())
        return mel.type_as(waveform)
