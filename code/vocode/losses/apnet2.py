# This module:
# 1. Implements the APNet2 training objective: log-amplitude regression,
#    the anti-wrapping phase losses (instantaneous, group-delay, and
#    integrated), STFT consistency, and the adversarial and
#    feature-matching terms, per the reference recipe
# 2. Provides the spectrum analyzer producing the STFT views those terms
#    compare
#
# Published composition:
# - Generator: ``45 * MSE(log-amplitude) + 100 * phase + 20 * spectrum +
#   waveform``, where the phase term is the sum of the instantaneous,
#   group-delay, and phase-time-difference anti-wrapping losses, the
#   spectrum term is ``consistency + 2.25 * (L1(real) + L1(imaginary))``,
#   and the waveform group is ``hinge_period + 0.1 * hinge_resolution +
#   FM_period + 0.1 * FM_resolution + 45 * L1(mel)``
# - Discriminator: ``hinge_period + 0.1 * hinge_resolution``
# - Validation: the same amplitude, phase, and spectrum terms plus
#   ``45 * L1(mel)``, with the adversarial and feature-matching terms
#   omitted
#
# Update semantics:
# - Two-phase: the driving model computes the discriminator scalar against
#   the detached synthesis and steps the discriminator optimizer, then
#   computes the generator scalar through the same discriminators. This
#   class supplies all three scalars and owns no optimizer state
#
# Report alignment:
# - The report records the APNet2, FreeV, and RNDVoC Project-Trained
#   Configurations as adding log-amplitude, anti-wrapping phase, and
#   STFT-consistency terms with real and imaginary reconstruction to mel,
#   hinge adversarial, and feature-matching supervision. Every one of those
#   named terms appears below, and this module supplies the FreeV family's
#   terms as well through delegation
#
# Design decisions:
# - Anti-wrapping losses compare phase differences through a sawtooth
#   projection, so a full-turn offset costs nothing while local phase
#   errors are penalized. Without this projection a phase of nearly two pi
#   and a phase of nearly zero would read as maximally different despite
#   describing the same angle
# - Phase is supervised along three axes at once: the instantaneous
#   difference constrains each bin directly, the group-delay difference
#   constrains how phase advances across frequency, and the
#   phase-time-difference term constrains how it advances across time. The
#   two derivative views are what make the objective sensitive to the
#   structure of the phase surface rather than only to its pointwise value
# - The adversarial form is the clamped hinge, not the least-squares form
#   of the HiFi-GAN family, and the multi-resolution ensemble enters at a
#   tenth of the multi-period ensemble on both the adversarial and the
#   feature-matching paths
# - Paired spectral tensors are cropped to their common extent rather than
#   being required to agree exactly, because the predicted and analyzed
#   spectra can differ by a boundary frame under the framing settings; the
#   mel term keeps the strict guard, since a mismatch there indicates a
#   genuine protocol disagreement
#
# Author: Rahul Sawhney

from typing import ClassVar

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat
from torch import nn

from vocode.losses.feature_matching import FeatureMatchingLoss
from vocode.losses.mel_reconstruction import MelReconstructionLoss

__all__: list[str] = ["Apnet2Loss", "Apnet2LossConfig", "Apnet2Spectrum", "Apnet2SpectrumAnalyzer"]


class Apnet2LossConfig(BaseModel):
    # Frozen loss weights of the reference composition.
    #
    # Fields:
    #     amplitude_weight: Multiplier on the mean squared error between
    #         the reference and predicted log amplitude spectra.
    #         Default: ``45.0``.
    #     phase_weight: Multiplier on the summed anti-wrapping phase term.
    #         It is the largest weight in the composition because the three
    #         phase losses are bounded by pi and would otherwise be dwarfed
    #         by the amplitude and spectral terms. Default: ``100.0``.
    #     spectrum_weight: Multiplier on the combined consistency and
    #         real-imaginary term. Default: ``20.0``.
    #     mel_weight: Multiplier on the L1 log-mel reconstruction term,
    #         applied on both the training and validation paths.
    #         Default: ``45.0``.
    #     resolution_discriminator_weight: Multiplier on the
    #         multi-resolution ensemble's hinge term, applied on the
    #         discriminator path and again to the generator's adversarial
    #         term. Default: ``0.1``.
    #     resolution_feature_matching_weight: Multiplier on the
    #         multi-resolution ensemble's feature-matching term. It is
    #         separate from the discriminator weight so the two resolution
    #         contributions can be tuned independently, even though both
    #         default to the same value. Default: ``0.1``.
    #     real_imaginary_weight: Multiplier on the summed L1 errors of the
    #         real and imaginary spectra inside the spectrum term, applied
    #         before the outer spectrum weight. Default: ``2.25``.
    #
    # Every field is a PositiveFloat on a strict, extra-forbidding, frozen
    # model, so a weight may be attenuated but never zeroed, negated, or
    # misspelled into silent acceptance, and the record cannot drift after
    # the run has logged it.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    amplitude_weight: PositiveFloat = 45.0
    phase_weight: PositiveFloat = 100.0
    spectrum_weight: PositiveFloat = 20.0
    mel_weight: PositiveFloat = 45.0
    resolution_discriminator_weight: PositiveFloat = 0.1
    resolution_feature_matching_weight: PositiveFloat = 0.1
    real_imaginary_weight: PositiveFloat = 2.25


class Apnet2Spectrum(BaseModel):
    # Frozen bundle of the four STFT views this objective supervises,
    # carried together so a caller cannot pass a phase from one analysis and
    # an amplitude from another. The analyzer produces these; the loss
    # consumes them as the reference side of its amplitude, phase, and
    # spectral terms.
    #
    # Fields:
    #     log_amplitude: Natural logarithm of the magnitude spectrum with
    #         an epsilon offset, supervised by the amplitude term.
    #     phase: Principal-value phase in radians from atan2, supervised
    #         by the three anti-wrapping terms.
    #     real_spectrum: Real component of the complex STFT.
    #     imaginary_spectrum: Imaginary component of the complex STFT.
    #
    # The model permits arbitrary types so torch tensors can be fields, and
    # is frozen so an analyzed reference cannot be mutated between the
    # training and validation paths that share it. Unlike the weight
    # records it is neither strict nor extra-forbidding, because it is an
    # internal transport structure rather than a user-facing setting.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    log_amplitude: torch.Tensor
    phase: torch.Tensor
    real_spectrum: torch.Tensor
    imaginary_spectrum: torch.Tensor


class Apnet2SpectrumAnalyzer:
    # Produces the four STFT views the APNet2 objective compares. This is
    # the analysis half of the family: the loss never computes a transform
    # itself, so both the reference spectra and any re-analyzed synthesis
    # pass through this one class and therefore share exactly one framing
    # protocol. It is a plain object rather than an nn.Module because it
    # holds no parameters and participates in no state dict.
    #
    # Integration: the driving model constructs one analyzer from its own
    # framing settings and reuses it for every batch. RndvocLoss reaches the
    # same class through its spectrum_analyzer_type property, so the RNDVoC
    # family analyzes audio identically without duplicating the transform.
    def __init__(self, n_fft: int, hop_size: int, win_size: int) -> None:
        # Binds the framing settings and the amplitude epsilon. The three
        # settings must match those the driving model uses elsewhere, since
        # nothing here can detect a disagreement with the model's own
        # spectra.
        #
        # Args:
        #     n_fft: Transform size in samples.
        #     hop_size: Advance between successive frames in samples.
        #     win_size: Hann window length in samples.
        self._n_fft: int = n_fft
        self._hop_size: int = hop_size
        self._win_size: int = win_size
        self._epsilon: float = 1e-5

    def analyze(self, waveform: torch.Tensor) -> Apnet2Spectrum:
        # Computes spectral analysis terms required by the loss function.
        # The waveform is first normalized to a two-dimensional batch
        # layout, then transformed with a centred Hann-windowed STFT. The
        # magnitude is offset by a fixed epsilon before its logarithm so a
        # silent bin yields a large negative value rather than negative
        # infinity, and the phase is taken as the principal value through
        # atan2, which is why every downstream phase comparison must go
        # through the anti-wrapping projection.
        #
        # Args:
        #     waveform: Audio as ``[time]``, ``[batch, time]``, or
        #         ``[batch, 1, time]``.
        #
        # Raises:
        #     ValueError: If the waveform has any other rank or a channel
        #         axis wider than one; the message reports the observed
        #         shape.
        #
        # Returns:
        #     An Apnet2Spectrum carrying the log amplitude, principal-value
        #     phase, and the real and imaginary components of one analysis.
        #     The Hann window is rebuilt on the waveform's device at every
        #     call rather than cached, so the analyzer follows the audio
        #     across devices without holding a buffer.
        prepared_waveform: torch.Tensor = self._prepare_waveform(waveform)
        stft: torch.Tensor = torch.stft(
            prepared_waveform,
            self._n_fft,
            hop_length=self._hop_size,
            win_length=self._win_size,
            window=torch.hann_window(self._win_size, device=prepared_waveform.device),
            center=True,
            return_complex=True
        )
        real_spectrum: torch.Tensor = stft.real
        imaginary_spectrum: torch.Tensor = stft.imag
        log_amplitude: torch.Tensor = torch.log(
            torch.sqrt(real_spectrum.pow(2) + imaginary_spectrum.pow(2)) + self._epsilon
        )
        phase: torch.Tensor = torch.atan2(imaginary_spectrum, real_spectrum)
        return Apnet2Spectrum(
            log_amplitude=log_amplitude,
            phase=phase,
            real_spectrum=real_spectrum,
            imaginary_spectrum=imaginary_spectrum
        )

    def _prepare_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        # Normalizes the input into the layout the next stage consumes,
        # which is the two-dimensional ``[batch, time]`` form torch.stft
        # accepts. An unbatched waveform gains a batch axis, an already
        # batched one passes through, and a single-channel three-dimensional
        # waveform has its channel axis squeezed out. A multi-channel input
        # is rejected rather than mixed down, because silently collapsing
        # channels would change what is being supervised.
        if waveform.ndim == 1:
            return waveform.unsqueeze(0)
        if waveform.ndim == 2:
            return waveform
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            return waveform.squeeze(1)
        raise ValueError(f"Expected waveform shape [time], [batch, time], or [batch, 1, time], got {tuple(waveform.shape)}")


class _HingeGanLoss(nn.Module):
    # Private hinge adversarial objective for the APNet2 composition,
    # holding both halves of the game over ensemble logit lists. It differs
    # from the package's shared LeastSquaresGanLoss in using a clamped hinge
    # rather than a squared error, and from the Vocos hinge in summing the
    # per-member terms rather than averaging them; the APNet2 recipe
    # controls ensemble balance through its configured resolution weight
    # instead. The class holds no parameters and no mutable state.
    def compute_generator_loss(self, fake_logits: list[torch.Tensor]) -> torch.Tensor:
        # Computes the generator objective for one GAN training step by
        # accumulating ``mean(clamp(1 - logit, min=0))`` over the ensemble.
        # A member whose logits already reach one contributes nothing, so
        # the term saturates instead of rewarding ever-larger logits.
        #
        # Args:
        #     fake_logits: One logit tensor per sub-discriminator, evaluated
        #         on synthesized audio.
        #
        # Returns:
        #     A scalar tensor holding the summed hinge term, placed on the
        #     device of the first logit. An empty list returns a CPU zero
        #     scalar.
        loss: torch.Tensor = torch.zeros((), device=fake_logits[0].device) if fake_logits else torch.zeros(())
        for fake_logit in fake_logits:
            loss: torch.Tensor = loss + torch.mean(torch.clamp(1.0 - fake_logit, min=0.0))
        return loss

    def compute_discriminator_loss(
        self,
        real_logits: list[torch.Tensor],
        fake_logits: list[torch.Tensor]
    ) -> torch.Tensor:
        # Computes the discriminator objective for one GAN training step.
        # Each aligned pair contributes ``mean(clamp(1 - real, min=0)) +
        # mean(clamp(1 + fake, min=0))``, so a member pays nothing once it
        # scores reference audio at or above one and synthesized audio at or
        # below minus one.
        #
        # Args:
        #     real_logits: One logit tensor per sub-discriminator, evaluated
        #         on reference audio; also fixes the accumulator's device.
        #     fake_logits: The matching logit tensors evaluated on the
        #         detached synthesis.
        #
        # Raises:
        #     ValueError: If the two lists disagree in length; the pairing
        #         uses strict zip, so a member-count mismatch fails closed.
        #
        # Returns:
        #     A scalar tensor holding the summed separation term. An empty
        #     real_logits list returns a CPU zero scalar.
        loss: torch.Tensor = torch.zeros((), device=real_logits[0].device) if real_logits else torch.zeros(())
        for real_logit, fake_logit in zip(real_logits, fake_logits, strict=True):
            real_loss: torch.Tensor = torch.mean(torch.clamp(1.0 - real_logit, min=0.0))
            fake_loss: torch.Tensor = torch.mean(torch.clamp(1.0 + fake_logit, min=0.0))
            loss: torch.Tensor = loss + real_loss + fake_loss
        return loss


class _PhaseLoss(nn.Module):
    # Private anti-wrapping phase objective of the APNet2 lineage,
    # returning the three views the composition sums: the instantaneous
    # phase error, the group-delay error along the frequency axis, and the
    # phase-time-difference error along the frame axis. Supervising the two
    # derivative views alongside the pointwise one is what makes the
    # objective sensitive to the shape of the phase surface rather than
    # only to its values. The class holds the two-pi constant and no other
    # state, and registers no parameters.
    def __init__(self) -> None:
        # Binds the full-turn constant used by the anti-wrapping
        # projection.
        super().__init__()
        self._two_pi: float = 6.283185307179586

    def forward(self, reference_phase: torch.Tensor, candidate_phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Computes this criterion's value for the given inputs. The two
        # phase surfaces are first cropped to their common extent. Two
        # first-order difference matrices are then built, one sized to the
        # frequency axis and one to the frame axis, and applied by
        # multiplication: the frequency operator acts on the transposed
        # surface to form group delay, and the frame operator acts directly
        # to form the phase time difference. Each of the three comparisons
        # is reduced through the anti-wrapping projection and averaged, so
        # every term is a mean angular error in radians bounded by pi.
        #
        # Args:
        #     reference_phase: Principal-value phase of the reference
        #         audio, laid out as ``[batch, bins, frames]``; its cropped
        #         extent sizes both difference matrices.
        #     candidate_phase: The predicted phase in the same layout.
        #
        # Returns:
        #     The instantaneous, group-delay, and phase-time-difference
        #     losses as three scalar tensors in that order. The caller sums
        #     them before applying the single configured phase weight, so
        #     the three views are equally weighted by construction.
        reference_phase, candidate_phase = self._crop_pair(reference_phase, candidate_phase)
        bin_count: int = reference_phase.shape[1]
        frame_count: int = reference_phase.shape[2]
        gd_matrix: torch.Tensor = self._difference_matrix(bin_count, reference_phase.device)
        ptd_matrix: torch.Tensor = self._difference_matrix(frame_count, reference_phase.device)
        reference_group_delay: torch.Tensor = torch.matmul(reference_phase.permute(0, 2, 1), gd_matrix)
        candidate_group_delay: torch.Tensor = torch.matmul(candidate_phase.permute(0, 2, 1), gd_matrix)
        reference_phase_time_difference: torch.Tensor = torch.matmul(reference_phase, ptd_matrix)
        candidate_phase_time_difference: torch.Tensor = torch.matmul(candidate_phase, ptd_matrix)
        instantaneous_phase_loss: torch.Tensor = torch.mean(self._anti_wrap(reference_phase - candidate_phase))
        group_delay_loss: torch.Tensor = torch.mean(
            self._anti_wrap(reference_group_delay - candidate_group_delay)
        )
        phase_time_difference_loss: torch.Tensor = torch.mean(
            self._anti_wrap(reference_phase_time_difference - candidate_phase_time_difference)
        )
        return instantaneous_phase_loss, group_delay_loss, phase_time_difference_loss

    def _difference_matrix(self, size: int, device: torch.device) -> torch.Tensor:
        # Builds the finite-difference matrix used by the APNet2 phase objective.
        # Subtracting the two upper triangles leaves ones on the first
        # superdiagonal alone, and subtracting the identity puts minus one
        # on the diagonal. Right-multiplying a surface by this operator
        # therefore replaces each position along the multiplied axis with
        # the difference between its predecessor and itself, giving the
        # first-order derivative view the group-delay and
        # phase-time-difference terms compare. The matrix is rebuilt on the
        # caller's device at every invocation rather than cached, so the
        # component follows the audio across devices without a buffer.
        upper_first: torch.Tensor = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        upper_second: torch.Tensor = torch.triu(torch.ones(size, size, device=device), diagonal=2)
        return upper_first - upper_second - torch.eye(size, device=device)

    def _anti_wrap(self, phase_difference: torch.Tensor) -> torch.Tensor:
        # Applies phase anti-wrapping so angular differences remain numerically stable.
        # Rounding the difference to the nearest whole turn and subtracting
        # that turn projects any angular error into the half-open interval
        # around zero, and the absolute value then bounds it by pi. A
        # difference of very nearly two pi therefore costs almost nothing,
        # which is correct because it describes the same angle; without
        # this projection such a pair would read as maximally wrong and the
        # gradient would push the phase in the wrong direction.
        return torch.abs(
            phase_difference - torch.round(phase_difference / self._two_pi) * self._two_pi
        )

    def _crop_pair(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Crops paired tensors to a shared shape before loss computation.
        # The two trailing axes are truncated to their common extent, so a
        # predicted spectrum that carries one boundary frame more or fewer
        # than the analyzed reference is still comparable. This is
        # deliberately more permissive than the mel reconstruction guard,
        # which rejects any disagreement: a spectral off-by-one is a framing
        # boundary effect, whereas a mel disagreement indicates the two
        # spectrograms were extracted under different protocols.
        minimum_bins: int = min(first.shape[-2], second.shape[-2])
        minimum_frames: int = min(first.shape[-1], second.shape[-1])
        return first[..., :minimum_bins, :minimum_frames], second[..., :minimum_bins, :minimum_frames]


class Apnet2Loss(nn.Module):
    # The APNet2 composite objective over amplitude, phase, consistency,
    # and adversarial terms. This is the richest composition in the
    # package: the generator is supervised in four distinct domains at
    # once, because the family predicts amplitude and phase as separate
    # streams and must therefore constrain each of them, their recombined
    # complex spectrum, and the waveform that spectrum implies. The
    # instance owns the frozen weight record and four stateless component
    # losses, and registers no parameters of its own, so despite
    # subclassing nn.Module it contributes nothing to the driving model's
    # optimizer state.
    #
    # Integration: the driving model constructs one instance from its
    # configuration, calls compute_discriminator_loss and then
    # compute_generator_loss once each per training batch under its own
    # optimizer toggles, and calls compute_validation_loss on the
    # evaluation path. FreevLoss wraps this class rather than reimplementing
    # it, so the FreeV family inherits every term unchanged.
    def __init__(self, configuration: Apnet2LossConfig) -> None:
        # Binds the frozen weight record and constructs the phase, hinge,
        # feature-matching, and mel component losses. Each is stateless, so
        # one instance of each serves both ensembles and both the training
        # and validation paths.
        #
        # Args:
        #     configuration: The frozen weight record, retained by
        #         reference and republished unchanged through the
        #         configuration property.
        super().__init__()
        self._configuration: Apnet2LossConfig = configuration
        self._phase_loss: _PhaseLoss = _PhaseLoss()
        self._hinge_loss: _HingeGanLoss = _HingeGanLoss()
        self._feature_matching_loss: FeatureMatchingLoss = FeatureMatchingLoss()
        self._mel_reconstruction_loss: MelReconstructionLoss = MelReconstructionLoss()

    def compute_discriminator_loss(
        self,
        real_period_logits: list[torch.Tensor],
        fake_period_logits: list[torch.Tensor],
        real_resolution_logits: list[torch.Tensor],
        fake_resolution_logits: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes the discriminator objective for one GAN training step as
        # the period ensemble's hinge separation plus the resolution
        # ensemble's, the latter scaled by the configured resolution
        # weight. The generator path applies that same weight to its own
        # resolution adversarial term, so the two ensembles stay in a fixed
        # ratio on both sides of the game.
        #
        # Args:
        #     real_period_logits: Multi-period logits for the reference
        #         audio.
        #     fake_period_logits: Multi-period logits for the synthesis,
        #         which the caller detaches before the discriminator forward
        #         pass so this term cannot reach the generator.
        #     real_resolution_logits: Multi-resolution logits for the
        #         reference audio.
        #     fake_resolution_logits: Multi-resolution logits for the
        #         detached synthesis.
        #
        # Raises:
        #     ValueError: If either ensemble's real and fake lists disagree
        #         in length, raised by the strict pairing inside the hinge
        #         component.
        #
        # Returns:
        #     The weighted total as a graph-carrying scalar tensor, paired
        #     with a component panel of detached Python floats keyed
        #     discriminator_loss_total, discriminator_loss_period, and
        #     discriminator_loss_resolution; the two per-ensemble entries
        #     are reported before the resolution weight is applied.
        period_loss: torch.Tensor = self._hinge_loss.compute_discriminator_loss(real_period_logits, fake_period_logits)
        resolution_loss: torch.Tensor = self._hinge_loss.compute_discriminator_loss(
            real_resolution_logits,
            fake_resolution_logits
        )
        total_loss: torch.Tensor = period_loss + self._configuration.resolution_discriminator_weight * resolution_loss
        components: dict[str, float] = {
            "discriminator_loss_total": float(total_loss.detach().item()),
            "discriminator_loss_period": float(period_loss.detach().item()),
            "discriminator_loss_resolution": float(resolution_loss.detach().item())
        }
        return total_loss, components

    def compute_generator_loss(
        self,
        reference_spectrum: Apnet2Spectrum,
        candidate_log_amplitude: torch.Tensor,
        candidate_phase: torch.Tensor,
        candidate_real_spectrum: torch.Tensor,
        candidate_imaginary_spectrum: torch.Tensor,
        final_spectrum: Apnet2Spectrum,
        reference_mel: torch.Tensor,
        candidate_mel: torch.Tensor,
        fake_period_logits: list[torch.Tensor],
        fake_resolution_logits: list[torch.Tensor],
        real_period_features: list[list[torch.Tensor]],
        fake_period_features: list[list[torch.Tensor]],
        real_resolution_features: list[list[torch.Tensor]],
        fake_resolution_features: list[list[torch.Tensor]]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes the generator objective for one GAN training step across
        # all four supervised domains. The amplitude term regresses the log
        # magnitude directly. The phase term sums the three anti-wrapping
        # views. The spectrum term combines STFT consistency, which measures
        # whether the predicted complex spectrum corresponds to any real
        # waveform, with the weighted L1 errors of its real and imaginary
        # parts against the reference. The waveform group then carries the
        # two ensembles' hinge and feature-matching terms together with the
        # mel term, and the four groups are scaled by their configured
        # weights to form the total.
        #
        # Args:
        #     reference_spectrum: Analyzed STFT views of the reference
        #         audio, supplying the amplitude, phase, real, and
        #         imaginary regression targets.
        #     candidate_log_amplitude: The predicted log amplitude stream.
        #     candidate_phase: The predicted phase stream.
        #     candidate_real_spectrum: Real part of the complex spectrum
        #         recombined from the two predicted streams.
        #     candidate_imaginary_spectrum: Imaginary part of that same
        #         recombined spectrum.
        #     final_spectrum: Re-analysis of the waveform actually
        #         synthesized from the recombined spectrum. The consistency
        #         term compares the recombined spectrum against this, so
        #         the two arguments must come from the same synthesis.
        #     reference_mel: Log-mel spectrogram of the reference audio.
        #     candidate_mel: Log-mel spectrogram of the synthesis; must
        #         match reference_mel in shape exactly.
        #     fake_period_logits: Multi-period logits for the synthesis,
        #         carrying gradient back to the generator.
        #     fake_resolution_logits: Multi-resolution logits for the
        #         synthesis.
        #     real_period_features: Multi-period intermediate activations
        #         for the reference audio; detached inside the matching
        #         term.
        #     fake_period_features: The corresponding activations for the
        #         synthesis.
        #     real_resolution_features: Multi-resolution intermediate
        #         activations for the reference audio; likewise detached.
        #     fake_resolution_features: The corresponding activations for
        #         the synthesis.
        #
        # Raises:
        #     ValueError: If reference_mel and candidate_mel disagree in
        #         shape. The spectral arguments carry no such requirement
        #         because they are cropped to their common extent.
        #
        # Returns:
        #     The weighted total as a graph-carrying scalar tensor, paired
        #     with a nine-entry component panel keyed generator_loss_total,
        #     the four domain entries (amplitude, phase, spectrum, mel), and
        #     the four per-ensemble adversarial and feature-matching
        #     entries. Every per-term entry is reported before its
        #     configured weight is applied.
        amplitude_loss: torch.Tensor = torch.nn.functional.mse_loss(
            *self._crop_pair(reference_spectrum.log_amplitude, candidate_log_amplitude)
        )
        instantaneous_phase_loss, group_delay_loss, phase_time_difference_loss = self._phase_loss(
            reference_spectrum.phase,
            candidate_phase
        )
        phase_loss: torch.Tensor = instantaneous_phase_loss + group_delay_loss + phase_time_difference_loss
        consistency_loss: torch.Tensor = self._stft_consistency_loss(
            candidate_real_spectrum,
            final_spectrum.real_spectrum,
            candidate_imaginary_spectrum,
            final_spectrum.imaginary_spectrum
        )
        real_loss: torch.Tensor = torch.nn.functional.l1_loss(
            *self._crop_pair(reference_spectrum.real_spectrum, candidate_real_spectrum)
        )
        imaginary_loss: torch.Tensor = torch.nn.functional.l1_loss(
            *self._crop_pair(reference_spectrum.imaginary_spectrum, candidate_imaginary_spectrum)
        )
        spectrum_loss: torch.Tensor = consistency_loss + self._configuration.real_imaginary_weight * (
            real_loss + imaginary_loss
        )
        adversarial_period: torch.Tensor = self._hinge_loss.compute_generator_loss(fake_period_logits)
        adversarial_resolution: torch.Tensor = self._hinge_loss.compute_generator_loss(fake_resolution_logits)
        feature_matching_period: torch.Tensor = self._feature_matching_loss(real_period_features, fake_period_features)
        feature_matching_resolution: torch.Tensor = self._feature_matching_loss(
            real_resolution_features,
            fake_resolution_features
        )
        mel_loss: torch.Tensor = self._mel_reconstruction_loss(reference_mel, candidate_mel)
        waveform_loss: torch.Tensor = (
            adversarial_period
            + self._configuration.resolution_discriminator_weight * adversarial_resolution
            + feature_matching_period
            + self._configuration.resolution_feature_matching_weight * feature_matching_resolution
            + self._configuration.mel_weight * mel_loss
        )
        total_loss: torch.Tensor = (
            self._configuration.amplitude_weight * amplitude_loss
            + self._configuration.phase_weight * phase_loss
            + self._configuration.spectrum_weight * spectrum_loss
            + waveform_loss
        )
        components: dict[str, float] = {
            "generator_loss_total": float(total_loss.detach().item()),
            "generator_loss_amplitude": float(amplitude_loss.detach().item()),
            "generator_loss_phase": float(phase_loss.detach().item()),
            "generator_loss_spectrum": float(spectrum_loss.detach().item()),
            "generator_loss_mel": float(mel_loss.detach().item()),
            "generator_loss_adversarial_period": float(adversarial_period.detach().item()),
            "generator_loss_adversarial_resolution": float(adversarial_resolution.detach().item()),
            "generator_loss_feature_matching_period": float(feature_matching_period.detach().item()),
            "generator_loss_feature_matching_resolution": float(feature_matching_resolution.detach().item())
        }
        return total_loss, components

    def compute_validation_loss(
        self,
        reference_spectrum: Apnet2Spectrum,
        candidate_log_amplitude: torch.Tensor,
        candidate_phase: torch.Tensor,
        candidate_real_spectrum: torch.Tensor,
        candidate_imaginary_spectrum: torch.Tensor,
        final_spectrum: Apnet2Spectrum,
        reference_mel: torch.Tensor,
        candidate_mel: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes validation loss without mutating optimizer state. The
        # amplitude, phase, and spectrum terms are computed exactly as on
        # the training path; only the adversarial and feature-matching
        # terms are dropped, and the mel term is applied directly at its own
        # weight rather than inside the waveform group. The reported value
        # therefore stays comparable across checkpoints without depending on
        # how well the discriminators happen to be trained at that moment.
        #
        # Args:
        #     reference_spectrum: Analyzed STFT views of the reference
        #         audio.
        #     candidate_log_amplitude: The predicted log amplitude stream.
        #     candidate_phase: The predicted phase stream.
        #     candidate_real_spectrum: Real part of the recombined complex
        #         spectrum.
        #     candidate_imaginary_spectrum: Imaginary part of that spectrum.
        #     final_spectrum: Re-analysis of the synthesized waveform,
        #         against which consistency is measured.
        #     reference_mel: Log-mel spectrogram of the reference audio.
        #     candidate_mel: Log-mel spectrogram of the synthesis; must
        #         match reference_mel in shape exactly.
        #
        # Raises:
        #     ValueError: If reference_mel and candidate_mel disagree in
        #         shape; the same guard applies on the evaluation path as in
        #         training.
        #
        # Returns:
        #     The weighted total as a scalar tensor, paired with a
        #     five-entry component panel keyed loss_total, loss_amplitude,
        #     loss_phase, loss_spectrum, and loss_mel. These keys use a bare
        #     loss prefix rather than the validation prefix used by the
        #     Vocos and HiFTNet families, but they remain distinct from this
        #     class's own generator keys, so the two paths never collide in
        #     the logged metric buffer.
        amplitude_loss: torch.Tensor = torch.nn.functional.mse_loss(
            *self._crop_pair(reference_spectrum.log_amplitude, candidate_log_amplitude)
        )
        instantaneous_phase_loss, group_delay_loss, phase_time_difference_loss = self._phase_loss(
            reference_spectrum.phase,
            candidate_phase
        )
        phase_loss: torch.Tensor = instantaneous_phase_loss + group_delay_loss + phase_time_difference_loss
        consistency_loss: torch.Tensor = self._stft_consistency_loss(
            candidate_real_spectrum,
            final_spectrum.real_spectrum,
            candidate_imaginary_spectrum,
            final_spectrum.imaginary_spectrum
        )
        real_loss: torch.Tensor = torch.nn.functional.l1_loss(
            *self._crop_pair(reference_spectrum.real_spectrum, candidate_real_spectrum)
        )
        imaginary_loss: torch.Tensor = torch.nn.functional.l1_loss(
            *self._crop_pair(reference_spectrum.imaginary_spectrum, candidate_imaginary_spectrum)
        )
        spectrum_loss: torch.Tensor = consistency_loss + self._configuration.real_imaginary_weight * (
            real_loss + imaginary_loss
        )
        mel_loss: torch.Tensor = self._mel_reconstruction_loss(reference_mel, candidate_mel)
        total_loss: torch.Tensor = (
            self._configuration.amplitude_weight * amplitude_loss
            + self._configuration.phase_weight * phase_loss
            + self._configuration.spectrum_weight * spectrum_loss
            + self._configuration.mel_weight * mel_loss
        )
        components: dict[str, float] = {
            "loss_total": float(total_loss.detach().item()),
            "loss_amplitude": float(amplitude_loss.detach().item()),
            "loss_phase": float(phase_loss.detach().item()),
            "loss_spectrum": float(spectrum_loss.detach().item()),
            "loss_mel": float(mel_loss.detach().item())
        }
        return total_loss, components

    def _stft_consistency_loss(
        self,
        first_real: torch.Tensor,
        second_real: torch.Tensor,
        first_imaginary: torch.Tensor,
        second_imaginary: torch.Tensor
    ) -> torch.Tensor:
        # Computes the STFT consistency term as the mean squared distance
        # between the predicted complex spectrum and the re-analysis of the
        # waveform synthesized from it. An arbitrary array of complex
        # numbers is not in general the STFT of any real signal, because
        # neighbouring frames overlap and must agree; this term measures
        # exactly that discrepancy, so minimizing it pushes the predicted
        # spectrum toward the set of realizable ones and keeps the
        # amplitude and phase streams mutually coherent.
        #
        # Args:
        #     first_real: Real part of the predicted spectrum.
        #     second_real: Real part of the re-analyzed synthesis.
        #     first_imaginary: Imaginary part of the predicted spectrum.
        #     second_imaginary: Imaginary part of the re-analyzed synthesis.
        #
        # Returns:
        #     A scalar tensor holding the mean squared complex distance,
        #     reduced over every axis after both pairs are cropped to their
        #     common extent.
        first_real, second_real = self._crop_pair(first_real, second_real)
        first_imaginary, second_imaginary = self._crop_pair(first_imaginary, second_imaginary)
        return torch.mean((first_real - second_real).pow(2) + (first_imaginary - second_imaginary).pow(2))

    def _crop_pair(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Crops paired tensors to a shared shape before loss computation.
        # The two trailing axes are truncated to their common extent, so a
        # predicted spectrum that carries one boundary frame more or fewer
        # than the analyzed reference is still comparable. This is
        # deliberately more permissive than the mel reconstruction guard,
        # which rejects any disagreement: a spectral off-by-one is a framing
        # boundary effect, whereas a mel disagreement indicates the two
        # spectrograms were extracted under different protocols.
        minimum_bins: int = min(first.shape[-2], second.shape[-2])
        minimum_frames: int = min(first.shape[-1], second.shape[-1])
        return first[..., :minimum_bins, :minimum_frames], second[..., :minimum_bins, :minimum_frames]

    @property
    def configuration(self) -> Apnet2LossConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration
