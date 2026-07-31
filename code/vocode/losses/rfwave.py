# This module:
# 1. Implements the RFWave training objective: the rectified-flow velocity
#    regression over noised band spectra plus the auxiliary magnitude and
#    overlap consistency terms, per the reference recipe
#
# Published composition:
# - ``velocity + 0.01 * (magnitude + overlap)``, where the velocity term is
#   measured in the waveform domain after time balancing
#
# Update semantics:
# - Single-phase: one optimizer step per batch, with no discriminator and
#   therefore no phase alternation. This family learns a velocity field
#   rather than playing an adversarial game, so none of the package's
#   adversarial or feature-matching machinery applies
#
# Report alignment:
# - The report records this Project-Trained Configuration as regressing
#   conditional flow velocities in the waveform domain with auxiliary
#   magnitude and band-overlap terms, which is the composition below. The
#   solver-step reduction the report evaluates as a deployment intervention
#   changes the executed iteration count at inference and leaves this
#   training objective untouched
#
# Design decisions:
# - Unlike every other objective in this package, this class does not
#   compose its own total. It publishes the four reductions as independent
#   methods and the driving model assembles the weighted sum, because the
#   velocity term needs an inverse STFT that only the model can perform and
#   the band structure the overlap term reads is the model's own
# - Time balancing divides both the prediction and the target by the square
#   root of the per-frame target variance, so frames of very different
#   energy contribute comparably to the velocity regression rather than
#   letting loud frames dominate
# - The magnitude term pairs a log-domain error with a Frobenius
#   convergence ratio, so it constrains both the relative shape of the
#   spectrum and its absolute scale; the ratio's denominator carries a
#   plus-one offset that keeps it finite for a silent target
# - The phase loss of the published recipe is disabled in the shipped
#   vocoder configuration and is therefore not implemented here at all,
#   rather than being implemented and weighted to zero
#
# Author: Rahul Sawhney

from typing import ClassVar

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat
from torch import nn
from torch.nn import functional

__all__: list[str] = ["RfwaveLoss", "RfwaveLossConfig"]


class RfwaveLossConfig(BaseModel):
    # Frozen loss weights of the reference composition.
    # The reference composition is rf_loss + 0.01 (stft_loss + overlap_loss); the phase
    # loss is disabled in the shipped vocoder configuration and is not implemented.
    #
    # Fields:
    #     auxiliary_weight: Multiplier applied jointly to the magnitude and
    #         overlap terms by the driving model. The velocity term always
    #         enters at one, so this weight sets the auxiliary terms'
    #         strength relative to it. Default: ``0.01``.
    #     log_epsilon: Floor applied to both magnitudes before the base-ten
    #         logarithm inside the magnitude term. It is a numerical floor
    #         rather than a weight: a silent frequency bin would otherwise
    #         take the logarithm of zero and produce a non-finite loss.
    #         Default: ``1e-8``.
    #
    # Both fields are PositiveFloat on a strict, extra-forbidding, frozen
    # model, so the auxiliary terms can be attenuated but never removed
    # through configuration, and the logarithm floor can never be zeroed.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    auxiliary_weight: PositiveFloat = 0.01
    log_epsilon: PositiveFloat = 1e-8


class RfwaveLoss(nn.Module):
    # Training objective component for the rectified-flow vocoder reproduction.
    # The velocity loss is measured in the waveform domain after time balancing (the
    # reference wave mode); magnitude and overlap terms enter at the auxiliary weight.
    # The instance holds only the frozen settings record and registers no
    # parameters, so despite subclassing nn.Module it contributes nothing to
    # the driving model's optimizer state.
    #
    # Integration: this class is a library of independent reductions rather
    # than a composer. The driving model calls time_balance, then obtains the
    # velocity-error waveform and the implied spectral endpoints from its own
    # network, then calls waveform_velocity_loss, magnitude_loss, and
    # overlap_loss, and finally assembles
    # ``velocity + auxiliary_weight * (magnitude + overlap)`` itself, reading
    # the weight from the configuration property. Nothing here enforces that
    # composition, so a caller that omits a term trains without it.
    def __init__(self, configuration: RfwaveLossConfig) -> None:
        # Binds the frozen settings record. No component losses are
        # constructed because every reduction in this family is inlined.
        #
        # Args:
        #     configuration: The frozen settings record, retained by
        #         reference and republished unchanged through the
        #         configuration property, from which the driving model reads
        #         the auxiliary weight when composing the total.
        super().__init__()
        self._configuration: RfwaveLossConfig = configuration

    def time_balance(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Normalizes prediction and target by the per-frame target variance (reference time balance).
        # The variance is taken along the feature axis independently for
        # every frame, and both tensors are divided by its square root, so a
        # frame's contribution to the subsequent velocity regression no
        # longer scales with that frame's energy. Only the target defines the
        # scale, so the operation is a pure rescaling of the comparison
        # rather than a normalization the prediction could game.
        #
        # Args:
        #     prediction: The network's predicted velocity field.
        #     target: The velocity target, which alone determines the
        #         per-frame scale.
        #
        # Returns:
        #     The rescaled prediction and target as a pair, in that order,
        #     both still carrying their input graphs. A variance floor of
        #     1e-6 is added before the square root, so an entirely constant
        #     frame rescales by a large but finite factor instead of
        #     dividing by zero.
        variance: torch.Tensor = target.var(dim=1, keepdim=True)
        scale: torch.Tensor = torch.sqrt(variance + 1e-6)
        return prediction / scale, target / scale

    def waveform_velocity_loss(self, placed_difference_waveform: torch.Tensor) -> torch.Tensor:
        # Computes the wave-mode rectified-flow loss from the inverse-STFT of the velocity error.
        # The reduction itself is a plain mean square; what makes it the
        # wave-mode variant is that the caller has already transformed the
        # balanced velocity error into the waveform domain, so the error is
        # measured where it will be heard rather than in the band-spectral
        # domain the network predicts in.
        #
        # Args:
        #     placed_difference_waveform: The inverse-STFT of the balanced
        #         velocity error, produced by the driving model's network
        #         from the outputs of time_balance.
        #
        # Returns:
        #     A scalar tensor holding the mean squared waveform error. This
        #     is the only term of the composition that enters unweighted.
        return placed_difference_waveform.pow(2.0).mean()

    def magnitude_loss(
        self,
        predicted_spectrum: torch.Tensor,
        target_spectrum: torch.Tensor
    ) -> torch.Tensor:
        # Computes the log10-magnitude MSE plus the Frobenius convergence term on implied endpoints.
        # Each spectrum arrives with its real and imaginary halves stacked
        # along the channel axis and is split back into the two components
        # before the magnitudes are formed. The two summed terms constrain
        # different things: the log-domain mean squared error is scale
        # invariant and therefore governs relative spectral shape, while the
        # Frobenius ratio compares absolute energies and therefore governs
        # overall level.
        #
        # Args:
        #     predicted_spectrum: The implied endpoint spectrum derived from
        #         the network's predicted velocity, real and imaginary
        #         halves stacked on the channel axis.
        #     target_spectrum: The corresponding implied endpoint spectrum
        #         derived from the velocity target, stacked identically.
        #
        # Returns:
        #     A scalar tensor holding the sum of the two terms. Both
        #     magnitudes are floored at the configured log epsilon before
        #     the logarithm, and the convergence ratio's denominator carries
        #     a plus-one offset, so a silent target yields a finite value
        #     rather than a division by zero.
        predicted_real: torch.Tensor
        predicted_imaginary: torch.Tensor
        predicted_real, predicted_imaginary = torch.chunk(predicted_spectrum, 2, dim=1)
        target_real: torch.Tensor
        target_imaginary: torch.Tensor
        target_real, target_imaginary = torch.chunk(target_spectrum, 2, dim=1)
        predicted_magnitude: torch.Tensor = torch.sqrt(predicted_real ** 2 + predicted_imaginary ** 2)
        target_magnitude: torch.Tensor = torch.sqrt(target_real ** 2 + target_imaginary ** 2)
        epsilon: float = self._configuration.log_epsilon
        log_magnitude_loss: torch.Tensor = functional.mse_loss(
            torch.log10(predicted_magnitude.clamp_min(epsilon)),
            torch.log10(target_magnitude.clamp_min(epsilon))
        )
        convergence_loss: torch.Tensor = (
            torch.norm(predicted_magnitude - target_magnitude, p="fro")
            / (torch.norm(target_magnitude, p="fro") + 1)
        )
        return log_magnitude_loss + convergence_loss

    def overlap_loss(
        self,
        band_predictions_real: list[torch.Tensor],
        band_predictions_imaginary: list[torch.Tensor],
        left_overlap: int,
        right_overlap: int
    ) -> torch.Tensor:
        # Penalizes disagreement between adjacent bands inside their shared overlap regions.
        # The network predicts the spectrum in overlapping frequency slabs,
        # so neighbouring slabs describe some of the same bins twice; this
        # term is what forces those duplicate descriptions to agree, without
        # which the recombined full-band spectrum would show seams at every
        # band boundary. The real and imaginary band lists are reduced
        # independently and their sum is averaged over the band count, so the
        # term does not grow with the number of bands the model uses.
        #
        # Args:
        #     band_predictions_real: Per-band real components, ordered from
        #         the lowest frequency band upward.
        #     band_predictions_imaginary: The matching imaginary components
        #         in the same band order.
        #     left_overlap: Width in bins of the region each band shares
        #         with its lower neighbour.
        #     right_overlap: Width in bins of the region each band shares
        #         with its upper neighbour.
        #
        # Returns:
        #     A scalar tensor holding the band-averaged overlap
        #     disagreement. Zero overlap widths make the term vanish, since
        #     every comparison is guarded on a positive width.
        real_loss: torch.Tensor = self._directional_overlap_loss(
            band_predictions_real,
            left_overlap,
            right_overlap
        )
        imaginary_loss: torch.Tensor = self._directional_overlap_loss(
            band_predictions_imaginary,
            left_overlap,
            right_overlap
        )
        return (real_loss + imaginary_loss) / len(band_predictions_real)

    def _directional_overlap_loss(
        self,
        band_predictions: list[torch.Tensor],
        left_overlap: int,
        right_overlap: int
    ) -> torch.Tensor:
        # Accumulates the reference left and right overlap MSE terms across the band list.
        # Every band is compared in both directions where a neighbour exists.
        # Looking left, the band's leading overlap bins are matched against
        # the trailing bins of the previous band's core, where the core is
        # that band with both of its own overlap margins removed. Looking
        # right, the band's trailing overlap bins are matched against the
        # leading bins of the next band's core. Comparing against the core
        # rather than the neighbour's own margin is what makes each shared
        # bin checked against an interior prediction instead of against
        # another boundary estimate.
        #
        # Args:
        #     band_predictions: One component's per-band predictions in
        #         ascending frequency order; all bands share a slab width.
        #     left_overlap: Width in bins of the lower-neighbour overlap.
        #         A width of zero disables every leftward comparison.
        #     right_overlap: Width in bins of the upper-neighbour overlap.
        #         A width of zero disables every rightward comparison.
        #
        # Returns:
        #     A scalar tensor holding the accumulated mean squared
        #     disagreement, seeded from the first band so it inherits that
        #     tensor's device and dtype. A single-band list yields zero,
        #     because neither neighbour exists.
        band_count: int = len(band_predictions)
        slab_width: int = band_predictions[0].size(1)
        accumulated: torch.Tensor = band_predictions[0].new_zeros(())
        for band_index in range(band_count):
            if band_index > 0 and left_overlap > 0:
                current_left: torch.Tensor = band_predictions[band_index][:, :left_overlap]
                previous_core: torch.Tensor = band_predictions[band_index - 1][:, left_overlap: slab_width - right_overlap]
                previous_right: torch.Tensor = previous_core[:, previous_core.size(1) - left_overlap:]
                accumulated: torch.Tensor = accumulated + functional.mse_loss(current_left, previous_right)
            if band_index < band_count - 1 and right_overlap > 0:
                current_right: torch.Tensor = band_predictions[band_index][:, slab_width - right_overlap:]
                next_core: torch.Tensor = band_predictions[band_index + 1][:, left_overlap: slab_width - right_overlap]
                next_left: torch.Tensor = next_core[:, :right_overlap]
                accumulated: torch.Tensor = accumulated + functional.mse_loss(current_right, next_left)
        return accumulated

    @property
    def configuration(self) -> RfwaveLossConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration
