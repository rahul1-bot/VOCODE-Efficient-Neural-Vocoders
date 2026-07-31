# This module:
# 1. Implements the FreeV training objective: the amplitude and
#    anti-wrapping phase supervision of the APNet2 lineage applied to the
#    prior-refined prediction, with the adversarial and feature-matching
#    terms, per the reference recipe
#
# Published composition:
# - Identical to APNet2 in every term and every default weight. What
#   distinguishes the family is the architecture that produces the tensors,
#   not the objective that scores them: FreeV predicts a refinement of an
#   analytic prior rather than the spectrum outright, and the same
#   supervision is applied to the refined result
#
# Report alignment:
# - The report records the APNet2, FreeV, and RNDVoC Project-Trained
#   Configurations as sharing one supervision description: log-amplitude,
#   anti-wrapping phase, and STFT-consistency terms with real and imaginary
#   reconstruction to mel, hinge adversarial, and feature-matching
#   supervision. Delegation is how this family satisfies that description
#   without a second implementation of it
#
# Design decisions:
# - The objective delegates to Apnet2Loss by composition rather than
#   duplicating it or subclassing it, so a correction to an APNet2 term
#   reaches this family automatically and the two cannot silently diverge
# - The configuration and spectrum records are declared as empty subclasses
#   rather than aliases, so the family owns nameable types for its own
#   configuration surface and error messages while inheriting every field
#   and default unchanged
# - The delegate is constructed from a field dump of the FreeV record, so
#   the wrapped Apnet2Loss always receives a genuine Apnet2LossConfig even
#   when the caller supplies a further subclass
#
# Author: Rahul Sawhney

import torch
from torch import nn

from vocode.losses.apnet2 import Apnet2Loss, Apnet2LossConfig, Apnet2Spectrum

__all__: list[str] = ["FreevLoss", "FreevLossConfig", "FreevSpectrum"]


class FreevLossConfig(Apnet2LossConfig):
    # Frozen weight record of the FreeV composition. It adds no field and
    # overrides no default, so every weight and every validation rule of
    # Apnet2LossConfig applies unchanged, including the frozen, strict, and
    # extra-forbidding model settings. The distinct type exists so this
    # family has its own name in configuration surfaces and validation
    # errors rather than reporting as the APNet2 record.
    pass


class FreevSpectrum(Apnet2Spectrum):
    # Frozen bundle of the four STFT views the FreeV objective supervises.
    # Like the configuration record it adds nothing to its base; it names
    # the family's spectrum transport while remaining substitutable
    # wherever an Apnet2Spectrum is expected, which is why the compute
    # methods below accept the base type in their signatures.
    pass


class FreevLoss(nn.Module):
    # The FreeV composite objective over amplitude, phase, consistency,
    # and adversarial terms. Every term is computed by a wrapped Apnet2Loss;
    # this class contributes the family's own types and a stable call
    # surface, not a different objective. It registers no parameters of its
    # own beyond what the delegate holds.
    #
    # Integration: the driving model constructs one instance from its
    # configuration and calls compute_discriminator_loss and then
    # compute_generator_loss once each per training batch under its own
    # optimizer toggles, and compute_validation_loss on the evaluation
    # path. The component panels are produced by the delegate, so their keys
    # are the APNet2 keys exactly.
    def __init__(self, configuration: FreevLossConfig) -> None:
        # Rebuilds the delegate's configuration from a field dump of the
        # FreeV record and constructs the wrapped Apnet2Loss around it, then
        # retains the original record for republication. Dumping and
        # reconstructing rather than passing the record through guarantees
        # the delegate holds a genuine Apnet2LossConfig even when the caller
        # supplies a further subclass.
        #
        # Args:
        #     configuration: The frozen weight record, retained by
        #         reference and republished unchanged through the
        #         configuration property. The delegate holds a separate,
        #         equal-valued instance, so identity comparison against the
        #         delegate's record fails while equality holds.
        super().__init__()
        apnet2_configuration: Apnet2LossConfig = Apnet2LossConfig(**configuration.model_dump())
        self._loss: Apnet2Loss = Apnet2Loss(apnet2_configuration)
        self._configuration: FreevLossConfig = configuration

    def compute_discriminator_loss(
        self,
        real_period_logits: list[torch.Tensor],
        fake_period_logits: list[torch.Tensor],
        real_resolution_logits: list[torch.Tensor],
        fake_resolution_logits: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Computes the discriminator objective for one GAN training step by
        # forwarding both ensembles' logits to the wrapped APNet2 objective,
        # which sums the period ensemble's hinge separation with the
        # resolution ensemble's under the configured resolution weight. The
        # returned scalar and component panel are the delegate's own.
        return self._loss.compute_discriminator_loss(
            real_period_logits=real_period_logits,
            fake_period_logits=fake_period_logits,
            real_resolution_logits=real_resolution_logits,
            fake_resolution_logits=fake_resolution_logits
        )

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
        # Computes the generator objective for one GAN training step by
        # forwarding every tensor to the wrapped APNet2 objective, which
        # supervises the amplitude, phase, spectrum, and waveform groups
        # under their configured weights. The reference_spectrum and
        # final_spectrum arguments are typed against the APNet2 record, so a
        # FreevSpectrum is accepted by substitution; the caller must still
        # ensure final_spectrum re-analyzes the same synthesis the candidate
        # tensors describe, since the consistency term compares the two.
        return self._loss.compute_generator_loss(
            reference_spectrum=reference_spectrum,
            candidate_log_amplitude=candidate_log_amplitude,
            candidate_phase=candidate_phase,
            candidate_real_spectrum=candidate_real_spectrum,
            candidate_imaginary_spectrum=candidate_imaginary_spectrum,
            final_spectrum=final_spectrum,
            reference_mel=reference_mel,
            candidate_mel=candidate_mel,
            fake_period_logits=fake_period_logits,
            fake_resolution_logits=fake_resolution_logits,
            real_period_features=real_period_features,
            fake_period_features=fake_period_features,
            real_resolution_features=real_resolution_features,
            fake_resolution_features=fake_resolution_features
        )

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
        # Computes validation loss without mutating optimizer state by
        # forwarding to the wrapped APNet2 objective, which drops the
        # adversarial and feature-matching terms and keeps the amplitude,
        # phase, spectrum, and mel supervision. The component panel is the
        # delegate's, so its entries carry the bare loss prefix rather than
        # a validation prefix.
        return self._loss.compute_validation_loss(
            reference_spectrum=reference_spectrum,
            candidate_log_amplitude=candidate_log_amplitude,
            candidate_phase=candidate_phase,
            candidate_real_spectrum=candidate_real_spectrum,
            candidate_imaginary_spectrum=candidate_imaginary_spectrum,
            final_spectrum=final_spectrum,
            reference_mel=reference_mel,
            candidate_mel=candidate_mel
        )

    @property
    def configuration(self) -> FreevLossConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration
