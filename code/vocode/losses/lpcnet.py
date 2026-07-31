# This module:
# 1. Implements the LPCNet training objective: cross-entropy between the
#    predicted mu-law excitation distribution and the teacher-forced
#    target excitation, per the reference recipe
#
# Published composition:
# - The objective is a single term, the mean negative log-likelihood of the
#   target excitation under the network's predicted distribution. This
#   family has no adversarial, spectral, or feature-matching term at all
#
# Update semantics:
# - Single-phase: one generator-equivalent optimizer step per batch. There
#   is no discriminator and therefore no phase alternation, which is what
#   distinguishes this autoregressive family from every GAN family in the
#   package
#
# Report alignment:
# - The report records this Project-Trained Configuration as minimizing
#   teacher-forced cross entropy over its quantized mu-law excitation, which
#   is exactly the single term below. This is the only autoregressive
#   configuration in the cohort and the only one whose objective involves no
#   discriminator
#
# Design decisions:
# - The excitation is the mu-law companded residual between the target
#   sample and the linear prediction, so the network is supervised on what
#   the linear predictor could not explain rather than on the waveform
#   itself
# - The distribution is evaluated as a binary tree rather than a flat
#   256-way softmax: eight successive binary decisions address the tree's
#   255 internal nodes, so the likelihood of one sample costs eight gathers
#   instead of materializing a full 256-level table
# - Branch probabilities are floored at a configured epsilon before the
#   logarithm, because a node output of exactly zero on the target's path
#   would otherwise produce a non-finite loss and poison the run
#
# Author: Rahul Sawhney

from typing import ClassVar

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat
from torch import nn

from vocode.transforms.lpc import LpcnetMuLaw

__all__: list[str] = ["LpcnetLoss", "LpcnetLossConfig"]


class LpcnetLossConfig(BaseModel):
    # Frozen loss settings of the reference composition.
    #
    # Fields:
    #     probability_epsilon: Lower bound applied to every branch
    #         probability before its logarithm is taken. This is a
    #         numerical floor rather than a weight: it caps the penalty of
    #         one branch at ``-log(probability_epsilon)``, so a node that is
    #         confidently wrong on the target's path yields a large but
    #         finite loss instead of an infinity that would poison the run.
    #         Default: ``1e-8``.
    #
    # The record carries no term weights because this objective has exactly
    # one term. The field is a PositiveFloat on a strict, extra-forbidding,
    # frozen model, so the floor can be tightened but never set to zero.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    probability_epsilon: PositiveFloat = 1e-8


class LpcnetLoss(nn.Module):
    # The LPCNet training objective, and the one family in this package with
    # no adversarial component at all.
    # The loss is the reference excitation cross entropy: the mu-law residual between the
    # target sample and the linear prediction indexes the tree distribution, evaluated along
    # the target's eight-node binary path so the full 256-level table is never materialized.
    # The instance owns the frozen settings record and the companding
    # transform, and registers no parameters, so despite subclassing
    # nn.Module it contributes nothing to the driving model's optimizer
    # state.
    #
    # Integration: the driving model constructs one instance from its
    # configuration and calls compute_loss once per training batch, then
    # steps its single optimizer. There is no discriminator phase and no
    # separate validation composition; the same method serves both paths.
    def __init__(self, configuration: LpcnetLossConfig) -> None:
        # Binds the frozen settings record, constructs the mu-law companding
        # transform used to map residuals into the index domain, and fixes
        # the tree depth at eight levels, which is what makes the addressed
        # distribution 256-valued.
        #
        # Args:
        #     configuration: The frozen settings record, retained by
        #         reference and republished unchanged through the
        #         configuration property.
        super().__init__()
        self._configuration: LpcnetLossConfig = configuration
        self._mulaw: LpcnetMuLaw = LpcnetMuLaw()
        self._level_count: int = 8

    def compute_loss(
        self,
        target_samples: torch.Tensor,
        prediction: torch.Tensor,
        node_outputs: torch.Tensor
    ) -> torch.Tensor:
        # Computes the rounded excitation cross entropy over one teacher-forced sequence batch.
        # The target index is formed first: the residual between the target
        # sample and the linear prediction is mu-law companded, rounded, and
        # clamped into the 0-255 domain. Its likelihood is then read off the
        # binary tree in eight passes. At each level the bits already decided
        # form the prefix that selects the active node, that node's output is
        # read as the probability of taking the one-branch, and the branch
        # the target actually takes contributes its log probability. Summing
        # the eight per-level terms yields the log likelihood of the target
        # index, and the returned loss is its negated mean.
        #
        # Args:
        #     target_samples: Teacher-forced target samples on the int16
        #         linear scale.
        #     prediction: The linear predictor's estimate of those samples.
        #         Only the residual is supervised, so the network is trained
        #         on what the linear predictor could not explain.
        #     node_outputs: Per-node branch probabilities with the tree's
        #         nodes on the last axis. Gathered indices run up to 255, so
        #         that axis must span at least 256 entries; entry zero is
        #         never addressed because the root is node one.
        #
        # Returns:
        #     A scalar tensor holding the mean negative log likelihood over
        #     every supervised position, carrying the graph of node_outputs.
        #     The value is bounded above by eight times
        #     ``-log(probability_epsilon)``, because each of the eight
        #     branch probabilities is floored before its logarithm.
        excitation: torch.Tensor = self._mulaw.linear_to_mulaw(target_samples - prediction)
        excitation_index: torch.Tensor = torch.round(excitation).clamp(0.0, 255.0).to(torch.long)
        log_probability: torch.Tensor = torch.zeros_like(excitation)
        for level in range(self._level_count):
            prefix: torch.Tensor = excitation_index >> (self._level_count - level)
            node_index: torch.Tensor = (1 << level) + prefix
            node_probability: torch.Tensor = torch.gather(node_outputs, dim=-1, index=node_index)
            branch_bit: torch.Tensor = (excitation_index >> (self._level_count - 1 - level)) & 1
            branch_probability: torch.Tensor = torch.where(
                branch_bit == 1,
                node_probability,
                1.0 - node_probability
            )
            log_probability: torch.Tensor = log_probability + torch.log(
                branch_probability.clamp_min(self._configuration.probability_epsilon)
            )
        return -log_probability.mean()

    @property
    def configuration(self) -> LpcnetLossConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration
