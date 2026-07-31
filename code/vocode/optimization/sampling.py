# This module:
# 1. Implements the ODE sampling-step reduction technique for the
#    iterative-flow vocoder: the trained weights are untouched and the
#    module's sampler is replaced with one integrating fewer Euler steps
#
# Harness contract (syntheticmind):
# - The technique maps a harness Module onto the same Module with only its
#   sampler collaborator exchanged, so evaluation measures the structural
#   step-count and quality trade-off on the identical checkpoint
#
# Design decisions:
# - The registered step counts (eight, four, two) halve the baseline
#   schedule progressively, giving the study a monotone intervention curve
# - The technique refuses modules without an ODE sampler because applying a
#   sampling-schedule intervention to a single-pass architecture would be
#   meaningless
# - The baseline step count is captured before replacement and recorded in
#   the recipe, so the intervention's starting point is part of the
#   evidence
#
# Author: Rahul Sawhney

from typing import ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict

from syntheticmind.core.module import Module
from syntheticmind.utilities.exceptions import MisconfigurationError

from vocode.models.rfwave.sampling import RfwaveOdeSampler
from vocode.optimization.registry import OptimizationTechnique, OptimizationVariantName

__all__: list[str] = ["OdeStepReduction", "OdeStepReductionConfig"]


class OdeStepReductionConfig(BaseModel):
    # Frozen intervention setting: the registered reduced step count.
    #
    # Fields:
    #     step_count: Number of Euler integration steps the exchanged
    #         sampler performs. The domain is closed over the three
    #         registered reduction points, which halve the schedule
    #         progressively, and the field carries no default because the
    #         reduction point is the identity of the intervention.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    step_count: Literal[8, 4, 2]


class OdeStepReduction(OptimizationTechnique):
    # Sampler exchange reducing the Euler integration schedule of the
    # iterative-flow vocoder while leaving its weights untouched.
    #
    # Mechanism: the flow architecture synthesizes by integrating an ordinary
    # differential equation, so its synthesis cost is proportional to the number
    # of integration steps and each step is one full network evaluation. Halving
    # the schedule therefore halves the work directly, without touching a single
    # parameter, which makes this the study's one structural inference-step
    # intervention rather than a numerical or a compilation one. The trade-off is
    # integration error, which the quality panel measures on the identical test
    # split.
    def __init__(self, configuration: OdeStepReductionConfig) -> None:
        # Binds the reduced step count and clears the baseline record. The
        # baseline is deliberately not assumed at construction; it is read from
        # the module the technique is applied to.
        #
        # Args:
        #     configuration: The registered reduction point this technique
        #         installs.
        self._configuration: OdeStepReductionConfig = configuration
        self._baseline_step_count: int | None = None

    @property
    def name(self) -> OptimizationVariantName:
        # Returns the canonical variant name this technique produces.
        return cast(OptimizationVariantName, f"ode_steps_{self._configuration.step_count}")

    def apply(self, module: Module) -> Module:
        # Applies the transformation to the trained module and returns it. The
        # existing sampler's step count is recorded first, so the recipe states
        # where the schedule started as well as where it landed, and a freshly
        # constructed sampler is installed rather than the existing one being
        # mutated, which leaves the original object intact for any other holder.
        #
        # Args:
        #     module: The harness Module carrying the restored baseline
        #         weights and its sampler collaborator.
        #
        # Raises:
        #     MisconfigurationError: If the module carries no sampler of the
        #         iterative-flow kind, naming the offending type; a
        #         sampling-schedule intervention has no meaning on a
        #         single-pass architecture.
        #
        # Returns:
        #     The same module, now integrating on the reduced schedule with
        #     every weight unchanged.
        current_sampler: object = getattr(module, "_sampler", None)
        if not isinstance(current_sampler, RfwaveOdeSampler):
            raise MisconfigurationError(
                f"Sampling-step reduction requires a module carrying an RfwaveOdeSampler, "
                f"got {type(current_sampler).__name__}; this technique applies only to the "
                f"iterative-flow architecture."
            )
        self._baseline_step_count: int | None = current_sampler.step_count
        reduced_sampler: RfwaveOdeSampler = RfwaveOdeSampler(
            step_count=self._configuration.step_count
        )
        setattr(module, "_sampler", reduced_sampler)
        return module

    @property
    def configuration(self) -> OdeStepReductionConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    def configuration_dump(self) -> dict[str, object]:
        # Returns the exact transformation configuration for the optimization recipe.
        return {
            "technique": "ode_step_reduction",
            "baseline_step_count": self._baseline_step_count,
            "reduced_step_count": self._configuration.step_count
        }
