# This module:
# 1. Implements the torch.compile acceleration technique: compiled execution
#    of the trained generator with unchanged weights, in the default and
#    reduce-overhead compiler modes
#
# Harness contract (syntheticmind):
# - The technique maps a harness Module onto the same Module with its
#   network (or the network's backbone submodule) replaced by the compiled
#   callable, so the harness prediction and test loops run unchanged
#
# Design decisions:
# - Architectures exposing a backbone submodule compile only that scope,
#   because their surrounding synthesis code contains data-dependent control
#   flow that graph capture would either break or silently fall out of
# - The compiled scope is recorded into the optimization recipe so every
#   capsule states which object was actually compiled
# - Weights are untouched by construction; the variant's quality rows must
#   match the baseline within seed spread, and compilation warm-up is
#   absorbed by the prediction-stage warm-up iterations
#
# Author: Rahul Sawhney

from typing import ClassVar, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from syntheticmind.core.module import Module

from vocode.optimization.registry import OptimizationTechnique, OptimizationVariantName

__all__: list[str] = ["TorchCompileAcceleration", "TorchCompileAccelerationConfig"]


class TorchCompileAccelerationConfig(BaseModel):
    # Frozen compiler settings: mode, backend, and whole-graph capture.
    #
    # Fields:
    #     mode: Compiler optimization mode. The default mode is the first
    #         registered configuration; ``"reduce-overhead"`` is the second,
    #         trading extra memory for lower per-call dispatch cost.
    #         Default: ``"default"``.
    #     backend: Compiler backend, pinned to the single registered choice
    #         so the two compilation variants differ by mode alone.
    #         Default: ``"inductor"``.
    #     fullgraph: Whether graph capture must cover the traced scope
    #         without any break. It stays disabled because the generators
    #         contain data-dependent control flow that would make whole-graph
    #         capture fail rather than fall back. Default: ``False``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    backend: Literal["inductor"] = "inductor"
    fullgraph: bool = False


class TorchCompileAcceleration(OptimizationTechnique):
    # Compiled execution of the trained generator through torch.compile,
    # scoped to the backbone where one exists and to the whole network
    # otherwise.
    #
    # Mechanism: compilation replaces an object with a wrapper that traces the
    # forward pass on first call and executes generated kernels thereafter, so
    # every parameter keeps its value and the intervention is purely one of
    # execution. That is what makes the quality rows a control: any deviation
    # from the baseline beyond seed spread would indicate a capture defect
    # rather than a trade-off. Because tracing and code generation happen on the
    # first call, the cost lands in the prediction stage's warm-up iterations
    # and outside the timed window.
    def __init__(self, configuration: TorchCompileAccelerationConfig | None = None) -> None:
        # Binds the compiler settings (defaulting to the default-mode
        # inductor configuration) and clears the compiled-scope record.
        self._configuration: TorchCompileAccelerationConfig = (
            configuration if configuration is not None else TorchCompileAccelerationConfig()
        )
        self._compiled_scope: str | None = None

    @property
    def name(self) -> OptimizationVariantName:
        # Returns the canonical variant name this technique produces; the default mode
        # keeps the historical variant name and the reduce-overhead mode names the
        # second registered compiler configuration.
        if self._configuration.mode == "default":
            return "torch_compile"
        return "torch_compile_overhead"

    def apply(self, module: Module) -> Module:
        # Compiles the appropriate scope in place: the backbone submodule
        # when the network exposes one, the whole network otherwise, and
        # records which scope was taken for the recipe. Narrowing to the
        # backbone keeps graph capture away from the surrounding synthesis
        # code, whose data-dependent control flow would either break capture or
        # silently drop back to eager execution and quietly weaken the claim.
        #
        # Args:
        #     module: The harness Module carrying the restored baseline
        #         weights; the compiled callable is installed in place of the
        #         chosen scope.
        #
        # Raises:
        #     RuntimeError: If this interpreter's torch refuses compilation
        #         altogether, in which case the module is left untouched and
        #         the compiled scope stays unrecorded rather than the run
        #         proceeding eagerly under a compiled variant name.
        #
        # Returns:
        #     The same module, now executing the compiled scope.
        if hasattr(module.network, "backbone"):
            compiled_backbone: nn.Module = cast(
                nn.Module,
                torch.compile(
                    getattr(module.network, "backbone"),
                    mode=self._configuration.mode,
                    backend=self._configuration.backend,
                    fullgraph=self._configuration.fullgraph
                )
            )
            setattr(module.network, "backbone", compiled_backbone)
            self._compiled_scope: str | None = "backbone"
            return module
        compiled_network: nn.Module = cast(
            nn.Module,
            torch.compile(
                module.network,
                mode=self._configuration.mode,
                backend=self._configuration.backend,
                fullgraph=self._configuration.fullgraph
            )
        )
        module.network = compiled_network
        self._compiled_scope: str | None = "network"
        return module

    def configuration_dump(self) -> dict[str, object]:
        # Returns the exact transformation configuration for the optimization recipe.
        return {
            "technique": "torch_compile",
            "compiled_scope": self._compiled_scope,
            **self._configuration.model_dump(mode="json")
        }

    @property
    def configuration(self) -> TorchCompileAccelerationConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration
