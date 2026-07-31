# This module:
# 1. Verifies the compiler configuration record of the torch.compile
#    acceleration technique: its defaults, its closed mode and backend
#    domains, and its immutability
# 2. Verifies the variant-name decision that maps compiler mode onto the two
#    registered compilation variants
# 3. Verifies the recipe configuration dump before any compilation, and the
#    harness Module-to-Module mapping contract of apply
#
# Design decisions:
# - The compiled-scope wiring is asserted behind a runtime capability probe:
#   torch refuses torch.compile on this interpreter ("torch.compile is not
#   supported on Python 3.14+"), so on this runtime the assertion is that
#   apply surfaces the refusal and leaves the network untouched rather than
#   silently falling back to eager execution
# - No compiled forward pass is ever invoked, because graph capture and
#   inductor code generation are minute-scale operations that belong to the
#   measured study lanes, not to this suite
# - Networks under test are two-parameter stand-ins; the technique's
#   decision depends only on whether a backbone submodule is exposed
#
# Author: Rahul Sawhney

import unittest
from typing import override

import torch
from pydantic import ValidationError
from torch import nn

from syntheticmind.core.module import Module

from vocode.optimization.compilation import TorchCompileAcceleration, TorchCompileAccelerationConfig


class TinyLinearNetwork(nn.Module):
    # Minimal network without a backbone submodule, compiled as a whole.
    def __init__(self) -> None:
        # Builds the single projection that stands in for the whole compiled scope.
        super().__init__()
        self.projection: nn.Linear = nn.Linear(4, 4)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Projects the conditioning mel through the only layer of the network.
        return self.projection(mel)


class TinyBackboneNetwork(nn.Module):
    # Minimal network exposing a backbone submodule, the narrower compiled scope.
    def __init__(self) -> None:
        # Builds a backbone submodule and a head, so the narrower scope is resolvable.
        super().__init__()
        self.backbone: nn.Linear = nn.Linear(4, 4)
        self.head: nn.Linear = nn.Linear(4, 2)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Runs the backbone and then the head.
        return self.head(self.backbone(mel))


class TinyHarnessModule(Module):
    # Minimal harness Module exposing the network attribute the technique transforms.
    def __init__(self, network: nn.Module) -> None:
        # Binds the synthesis network the technique reads and replaces.
        super().__init__()
        self.network: nn.Module = network

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Delegates synthesis to the bound network.
        return self.network(mel)


class CompilationRuntimeProbe:
    # Reports whether this runtime's torch accepts a torch.compile wrapper at all.
    #
    # Integration: the probe lets each wiring case assert one contract under
    # either runtime instead of skipping. Where compilation is unavailable, the
    # assertion is that apply surfaces the refusal and leaves the network
    # untouched, which is the behaviour that stops a capsule from silently
    # measuring eager execution under a compiled variant name; where it is
    # available, the assertion is the scope wiring itself.
    def is_available(self) -> bool:
        # Wraps a throwaway module and reports whether this torch accepts the call.
        # Only the wrapper call is made and never invoked, so the probe costs no
        # graph capture. The refusal this runtime raises is a RuntimeError, so
        # that is the one exception treated as unavailability; anything else
        # propagates rather than being read as a missing feature.
        #
        # Returns:
        #     Whether this interpreter's torch accepts a compile wrapper.
        try:
            torch.compile(nn.Identity())
        except RuntimeError:
            return False
        return True


class TorchCompileConfigurationTest(unittest.TestCase):
    # Verifies the frozen compiler settings record and its closed value domains.
    def setUp(self) -> None:
        # Binds the default settings record the default and mutation cases read.
        self._configuration: TorchCompileAccelerationConfig = TorchCompileAccelerationConfig()

    def test_default_configuration_selects_default_mode_inductor_partial_graphs(self) -> None:
        # The default compiler configuration is default-mode inductor without whole-graph capture.
        self.assertEqual(self._configuration.mode, "default")
        self.assertEqual(self._configuration.backend, "inductor")
        self.assertFalse(self._configuration.fullgraph)

    def test_registered_modes_are_accepted(self) -> None:
        # The three registered compiler modes construct.
        mode: str
        for mode in ("default", "reduce-overhead", "max-autotune"):
            self.assertEqual(TorchCompileAccelerationConfig(mode=mode).mode, mode)

    def test_unregistered_mode_is_refused(self) -> None:
        # A mode outside the closed domain is a construction error.
        with self.assertRaises(ValidationError):
            TorchCompileAccelerationConfig(mode="turbo")

    def test_unregistered_backend_is_refused(self) -> None:
        # Only the inductor backend is registered for the study.
        with self.assertRaises(ValidationError):
            TorchCompileAccelerationConfig(backend="eager")

    def test_configuration_rejects_mutation_and_unknown_fields(self) -> None:
        # The compiler settings are frozen and closed to extra keys.
        with self.assertRaises(ValidationError):
            self._configuration.mode = "reduce-overhead"
        with self.assertRaises(ValidationError):
            TorchCompileAccelerationConfig(unknown_field=1)

    def test_configuration_is_strictly_typed(self) -> None:
        # A string standing in for the boolean whole-graph flag is refused.
        with self.assertRaises(ValidationError):
            TorchCompileAccelerationConfig(fullgraph="true")


class TorchCompileVariantNamingTest(unittest.TestCase):
    # Verifies the mode-to-variant-name decision of the compilation technique.
    def setUp(self) -> None:
        # Binds a technique built without settings, which is the default-mode lane.
        self._default_technique: TorchCompileAcceleration = TorchCompileAcceleration()

    def test_default_mode_keeps_the_historical_variant_name(self) -> None:
        # The default compiler configuration produces the first registered compilation variant.
        self.assertEqual(self._default_technique.name, "torch_compile")

    def test_reduce_overhead_mode_names_the_second_registered_configuration(self) -> None:
        # The reduce-overhead configuration is the second registered compilation variant.
        technique: TorchCompileAcceleration = TorchCompileAcceleration(
            TorchCompileAccelerationConfig(mode="reduce-overhead")
        )
        self.assertEqual(technique.name, "torch_compile_overhead")

    def test_every_non_default_mode_names_the_second_registered_configuration(self) -> None:
        # Only the default mode holds the historical name; other modes take the overhead name.
        technique: TorchCompileAcceleration = TorchCompileAcceleration(
            TorchCompileAccelerationConfig(mode="max-autotune")
        )
        self.assertEqual(technique.name, "torch_compile_overhead")

    def test_omitted_configuration_binds_the_default_record(self) -> None:
        # Constructing without settings binds the default-mode inductor configuration.
        self.assertEqual(self._default_technique.configuration, TorchCompileAccelerationConfig())

    def test_supplied_configuration_is_bound_unchanged(self) -> None:
        # The supplied settings record is the one the technique reports.
        configuration: TorchCompileAccelerationConfig = TorchCompileAccelerationConfig(
            mode="reduce-overhead",
            fullgraph=True
        )
        self.assertIs(TorchCompileAcceleration(configuration).configuration, configuration)


class TorchCompileConfigurationDumpTest(unittest.TestCase):
    # Verifies the recipe dump the technique records before any compilation happens.
    def setUp(self) -> None:
        # Binds a default technique whose dump is read before any compilation.
        self._technique: TorchCompileAcceleration = TorchCompileAcceleration()

    def test_dump_before_apply_records_an_unresolved_compiled_scope(self) -> None:
        # Until apply runs, no object has been compiled and the scope stays unrecorded.
        self.assertEqual(
            self._technique.configuration_dump(),
            {
                "technique": "torch_compile",
                "compiled_scope": None,
                "mode": "default",
                "backend": "inductor",
                "fullgraph": False
            }
        )

    def test_dump_carries_the_bound_compiler_settings(self) -> None:
        # The recipe states the exact compiler settings the capsule executed under.
        technique: TorchCompileAcceleration = TorchCompileAcceleration(
            TorchCompileAccelerationConfig(mode="reduce-overhead", fullgraph=True)
        )
        dump: dict[str, object] = technique.configuration_dump()
        self.assertEqual(dump["mode"], "reduce-overhead")
        self.assertEqual(dump["backend"], "inductor")
        self.assertTrue(dump["fullgraph"])
        self.assertEqual(dump["technique"], "torch_compile")


class TorchCompileScopeWiringTest(unittest.TestCase):
    # Verifies the Module-to-Module mapping contract of apply under this runtime's compiler support.
    def setUp(self) -> None:
        # Seeds construction and binds the runtime probe alongside a default technique.
        torch.manual_seed(0)
        self._probe: CompilationRuntimeProbe = CompilationRuntimeProbe()
        self._technique: TorchCompileAcceleration = TorchCompileAcceleration()

    def test_network_scope_is_taken_when_no_backbone_is_exposed(self) -> None:
        # Without a backbone the whole network is the compiled scope; an unavailable
        # compiler must surface its refusal instead of silently staying eager.
        module: TinyHarnessModule = TinyHarnessModule(TinyLinearNetwork())
        original_network: nn.Module = module.network
        if not self._probe.is_available():
            with self.assertRaises(RuntimeError):
                self._technique.apply(module)
            self.assertIs(module.network, original_network)
            self.assertIsNone(self._technique.configuration_dump()["compiled_scope"])
            return
        returned: Module = self._technique.apply(module)
        self.assertIs(returned, module)
        self.assertIsNot(returned.network, original_network)
        self.assertEqual(self._technique.configuration_dump()["compiled_scope"], "network")

    def test_backbone_scope_is_taken_when_a_backbone_is_exposed(self) -> None:
        # A backbone-bearing network compiles only that submodule; an unavailable
        # compiler must leave the backbone untouched.
        module: TinyHarnessModule = TinyHarnessModule(TinyBackboneNetwork())
        original_network: nn.Module = module.network
        original_backbone: nn.Module = getattr(module.network, "backbone")
        if not self._probe.is_available():
            with self.assertRaises(RuntimeError):
                self._technique.apply(module)
            self.assertIs(getattr(module.network, "backbone"), original_backbone)
            self.assertIsNone(self._technique.configuration_dump()["compiled_scope"])
            return
        returned: Module = self._technique.apply(module)
        self.assertIs(returned, module)
        self.assertIs(returned.network, original_network)
        self.assertIsNot(getattr(returned.network, "backbone"), original_backbone)
        self.assertEqual(self._technique.configuration_dump()["compiled_scope"], "backbone")

    def test_weights_are_never_touched_by_the_compilation_technique(self) -> None:
        # Compilation changes execution, not parameters; the quality rows must match the baseline.
        module: TinyHarnessModule = TinyHarnessModule(TinyLinearNetwork())
        original_weight: torch.Tensor = getattr(module.network, "projection").weight.detach().clone()
        if self._probe.is_available():
            self._technique.apply(module)
        else:
            with self.assertRaises(RuntimeError):
                self._technique.apply(module)
        current_weight: torch.Tensor = getattr(module.network, "projection").weight.detach()
        self.assertTrue(
            torch.equal(current_weight, original_weight),
            msg="The compilation technique must leave every parameter unchanged."
        )


if __name__ == "__main__":
    unittest.main()
