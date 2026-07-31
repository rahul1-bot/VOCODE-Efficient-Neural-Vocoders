# This module:
# 1. Verifies the frozen deployment settings: defaults, positive-integer
#    domains, immutability, and strict typing
# 2. Verifies the static-INT8 switch decision that names the two deployment
#    variants
# 3. Verifies the binding contract: apply refuses to run before the run
#    configuration is bound, because export paths and calibration data have
#    no destination without it
# 4. Verifies the recipe dump recorded before any export, and the session
#    adapter against the ONNX Runtime backend actually present in this
#    environment
#
# Design decisions:
# - apply is never invoked with a bound configuration: the deployment chain
#   captures calibration mels by running real prediction steps over the
#   LJSpeech training partition, which is a dataset dependency this suite
#   refuses; only the pre-apply refusal path is exercised
# - onnxruntime is not installed in this environment, so the session adapter
#   assertion is the surfaced dependency failure rather than a skip; the
#   roundtrip branch stays in place for environments that do carry the
#   backend
# - The run configuration handed to bind is the real validated record built
#   inside a temporary artifact root, so the binding contract is exercised
#   against the object the command-line surface produces
#
# Author: Rahul Sawhney

import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import override

import torch
from pydantic import ValidationError
from torch import nn

from syntheticmind.core.module import Module
from syntheticmind.utilities.exceptions import MisconfigurationError

from vocode.configs.layout import ExperimentArtifactLayout
from vocode.configs.run import ExperimentConfiguration
from vocode.data.ljspeech_datamodule import LJSpeechDataConfig
from vocode.metrics.rtf import RealTimeFactorConfig
from vocode.optimization.deployment import OnnxNetworkModule, OnnxRuntimeDeployment, OnnxRuntimeDeploymentConfig
from vocode.optimization.export import OnnxExporter


class TinySpectralGenerator(nn.Module):
    # Minimal mel-to-waveform generator producing an exportable convolutional graph.
    def __init__(self) -> None:
        # Builds the single convolution that makes an exportable graph.
        super().__init__()
        self.head: nn.Conv1d = nn.Conv1d(4, 1, 3, padding=1)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Maps the conditioning mel onto a single waveform channel.
        return self.head(mel)


class TinyHarnessModule(Module):
    # Minimal harness Module exposing the network attribute the technique replaces.
    def __init__(self, network: nn.Module) -> None:
        # Binds the synthesis network the deployment technique replaces.
        super().__init__()
        self.network: nn.Module = network

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Delegates synthesis to the bound network.
        return self.network(mel)


class MelBatchBuilder:
    # Builds the minimal conditioning mel batch the deployment lane executes on.
    def build(self) -> torch.Tensor:
        # Seeds construction and returns one deterministic conditioning batch.
        torch.manual_seed(0)
        return torch.randn(1, 4, 8)


class DeploymentConfigurationBuilder:
    # Builds a validated optimized-variant run configuration inside a temporary root.
    # The record is the real validated one rather than a stub, so the binding
    # contract is exercised against exactly the object the command-line surface
    # produces; every path it names is anchored in the temporary root, so no case
    # can reach the corpus, the author weights, or the real artifact tree.
    def __init__(self, root: Path) -> None:
        # Binds the temporary root every built configuration is anchored in.
        #
        # Args:
        #     root: The per-case temporary directory all built paths descend
        #         from.
        self._root: Path = root

    def build(self, variant_name: str) -> ExperimentConfiguration:
        # Builds an optimized-variant run description for the requested variant.
        # The architecture is one of the four carrying a registered export lane,
        # so the configuration is a coherent deployment cell rather than one the
        # registry would refuse.
        #
        # Args:
        #     variant_name: The deployment variant the run declares.
        #
        # Returns:
        #     A validated run description on the processor lane whose paths
        #     all lie inside the temporary root.
        layout: ExperimentArtifactLayout = ExperimentArtifactLayout(
            artifact_root=self._root / "artifacts",
            evidence_category="project_optimized_variants",
            architecture_name="hifigan_v1",
            hardware_name="cpu",
            precision_name="fp32",
            seed=0,
            run_id="run-0001",
            variant_name=variant_name
        )
        return ExperimentConfiguration(
            experiment_name="deployment-unit",
            run_id="run-0001",
            hypothesis="Exported runtime execution preserves the measured gain.",
            interpretation_notes="Unit-suite configuration; no deployment chain is executed.",
            evidence_category="project_optimized_variants",
            stage="test",
            dataset_split_name="test",
            architecture_name="hifigan_v1",
            seed=0,
            artifact_layout=layout,
            published_weights_root=self._root / "published_weights",
            project_checkpoint_path=self._root / "baseline.ckpt",
            optimization_variant_name=variant_name,
            optimization_hypothesis_id="H-OPT-2",
            code_commit_hash="0123456789abcdef",
            data_configuration=LJSpeechDataConfig(dataset_root=self._root / "corpus"),
            real_time_factor_configuration=RealTimeFactorConfig(),
            accelerator="cpu"
        )


class RuntimeDependencyProbe:
    # Reports whether an optional runtime backend is importable in this environment.
    #
    # Integration: the probe lets a case assert one contract under either
    # environment instead of skipping. Where the runtime backend is absent, the
    # assertion is that constructing a session surfaces the missing dependency;
    # where it is present, the assertion is the session adapter's own behaviour.
    def is_installed(self, module_name: str) -> bool:
        # Reports whether the named backend resolves, without importing it.
        #
        # Args:
        #     module_name: Top-level module name of the optional backend.
        #
        # Returns:
        #     Whether an import of that name would resolve in this
        #     environment.
        return importlib.util.find_spec(module_name) is not None


class OnnxRuntimeDeploymentConfigurationTest(unittest.TestCase):
    # Verifies the frozen deployment settings record and its validated domains.
    def setUp(self) -> None:
        # Binds the default settings record the default and mutation cases read.
        self._configuration: OnnxRuntimeDeploymentConfig = OnnxRuntimeDeploymentConfig()

    def test_default_configuration_is_the_single_precision_deployment_lane(self) -> None:
        # The default lane exports without calibration under the pinned opset.
        self.assertFalse(self._configuration.static_int8)
        self.assertEqual(self._configuration.opset_version, 17)
        self.assertEqual(self._configuration.calibration_utterance_count, 48)
        self.assertEqual(self._configuration.intra_op_threads, 8)

    def test_non_positive_settings_are_refused(self) -> None:
        # Opset, calibration budget, and thread count must all be positive.
        with self.assertRaises(ValidationError):
            OnnxRuntimeDeploymentConfig(opset_version=0)
        with self.assertRaises(ValidationError):
            OnnxRuntimeDeploymentConfig(calibration_utterance_count=0)
        with self.assertRaises(ValidationError):
            OnnxRuntimeDeploymentConfig(intra_op_threads=0)

    def test_configuration_is_strictly_typed(self) -> None:
        # A string standing in for the quantization switch is refused.
        with self.assertRaises(ValidationError):
            OnnxRuntimeDeploymentConfig(static_int8="true")

    def test_configuration_rejects_mutation_and_unknown_fields(self) -> None:
        # The deployment settings are frozen and closed to extra keys.
        with self.assertRaises(ValidationError):
            self._configuration.static_int8 = True
        with self.assertRaises(ValidationError):
            OnnxRuntimeDeploymentConfig(unknown_field=1)


class OnnxRuntimeDeploymentNamingTest(unittest.TestCase):
    # Verifies the quantization-switch decision naming the two deployment variants.
    def setUp(self) -> None:
        # Binds a technique built without settings, which is the fp32 lane.
        self._technique: OnnxRuntimeDeployment = OnnxRuntimeDeployment()

    def test_default_lane_names_the_single_precision_variant(self) -> None:
        # Without calibration the technique produces the fp32 deployment variant.
        self.assertEqual(self._technique.name, "onnx_fp32")

    def test_calibrated_lane_names_the_static_integer_variant(self) -> None:
        # The static INT8 switch produces the calibrated deployment variant.
        technique: OnnxRuntimeDeployment = OnnxRuntimeDeployment(
            OnnxRuntimeDeploymentConfig(static_int8=True)
        )
        self.assertEqual(technique.name, "onnx_int8_static")

    def test_omitted_configuration_binds_the_default_record(self) -> None:
        # Constructing without settings binds the fp32 deployment lane.
        self.assertEqual(self._technique.configuration, OnnxRuntimeDeploymentConfig())

    def test_supplied_configuration_is_bound_unchanged(self) -> None:
        # The supplied settings record is the one the technique reports.
        configuration: OnnxRuntimeDeploymentConfig = OnnxRuntimeDeploymentConfig(
            static_int8=True,
            intra_op_threads=2
        )
        self.assertIs(OnnxRuntimeDeployment(configuration).configuration, configuration)


class OnnxRuntimeDeploymentBindingTest(unittest.TestCase):
    # Verifies that the deployment chain refuses to run before its run configuration is bound.
    # The refusal path is the only part of apply this suite exercises, because
    # the bound path captures calibration mels by running real prediction steps
    # over the corpus, which is a dataset dependency a unit suite must not carry.
    def setUp(self) -> None:
        # Seeds construction and opens the temporary root, the configuration builder,
        # an unbound technique, and the module its refusal is asserted against.
        torch.manual_seed(0)
        self._temporary_directory: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._builder: DeploymentConfigurationBuilder = DeploymentConfigurationBuilder(self._root)
        self._technique: OnnxRuntimeDeployment = OnnxRuntimeDeployment()
        self._module: TinyHarnessModule = TinyHarnessModule(TinySpectralGenerator())

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_apply_before_binding_is_refused(self) -> None:
        # Export paths, calibration data, and artifacts must land inside a run capsule.
        with self.assertRaisesRegex(MisconfigurationError, "requires bind"):
            self._technique.apply(self._module)

    def test_refused_apply_leaves_the_network_in_place(self) -> None:
        # The refusal happens before any transformation of the module.
        original_network: nn.Module = self._module.network
        with self.assertRaises(MisconfigurationError):
            self._technique.apply(self._module)
        self.assertIs(self._module.network, original_network)
        self.assertFalse(hasattr(self._module, "deployable_artifact_bytes"))

    def test_binding_a_run_configuration_leaves_the_recipe_unchanged(self) -> None:
        # Binding states where artifacts will land; it records no measurement.
        configuration: ExperimentConfiguration = self._builder.build("onnx_fp32")
        dump_before_binding: dict[str, object] = self._technique.configuration_dump()
        self._technique.bind(configuration)
        self.assertEqual(self._technique.configuration_dump(), dump_before_binding)


class OnnxRuntimeDeploymentRecipeTest(unittest.TestCase):
    # Verifies the recipe dump recorded before any export has happened.
    def setUp(self) -> None:
        # Binds an unbound default technique whose dump is read before any export.
        self._technique: OnnxRuntimeDeployment = OnnxRuntimeDeployment()

    def test_dump_before_apply_records_no_artifact_and_no_session(self) -> None:
        # Nothing is claimed about a deployment that has not been executed.
        self.assertEqual(
            self._technique.configuration_dump(),
            {
                "technique": "onnx_runtime_deployment",
                "static_int8": False,
                "opset_version": 17,
                "calibration_mel_count": 0,
                "intra_op_threads": 8,
                "resolved_providers": [],
                "session_build_seconds": None,
                "fp32_artifact": None,
                "deployed_artifact": None
            }
        )

    def test_dump_carries_the_bound_deployment_settings(self) -> None:
        # The recipe states the exact settings the capsule executed under.
        technique: OnnxRuntimeDeployment = OnnxRuntimeDeployment(
            OnnxRuntimeDeploymentConfig(static_int8=True, opset_version=16, intra_op_threads=2)
        )
        dump: dict[str, object] = technique.configuration_dump()
        self.assertTrue(dump["static_int8"])
        self.assertEqual(dump["opset_version"], 16)
        self.assertEqual(dump["intra_op_threads"], 2)


class OnnxSessionAdapterTest(unittest.TestCase):
    # Verifies the session adapter standing in the network position of the module.
    def setUp(self) -> None:
        # Exports one fp32 artifact into a temporary root for the adapter to load.
        torch.manual_seed(0)
        self._temporary_directory: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._probe: RuntimeDependencyProbe = RuntimeDependencyProbe()
        self._mel: torch.Tensor = MelBatchBuilder().build()
        self._artifact_path: Path = self._root / "tiny_fp32.onnx"
        OnnxExporter().export(TinySpectralGenerator(), self._mel, self._artifact_path)

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_session_adapter_executes_the_exported_artifact(self) -> None:
        # The adapter runs the deployed graph behind the module's tensor interface;
        # without the runtime backend the missing dependency must surface instead.
        if not self._probe.is_installed("onnxruntime"):
            with self.assertRaises(ModuleNotFoundError):
                OnnxNetworkModule(self._artifact_path, 2)
            return
        session_module: OnnxNetworkModule = OnnxNetworkModule(self._artifact_path, 2)
        output: torch.Tensor = session_module(self._mel)
        self.assertEqual(tuple(output.shape), (1, 1, 8))
        self.assertTrue(torch.isfinite(output).all())
        self.assertIn("CPUExecutionProvider", session_module.resolved_providers)
        self.assertGreaterEqual(session_module.session_build_seconds, 0.0)
        self.assertIsNotNone(session_module.first_run_seconds)

    def test_session_adapter_stands_in_the_network_position(self) -> None:
        # The adapter is a torch module, so the harness loops drive it unchanged;
        # without the runtime backend the missing dependency must surface instead.
        if not self._probe.is_installed("onnxruntime"):
            with self.assertRaises(ModuleNotFoundError):
                OnnxNetworkModule(self._artifact_path, 2)
            return
        module: TinyHarnessModule = TinyHarnessModule(TinySpectralGenerator())
        session_module: OnnxNetworkModule = OnnxNetworkModule(self._artifact_path, 2)
        module.network: nn.Module = session_module
        output: torch.Tensor = module(self._mel)
        self.assertIs(module.network, session_module)
        self.assertEqual(tuple(output.shape), (1, 1, 8))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
