# This module:
# 1. Implements the ONNX Runtime deployment technique: export of the trained
#    generator from its own captured input distribution, optional calibrated
#    static INT8 quantization, and execution of the deployed artifact
#    through an ONNX Runtime session standing in the network position
# 2. Answers the Study 2 question of which apparent gains survive execution
#    on a real backend rather than inspection of an in-memory PyTorch object
#
# Harness contract (syntheticmind):
# - The session adapter is a torch.nn.Module standing exactly where the
#   network stood, so the module's own feature extraction, prediction, and
#   test steps drive the deployed session and every existing metric,
#   timing, and logging component measures it unchanged
# - The technique additionally declares deployable_artifact_bytes and
#   deployable_parameter_count on the module, which the metric panel
#   prefers over live-object measurements
#
# Design decisions:
# - Calibration mels are captured from the training partition through a
#   forward-pre-hook on the network during real prediction steps, so the
#   exported graph and the quantization ranges see the model's true input
#   distribution rather than synthetic probes
# - The technique requires an explicit bind to the run configuration before
#   apply, because export paths, calibration data, and artifacts must land
#   inside the run capsule
# - Session construction cost and resolved execution providers are serialized.
#   The adapter measures its first forward call in memory, but the completed
#   Study 2 capsules did not serialize that value; cold-start inference is
#   therefore outside the retained deployment evidence.
#
# Author: Rahul Sawhney

import time
from pathlib import Path
from typing import ClassVar

import numpy
import torch
from loguru import logger as log
from pydantic import BaseModel, ConfigDict, PositiveInt
from torch import nn
from torch.utils.hooks import RemovableHandle

from syntheticmind.core.module import Module
from syntheticmind.utilities.exceptions import MisconfigurationError

from vocode.configs.run import ExperimentConfiguration
from vocode.data.ljspeech_datamodule import LJSpeechDataModule
from vocode.optimization.export import OnnxArtifactRecord, OnnxExporter, OnnxStaticQuantizer
from vocode.optimization.registry import OptimizationTechnique, OptimizationVariantName

__all__: list[str] = [
    "OnnxNetworkModule",
    "OnnxRuntimeDeployment",
    "OnnxRuntimeDeploymentConfig"
]


class OnnxNetworkModule(nn.Module):
    # Network adapter executing the deployable ONNX artifact through ONNX Runtime.
    # It stands in the network position of the source module, so the module's own
    # feature extraction, prediction, and test steps drive the deployed session and
    # every existing metric, timing, and logging component measures it unchanged.
    #
    # Integration: the adapter is a torch.nn.Module carrying no parameters, which
    # is what lets it be assigned into the network position without any change to
    # the surrounding module. Its forward accepts and returns torch tensors,
    # converting to and from numpy at the session boundary and restoring the
    # input's device on the way out, so the harness loops, the metric panel, and
    # the real-time-factor monitor time and measure the deployed graph exactly as
    # they would an in-memory network. Because the artifact carries its own
    # weights, live-object parameter counts no longer describe what executes;
    # the deployment technique therefore declares the artifact's size and
    # initializer count on the module for the metric panel to prefer.
    def __init__(self, session_path: Path, intra_op_threads: int) -> None:
        # Builds the CPU inference session with the configured thread count,
        # timing the construction and recording the resolved providers. Session
        # construction is where ONNX Runtime performs graph optimization and
        # provider assignment, so its cost is measured here and serialized into
        # the recipe rather than being absorbed into the first inference.
        #
        # Args:
        #     session_path: The deployable ONNX artifact this session executes,
        #         which is the fp32 export or the calibrated INT8 artifact
        #         depending on the lane.
        #     intra_op_threads: Thread count for intra-operator parallelism,
        #         pinned per run so timing is comparable across capsules.
        super().__init__()
        import onnxruntime
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = intra_op_threads
        session_build_start: float = time.perf_counter()
        self._session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            str(session_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"]
        )
        self._session_build_seconds: float = time.perf_counter() - session_build_start
        self._resolved_providers: tuple[str, ...] = tuple(self._session.get_providers())
        self._first_run_seconds: float | None = None

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Runs the deployed session on one mel batch and returns the waveform tensor.
        # The input is detached, moved to host memory, and cast to single
        # precision because the session's declared input type is fp32 on both
        # lanes; on the INT8 lane the quantization boundary lives inside the
        # graph, so the interface dtype is unchanged. The first call is timed
        # separately, since it carries the runtime's lazy per-node initialization
        # that later calls do not.
        #
        # Args:
        #     mel: The conditioning mel batch, whose device the returned
        #         waveform is restored to.
        #
        # Returns:
        #     The synthesized waveform as a torch tensor on the input's device.
        mel_array: numpy.ndarray = mel.detach().cpu().float().numpy()
        first_run_start: float = time.perf_counter()
        session_outputs: list[numpy.ndarray] = self._session.run(None, {"mel": mel_array})
        if self._first_run_seconds is None:
            self._first_run_seconds: float | None = time.perf_counter() - first_run_start
        return torch.from_numpy(session_outputs[0]).to(mel.device)

    @property
    def session_build_seconds(self) -> float:
        # Returns the measured session construction cost.
        return self._session_build_seconds

    @property
    def first_run_seconds(self) -> float | None:
        # Returns the measured first-inference cold cost once available.
        return self._first_run_seconds

    @property
    def resolved_providers(self) -> tuple[str, ...]:
        # Returns the execution providers the session actually assigned.
        return self._resolved_providers


class OnnxRuntimeDeploymentConfig(BaseModel):
    # Frozen deployment settings: the static-INT8 switch, pinned opset,
    # calibration utterance budget, and session thread count.
    #
    # Fields:
    #     static_int8: Whether the deployed artifact is the calibrated static
    #         INT8 graph rather than the fp32 export. The fp32 export is
    #         produced on both lanes, because it is the same-lane denominator
    #         the INT8 deployment claim is measured against. This switch also
    #         decides which of the two deployment variant names the technique
    #         reports. Default: ``False``.
    #     opset_version: ONNX operator-set version pinned for the export and
    #         recorded into every artifact record, so artifact identity stays
    #         reproducible across runtime upgrades. Default: ``17``.
    #     calibration_utterance_count: Budget of captured mel batches the
    #         calibration pass collects from the execution-seeded shuffled
    #         training loader before it stops. Default: ``48``.
    #     intra_op_threads: Intra-operator thread count of the inference
    #         session, pinned so timing is comparable across capsules on the
    #         processor lane. Default: ``8``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    static_int8: bool = False
    opset_version: PositiveInt = 17
    calibration_utterance_count: PositiveInt = 48
    intra_op_threads: PositiveInt = 8


class OnnxRuntimeDeployment(OptimizationTechnique):
    # Exported-runtime deployment technique for convolutional mel-to-waveform generators.
    # The trained network is exported to ONNX from its own captured input distribution,
    # optionally statically quantized with training-partition calibration mels, and then
    # replaced by the ONNX Runtime session adapter so the identical module pipeline
    # measures the deployed artifact.
    #
    # Static INT8 calibration mechanism: quantizing activations statically
    # requires knowing their numeric range in advance, and this technique derives
    # that range per execution rather than from any stored table. Every execution
    # of the deployment path captures its own calibration mel batches by
    # installing a forward-pre-hook on the network and running real prediction
    # steps over the execution-seeded shuffled training loader, so the ranges
    # describe the model's own input distribution and not a synthetic probe.
    # Those captured tensors are then replayed once, in capture order, through a
    # single-use sequential reader constructed inside the quantization call; the
    # reader advances monotonically and reports exhaustion by returning nothing,
    # so it serves exactly one quantization execution and is never rewound or
    # shared between executions. Nothing is cached between executions, which
    # means an INT8 artifact's activation ranges, and therefore its bytes and its
    # content hash, are a property of the execution that produced it. The capture
    # pass runs before export, so the exported fp32 graph and the INT8
    # calibration observe the same captured distribution, and the first captured
    # tensor additionally serves as the export's tracing input.
    #
    # Consequence for the evidence: every other registered variant repeats
    # inference from one frozen artifact across the three evaluation executions,
    # so its execution spread reflects execution variability alone. Static ONNX
    # INT8 is the single registered exception, because each execution rebuilds
    # its quantized graph on its own calibration batches. Its executions
    # therefore combine calibration-set variation with runtime variation rather
    # than repeating one fixed artifact, and its per-execution quality disperses
    # where the other variants repeat exactly.
    #
    # Integration: this technique needs run-level context no other technique
    # needs, because its artifacts and its calibration data must land inside the
    # run capsule. It therefore extends the base contract with bind, which the
    # optimized-variant evaluator calls with the run configuration before apply;
    # apply refuses to execute unbound rather than defaulting to a directory
    # outside the capsule.
    def __init__(self, configuration: OnnxRuntimeDeploymentConfig | None = None) -> None:
        # Binds the settings (defaulting to the fp32 lane) and clears the
        # bound-configuration, artifact, and session records.
        self._configuration: OnnxRuntimeDeploymentConfig = (
            configuration if configuration is not None else OnnxRuntimeDeploymentConfig()
        )
        self._experiment_configuration: ExperimentConfiguration | None = None
        self._fp32_record: OnnxArtifactRecord | None = None
        self._deployed_record: OnnxArtifactRecord | None = None
        self._session_build_seconds: float | None = None
        self._resolved_providers: tuple[str, ...] = ()
        self._calibration_mel_count: int = 0

    @property
    def name(self) -> OptimizationVariantName:
        # Returns the canonical variant name this technique produces.
        if self._configuration.static_int8:
            return "onnx_int8_static"
        return "onnx_fp32"

    def bind(self, experiment_configuration: ExperimentConfiguration) -> None:
        # Binds the run configuration so export, calibration, and artifacts reach the capsule.
        # Binding records where artifacts will land and which corpus the
        # calibration pass reads; it performs no measurement and leaves the
        # recipe dump unchanged.
        #
        # Args:
        #     experiment_configuration: The validated run description whose
        #         run directory receives the deployment artifacts and whose
        #         data configuration supplies the calibration corpus.
        self._experiment_configuration: ExperimentConfiguration | None = experiment_configuration

    def apply(self, module: Module) -> Module:
        # Executes the deployment chain: capture calibration mels, export
        # the fp32 graph, optionally produce the calibrated INT8 artifact,
        # construct the session adapter over the deployed artifact, install
        # it in the network position, and declare the deployable size and
        # parameter facts on the module. The order matters: capture precedes
        # export because the first captured tensor is the export's tracing
        # input, and the fp32 artifact is produced on both lanes because the
        # INT8 lane quantizes it and the fp32 lane is its same-lane denominator.
        #
        # Args:
        #     module: The harness Module carrying the restored baseline
        #         weights, whose network is exported and then replaced by the
        #         session adapter.
        #
        # Raises:
        #     MisconfigurationError: If bind was not called first, so export
        #         and calibration have no destination inside a run capsule.
        #     RuntimeError: If the calibration pass captured no network input.
        #
        # Returns:
        #     The same module, now synthesizing through the deployed artifact
        #     and carrying the deployable_artifact_bytes and
        #     deployable_parameter_count attributes the metric panel prefers.
        configuration: ExperimentConfiguration = self._require_bound_configuration()
        calibration_mels: list[numpy.ndarray] = self._capture_calibration_mels(module, configuration)
        deployment_directory: Path = configuration.run_directory / "deployment"
        exporter: OnnxExporter = OnnxExporter(self._configuration.opset_version)
        fp32_path: Path = deployment_directory / f"{configuration.architecture_name}_fp32.onnx"
        self._fp32_record: OnnxArtifactRecord | None = exporter.export(
            module.network,
            torch.from_numpy(calibration_mels[0]),
            fp32_path
        )
        deployed_record: OnnxArtifactRecord = self._fp32_record
        if self._configuration.static_int8:
            int8_path: Path = deployment_directory / f"{configuration.architecture_name}_int8_static.onnx"
            OnnxStaticQuantizer().quantize(fp32_path, int8_path, calibration_mels)
            deployed_record: OnnxArtifactRecord = exporter.describe(int8_path)
        self._deployed_record: OnnxArtifactRecord | None = deployed_record
        session_module: OnnxNetworkModule = OnnxNetworkModule(
            deployed_record.artifact_path,
            int(self._configuration.intra_op_threads)
        )
        self._session_build_seconds: float | None = session_module.session_build_seconds
        self._resolved_providers: tuple[str, ...] = session_module.resolved_providers
        module.network = session_module
        setattr(module, "deployable_artifact_bytes", deployed_record.artifact_bytes)
        setattr(module, "deployable_parameter_count", deployed_record.initializer_parameter_count)
        log.info(
            f"Deployed {deployed_record.artifact_path.name} "
            f"({deployed_record.artifact_bytes} bytes) on providers {self._resolved_providers}"
        )
        return module

    @property
    def configuration(self) -> OnnxRuntimeDeploymentConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    def _require_bound_configuration(self) -> ExperimentConfiguration:
        # Requires the run configuration bound through bind(); export and
        # calibration have no defined destination without it.
        if self._experiment_configuration is None:
            raise MisconfigurationError(
                "OnnxRuntimeDeployment requires bind(experiment_configuration) before apply."
            )
        return self._experiment_configuration

    def _capture_calibration_mels(
        self,
        module: Module,
        configuration: ExperimentConfiguration
    ) -> list[numpy.ndarray]:
        # Captures the network's real input mels from the training partition through a hook.
        # The hook is installed on the network rather than on the module, so what
        # is captured is exactly the tensor the exported graph will receive,
        # after the module's own feature extraction has run. Capture happens
        # inside real prediction steps under inference mode with the module in
        # evaluation mode, and the hook is removed in a finally block so a
        # failure mid-capture cannot leave it attached to the network the
        # deployment then measures.
        #
        # Args:
        #     module: The harness Module whose network is hooked and whose
        #         prediction step drives the capture.
        #     configuration: The bound run description supplying the corpus.
        #
        # Raises:
        #     RuntimeError: If the pass completed without capturing a single
        #         network input, because neither export nor calibration has a
        #         defined meaning without one.
        #
        # Returns:
        #     The captured mel batches in capture order, one per network
        #     invocation, truncated at the configured budget.
        datamodule: LJSpeechDataModule = LJSpeechDataModule(configuration.data_configuration)
        datamodule.setup("fit")
        captured_mels: list[numpy.ndarray] = []

        def capture_network_input(hooked_module: nn.Module, hook_arguments: tuple[torch.Tensor, ...]) -> None:
            # Records the first positional argument of every network call as a
            # detached host-side single-precision array, which is the form both
            # the exporter and the calibrator consume.
            del hooked_module
            captured_mels.append(hook_arguments[0].detach().cpu().float().numpy())

        hook_handle: RemovableHandle = module.network.register_forward_pre_hook(capture_network_input)
        module.eval()
        collected_utterances: int = 0
        try:
            with torch.inference_mode():
                # The budget is tested before each step and counts captured
                # network invocations rather than consumed loader batches,
                # because one prediction step may invoke the network more than
                # once.
                for batch_index, batch in enumerate(datamodule.train_dataloader()):
                    if collected_utterances >= self._configuration.calibration_utterance_count:
                        break
                    module.predict_step(batch, batch_index)
                    collected_utterances: int = len(captured_mels)
        finally:
            hook_handle.remove()
        if not captured_mels:
            raise RuntimeError(
                "Calibration capture collected no network input mels; the deployment "
                "lane cannot export or calibrate without them."
            )
        self._calibration_mel_count: int = len(captured_mels)
        log.info(f"Captured {self._calibration_mel_count} calibration mels from the training partition.")
        return captured_mels

    def configuration_dump(self) -> dict[str, object]:
        # Returns the exact transformation configuration for the optimization recipe.
        fp32_dump: dict[str, object] | None = (
            self._artifact_dump(self._fp32_record) if self._fp32_record is not None else None
        )
        deployed_dump: dict[str, object] | None = (
            self._artifact_dump(self._deployed_record) if self._deployed_record is not None else None
        )
        return {
            "technique": "onnx_runtime_deployment",
            "static_int8": self._configuration.static_int8,
            "opset_version": int(self._configuration.opset_version),
            "calibration_mel_count": self._calibration_mel_count,
            "intra_op_threads": int(self._configuration.intra_op_threads),
            "resolved_providers": list(self._resolved_providers),
            "session_build_seconds": self._session_build_seconds,
            "fp32_artifact": fp32_dump,
            "deployed_artifact": deployed_dump
        }

    def _artifact_dump(self, record: OnnxArtifactRecord) -> dict[str, object]:
        # Serializes one artifact record for the durable optimization recipe.
        return {
            "artifact_path": str(record.artifact_path),
            "artifact_sha256": record.artifact_sha256,
            "artifact_bytes": record.artifact_bytes,
            "initializer_parameter_count": record.initializer_parameter_count,
            "opset_version": record.opset_version,
            "node_type_counts": dict(record.node_type_counts)
        }
