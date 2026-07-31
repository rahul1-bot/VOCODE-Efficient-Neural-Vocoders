# This module:
# 1. Exports one mel-to-waveform generator to a deployable ONNX graph with
#    dynamic batch and frame axes, and describes the exported artifact
#    (hash, byte size, initializer parameter count, node composition)
# 2. Produces the calibrated static INT8 QDQ artifact from the fp32 export
#    using captured training mels as calibration data
#
# Design decisions:
# - The artifact record inventories the exported graph itself, so the
#   deployment lane reports the exact object it executes rather than the
#   PyTorch module it came from
# - Export uses the stable TorchScript-based exporter path with a pinned
#   opset, keeping artifact identity reproducible
# - Static quantization uses the QDQ format with per-channel weights over
#   the convolution and matrix-multiplication operator families, the
#   configuration ONNX Runtime executes fastest on CPU targets
# - Calibration data must be non-empty by contract; a silently empty
#   calibration set would produce an artifact with meaningless ranges
#
# Author: Rahul Sawhney

import hashlib
from collections import Counter
from pathlib import Path
from typing import ClassVar

import numpy
import torch
from pydantic import BaseModel, ConfigDict, PositiveInt
from torch import nn

__all__: list[str] = ["OnnxArtifactRecord", "OnnxExporter", "OnnxStaticQuantizer"]


class OnnxArtifactRecord(BaseModel):
    # Frozen identity and composition record of one exported ONNX artifact:
    # path, content hash, byte size, initializer parameter count, opset,
    # and per-operator node counts. Every field describes the file on disk
    # rather than the PyTorch module it came from, which is what allows the
    # deployment lane to report the exact object it executes.
    #
    # Fields:
    #     artifact_path: Location of the described file inside the run
    #         capsule's deployment directory.
    #     artifact_sha256: Content hash of the file bytes, which is the
    #         artifact's identity in the recipe.
    #     artifact_bytes: Size of the file on disk, which the metric panel
    #         prefers over a live-object size for deployed variants.
    #     initializer_parameter_count: Total elements across the graph's
    #         initializers, that is, the parameter count of the graph that
    #         actually executes.
    #     opset_version: The exporter's pinned operator-set version. It is
    #         recorded from the exporter rather than read back from the
    #         graph, so describing a file with a differently pinned exporter
    #         records that exporter's value.
    #     node_type_counts: Occurrences per operator type in the graph,
    #         which is how a quantized artifact is told from its fp32 source
    #         by composition rather than by filename.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True
    )
    artifact_path: Path
    artifact_sha256: str
    artifact_bytes: int
    initializer_parameter_count: int
    opset_version: int
    node_type_counts: dict[str, int]


class OnnxExporter:
    # Exporter producing the deployable ONNX graph for one mel-to-waveform generator.
    # The exported artifact is inspected for its initializer parameter count and node
    # composition so the deployment lane reports the exact object it executes.
    def __init__(self, opset_version: PositiveInt = 17) -> None:
        # Binds the pinned opset version recorded into every artifact.
        self._opset_version: int = int(opset_version)

    def export(
        self,
        network: nn.Module,
        sample_mel: torch.Tensor,
        artifact_path: Path
    ) -> OnnxArtifactRecord:
        # Exports the network to ONNX with dynamic batch and frame axes and records identity.
        # The exporter owns the creation of its destination directory, switches
        # the network to evaluation mode, and traces under inference mode. The
        # declared dynamic axes are the deployment claim of the artifact: batch
        # and frame count on the input, batch and sample count on the output, so
        # one exported graph serves every utterance length rather than only the
        # traced one. Graph capture takes the stable tracing path rather than the
        # newer exporter, which is what keeps artifact bytes reproducible for a
        # given set of weights.
        #
        # Args:
        #     network: The generator network to export; it is left in
        #         evaluation mode afterwards.
        #     sample_mel: One representative conditioning batch used to trace
        #         the graph, drawn from the model's own captured input
        #         distribution.
        #     artifact_path: Destination file, whose parent directory is
        #         created if absent.
        #
        # Returns:
        #     The identity and composition record of the written file.
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        network.eval()
        with torch.inference_mode():
            torch.onnx.export(
                network,
                (sample_mel,),
                str(artifact_path),
                input_names=["mel"],
                output_names=["waveform"],
                dynamic_axes={
                    "mel": {0: "batch", 2: "frames"},
                    "waveform": {0: "batch", 2: "samples"}
                },
                opset_version=self._opset_version,
                dynamo=False
            )
        return self.describe(artifact_path)

    def describe(self, artifact_path: Path) -> OnnxArtifactRecord:
        # Inspects one ONNX artifact and returns its identity and composition record.
        # This is the path by which an artifact this exporter did not write, such
        # as the quantized graph the static quantizer produces, is brought into
        # the same record form. The onnx import is function-local so that
        # importing this module never requires the library in processes that only
        # need the exporter's type surface.
        #
        # Args:
        #     artifact_path: An existing ONNX file to inspect.
        #
        # Raises:
        #     OSError: If the file does not exist or cannot be read.
        #
        # Returns:
        #     The identity and composition record of the inspected file,
        #     carrying this exporter's pinned opset.
        import onnx
        loaded_model: onnx.ModelProto = onnx.load(str(artifact_path))
        initializer_parameter_count: int = sum(
            int(numpy.prod(tensor.dims)) for tensor in loaded_model.graph.initializer
        )
        node_type_counts: Counter[str] = Counter(node.op_type for node in loaded_model.graph.node)
        return OnnxArtifactRecord(
            artifact_path=artifact_path,
            artifact_sha256=self._compute_sha256(artifact_path),
            artifact_bytes=artifact_path.stat().st_size,
            initializer_parameter_count=initializer_parameter_count,
            opset_version=self._opset_version,
            node_type_counts=dict(node_type_counts)
        )

    def _compute_sha256(self, file_path: Path) -> str:
        # Computes the integrity hash recorded beside the artifact.
        digest: hashlib._Hash = hashlib.sha256()
        with file_path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


class OnnxStaticQuantizer:
    # Calibrated static INT8 quantizer over one exported convolutional ONNX graph.
    # Calibration mels come from the training partition through the deployed graph's own
    # input distribution, and quantization uses the QDQ format with per-channel weights.
    #
    # Mechanism: static quantization differs from the dynamic kind in that
    # activation ranges are fixed into the graph ahead of time instead of being
    # derived per inference, which is what makes the deployed graph faster and
    # what makes calibration necessary. The calibrator obtains those ranges by
    # executing the fp32 graph once for each supplied calibration tensor and
    # observing the activations, so the ranges are a measured property of the
    # calibration set. Because the reader constructed for that pass is single-use,
    # calibration is a per-execution act: one call quantizes one graph from one
    # captured set, and the resulting artifact's bytes belong to that execution.
    # The quantizer holds no state of its own between calls.
    def quantize(
        self,
        fp32_artifact_path: Path,
        int8_artifact_path: Path,
        calibration_mels: list[numpy.ndarray]
    ) -> None:
        # Produces the static INT8 QDQ artifact from the fp32 export and calibration data.
        # The QDQ format inserts explicit quantize and dequantize pairs around
        # the quantized operators rather than replacing them with integer
        # operators outright, which is the representation ONNX Runtime executes
        # fastest on processor targets. Weights are quantized per output channel
        # as signed eight-bit values and activations as unsigned ones, and only
        # the convolution and matrix-multiplication operator families are
        # quantized, so the remaining operators keep single-precision execution.
        # The backend import is function-local, so the deployment module can be
        # imported without the quantization backend installed.
        #
        # Args:
        #     fp32_artifact_path: The exported single-precision graph, which
        #         is read but not modified.
        #     int8_artifact_path: Destination of the calibrated INT8 graph.
        #     calibration_mels: The captured network inputs replayed once, in
        #         order, to measure activation ranges.
        #
        # Raises:
        #     RuntimeError: If the calibration set is empty, because the
        #         resulting artifact would carry meaningless ranges. The check
        #         precedes any write, so a refused call leaves no destination
        #         file behind.
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
        if not calibration_mels:
            raise RuntimeError(
                "Static INT8 quantization requires a non-empty calibration mel set."
            )

        class MelCalibrationReader(CalibrationDataReader):
            # Sequential reader feeding captured training mels to the calibrator.
            # The reader is single-use: its position advances monotonically and
            # is never reset, so one instance serves exactly one quantization
            # execution. It is declared inside the call because the base class
            # comes from the function-local backend import.
            def __init__(self, mels: list[numpy.ndarray]) -> None:
                # Copies the supplied sequence so the caller's list cannot
                # change the calibration set mid-pass, and opens the position
                # at the first entry.
                self._mels: list[numpy.ndarray] = list(mels)
                self._position: int = 0

            def get_next(self) -> dict[str, numpy.ndarray] | None:
                # Yields the next calibration mel under the graph's input
                # name, or None when the set is exhausted. Returning None is
                # how the calibrator learns the pass is over, so the reader
                # reports exhaustion rather than raising or wrapping around.
                #
                # Returns:
                #     A single-entry feed mapping the graph's input name onto
                #     the next captured tensor, or ``None`` once every
                #     captured tensor has been served.
                if self._position >= len(self._mels):
                    return None
                current_mel: numpy.ndarray = self._mels[self._position]
                self._position: int = self._position + 1
                return {"mel": current_mel}

        quantize_static(
            model_input=str(fp32_artifact_path),
            model_output=str(int8_artifact_path),
            calibration_data_reader=MelCalibrationReader(calibration_mels),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            op_types_to_quantize=["Conv", "ConvTranspose", "MatMul", "Gemm"]
        )
