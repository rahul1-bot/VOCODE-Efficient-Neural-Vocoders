# This module:
# 1. Verifies the ONNX export of a minimal mel-to-waveform generator into a
#    temporary directory: directory creation, the recorded content hash,
#    byte size, initializer parameter count, pinned opset, and node
#    composition
# 2. Verifies the declared dynamic batch and frame axes of the exported
#    graph, which are the deployment claim of the exported artifact
# 3. Verifies the artifact description path on an already exported file and
#    the immutability of the artifact record
# 4. Verifies the calibration contract of the static INT8 quantizer against
#    the runtime backend actually present in this environment
#
# Design decisions:
# - The exported network is a single convolution, so the whole export costs
#   milliseconds while still producing a real graph with initializers, node
#   types, and named dynamic axes to assert against
# - The recorded hash is checked against an independently computed digest of
#   the file bytes rather than against a pinned constant, so the assertion
#   survives a change of torch or opset while still proving integrity
# - The opset recorded in the artifact record is the exporter's pinned value
#   rather than a value read back from the graph; that behaviour is asserted
#   explicitly so it cannot change silently
# - Static INT8 calibration needs the ONNX Runtime quantization backend,
#   which is not installed in this environment; the dependency boundary is
#   asserted rather than skipped, and no calibration data pipeline is built
#
# Author: Rahul Sawhney

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import override

import numpy
import onnx
import torch
from pydantic import ValidationError
from torch import nn

from vocode.optimization.export import OnnxArtifactRecord, OnnxExporter, OnnxStaticQuantizer


class TinySpectralGenerator(nn.Module):
    # Minimal mel-to-waveform generator producing an exportable convolutional graph.
    # A single padded convolution is enough to produce a real graph carrying
    # initializers, a named operator type, and declared dynamic axes, while
    # keeping every export in this suite a millisecond-scale operation. Its
    # twelve weights and one bias make the expected initializer count thirteen.
    def __init__(self) -> None:
        # Builds the single convolution that makes an exportable graph.
        super().__init__()
        self.head: nn.Conv1d = nn.Conv1d(4, 1, 3, padding=1)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Maps the conditioning mel onto a single waveform channel.
        return self.head(mel)


class MelBatchBuilder:
    # Builds the minimal conditioning mel batch the export traces through.
    def build(self) -> torch.Tensor:
        # Seeds construction and returns one deterministic conditioning batch.
        # Seeding matters because the traced batch's values reach the exported
        # initializers only indirectly, but a deterministic batch keeps the
        # byte-reproducibility assertion meaningful across runs.
        #
        # Returns:
        #     One batch of four mel channels over eight frames.
        torch.manual_seed(0)
        return torch.randn(1, 4, 8)


class RuntimeDependencyProbe:
    # Reports whether an optional runtime backend is importable in this environment.
    #
    # Integration: the probe lets a case assert one contract under either
    # environment instead of skipping. Where the backend is absent, the assertion
    # is that its absence surfaces and that no artifact is written; where it is
    # present, the assertion is the calibration contract itself.
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


class OnnxExportArtifactTest(unittest.TestCase):
    # Verifies the exported artifact and the identity record describing it.
    def setUp(self) -> None:
        # Seeds construction and opens the temporary root, the network under export,
        # its tracing batch, and the exporter.
        torch.manual_seed(0)
        self._temporary_directory: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._network: TinySpectralGenerator = TinySpectralGenerator()
        self._sample_mel: torch.Tensor = MelBatchBuilder().build()
        self._exporter: OnnxExporter = OnnxExporter()

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_export_creates_the_artifact_and_its_parent_directory(self) -> None:
        # The exporter owns the creation of the deployment directory tree.
        artifact_path: Path = self._root / "deployment" / "tiny_fp32.onnx"
        self.assertFalse(
            artifact_path.parent.exists(),
            msg="The deployment directory must not exist before the export creates it."
        )
        record: OnnxArtifactRecord = self._exporter.export(self._network, self._sample_mel, artifact_path)
        self.assertTrue(artifact_path.parent.is_dir())
        self.assertTrue(artifact_path.exists())
        self.assertEqual(record.artifact_path, artifact_path)

    def test_recorded_hash_and_size_describe_the_written_bytes(self) -> None:
        # The record identifies the exact bytes on disk, not the source module.
        # The digest is recomputed independently from the file rather than
        # compared against a pinned constant, so the assertion survives a change
        # of torch or opset while still proving the record describes the bytes
        # that were written; the length check pins the digest algorithm.
        artifact_path: Path = self._root / "tiny_fp32.onnx"
        record: OnnxArtifactRecord = self._exporter.export(self._network, self._sample_mel, artifact_path)
        expected_digest: str = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        self.assertEqual(record.artifact_sha256, expected_digest)
        self.assertEqual(len(record.artifact_sha256), 64)
        self.assertEqual(record.artifact_bytes, artifact_path.stat().st_size)

    def test_initializer_parameter_count_matches_the_exported_network(self) -> None:
        # The deployment lane reports the parameter count of the graph it executes.
        artifact_path: Path = self._root / "tiny_fp32.onnx"
        record: OnnxArtifactRecord = self._exporter.export(self._network, self._sample_mel, artifact_path)
        expected_count: int = sum(parameter.numel() for parameter in self._network.parameters())
        self.assertEqual(record.initializer_parameter_count, expected_count)
        self.assertEqual(expected_count, 13)

    def test_node_composition_and_pinned_opset_are_recorded(self) -> None:
        # The record inventories the operator graph under the pinned opset.
        artifact_path: Path = self._root / "tiny_fp32.onnx"
        record: OnnxArtifactRecord = self._exporter.export(self._network, self._sample_mel, artifact_path)
        self.assertEqual(record.node_type_counts, {"Conv": 1})
        self.assertEqual(record.opset_version, 17)

    def test_exporter_honours_an_explicitly_pinned_opset(self) -> None:
        # The pinned opset is a constructor argument recorded into every artifact.
        artifact_path: Path = self._root / "tiny_opset16.onnx"
        record: OnnxArtifactRecord = OnnxExporter(16).export(
            self._network,
            self._sample_mel,
            artifact_path
        )
        self.assertEqual(record.opset_version, 16)

    def test_exported_graph_declares_the_dynamic_batch_and_frame_axes(self) -> None:
        # Dynamic batch and frame axes are the deployment contract of the artifact.
        artifact_path: Path = self._root / "tiny_fp32.onnx"
        self._exporter.export(self._network, self._sample_mel, artifact_path)
        loaded_model: onnx.ModelProto = onnx.load(str(artifact_path))
        graph_input: onnx.ValueInfoProto = loaded_model.graph.input[0]
        graph_output: onnx.ValueInfoProto = loaded_model.graph.output[0]
        self.assertEqual(graph_input.name, "mel")
        self.assertEqual(graph_output.name, "waveform")
        self.assertEqual(graph_input.type.tensor_type.shape.dim[0].dim_param, "batch")
        self.assertEqual(graph_input.type.tensor_type.shape.dim[2].dim_param, "frames")
        self.assertEqual(graph_output.type.tensor_type.shape.dim[0].dim_param, "batch")
        self.assertEqual(graph_output.type.tensor_type.shape.dim[2].dim_param, "samples")

    def test_export_of_identical_weights_is_byte_reproducible(self) -> None:
        # Artifact identity is reproducible, which is why the export path is pinned.
        first_path: Path = self._root / "first.onnx"
        second_path: Path = self._root / "second.onnx"
        first_record: OnnxArtifactRecord = self._exporter.export(
            self._network,
            self._sample_mel,
            first_path
        )
        second_record: OnnxArtifactRecord = self._exporter.export(
            self._network,
            self._sample_mel,
            second_path
        )
        self.assertEqual(first_record.artifact_sha256, second_record.artifact_sha256)
        self.assertEqual(first_record.artifact_bytes, second_record.artifact_bytes)


class OnnxArtifactDescriptionTest(unittest.TestCase):
    # Verifies description of an already exported artifact.
    def setUp(self) -> None:
        # Exports one artifact up front, so description runs against real bytes.
        torch.manual_seed(0)
        self._temporary_directory: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._exporter: OnnxExporter = OnnxExporter()
        self._artifact_path: Path = self._root / "tiny_fp32.onnx"
        self._export_record: OnnxArtifactRecord = self._exporter.export(
            TinySpectralGenerator(),
            MelBatchBuilder().build(),
            self._artifact_path
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_description_reproduces_the_export_record(self) -> None:
        # Describing an artifact yields the same identity the export recorded.
        described: OnnxArtifactRecord = self._exporter.describe(self._artifact_path)
        self.assertEqual(described.artifact_sha256, self._export_record.artifact_sha256)
        self.assertEqual(described.artifact_bytes, self._export_record.artifact_bytes)
        self.assertEqual(
            described.initializer_parameter_count,
            self._export_record.initializer_parameter_count
        )
        self.assertEqual(described.node_type_counts, self._export_record.node_type_counts)

    def test_description_reports_the_exporter_pinned_opset(self) -> None:
        # The recorded opset is the exporter's pinned value, not a value read from the graph.
        # The case describes a file exported under one opset with an exporter
        # pinned to another, which is the only construction that can tell the two
        # behaviours apart; the recorded value follows the exporter, and that is
        # asserted explicitly so it cannot change silently.
        described: OnnxArtifactRecord = OnnxExporter(13).describe(self._artifact_path)
        self.assertEqual(described.opset_version, 13)

    def test_description_of_a_missing_artifact_fails(self) -> None:
        # A missing artifact cannot be described into a record.
        with self.assertRaises(OSError):
            self._exporter.describe(self._root / "absent.onnx")


class OnnxArtifactRecordImmutabilityTest(unittest.TestCase):
    # Verifies that the artifact identity record is a frozen, closed value object.
    def setUp(self) -> None:
        # Builds one artifact record from hand-written field values.
        self._record: OnnxArtifactRecord = OnnxArtifactRecord(
            artifact_path=Path("tiny_fp32.onnx"),
            artifact_sha256="0" * 64,
            artifact_bytes=359,
            initializer_parameter_count=13,
            opset_version=17,
            node_type_counts={"Conv": 1}
        )

    def test_record_rejects_mutation(self) -> None:
        # An artifact record describes bytes already written and cannot drift.
        with self.assertRaises(ValidationError):
            self._record.artifact_bytes = 0

    def test_record_rejects_unknown_fields(self) -> None:
        # The record is closed to extra keys.
        with self.assertRaises(ValidationError):
            OnnxArtifactRecord(
                artifact_path=Path("tiny_fp32.onnx"),
                artifact_sha256="0" * 64,
                artifact_bytes=359,
                initializer_parameter_count=13,
                opset_version=17,
                node_type_counts={"Conv": 1},
                unknown_field=1
            )

    def test_record_is_strictly_typed(self) -> None:
        # A string standing in for the byte size is refused.
        with self.assertRaises(ValidationError):
            OnnxArtifactRecord(
                artifact_path=Path("tiny_fp32.onnx"),
                artifact_sha256="0" * 64,
                artifact_bytes="359",
                initializer_parameter_count=13,
                opset_version=17,
                node_type_counts={"Conv": 1}
            )


class OnnxStaticQuantizerCalibrationTest(unittest.TestCase):
    # Verifies the calibration contract of the static INT8 quantizer.
    def setUp(self) -> None:
        # Exports the fp32 artifact the quantizer calibrates from and binds the probe.
        torch.manual_seed(0)
        self._temporary_directory: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._quantizer: OnnxStaticQuantizer = OnnxStaticQuantizer()
        self._probe: RuntimeDependencyProbe = RuntimeDependencyProbe()
        self._fp32_path: Path = self._root / "tiny_fp32.onnx"
        OnnxExporter().export(TinySpectralGenerator(), MelBatchBuilder().build(), self._fp32_path)

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_empty_calibration_set_is_refused_once_the_backend_is_present(self) -> None:
        # An empty calibration set would produce meaningless quantization ranges; without
        # the quantization backend the missing dependency must surface instead.
        int8_path: Path = self._root / "tiny_int8.onnx"
        empty_calibration: list[numpy.ndarray] = []
        if not self._probe.is_installed("onnxruntime"):
            with self.assertRaises(ModuleNotFoundError):
                self._quantizer.quantize(self._fp32_path, int8_path, empty_calibration)
            self.assertFalse(int8_path.exists())
            return
        with self.assertRaisesRegex(RuntimeError, "non-empty calibration"):
            self._quantizer.quantize(self._fp32_path, int8_path, empty_calibration)
        self.assertFalse(int8_path.exists())


if __name__ == "__main__":
    unittest.main()
