# This module:
# 1. Verifies the in-memory storage definition: the summed element bytes of
#    every registered parameter and buffer, reported in megabytes
# 2. Verifies the serialized storage definition: the exact byte length of the
#    persisted state dictionary, reported in megabytes
# 3. Verifies the contrast between the two definitions, which is what makes
#    reporting both meaningful rather than redundant
#
# Design decisions:
# - The in-memory expectations are exact, because element count times element
#   size divided by one mebibyte is deterministic pure arithmetic
# - The serialized expectations compare against an independently produced
#   in-memory serialization of the same state dictionary, which pins the unit
#   conversion and the choice of serialized object without pinning the pickle
#   payload length of a particular torch release
# - Non-persistent buffers are the sharpest available contrast between the two
#   definitions: they occupy live memory but never reach the shipped artifact
#
# Author: Rahul Sawhney

import io
import unittest

import torch
from torch import nn

from vocode.metrics.size import ModelSize


class PersistentBufferNetwork(nn.Module):
    # Network whose buffer is persistent, so it appears in both the live object
    # and the serialized state dictionary.
    def __init__(self, buffer_length: int) -> None:
        # Registers one weighted layer beside a buffer that ships with the state.
        super().__init__()
        self.projection: nn.Linear = nn.Linear(4, 3)
        self.register_buffer("running_scale", torch.zeros(buffer_length), persistent=True)


class ScratchBufferNetwork(nn.Module):
    # Network whose buffer is non-persistent, so it occupies live memory but is
    # excluded from the serialized state dictionary.
    def __init__(self, buffer_length: int) -> None:
        # Registers the same geometry with a buffer the state dictionary omits.
        super().__init__()
        self.projection: nn.Linear = nn.Linear(4, 3)
        self.register_buffer("running_scale", torch.zeros(buffer_length), persistent=False)


class InMemoryModelSizeTest(unittest.TestCase):
    # Verifies the in-memory definition over networks whose byte totals are
    # known exactly from their declared geometry and dtype.
    def setUp(self) -> None:
        # Prepares the metric and the mebibyte divisor the expectations use.
        self._metric: ModelSize = ModelSize()
        self._bytes_per_megabyte: float = 1024.0 * 1024.0

    def test_single_layer_reports_its_parameter_bytes_in_megabytes(self) -> None:
        # Fifteen float32 parameters occupy sixty bytes.
        network: nn.Linear = nn.Linear(4, 3)
        self.assertEqual(self._metric(network), (4 * 3 + 3) * 4 / self._bytes_per_megabyte)

    def test_nested_submodules_are_traversed(self) -> None:
        # The measurement reaches every parameter in the module tree.
        network: nn.Sequential = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        expected_elements: int = (4 * 3 + 3) + (3 * 2 + 2)
        self.assertEqual(self._metric(network), expected_elements * 4 / self._bytes_per_megabyte)

    def test_registered_buffers_are_included(self) -> None:
        # Buffers occupy live memory and belong to the in-memory definition.
        network: PersistentBufferNetwork = PersistentBufferNetwork(64)
        expected_elements: int = (4 * 3 + 3) + 64
        self.assertEqual(self._metric(network), expected_elements * 4 / self._bytes_per_megabyte)

    def test_half_precision_parameters_halve_the_reported_size(self) -> None:
        # The measurement reads element size rather than assuming float32.
        network: nn.Linear = nn.Linear(4, 3).half()
        self.assertEqual(self._metric(network), (4 * 3 + 3) * 2 / self._bytes_per_megabyte)

    def test_parameter_free_network_reports_zero(self) -> None:
        # A network holding neither parameters nor buffers occupies nothing.
        network: nn.ReLU = nn.ReLU()
        self.assertEqual(self._metric(network), 0.0)

    def test_measurement_is_repeatable(self) -> None:
        # The measurement is a pure read and must not drift between calls.
        network: nn.Linear = nn.Linear(4, 3)
        self.assertEqual(self._metric(network), self._metric(network))


class SerializedModelSizeTest(unittest.TestCase):
    # Verifies the serialized definition against an independently produced
    # in-memory serialization of the same state dictionary.
    def setUp(self) -> None:
        # Prepares the metric and the mebibyte divisor the expectations use.
        self._metric: ModelSize = ModelSize()
        self._bytes_per_megabyte: float = 1024.0 * 1024.0

    def test_serialized_size_matches_the_persisted_state_dictionary_bytes(self) -> None:
        # The reported value is the state dictionary's exact byte length in megabytes.
        network: nn.Linear = nn.Linear(4, 3)
        reference_buffer: io.BytesIO = io.BytesIO()
        torch.save(network.state_dict(), reference_buffer)
        expected_megabytes: float = (
            reference_buffer.getbuffer().nbytes / self._bytes_per_megabyte
        )
        self.assertEqual(self._metric.serialized_megabytes(network), expected_megabytes)

    def test_serialized_size_is_positive_for_a_parameter_free_network(self) -> None:
        # An empty state dictionary still carries a serialization header.
        network: nn.ReLU = nn.ReLU()
        self.assertGreater(self._metric.serialized_megabytes(network), 0.0)

    def test_serialized_size_grows_with_the_weight_count(self) -> None:
        # A wider network must serialize to strictly more bytes.
        small_network: nn.Linear = nn.Linear(4, 3)
        large_network: nn.Linear = nn.Linear(64, 32)
        self.assertGreater(
            self._metric.serialized_megabytes(large_network),
            self._metric.serialized_megabytes(small_network)
        )

    def test_serialized_measurement_is_repeatable(self) -> None:
        # Repeated serialization of unchanged weights must agree exactly.
        network: nn.Linear = nn.Linear(4, 3)
        self.assertEqual(
            self._metric.serialized_megabytes(network),
            self._metric.serialized_megabytes(network)
        )

    def test_serialization_leaves_the_network_untouched(self) -> None:
        # Measuring a network must not perturb the weights it measures.
        network: nn.Linear = nn.Linear(4, 3)
        weight_before: torch.Tensor = network.weight.detach().clone()
        self._metric.serialized_megabytes(network)
        self.assertTrue(torch.equal(network.weight.detach(), weight_before))


class ModelSizeDefinitionContrastTest(unittest.TestCase):
    # Verifies that the two definitions answer different questions, using
    # non-persistent buffers, which live in memory but never ship.
    def setUp(self) -> None:
        # Prepares the metric shared by both definitions.
        self._metric: ModelSize = ModelSize()

    def test_in_memory_definition_includes_non_persistent_buffers(self) -> None:
        # A scratch buffer occupies live memory exactly like a persistent one.
        scratch_network: ScratchBufferNetwork = ScratchBufferNetwork(64)
        persistent_network: PersistentBufferNetwork = PersistentBufferNetwork(64)
        self.assertEqual(self._metric(scratch_network), self._metric(persistent_network))

    def test_serialized_definition_excludes_non_persistent_buffers(self) -> None:
        # A scratch buffer never reaches the state dictionary that ships.
        scratch_network: ScratchBufferNetwork = ScratchBufferNetwork(64)
        persistent_network: PersistentBufferNetwork = PersistentBufferNetwork(64)
        self.assertLess(
            self._metric.serialized_megabytes(scratch_network),
            self._metric.serialized_megabytes(persistent_network)
        )

    def test_both_definitions_report_megabytes_of_the_same_network(self) -> None:
        # A network large enough to dominate serialization overhead agrees to a megabyte scale.
        network: nn.Linear = nn.Linear(512, 512)
        self.assertAlmostEqual(
            self._metric.serialized_megabytes(network),
            self._metric(network),
            places=2
        )


if __name__ == "__main__":
    unittest.main()
