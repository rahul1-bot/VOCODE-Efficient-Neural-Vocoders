# This module:
# 1. Verifies the Vocos generator network configuration record: the published
#    topology defaults, the closed padding vocabulary, and the frozen
#    extra-forbidding validation contract
# 2. Verifies the network's synthesis behavior: the inverse-STFT waveform
#    length arithmetic under both padding grids, dtype and finiteness of the
#    synthesis, batch independence, and seeded construction reproducibility
# 3. Verifies that the recipe network carries the recorded parameter budget
#    of the reproduced Vocos baseline
#
# Design decisions:
# - Forwards use one- or two-item batches of at most sixteen mel frames, so
#   the full recipe network constructs and synthesizes in well under a second
#   on CPU and no reduced surrogate topology is needed
# - Assertions bound the length arithmetic, dtype, and finiteness rather than
#   pinning sample values, because randomly initialized weights carry no
#   meaningful amplitudes
# - The parameter budget is asserted exactly, because it is the provenance
#   anchor the parameter-matched VocosFormer row is compared against
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import PositiveInt, ValidationError

from vocode.models.vocos.network import VocosNetwork, VocosNetworkConfig


class ConditioningMelBuilder:
    # Produces deterministic conditioning mels on the grid the Vocos network consumes.
    def __init__(self, band_count: int, frame_count: int) -> None:
        # Binds the mel grid this builder emits.
        self._band_count: int = band_count
        self._frame_count: int = frame_count

    def build(self, seed: int, batch_size: int) -> torch.Tensor:
        # Seeds the generator and returns one batched conditioning mel.
        torch.manual_seed(seed)
        return torch.randn(batch_size, self._band_count, self._frame_count)

    @property
    def frame_count(self) -> int:
        # Returns the frame count of every mel this builder emits.
        return self._frame_count


class VocosNetworkConfigurationTest(unittest.TestCase):
    # Verifies that the network configuration pins the published topology and rejects invalid records.
    def setUp(self) -> None:
        # Builds the unmodified topology record under test.
        self._configuration: VocosNetworkConfig = VocosNetworkConfig()

    def test_default_topology_matches_the_published_recipe(self) -> None:
        # The unmodified record is the charactr 24 kHz ConvNeXt and ISTFT grid.
        self.assertEqual(self._configuration.input_channels, 100)
        self.assertEqual(self._configuration.hidden_dimension, 512)
        self.assertEqual(self._configuration.intermediate_dimension, 1536)
        self.assertEqual(self._configuration.layer_count, 8)
        self.assertEqual(self._configuration.n_fft, 1024)
        self.assertEqual(self._configuration.hop_length, 256)
        self.assertEqual(self._configuration.padding, "center")
        self.assertIsNone(self._configuration.layer_scale_initial_value)

    def test_configuration_record_is_frozen(self) -> None:
        # Topology fields cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.hop_length: PositiveInt = 512

    def test_unknown_field_is_rejected(self) -> None:
        # Extra fields are forbidden so a misspelled experiment setting fails loudly.
        with self.assertRaises(ValidationError):
            VocosNetworkConfig(hidden_dimensions=512)

    def test_non_positive_layer_count_is_rejected(self) -> None:
        # A depth of zero would silently produce an identity backbone.
        with self.assertRaises(ValidationError):
            VocosNetworkConfig(layer_count=0)

    def test_padding_outside_the_closed_vocabulary_is_rejected(self) -> None:
        # The ISTFT grid accepts only the two published padding modes.
        with self.assertRaises(ValidationError):
            VocosNetworkConfig(padding="reflect")


class VocosNetworkCenterPaddingSynthesisTest(unittest.TestCase):
    # Verifies the centered inverse-STFT synthesis path of the recipe network.
    def setUp(self) -> None:
        # Builds the recipe network on the centered ISTFT grid in evaluation mode.
        self._configuration: VocosNetworkConfig = VocosNetworkConfig()
        self._builder: ConditioningMelBuilder = ConditioningMelBuilder(band_count=100, frame_count=16)
        torch.manual_seed(20260730)
        self._network: VocosNetwork = VocosNetwork(self._configuration).eval()

    def test_waveform_length_follows_the_centered_istft_arithmetic(self) -> None:
        # Centered synthesis expands frames to (frames - 1) * hop samples.
        mel: torch.Tensor = self._builder.build(seed=11, batch_size=1)
        with torch.no_grad():
            waveform: torch.Tensor = self._network(mel)
        expected_samples: int = (self._builder.frame_count - 1) * self._configuration.hop_length
        self.assertEqual(
            tuple(waveform.shape),
            (1, expected_samples),
            msg=f"Expected one waveform of {expected_samples} samples, got {tuple(waveform.shape)}"
        )

    def test_synthesis_is_finite_float32(self) -> None:
        # The head reconstructs in float32 and must not emit non-finite samples.
        mel: torch.Tensor = self._builder.build(seed=12, batch_size=1)
        with torch.no_grad():
            waveform: torch.Tensor = self._network(mel)
        self.assertEqual(waveform.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(waveform).all()), msg="Synthesis contains non-finite samples")

    def test_batch_dimension_is_preserved(self) -> None:
        # Two conditioning mels synthesize two independent waveforms of equal length.
        mel: torch.Tensor = self._builder.build(seed=13, batch_size=2)
        with torch.no_grad():
            waveform: torch.Tensor = self._network(mel)
        expected_samples: int = (self._builder.frame_count - 1) * self._configuration.hop_length
        self.assertEqual(tuple(waveform.shape), (2, expected_samples))

    def test_configuration_property_returns_the_bound_record(self) -> None:
        # The network exposes exactly the record it was constructed from.
        self.assertIs(self._network.configuration, self._configuration)


class VocosNetworkSamePaddingSynthesisTest(unittest.TestCase):
    # Verifies the folded same-padding inverse-STFT path, whose length arithmetic differs from the centered path.
    def setUp(self) -> None:
        # Builds the recipe network on the folded same-padding grid in evaluation mode.
        self._configuration: VocosNetworkConfig = VocosNetworkConfig(padding="same")
        self._builder: ConditioningMelBuilder = ConditioningMelBuilder(band_count=100, frame_count=16)
        torch.manual_seed(20260731)
        self._network: VocosNetwork = VocosNetwork(self._configuration).eval()

    def test_waveform_length_follows_the_same_padding_arithmetic(self) -> None:
        # The folded path trims the window overhang, leaving frames * hop samples.
        mel: torch.Tensor = self._builder.build(seed=14, batch_size=1)
        with torch.no_grad():
            waveform: torch.Tensor = self._network(mel)
        expected_samples: int = self._builder.frame_count * self._configuration.hop_length
        self.assertEqual(
            tuple(waveform.shape),
            (1, expected_samples),
            msg=f"Expected one waveform of {expected_samples} samples, got {tuple(waveform.shape)}"
        )

    def test_same_padding_synthesis_is_finite(self) -> None:
        # The envelope division is clamped, so no sample may become non-finite.
        mel: torch.Tensor = self._builder.build(seed=15, batch_size=1)
        with torch.no_grad():
            waveform: torch.Tensor = self._network(mel)
        self.assertTrue(bool(torch.isfinite(waveform).all()), msg="Folded synthesis contains non-finite samples")


class VocosNetworkReproducibilityTest(unittest.TestCase):
    # Verifies that the recipe network is seed-reproducible and carries the recorded parameter budget.
    def setUp(self) -> None:
        # Prepares the recipe record and the mel builder shared by the reproducibility checks.
        self._configuration: VocosNetworkConfig = VocosNetworkConfig()
        self._builder: ConditioningMelBuilder = ConditioningMelBuilder(band_count=100, frame_count=16)

    def test_identical_seeds_produce_identical_synthesis(self) -> None:
        # Construction and synthesis under one seed are bit-identical across runs.
        torch.manual_seed(4242)
        first_network: VocosNetwork = VocosNetwork(self._configuration).eval()
        torch.manual_seed(4242)
        second_network: VocosNetwork = VocosNetwork(self._configuration).eval()
        mel: torch.Tensor = self._builder.build(seed=16, batch_size=1)
        with torch.no_grad():
            first_waveform: torch.Tensor = first_network(mel)
            second_waveform: torch.Tensor = second_network(mel)
        self.assertTrue(
            torch.equal(first_waveform, second_waveform),
            msg="Seeded construction is not reproducible across two builds"
        )

    def test_recipe_parameter_budget_matches_the_reproduced_baseline(self) -> None:
        # The Vocos baseline is the anchor the parameter-matched rows are compared against.
        torch.manual_seed(99)
        network: VocosNetwork = VocosNetwork(self._configuration)
        parameter_count: int = sum(parameter.numel() for parameter in network.parameters())
        self.assertEqual(
            parameter_count,
            13_531_650,
            msg=f"Vocos baseline budget drifted to {parameter_count} parameters"
        )


if __name__ == "__main__":
    unittest.main()
