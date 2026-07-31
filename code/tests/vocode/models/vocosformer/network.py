# This module:
# 1. Verifies the VocosFormer network configuration record: the trimmed
#    ConvNeXt topology, the position-network settings, and the frozen
#    extra-forbidding validation contract
# 2. Verifies the network's synthesis behavior: the inverse-STFT waveform
#    length arithmetic inherited from the Vocos chassis, dtype and finiteness,
#    frame-count independence, and the dropout regime of the position network
# 3. Verifies the parameter-matching claim of the adaptation: the exact
#    recorded budget and its distance from the reproduced Vocos baseline
#
# Design decisions:
# - Forwards run in evaluation mode so the position network's dropout is
#   inactive and the synthesis is reproducible; one test deliberately runs in
#   training mode to confirm dropout is wired and active there
# - The parameter budget is asserted both exactly and as a relative distance
#   from the Vocos baseline, because "parameter-matched within two percent" is
#   the experimental claim that makes the attention contribution attributable
# - Length and finiteness bounds replace sample-value pinning, since randomly
#   initialized weights carry no meaningful amplitudes
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import PositiveInt, ValidationError

from vocode.models.vocos.network import VocosNetwork, VocosNetworkConfig
from vocode.models.vocosformer.network import VocosformerNetwork, VocosformerNetworkConfig


class ConditioningMelBuilder:
    # Produces deterministic conditioning mels on the grid the VocosFormer network consumes.
    def __init__(self, band_count: int) -> None:
        # Binds the mel band count this builder emits.
        self._band_count: int = band_count

    def build(self, seed: int, frame_count: int) -> torch.Tensor:
        # Seeds the generator and returns one conditioning mel of the requested length.
        torch.manual_seed(seed)
        return torch.randn(1, self._band_count, frame_count)


class VocosformerNetworkConfigurationTest(unittest.TestCase):
    # Verifies that the configuration pins the trimmed topology and the position-network settings.
    def setUp(self) -> None:
        # Builds the unmodified adaptation record under test.
        self._configuration: VocosformerNetworkConfig = VocosformerNetworkConfig()

    def test_convnext_stack_is_trimmed_against_the_vocos_baseline(self) -> None:
        # Depth and width are reduced to absorb the position network's parameters.
        baseline: VocosNetworkConfig = VocosNetworkConfig()
        self.assertEqual(self._configuration.intermediate_dimension, 1344)
        self.assertEqual(self._configuration.layer_count, 4)
        self.assertLess(self._configuration.intermediate_dimension, baseline.intermediate_dimension)
        self.assertLess(self._configuration.layer_count, baseline.layer_count)

    def test_chassis_settings_match_the_vocos_baseline(self) -> None:
        # Everything outside the position network and the trimmed stack is unchanged.
        baseline: VocosNetworkConfig = VocosNetworkConfig()
        self.assertEqual(self._configuration.input_channels, baseline.input_channels)
        self.assertEqual(self._configuration.hidden_dimension, baseline.hidden_dimension)
        self.assertEqual(self._configuration.n_fft, baseline.n_fft)
        self.assertEqual(self._configuration.hop_length, baseline.hop_length)
        self.assertEqual(self._configuration.padding, baseline.padding)

    def test_position_network_defaults_are_pinned(self) -> None:
        # The inserted network uses the reference group count and dropout rate.
        self.assertEqual(self._configuration.position_group_count, 32)
        self.assertAlmostEqual(self._configuration.position_dropout, 0.1)

    def test_configuration_record_is_frozen(self) -> None:
        # Topology fields cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.layer_count: PositiveInt = 8

    def test_unknown_field_is_rejected(self) -> None:
        # Extra fields are forbidden so a misspelled experiment setting fails loudly.
        with self.assertRaises(ValidationError):
            VocosformerNetworkConfig(position_groups=32)

    def test_padding_outside_the_closed_vocabulary_is_rejected(self) -> None:
        # The ISTFT grid accepts only the two published padding modes.
        with self.assertRaises(ValidationError):
            VocosformerNetworkConfig(padding="valid")


class VocosformerNetworkSynthesisTest(unittest.TestCase):
    # Verifies the attention-augmented network's synthesis contract on minimal seeded mels.
    def setUp(self) -> None:
        # Builds the adaptation network in evaluation mode so dropout is inactive.
        self._configuration: VocosformerNetworkConfig = VocosformerNetworkConfig()
        self._builder: ConditioningMelBuilder = ConditioningMelBuilder(band_count=100)
        torch.manual_seed(20260808)
        self._network: VocosformerNetwork = VocosformerNetwork(self._configuration).eval()

    def test_waveform_length_follows_the_centered_istft_arithmetic(self) -> None:
        # The position network preserves frame rate, so the Vocos length arithmetic is unchanged.
        mel: torch.Tensor = self._builder.build(seed=51, frame_count=16)
        with torch.no_grad():
            waveform: torch.Tensor = self._network(mel)
        expected_samples: int = 15 * self._configuration.hop_length
        self.assertEqual(
            tuple(waveform.shape),
            (1, expected_samples),
            msg=f"Expected one waveform of {expected_samples} samples, got {tuple(waveform.shape)}"
        )

    def test_frame_count_scales_the_synthesis_length(self) -> None:
        # Full self-attention over frames must remain length-agnostic.
        short_mel: torch.Tensor = self._builder.build(seed=52, frame_count=8)
        long_mel: torch.Tensor = self._builder.build(seed=53, frame_count=24)
        with torch.no_grad():
            short_waveform: torch.Tensor = self._network(short_mel)
            long_waveform: torch.Tensor = self._network(long_mel)
        self.assertEqual(tuple(short_waveform.shape), (1, 7 * self._configuration.hop_length))
        self.assertEqual(tuple(long_waveform.shape), (1, 23 * self._configuration.hop_length))

    def test_synthesis_is_finite_float32(self) -> None:
        # The head reconstructs in float32 and must not emit non-finite samples.
        mel: torch.Tensor = self._builder.build(seed=54, frame_count=16)
        with torch.no_grad():
            waveform: torch.Tensor = self._network(mel)
        self.assertEqual(waveform.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(waveform).all()), msg="Synthesis contains non-finite samples")

    def test_evaluation_mode_synthesis_is_deterministic(self) -> None:
        # With dropout inactive, repeated synthesis of one mel is bit-identical.
        mel: torch.Tensor = self._builder.build(seed=55, frame_count=16)
        with torch.no_grad():
            first_waveform: torch.Tensor = self._network(mel)
            second_waveform: torch.Tensor = self._network(mel)
        self.assertTrue(torch.equal(first_waveform, second_waveform))

    def test_training_mode_activates_position_network_dropout(self) -> None:
        # In training mode the residual blocks drop activations, so synthesis varies.
        self._network.train()
        mel: torch.Tensor = self._builder.build(seed=56, frame_count=16)
        with torch.no_grad():
            first_waveform: torch.Tensor = self._network(mel)
            second_waveform: torch.Tensor = self._network(mel)
        self.assertFalse(
            torch.equal(first_waveform, second_waveform),
            msg="Dropout in the position network is inactive during training"
        )

    def test_configuration_property_returns_the_bound_record(self) -> None:
        # The network exposes exactly the record it was constructed from.
        self.assertIs(self._network.configuration, self._configuration)


class VocosformerParameterMatchingTest(unittest.TestCase):
    # Verifies that the adaptation holds its capacity at the reproduced Vocos baseline.
    def setUp(self) -> None:
        # Builds the adaptation and the baseline networks for the capacity comparison.
        torch.manual_seed(20260809)
        self._adapted_network: VocosformerNetwork = VocosformerNetwork(VocosformerNetworkConfig())
        self._baseline_network: VocosNetwork = VocosNetwork(VocosNetworkConfig())

    def test_recorded_parameter_budgets_are_exact(self) -> None:
        # Both budgets are provenance figures reported alongside the study results.
        adapted_count: int = sum(parameter.numel() for parameter in self._adapted_network.parameters())
        baseline_count: int = sum(parameter.numel() for parameter in self._baseline_network.parameters())
        self.assertEqual(adapted_count, 13_778_690, msg=f"VocosFormer budget drifted to {adapted_count}")
        self.assertEqual(baseline_count, 13_531_650, msg=f"Vocos baseline budget drifted to {baseline_count}")

    def test_capacity_stays_within_two_percent_of_the_baseline(self) -> None:
        # Backbone capacity must not masquerade as the attention contribution.
        adapted_count: int = sum(parameter.numel() for parameter in self._adapted_network.parameters())
        baseline_count: int = sum(parameter.numel() for parameter in self._baseline_network.parameters())
        relative_difference: float = abs(adapted_count - baseline_count) / baseline_count
        self.assertLess(
            relative_difference,
            0.02,
            msg=f"Parameter matching drifted to {relative_difference:.4f} relative difference"
        )


if __name__ == "__main__":
    unittest.main()
