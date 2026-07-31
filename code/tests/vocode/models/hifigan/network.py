# This module:
# 1. Verifies the HiFi-GAN generator network rejects inconsistent
#    topology arguments: mismatched upsample and resblock argument
#    lengths, wrong dilation counts per residual-block variant, and an
#    unsupported resblock kind
# 2. Verifies the reference V1 and V3 topologies: the upsampling stage
#    count, the residual-block variant and count of the multi-receptive
#    field fusion, and the uniform width scaling of the half-width control
# 3. Verifies mel-to-waveform synthesis: the upsampling factor of the
#    time axis, the single output channel, the tanh bound, finiteness,
#    and seeded determinism
#
# Design decisions:
# - The reference V1 and V3 recipes are instantiated in full because both
#   build and run a sixteen-frame forward on CPU in a fraction of a
#   second; topology-only assertions use a compact recipe so failures name
#   the property rather than the recipe
# - Waveform values are bounded rather than pinned, because the generator
#   output at initialization carries no reference value; the tanh range is
#   the architectural invariant worth asserting
# - Weight initialization is deliberately not asserted: the reference
#   normal initialization is applied through the weight-normalization
#   parametrization, so the stored parameters keep the framework default
#   and pinning either value would encode current behavior rather than a
#   contract
#
# Author: Rahul Sawhney

import unittest

import torch
from torch import nn

from vocode.models.hifigan.network import HifiganNetwork
from vocode.models.hifigan.resblock import ResBlock1, ResBlock2


class HifiganNetworkRecipe:
    # Builds generator networks at the reference V1 and V3 topologies and
    # at a compact topology used where only structure is under test.
    def reference_v1(self, channel_multiplier: float = 1.0) -> HifiganNetwork:
        # Builds the published full-width V1 stack; the multiplier drives the
        # project half-width control off the same recipe.
        return HifiganNetwork(
            input_mel_channels=80,
            upsample_initial_channels=512,
            upsample_rates=(8, 8, 2, 2),
            upsample_kernel_sizes=(16, 16, 4, 4),
            resblock_kernel_sizes=(3, 7, 11),
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            resblock_kind="1",
            leaky_relu_slope=0.1,
            channel_multiplier=channel_multiplier
        )

    def reference_v3(self) -> HifiganNetwork:
        # Builds the published V3 stack: three upsampling stages fused by ResBlock2.
        return HifiganNetwork(
            input_mel_channels=80,
            upsample_initial_channels=256,
            upsample_rates=(8, 8, 4),
            upsample_kernel_sizes=(16, 16, 8),
            resblock_kernel_sizes=(3, 5, 7),
            resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
            resblock_kind="2",
            leaky_relu_slope=0.1
        )

    def compact(self) -> HifiganNetwork:
        # Builds a narrow two-stage stack so structural failures name the
        # property under test rather than a reference recipe.
        return HifiganNetwork(
            input_mel_channels=80,
            upsample_initial_channels=32,
            upsample_rates=(8, 8),
            upsample_kernel_sizes=(16, 16),
            resblock_kernel_sizes=(3,),
            resblock_dilation_sizes=((1, 3, 5),),
            resblock_kind="1"
        )


class HifiganNetworkConstructionValidationTest(unittest.TestCase):
    # Verifies the generator refuses topology arguments that cannot be
    # assembled into a consistent upsampling and fusion stack.
    def setUp(self) -> None:
        # Builds the accepted topology arguments each rejection case mutates once.
        torch.manual_seed(0)
        self._valid_arguments: dict[str, object] = {
            "input_mel_channels": 80,
            "upsample_initial_channels": 32,
            "upsample_rates": (8, 8),
            "upsample_kernel_sizes": (16, 16),
            "resblock_kernel_sizes": (3,),
            "resblock_dilation_sizes": ((1, 3, 5),),
            "resblock_kind": "1"
        }

    def test_upsample_rate_and_kernel_lengths_must_agree(self) -> None:
        # Each upsampling stage needs exactly one kernel size.
        invalid_arguments: dict[str, object] = dict(self._valid_arguments)
        invalid_arguments["upsample_kernel_sizes"] = (16,)
        with self.assertRaisesRegex(ValueError, "upsample_rates length"):
            HifiganNetwork(**invalid_arguments)

    def test_resblock_kernel_and_dilation_lengths_must_agree(self) -> None:
        # Each fusion branch needs exactly one dilation set.
        invalid_arguments: dict[str, object] = dict(self._valid_arguments)
        invalid_arguments["resblock_kernel_sizes"] = (3, 7)
        with self.assertRaisesRegex(ValueError, "resblock_kernel_sizes length"):
            HifiganNetwork(**invalid_arguments)

    def test_first_resblock_variant_requires_three_dilations(self) -> None:
        # ResBlock1 pairs one dilated and one refinement convolution per dilation.
        invalid_arguments: dict[str, object] = dict(self._valid_arguments)
        invalid_arguments["resblock_dilation_sizes"] = ((1, 3),)
        with self.assertRaisesRegex(ValueError, "ResBlock1 expects 3 dilations"):
            HifiganNetwork(**invalid_arguments)

    def test_second_resblock_variant_requires_two_dilations(self) -> None:
        # ResBlock2 carries a single convolution per dilation.
        invalid_arguments: dict[str, object] = dict(self._valid_arguments)
        invalid_arguments["resblock_kind"] = "2"
        with self.assertRaisesRegex(ValueError, "ResBlock2 expects 2 dilations"):
            HifiganNetwork(**invalid_arguments)

    def test_unsupported_resblock_kind_is_rejected(self) -> None:
        # The residual-block vocabulary is closed to the two reference variants.
        invalid_arguments: dict[str, object] = dict(self._valid_arguments)
        invalid_arguments["resblock_kind"] = "3"
        with self.assertRaisesRegex(ValueError, "Unsupported resblock_kind"):
            HifiganNetwork(**invalid_arguments)


class HifiganNetworkReferenceTopologyTest(unittest.TestCase):
    # Verifies the V1 and V3 reference recipes assemble the published
    # upsampling stage count and residual-block variant and count.
    def setUp(self) -> None:
        # Seeds construction and binds the recipe builder for the topology counts.
        torch.manual_seed(0)
        self._recipe: HifiganNetworkRecipe = HifiganNetworkRecipe()

    def test_v1_builds_four_upsampling_stages(self) -> None:
        # The V1 recipe upsamples in four transposed-convolution stages.
        network: HifiganNetwork = self._recipe.reference_v1()
        upsample_count: int = sum(
            1 for submodule in network.modules() if isinstance(submodule, nn.ConvTranspose1d)
        )
        self.assertEqual(upsample_count, 4)

    def test_v1_fuses_three_first_variant_blocks_per_stage(self) -> None:
        # Four stages times three kernel variants is twelve residual blocks.
        network: HifiganNetwork = self._recipe.reference_v1()
        first_variant_count: int = sum(
            1 for submodule in network.modules() if isinstance(submodule, ResBlock1)
        )
        second_variant_count: int = sum(
            1 for submodule in network.modules() if isinstance(submodule, ResBlock2)
        )
        self.assertEqual(first_variant_count, 12)
        self.assertEqual(second_variant_count, 0)

    def test_v3_builds_three_upsampling_stages_with_the_light_variant(self) -> None:
        # The V3 recipe upsamples in three stages fused by ResBlock2.
        network: HifiganNetwork = self._recipe.reference_v3()
        upsample_count: int = sum(
            1 for submodule in network.modules() if isinstance(submodule, nn.ConvTranspose1d)
        )
        second_variant_count: int = sum(
            1 for submodule in network.modules() if isinstance(submodule, ResBlock2)
        )
        first_variant_count: int = sum(
            1 for submodule in network.modules() if isinstance(submodule, ResBlock1)
        )
        self.assertEqual(upsample_count, 3)
        self.assertEqual(second_variant_count, 9)
        self.assertEqual(first_variant_count, 0)

    def test_generator_emits_a_single_waveform_channel(self) -> None:
        # The post-convolution collapses the feature stack to one channel.
        network: HifiganNetwork = self._recipe.compact()
        parameter_names: list[str] = list(network.state_dict().keys())
        self.assertIn("_post_convolution.bias", parameter_names)
        self.assertEqual(network.state_dict()["_post_convolution.bias"].shape[0], 1)


class HifiganNetworkForwardTest(unittest.TestCase):
    # Verifies mel-to-waveform synthesis: the upsampling factor, the
    # single output channel, the tanh bound, finiteness, and determinism.
    def setUp(self) -> None:
        # Seeds construction and binds the recipe builder with one short mel.
        torch.manual_seed(0)
        self._recipe: HifiganNetworkRecipe = HifiganNetworkRecipe()
        self._mel: torch.Tensor = torch.randn(1, 80, 16)

    def test_v1_upsamples_the_time_axis_by_the_rate_product(self) -> None:
        # The V1 rates multiply to the two hundred fifty-six sample hop.
        network: HifiganNetwork = self._recipe.reference_v1()
        with torch.no_grad():
            waveform: torch.Tensor = network(self._mel)
        self.assertEqual(tuple(waveform.shape), (1, 1, 16 * 256))

    def test_v3_upsamples_the_time_axis_by_the_rate_product(self) -> None:
        # The V3 rates reach the same hop through three stages.
        network: HifiganNetwork = self._recipe.reference_v3()
        with torch.no_grad():
            waveform: torch.Tensor = network(self._mel)
        self.assertEqual(tuple(waveform.shape), (1, 1, 16 * 256))

    def test_compact_recipe_upsamples_by_its_own_rate_product(self) -> None:
        # The upsampling factor follows the configured rates, not a constant.
        network: HifiganNetwork = self._recipe.compact()
        with torch.no_grad():
            waveform: torch.Tensor = network(self._mel)
        self.assertEqual(tuple(waveform.shape), (1, 1, 16 * 64))

    def test_waveform_is_bounded_by_the_output_activation(self) -> None:
        # The tanh post-activation bounds synthesis to the waveform range.
        network: HifiganNetwork = self._recipe.compact()
        with torch.no_grad():
            waveform: torch.Tensor = network(self._mel)
        self.assertGreaterEqual(float(waveform.min()), -1.0)
        self.assertLessEqual(float(waveform.max()), 1.0)

    def test_waveform_is_finite_float32(self) -> None:
        # A non-finite sample would corrupt every downstream metric.
        network: HifiganNetwork = self._recipe.compact()
        with torch.no_grad():
            waveform: torch.Tensor = network(self._mel)
        self.assertEqual(waveform.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(waveform).all()))

    def test_forward_accepts_a_batch_of_two(self) -> None:
        # Synthesis is batch-agnostic along the leading dimension.
        network: HifiganNetwork = self._recipe.compact()
        with torch.no_grad():
            waveform: torch.Tensor = network(torch.randn(2, 80, 8))
        self.assertEqual(tuple(waveform.shape), (2, 1, 8 * 64))

    def test_forward_rejects_a_mel_with_the_wrong_band_count(self) -> None:
        # The pre-convolution is bound to the configured mel band count.
        network: HifiganNetwork = self._recipe.compact()
        with self.assertRaises(RuntimeError):
            network(torch.randn(1, 40, 8))

    def test_forward_is_deterministic_under_a_fixed_seed(self) -> None:
        # Two identically seeded generators agree exactly on the same mel.
        torch.manual_seed(11)
        first_network: HifiganNetwork = self._recipe.compact()
        torch.manual_seed(11)
        second_network: HifiganNetwork = self._recipe.compact()
        mel: torch.Tensor = torch.randn(1, 80, 8)
        with torch.no_grad():
            self.assertTrue(bool(torch.equal(first_network(mel), second_network(mel))))


class HifiganNetworkChannelMultiplierTest(unittest.TestCase):
    # Verifies the channel multiplier scales stage widths uniformly, which
    # is how the project half-width control halves capacity.
    def setUp(self) -> None:
        # Seeds construction and binds the recipe builder for the width comparisons.
        torch.manual_seed(0)
        self._recipe: HifiganNetworkRecipe = HifiganNetworkRecipe()

    def test_half_width_control_reduces_the_parameter_count(self) -> None:
        # Halving the width must reduce capacity, not merely relabel it.
        full_width_parameters: int = sum(
            parameter.numel() for parameter in self._recipe.reference_v1().parameters()
        )
        half_width_parameters: int = sum(
            parameter.numel() for parameter in self._recipe.reference_v1(0.5).parameters()
        )
        self.assertLess(half_width_parameters, full_width_parameters)

    def test_half_width_control_preserves_the_topology(self) -> None:
        # Capacity changes while the stage and fusion counts stay fixed.
        half_width_network: HifiganNetwork = self._recipe.reference_v1(0.5)
        upsample_count: int = sum(
            1 for submodule in half_width_network.modules() if isinstance(submodule, nn.ConvTranspose1d)
        )
        block_count: int = sum(
            1 for submodule in half_width_network.modules() if isinstance(submodule, ResBlock1)
        )
        self.assertEqual(upsample_count, 4)
        self.assertEqual(block_count, 12)

    def test_half_width_control_preserves_the_upsampling_factor(self) -> None:
        # The control halves width without touching the time axis.
        half_width_network: HifiganNetwork = self._recipe.reference_v1(0.5)
        with torch.no_grad():
            waveform: torch.Tensor = half_width_network(torch.randn(1, 80, 8))
        self.assertEqual(tuple(waveform.shape), (1, 1, 8 * 256))

    def test_half_width_control_halves_the_first_stage_width(self) -> None:
        # The multiplier scales the pre-convolution output width directly.
        full_width_bias: torch.Tensor = self._recipe.reference_v1().state_dict()["_pre_convolution.bias"]
        half_width_bias: torch.Tensor = self._recipe.reference_v1(0.5).state_dict()["_pre_convolution.bias"]
        self.assertEqual(full_width_bias.shape[0], 512)
        self.assertEqual(half_width_bias.shape[0], 256)


if __name__ == "__main__":
    unittest.main()
