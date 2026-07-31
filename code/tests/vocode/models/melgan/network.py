# This module:
# 1. Verifies the MelGAN generator network at the Seungwon Park reference
#    topology: one transposed-convolution stage and one residual stack per
#    upsample factor, the single tanh-bounded output channel, and the
#    positional layout of the upstream sequential container
# 2. Verifies mel-to-waveform synthesis: the upsampling factor of the time
#    axis, the output bound, finiteness, batch handling, and seeded
#    determinism
# 3. Verifies the upstream log-mel normalization: the recorded shift and
#    scale, and that the forward pass actually applies them ahead of the
#    convolutional stack
# 4. Verifies the residual stack: its three dilated blocks and three
#    learned shortcuts, their weight normalization, and time-axis
#    invariance
#
# Design decisions:
# - The reference recipe is instantiated in full because the MelGAN
#   generator builds and runs a sixteen-frame forward on CPU in a fraction
#   of a second
# - The upstream log-mel shift and scale are verified twice: as recorded
#   constants, and by feeding the pre-normalized mel straight into the
#   sequential container and requiring bit-identical agreement with the
#   forward pass, which proves the constants are applied and not merely
#   declared; that second check reads the private container because the
#   normalization is fused into forward and has no other observable surface
# - Waveform values are bounded rather than pinned, because the generator
#   output at initialization carries no reference value
#
# Author: Rahul Sawhney

import unittest

import torch
from torch import nn

from vocode.models.melgan.network import MelganNetwork, ResStack


class MelganNetworkRecipe:
    # Builds generator networks at the Seungwon Park reference topology and
    # at a compact topology used where only structure is under test.
    def reference(self) -> MelganNetwork:
        # Builds the published Seungwon Park four-stage generator.
        return MelganNetwork(
            input_mel_channels=80,
            ngf=32,
            upsample_factors=(8, 8, 2, 2),
            leaky_relu_slope=0.2
        )

    def compact(self) -> MelganNetwork:
        # Builds a narrow two-stage generator so structural failures name the
        # property under test rather than the reference recipe.
        return MelganNetwork(
            input_mel_channels=80,
            ngf=4,
            upsample_factors=(8, 8),
            leaky_relu_slope=0.2
        )


class MelganNetworkTopologyTest(unittest.TestCase):
    # Verifies the reference recipe assembles one upsampling stage and one
    # residual stack per factor and emits a single bounded waveform channel.
    def setUp(self) -> None:
        # Builds one seeded generator at the reference topology.
        torch.manual_seed(0)
        self._network: MelganNetwork = MelganNetworkRecipe().reference()

    def test_network_builds_one_upsampling_stage_per_factor(self) -> None:
        # Four upsample factors produce four transposed convolutions.
        upsample_count: int = sum(
            1 for submodule in self._network.modules() if isinstance(submodule, nn.ConvTranspose1d)
        )
        self.assertEqual(upsample_count, 4)

    def test_network_builds_one_residual_stack_per_upsampling_stage(self) -> None:
        # Each stage is followed by its own residual dilated stack.
        stack_count: int = sum(
            1 for submodule in self._network.modules() if isinstance(submodule, ResStack)
        )
        self.assertEqual(stack_count, 4)

    def test_network_ends_in_a_single_bounded_output_channel(self) -> None:
        # The output convolution collapses to one channel before the tanh.
        activation_count: int = sum(
            1 for submodule in self._network.modules() if isinstance(submodule, nn.Tanh)
        )
        convolutions: list[nn.Conv1d] = [
            submodule for submodule in self._network.modules() if isinstance(submodule, nn.Conv1d)
        ]
        self.assertEqual(activation_count, 1)
        self.assertEqual(convolutions[-1].out_channels, 1)

    def test_network_keeps_the_upstream_positional_layout(self) -> None:
        # A name-only state-dict remap requires the sequential container.
        parameter_names: list[str] = list(self._network.state_dict().keys())
        parameter_name: str
        for parameter_name in parameter_names:
            self.assertTrue(
                parameter_name.startswith("_network."),
                msg=f"{parameter_name} is outside the sequential container"
            )

    def test_network_scales_its_first_stage_width_from_the_growth_factor(self) -> None:
        # The input convolution widens to the growth factor times two per stage.
        input_convolution_bias: torch.Tensor = self._network.state_dict()["_network.1.bias"]
        self.assertEqual(input_convolution_bias.shape[0], 32 * (2 ** 4))


class MelganNetworkForwardTest(unittest.TestCase):
    # Verifies mel-to-waveform synthesis produces the upsampled, bounded,
    # finite waveform the discriminator ensemble and metrics consume.
    def setUp(self) -> None:
        # Seeds construction and binds the recipe builder with one short mel.
        torch.manual_seed(0)
        self._recipe: MelganNetworkRecipe = MelganNetworkRecipe()
        self._mel: torch.Tensor = torch.randn(1, 80, 16)

    def test_reference_recipe_upsamples_by_the_factor_product(self) -> None:
        # The reference factors multiply to the two hundred fifty-six sample hop.
        network: MelganNetwork = self._recipe.reference()
        with torch.no_grad():
            waveform: torch.Tensor = network(self._mel)
        self.assertEqual(tuple(waveform.shape), (1, 1, 16 * 256))

    def test_compact_recipe_upsamples_by_its_own_factor_product(self) -> None:
        # The upsampling factor follows the configured factors, not a constant.
        network: MelganNetwork = self._recipe.compact()
        with torch.no_grad():
            waveform: torch.Tensor = network(self._mel)
        self.assertEqual(tuple(waveform.shape), (1, 1, 16 * 64))

    def test_waveform_is_bounded_by_the_output_activation(self) -> None:
        # The tanh output bounds synthesis to the waveform range.
        network: MelganNetwork = self._recipe.compact()
        with torch.no_grad():
            waveform: torch.Tensor = network(self._mel)
        self.assertGreaterEqual(float(waveform.min()), -1.0)
        self.assertLessEqual(float(waveform.max()), 1.0)

    def test_waveform_is_finite_float32(self) -> None:
        # A non-finite sample would corrupt every downstream metric.
        network: MelganNetwork = self._recipe.compact()
        with torch.no_grad():
            waveform: torch.Tensor = network(self._mel)
        self.assertEqual(waveform.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(waveform).all()))

    def test_forward_accepts_a_batch_of_two(self) -> None:
        # Synthesis is batch-agnostic along the leading dimension.
        network: MelganNetwork = self._recipe.compact()
        with torch.no_grad():
            waveform: torch.Tensor = network(torch.randn(2, 80, 8))
        self.assertEqual(tuple(waveform.shape), (2, 1, 8 * 64))

    def test_forward_rejects_a_mel_with_the_wrong_band_count(self) -> None:
        # The input convolution is bound to the configured mel band count.
        network: MelganNetwork = self._recipe.compact()
        with self.assertRaises(RuntimeError):
            network(torch.randn(1, 40, 8))

    def test_forward_is_deterministic_under_a_fixed_seed(self) -> None:
        # Two identically seeded generators agree exactly on the same mel.
        torch.manual_seed(5)
        first_network: MelganNetwork = self._recipe.compact()
        torch.manual_seed(5)
        second_network: MelganNetwork = self._recipe.compact()
        mel: torch.Tensor = torch.randn(1, 80, 8)
        with torch.no_grad():
            self.assertTrue(bool(torch.equal(first_network(mel), second_network(mel))))


class MelganNetworkInputNormalizationTest(unittest.TestCase):
    # Verifies the upstream log-mel normalization the released checkpoint
    # requires: the recorded constants and their application in the forward.
    def test_network_records_the_upstream_shift_and_scale(self) -> None:
        # The released weights need the (mel + 5) / 5 input normalization.
        self.assertEqual(MelganNetwork._INPUT_MEL_SHIFT, 5.0)
        self.assertEqual(MelganNetwork._INPUT_MEL_SCALE, 5.0)

    def test_forward_applies_the_normalization_before_the_stack(self) -> None:
        # Feeding the pre-normalized mel straight to the sequential container
        # reproduces the forward exactly, which proves the shift and scale are
        # applied to the input rather than only recorded on the class.
        torch.manual_seed(0)
        network: MelganNetwork = MelganNetworkRecipe().compact()
        mel: torch.Tensor = torch.randn(1, 80, 8)
        normalized_mel: torch.Tensor = (mel + 5.0) / 5.0
        with torch.no_grad():
            through_forward: torch.Tensor = network(mel)
            through_container: torch.Tensor = network._network(normalized_mel)
        self.assertTrue(bool(torch.equal(through_forward, through_container)))


class ResStackTest(unittest.TestCase):
    # Verifies the residual stack combines three dilated blocks with three
    # learned shortcut convolutions without changing its input shape.
    def setUp(self) -> None:
        # Builds one seeded stack and the feature map every forward reads.
        torch.manual_seed(0)
        self._stack: ResStack = ResStack(channels=16)
        self._features: torch.Tensor = torch.randn(1, 16, 32)

    def test_stack_preserves_the_input_shape(self) -> None:
        # Additive combination requires an unchanged channel and time layout.
        with torch.no_grad():
            output: torch.Tensor = self._stack(self._features)
        self.assertEqual(output.shape, self._features.shape)

    def test_stack_holds_three_blocks_and_three_shortcuts(self) -> None:
        # Two convolutions per block plus one shortcut each gives nine.
        convolution_count: int = sum(
            1 for submodule in self._stack.modules() if isinstance(submodule, nn.Conv1d)
        )
        parameter_names: list[str] = list(self._stack.state_dict().keys())
        shortcut_names: list[str] = [
            name for name in parameter_names if name.startswith("_shortcuts.")
        ]
        block_names: list[str] = [
            name for name in parameter_names if name.startswith("_blocks.")
        ]
        self.assertEqual(convolution_count, 9)
        self.assertEqual(len(shortcut_names), 9)
        self.assertEqual(len(block_names), 18)

    def test_every_convolution_is_weight_normalized(self) -> None:
        # Weight normalization matches the released checkpoint layout.
        magnitude_names: list[str] = [
            name for name in self._stack.state_dict()
            if name.endswith("parametrizations.weight.original0")
        ]
        self.assertEqual(len(magnitude_names), 9)

    def test_stack_returns_finite_output(self) -> None:
        # A non-finite activation would poison the upsampling chain.
        with torch.no_grad():
            output: torch.Tensor = self._stack(self._features)
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_stack_accepts_a_batch_of_two(self) -> None:
        # The stack is batch-agnostic along the leading dimension.
        with torch.no_grad():
            output: torch.Tensor = self._stack(torch.randn(2, 16, 32))
        self.assertEqual(tuple(output.shape), (2, 16, 32))

    def test_stack_transforms_its_input(self) -> None:
        # An initialized stack is not a pass-through.
        with torch.no_grad():
            output: torch.Tensor = self._stack(self._features)
        self.assertFalse(bool(torch.equal(output, self._features)))


if __name__ == "__main__":
    unittest.main()
