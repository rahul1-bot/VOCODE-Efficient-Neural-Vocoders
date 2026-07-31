# This module:
# 1. Verifies the ResBlock1 variant of the HiFi-GAN residual block: the
#    paired dilated and refinement convolutions, their weight
#    normalization, time-axis length invariance across the reference
#    kernel and dilation sets, and the residual skip path
# 2. Verifies the same properties for the lighter single-convolution
#    ResBlock2 variant used by the V3 recipe
#
# Design decisions:
# - Length invariance is asserted across every kernel and dilation pair
#   the reference recipes use, because a padding error would only surface
#   as a residual-addition failure at one specific combination
# - The residual wiring is proven by neutralizing the convolutions: with
#   zeroed weight magnitudes and biases the block must reproduce its input
#   exactly, which is a deterministic identity rather than a tolerance
# - Blocks are built at small channel counts and short time axes, since
#   the properties under test are shape and wiring properties that do not
#   depend on the reference widths
#
# Author: Rahul Sawhney

import unittest

import torch
from torch import nn

from vocode.models.hifigan.resblock import ResBlock1, ResBlock2


class ResidualBlockNeutralizer:
    # Zeroes the weight magnitude and bias of every convolution in a block,
    # reducing the block's forward pass to its residual skip path alone.
    def neutralize(self, block: nn.Module) -> None:
        # Zeroes the weight-normalization magnitude rather than the direction,
        # because the parametrization recomputes the effective weight from both.
        # Assigning into the computed weight would not survive the next
        # access, so the magnitude parameter is the only writable point
        # that reduces the effective weight to zero.
        #
        # Args:
        #     block: The residual block to neutralize, mutated in place.
        #         Every one-dimensional convolution it holds is zeroed,
        #         which for either variant is all of them.
        submodule: nn.Module
        for submodule in block.modules():
            if not isinstance(submodule, nn.Conv1d):
                continue
            weight_magnitude: torch.Tensor = submodule.parametrizations.weight.original0
            weight_magnitude.data.zero_()
            if submodule.bias is not None:
                submodule.bias.data.zero_()


class ResBlock1ForwardTest(unittest.TestCase):
    # Verifies the V1 and V2 residual block preserves its input shape,
    # produces finite output, and reduces to the identity on its skip path.
    def setUp(self) -> None:
        # Builds one seeded V1 block and the feature map every forward reads.
        torch.manual_seed(0)
        self._block: ResBlock1 = ResBlock1(channels=8, kernel_size=3, dilations=(1, 3, 5))
        self._features: torch.Tensor = torch.randn(1, 8, 32)

    def test_forward_preserves_the_input_shape(self) -> None:
        # Residual addition requires an unchanged channel and time layout.
        with torch.no_grad():
            output: torch.Tensor = self._block(self._features)
        self.assertEqual(output.shape, self._features.shape)

    def test_forward_preserves_the_time_axis_for_every_reference_kernel(self) -> None:
        # Padding is computed per kernel and dilation across the V1 recipe.
        kernel_size: int
        for kernel_size in (3, 7, 11):
            block: ResBlock1 = ResBlock1(channels=4, kernel_size=kernel_size, dilations=(1, 3, 5))
            with torch.no_grad():
                output: torch.Tensor = block(torch.randn(1, 4, 29))
            self.assertEqual(
                output.shape[-1],
                29,
                msg=f"kernel_size={kernel_size} changed the time axis length"
            )

    def test_forward_returns_finite_float32_output(self) -> None:
        # A non-finite activation would poison the generator fusion.
        with torch.no_grad():
            output: torch.Tensor = self._block(self._features)
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_forward_accepts_a_batch_of_two(self) -> None:
        # The block is batch-agnostic along the leading dimension.
        with torch.no_grad():
            output: torch.Tensor = self._block(torch.randn(2, 8, 32))
        self.assertEqual(tuple(output.shape), (2, 8, 32))

    def test_forward_transforms_its_input(self) -> None:
        # An initialized block is not a pass-through.
        with torch.no_grad():
            output: torch.Tensor = self._block(self._features)
        self.assertFalse(bool(torch.equal(output, self._features)))

    def test_neutralized_block_reproduces_its_input_exactly(self) -> None:
        # With zeroed convolutions only the residual skip path remains.
        ResidualBlockNeutralizer().neutralize(self._block)
        with torch.no_grad():
            output: torch.Tensor = self._block(self._features)
        self.assertTrue(bool(torch.equal(output, self._features)))

    def test_forward_is_deterministic_under_a_fixed_seed(self) -> None:
        # Two identically seeded blocks agree exactly on the same input.
        torch.manual_seed(7)
        first_block: ResBlock1 = ResBlock1(channels=4, kernel_size=3, dilations=(1, 3, 5))
        torch.manual_seed(7)
        second_block: ResBlock1 = ResBlock1(channels=4, kernel_size=3, dilations=(1, 3, 5))
        features: torch.Tensor = torch.randn(1, 4, 16)
        with torch.no_grad():
            self.assertTrue(bool(torch.equal(first_block(features), second_block(features))))


class ResBlock1StructureTest(unittest.TestCase):
    # Verifies the V1 and V2 residual block carries one dilated and one
    # refinement convolution per dilation, all weight normalized.
    def setUp(self) -> None:
        # Builds one seeded V1 block and sorts its state-dictionary key set.
        torch.manual_seed(0)
        self._block: ResBlock1 = ResBlock1(channels=8, kernel_size=3, dilations=(1, 3, 5))
        self._parameter_names: list[str] = sorted(self._block.state_dict().keys())

    def test_block_holds_two_convolutions_per_dilation(self) -> None:
        # Three dilations produce three dilated and three refinement convolutions.
        convolution_count: int = sum(
            1 for submodule in self._block.modules() if isinstance(submodule, nn.Conv1d)
        )
        self.assertEqual(convolution_count, 6)

    def test_block_separates_dilated_from_refinement_convolutions(self) -> None:
        # The two convolution roles are held in distinct module lists.
        dilated_names: list[str] = [
            name for name in self._parameter_names if name.startswith("_dilated_convolutions.")
        ]
        refinement_names: list[str] = [
            name for name in self._parameter_names if name.startswith("_refinement_convolutions.")
        ]
        self.assertEqual(len(dilated_names), 9)
        self.assertEqual(len(refinement_names), 9)

    def test_every_convolution_is_weight_normalized(self) -> None:
        # Weight normalization matches the reference training dynamics.
        magnitude_names: list[str] = [
            name for name in self._parameter_names if name.endswith("parametrizations.weight.original0")
        ]
        direction_names: list[str] = [
            name for name in self._parameter_names if name.endswith("parametrizations.weight.original1")
        ]
        self.assertEqual(len(magnitude_names), 6)
        self.assertEqual(len(direction_names), 6)


class ResBlock2ForwardTest(unittest.TestCase):
    # Verifies the lighter V3 residual block preserves its input shape,
    # produces finite output, and reduces to the identity on its skip path.
    def setUp(self) -> None:
        # Builds one seeded V3 block and the feature map every forward reads.
        torch.manual_seed(0)
        self._block: ResBlock2 = ResBlock2(channels=8, kernel_size=3, dilations=(1, 3))
        self._features: torch.Tensor = torch.randn(1, 8, 32)

    def test_forward_preserves_the_input_shape(self) -> None:
        # Residual addition requires an unchanged channel and time layout.
        with torch.no_grad():
            output: torch.Tensor = self._block(self._features)
        self.assertEqual(output.shape, self._features.shape)

    def test_forward_preserves_the_time_axis_for_every_v3_dilation_pair(self) -> None:
        # The V3 recipe pairs each kernel size with its own dilation pair.
        kernel_size: int
        dilations: tuple[int, int]
        for kernel_size, dilations in ((3, (1, 2)), (5, (2, 6)), (7, (3, 12))):
            block: ResBlock2 = ResBlock2(channels=4, kernel_size=kernel_size, dilations=dilations)
            with torch.no_grad():
                output: torch.Tensor = block(torch.randn(1, 4, 29))
            self.assertEqual(
                output.shape[-1],
                29,
                msg=f"kernel_size={kernel_size} dilations={dilations} changed the time axis length"
            )

    def test_forward_returns_finite_float32_output(self) -> None:
        # A non-finite activation would poison the generator fusion.
        with torch.no_grad():
            output: torch.Tensor = self._block(self._features)
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_forward_accepts_a_batch_of_two(self) -> None:
        # The block is batch-agnostic along the leading dimension.
        with torch.no_grad():
            output: torch.Tensor = self._block(torch.randn(2, 8, 32))
        self.assertEqual(tuple(output.shape), (2, 8, 32))

    def test_neutralized_block_reproduces_its_input_exactly(self) -> None:
        # With zeroed convolutions only the residual skip path remains.
        ResidualBlockNeutralizer().neutralize(self._block)
        with torch.no_grad():
            output: torch.Tensor = self._block(self._features)
        self.assertTrue(bool(torch.equal(output, self._features)))


class ResBlock2StructureTest(unittest.TestCase):
    # Verifies the lighter V3 residual block carries one weight-normalized
    # convolution per dilation and no refinement stage.
    def setUp(self) -> None:
        # Builds one seeded V3 block and sorts its state-dictionary key set.
        torch.manual_seed(0)
        self._block: ResBlock2 = ResBlock2(channels=8, kernel_size=3, dilations=(1, 3))
        self._parameter_names: list[str] = sorted(self._block.state_dict().keys())

    def test_block_holds_one_convolution_per_dilation(self) -> None:
        # The lighter variant drops the refinement convolution entirely.
        convolution_count: int = sum(
            1 for submodule in self._block.modules() if isinstance(submodule, nn.Conv1d)
        )
        self.assertEqual(convolution_count, 2)

    def test_block_declares_no_refinement_stage(self) -> None:
        # Only the single convolution list exists in the state dictionary.
        refinement_names: list[str] = [
            name for name in self._parameter_names if "_refinement_convolutions." in name
        ]
        convolution_names: list[str] = [
            name for name in self._parameter_names if name.startswith("_convolutions.")
        ]
        self.assertEqual(refinement_names, [])
        self.assertEqual(len(convolution_names), 6)

    def test_every_convolution_is_weight_normalized(self) -> None:
        # Weight normalization matches the reference checkpoint layout.
        magnitude_names: list[str] = [
            name for name in self._parameter_names if name.endswith("parametrizations.weight.original0")
        ]
        self.assertEqual(len(magnitude_names), 2)


if __name__ == "__main__":
    unittest.main()
