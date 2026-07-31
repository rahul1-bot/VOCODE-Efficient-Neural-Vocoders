# This module:
# 1. Verifies the BigVGAN generator network: the upsampling channel
#    schedule, the residual dilation geometry, the synthesis shape
#    contract, and the bounded output range of both final stages
# 2. Verifies the anti-aliased periodic activation chain: the snake and
#    snakebeta closed forms at initialization and the length invariance of
#    the filtered up-down resampling pair
# 3. Verifies the weight-normalization parameter layout that the published
#    weight adapter renames into
#
# Design decisions:
# - Topology, range, and activation assertions run on a miniature recipe so
#   the file stays inside the runtime budget; the published 24 kHz 100-band
#   recipe is constructed once for its channel schedule and exercised with a
#   single four-frame synthesis, the largest affordable BigVGAN forward
# - Snake and snakebeta are asserted against their closed forms at the
#   initialized parameters, where the log-scale alpha and beta both equal
#   one, which is exact deterministic mathematics rather than a pinned float
# - The resampling kernels are asserted through the unit-sum invariant of
#   the windowed-sinc design instead of tabulated coefficients
#
# Author: Rahul Sawhney

import unittest
from typing import Literal

import torch

from vocode.models.bigvgan.network import BigvganNetwork


class MiniatureGeneratorRecipe:
    # Configuration object building the miniature BigVGAN generators used by the assertions.
    # Exactly the three fields the assertions vary are parameters; every other
    # setting is fixed in the body, so a test that constructs a variant states
    # only what makes that variant different from the others.
    def __init__(
        self,
        activation: Literal["snake", "snakebeta"],
        resblock: Literal["1", "2"],
        use_tanh_at_final: bool
    ) -> None:
        # Binds the activation, residual variant, and final-stage choice this recipe varies.
        #
        # Args:
        #     activation: Periodic activation the generator is built with;
        #         selecting it decides whether a beta parameter exists.
        #     resblock: Residual variant; selecting it decides whether the
        #         blocks carry a refinement convolution list.
        #     use_tanh_at_final: Output bounding; selecting it decides whether
        #         the range is open or closed at unity.
        self._activation: Literal["snake", "snakebeta"] = activation
        self._resblock: Literal["1", "2"] = resblock
        self._use_tanh_at_final: bool = use_tanh_at_final

    def build(self) -> BigvganNetwork:
        # Returns the miniature generator at the bound variant selection.
        # The reduced widths and the two upsampling stages preserve every
        # structural relation under test, the channel halving, the block count
        # arithmetic, and the length expansion, while making construction and a
        # forward pass negligible.
        #
        # Returns:
        #     A generator whose stages expand each frame by four samples in
        #     total and whose residual bank holds two blocks per stage.
        return BigvganNetwork(
            num_mels=8,
            upsample_initial_channel=8,
            upsample_rates=(2, 2),
            upsample_kernel_sizes=(4, 4),
            resblock_kernel_sizes=(3, 5),
            resblock_dilation_sizes=((1, 3), (1, 3)),
            resblock=self._resblock,
            activation=self._activation,
            snake_logscale=True,
            use_tanh_at_final=self._use_tanh_at_final
        )


class PublishedGeneratorRecipe:
    # Builds the published NVIDIA BigVGAN-base 24 kHz 100-band generator topology.
    # It exists because the channel schedule and block count of the real
    # network are part of the reproduction claim and cannot be inferred from
    # the miniature recipe; the strict author load depends on both.
    def build(self) -> BigvganNetwork:
        # Returns the published generator at its full channel schedule.
        # This construction allocates the full parameter set, so the class
        # using it builds once in setUp and keeps its forward pass to the
        # smallest conditioning mel that still exercises the whole chain.
        #
        # Returns:
        #     The generator at the published topology, expanding each frame to
        #     256 samples across four stages.
        return BigvganNetwork(
            num_mels=100,
            upsample_initial_channel=512,
            upsample_rates=(8, 8, 2, 2),
            upsample_kernel_sizes=(16, 16, 4, 4),
            resblock_kernel_sizes=(3, 7, 11),
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            resblock="1",
            activation="snakebeta",
            snake_logscale=True
        )


class BigvganGeneratorTopologyTest(unittest.TestCase):
    # Verifies the upsampling channel schedule and the residual block geometry of the generator.
    def setUp(self) -> None:
        # Builds the miniature snakebeta generator whose topology is under test.
        torch.manual_seed(1234)
        self._recipe: MiniatureGeneratorRecipe = MiniatureGeneratorRecipe(
            activation="snakebeta",
            resblock="1",
            use_tanh_at_final=True
        )
        self._network: BigvganNetwork = self._recipe.build()

    def test_conditioning_convolution_accepts_the_configured_mel_bands(self) -> None:
        # The pre-convolution maps mel bands onto the initial upsampling width.
        self.assertEqual(self._network.conv_pre.in_channels, 8)
        self.assertEqual(self._network.conv_pre.out_channels, 8)

    def test_upsample_stage_count_matches_the_configured_rates(self) -> None:
        # One transposed-convolution stage exists per configured upsample rate.
        self.assertEqual(len(self._network.ups), 2)

    def test_upsample_channels_halve_at_every_stage(self) -> None:
        # Each upsampling stage halves the channel width of the previous stage.
        expected_widths: tuple[tuple[int, int], ...] = ((8, 4), (4, 2))
        stage_index: int
        expected_pair: tuple[int, int]
        for stage_index, expected_pair in enumerate(expected_widths):
            layer: torch.nn.ConvTranspose1d = self._network.ups[stage_index][0]
            self.assertEqual(
                (layer.in_channels, layer.out_channels),
                expected_pair,
                msg=f"Unexpected channel widths at upsample stage {stage_index}"
            )

    def test_residual_block_count_is_stages_times_kernels(self) -> None:
        # Every upsampling stage carries one residual block per configured kernel size.
        self.assertEqual(len(self._network.resblocks), 4)

    def test_residual_dilations_drive_length_preserving_padding(self) -> None:
        # Dilated convolutions pad by half the dilated kernel extent so the time axis is invariant.
        first_block: torch.nn.Module = self._network.resblocks[0]
        dilation_index: int
        expected_dilation: int
        for dilation_index, expected_dilation in enumerate((1, 3)):
            convolution: torch.nn.Conv1d = first_block.convs1[dilation_index]
            self.assertEqual(convolution.dilation, (expected_dilation,))
            self.assertEqual(
                convolution.padding,
                (expected_dilation,),
                msg=f"Kernel size three at dilation {expected_dilation} requires that padding"
            )

    def test_refinement_convolutions_are_undilated(self) -> None:
        # The second convolution of each residual pair stays at dilation one.
        first_block: torch.nn.Module = self._network.resblocks[0]
        convolution: torch.nn.Conv1d
        for convolution in first_block.convs2:
            self.assertEqual(convolution.dilation, (1,))

    def test_second_residual_variant_drops_the_refinement_convolutions(self) -> None:
        # The resblock two variant carries a single convolution list rather than a pair.
        variant_recipe: MiniatureGeneratorRecipe = MiniatureGeneratorRecipe(
            activation="snakebeta",
            resblock="2",
            use_tanh_at_final=True
        )
        variant_network: BigvganNetwork = variant_recipe.build()
        variant_block: torch.nn.Module = variant_network.resblocks[0]
        self.assertTrue(hasattr(variant_block, "convs"))
        self.assertFalse(hasattr(variant_block, "convs2"))
        self.assertEqual(len(variant_block.activations), len(variant_block.convs))

    def test_output_convolution_projects_to_one_waveform_channel(self) -> None:
        # The post-convolution reduces the final width to the single waveform channel.
        self.assertEqual(self._network.conv_post.in_channels, 2)
        self.assertEqual(self._network.conv_post.out_channels, 1)


class BigvganSynthesisShapeTest(unittest.TestCase):
    # Verifies the mel-to-waveform shape contract, the bounded output range, and the input guard.
    def setUp(self) -> None:
        # Builds the tanh-headed miniature generator and the six-frame conditioning mel.
        torch.manual_seed(1234)
        self._recipe: MiniatureGeneratorRecipe = MiniatureGeneratorRecipe(
            activation="snakebeta",
            resblock="1",
            use_tanh_at_final=True
        )
        self._network: BigvganNetwork = self._recipe.build()
        self._mel: torch.Tensor = torch.randn(1, 8, 6)

    def test_synthesis_length_is_the_product_of_the_upsample_rates(self) -> None:
        # Rates two and two expand every conditioning frame into four waveform samples.
        with torch.no_grad():
            waveform: torch.Tensor = self._network(self._mel)
        self.assertEqual(tuple(waveform.shape), (1, 1, 24))

    def test_batch_dimension_is_preserved(self) -> None:
        # Batched conditioning produces one waveform row per batch element.
        batched_mel: torch.Tensor = torch.randn(2, 8, 6)
        with torch.no_grad():
            waveform: torch.Tensor = self._network(batched_mel)
        self.assertEqual(tuple(waveform.shape), (2, 1, 24))

    def test_synthesis_is_finite(self) -> None:
        # The synthesis chain produces no non-finite samples at initialization.
        with torch.no_grad():
            waveform: torch.Tensor = self._network(self._mel)
        self.assertTrue(bool(torch.isfinite(waveform).all()))

    def test_tanh_final_stage_bounds_the_waveform(self) -> None:
        # The hyperbolic-tangent head keeps every sample strictly inside the unit interval.
        with torch.no_grad():
            waveform: torch.Tensor = self._network(self._mel)
        self.assertLess(float(waveform.abs().max()), 1.0)

    def test_clamped_final_stage_bounds_the_waveform(self) -> None:
        # The clamped head keeps every sample inside the closed unit interval.
        clamped_recipe: MiniatureGeneratorRecipe = MiniatureGeneratorRecipe(
            activation="snakebeta",
            resblock="1",
            use_tanh_at_final=False
        )
        clamped_network: BigvganNetwork = clamped_recipe.build()
        with torch.no_grad():
            waveform: torch.Tensor = clamped_network(self._mel)
        self.assertLessEqual(float(waveform.abs().max()), 1.0)

    def test_mel_without_three_dimensions_is_rejected(self) -> None:
        # A two-dimensional mel is refused with the expected-layout message.
        with self.assertRaisesRegex(ValueError, "Expected mel shape"):
            self._network(torch.randn(8, 6))


class BigvganAntiAliasedActivationTest(unittest.TestCase):
    # Verifies the periodic activation closed forms and the length invariance of the resampling pair.
    def setUp(self) -> None:
        # Builds the snakebeta generator and the feature block the activation chain is applied to.
        torch.manual_seed(1234)
        self._beta_recipe: MiniatureGeneratorRecipe = MiniatureGeneratorRecipe(
            activation="snakebeta",
            resblock="1",
            use_tanh_at_final=True
        )
        self._network: BigvganNetwork = self._beta_recipe.build()
        self._features: torch.Tensor = torch.randn(1, 2, 16)

    def test_snakebeta_reduces_to_the_closed_form_at_initialization(self) -> None:
        # Log-scale alpha and beta both start at one, so the activation is x plus sine squared.
        # Log-scale storage initializes both parameters at zero and
        # exponentiates them at use, which is what pins the initial factor at
        # exactly one and makes the expected value closed-form rather than a
        # recorded constant. The reciprocal's guard epsilon is carried into the
        # expectation so the comparison targets the implemented expression and
        # not an idealized one. The activation is read off the generator's own
        # head rather than constructed here, so the test also confirms the
        # configured variant is what the network actually built.
        activation: torch.nn.Module = self._network.activation_post.act
        with torch.no_grad():
            activated: torch.Tensor = activation(self._features)
        expected: torch.Tensor = self._features + torch.sin(self._features).pow(2) / (1.0 + 1e-9)
        self.assertTrue(bool(torch.allclose(activated, expected, atol=1e-6)))

    def test_snakebeta_exposes_an_independent_beta_parameter(self) -> None:
        # The snakebeta variant learns the amplitude term separately from the frequency term.
        activation: torch.nn.Module = self._network.activation_post.act
        self.assertTrue(hasattr(activation, "beta"))
        self.assertEqual(tuple(activation.alpha.shape), (2,))
        self.assertEqual(tuple(activation.beta.shape), (2,))

    def test_snake_variant_has_no_beta_parameter(self) -> None:
        # The snake variant reuses alpha for both terms and therefore owns no beta.
        snake_recipe: MiniatureGeneratorRecipe = MiniatureGeneratorRecipe(
            activation="snake",
            resblock="1",
            use_tanh_at_final=True
        )
        snake_network: BigvganNetwork = snake_recipe.build()
        activation: torch.nn.Module = snake_network.activation_post.act
        self.assertFalse(hasattr(activation, "beta"))

    def test_upsampling_kernel_is_unit_sum(self) -> None:
        # The windowed-sinc interpolation kernel is normalized so it preserves signal level.
        # Unit sum is unit gain at DC, which is the invariant that keeps the
        # anti-aliasing apparatus transparent: a kernel summing to anything
        # else would rescale the signal every time an activation is applied,
        # compounding across every block. Asserting the design invariant rather
        # than tabulated taps keeps the test valid under any kernel length,
        # and the places bound accommodates the float32 accumulation of the
        # normalizing division.
        kernel: torch.Tensor = self._network.activation_post.upsample.get_buffer("filter")
        self.assertEqual(kernel.shape[0], 1)
        self.assertEqual(kernel.shape[1], 1)
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=5)

    def test_lowpass_kernel_is_unit_sum(self) -> None:
        # The decimation low-pass kernel carries the same unit-sum normalization.
        kernel: torch.Tensor = self._network.activation_post.downsample.lowpass.get_buffer("filter")
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=5)

    def test_filtered_activation_preserves_the_time_axis(self) -> None:
        # Upsampling by two and decimating by two returns the original frame count.
        with torch.no_grad():
            activated: torch.Tensor = self._network.activation_post(self._features)
        self.assertEqual(tuple(activated.shape), tuple(self._features.shape))

    def test_residual_block_preserves_channels_and_time(self) -> None:
        # Residual blocks are shape-preserving refinements of their input.
        block_input: torch.Tensor = torch.randn(1, 4, 12)
        with torch.no_grad():
            block_output: torch.Tensor = self._network.resblocks[0](block_input)
        self.assertEqual(tuple(block_output.shape), tuple(block_input.shape))


class BigvganPublishedRecipeTest(unittest.TestCase):
    # Verifies the published 24 kHz 100-band topology and its 256-samples-per-frame synthesis.
    def setUp(self) -> None:
        # Constructs the published generator once for the whole class.
        torch.manual_seed(1234)
        self._recipe: PublishedGeneratorRecipe = PublishedGeneratorRecipe()
        self._network: BigvganNetwork = self._recipe.build()

    def test_published_channel_schedule_descends_from_five_hundred_twelve(self) -> None:
        # The base recipe halves 512 channels across its four upsampling stages.
        expected_widths: tuple[tuple[int, int], ...] = ((512, 256), (256, 128), (128, 64), (64, 32))
        stage_index: int
        expected_pair: tuple[int, int]
        for stage_index, expected_pair in enumerate(expected_widths):
            layer: torch.nn.ConvTranspose1d = self._network.ups[stage_index][0]
            self.assertEqual(
                (layer.in_channels, layer.out_channels),
                expected_pair,
                msg=f"Unexpected published channel widths at stage {stage_index}"
            )

    def test_published_recipe_carries_twelve_residual_blocks(self) -> None:
        # Four upsampling stages times three kernel sizes give twelve blocks.
        self.assertEqual(len(self._network.resblocks), 12)

    def test_published_recipe_expands_each_frame_to_two_hundred_fifty_six_samples(self) -> None:
        # Rates eight, eight, two, and two multiply to the published hop size.
        mel: torch.Tensor = torch.randn(1, 100, 4)
        with torch.no_grad():
            waveform: torch.Tensor = self._network(mel)
        self.assertEqual(tuple(waveform.shape), (1, 1, 1024))
        self.assertTrue(bool(torch.isfinite(waveform).all()))


class BigvganWeightNormalizationLayoutTest(unittest.TestCase):
    # Verifies the weight-normalization state layout the published weight adapter renames into.
    def setUp(self) -> None:
        # Builds the miniature generator and collects its state-dictionary key inventory.
        torch.manual_seed(1234)
        self._recipe: MiniatureGeneratorRecipe = MiniatureGeneratorRecipe(
            activation="snakebeta",
            resblock="1",
            use_tanh_at_final=True
        )
        self._network: BigvganNetwork = self._recipe.build()
        self._state_keys: set[str] = set(self._network.state_dict().keys())

    def test_convolutions_store_parametrized_weight_components(self) -> None:
        # Weight normalization splits every kernel into the magnitude and direction tensors.
        self.assertIn("conv_pre.parametrizations.weight.original0", self._state_keys)
        self.assertIn("conv_pre.parametrizations.weight.original1", self._state_keys)
        self.assertIn("conv_post.parametrizations.weight.original0", self._state_keys)

    def test_upstream_weight_norm_names_are_absent(self) -> None:
        # The upstream weight_g and weight_v names never appear locally, which is why the adapter renames.
        upstream_named_keys: list[str] = [
            key for key in self._state_keys if key.endswith((".weight_g", ".weight_v"))
        ]
        self.assertEqual(upstream_named_keys, [])

    def test_resampling_filters_are_persistent_state(self) -> None:
        # The anti-aliasing kernels travel with the state dictionary as buffers.
        filter_keys: list[str] = [key for key in self._state_keys if key.endswith(".filter")]
        self.assertGreater(len(filter_keys), 0)
