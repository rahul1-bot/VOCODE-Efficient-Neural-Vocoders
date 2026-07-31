# This module:
# 1. Verifies the FreeV generator network's pseudo-inverse mel prior: the
#    projection buffer's shape, its exclusion from checkpoints, and the
#    identity that the unrefined amplitude estimate is exactly the clamped
#    log of that analytic projection
# 2. Verifies the spectral prediction contract: both streams emit one value
#    per STFT bin and frame, phase stays inside the arctangent codomain, the
#    emitted spectra are the polar form of the predicted log-amplitude, and
#    the inverse STFT yields the waveform length arithmetic
# 3. Verifies the structural parity the strict author load depends on,
#    including the two reference normalization modules the forward pass
#    never invokes
#
# Design decisions:
# - The prior identity is isolated by building one network with zero
#   amplitude-refinement blocks, so the amplitude output is the analytic
#   estimate alone and can be recomputed from the projection buffer; this is
#   the architecture's defining efficiency claim and is otherwise masked by
#   the learned residual
# - The full official topology is used everywhere else, because construction
#   plus a sixteen-frame forward stays a fraction of a second on CPU
# - The polar identity is asserted with a tolerance rather than exact
#   equality, because it is recomputed through transcendental functions in
#   float32
#
# Author: Rahul Sawhney

import math
import unittest

import torch
from pydantic import ValidationError

from vocode.models.freev.network import FreevGeneratorOutput, FreevNetwork


class ConditioningMelBuilder:
    # Produces deterministic conditioning mels on the grid the FreeV network consumes.
    # The values are standard normal rather than a real log-mel, which is
    # acceptable because the prior identity under test is an algebraic property
    # of the projection and holds for any finite input the exponential does not
    # overflow on.
    def __init__(self, band_count: int) -> None:
        # Binds the mel band count this builder emits.
        #
        # Args:
        #     band_count: Band count of every emitted mel; it must match the
        #         column count of the pseudo-inverse projection or the lift to
        #         the bin grid is undefined.
        self._band_count: int = band_count

    def build(self, seed: int, frame_count: int) -> torch.Tensor:
        # Seeds the generator and returns one conditioning mel of the requested length.
        # Seeding immediately before sampling makes each test's input a
        # function of its own seed alone, so tests cannot influence one another
        # through the global generator regardless of execution order.
        #
        # Args:
        #     seed: Global generator seed applied before sampling.
        #     frame_count: Conditioning frames, which determines the
        #         synthesis length through the inverse-STFT arithmetic.
        #
        # Returns:
        #     A single-item batch shaped ``[1, band_count, frame_count]``.
        torch.manual_seed(seed)
        return torch.randn(1, self._band_count, frame_count)


class RecipeNetworkBuilder:
    # Builds the FreeV generator at the official topology, optionally without amplitude refinement.
    # The refinement depth is a build parameter because setting it to zero is
    # what exposes the analytic prior directly: with no residual block on the
    # amplitude branch, the emitted log-amplitude is the projection's own
    # output and can be recomputed independently from the buffer.
    def __init__(self, n_fft: int, hop_size: int) -> None:
        # Binds the STFT grid shared by the amplitude and phase streams.
        #
        # Args:
        #     n_fft: Transform size; it fixes both the bin count the heads emit
        #         and the row count of the projection.
        #     hop_size: Hop the inverse STFT expands conditioning frames by.
        self._n_fft: int = n_fft
        self._hop_size: int = hop_size

    def build(self, seed: int, amplitude_refinement_layer_count: int) -> FreevNetwork:
        # Seeds the generator and returns the constructed network in evaluation mode.
        # The seed fixes the truncated-normal initialization, so a failure is
        # reproducible from the seed alone. Evaluation mode records that the
        # assertions describe inference; this network holds no module whose
        # behavior differs between the two modes, so the call is a statement of
        # intent rather than a behavioral switch.
        #
        # Args:
        #     seed: Global generator seed applied before construction.
        #     amplitude_refinement_layer_count: Depth of the amplitude
        #         refinement stack. One reproduces the official recipe; zero
        #         removes the learned correction so the branch emits the
        #         analytic estimate unmodified.
        #
        # Returns:
        #     The constructed network at the official topology apart from the
        #     requested refinement depth.
        torch.manual_seed(seed)
        network: FreevNetwork = FreevNetwork(
            num_mels=80,
            n_fft=self._n_fft,
            hop_size=self._hop_size,
            win_size=1024,
            sampling_rate=22050,
            fmin=0.0,
            fmax=8000.0,
            psp_channel=512,
            psp_input_conv_kernel_size=7,
            psp_output_r_conv_kernel_size=7,
            psp_output_i_conv_kernel_size=7,
            convnext_layer_count=8,
            amplitude_refinement_layer_count=amplitude_refinement_layer_count,
            convnext_intermediate_dimension=1536
        )
        return network.eval()

    @property
    def bin_count(self) -> int:
        # Returns the one-sided STFT bin count both streams emit.
        return self._n_fft // 2 + 1

    @property
    def hop_size(self) -> int:
        # Returns the hop the inverse STFT expands frames by.
        return self._hop_size


class FreevPseudoInversePriorTest(unittest.TestCase):
    # Verifies that the amplitude stream is seeded by the fixed analytic projection of the mel basis.
    def setUp(self) -> None:
        # Builds the network without amplitude refinement so the analytic prior is the output.
        self._network_builder: RecipeNetworkBuilder = RecipeNetworkBuilder(n_fft=1024, hop_size=256)
        self._mel_builder: ConditioningMelBuilder = ConditioningMelBuilder(band_count=80)
        self._network: FreevNetwork = self._network_builder.build(
            seed=20260821,
            amplitude_refinement_layer_count=0
        )

    def test_projection_buffer_maps_mel_bands_back_to_stft_bins(self) -> None:
        # The pseudo-inverse of the mel basis lifts eighty bands to the full bin grid.
        projection: torch.Tensor = self._network.get_buffer("_inverse_mel_filter_bank")
        self.assertEqual(tuple(projection.shape), (self._network_builder.bin_count, 80))
        self.assertTrue(bool(torch.isfinite(projection).all()))

    def test_projection_buffer_is_excluded_from_checkpoints(self) -> None:
        # The prior is recomputed at construction, so it must not appear in the strict author load.
        # It is registered non-persistently for exactly this reason: the author
        # release carries no such entry, and a strict load rejects any key the
        # network declares but the release does not supply, so persisting it
        # would break the reproduction anchor outright.
        self.assertNotIn("_inverse_mel_filter_bank", self._network.state_dict())

    def test_unrefined_amplitude_is_the_analytic_projection(self) -> None:
        # Without refinement blocks the amplitude estimate is the clamped log of the prior itself.
        # This is the architecture's central claim made checkable: the expected
        # value is recomputed here from the buffer alone, with no learned
        # parameter involved, which establishes that the amplitude branch
        # begins from an analytic estimate rather than a projection it had to
        # learn. At the official depth of one the learned residual masks the
        # identity, which is why the fixture removes it. The tolerance covers
        # recomputing the same matrix product and logarithm in float32.
        mel: torch.Tensor = self._mel_builder.build(seed=111, frame_count=16)
        projection: torch.Tensor = self._network.get_buffer("_inverse_mel_filter_bank")
        expected_amplitude: torch.Tensor = (projection @ torch.exp(mel.float())).abs().clamp_min(1e-5).log()
        with torch.no_grad():
            output: FreevGeneratorOutput = self._network.predict_components(mel)
        self.assertTrue(
            torch.allclose(output.log_amplitude, expected_amplitude, atol=1e-5),
            msg="The unrefined amplitude estimate is not the pseudo-inverse mel projection"
        )


class FreevNetworkSpectralPredictionTest(unittest.TestCase):
    # Verifies the shapes, dtype, and length arithmetic of the amplitude and phase streams.
    def setUp(self) -> None:
        # Builds the official network in evaluation mode.
        self._network_builder: RecipeNetworkBuilder = RecipeNetworkBuilder(n_fft=1024, hop_size=256)
        self._mel_builder: ConditioningMelBuilder = ConditioningMelBuilder(band_count=80)
        self._network: FreevNetwork = self._network_builder.build(
            seed=20260822,
            amplitude_refinement_layer_count=1
        )

    def test_both_streams_emit_one_value_per_bin_and_frame(self) -> None:
        # Amplitude and phase are predicted on the full one-sided STFT grid.
        mel: torch.Tensor = self._mel_builder.build(seed=112, frame_count=16)
        with torch.no_grad():
            output: FreevGeneratorOutput = self._network.predict_components(mel)
        expected_shape: tuple[int, int, int] = (1, self._network_builder.bin_count, 16)
        self.assertEqual(tuple(output.log_amplitude.shape), expected_shape)
        self.assertEqual(tuple(output.phase.shape), expected_shape)
        self.assertEqual(tuple(output.real_spectrum.shape), expected_shape)
        self.assertEqual(tuple(output.imaginary_spectrum.shape), expected_shape)

    def test_waveform_length_follows_the_centered_istft_arithmetic(self) -> None:
        # The combination expands frames to (frames - 1) * hop samples on one channel.
        mel: torch.Tensor = self._mel_builder.build(seed=113, frame_count=16)
        with torch.no_grad():
            output: FreevGeneratorOutput = self._network.predict_components(mel)
        expected_samples: int = 15 * self._network_builder.hop_size
        self.assertEqual(
            tuple(output.waveform.shape),
            (1, 1, expected_samples),
            msg=f"Expected one channel of {expected_samples} samples, got {tuple(output.waveform.shape)}"
        )

    def test_forward_returns_the_component_waveform(self) -> None:
        # The synthesis entry point is the waveform member of the component bundle.
        mel: torch.Tensor = self._mel_builder.build(seed=114, frame_count=16)
        with torch.no_grad():
            waveform: torch.Tensor = self._network(mel)
            output: FreevGeneratorOutput = self._network.predict_components(mel)
        self.assertTrue(torch.equal(waveform, output.waveform))

    def test_components_are_finite_float32(self) -> None:
        # The complex reconstruction runs in float32 and must not emit non-finite values.
        mel: torch.Tensor = self._mel_builder.build(seed=115, frame_count=16)
        with torch.no_grad():
            output: FreevGeneratorOutput = self._network.predict_components(mel)
        self.assertEqual(output.log_amplitude.dtype, torch.float32)
        self.assertEqual(output.phase.dtype, torch.float32)
        self.assertEqual(output.waveform.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(output.log_amplitude).all()))
        self.assertTrue(bool(torch.isfinite(output.phase).all()))
        self.assertTrue(bool(torch.isfinite(output.waveform).all()))

    def test_phase_lies_inside_the_arctangent_codomain(self) -> None:
        # Predicting real and imaginary parts yields wrapped phase without unwrapping.
        mel: torch.Tensor = self._mel_builder.build(seed=116, frame_count=16)
        with torch.no_grad():
            output: FreevGeneratorOutput = self._network.predict_components(mel)
        self.assertGreaterEqual(float(output.phase.min()), -math.pi)
        self.assertLessEqual(float(output.phase.max()), math.pi)

    def test_spectra_are_the_polar_form_of_the_predicted_amplitude(self) -> None:
        # The squared magnitude of the emitted spectrum must equal the exponentiated log-amplitude.
        mel: torch.Tensor = self._mel_builder.build(seed=117, frame_count=16)
        with torch.no_grad():
            output: FreevGeneratorOutput = self._network.predict_components(mel)
        squared_magnitude: torch.Tensor = output.real_spectrum.pow(2) + output.imaginary_spectrum.pow(2)
        expected_magnitude: torch.Tensor = torch.exp(output.log_amplitude).pow(2)
        self.assertTrue(
            torch.allclose(squared_magnitude, expected_magnitude, atol=1e-4),
            msg="Emitted spectra are not the polar form of the predicted log-amplitude"
        )


class FreevNetworkReferenceParityTest(unittest.TestCase):
    # Verifies the checkpoint-parity structure and the conditioning-layout guard.
    def setUp(self) -> None:
        # Builds the official network whose structure the strict author load depends on.
        self._network_builder: RecipeNetworkBuilder = RecipeNetworkBuilder(n_fft=1024, hop_size=256)
        self._mel_builder: ConditioningMelBuilder = ConditioningMelBuilder(band_count=80)
        self._network: FreevNetwork = self._network_builder.build(
            seed=20260823,
            amplitude_refinement_layer_count=1
        )

    def test_unused_reference_normalizations_are_materialized(self) -> None:
        # The released checkpoint carries these modules, so a strict load requires them to exist.
        # They are the mirror image of the projection buffer: that one is
        # excluded because the release does not carry it, while these are
        # declared despite never being invoked because the release does. Both
        # keys and shapes are asserted, since a strict load matches on both and
        # a normalization built at the wrong width would fail only when the
        # genuine release bytes are present.
        state_keys: set[str] = set(self._network.state_dict().keys())
        self.assertIn("norm2.weight", state_keys)
        self.assertIn("norm2.bias", state_keys)
        self.assertIn("final_layer_norm2.weight", state_keys)
        self.assertIn("final_layer_norm2.bias", state_keys)
        self.assertEqual(tuple(self._network.norm2.normalized_shape), (512,))
        self.assertEqual(tuple(self._network.final_layer_norm2.normalized_shape), (512,))

    def test_amplitude_refinement_stack_holds_the_declared_depth(self) -> None:
        # The refinement stream is one block deep in the official recipe.
        # The eight-to-one ratio against the phase stack is the efficiency
        # claim in numbers: amplitude needs only a correction because the prior
        # supplies the estimate, whereas phase is discarded entirely by the mel
        # and has to be produced from nothing.
        self.assertEqual(len(self._network.convnext2), 1)
        self.assertEqual(len(self._network.convnext), 8)

    def test_unbatched_mel_is_rejected(self) -> None:
        # The streams require an explicit batch axis rather than guessing the layout.
        with self.assertRaises(ValueError):
            self._network.predict_components(torch.zeros(80, 16))

    def test_four_dimensional_mel_is_rejected(self) -> None:
        # A channel axis is not part of the conditioning contract.
        with self.assertRaises(ValueError):
            self._network.predict_components(torch.zeros(1, 1, 80, 16))

    def test_component_bundle_is_frozen(self) -> None:
        # Predicted components cannot be rewritten between prediction and loss computation.
        mel: torch.Tensor = self._mel_builder.build(seed=118, frame_count=8)
        with torch.no_grad():
            output: FreevGeneratorOutput = self._network.predict_components(mel)
        with self.assertRaises(ValidationError):
            output.phase = torch.zeros(1)
