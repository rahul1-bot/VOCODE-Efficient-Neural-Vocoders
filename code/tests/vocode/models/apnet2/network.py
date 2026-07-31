# This module:
# 1. Verifies the APNet2 generator network's spectral prediction contract:
#    the amplitude and phase streams both emit one value per STFT bin and
#    frame, and the inverse-STFT combination yields the waveform
# 2. Verifies the mathematical properties the loss composition depends on:
#    the phase stream produces wrapped phase inside the arctangent codomain,
#    and the real and imaginary spectra are the polar form of the predicted
#    log-amplitude
# 3. Verifies the conditioning-layout guard and the frozen output record
#
# Design decisions:
# - The network is built at the full redmist328 topology because construction
#   plus a sixteen-frame forward stays a fraction of a second on CPU, so no
#   reduced surrogate is needed and the tested shapes are the real ones
# - The polar identity is asserted with a tolerance rather than exact
#   equality, because it is recomputed through transcendental functions in
#   float32
# - Amplitude values themselves are only bounded for finiteness: an
#   untrained network's log-amplitudes carry no meaningful scale
#
# Author: Rahul Sawhney

import math
import unittest

import torch
from pydantic import ValidationError

from vocode.models.apnet2.network import Apnet2GeneratorOutput, Apnet2Network


class ConditioningMelBuilder:
    # Produces deterministic conditioning mels on the grid the APNet2 network consumes.
    # The values are standard normal rather than a real mel, because every
    # assertion here concerns shape, dtype, and algebraic identity, none of
    # which depend on the conditioning being acoustically plausible.
    def __init__(self, band_count: int) -> None:
        # Binds the mel band count this builder emits.
        #
        # Args:
        #     band_count: Band count of every emitted mel; it must match the
        #         network's input width or the entry projections reject it.
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
    # Builds the APNet2 generator at the redmist328 topology under a fixed seed.
    # Constructing the published topology rather than a reduced surrogate is
    # what makes the asserted shapes the real ones, and it is affordable
    # because nothing here trains.
    def __init__(self, n_fft: int, hop_size: int) -> None:
        # Binds the STFT grid shared by the amplitude and phase streams.
        #
        # Args:
        #     n_fft: Transform size; it fixes the bin count both heads emit.
        #     hop_size: Hop the inverse STFT expands conditioning frames by.
        self._n_fft: int = n_fft
        self._hop_size: int = hop_size

    def build(self, seed: int) -> Apnet2Network:
        # Seeds the generator and returns the constructed network in evaluation mode.
        # The seed fixes the truncated-normal initialization, so a failure is
        # reproducible from the seed alone. Evaluation mode records that the
        # assertions describe inference; this network holds no module whose
        # behavior differs between the two modes, so the call is a statement of
        # intent rather than a behavioral switch.
        #
        # Args:
        #     seed: Global generator seed applied before construction.
        #
        # Returns:
        #     The constructed network at the full redmist328 topology.
        torch.manual_seed(seed)
        network: Apnet2Network = Apnet2Network(
            num_mels=80,
            n_fft=self._n_fft,
            hop_size=self._hop_size,
            win_size=1024,
            asp_channel=512,
            psp_channel=512,
            asp_input_conv_kernel_size=7,
            asp_output_conv_kernel_size=7,
            psp_input_conv_kernel_size=7,
            psp_output_r_conv_kernel_size=7,
            psp_output_i_conv_kernel_size=7,
            convnext_layer_count=8,
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


class Apnet2NetworkSpectralPredictionTest(unittest.TestCase):
    # Verifies the shapes, dtype, and length arithmetic of the parallel amplitude and phase streams.
    def setUp(self) -> None:
        # Builds the reference network in evaluation mode.
        self._network_builder: RecipeNetworkBuilder = RecipeNetworkBuilder(n_fft=1024, hop_size=256)
        self._mel_builder: ConditioningMelBuilder = ConditioningMelBuilder(band_count=80)
        self._network: Apnet2Network = self._network_builder.build(seed=20260812)

    def test_both_streams_emit_one_value_per_bin_and_frame(self) -> None:
        # Amplitude and phase are predicted on the full one-sided STFT grid.
        mel: torch.Tensor = self._mel_builder.build(seed=71, frame_count=16)
        with torch.no_grad():
            output: Apnet2GeneratorOutput = self._network.predict_components(mel)
        expected_shape: tuple[int, int, int] = (1, self._network_builder.bin_count, 16)
        self.assertEqual(tuple(output.log_amplitude.shape), expected_shape)
        self.assertEqual(tuple(output.phase.shape), expected_shape)
        self.assertEqual(tuple(output.real_spectrum.shape), expected_shape)
        self.assertEqual(tuple(output.imaginary_spectrum.shape), expected_shape)

    def test_waveform_length_follows_the_centered_istft_arithmetic(self) -> None:
        # The combination expands frames to (frames - 1) * hop samples on one channel.
        # Centered framing pads the signal by half a window at each end, and
        # the inverse transform removes that padding, so the result is one hop
        # short of the naive frames-times-hop figure. Pinning the exact
        # arithmetic matters because the module truncates the reference to this
        # length before every loss and metric comparison.
        mel: torch.Tensor = self._mel_builder.build(seed=72, frame_count=16)
        with torch.no_grad():
            output: Apnet2GeneratorOutput = self._network.predict_components(mel)
        expected_samples: int = 15 * self._network_builder.hop_size
        self.assertEqual(
            tuple(output.waveform.shape),
            (1, 1, expected_samples),
            msg=f"Expected one channel of {expected_samples} samples, got {tuple(output.waveform.shape)}"
        )

    def test_forward_returns_the_component_waveform(self) -> None:
        # The synthesis entry point is the waveform member of the component bundle.
        mel: torch.Tensor = self._mel_builder.build(seed=73, frame_count=16)
        with torch.no_grad():
            waveform: torch.Tensor = self._network(mel)
            output: Apnet2GeneratorOutput = self._network.predict_components(mel)
        self.assertTrue(torch.equal(waveform, output.waveform))

    def test_components_are_finite_float32(self) -> None:
        # The complex reconstruction runs in float32 and must not emit non-finite values.
        mel: torch.Tensor = self._mel_builder.build(seed=74, frame_count=16)
        with torch.no_grad():
            output: Apnet2GeneratorOutput = self._network.predict_components(mel)
        self.assertEqual(output.log_amplitude.dtype, torch.float32)
        self.assertEqual(output.phase.dtype, torch.float32)
        self.assertEqual(output.waveform.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(output.log_amplitude).all()))
        self.assertTrue(bool(torch.isfinite(output.phase).all()))
        self.assertTrue(bool(torch.isfinite(output.waveform).all()))

    def test_frame_count_scales_the_synthesis_length(self) -> None:
        # Both streams operate at frame rate, so length follows the conditioning mel.
        # Two lengths are synthesized from one network to show the relation is
        # the frame count's alone: seven and twenty-three hops are the same
        # (frames - 1) rule applied to eight and twenty-four frames, which
        # would not hold if any stage held state across calls.
        short_mel: torch.Tensor = self._mel_builder.build(seed=75, frame_count=8)
        long_mel: torch.Tensor = self._mel_builder.build(seed=76, frame_count=24)
        with torch.no_grad():
            short_waveform: torch.Tensor = self._network(short_mel)
            long_waveform: torch.Tensor = self._network(long_mel)
        self.assertEqual(tuple(short_waveform.shape), (1, 1, 7 * self._network_builder.hop_size))
        self.assertEqual(tuple(long_waveform.shape), (1, 1, 23 * self._network_builder.hop_size))


class Apnet2NetworkSpectralConsistencyTest(unittest.TestCase):
    # Verifies the phase wrapping and polar-form identities the loss composition relies on.
    def setUp(self) -> None:
        # Builds the reference network whose spectral identities are under test.
        self._network_builder: RecipeNetworkBuilder = RecipeNetworkBuilder(n_fft=1024, hop_size=256)
        self._mel_builder: ConditioningMelBuilder = ConditioningMelBuilder(band_count=80)
        self._network: Apnet2Network = self._network_builder.build(seed=20260813)

    def test_phase_lies_inside_the_arctangent_codomain(self) -> None:
        # Predicting real and imaginary parts yields wrapped phase without unwrapping.
        # The bound is the two-argument arctangent's codomain, so it holds for
        # any projection pair whatsoever, trained or not. The anti-wrapping
        # phase losses are defined only on wrapped phase, which is why this is
        # asserted as a structural property rather than a learned one.
        mel: torch.Tensor = self._mel_builder.build(seed=77, frame_count=16)
        with torch.no_grad():
            output: Apnet2GeneratorOutput = self._network.predict_components(mel)
        self.assertGreaterEqual(float(output.phase.min()), -math.pi)
        self.assertLessEqual(float(output.phase.max()), math.pi)

    def test_spectra_are_the_polar_form_of_the_predicted_amplitude(self) -> None:
        # The squared magnitude of the emitted spectrum must equal the exponentiated log-amplitude.
        # This is the identity that lets the amplitude loss supervise
        # log_amplitude while the consistency loss supervises the recombined
        # spectra: the two describe one quantity. The tolerance accommodates
        # recomputing exp, cos, and sin in float32, whose rounding makes exact
        # equality unattainable while the identity still holds.
        mel: torch.Tensor = self._mel_builder.build(seed=78, frame_count=16)
        with torch.no_grad():
            output: Apnet2GeneratorOutput = self._network.predict_components(mel)
        squared_magnitude: torch.Tensor = output.real_spectrum.pow(2) + output.imaginary_spectrum.pow(2)
        expected_magnitude: torch.Tensor = torch.exp(output.log_amplitude).pow(2)
        self.assertTrue(
            torch.allclose(squared_magnitude, expected_magnitude, atol=1e-4),
            msg="Emitted spectra are not the polar form of the predicted log-amplitude"
        )


class Apnet2NetworkValidationTest(unittest.TestCase):
    # Verifies the conditioning-layout guard and the immutability of the component bundle.
    def setUp(self) -> None:
        # Builds the reference network whose input guards are under test.
        self._network_builder: RecipeNetworkBuilder = RecipeNetworkBuilder(n_fft=1024, hop_size=256)
        self._mel_builder: ConditioningMelBuilder = ConditioningMelBuilder(band_count=80)
        self._network: Apnet2Network = self._network_builder.build(seed=20260814)

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
        mel: torch.Tensor = self._mel_builder.build(seed=79, frame_count=8)
        with torch.no_grad():
            output: Apnet2GeneratorOutput = self._network.predict_components(mel)
        with self.assertRaises(ValidationError):
            output.phase = torch.zeros(1)
