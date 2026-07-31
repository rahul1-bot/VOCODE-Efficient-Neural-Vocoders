# This module:
# 1. Verifies the RFWave network configuration record: the reference
#    backbone and band settings, the band-to-channel consistency the flow
#    depends on, immutability, and the rejection of unknown fields
# 2. Verifies the spectral transform pair: the stacked real and imaginary
#    layout, the inverse-scaling round trip, and dtype preservation
# 3. Verifies the pseudo-QMF equalizer: near-perfect analysis-synthesis
#    reconstruction and the training-only running statistics
# 4. Verifies the backbone: the velocity output shape, the time
#    conditioning, the neutral band conditioning at initialization, and
#    determinism
#
# Design decisions:
# - Reconstruction is asserted over the interior of the signal, because both
#   the centred transform and the filterbank have edge transients by
#   construction; the interior bound is the near-perfect-reconstruction
#   contract these components exist to provide
# - The equalizer is exercised at the reference tap count, since the
#   reconstruction quality is a property of that prototype filter design
# - The backbone runs at reduced width and depth; only shapes and
#   conditioning behavior are under test, both of which are width-invariant
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError

from vocode.models.rfwave.network import (
    RfwaveBackbone,
    RfwaveNetworkConfig,
    RfwavePqmfEqualizer,
    RfwaveSpectralTransform,
)


class ReducedBackboneRecipe:
    # Builds the reduced RFWave backbone configuration used by the shape and conditioning assertions.
    def build(self) -> RfwaveNetworkConfig:
        # Returns the reduced width and depth; the assertions below are width-invariant.
        return RfwaveNetworkConfig(
            hidden_dimension=32,
            intermediate_dimension=64,
            layer_count=2,
            band_count=4,
            n_fft=64,
            hop_length=16,
            output_channels=24,
            left_overlap=2,
            right_overlap=2,
            pqmf_taps=16
        )


class RfwaveNetworkConfigurationTest(unittest.TestCase):
    # Verifies the reference backbone and band settings and the validation behavior of the record.
    def setUp(self) -> None:
        # Reads the default network record, which is the reference geometry.
        self._configuration: RfwaveNetworkConfig = RfwaveNetworkConfig()

    def test_backbone_defaults_follow_the_reference_configuration(self) -> None:
        # The reference backbone is eight ConvNeXt blocks at width 512 with intermediate width 1536.
        self.assertEqual(self._configuration.hidden_dimension, 512)
        self.assertEqual(self._configuration.intermediate_dimension, 1536)
        self.assertEqual(self._configuration.layer_count, 8)
        self.assertEqual(self._configuration.mel_channels, 100)

    def test_band_and_transform_defaults_follow_the_reference_configuration(self) -> None:
        # Eight subbands over the 1024-point transform at hop 256 define the reference geometry.
        self.assertEqual(self._configuration.band_count, 8)
        self.assertEqual(self._configuration.n_fft, 1024)
        self.assertEqual(self._configuration.hop_length, 256)
        self.assertEqual(self._configuration.left_overlap, 8)
        self.assertEqual(self._configuration.right_overlap, 8)

    def test_output_channels_match_the_overlapping_band_width(self) -> None:
        # Each band carries its own bins plus both overlaps, in real and
        # imaginary parts. This is the one cross-field constraint in the
        # record and the only assertion here that is not direct transcription:
        # the head width is not an independent choice but a consequence of the
        # transform size, the band count, and the overlaps, so changing any of
        # those without changing it produces a network whose output cannot be
        # placed back into a spectrum. The relation is recomputed from the
        # other fields rather than compared against a literal, so it states
        # the constraint rather than a value.
        bins_per_band: int = self._configuration.n_fft // 2 // self._configuration.band_count
        overlap: int = self._configuration.left_overlap + self._configuration.right_overlap
        self.assertEqual(
            self._configuration.output_channels,
            2 * (bins_per_band + overlap),
            msg="The velocity head width must match the joint subband slab width"
        )

    def test_conditioning_defaults_follow_the_reference_configuration(self) -> None:
        # Fourier features and the time-embedding scale follow the reference settings.
        self.assertEqual(self._configuration.fourier_start_exponent, 6)
        self.assertEqual(self._configuration.fourier_stop_exponent, 8)
        self.assertEqual(self._configuration.time_embedding_scale, 1000.0)

    def test_equalizer_defaults_follow_the_reference_configuration(self) -> None:
        # The pseudo-QMF prototype is the reference 124-tap Kaiser design.
        self.assertEqual(self._configuration.pqmf_taps, 124)
        self.assertEqual(self._configuration.pqmf_cutoff_ratio, 0.071)
        self.assertEqual(self._configuration.pqmf_kaiser_beta, 9.0)

    def test_configuration_is_immutable(self) -> None:
        # Frozen settings cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.band_count = 4

    def test_unknown_field_is_rejected(self) -> None:
        # Extra fields are forbidden so silent typos cannot enter an experiment record.
        with self.assertRaises(ValidationError):
            RfwaveNetworkConfig(unknown_setting=1)


class RfwaveSpectralTransformTest(unittest.TestCase):
    # Verifies the stacked spectral layout and the scaled forward and inverse transform pair.
    def setUp(self) -> None:
        # Builds the 64-point transform and a waveform long enough to have a reconstructable interior.
        torch.manual_seed(1234)
        self._transform: RfwaveSpectralTransform = RfwaveSpectralTransform(n_fft=64, hop_length=16)
        self._waveform: torch.Tensor = torch.randn(2, 512)

    def test_spectrum_stacks_real_and_imaginary_channels(self) -> None:
        # The complex spectrum is carried as concatenated real and imaginary channels.
        with torch.no_grad():
            spectrum: torch.Tensor = self._transform.stft(self._waveform)
        self.assertEqual(spectrum.shape[0], 2)
        self.assertEqual(spectrum.shape[1], 2 * (64 // 2 + 1))

    def test_centred_analysis_produces_one_frame_per_hop(self) -> None:
        # The centred transform yields one frame per hop plus the closing frame.
        with torch.no_grad():
            spectrum: torch.Tensor = self._transform.stft(self._waveform)
        self.assertEqual(spectrum.shape[2], 512 // 16 + 1)

    def test_inverse_transform_restores_the_waveform_interior(self) -> None:
        # The inverse scaling undoes the forward scaling, so the interior reconstructs exactly.
        with torch.no_grad():
            spectrum: torch.Tensor = self._transform.stft(self._waveform)
            reconstructed: torch.Tensor = self._transform.istft(spectrum)
        self.assertEqual(tuple(reconstructed.shape), tuple(self._waveform.shape))
        self.assertTrue(
            bool(torch.allclose(reconstructed[:, 64:-64], self._waveform[:, 64:-64], atol=1e-4)),
            msg="The scaled transform pair must reconstruct the signal interior"
        )

    def test_transform_preserves_the_input_dtype(self) -> None:
        # Analysis runs in float32 internally but returns the caller's dtype.
        with torch.no_grad():
            spectrum: torch.Tensor = self._transform.stft(self._waveform)
        self.assertEqual(spectrum.dtype, self._waveform.dtype)


class RfwavePqmfEqualizerTest(unittest.TestCase):
    # Verifies the filterbank reconstruction and the training-only running
    # subband statistics. The statistics assertions matter beyond bookkeeping:
    # they are buffers rather than parameters, so they are checkpointed with
    # the model and the equalization applied at measurement time depends on
    # them. An evaluation pass that moved them would make a measurement run
    # alter the model it was measuring.
    def setUp(self) -> None:
        # Builds the equalizer at the reference 124-tap prototype, in
        # evaluation mode. The tap count is deliberately not reduced, because
        # reconstruction quality is a property of that specific prototype
        # design and a shorter filter would separate the bands less sharply.
        torch.manual_seed(1234)
        self._equalizer: RfwavePqmfEqualizer = RfwavePqmfEqualizer(
            band_count=8,
            taps=124,
            cutoff_ratio=0.071,
            kaiser_beta=9.0
        )
        self._equalizer.eval()
        self._waveform: torch.Tensor = torch.randn(1, 2048)

    def test_projection_preserves_the_waveform_layout(self) -> None:
        # Analysis and synthesis run at full rate, so the sample count is unchanged.
        with torch.no_grad():
            projected: torch.Tensor = self._equalizer.project(self._waveform)
        self.assertEqual(tuple(projected.shape), tuple(self._waveform.shape))

    def test_restoration_inverts_the_projection_in_the_interior(self) -> None:
        # The cosine-modulated filterbank is a near-perfect-reconstruction
        # pair, so the round trip is asserted with a tolerance rather than
        # exactly. Two separate effects justify that. The reconstruction is
        # only near-perfect by design, since the filter family cancels
        # aliasing between adjacent bands rather than eliminating it, and the
        # comparison is restricted to the interior because both convolutions
        # have edge transients where no filter has yet seen a full window.
        # Excluding the edges is therefore stating the contract, not evading
        # a failure.
        with torch.no_grad():
            projected: torch.Tensor = self._equalizer.project(self._waveform)
            restored: torch.Tensor = self._equalizer.restore(projected)
        interior_error: float = float((restored[:, 200:-200] - self._waveform[:, 200:-200]).abs().max())
        self.assertLess(
            interior_error,
            0.05,
            msg="Equalize then restore must return the interior of the waveform"
        )

    def test_statistics_start_neutral(self) -> None:
        # Before any observation the equalizer normalizes by zero mean and
        # unit variance, which makes equalization the identity. A freshly
        # constructed equalizer therefore passes audio through unchanged
        # rather than distorting it by whatever an arbitrary initialization
        # happened to be, and any deviation from identity is something
        # training observed.
        self.assertTrue(bool(torch.equal(self._equalizer.get_buffer("_mean_statistics"), torch.zeros(8))))
        self.assertTrue(bool(torch.equal(self._equalizer.get_buffer("_variance_statistics"), torch.ones(8))))
        self.assertEqual(float(self._equalizer.get_buffer("_observed_batches")), 0.0)

    def test_evaluation_mode_leaves_statistics_untouched(self) -> None:
        # Measurement runs must not move the running statistics of a trained
        # equalizer. This is what makes evaluation repeatable and what allows
        # the flow tests to assert that the data endpoint is deterministic: in
        # training mode the statistics shift on every call, so the same
        # waveform would equalize differently the second time.
        with torch.no_grad():
            self._equalizer.project(self._waveform)
        self.assertEqual(float(self._equalizer.get_buffer("_observed_batches")), 0.0)
        self.assertTrue(bool(torch.equal(self._equalizer.get_buffer("_mean_statistics"), torch.zeros(8))))

    def test_training_mode_accumulates_statistics(self) -> None:
        # Training observations move the exponential statistics and count the batch.
        self._equalizer.train()
        initial_mean: torch.Tensor = self._equalizer.get_buffer("_mean_statistics").clone()
        with torch.no_grad():
            self._equalizer.project(self._waveform)
        self.assertEqual(float(self._equalizer.get_buffer("_observed_batches")), 1.0)
        self.assertFalse(bool(torch.equal(self._equalizer.get_buffer("_mean_statistics"), initial_mean)))


class RfwaveBackboneTest(unittest.TestCase):
    # Verifies the velocity head shape and the time and band conditioning of
    # the backbone. The two conditioning assertions are deliberately opposite:
    # the flow time must change the prediction, because a field that ignored
    # it could not describe a path, while the band index must not change it at
    # initialization, because band conditioning is entirely learned. Together
    # they establish that each conditioning route is wired to the behavior it
    # is supposed to control.
    def setUp(self) -> None:
        # Builds the reduced backbone and one four-band conditioning set at a
        # fixed flow time. The four rows carry four distinct band indices, so
        # the band-conditioning assertion can vary that one input while
        # holding everything else fixed.
        torch.manual_seed(1234)
        self._configuration: RfwaveNetworkConfig = ReducedBackboneRecipe().build()
        self._backbone: RfwaveBackbone = RfwaveBackbone(self._configuration)
        self._noisy_state: torch.Tensor = torch.randn(4, 24, 5)
        self._mel: torch.Tensor = torch.randn(4, 100, 5)
        self._band_index: torch.Tensor = torch.arange(4)
        self._time_values: torch.Tensor = torch.full((4,), 0.3)

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The backbone exposes exactly the record it was constructed with.
        self.assertIs(self._backbone.configuration, self._configuration)

    def test_velocity_has_the_band_slab_shape(self) -> None:
        # The head predicts one velocity per slab channel and frame of every band.
        with torch.no_grad():
            velocity: torch.Tensor = self._backbone(
                self._noisy_state,
                self._time_values,
                self._mel,
                self._band_index
            )
        self.assertEqual(tuple(velocity.shape), (4, 24, 5))

    def test_velocity_is_finite(self) -> None:
        # The Fourier features and the block stack produce no non-finite velocities.
        with torch.no_grad():
            velocity: torch.Tensor = self._backbone(
                self._noisy_state,
                self._time_values,
                self._mel,
                self._band_index
            )
        self.assertTrue(bool(torch.isfinite(velocity).all()))

    def test_prediction_is_deterministic(self) -> None:
        # The backbone carries no sampling and no normalization with running
        # state, so identical inputs give identical velocities without any
        # seed being reset. That determinism is what confines this family's
        # stochasticity to the noise endpoint alone, which is why a seeded
        # synthesis reproduces exactly.
        with torch.no_grad():
            first: torch.Tensor = self._backbone(
                self._noisy_state,
                self._time_values,
                self._mel,
                self._band_index
            )
            second: torch.Tensor = self._backbone(
                self._noisy_state,
                self._time_values,
                self._mel,
                self._band_index
            )
        self.assertTrue(bool(torch.equal(first, second)))

    def test_flow_time_changes_the_predicted_velocity(self) -> None:
        # The time embedding enters every block, so the velocity field varies along the path.
        with torch.no_grad():
            early: torch.Tensor = self._backbone(
                self._noisy_state,
                torch.full((4,), 0.1),
                self._mel,
                self._band_index
            )
            late: torch.Tensor = self._backbone(
                self._noisy_state,
                torch.full((4,), 0.9),
                self._mel,
                self._band_index
            )
        self.assertFalse(bool(torch.equal(early, late)))

    def test_band_conditioning_starts_neutral(self) -> None:
        # Adaptive normalization initializes to unit scale and zero shift, so
        # every band's lookup returns the same values and the backbone is
        # initially band-agnostic. Collapsing all four rows onto one band index
        # and observing no change is what proves the neutrality: any
        # difference would mean the band route carries an initialization
        # artifact rather than learned specialization. The assertion would
        # legitimately fail on a trained model, which is exactly the point.
        with torch.no_grad():
            distinct_bands: torch.Tensor = self._backbone(
                self._noisy_state,
                self._time_values,
                self._mel,
                self._band_index
            )
            single_band: torch.Tensor = self._backbone(
                self._noisy_state,
                self._time_values,
                self._mel,
                torch.zeros(4, dtype=torch.long)
            )
        self.assertTrue(
            bool(torch.equal(distinct_bands, single_band)),
            msg="Band conditioning is learned, so it must be neutral at initialization"
        )

    def test_convolution_biases_start_at_zero(self) -> None:
        # The reference initialization pairs truncated-normal weights with zero biases.
        state: dict[str, torch.Tensor] = self._backbone.state_dict()
        self.assertTrue(bool(torch.equal(state["_embedding.bias"], torch.zeros(32))))
        self.assertTrue(bool(torch.equal(state["_output_projection.bias"], torch.zeros(24))))
