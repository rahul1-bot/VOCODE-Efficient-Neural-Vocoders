# This module:
# 1. Verifies the RNDVoC normalization and mixing blocks: the channel and
#    band-wise normalizations, the identity behavior of the global response
#    normalization at initialization, the grouped linear mixer, and the
#    shape contracts of the band and time modules
# 2. Verifies the shared band split and merge: the twenty-four band encoding
#    of the 513-bin spectrum and the strictly positive magnitude and bounded
#    phase the decoder emits
# 3. Verifies the range-null decomposition itself: the pseudo-inverse
#    projector is idempotent and the null-space filter is invisible to the
#    mel projection, which is the architecture's defining property
# 4. Verifies the synthesis bundle: component shapes, the inverse-transform
#    sample count, and the internal consistency of amplitude and phase
#
# Design decisions:
# - The projector identities are asserted on the registered analysis buffers,
#   because the decomposition is exact linear algebra there rather than a
#   learned approximation; tolerances follow float32 conditioning of a
#   513-by-80 pseudo-inverse
# - The generator runs at one refinement stage with narrow feature widths;
#   the published stage count and widths are configuration values asserted
#   in the module tests, and nothing here depends on them
# - The transform geometry stays at the published 1024-point analysis
#   because the band split encodes that bin count in its region kernels
#
# Author: Rahul Sawhney

import unittest

import torch

from vocode.models.rndvoc.network import (
    RndvocBandShuffler,
    RndvocBandwiseC2LayerNorm,
    RndvocBandwiseLayerNorm,
    RndvocBandWiseTimeModule,
    RndvocChannelNormalization,
    RndvocGeneratorOutput,
    RndvocGlobalResponseNormalization,
    RndvocGroupedLinear,
    RndvocNetwork,
    RndvocSharedBandMerge,
    RndvocSharedBandSplit,
    RndvocVocModule,
)


class MiniatureNetworkRecipe:
    # Builds the miniature RNDVoC generator used by the decomposition and
    # synthesis assertions. The reduction is confined to what the network
    # learns: the stage count, the repeat count, and the feature widths. The
    # analysis geometry is untouched, because both the decomposition and the
    # band split depend on it directly and reducing it would change what these
    # assertions are about rather than merely how long they take.
    def build(self) -> RndvocNetwork:
        # Returns the generator at one refinement stage, keeping the published
        # 1024-point geometry. That geometry is load-bearing in two separate
        # ways: it fixes the 513-bin spectrum whose partition the band split
        # hard-codes in its region kernels, and it fixes the shape of the
        # pseudo-inverse the decomposition identities are stated over.
        return RndvocNetwork(
            sample_rate=22050,
            num_mels=80,
            n_fft=1024,
            hop_size=256,
            win_size=1024,
            fmin=0.0,
            fmax=8000.0,
            null_stage_count=1,
            repeat_count=1,
            input_dimension=8,
            squeeze_dimension=4,
            hidden_dimension=8,
            kernel_size=3
        )


class RndvocChannelNormalizationTest(unittest.TestCase):
    # Verifies the per-position channel normalization and its identity-preserving initialization.
    def setUp(self) -> None:
        # Builds the five-channel normalization and the feature block it rescales.
        torch.manual_seed(1234)
        self._normalization: RndvocChannelNormalization = RndvocChannelNormalization(channel_count=5)
        self._features: torch.Tensor = torch.randn(2, 5, 7)

    def test_shape_is_preserved(self) -> None:
        # Normalization rescales without changing the feature layout.
        with torch.no_grad():
            normalized: torch.Tensor = self._normalization(self._features)
        self.assertEqual(tuple(normalized.shape), (2, 5, 7))

    def test_channels_are_centered(self) -> None:
        # Each position is centred across the channel axis.
        with torch.no_grad():
            normalized: torch.Tensor = self._normalization(self._features)
        self.assertLess(float(normalized.mean(dim=1).abs().max()), 1e-5)

    def test_channels_are_scaled_to_unit_deviation(self) -> None:
        # Each position is scaled to unit deviation across the channel axis.
        with torch.no_grad():
            normalized: torch.Tensor = self._normalization(self._features)
        deviations: torch.Tensor = normalized.std(dim=1, unbiased=False)
        self.assertTrue(bool(torch.allclose(deviations, torch.ones_like(deviations), atol=1e-3)))

    def test_gain_and_bias_start_neutral(self) -> None:
        # The affine terms begin at identity so initialization does not shift the signal.
        self.assertTrue(bool(torch.equal(self._normalization.gain, torch.ones(1, 5, 1))))
        self.assertTrue(bool(torch.equal(self._normalization.bias, torch.zeros(1, 5, 1))))


class RndvocBandwiseNormalizationTest(unittest.TestCase):
    # Verifies the band-wise normalizations used inside the band and time modules.
    def setUp(self) -> None:
        # Builds both band-wise normalizations: the flattened one and the complex encoder one.
        torch.manual_seed(1234)
        self._band_normalization: RndvocBandwiseLayerNorm = RndvocBandwiseLayerNorm(
            band_count=2,
            feature_dimension=4
        )
        self._complex_normalization: RndvocBandwiseC2LayerNorm = RndvocBandwiseC2LayerNorm(
            band_count=3,
            feature_dimension=4
        )

    def test_bandwise_normalization_preserves_the_flattened_layout(self) -> None:
        # The flattened batch-and-band layout returns unchanged in shape.
        features: torch.Tensor = torch.randn(2, 4, 5)
        with torch.no_grad():
            normalized: torch.Tensor = self._band_normalization(features)
        self.assertEqual(tuple(normalized.shape), (2, 4, 5))

    def test_bandwise_normalization_centers_the_channel_axis(self) -> None:
        # Statistics are taken per band and frame across the channel axis.
        features: torch.Tensor = torch.randn(2, 4, 5)
        with torch.no_grad():
            normalized: torch.Tensor = self._band_normalization(features)
        self.assertLess(float(normalized.mean(dim=1).abs().max()), 1e-5)

    def test_complex_normalization_preserves_the_band_layout(self) -> None:
        # The encoder normalization keeps the feature, band, and frame axes intact.
        features: torch.Tensor = torch.randn(2, 4, 3, 6)
        with torch.no_grad():
            normalized: torch.Tensor = self._complex_normalization(features)
        self.assertEqual(tuple(normalized.shape), (2, 4, 3, 6))

    def test_complex_normalization_centers_the_feature_axis(self) -> None:
        # Statistics are taken across the feature axis of every band and frame.
        features: torch.Tensor = torch.randn(2, 4, 3, 6)
        with torch.no_grad():
            normalized: torch.Tensor = self._complex_normalization(features)
        self.assertLess(float(normalized.mean(dim=1).abs().max()), 1e-5)


class RndvocGlobalResponseNormalizationTest(unittest.TestCase):
    # Verifies that the global response normalization starts as an exact
    # identity. Exactness matters rather than mere closeness: the block sits
    # inside every temporal residual block of every stage, so an initialization
    # that perturbed its input even slightly would compound through the depth
    # of the decoder before training had adjusted anything.
    def setUp(self) -> None:
        # Builds the response normalization and the feature block it must pass through untouched.
        torch.manual_seed(1234)
        self._normalization: RndvocGlobalResponseNormalization = RndvocGlobalResponseNormalization(channel_count=5)
        self._features: torch.Tensor = torch.randn(2, 5, 7)

    def test_initialization_is_the_identity(self) -> None:
        # Both response weights start at zero, so the residual path passes the input through.
        with torch.no_grad():
            normalized: torch.Tensor = self._normalization(self._features)
        self.assertTrue(bool(torch.equal(normalized, self._features)))

    def test_response_weights_start_at_zero(self) -> None:
        # The reference initialization keeps the block neutral until it learns a response.
        self.assertTrue(bool(torch.equal(self._normalization.gamma, torch.zeros(1, 5, 1))))
        self.assertTrue(bool(torch.equal(self._normalization.beta, torch.zeros(1, 5, 1))))


class RndvocGroupedLinearTest(unittest.TestCase):
    # Verifies that the grouped linear mixer applies one independent map per group.
    def setUp(self) -> None:
        # Builds a four-group mixer and the features whose per-group product is recomputed below.
        torch.manual_seed(1234)
        self._mixer: RndvocGroupedLinear = RndvocGroupedLinear(in_features=3, out_features=2, group_count=4)
        self._features: torch.Tensor = torch.randn(6, 4, 3)

    def test_output_width_is_the_configured_out_features(self) -> None:
        # Each group maps its own input width onto the configured output width.
        with torch.no_grad():
            mixed: torch.Tensor = self._mixer(self._features)
        self.assertEqual(tuple(mixed.shape), (6, 4, 2))

    def test_each_group_applies_its_own_affine_map(self) -> None:
        # The contraction equals the per-group matrix product plus that
        # group's bias, recomputed here as an explicit loop. Restating the
        # operation in a second, independently verifiable form is what makes
        # this assertion meaningful: index notation is readily mis-specified
        # in a way that still produces plausible shapes, and the loop would
        # not match if the contraction mixed groups.
        with torch.no_grad():
            mixed: torch.Tensor = self._mixer(self._features)
            expected: torch.Tensor = torch.stack(
                [
                    self._features[:, group] @ self._mixer.weight[group].transpose(0, 1) + self._mixer.bias[group]
                    for group in range(4)
                ],
                dim=1
            )
        self.assertTrue(bool(torch.allclose(mixed, expected, atol=1e-6)))


class RndvocMixingModuleTest(unittest.TestCase):
    # Verifies the shape contracts of the band shuffler, the time module, and their composition.
    def setUp(self) -> None:
        # Builds the band shuffler, the time module, and the stage that composes both.
        torch.manual_seed(1234)
        self._shuffler: RndvocBandShuffler = RndvocBandShuffler(
            band_count=24,
            input_dimension=8,
            squeeze_dimension=4
        )
        self._time_module: RndvocBandWiseTimeModule = RndvocBandWiseTimeModule(
            band_count=24,
            repeat_count=1,
            input_dimension=8,
            hidden_dimension=8,
            kernel_size=3
        )
        self._voc_module: RndvocVocModule = RndvocVocModule(
            band_count=24,
            repeat_count=1,
            input_dimension=8,
            squeeze_dimension=4,
            hidden_dimension=8,
            kernel_size=3
        )

    def test_band_shuffler_preserves_the_frame_major_layout(self) -> None:
        # The shuffler mixes across bands without changing the tensor layout.
        features: torch.Tensor = torch.randn(1, 3, 8, 24)
        with torch.no_grad():
            mixed: torch.Tensor = self._shuffler(features)
        self.assertEqual(tuple(mixed.shape), (1, 3, 8, 24))

    def test_time_module_preserves_the_band_major_layout(self) -> None:
        # The time module refines along frames without changing the tensor layout.
        features: torch.Tensor = torch.randn(1, 24, 8, 5)
        with torch.no_grad():
            refined: torch.Tensor = self._time_module(features)
        self.assertEqual(tuple(refined.shape), (1, 24, 8, 5))

    def test_voc_module_returns_the_frame_major_layout_it_received(self) -> None:
        # The stage transposes into the time module and back, so the caller sees one layout.
        features: torch.Tensor = torch.randn(1, 5, 8, 24)
        with torch.no_grad():
            refined: torch.Tensor = self._voc_module(features)
        self.assertEqual(tuple(refined.shape), (1, 5, 8, 24))

    def test_voc_module_output_is_finite(self) -> None:
        # The composed stage introduces no non-finite values at initialization.
        features: torch.Tensor = torch.randn(1, 5, 8, 24)
        with torch.no_grad():
            refined: torch.Tensor = self._voc_module(features)
        self.assertTrue(bool(torch.isfinite(refined).all()))


class RndvocBandSplitAndMergeTest(unittest.TestCase):
    # Verifies the twenty-four band encoding of the spectrum and the decoded
    # magnitude and phase. The pair must invert each other's partition
    # exactly, and the round-trip assertion is what establishes that: the
    # encoder drops the highest bin so its regions divide evenly, and the
    # decoder restores it by duplication, so a change to either side alone
    # would produce a spectrum of the wrong width for the inverse transform.
    def setUp(self) -> None:
        # Builds the shared band encoder and decoder at feature width four.
        torch.manual_seed(1234)
        self._split: RndvocSharedBandSplit = RndvocSharedBandSplit(feature_dimension=4)
        self._merge: RndvocSharedBandMerge = RndvocSharedBandMerge(feature_dimension=4)

    def test_band_count_covers_the_three_frequency_regions(self) -> None:
        # Twelve, eight, and four bands encode the low, middle, and high
        # regions. The counts fall as frequency rises because the kernels
        # widen, which is the perceptual motivation made structural: the
        # low region is resolved most finely because pitch and formant
        # structure live there.
        self.assertEqual(self._split.band_count, 24)

    def test_split_encodes_the_spectrum_into_bands(self) -> None:
        # A 513-bin complex spectrum becomes one feature vector per band and frame.
        spectrum: torch.Tensor = torch.randn(2, 513, 6, 2)
        with torch.no_grad():
            encoded: torch.Tensor = self._split(spectrum)
        self.assertEqual(tuple(encoded.shape), (2, 24, 4, 6))

    def test_merge_reconstructs_every_frequency_bin(self) -> None:
        # The decoder restores the full 513-bin resolution the transform expects.
        features: torch.Tensor = torch.randn(2, 24, 4, 6)
        with torch.no_grad():
            magnitude, phase = self._merge(features)
        self.assertEqual(tuple(magnitude.shape), (2, 513, 6))
        self.assertEqual(tuple(phase.shape), (2, 513, 6))

    def test_decoded_magnitude_is_strictly_positive(self) -> None:
        # The magnitude decoder exponentiates, so no decoded magnitude can be negative.
        features: torch.Tensor = torch.randn(2, 24, 4, 6)
        with torch.no_grad():
            magnitude, _ = self._merge(features)
        self.assertTrue(bool((magnitude > 0.0).all()))

    def test_decoded_phase_is_a_wrapped_angle(self) -> None:
        # The phase decoder is a two-argument arctangent, so its range is the wrapped circle.
        features: torch.Tensor = torch.randn(2, 24, 4, 6)
        with torch.no_grad():
            _, phase = self._merge(features)
        self.assertLessEqual(float(phase.abs().max()), 3.14159266)

    def test_split_and_merge_round_trip_through_the_band_layout(self) -> None:
        # The encoder output feeds the decoder directly, which is how the stages are composed.
        spectrum: torch.Tensor = torch.randn(1, 513, 6, 2)
        with torch.no_grad():
            encoded: torch.Tensor = self._split(spectrum)
            magnitude, phase = self._merge(encoded)
        self.assertEqual(tuple(magnitude.shape), (1, 513, 6))
        self.assertTrue(bool(torch.isfinite(phase).all()))


class RndvocRangeNullDecompositionTest(unittest.TestCase):
    # Verifies that the registered analysis buffers implement an exact
    # range-null decomposition of the mel projection. This is the most
    # important class in the file, because the decomposition is the
    # architecture's entire premise: if these identities failed, the network's
    # learned contribution could contradict the analytically recovered
    # component and the design would reduce to an ordinary spectral decoder
    # carrying a useless extra projection.
    #
    # The identities are asserted on the buffers rather than through a forward
    # pass, which makes them exact linear-algebra statements about the
    # construction rather than observations about a particular input. Two
    # properties together establish the decomposition: that the projector is
    # idempotent, so it genuinely projects, and that the mel basis annihilates
    # its complement, so the network's confined contribution is invisible to
    # the conditioning.
    def setUp(self) -> None:
        # Reads the three registered analysis buffers the decomposition
        # identities are stated over. They are read through the buffer
        # accessor rather than as attributes, which is what makes their
        # registration part of the asserted contract.
        torch.manual_seed(1234)
        self._network: RndvocNetwork = MiniatureNetworkRecipe().build()
        self._mel_basis: torch.Tensor = self._network.get_buffer("_mel_basis")
        self._inverse_mel_basis: torch.Tensor = self._network.get_buffer("_inverse_mel_basis")
        self._projection: torch.Tensor = self._network.get_buffer("_projection")

    def test_basis_maps_spectral_bins_onto_mel_bands(self) -> None:
        # The forward basis is the eighty-band filterbank over the 513-bin
        # spectrum, and its pseudo-inverse maps back the other way. The
        # asymmetry between the two dimensions is precisely what creates the
        # null space: the map has at most eighty independent directions to
        # describe a 513-dimensional spectrum, so a subspace of dimension at
        # least four hundred thirty-three is annihilated outright, and that
        # subspace is exactly what the network exists to predict.
        self.assertEqual(tuple(self._mel_basis.shape), (80, 513))
        self.assertEqual(tuple(self._inverse_mel_basis.shape), (513, 80))

    def test_projection_is_the_range_space_projector(self) -> None:
        # The recorded projector is the pseudo-inverse composition over the spectral axis.
        self.assertEqual(tuple(self._projection.shape), (513, 513))
        self.assertTrue(
            bool(torch.allclose(self._projection, self._inverse_mel_basis @ self._mel_basis, atol=1e-6))
        )

    def test_projection_is_idempotent(self) -> None:
        # Projecting twice onto the range space equals projecting once, which
        # is the defining property of a projector and the reason the analytic
        # component is well defined at all. The tolerance is looser than the
        # equality above because it compares a product of two ill-conditioned
        # matrices rather than one; the pseudo-inverse of a strongly
        # rank-deficient filterbank amplifies float32 rounding, so the margin
        # reflects the conditioning of the construction rather than any
        # approximation in the identity itself.
        self.assertTrue(
            bool(torch.allclose(self._projection @ self._projection, self._projection, atol=1e-5)),
            msg="The range-space projector must be idempotent for the decomposition to hold"
        )

    def test_null_space_filter_is_invisible_to_the_mel_projection(self) -> None:
        # Content the network predicts in the null space cannot alter the
        # conditioning mel. This is the architecture's premise stated as an
        # equation: composing the mel basis with the complementary projector
        # must annihilate everything, so whatever the network adds re-analyzes
        # to a mel of zero and cannot disturb the component recovered
        # analytically. Asserting it on the full composed matrix rather than
        # on a sampled vector establishes it for every possible prediction
        # rather than for one draw.
        identity: torch.Tensor = torch.eye(self._projection.shape[0])
        null_filter: torch.Tensor = identity - self._projection
        residual: torch.Tensor = self._mel_basis @ null_filter
        self.assertLess(
            float(residual.abs().max()),
            1e-6,
            msg="The mel basis must annihilate the null-space filter, which is the architecture's premise"
        )


class RndvocSynthesisTest(unittest.TestCase):
    # Verifies the decomposed component bundle, the reconstruction length, and
    # the bundle's internal consistency. The consistency assertions are the
    # substantive ones: the bundle reports one spectrum in two
    # parameterizations, and the objective penalizes both, so a disagreement
    # between them would mean the objective was scoring two different signals
    # while appearing to score one.
    def setUp(self) -> None:
        # Builds the miniature generator and the six-frame conditioning mel it synthesizes from.
        torch.manual_seed(1234)
        self._network: RndvocNetwork = MiniatureNetworkRecipe().build()
        self._mel: torch.Tensor = torch.randn(1, 80, 6)

    def test_alpha_scaling_starts_neutral(self) -> None:
        # The analytic encoding enters the merge unscaled at initialization,
        # so the skip past every refinement stage begins at full strength.
        # This is what lets the decoder see a usable signal before any stage
        # has learned anything, and it is why deepening the stack does not
        # degrade the starting point.
        self.assertTrue(bool(torch.equal(self._network.alpha, torch.ones(1, 1, 8, 24))))

    def test_component_bundle_carries_the_decomposed_spectra(self) -> None:
        # The bundle exposes the log amplitude, phase, and rectangular spectra for the losses.
        with torch.no_grad():
            components: RndvocGeneratorOutput = self._network.predict_components(self._mel)
        self.assertIsInstance(components, RndvocGeneratorOutput)
        self.assertEqual(tuple(components.log_amplitude.shape), (1, 513, 6))
        self.assertEqual(tuple(components.phase.shape), (1, 513, 6))
        self.assertEqual(tuple(components.real_spectrum.shape), (1, 513, 6))
        self.assertEqual(tuple(components.imaginary_spectrum.shape), (1, 513, 6))

    def test_reconstruction_length_follows_the_hop_size(self) -> None:
        # The inverse transform emits hop-size samples for every frame after
        # the first, so six frames at hop two hundred fifty-six reconstruct to
        # one thousand two hundred eighty samples rather than to the full
        # product. The lost frame is the transform's own edge behavior, which
        # is why the training and validation paths truncate the reference
        # rather than assuming the two lengths agree.
        with torch.no_grad():
            components: RndvocGeneratorOutput = self._network.predict_components(self._mel)
        self.assertEqual(tuple(components.waveform.shape), (1, 1280))

    def test_forward_returns_the_bundled_waveform(self) -> None:
        # The forward pass is the waveform member of the component bundle.
        with torch.no_grad():
            waveform: torch.Tensor = self._network(self._mel)
            components: RndvocGeneratorOutput = self._network.predict_components(self._mel)
        self.assertTrue(bool(torch.equal(waveform, components.waveform)))

    def test_log_amplitude_matches_the_rectangular_spectra(self) -> None:
        # The reported amplitude is the magnitude of the reported real and
        # imaginary parts, so the two parameterizations in the bundle describe
        # one spectrum. The tolerance accommodates the small additive floor
        # applied inside the logarithm and the round trip through exponential
        # and logarithm, both of which are exact in intent but not in float32.
        with torch.no_grad():
            components: RndvocGeneratorOutput = self._network.predict_components(self._mel)
            magnitude: torch.Tensor = torch.sqrt(
                components.real_spectrum.pow(2) + components.imaginary_spectrum.pow(2)
            )
        self.assertTrue(bool(torch.allclose(torch.exp(components.log_amplitude), magnitude, rtol=1e-4, atol=1e-3)))

    def test_phase_matches_the_rectangular_spectra(self) -> None:
        # The reported phase reconstructs the real part from the reported magnitude.
        with torch.no_grad():
            components: RndvocGeneratorOutput = self._network.predict_components(self._mel)
            magnitude: torch.Tensor = torch.sqrt(
                components.real_spectrum.pow(2) + components.imaginary_spectrum.pow(2)
            )
            reconstructed_real: torch.Tensor = magnitude * torch.cos(components.phase)
        self.assertTrue(bool(torch.allclose(reconstructed_real, components.real_spectrum, rtol=1e-4, atol=1e-3)))

    def test_synthesis_is_finite(self) -> None:
        # The decomposition and the inverse transform produce no non-finite samples.
        with torch.no_grad():
            waveform: torch.Tensor = self._network(self._mel)
        self.assertTrue(bool(torch.isfinite(waveform).all()))

    def test_batch_dimension_is_preserved(self) -> None:
        # Batched conditioning produces one waveform row per batch element.
        with torch.no_grad():
            waveform: torch.Tensor = self._network(torch.randn(2, 80, 6))
        self.assertEqual(tuple(waveform.shape), (2, 1280))
