# This module:
# 1. Verifies the HiFTNet network configuration record: the reference
#    defaults, immutability, and the rejection of unknown fields
# 2. Verifies the neural source-filter synthesis chain: the frame-to-sample
#    expansion, the spectral component bundle, the strictly positive
#    magnitude and bounded phase, and the input-layout guard
# 3. Verifies the F0-extractor checkpoint boundary: construction without a
#    checkpoint, the silent skip of an absent path, the malformed-payload
#    refusals, and the reference-naming adaptation on a fabricated payload
#
# Design decisions:
# - The pretrained F0 predictor of the reference recipe is never retrieved:
#   the extractor is exercised at its random initialization, and the
#   checkpoint path is tested only with fabricated payloads written into a
#   temporary directory, so no author binary and no network call is involved
# - The fabricated reference payload is produced by renaming a locally
#   initialized extractor state into the author naming, which asserts the
#   adapter against the real parameter inventory rather than a guessed one
# - Synthesis assertions run on a reduced generator; the reference topology
#   is exercised once for its frame-to-sample expansion contract
# - The harmonic source injects noise from the global generator, so
#   reproducibility is asserted under a fixed seed rather than by assuming
#   a deterministic forward pass
#
# Author: Rahul Sawhney

import tempfile
import unittest
from pathlib import Path

import torch
from pydantic import ValidationError

from vocode.models.hiftnet.network import HiftnetGeneratorOutput, HiftnetNetwork, HiftnetNetworkConfig


class ReducedGeneratorRecipe:
    # Builds the reduced HiFTNet network configurations used by the synthesis
    # assertions. The band count is held at eighty because the pitch network's
    # shape chain depends on it, and the inverse-STFT head is left at its
    # reference values because the spectral assertions are derived from them;
    # only the width, the upsampling rates, and the residual kernel set are
    # reduced, none of which the assertions below depend on.
    #
    # The checkpoint path is a constructor parameter rather than a build
    # argument because the checkpoint tests build many configurations that
    # differ only in that path, and binding it here keeps each of those call
    # sites to a single expression.
    def __init__(self, f0_checkpoint_path: Path | None) -> None:
        # Binds the pitch-checkpoint path this recipe stamps on every
        # configuration.
        #
        # Args:
        #     f0_checkpoint_path: Path stamped onto every configuration this
        #         recipe builds. ``None`` produces a network at random pitch
        #         initialization; a path is used by the checkpoint tests to
        #         drive the loader against fabricated payloads.
        self._f0_checkpoint_path: Path | None = f0_checkpoint_path

    def build(self) -> HiftnetNetworkConfig:
        # Returns the reduced network configuration at the bound checkpoint
        # path. Each call constructs a fresh record, so a caller building
        # several networks cannot share mutable state between them.
        return HiftnetNetworkConfig(
            input_mel_channels=80,
            upsample_initial_channel=32,
            upsample_rates=(4, 4),
            upsample_kernel_sizes=(8, 8),
            resblock_kernel_sizes=(3,),
            resblock_dilation_sizes=((1, 3, 5),),
            f0_checkpoint_path=self._f0_checkpoint_path
        )


class UpstreamF0StateBuilder:
    # Renames a locally initialized F0-extractor state into the reference
    # checkpoint naming, inverting the adapter under test.
    #
    # Deriving the fixture from the live network rather than writing key names
    # by hand is what makes the adaptation assertions meaningful. A
    # hand-written fixture would only prove that the adapter handles the keys
    # its author remembered; a fixture built by renaming the real parameter
    # inventory covers every parameter the network actually declares, so a
    # renaming rule that misses one is exposed as a missing key at load time.
    # The inversion must be maintained alongside the adapter: if the two were
    # to drift apart, this fixture would stop describing the author release.
    def __init__(self, local_state: dict[str, torch.Tensor]) -> None:
        # Copies the local state so the fixture cannot mutate the caller's
        # mapping.
        #
        # Args:
        #     local_state: Extractor parameters keyed as this implementation
        #         names them, typically read straight out of a constructed
        #         network.
        self._local_state: dict[str, torch.Tensor] = dict(local_state)

    def build(self) -> dict[str, torch.Tensor]:
        # Returns the same tensors keyed under the author naming the adapter
        # expects. The tensor objects are shared rather than cloned, which is
        # what lets the loading assertions compare loaded parameters against
        # the originals by exact equality.
        return {
            self._to_upstream_key(local_key): tensor
            for local_key, tensor in self._local_state.items()
        }

    def _to_upstream_key(self, local_key: str) -> str:
        # Maps one local attribute path onto its author-release counterpart.
        # Prefix matching is used rather than substring replacement, so a rule
        # can only rewrite the head of a key and cannot corrupt a member name
        # further down the path; the residual-block members are handled by a
        # second pass over the remainder. An unrecognized key is returned
        # unchanged, which is correct because such a key names a parameter
        # whose local and upstream spellings already agree.
        if local_key.startswith("_conv_block."):
            return f"conv_block.{local_key[len('_conv_block.'):]}"
        block_number: str
        for block_number in ("1", "2", "3"):
            prefix: str = f"_res_block_{block_number}."
            if local_key.startswith(prefix):
                return f"res_block{block_number}.{self._to_upstream_block_member(local_key[len(prefix):])}"
        if local_key.startswith("_pool_batch_norm."):
            return f"pool_block.0.{local_key[len('_pool_batch_norm.'):]}"
        if local_key.startswith("_bilstm_classifier."):
            return f"bilstm_classifier.{local_key[len('_bilstm_classifier.'):]}"
        if local_key.startswith("_classifier."):
            return f"classifier.{local_key[len('_classifier.'):]}"
        return local_key

    def _to_upstream_block_member(self, member_key: str) -> str:
        # Maps the three residual-block member names onto the author spellings.
        renamed: str = member_key.replace("_pre_convolution.", "pre_conv.")
        renamed: str = renamed.replace("_projection.", "conv1by1.")
        renamed: str = renamed.replace("_convolution.", "conv.")
        return renamed


class ExtractorStateReader:
    # Reads the F0-extractor slice out of a network state dictionary with the
    # prefix removed. Stripping the prefix is what makes the result directly
    # comparable with a standalone pitch checkpoint, whose keys are unprefixed
    # because the release describes the extractor alone rather than a
    # generator containing one.
    def __init__(self, network: HiftnetNetwork) -> None:
        # Binds the network whose extractor slice this reader extracts.
        self._network: HiftnetNetwork = network

    def read(self) -> dict[str, torch.Tensor]:
        # Returns the extractor parameters keyed as the extractor itself names
        # them.
        #
        # Returns:
        #     The extractor slice of the network's state dictionary. Entries
        #     outside the extractor are excluded, so a comparison between two
        #     networks isolates the pitch parameters from everything the
        #     checkpoint boundary does not touch.
        prefix: str = "_f0_model."
        return {
            key[len(prefix):]: value
            for key, value in self._network.state_dict().items()
            if key.startswith(prefix)
        }


class HiftnetNetworkConfigurationTest(unittest.TestCase):
    # Verifies the reference defaults and the validation behavior of the network configuration.
    def setUp(self) -> None:
        # Reads the default network configuration, which is the reference topology.
        self._configuration: HiftnetNetworkConfig = HiftnetNetworkConfig()

    def test_defaults_follow_the_reference_topology(self) -> None:
        # The reference generator upsamples eight and eight from 512 channels at 22.05 kHz.
        self.assertEqual(self._configuration.input_mel_channels, 80)
        self.assertEqual(self._configuration.sampling_rate, 22050)
        self.assertEqual(self._configuration.upsample_rates, (8, 8))
        self.assertEqual(self._configuration.upsample_kernel_sizes, (16, 16))
        self.assertEqual(self._configuration.upsample_initial_channel, 512)

    def test_inverse_stft_head_defaults_match_the_reference(self) -> None:
        # The final reconstruction is a sixteen-point inverse transform at hop four.
        self.assertEqual(self._configuration.gen_istft_n_fft, 16)
        self.assertEqual(self._configuration.gen_istft_hop_size, 4)

    def test_source_module_defaults_match_the_reference(self) -> None:
        # Sine amplitude, noise deviation, and the voicing threshold drive the harmonic source.
        self.assertEqual(self._configuration.sine_amplitude, 0.1)
        self.assertEqual(self._configuration.noise_standard_deviation, 0.003)
        self.assertEqual(self._configuration.voiced_threshold, 10.0)

    def test_pitch_checkpoint_path_defaults_to_absent(self) -> None:
        # The pretrained pitch predictor is opt-in, so the default configuration needs no binary.
        self.assertIsNone(self._configuration.f0_checkpoint_path)

    def test_configuration_is_immutable(self) -> None:
        # Frozen settings cannot drift after construction.
        with self.assertRaises(ValidationError):
            self._configuration.upsample_initial_channel = 256

    def test_unknown_field_is_rejected(self) -> None:
        # Extra fields are forbidden so silent typos cannot enter an experiment record.
        with self.assertRaises(ValidationError):
            HiftnetNetworkConfig(unknown_setting=1)


class HiftnetSourceFilterSynthesisTest(unittest.TestCase):
    # Verifies the source-filter synthesis chain, its shape contract, and its
    # spectral bundle. The assertions are structural rather than perceptual:
    # at random initialization the synthesized audio carries no meaning, so
    # what can be established is that the chain runs end to end, that each
    # frame expands by exactly the declared factor, that the head's two
    # channel groups obey the ranges their activations impose, and that the
    # result contains no non-finite values.
    def setUp(self) -> None:
        # Builds the reduced generator and the eight-frame conditioning mel it
        # synthesizes from. The seed fixes both the initialization and the
        # noise the harmonic source draws, so a failure here is attributable
        # to the code rather than to an unlucky draw.
        torch.manual_seed(1234)
        self._configuration: HiftnetNetworkConfig = ReducedGeneratorRecipe(f0_checkpoint_path=None).build()
        self._network: HiftnetNetwork = HiftnetNetwork(self._configuration)
        self._mel: torch.Tensor = torch.randn(1, 80, 8)

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The network exposes exactly the record it was constructed with.
        self.assertIs(self._network.configuration, self._configuration)

    def test_each_frame_expands_by_the_upsample_and_hop_product(self) -> None:
        # Rates four and four with hop four reconstruct sixty-four samples per frame.
        with torch.no_grad():
            waveform: torch.Tensor = self._network(self._mel)
        self.assertEqual(tuple(waveform.shape), (1, 1, 512))

    def test_batch_dimension_is_preserved(self) -> None:
        # Batched conditioning produces one waveform row per batch element.
        with torch.no_grad():
            waveform: torch.Tensor = self._network(torch.randn(2, 80, 8))
        self.assertEqual(tuple(waveform.shape), (2, 1, 512))

    def test_component_bundle_carries_magnitude_phase_and_waveform(self) -> None:
        # The bundle exposes the predicted spectra alongside the reconstructed
        # waveform. The nine spectral channels are the bin count of the
        # sixteen-point transform, and the frame count is the one the inverse
        # transform needs to produce five hundred twelve samples at hop four;
        # asserting both spectra at the same shape confirms the head's output
        # channels split evenly rather than being partitioned unequally.
        with torch.no_grad():
            components: HiftnetGeneratorOutput = self._network.predict_components(self._mel)
        self.assertIsInstance(components, HiftnetGeneratorOutput)
        self.assertEqual(tuple(components.magnitude.shape), (1, 9, 129))
        self.assertEqual(tuple(components.phase.shape), (1, 9, 129))
        self.assertEqual(tuple(components.waveform.shape), (1, 1, 512))

    def test_predicted_magnitude_is_strictly_positive(self) -> None:
        # The magnitude head exponentiates, so no predicted magnitude can be zero or negative.
        with torch.no_grad():
            components: HiftnetGeneratorOutput = self._network.predict_components(self._mel)
        self.assertTrue(bool((components.magnitude > 0.0).all()))

    def test_predicted_phase_stays_inside_the_sine_range(self) -> None:
        # The phase head is a sine, so its outputs are bounded by one in magnitude.
        with torch.no_grad():
            components: HiftnetGeneratorOutput = self._network.predict_components(self._mel)
        self.assertLessEqual(float(components.phase.abs().max()), 1.0)

    def test_synthesis_is_finite(self) -> None:
        # The harmonic source and the inverse transform produce no non-finite samples.
        with torch.no_grad():
            waveform: torch.Tensor = self._network(self._mel)
        self.assertTrue(bool(torch.isfinite(waveform).all()))

    def test_synthesis_is_reproducible_under_a_fixed_seed(self) -> None:
        # The stochastic harmonic source repeats exactly when the global seed
        # is repeated. This is the reproducibility contract that matters for
        # the study: the source module draws noise on every forward pass, so
        # synthesis is not a pure function of its input, and the guarantee
        # available is determinism given the seed rather than determinism as
        # such. Exact equality is asserted rather than approximate agreement,
        # because a seeded draw that differs at all indicates an extra or
        # reordered consumption of the generator.
        torch.manual_seed(7)
        with torch.no_grad():
            first_waveform: torch.Tensor = self._network(self._mel)
        torch.manual_seed(7)
        with torch.no_grad():
            second_waveform: torch.Tensor = self._network(self._mel)
        self.assertTrue(bool(torch.equal(first_waveform, second_waveform)))

    def test_mel_without_three_dimensions_is_rejected(self) -> None:
        # A two-dimensional mel is refused with the expected-layout message.
        with self.assertRaisesRegex(ValueError, "Expected mel shape"):
            self._network(torch.randn(80, 8))

    def test_component_bundle_is_immutable(self) -> None:
        # The frozen bundle cannot be edited between prediction and loss computation.
        with torch.no_grad():
            components: HiftnetGeneratorOutput = self._network.predict_components(self._mel)
        with self.assertRaises(ValidationError):
            components.waveform = torch.zeros(1)


class HiftnetReferenceTopologySynthesisTest(unittest.TestCase):
    # Verifies that the reference generator topology reconstructs 256 samples
    # per frame. This assertion is separated from the reduced-generator tests
    # because it is the one property the reduction cannot stand in for: the
    # total expansion must equal the mel protocol's hop length, and only the
    # full-size topology satisfies that.
    def setUp(self) -> None:
        # Constructs the reference generator once for its single expansion
        # assertion. The full topology is expensive to build relative to the
        # reduced one, which is why exactly one assertion is paid for here.
        torch.manual_seed(1234)
        self._network: HiftnetNetwork = HiftnetNetwork(HiftnetNetworkConfig())

    def test_reference_topology_expands_each_frame_to_two_hundred_fifty_six_samples(self) -> None:
        # Rates eight and eight with hop four reconstruct the 22.05 kHz frame
        # rate, so eight frames become two thousand forty-eight samples. This
        # is the invariant that keeps synthesis aligned with the conditioning
        # protocol: were the expansion to disagree with the mel hop, every
        # reconstruction comparison in the study would be measuring a
        # time-warped signal.
        with torch.no_grad():
            waveform: torch.Tensor = self._network(torch.randn(1, 80, 8))
        self.assertEqual(tuple(waveform.shape), (1, 1, 2048))
        self.assertTrue(bool(torch.isfinite(waveform).all()))


class HiftnetPitchExtractorCheckpointTest(unittest.TestCase):
    # Verifies the F0-extractor checkpoint boundary: absent paths, malformed
    # payloads, and adaptation. This boundary is the one external dependency
    # of the architecture, so its behavior is characterized in both
    # directions: an absent release must be tolerated silently, because
    # construction and testing must not require a download, while a present
    # but disagreeing release must be refused, because a partially initialized
    # pitch model would degrade synthesis without announcing itself.
    def setUp(self) -> None:
        # Opens a temporary payload directory and builds the reference network
        # under a fixed seed. The seed is what makes the comparison networks
        # below meaningful: two networks constructed from the same seed start
        # identical, so any difference in their extractor state is
        # attributable to the checkpoint boundary alone.
        torch.manual_seed(1234)
        self._temporary_root: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_root.name)
        self._recipe: ReducedGeneratorRecipe = ReducedGeneratorRecipe(f0_checkpoint_path=None)
        self._network: HiftnetNetwork = HiftnetNetwork(self._recipe.build())

    def tearDown(self) -> None:
        # Removes the fabricated payload directory.
        self._temporary_root.cleanup()

    def test_network_constructs_without_a_pitch_checkpoint(self) -> None:
        # The architecture is usable at random pitch-predictor initialization, which is what these tests use.
        self.assertIsInstance(self._network, HiftnetNetwork)
        self.assertIsNone(self._network.configuration.f0_checkpoint_path)

    def test_absent_checkpoint_path_is_skipped_silently(self) -> None:
        # A configured but missing file leaves the extractor at its
        # initialization instead of failing. The comparison network is rebuilt
        # under the same seed as the one in setUp, so proving the two
        # extractor states are identical proves the loader touched nothing.
        # The consequence documented here is deliberate but sharp: a
        # reproduction run with a mistyped path proceeds silently at random
        # pitch, so path correctness must be established outside this
        # boundary.
        missing_path: Path = self._root / "absent_f0.pt"
        torch.manual_seed(1234)
        network: HiftnetNetwork = HiftnetNetwork(ReducedGeneratorRecipe(f0_checkpoint_path=missing_path).build())
        skipped_state: dict[str, torch.Tensor] = ExtractorStateReader(network).read()
        initialized_state: dict[str, torch.Tensor] = ExtractorStateReader(self._network).read()
        self.assertEqual(network.configuration.f0_checkpoint_path, missing_path)
        self.assertEqual(sorted(skipped_state.keys()), sorted(initialized_state.keys()))
        self.assertTrue(
            bool(torch.equal(skipped_state["_classifier.weight"], initialized_state["_classifier.weight"])),
            msg="A missing checkpoint must leave the extractor at the initialization the seed produced"
        )

    def test_checkpoint_without_model_entry_is_rejected(self) -> None:
        # The reference payload keys its state under model, and anything else fails loudly.
        payload_path: Path = self._root / "no_model.pt"
        torch.save({"weights": {}}, payload_path)
        with self.assertRaisesRegex(ValueError, "must contain key"):
            HiftnetNetwork(ReducedGeneratorRecipe(f0_checkpoint_path=payload_path).build())

    def test_checkpoint_with_non_mapping_model_entry_is_rejected(self) -> None:
        # A model entry that is not a state dictionary cannot be adapted.
        payload_path: Path = self._root / "bad_model.pt"
        torch.save({"model": [1, 2, 3]}, payload_path)
        with self.assertRaisesRegex(TypeError, "must be a state dict"):
            HiftnetNetwork(ReducedGeneratorRecipe(f0_checkpoint_path=payload_path).build())

    def test_incomplete_checkpoint_is_rejected_after_adaptation(self) -> None:
        # Partial coverage of the extractor parameters is a contract break,
        # not a warning. This is the assertion that gives the relaxed loading
        # flag its meaning: the loader passes a non-strict flag so the
        # deliberately dropped detector entries do not fail, and then checks
        # the residual key lists itself, so a genuinely incomplete payload is
        # still refused rather than silently leaving most of the pitch model
        # at random initialization.
        payload_path: Path = self._root / "partial.pt"
        torch.save({"model": {"conv_block.0.weight": torch.zeros(64, 1, 3, 3)}}, payload_path)
        with self.assertRaisesRegex(RuntimeError, "missing keys after adaptation"):
            HiftnetNetwork(ReducedGeneratorRecipe(f0_checkpoint_path=payload_path).build())

    def test_reference_named_checkpoint_loads_completely(self) -> None:
        # A payload in author naming covers every extractor parameter after
        # adaptation. The per-parameter equality loop is the substance of the
        # assertion: matching key sets alone would pass even if the adapter
        # routed a tensor to the wrong slot, so every parameter is compared to
        # the one it was derived from. The size floor asserted first guards
        # the fixture rather than the adapter, since a comparison over an
        # accidentally empty state would otherwise succeed vacuously.
        local_state: dict[str, torch.Tensor] = ExtractorStateReader(self._network).read()
        upstream_state: dict[str, torch.Tensor] = UpstreamF0StateBuilder(local_state).build()
        payload_path: Path = self._root / "reference_f0.pt"
        torch.save({"model": upstream_state}, payload_path)
        loaded_network: HiftnetNetwork = HiftnetNetwork(
            ReducedGeneratorRecipe(f0_checkpoint_path=payload_path).build()
        )
        loaded_state: dict[str, torch.Tensor] = ExtractorStateReader(loaded_network).read()
        self.assertIn("_classifier.weight", local_state)
        self.assertGreater(len(local_state), 50)
        self.assertEqual(sorted(loaded_state.keys()), sorted(local_state.keys()))
        parameter_name: str
        for parameter_name in local_state:
            self.assertTrue(
                bool(torch.equal(loaded_state[parameter_name], local_state[parameter_name])),
                msg=f"Adapted checkpoint did not transfer {parameter_name}"
            )

    def test_unused_detector_state_is_ignored(self) -> None:
        # The reference release bundles pitch-detector branches this
        # implementation does not carry. Adding them to an otherwise complete
        # payload proves both halves of the intended behavior at once: the
        # extra entries are dropped rather than raising as unexpected keys,
        # and the parameters that are loaded arrive undisturbed, so the drop
        # rule cannot be consuming keys beyond the detector branches.
        local_state: dict[str, torch.Tensor] = ExtractorStateReader(self._network).read()
        upstream_state: dict[str, torch.Tensor] = UpstreamF0StateBuilder(local_state).build()
        upstream_state["detector_conv.0.weight"] = torch.zeros(2)
        upstream_state["bilstm_detector.weight_ih_l0"] = torch.zeros(2)
        payload_path: Path = self._root / "with_detector.pt"
        torch.save({"model": upstream_state}, payload_path)
        loaded_network: HiftnetNetwork = HiftnetNetwork(
            ReducedGeneratorRecipe(f0_checkpoint_path=payload_path).build()
        )
        loaded_state: dict[str, torch.Tensor] = ExtractorStateReader(loaded_network).read()
        self.assertEqual(sorted(loaded_state.keys()), sorted(local_state.keys()))
        self.assertNotIn("detector_conv.0.weight", loaded_state)
        self.assertTrue(
            bool(torch.equal(loaded_state["_classifier.weight"], local_state["_classifier.weight"])),
            msg="The detector branches must be dropped without disturbing the adapted parameters"
        )
