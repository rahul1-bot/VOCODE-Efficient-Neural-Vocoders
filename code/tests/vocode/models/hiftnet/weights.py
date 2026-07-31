# This module:
# 1. Verifies the HiFTNet published-weight provenance record: the release
#    identity, the retrieval contract, the recorded digest, and the local
#    path resolution
# 2. Verifies the loading guards: the module-type refusal, the missing-file
#    refusal, and the digest-mismatch refusal, all raised before any tensor
#    is read
# 3. Verifies the state-key adaptation: the generator, source-module, and
#    F0-extractor renamings, and the sixteen-entry detector-state contract
#    that guards against an upstream layout change
#
# Design decisions:
# - No author binary is ever retrieved and no network call is ever made:
#   every fixture is a fabricated file or an in-memory mapping, so the
#   successful retrieval-and-strict-load path is deliberately out of scope
#   and remains untested here
# - The digest guard runs before extraction, so the checkpoint-structure
#   refusals cannot be reached through load with a fabricated file; they are
#   exercised on the extraction and adaptation helpers directly
# - Adaptation fixtures carry exactly sixteen detector entries, because the
#   adapter treats any other count as a changed upstream contract; that
#   count is itself asserted from both sides
#
# Author: Rahul Sawhney

import tempfile
import unittest
from pathlib import Path

import torch

from vocode.metrics.pesq import PesqConfig
from vocode.metrics.stoi import StoiConfig
from vocode.models.hiftnet.hiftnet import Hiftnet, HiftnetConfig
from vocode.models.hiftnet.weights import HiftnetWeights
from vocode.models.vocoder import PublishedWeightProvenance
from vocode.transforms.mel import MelConfig


class ReducedModuleRecipe:
    # Builds the reduced HiFTNet module used as the load target of the guard
    # assertions. The module's topology is irrelevant here, because every
    # assertion in this file stops before any tensor is applied to it; a small
    # module is used purely so constructing the target of a refusal costs
    # almost nothing.
    def build(self) -> Hiftnet:
        # Returns a constructed module small enough to be the target of every
        # refusal path.
        #
        # Returns:
        #     A Hiftnet whose pitch-checkpoint path is absent, so building the
        #     load target performs no filesystem access of its own and cannot
        #     contaminate the guard assertions.
        configuration: HiftnetConfig = HiftnetConfig(
            input_mel_channels=80,
            sampling_rate=22050,
            upsample_rates=(4, 4),
            upsample_kernel_sizes=(8, 8),
            upsample_initial_channel=32,
            resblock_kernel_sizes=(3,),
            resblock_dilation_sizes=((1, 3, 5),),
            gen_istft_n_fft=16,
            gen_istft_hop_size=4,
            f0_checkpoint_path=None,
            mel_protocol=MelConfig.hiftnet_yl4579(),
            pesq_protocol=PesqConfig(),
            stoi_protocol=StoiConfig()
        )
        return Hiftnet(configuration)


class UpstreamCheckpointBuilder:
    # Builds fabricated upstream state mappings carrying a chosen number of
    # detector entries. Making that count a constructor parameter is what lets
    # the same generator fixture be presented to the adapter three ways: at
    # the contractual count, below it, and above it. No real release is ever
    # read, so these fixtures define the upstream layout the adapter is tested
    # against.
    def __init__(self, detector_entry_count: int) -> None:
        # Binds how many detector entries the fabricated release declares.
        #
        # Args:
        #     detector_entry_count: Number of voicing-detector entries to
        #         emit. Sixteen is the count the adapter accepts; any other
        #         value is expected to be refused.
        self._detector_entry_count: int = detector_entry_count

    def build(self, generator_entries: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Returns the generator entries alongside the declared count of
        # detector entries.
        #
        # Args:
        #     generator_entries: Upstream-named generator state to preserve
        #         verbatim, so the detector count is the only variable across
        #         the three fixtures.
        #
        # Returns:
        #     The merged mapping. The detector entries use the upstream
        #     prefix the adapter matches on, and their tensors are minimal
        #     because only the keys are inspected.
        detector_entries: dict[str, torch.Tensor] = {
            f"F0_model.detector_conv.{entry_index}.weight": torch.zeros(1)
            for entry_index in range(self._detector_entry_count)
        }
        return generator_entries | detector_entries


class HiftnetWeightProvenanceTest(unittest.TestCase):
    # Verifies the recorded release identity and retrieval contract of the
    # published checkpoint. The provenance record is the study's citation of
    # which exact bytes anchor this architecture, so these assertions pin the
    # values that a reader would need to obtain the same file independently;
    # any edit to them changes what the reproduction claim refers to.
    def setUp(self) -> None:
        # Builds the weight adapter and reads its provenance record.
        self._weights: HiftnetWeights = HiftnetWeights()
        self._provenance: PublishedWeightProvenance = self._weights.provenance

    def test_release_identity_names_the_published_ljspeech_variant(self) -> None:
        # The anchor is the yl4579 LJSpeech release at step 155000.
        self.assertEqual(self._provenance.architecture_name, "hiftnet")
        self.assertEqual(self._provenance.variant_name, "yl4579_ljspeech_g_00155000")
        self.assertIn("yl4579/HiFTNet", self._provenance.author_source_uri)

    def test_retrieval_contract_names_the_hugging_face_file(self) -> None:
        # Retrieval goes through the Hugging Face file API at the recorded
        # repository and filename. The retrieval repository is a
        # project-controlled mirror rather than the author's, because the
        # original release is a zip archive; the recorded digest is what makes
        # that substitution auditable, which is why the previous assertion
        # pins the author source separately.
        self.assertEqual(self._provenance.retrieval_kind, "huggingface_file")
        self.assertEqual(self._provenance.retrieval_uri, "r-sawhney/vocode-phase1-assets")
        self.assertEqual(self._provenance.retrieval_filename, "hiftnet_lj/g_00155000")

    def test_recorded_digest_is_a_full_sha256(self) -> None:
        # A traceable anchor requires the complete sixty-four character hexadecimal digest.
        self.assertEqual(len(self._provenance.expected_sha256), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in self._provenance.expected_sha256))

    def test_serialization_layout_names_the_generator_entry(self) -> None:
        # The release is a checkpoint dictionary whose generator entry carries the state.
        self.assertEqual(self._provenance.serialization, "checkpoint_dict")
        self.assertEqual(self._provenance.state_dict_key, "generator")

    def test_local_path_resolves_under_the_published_weights_root(self) -> None:
        # The release lives at its recorded relative path beneath the caller's weights root.
        root: Path = Path("/tmp/vocode-published-weights")
        self.assertEqual(self._provenance.resolve_local_path(root), root / "hiftnet_lj" / "g_00155000")

    def test_provenance_record_is_immutable(self) -> None:
        # A frozen record cannot be edited into agreement with a substituted release.
        with self.assertRaises(ValueError):
            self._provenance.expected_sha256 = "0" * 64


class HiftnetWeightLoadGuardTest(unittest.TestCase):
    # Verifies that loading refuses wrong modules, missing files, and
    # mismatched digests. Each assertion drives the real load entry point and
    # relies on the guard ordering to stop before the network is reached: the
    # type check precedes retrieval, and the digest check precedes
    # deserialization. That ordering is what makes these tests safe to run
    # offline, since no path here can reach a download or a tensor read.
    def setUp(self) -> None:
        # Opens a temporary weights root and resolves the release path the
        # guards inspect. A temporary directory is used so the fabricated
        # files cannot collide with, or be mistaken for, a genuine cached
        # release on the developer's machine.
        torch.manual_seed(1234)
        self._weights: HiftnetWeights = HiftnetWeights()
        self._temporary_root: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_root.name)
        self._local_path: Path = self._weights.provenance.resolve_local_path(self._root)

    def tearDown(self) -> None:
        # Removes the fabricated release tree.
        self._temporary_root.cleanup()

    def test_loading_onto_a_foreign_module_is_refused(self) -> None:
        # The type guard runs before retrieval, so a foreign module never triggers a download.
        with self.assertRaisesRegex(TypeError, "require Hiftnet"):
            self._weights.load(torch.nn.Linear(2, 2), self._root)

    def test_missing_release_file_is_refused(self) -> None:
        # A resolved path that is not a readable file fails before any tensor
        # is read. A directory is created at the release path rather than
        # leaving it absent, because an absent path would trigger the download
        # branch; occupying the path with a non-file drives the existence
        # check straight into the digest guard's file test instead.
        self._local_path.mkdir(parents=True, exist_ok=True)
        with self.assertRaisesRegex(FileNotFoundError, "weight file is missing"):
            self._weights.load(ReducedModuleRecipe().build(), self._root)

    def test_digest_mismatch_is_refused(self) -> None:
        # A substituted or corrupted release cannot masquerade as the recorded
        # anchor. The fabricated content is not a checkpoint at all, which is
        # sufficient precisely because the digest is verified before
        # deserialization is attempted; reaching a deserialization error here
        # instead would itself prove the guard ordering had regressed.
        self._local_path.parent.mkdir(parents=True, exist_ok=True)
        self._local_path.write_bytes(b"not the published release")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self._weights.load(ReducedModuleRecipe().build(), self._root)


class HiftnetCheckpointExtractionTest(unittest.TestCase):
    # Verifies the checkpoint-structure guards and the tensor extraction of
    # the generator entry. These assertions call the extraction helper
    # directly rather than going through load, because the digest guard would
    # reject any fabricated file long before extraction ran; testing the
    # helper in isolation is the only way to reach these branches without
    # possessing the genuine release.
    def setUp(self) -> None:
        # Builds the weight adapter whose extraction helper is under test.
        self._weights: HiftnetWeights = HiftnetWeights()

    def test_non_mapping_checkpoint_is_refused(self) -> None:
        # A checkpoint that is not a mapping cannot carry a keyed generator state.
        with self.assertRaisesRegex(TypeError, "must be a mapping"):
            self._weights._extract_state_dict([1, 2, 3])

    def test_checkpoint_without_generator_entry_is_refused(self) -> None:
        # The recorded serialization layout demands a mapping at the generator key.
        with self.assertRaisesRegex(TypeError, "mapping at generator"):
            self._weights._extract_state_dict({"model": {}})

    def test_non_tensor_state_entry_is_refused(self) -> None:
        # Every extracted entry must be a tensor, and the offending key is
        # named. Matching on the key rather than on generic wording is what
        # proves the message is actionable: an operator reading the failure
        # learns which entry of the release broke the contract.
        with self.assertRaisesRegex(TypeError, "conv_pre.bias"):
            self._weights._extract_state_dict({"generator": {"conv_pre.bias": 1.0}})

    def test_generator_entry_is_extracted_as_tensors(self) -> None:
        # A well-formed release yields the flat key-to-tensor mapping the adapter renames.
        extracted: dict[str, torch.Tensor] = self._weights._extract_state_dict(
            {"generator": {"conv_pre.bias": torch.zeros(2)}}
        )
        self.assertEqual(list(extracted.keys()), ["conv_pre.bias"])
        self.assertTrue(bool(torch.equal(extracted["conv_pre.bias"], torch.zeros(2))))


class HiftnetStateKeyAdaptationTest(unittest.TestCase):
    # Verifies the upstream-to-local renaming of the generator, source-module,
    # and extractor state. Key adaptation is where a reproduction most easily
    # goes wrong without failing: a rule that silently misses leaves a
    # parameter at its random initialization, and strict loading downstream
    # cannot detect what was never presented. These assertions therefore check
    # the produced key names directly rather than checking only that loading
    # succeeded.
    def setUp(self) -> None:
        # Builds one upstream-named fixture covering every renaming family the
        # adapter handles: the generator convolutions, the residual blocks
        # with both convolution roles and both activation coefficients, the
        # source and noise paths, and the full pitch-extractor tree. Covering
        # every family in a single fixture means each assertion below runs
        # against the same input, so a rule that consumes text belonging to
        # another rule is exposed rather than hidden by a narrower fixture.
        torch.manual_seed(1234)
        self._weights: HiftnetWeights = HiftnetWeights()
        self._builder: UpstreamCheckpointBuilder = UpstreamCheckpointBuilder(detector_entry_count=16)
        self._generator_entries: dict[str, torch.Tensor] = {
            "conv_pre.weight_g": torch.zeros(1),
            "conv_pre.weight_v": torch.ones(1),
            "ups.0.weight_g": torch.zeros(1),
            "resblocks.0.convs1.0.weight_g": torch.zeros(1),
            "resblocks.0.convs2.0.weight_v": torch.zeros(1),
            "resblocks.0.alpha1.0": torch.zeros(1),
            "resblocks.0.alpha2.0": torch.zeros(1),
            "noise_convs.0.weight": torch.zeros(1),
            "noise_res.0.convs1.0.weight_g": torch.zeros(1),
            "m_source.l_linear.weight": torch.zeros(1),
            "conv_post.weight_g": torch.zeros(1),
            "F0_model.conv_block.0.weight": torch.zeros(1),
            "F0_model.res_block1.pre_conv.0.weight": torch.zeros(1),
            "F0_model.res_block2.conv1by1.weight": torch.zeros(1),
            "F0_model.res_block3.conv.0.weight": torch.zeros(1),
            "F0_model.pool_block.0.weight": torch.zeros(1),
            "F0_model.bilstm_classifier.weight_ih_l0": torch.zeros(1),
            "F0_model.classifier.weight": torch.zeros(1)
        }
        self._upstream_state: dict[str, torch.Tensor] = self._builder.build(self._generator_entries)

    def test_generator_convolutions_are_renamed_to_local_attributes(self) -> None:
        # The upstream generator names map onto the descriptive local
        # attribute names. The expected keys also carry the parametrization
        # suffixes, which proves the second half of the adaptation: the
        # release stores weight normalization under the older paired-tensor
        # names, and those must be rewritten to the registry paths the current
        # framework expects or the tensors would arrive as unexpected keys.
        adapted: dict[str, torch.Tensor] = self._weights._adapt_state_dict(self._upstream_state)
        self.assertIn("_pre_convolution.parametrizations.weight.original0", adapted)
        self.assertIn("_upsample_layers.0.parametrizations.weight.original0", adapted)
        self.assertIn("_post_convolution.parametrizations.weight.original0", adapted)

    def test_residual_blocks_are_renamed_with_their_convolution_roles(self) -> None:
        # Dilated and refinement convolution lists carry their role in the local naming.
        adapted: dict[str, torch.Tensor] = self._weights._adapt_state_dict(self._upstream_state)
        self.assertIn("_residual_blocks.0._dilated_convolutions.0.parametrizations.weight.original0", adapted)
        self.assertIn("_residual_blocks.0._refinement_convolutions.0.parametrizations.weight.original1", adapted)
        self.assertIn("_residual_blocks.0._alpha_1.0", adapted)
        self.assertIn("_residual_blocks.0._alpha_2.0", adapted)

    def test_source_module_and_noise_path_are_renamed(self) -> None:
        # The harmonic source projection and the noise convolutions carry local names.
        adapted: dict[str, torch.Tensor] = self._weights._adapt_state_dict(self._upstream_state)
        self.assertIn("_source_module._linear.weight", adapted)
        self.assertIn("_noise_convolutions.0.weight", adapted)
        self.assertIn("_noise_residuals.0._dilated_convolutions.0.parametrizations.weight.original0", adapted)

    def test_pitch_extractor_state_is_renamed(self) -> None:
        # The bundled F0 extractor is renamed into the local extractor attribute tree.
        adapted: dict[str, torch.Tensor] = self._weights._adapt_state_dict(self._upstream_state)
        self.assertIn("_f0_model._conv_block.0.weight", adapted)
        self.assertIn("_f0_model._res_block_1._pre_convolution.0.weight", adapted)
        self.assertIn("_f0_model._res_block_2._projection.weight", adapted)
        self.assertIn("_f0_model._res_block_3._convolution.0.weight", adapted)
        self.assertIn("_f0_model._pool_batch_norm.weight", adapted)
        self.assertIn("_f0_model._bilstm_classifier.weight_ih_l0", adapted)
        self.assertIn("_f0_model._classifier.weight", adapted)

    def test_detector_entries_are_dropped_from_the_adapted_state(self) -> None:
        # The unused detector branches never reach the local module. The count
        # equality is the stronger half of this assertion: it proves that
        # exactly the detector entries were removed and nothing else, which a
        # substring scan over the surviving keys alone could not establish.
        adapted: dict[str, torch.Tensor] = self._weights._adapt_state_dict(self._upstream_state)
        self.assertEqual(len(adapted), len(self._generator_entries))
        adapted_key: str
        for adapted_key in adapted:
            self.assertNotIn("detector", adapted_key)

    def test_missing_detector_entries_break_the_upstream_contract(self) -> None:
        # A release without exactly sixteen detector entries is an audited
        # layout change. Refusing a release that omits entries matters more
        # than it first appears: fewer skipped keys can also mean a renamed
        # branch whose parameters are now silently discarded elsewhere, so the
        # count is checked from below as well as above.
        without_detector_state: dict[str, torch.Tensor] = UpstreamCheckpointBuilder(
            detector_entry_count=0
        ).build(self._generator_entries)
        with self.assertRaisesRegex(RuntimeError, "detector-state contract changed"):
            self._weights._adapt_state_dict(without_detector_state)

    def test_extra_detector_entries_break_the_upstream_contract(self) -> None:
        # More detector entries than recorded is equally an audited layout
        # change. The assertion matches the observed count in the message
        # rather than only the failure wording, which proves the diagnostic
        # reports what was actually found and so tells an auditor how far the
        # release has drifted.
        with_extra_detector_state: dict[str, torch.Tensor] = UpstreamCheckpointBuilder(
            detector_entry_count=17
        ).build(self._generator_entries)
        with self.assertRaisesRegex(RuntimeError, "received 17"):
            self._weights._adapt_state_dict(with_extra_detector_state)
