# This module:
# 1. Verifies the Vocos published-weight provenance record: the author source,
#    retrieval route, recorded SHA-256, local layout, serialization kind, and
#    the frozen path resolution against a weights root
# 2. Verifies the loader's refusal paths: a module of the wrong architecture,
#    a release whose bytes do not match the recorded digest, and a release
#    path that is not a file
# 3. Verifies the upstream-to-project key adaptation and the state-dictionary
#    validation that guard the strict load
#
# Design decisions:
# - No author binary is ever retrieved or read. Every filesystem fixture is
#   fabricated inside a temporary directory, so the tests are offline and
#   leave the published-weights cache untouched
# - The digest and missing-file refusals are driven through the public load
#   entry point, because a fabricated file at the expected path reaches the
#   verification gate without any download
# - Key adaptation and state-dictionary validation are exercised directly on
#   the adapter helpers: the public entry point can only reach them after the
#   SHA-256 gate accepts the genuine release bytes, which no fabricated file
#   can satisfy
# - Adapted keys are asserted to exist in a freshly built network's state
#   dictionary, so the mapping is checked against the real parameter layout
#   rather than against a transcription of itself
#
# Author: Rahul Sawhney

import tempfile
import unittest
from pathlib import Path

import torch
from pydantic import ValidationError

from syntheticmind.core.module import Module

from vocode.models.vocoder import PublishedWeightProvenance
from vocode.models.vocos.network import VocosNetwork, VocosNetworkConfig
from vocode.models.vocos.vocos import Vocos, VocosConfig
from vocode.models.vocos.weights import VocosWeights


class UnsupportedStubModule(Module):
    # Stands in for a harness module of a different architecture at the loader's type gate.
    # A real sibling family's module would serve equally, but an empty
    # stub keeps the type-gate test free of any other family's
    # construction cost and makes the rejection unambiguously about the
    # type rather than about a network mismatch.
    def __init__(self) -> None:
        # Constructs an empty harness module that carries no synthesis network.
        # The absent network is deliberate: reaching it would mean the
        # type gate had already been passed, so any attribute error here
        # would itself be a test failure.
        super().__init__()
        self._architecture_label: str = "unsupported_stub"


class FabricatedReleaseTree:
    # Materializes decoy release artifacts under a temporary published-weights root.
    def __init__(self, root: Path, relative_path: Path) -> None:
        # Binds the weights root and the release layout under it.
        self._root: Path = root
        self._relative_path: Path = relative_path

    def write_corrupted_release(self) -> Path:
        # Writes bytes that cannot match the recorded release digest.
        target: Path = self._root / self._relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fabricated bytes that are not the author release")
        return target

    def create_directory_at_release_path(self) -> Path:
        # Creates a directory where the release file is expected.
        target: Path = self._root / self._relative_path
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def root(self) -> Path:
        # Returns the temporary published-weights root.
        return self._root


class VocosPublishedWeightProvenanceTest(unittest.TestCase):
    # Verifies the recorded release identity of the charactr Vocos checkpoint.
    def setUp(self) -> None:
        # Builds the weight adapter and reads its provenance record.
        self._adapter: VocosWeights = VocosWeights()
        self._provenance: PublishedWeightProvenance = self._adapter.provenance

    def test_release_identity_matches_the_charactr_publication(self) -> None:
        # The reference anchor is the charactr 24 kHz mel release on the hub.
        self.assertEqual(self._provenance.architecture_name, "vocos")
        self.assertEqual(self._provenance.variant_name, "charactr_mel_24khz")
        self.assertEqual(self._provenance.author_source_uri, "https://github.com/charactr-platform/vocos")
        self.assertEqual(self._provenance.retrieval_kind, "huggingface_file")
        self.assertEqual(self._provenance.retrieval_uri, "charactr/vocos-mel-24khz")
        self.assertEqual(self._provenance.retrieval_filename, "pytorch_model.bin")

    def test_recorded_digest_and_serialization_are_pinned(self) -> None:
        # The release is a bare state dictionary verified against this digest.
        self.assertEqual(
            self._provenance.expected_sha256,
            "97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a"
        )
        self.assertEqual(self._provenance.serialization, "pytorch_state_dict")
        self.assertIsNone(self._provenance.state_dict_key)

    def test_local_path_resolves_under_the_weights_root(self) -> None:
        # The release lives at a fixed relative layout under any weights root.
        root: Path = Path("/fabricated/weights/root")
        self.assertEqual(
            self._provenance.resolve_local_path(root),
            root / Path("vocos_mel_24khz/pytorch_model.bin")
        )

    def test_provenance_record_is_frozen(self) -> None:
        # A release identity cannot be edited after it is recorded.
        with self.assertRaises(ValidationError):
            self._provenance.expected_sha256: str = "0" * 64


class VocosWeightKeyAdaptationTest(unittest.TestCase):
    # Verifies that upstream release keys are remapped onto the project network's parameter layout.
    def setUp(self) -> None:
        # Builds the adapter and the parameter key set of a freshly constructed network.
        self._adapter: VocosWeights = VocosWeights()
        torch.manual_seed(20260805)
        self._network_keys: set[str] = set(VocosNetwork(VocosNetworkConfig()).state_dict().keys())

    def test_backbone_and_head_keys_map_onto_project_parameters(self) -> None:
        # Every adapted key must name a parameter the strict load will bind.
        upstream_state_dict: dict[str, torch.Tensor] = {
            "backbone.embed.weight": torch.zeros(1),
            "backbone.norm.bias": torch.zeros(1),
            "backbone.convnext.0.dwconv.weight": torch.zeros(1),
            "backbone.convnext.0.pwconv1.weight": torch.zeros(1),
            "backbone.convnext.0.pwconv2.bias": torch.zeros(1),
            "backbone.convnext.3.gamma": torch.zeros(1),
            "backbone.final_layer_norm.weight": torch.zeros(1),
            "head.out.weight": torch.zeros(1)
        }
        adapted: dict[str, torch.Tensor] = self._adapter._adapt_state_dict(upstream_state_dict)
        self.assertEqual(len(adapted), len(upstream_state_dict))
        adapted_key: str
        for adapted_key in adapted:
            self.assertIn(
                adapted_key,
                self._network_keys,
                msg=f"Adapted key {adapted_key} names no parameter of the project network"
            )

    def test_individual_key_translations_are_exact(self) -> None:
        # The mapping is a fixed rename table between the release and the project layout.
        adapted: dict[str, torch.Tensor] = self._adapter._adapt_state_dict({
            "backbone.embed.weight": torch.zeros(1),
            "backbone.convnext.2.dwconv.bias": torch.zeros(1),
            "head.out.bias": torch.zeros(1)
        })
        self.assertIn("_backbone._embedding.weight", adapted)
        self.assertIn("_backbone._blocks.2._depthwise_convolution.bias", adapted)
        self.assertIn("_head._output_projection.bias", adapted)

    def test_external_feature_state_is_dropped(self) -> None:
        # The release ships extractor and window buffers the project network does not own.
        adapted: dict[str, torch.Tensor] = self._adapter._adapt_state_dict({
            "feature_extractor.mel_spec.mel_scale.fb": torch.zeros(1),
            "feature_extractor.mel_spec.spectrogram.window": torch.zeros(1),
            "head.istft.window": torch.zeros(1),
            "head.out.weight": torch.zeros(1)
        })
        self.assertEqual(list(adapted.keys()), ["_head._output_projection.weight"])


class VocosStateDictValidationTest(unittest.TestCase):
    # Verifies that a malformed release payload is rejected before any tensor reaches the network.
    def setUp(self) -> None:
        # Builds the weight adapter under test.
        self._adapter: VocosWeights = VocosWeights()

    def test_non_mapping_payload_is_rejected(self) -> None:
        # A release that is not a mapping cannot be a state dictionary.
        with self.assertRaises(TypeError):
            self._adapter._extract_state_dict([torch.zeros(1)])

    def test_non_tensor_entry_is_rejected(self) -> None:
        # Every entry of a state dictionary must be a tensor.
        with self.assertRaises(TypeError):
            self._adapter._extract_state_dict({"backbone.embed.weight": 5})

    def test_tensor_entries_are_preserved_under_their_keys(self) -> None:
        # A well-formed payload passes through with its keys intact.
        payload: dict[str, torch.Tensor] = {"backbone.embed.weight": torch.zeros(3)}
        extracted: dict[str, torch.Tensor] = self._adapter._extract_state_dict(payload)
        self.assertEqual(list(extracted.keys()), ["backbone.embed.weight"])
        self.assertTrue(torch.equal(extracted["backbone.embed.weight"], payload["backbone.embed.weight"]))


class VocosWeightLoadingRefusalTest(unittest.TestCase):
    # Verifies that the loader refuses wrong architectures, corrupted releases, and non-file release paths.
    def setUp(self) -> None:
        # Opens a temporary weights root and binds the fabricated release layout under it.
        self._adapter: VocosWeights = VocosWeights()
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._tree: FabricatedReleaseTree = FabricatedReleaseTree(
            root=Path(self._temporary_directory.name),
            relative_path=self._adapter.provenance.local_relative_path
        )

    def tearDown(self) -> None:
        # Removes the fabricated release tree.
        self._temporary_directory.cleanup()

    def test_module_of_another_architecture_is_rejected(self) -> None:
        # Author weights may only be loaded onto the architecture they belong to.
        with self.assertRaisesRegex(TypeError, "Vocos published weights require Vocos"):
            self._adapter.load(UnsupportedStubModule(), self._tree.root)

    def test_release_with_a_mismatched_digest_is_rejected(self) -> None:
        # A substituted or corrupted file can never masquerade as the reference.
        self._tree.write_corrupted_release()
        torch.manual_seed(20260806)
        module: Vocos = Vocos(VocosConfig())
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self._adapter.load(module, self._tree.root)

    def test_release_path_that_is_not_a_file_is_rejected(self) -> None:
        # A directory at the release path must fail before any read is attempted.
        self._tree.create_directory_at_release_path()
        torch.manual_seed(20260807)
        module: Vocos = Vocos(VocosConfig())
        with self.assertRaises(FileNotFoundError):
            self._adapter.load(module, self._tree.root)


if __name__ == "__main__":
    unittest.main()
