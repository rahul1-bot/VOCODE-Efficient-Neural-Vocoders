# This module:
# 1. Verifies the FreeV published-weight provenance record: the author
#    source, retrieval route, recorded SHA-256, local layout, checkpoint
#    serialization with its generator key, and path resolution against a
#    weights root
# 2. Verifies the loader's refusal paths: a module of the wrong architecture,
#    a release whose bytes do not match the recorded digest, and a release
#    path that is not a file
# 3. Verifies the checkpoint-dictionary validation guarding the strict load:
#    the payload must be a mapping carrying a mapping of tensors at the
#    generator key
#
# Design decisions:
# - No author binary is ever retrieved or read. Every filesystem fixture is
#   fabricated inside a temporary directory, so the tests are offline and
#   leave the published-weights cache untouched
# - The digest and missing-file refusals are driven through the public load
#   entry point, because a fabricated file at the expected path reaches the
#   verification gate without any download
# - Checkpoint validation is exercised directly on the adapter helper: the
#   public entry point can only reach it after the SHA-256 gate accepts the
#   genuine release bytes, which no fabricated file can satisfy
#
# Author: Rahul Sawhney

import tempfile
import unittest
from pathlib import Path

import torch
from pydantic import ValidationError

from syntheticmind.core.module import Module

from vocode.models.freev.freev import Freev, FreevConfig
from vocode.models.freev.weights import FreevWeights
from vocode.models.vocoder import PublishedWeightProvenance


class UnsupportedStubModule(Module):
    # Stands in for a harness module of a different architecture at the loader's type gate.
    # It is a genuine harness Module rather than an arbitrary object, so the
    # refusal is shown to rest on the specific architecture check and not on
    # the argument merely failing to be a module at all.
    def __init__(self) -> None:
        # Constructs an empty harness module that carries no synthesis network.
        # The absence of a network is deliberate: reaching the load would raise
        # an attribute error instead of the type error under test, so the
        # assertion cannot pass for the wrong reason.
        super().__init__()
        self._architecture_label: str = "unsupported_stub"


class FabricatedReleaseTree:
    # Materializes decoy release artifacts under a temporary published-weights root.
    # Placing a file exactly where the provenance resolves lets the refusal
    # paths be driven through the public load entry point without any
    # retrieval, because the loader only downloads when the resolved path does
    # not already exist.
    def __init__(self, root: Path, relative_path: Path) -> None:
        # Binds the weights root and the release layout under it.
        #
        # Args:
        #     root: Temporary directory standing in for the published-weights
        #         root, so no real cache entry is ever touched.
        #     relative_path: The release's recorded relative layout, taken from
        #         the adapter's own provenance rather than restated, so the
        #         fixture cannot drift from the path the loader resolves.
        self._root: Path = root
        self._relative_path: Path = relative_path

    def write_corrupted_release(self) -> Path:
        # Writes bytes that cannot match the recorded release digest.
        # Any content other than the released bytes reproduces the mismatch, so
        # a short readable string is used in place of a large binary fixture.
        #
        # Returns:
        #     The path written, which is where the loader resolves the release.
        target: Path = self._root / self._relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fabricated bytes that are not the author release")
        return target

    def create_directory_at_release_path(self) -> Path:
        # Creates a directory where the release file is expected.
        # This is the one case that reaches the digest gate while defeating the
        # existence check, which is what separates the not-a-file refusal from
        # the mismatch refusal.
        #
        # Returns:
        #     The directory created at the release path.
        target: Path = self._root / self._relative_path
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def root(self) -> Path:
        # Returns the temporary published-weights root.
        return self._root


class FreevPublishedWeightProvenanceTest(unittest.TestCase):
    # Verifies the recorded release identity of the official FreeV checkpoint.
    def setUp(self) -> None:
        # Builds the weight adapter and reads its provenance record.
        self._adapter: FreevWeights = FreevWeights()
        self._provenance: PublishedWeightProvenance = self._adapter.provenance

    def test_release_identity_matches_the_bakerbunker_publication(self) -> None:
        # The reference anchor is the author's one-million-step LJSpeech generator.
        self.assertEqual(self._provenance.architecture_name, "freev")
        self.assertEqual(self._provenance.variant_name, "bakerbunker_ljspeech_g_01000000")
        self.assertEqual(self._provenance.author_source_uri, "https://github.com/BakerBunker/FreeV")
        self.assertEqual(self._provenance.retrieval_kind, "huggingface_file")
        self.assertEqual(self._provenance.retrieval_uri, "Bakerbunker/FreeV_Model_Logs")
        self.assertEqual(self._provenance.retrieval_filename, "freev_g_01000000")

    def test_recorded_digest_and_serialization_are_pinned(self) -> None:
        # The release is a training checkpoint whose generator entry holds the weights.
        self.assertEqual(
            self._provenance.expected_sha256,
            "a2e92b9cb1278f49d41c43f8e05f09f61757e257fc1e63d2f97f4c91f4d45b9c"
        )
        self.assertEqual(self._provenance.serialization, "checkpoint_dict")
        self.assertEqual(self._provenance.state_dict_key, "generator")

    def test_local_path_resolves_under_the_weights_root(self) -> None:
        # The release lives at a fixed relative layout under any weights root.
        root: Path = Path("/fabricated/weights/root")
        self.assertEqual(self._provenance.resolve_local_path(root), root / Path("freev_lj/freev_g_01000000"))

    def test_provenance_record_is_frozen(self) -> None:
        # A release identity cannot be edited after it is recorded.
        with self.assertRaises(ValidationError):
            self._provenance.expected_sha256 = "0" * 64


class FreevCheckpointValidationTest(unittest.TestCase):
    # Verifies that a malformed checkpoint payload is rejected before any tensor reaches the network.
    def setUp(self) -> None:
        # Builds the weight adapter under test.
        self._adapter: FreevWeights = FreevWeights()

    def test_non_mapping_payload_is_rejected(self) -> None:
        # A checkpoint that is not a mapping cannot carry a generator entry.
        with self.assertRaises(TypeError):
            self._adapter._extract_state_dict([torch.zeros(1)])

    def test_payload_without_a_generator_entry_is_rejected(self) -> None:
        # The recorded serialization promises the weights live under the generator key.
        with self.assertRaisesRegex(TypeError, "generator"):
            self._adapter._extract_state_dict({"optimizer": {}})

    def test_non_tensor_entry_is_rejected(self) -> None:
        # Every entry of the generator state dictionary must be a tensor.
        with self.assertRaises(TypeError):
            self._adapter._extract_state_dict({"generator": {"PSP_input_conv.weight": 5}})

    def test_generator_entries_are_extracted_under_their_keys(self) -> None:
        # A well-formed checkpoint yields the generator state dictionary unchanged.
        payload: dict[str, dict[str, torch.Tensor]] = {"generator": {"PSP_input_conv.bias": torch.zeros(3)}}
        extracted: dict[str, torch.Tensor] = self._adapter._extract_state_dict(payload)
        self.assertEqual(list(extracted.keys()), ["PSP_input_conv.bias"])
        self.assertTrue(
            torch.equal(extracted["PSP_input_conv.bias"], payload["generator"]["PSP_input_conv.bias"])
        )


class FreevWeightLoadingRefusalTest(unittest.TestCase):
    # Verifies that the loader refuses wrong architectures, corrupted releases, and non-file release paths.
    def setUp(self) -> None:
        # Opens a temporary weights root and binds the fabricated release layout under it.
        self._adapter: FreevWeights = FreevWeights()
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
        with self.assertRaisesRegex(TypeError, "FreeV published weights require Freev"):
            self._adapter.load(UnsupportedStubModule(), self._tree.root)

    def test_release_with_a_mismatched_digest_is_rejected(self) -> None:
        # A substituted or corrupted file can never masquerade as the reference.
        self._tree.write_corrupted_release()
        torch.manual_seed(20260826)
        module: Freev = Freev(FreevConfig.official())
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self._adapter.load(module, self._tree.root)

    def test_release_path_that_is_not_a_file_is_rejected(self) -> None:
        # A directory at the release path must fail before any read is attempted.
        self._tree.create_directory_at_release_path()
        torch.manual_seed(20260827)
        module: Freev = Freev(FreevConfig.official())
        with self.assertRaises(FileNotFoundError):
            self._adapter.load(module, self._tree.root)
