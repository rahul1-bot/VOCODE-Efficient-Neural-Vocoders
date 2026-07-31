# This module:
# 1. Verifies the provenance records of the three jik876 HiFi-GAN
#    releases: the author source, the retrieval route, the recorded
#    SHA-256 identity, the local layout, and the serialization contract
# 2. Verifies the loader guards: a foreign module type, a release path
#    that is not a readable file, and a content hash that disagrees with
#    the recorded identity
# 3. Verifies the state-dictionary contract on fabricated releases: a
#    non-mapping payload, a payload without the generator entry, a
#    non-tensor state entry, and the upstream-to-project key adaptation
#
# Design decisions:
# - No author binary is ever retrieved or read: every load path is driven
#   with a fabricated checkpoint written into a temporary weights root,
#   and the adapter is constructed with a provenance record whose hash is
#   the fabricated file's own digest, so verification passes without a
#   download
# - The download branch is deliberately never reached: fabricated releases
#   are written at the resolved local path before loading, because a
#   missing file would drive the adapter into its retrieval route
# - Key adaptation is verified through the strict load failure, which
#   names the adapted project keys; that is the only observable surface
#   for the rename without a genuine release
# - Loading a release that actually matches the project network is out of
#   scope, because it requires the retained author binaries
#
# Author: Rahul Sawhney

import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import BinaryIO

import torch

from vocode.models.hifigan.hifigan import Hifigan, HifiganConfig
from vocode.models.hifigan.weights import HifiganWeights
from vocode.models.melgan.melgan import Melgan, MelganConfig
from vocode.models.vocoder import PublishedWeightProvenance


class FabricatedRelease:
    # Writes a fabricated author release into a temporary weights root and
    # builds the provenance record whose recorded hash matches the written
    # bytes, so the loader runs without retrieving an author binary.
    def __init__(self, weights_root: Path, architecture_name: str) -> None:
        # Binds the temporary root and the architecture the fabricated record anchors.
        self._weights_root: Path = weights_root
        self._architecture_name: str = architecture_name
        self._relative_path: Path = Path("fabricated_release/generator.pt")

    def write(self, payload: object) -> PublishedWeightProvenance:
        # Writes the payload first, then records its own digest as the expected
        # hash, so the verification gate accepts it without a download.
        # The ordering is what makes the fixture work at all: the digest
        # cannot be known before the bytes exist, and a provenance record
        # built with any other hash would be rejected at verification
        # before the payload-shape guards under test are ever reached.
        #
        # Args:
        #     payload: The object to serialize as the fabricated release.
        #         Malformed payloads are the point, since each drives one
        #         guard of the state-dictionary contract.
        #
        # Returns:
        #     A provenance record anchored to the written file, whose
        #     retrieval fields point at an unreachable host because the
        #     retrieval route is never entered.
        release_path: Path = self._weights_root / self._relative_path
        release_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, release_path)
        return PublishedWeightProvenance(
            architecture_name=self._architecture_name,
            variant_name="fabricated",
            author_source_uri="https://example.invalid/hifi-gan",
            retrieval_kind="http",
            retrieval_uri="https://example.invalid/hifi-gan/generator.pt",
            retrieval_filename=None,
            expected_sha256=self._digest(release_path),
            local_relative_path=self._relative_path,
            serialization="checkpoint_dict",
            state_dict_key="generator"
        )

    def _digest(self, release_path: Path) -> str:
        # Returns the SHA-256 of the written release bytes.
        handle: BinaryIO
        with release_path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()


class HifiganWeightsProvenanceTest(unittest.TestCase):
    # Verifies the three release records identify the jik876 LJSpeech
    # checkpoints by source, retrieval route, hash, and local layout.
    def setUp(self) -> None:
        # Builds the three release adapters; no release is retrieved or read.
        self._adapters: dict[str, HifiganWeights] = {
            "hifigan_v1": HifiganWeights.v1(),
            "hifigan_v2": HifiganWeights.v2(),
            "hifigan_v3": HifiganWeights.v3()
        }

    def test_each_factory_anchors_its_own_architecture(self) -> None:
        # A release record must never be wired to another width.
        architecture_name: str
        adapter: HifiganWeights
        for architecture_name, adapter in self._adapters.items():
            self.assertEqual(adapter.provenance.architecture_name, architecture_name)

    def test_each_release_names_the_author_repository(self) -> None:
        # Provenance points back at the author source, not a mirror.
        adapter: HifiganWeights
        for adapter in self._adapters.values():
            self.assertEqual(adapter.provenance.author_source_uri, "https://github.com/jik876/hifi-gan")

    def test_each_release_is_retrieved_from_the_author_drive_folder(self) -> None:
        # The jik876 releases are published as a shared drive folder.
        adapter: HifiganWeights
        for adapter in self._adapters.values():
            self.assertEqual(adapter.provenance.retrieval_kind, "google_drive_folder")
            self.assertIsNotNone(adapter.provenance.retrieval_filename)

    def test_each_release_records_a_distinct_content_hash(self) -> None:
        # Three distinct checkpoints must carry three distinct identities.
        recorded_hashes: list[str] = [
            adapter.provenance.expected_sha256 for adapter in self._adapters.values()
        ]
        self.assertEqual(len(set(recorded_hashes)), 3)
        recorded_hash: str
        for recorded_hash in recorded_hashes:
            self.assertEqual(len(recorded_hash), 64)

    def test_each_release_lands_in_its_own_local_directory(self) -> None:
        # Local layouts must not collide between the three widths.
        local_paths: list[Path] = [
            adapter.provenance.local_relative_path for adapter in self._adapters.values()
        ]
        self.assertEqual(len(set(local_paths)), 3)
        self.assertEqual(self._adapters["hifigan_v1"].provenance.local_relative_path.name, "generator_v1")

    def test_each_release_is_a_checkpoint_keyed_at_generator(self) -> None:
        # The author payload nests the generator state under one key.
        adapter: HifiganWeights
        for adapter in self._adapters.values():
            self.assertEqual(adapter.provenance.serialization, "checkpoint_dict")
            self.assertEqual(adapter.provenance.state_dict_key, "generator")

    def test_local_path_resolution_composes_with_the_weights_root(self) -> None:
        # The adapter reads its release from under the configured root.
        weights_root: Path = Path("/published_weights")
        resolved_path: Path = self._adapters["hifigan_v2"].provenance.resolve_local_path(weights_root)
        self.assertEqual(resolved_path, weights_root / Path("hifigan_v2_lj/generator_v2"))


class HifiganWeightsLoadGuardTest(unittest.TestCase):
    # Verifies the loader refuses a foreign module, an unreadable release
    # path, and a release whose bytes disagree with the recorded hash.
    def setUp(self) -> None:
        # Opens a temporary weights root so no guard test can reach the real cache.
        torch.manual_seed(0)
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._weights_root: Path = Path(self._temporary_directory.name)
        self._adapter: HifiganWeights = HifiganWeights.v1()
        self._module: Hifigan = Hifigan(HifiganConfig.v3())

    def tearDown(self) -> None:
        # Removes the temporary weights root.
        self._temporary_directory.cleanup()

    def test_loader_rejects_a_module_of_another_family(self) -> None:
        # HiFi-GAN weights can only be loaded onto a HiFi-GAN module.
        foreign_module: Melgan = Melgan(MelganConfig.seungwon())
        with self.assertRaisesRegex(TypeError, "require Hifigan"):
            self._adapter.load(foreign_module, self._weights_root)

    def test_loader_rejects_a_release_path_that_is_not_a_file(self) -> None:
        # A directory at the release path is not a readable release.
        occupied_path: Path = self._adapter.provenance.resolve_local_path(self._weights_root)
        occupied_path.mkdir(parents=True, exist_ok=True)
        with self.assertRaisesRegex(FileNotFoundError, "missing"):
            self._adapter.load(self._module, self._weights_root)

    def test_loader_rejects_a_release_whose_hash_disagrees(self) -> None:
        # A substituted release must never masquerade as the reference.
        release_path: Path = self._adapter.provenance.resolve_local_path(self._weights_root)
        release_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"generator": {}}, release_path)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self._adapter.load(self._module, self._weights_root)

    def test_hash_rejection_reports_both_the_expected_and_actual_digest(self) -> None:
        # The failure carries enough context to identify the bad cache entry.
        release_path: Path = self._adapter.provenance.resolve_local_path(self._weights_root)
        release_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"generator": {}}, release_path)
        with self.assertRaisesRegex(ValueError, self._adapter.provenance.expected_sha256):
            self._adapter.load(self._module, self._weights_root)


class HifiganWeightsStateDictionaryTest(unittest.TestCase):
    # Verifies the state-dictionary contract and the upstream-to-project
    # key adaptation using fabricated releases in a temporary root.
    def setUp(self) -> None:
        # Opens a temporary weights root and binds the fabricated release writer.
        torch.manual_seed(0)
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._weights_root: Path = Path(self._temporary_directory.name)
        self._release: FabricatedRelease = FabricatedRelease(self._weights_root, "hifigan_v3")
        self._module: Hifigan = Hifigan(HifiganConfig.v3())

    def tearDown(self) -> None:
        # Removes the temporary weights root and every fabricated release under it.
        self._temporary_directory.cleanup()

    def test_loader_rejects_a_payload_that_is_not_a_mapping(self) -> None:
        # An author checkpoint is a mapping of named state.
        provenance: PublishedWeightProvenance = self._release.write([1, 2, 3])
        with self.assertRaisesRegex(TypeError, "must be a mapping"):
            HifiganWeights(provenance).load(self._module, self._weights_root)

    def test_loader_rejects_a_payload_without_the_generator_entry(self) -> None:
        # The generator state must live under the recorded key.
        provenance: PublishedWeightProvenance = self._release.write({"discriminator": {}})
        with self.assertRaisesRegex(TypeError, "mapping at generator"):
            HifiganWeights(provenance).load(self._module, self._weights_root)

    def test_loader_rejects_a_non_tensor_state_entry(self) -> None:
        # Every entry of the release state must be a tensor.
        provenance: PublishedWeightProvenance = self._release.write(
            {"generator": {"conv_pre.bias": "not-a-tensor"}}
        )
        with self.assertRaisesRegex(TypeError, "must be torch.Tensor"):
            HifiganWeights(provenance).load(self._module, self._weights_root)

    def test_loader_renames_upstream_keys_to_the_project_layout(self) -> None:
        # The strict load names the adapted project keys, proving the rename.
        provenance: PublishedWeightProvenance = self._release.write(
            {"generator": {"conv_pre.weight_g": torch.zeros(2)}}
        )
        with self.assertRaisesRegex(RuntimeError, r"_pre_convolution\.parametrizations\.weight\.original0"):
            HifiganWeights(provenance).load(self._module, self._weights_root)

    def test_loader_renames_nested_resblock_keys(self) -> None:
        # Residual-block and convolution-list names are adapted as well.
        provenance: PublishedWeightProvenance = self._release.write(
            {"generator": {"resblocks.0.convs1.0.weight_v": torch.zeros(2)}}
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"_residual_blocks\.0\._dilated_convolutions\.0\.parametrizations\.weight\.original1"
        ):
            HifiganWeights(provenance).load(self._module, self._weights_root)

    def test_loader_refuses_a_release_that_does_not_match_the_network(self) -> None:
        # Strict loading proves architectural agreement instead of assuming it.
        provenance: PublishedWeightProvenance = self._release.write({"generator": {}})
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            HifiganWeights(provenance).load(self._module, self._weights_root)


if __name__ == "__main__":
    unittest.main()
