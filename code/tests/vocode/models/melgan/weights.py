# This module:
# 1. Verifies the provenance record of the Seungwon Park MelGAN release:
#    the author source, the HTTP retrieval route, the recorded SHA-256
#    identity, the local layout, and the serialization contract
# 2. Verifies the loader guards reachable without the author binary: a
#    foreign module type, a release path that is not a readable file, and
#    a content hash that disagrees with the recorded identity
#
# Design decisions:
# - No release is ever retrieved: the adapter downloads only when the
#   resolved local path does not exist, so every test writes a file or a
#   directory at that path first and the HTTP route is never entered
# - The state-dictionary extraction and the upstream-to-project key
#   adaptation are not exercised here: this adapter hard-codes its
#   provenance, so no fabricated file can satisfy the recorded hash and
#   reach those paths, and they stay out of scope rather than being
#   reached through private methods; the equivalent adaptation logic is
#   covered on the HiFi-GAN adapter, which accepts an injected provenance
# - Loading a release that actually matches the project network is out of
#   scope, because it requires the retained author binary
#
# Author: Rahul Sawhney

import tempfile
import unittest
from pathlib import Path

import torch

from vocode.models.hifigan.hifigan import Hifigan, HifiganConfig
from vocode.models.melgan.melgan import Melgan, MelganConfig
from vocode.models.melgan.weights import MelganWeights
from vocode.models.vocoder import PublishedWeightProvenance


class MelganWeightsProvenanceTest(unittest.TestCase):
    # Verifies the release record identifies the Seungwon Park LJSpeech
    # checkpoint by source, retrieval route, hash, and local layout.
    def setUp(self) -> None:
        # Reads the hard-coded release record; nothing is retrieved or loaded.
        self._adapter: MelganWeights = MelganWeights()
        self._provenance: PublishedWeightProvenance = self._adapter.provenance

    def test_record_anchors_the_melgan_architecture(self) -> None:
        # The adapter is wired to exactly one architecture of the vocabulary.
        self.assertEqual(self._provenance.architecture_name, "melgan")
        self.assertEqual(self._provenance.variant_name, "seungwon_nvidia_tacotron2_lj11_epoch6400")

    def test_record_names_the_author_repository(self) -> None:
        # Provenance points back at the author source, not a mirror.
        self.assertEqual(self._provenance.author_source_uri, "https://github.com/seungwonpark/melgan")

    def test_record_retrieves_over_http_without_a_nested_filename(self) -> None:
        # The release is a single asset addressed by its own URL.
        self.assertEqual(self._provenance.retrieval_kind, "http")
        self.assertIsNone(self._provenance.retrieval_filename)
        self.assertTrue(self._provenance.retrieval_uri.endswith("nvidia_tacotron2_LJ11_epoch6400.pt"))

    def test_record_carries_a_full_content_hash(self) -> None:
        # A release without a full hash is not traceable to exact bytes.
        self.assertEqual(len(self._provenance.expected_sha256), 64)
        self.assertEqual(self._provenance.expected_sha256, self._provenance.expected_sha256.lower())

    def test_record_is_a_checkpoint_keyed_at_the_generator_entry(self) -> None:
        # The author payload nests the generator state under model_g.
        self.assertEqual(self._provenance.serialization, "checkpoint_dict")
        self.assertEqual(self._provenance.state_dict_key, "model_g")

    def test_local_path_resolution_composes_with_the_weights_root(self) -> None:
        # The adapter reads its release from under the configured root.
        weights_root: Path = Path("/published_weights")
        resolved_path: Path = self._provenance.resolve_local_path(weights_root)
        self.assertEqual(
            resolved_path,
            weights_root / Path("melgan_lj/nvidia_tacotron2_LJ11_epoch6400.pt")
        )

    def test_local_layout_isolates_the_release_in_its_own_directory(self) -> None:
        # Family subdirectories keep releases from colliding on disk.
        self.assertEqual(self._provenance.local_relative_path.parent, Path("melgan_lj"))


class MelganWeightsLoadGuardTest(unittest.TestCase):
    # Verifies the loader refuses a foreign module, an unreadable release
    # path, and a release whose bytes disagree with the recorded hash.
    def setUp(self) -> None:
        # Opens a temporary weights root and resolves the release path under it,
        # so every guard runs before the adapter can enter its retrieval route.
        torch.manual_seed(0)
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._weights_root: Path = Path(self._temporary_directory.name)
        self._adapter: MelganWeights = MelganWeights()
        self._release_path: Path = self._adapter.provenance.resolve_local_path(self._weights_root)

    def tearDown(self) -> None:
        # Removes the temporary weights root.
        self._temporary_directory.cleanup()

    def test_loader_rejects_a_module_of_another_family(self) -> None:
        # MelGAN weights can only be loaded onto a MelGAN module.
        foreign_module: Hifigan = Hifigan(HifiganConfig.v3())
        with self.assertRaisesRegex(TypeError, "require Melgan"):
            self._adapter.load(foreign_module, self._weights_root)

    def test_foreign_module_rejection_precedes_any_retrieval(self) -> None:
        # The type guard runs before the release path is even resolved.
        foreign_module: Hifigan = Hifigan(HifiganConfig.v3())
        with self.assertRaises(TypeError):
            self._adapter.load(foreign_module, self._weights_root)
        self.assertFalse(self._release_path.exists())

    def test_loader_rejects_a_release_path_that_is_not_a_file(self) -> None:
        # A directory at the release path is not a readable release.
        self._release_path.mkdir(parents=True, exist_ok=True)
        module: Melgan = Melgan(MelganConfig.seungwon())
        with self.assertRaisesRegex(FileNotFoundError, "missing"):
            self._adapter.load(module, self._weights_root)

    def test_loader_rejects_a_release_whose_hash_disagrees(self) -> None:
        # A substituted release must never masquerade as the reference.
        self._release_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_g": {}}, self._release_path)
        module: Melgan = Melgan(MelganConfig.seungwon())
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self._adapter.load(module, self._weights_root)

    def test_hash_rejection_reports_the_expected_digest(self) -> None:
        # The failure carries enough context to identify the bad cache entry.
        self._release_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_g": {}}, self._release_path)
        module: Melgan = Melgan(MelganConfig.seungwon())
        with self.assertRaisesRegex(ValueError, self._adapter.provenance.expected_sha256):
            self._adapter.load(module, self._weights_root)

    def test_hash_rejection_leaves_the_generator_untouched(self) -> None:
        # A refused release must not partially overwrite the network.
        self._release_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_g": {}}, self._release_path)
        module: Melgan = Melgan(MelganConfig.seungwon())
        parameter_name: str = "_network.1.bias"
        original_values: torch.Tensor = module.network.state_dict()[parameter_name].clone()
        with self.assertRaises(ValueError):
            self._adapter.load(module, self._weights_root)
        self.assertTrue(
            bool(torch.equal(module.network.state_dict()[parameter_name], original_values))
        )


if __name__ == "__main__":
    unittest.main()
