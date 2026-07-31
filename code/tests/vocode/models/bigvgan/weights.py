# This module:
# 1. Verifies the BigVGAN published-weight provenance record: the release
#    identity, the retrieval contract, the recorded digest, and the local
#    path resolution
# 2. Verifies the loading guards: the module-type refusal, the missing-file
#    refusal, and the digest-mismatch refusal, all raised before any tensor
#    is read
# 3. Verifies the checkpoint extraction and the upstream-to-local key
#    adaptation that renames the weight-normalization components
#
# Design decisions:
# - No author binary is ever retrieved and no network call is ever made:
#   every fixture is a fabricated file in a temporary directory, so the
#   successful retrieval-and-strict-load path is deliberately out of scope
#   and remains untested here
# - The digest guard runs before extraction, so the checkpoint-structure
#   refusals cannot be reached through load with a fabricated file; they are
#   exercised on the extraction and adaptation helpers directly
# - The digest-mismatch fixture is a short text file: any content other than
#   the released bytes reproduces the mismatch the guard exists to catch
#
# Author: Rahul Sawhney

import tempfile
import unittest
from pathlib import Path

import torch

from vocode.metrics.pesq import PesqConfig
from vocode.metrics.stoi import StoiConfig
from vocode.models.bigvgan.bigvgan import Bigvgan, BigvganConfig
from vocode.models.bigvgan.weights import BigvganWeights
from vocode.models.vocoder import PublishedWeightProvenance
from vocode.transforms.mel import MelConfig


class MiniatureModuleRecipe:
    # Builds the miniature BigVGAN module used as the load target of the guard assertions.
    # The guards under test refuse before any parameter is touched, so the
    # target's topology is irrelevant to them and only its type matters; a
    # reduced module keeps every refusal assertion cheap.
    def build(self) -> Bigvgan:
        # Returns a constructed module small enough to be the target of every refusal path.
        # It would not survive a successful strict load, which is consistent
        # with this file's scope: the retrieval-and-load path is deliberately
        # untested here because it needs the genuine release bytes.
        #
        # Returns:
        #     A valid Bigvgan instance that satisfies the loader's type gate.
        configuration: BigvganConfig = BigvganConfig(
            input_mel_channels=100,
            upsample_initial_channel=16,
            upsample_rates=(2, 2),
            upsample_kernel_sizes=(4, 4),
            resblock_kernel_sizes=(3,),
            resblock_dilation_sizes=((1, 3),),
            resblock="1",
            activation="snakebeta",
            snake_logscale=True,
            resolutions=((32, 8, 32), (64, 16, 64), (16, 4, 16)),
            mpd_reshapes=(2,),
            use_spectral_norm=False,
            discriminator_channel_multiplier=0.125,
            mel_protocol=MelConfig.bigvgan_nvidia_base_24khz_100band(),
            reconstruction_mel_protocol=MelConfig.bigvgan_nvidia_base_reconstruction_24khz_100band(),
            pesq_protocol=PesqConfig(),
            stoi_protocol=StoiConfig()
        )
        return Bigvgan(configuration)


class BigvganWeightProvenanceTest(unittest.TestCase):
    # Verifies the recorded release identity and retrieval contract of the published checkpoint.
    def setUp(self) -> None:
        # Builds the weight adapter and reads its provenance record.
        self._weights: BigvganWeights = BigvganWeights()
        self._provenance: PublishedWeightProvenance = self._weights.provenance

    def test_release_identity_names_the_published_base_variant(self) -> None:
        # The anchor is the NVIDIA base 24 kHz 100-band release of the BigVGAN family.
        self.assertEqual(self._provenance.architecture_name, "bigvgan")
        self.assertEqual(self._provenance.variant_name, "nvidia_base_24khz_100band")
        self.assertEqual(self._provenance.author_source_uri, "https://github.com/NVIDIA/BigVGAN")

    def test_retrieval_contract_names_the_hugging_face_file(self) -> None:
        # Retrieval goes through the Hugging Face file API at the recorded repository and filename.
        self.assertEqual(self._provenance.retrieval_kind, "huggingface_file")
        self.assertEqual(self._provenance.retrieval_uri, "nvidia/bigvgan_base_24khz_100band")
        self.assertEqual(self._provenance.retrieval_filename, "bigvgan_generator.pt")

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
        self.assertEqual(
            self._provenance.resolve_local_path(root),
            root / "bigvgan_base_24khz_100band" / "bigvgan_generator.pt"
        )

    def test_provenance_record_is_immutable(self) -> None:
        # A frozen record cannot be edited into agreement with a substituted release.
        with self.assertRaises(ValueError):
            self._provenance.expected_sha256 = "0" * 64


class BigvganWeightLoadGuardTest(unittest.TestCase):
    # Verifies that loading refuses wrong modules, missing files, and mismatched digests.
    def setUp(self) -> None:
        # Opens a temporary weights root and resolves the release path the guards inspect.
        torch.manual_seed(1234)
        self._weights: BigvganWeights = BigvganWeights()
        self._module: Bigvgan = MiniatureModuleRecipe().build()
        self._temporary_root: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_root.name)
        self._local_path: Path = self._weights.provenance.resolve_local_path(self._root)

    def tearDown(self) -> None:
        # Removes the fabricated release tree.
        self._temporary_root.cleanup()

    def test_loading_onto_a_foreign_module_is_refused(self) -> None:
        # The type guard runs before retrieval, so a foreign module never triggers a download.
        with self.assertRaisesRegex(TypeError, "require Bigvgan"):
            self._weights.load(torch.nn.Linear(2, 2), self._root)

    def test_missing_release_file_is_refused(self) -> None:
        # A resolved path that is not a readable file fails before any tensor is read.
        self._local_path.mkdir(parents=True, exist_ok=True)
        with self.assertRaisesRegex(FileNotFoundError, "weight file is missing"):
            self._weights.load(self._module, self._root)

    def test_digest_mismatch_is_refused(self) -> None:
        # A substituted or corrupted release cannot masquerade as the recorded anchor.
        self._local_path.parent.mkdir(parents=True, exist_ok=True)
        self._local_path.write_bytes(b"not the published release")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self._weights.load(self._module, self._root)

    def test_digest_mismatch_message_names_the_expected_digest(self) -> None:
        # The failure message carries both digests so the corrupted cache entry is identifiable.
        self._local_path.parent.mkdir(parents=True, exist_ok=True)
        self._local_path.write_bytes(b"not the published release")
        with self.assertRaises(ValueError) as failure:
            self._weights.load(self._module, self._root)
        self.assertIn(self._weights.provenance.expected_sha256, str(failure.exception))


class BigvganCheckpointExtractionTest(unittest.TestCase):
    # Verifies the checkpoint-structure guards and the tensor extraction of the generator entry.
    def setUp(self) -> None:
        # Builds the weight adapter whose extraction helper is under test.
        self._weights: BigvganWeights = BigvganWeights()

    def test_non_mapping_checkpoint_is_refused(self) -> None:
        # A checkpoint that is not a mapping cannot carry a keyed generator state.
        with self.assertRaisesRegex(TypeError, "must be a mapping"):
            self._weights._extract_state_dict([1, 2, 3])

    def test_checkpoint_without_generator_entry_is_refused(self) -> None:
        # The recorded serialization layout demands a mapping at the generator key.
        with self.assertRaisesRegex(TypeError, "mapping at generator"):
            self._weights._extract_state_dict({"model": {}})

    def test_non_tensor_state_entry_is_refused(self) -> None:
        # Every extracted entry must be a tensor, and the offending key is named.
        with self.assertRaisesRegex(TypeError, "conv_pre.bias"):
            self._weights._extract_state_dict({"generator": {"conv_pre.bias": 1.0}})

    def test_generator_entry_is_extracted_as_tensors(self) -> None:
        # A well-formed release yields the flat key-to-tensor mapping the adapter renames.
        extracted: dict[str, torch.Tensor] = self._weights._extract_state_dict(
            {"generator": {"conv_pre.bias": torch.zeros(2)}}
        )
        self.assertEqual(list(extracted.keys()), ["conv_pre.bias"])
        self.assertTrue(torch.equal(extracted["conv_pre.bias"], torch.zeros(2)))


class BigvganStateKeyAdaptationTest(unittest.TestCase):
    # Verifies the upstream-to-local renaming of the weight-normalization state components.
    def setUp(self) -> None:
        # Builds the adapter and one upstream-named state fragment covering both key kinds.
        torch.manual_seed(1234)
        self._weights: BigvganWeights = BigvganWeights()
        self._upstream: dict[str, torch.Tensor] = {
            "conv_pre.weight_g": torch.zeros(1),
            "conv_pre.weight_v": torch.ones(1),
            "conv_pre.bias": torch.full((1,), 2.0)
        }

    def test_weight_magnitude_component_is_renamed(self) -> None:
        # The upstream magnitude tensor becomes the first parametrization component.
        adapted: dict[str, torch.Tensor] = self._weights._adapt_state_dict(self._upstream)
        self.assertIn("conv_pre.parametrizations.weight.original0", adapted)

    def test_weight_direction_component_is_renamed(self) -> None:
        # The upstream direction tensor becomes the second parametrization component.
        adapted: dict[str, torch.Tensor] = self._weights._adapt_state_dict(self._upstream)
        self.assertIn("conv_pre.parametrizations.weight.original1", adapted)

    def test_unparametrized_keys_are_left_untouched(self) -> None:
        # Biases carry no weight normalization and therefore keep their upstream names.
        adapted: dict[str, torch.Tensor] = self._weights._adapt_state_dict(self._upstream)
        self.assertIn("conv_pre.bias", adapted)
        self.assertEqual(len(adapted), len(self._upstream))

    def test_adaptation_preserves_the_tensor_values(self) -> None:
        # Renaming is a relabeling only; no tensor is transformed on the way in.
        adapted: dict[str, torch.Tensor] = self._weights._adapt_state_dict(self._upstream)
        self.assertTrue(
            torch.equal(adapted["conv_pre.parametrizations.weight.original1"], torch.ones(1)),
            msg="The direction component must arrive unmodified under its local name"
        )
