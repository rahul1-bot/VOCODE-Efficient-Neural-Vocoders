# This module:
# 1. Verifies the closed architecture-name vocabulary of the study against
#    the registered architectures
# 2. Verifies the published-weight provenance record: the mandatory
#    SHA-256 field, the closed retrieval and serialization vocabularies,
#    strict typing, immutability, and local-path resolution
# 3. Verifies that the real project models and the real published-weight
#    adapters satisfy the two structural protocols of the model layer
#
# Design decisions:
# - Protocol conformance is asserted structurally rather than through
#   isinstance, because both protocols are deliberately not
#   runtime-checkable; the surface is verified by attribute type and by
#   calling the synthesis method
# - Provenance is exercised as a pure record: local-path resolution is a
#   path computation with no filesystem access, and no adapter is ever
#   asked to retrieve, verify, or load a release here
# - Synthesis conformance runs one seeded forward on a sixteen-frame mel
#   through the smallest reference recipe of each family, which keeps the
#   protocol check honest without approaching a training-scale cost
#
# Author: Rahul Sawhney

import unittest
from pathlib import Path
from typing import get_args, get_protocol_members

import torch
from pydantic import ValidationError
from torch import nn

from vocode.models.hifigan.hifigan import Hifigan, HifiganConfig
from vocode.models.hifigan.weights import HifiganWeights
from vocode.models.melgan.melgan import Melgan, MelganConfig
from vocode.models.melgan.weights import MelganWeights
from vocode.models.registry import ModelRegistry
from vocode.models.vocoder import ArchitectureName, PublishedWeightProvenance, PublishedWeights, Vocoder
from vocode.transforms.mel import MelConfig


class ProvenanceFieldPayload:
    # Builds valid published-weight provenance payloads and single-field
    # mutations of them, so validation tests state only the field under test.
    def __init__(self) -> None:
        # Records one complete MelGAN release payload as the accepted baseline.
        self._fields: dict[str, str | Path | None] = {
            "architecture_name": "melgan",
            "variant_name": "seungwon_reference",
            "author_source_uri": "https://github.com/seungwonpark/melgan",
            "retrieval_kind": "http",
            "retrieval_uri": "https://example.invalid/release.pt",
            "retrieval_filename": None,
            "expected_sha256": "0" * 64,
            "local_relative_path": Path("melgan_lj/release.pt"),
            "serialization": "checkpoint_dict",
            "state_dict_key": "model_g"
        }

    def valid(self) -> dict[str, str | Path | None]:
        # Returns a copy of the accepted payload.
        return dict(self._fields)

    def with_field(self, field_name: str, field_value: object) -> dict[str, object]:
        # Returns the accepted payload with exactly one field replaced.
        mutated_fields: dict[str, object] = dict(self._fields)
        mutated_fields[field_name] = field_value
        return mutated_fields


class ArchitectureVocabularyTest(unittest.TestCase):
    # Verifies the architecture-name literal is the closed thirteen-name
    # vocabulary the registry implements.
    def test_vocabulary_holds_thirteen_architecture_names(self) -> None:
        # The study vocabulary is fixed at thirteen architectures.
        vocabulary_names: tuple[str, ...] = get_args(ArchitectureName.__value__)
        self.assertEqual(len(vocabulary_names), 13)

    def test_vocabulary_matches_the_registered_architectures(self) -> None:
        # Adding a name is a registry-level change, never a runtime string.
        vocabulary_names: tuple[str, ...] = get_args(ArchitectureName.__value__)
        registered_names: tuple[str, ...] = tuple(
            record.architecture_name for record in ModelRegistry().all_records()
        )
        self.assertEqual(vocabulary_names, registered_names)

    def test_vocabulary_covers_the_three_hifigan_widths_and_melgan(self) -> None:
        # The families under test here are part of the closed vocabulary.
        vocabulary_names: tuple[str, ...] = get_args(ArchitectureName.__value__)
        expected_name: str
        for expected_name in ("hifigan_v1", "hifigan_v2", "hifigan_v3", "melgan"):
            self.assertIn(expected_name, vocabulary_names)


class PublishedWeightProvenanceValidationTest(unittest.TestCase):
    # Verifies the frozen provenance record enforces its mandatory
    # SHA-256 field, closed vocabularies, strict types, and immutability.
    def setUp(self) -> None:
        # Builds the payload source the single-field rejection cases mutate.
        self._payload: ProvenanceFieldPayload = ProvenanceFieldPayload()

    def test_provenance_accepts_a_complete_release_record(self) -> None:
        # A complete payload constructs and preserves every release field.
        provenance: PublishedWeightProvenance = PublishedWeightProvenance(**self._payload.valid())
        self.assertEqual(provenance.architecture_name, "melgan")
        self.assertEqual(provenance.retrieval_kind, "http")
        self.assertEqual(provenance.serialization, "checkpoint_dict")
        self.assertEqual(provenance.state_dict_key, "model_g")
        self.assertEqual(len(provenance.expected_sha256), 64)

    def test_provenance_requires_the_content_hash(self) -> None:
        # A release without a recorded hash is not traceable to exact bytes.
        incomplete_fields: dict[str, str | Path | None] = self._payload.valid()
        del incomplete_fields["expected_sha256"]
        with self.assertRaises(ValidationError):
            PublishedWeightProvenance(**incomplete_fields)

    def test_provenance_rejects_a_null_content_hash(self) -> None:
        # The hash field is mandatory rather than optional.
        with self.assertRaises(ValidationError):
            PublishedWeightProvenance(**self._payload.with_field("expected_sha256", None))

    def test_provenance_rejects_an_unregistered_architecture_name(self) -> None:
        # Provenance can only anchor an architecture in the closed vocabulary.
        with self.assertRaises(ValidationError):
            PublishedWeightProvenance(**self._payload.with_field("architecture_name", "griffinlim"))

    def test_provenance_rejects_an_unknown_retrieval_kind(self) -> None:
        # Retrieval methods are a closed set of audited transports.
        with self.assertRaises(ValidationError):
            PublishedWeightProvenance(**self._payload.with_field("retrieval_kind", "ftp"))

    def test_provenance_rejects_an_unknown_serialization_layout(self) -> None:
        # Serialization layouts are a closed set the loaders implement.
        with self.assertRaises(ValidationError):
            PublishedWeightProvenance(**self._payload.with_field("serialization", "safetensors"))

    def test_provenance_rejects_a_string_local_path(self) -> None:
        # strict=True keeps the local layout a pathlib path.
        with self.assertRaises(ValidationError):
            PublishedWeightProvenance(**self._payload.with_field("local_relative_path", "melgan_lj/release.pt"))

    def test_provenance_rejects_an_extra_field(self) -> None:
        # extra="forbid" blocks undeclared release metadata.
        with self.assertRaises(ValidationError):
            PublishedWeightProvenance(**self._payload.with_field("license_name", "MIT"))

    def test_provenance_accepts_an_absent_state_dict_key(self) -> None:
        # A flat state-dictionary release carries no nested key.
        flat_fields: dict[str, object] = self._payload.with_field("state_dict_key", None)
        flat_fields["serialization"] = "pytorch_state_dict"
        provenance: PublishedWeightProvenance = PublishedWeightProvenance(**flat_fields)
        self.assertIsNone(provenance.state_dict_key)

    def test_provenance_rejects_mutation_after_construction(self) -> None:
        # frozen=True keeps a release identity fixed once recorded.
        provenance: PublishedWeightProvenance = PublishedWeightProvenance(**self._payload.valid())
        with self.assertRaises(ValidationError):
            provenance.expected_sha256: str = "1" * 64


class PublishedWeightProvenanceLocalPathTest(unittest.TestCase):
    # Verifies local-path resolution composes the weights root with the
    # recorded relative layout without touching the filesystem.
    def setUp(self) -> None:
        # Builds one recorded release whose relative layout the resolutions compose against.
        self._provenance: PublishedWeightProvenance = PublishedWeightProvenance(
            **ProvenanceFieldPayload().valid()
        )

    def test_resolution_appends_the_relative_layout_to_the_root(self) -> None:
        # The release lives under the weights root at its recorded layout.
        weights_root: Path = Path("/published_weights")
        resolved_path: Path = self._provenance.resolve_local_path(weights_root)
        self.assertEqual(resolved_path, weights_root / Path("melgan_lj/release.pt"))

    def test_resolution_preserves_the_nested_release_directory(self) -> None:
        # Family subdirectories survive resolution so releases never collide.
        resolved_path: Path = self._provenance.resolve_local_path(Path("/published_weights"))
        self.assertEqual(resolved_path.name, "release.pt")
        self.assertEqual(resolved_path.parent.name, "melgan_lj")

    def test_resolution_accepts_a_relative_weights_root(self) -> None:
        # A relative root resolves to a relative release path.
        resolved_path: Path = self._provenance.resolve_local_path(Path("artifacts/weights"))
        self.assertEqual(resolved_path, Path("artifacts/weights/melgan_lj/release.pt"))

    def test_resolution_performs_no_filesystem_access(self) -> None:
        # Resolution is a pure path computation on a nonexistent root.
        resolved_path: Path = self._provenance.resolve_local_path(Path("/nonexistent_weights_root"))
        self.assertFalse(resolved_path.exists())


class PublishedWeightsProtocolConformanceTest(unittest.TestCase):
    # Verifies the published-weights contract surface and that the real
    # HiFi-GAN and MelGAN adapters satisfy it.
    def setUp(self) -> None:
        # Builds one real adapter per family; no release is retrieved or loaded.
        self._hifigan_adapter: HifiganWeights = HifiganWeights.v1()
        self._melgan_adapter: MelganWeights = MelganWeights()

    def test_contract_declares_provenance_and_load(self) -> None:
        # The adapter contract is exactly a provenance record and a loader.
        self.assertEqual(get_protocol_members(PublishedWeights), frozenset({"provenance", "load"}))

    def test_adapters_expose_a_provenance_record(self) -> None:
        # Provenance travels with the weights into the evidence rows.
        adapter: PublishedWeights
        for adapter in (self._hifigan_adapter, self._melgan_adapter):
            self.assertIsInstance(adapter.provenance, PublishedWeightProvenance)

    def test_adapters_expose_a_load_surface(self) -> None:
        # Every adapter loads a release onto a harness module.
        adapter: PublishedWeights
        for adapter in (self._hifigan_adapter, self._melgan_adapter):
            self.assertTrue(callable(adapter.load))

    def test_adapters_declare_the_protocol_in_their_inheritance(self) -> None:
        # Explicit protocol inheritance makes the contract discoverable.
        adapter_type: type[PublishedWeights]
        for adapter_type in (HifiganWeights, MelganWeights):
            self.assertIn(PublishedWeights, adapter_type.__mro__)

    def test_protocol_is_not_runtime_checkable(self) -> None:
        # The contract is structural documentation, not a runtime gate.
        with self.assertRaises(TypeError):
            isinstance(self._melgan_adapter, PublishedWeights)


class VocoderProtocolConformanceTest(unittest.TestCase):
    # Verifies the real HiFi-GAN and MelGAN modules satisfy the synthesis
    # contract the metric and profiling components consume.
    def setUp(self) -> None:
        # Builds the smallest reference recipe of each family and one short mel.
        torch.manual_seed(0)
        self._hifigan: Hifigan = Hifigan(HifiganConfig.v3()).eval()
        self._melgan: Melgan = Melgan(MelganConfig.seungwon()).eval()
        self._mel: torch.Tensor = torch.randn(1, 80, 16)

    def test_contract_declares_the_synthesis_surface(self) -> None:
        # The synthesis contract is a network, two mel protocols, and synthesize.
        self.assertEqual(
            get_protocol_members(Vocoder),
            frozenset({"network", "mel_protocol", "metric_mel_protocol", "synthesize"})
        )

    def test_models_expose_a_synthesis_network(self) -> None:
        # The network member is the live torch module of the family.
        vocoder: Vocoder
        for vocoder in (self._hifigan, self._melgan):
            self.assertIsInstance(vocoder.network, nn.Module)

    def test_models_expose_both_mel_protocols(self) -> None:
        # Conditioning and metric protocols are both frozen mel records.
        vocoder: Vocoder
        for vocoder in (self._hifigan, self._melgan):
            self.assertIsInstance(vocoder.mel_protocol, MelConfig)
            self.assertIsInstance(vocoder.metric_mel_protocol, MelConfig)

    def test_hifigan_separates_conditioning_from_metric_protocol(self) -> None:
        # HiFi-GAN conditions band-limited and measures full-band.
        self.assertEqual(self._hifigan.mel_protocol.fmax, 8000.0)
        self.assertIsNone(self._hifigan.metric_mel_protocol.fmax)
        self.assertNotEqual(self._hifigan.mel_protocol, self._hifigan.metric_mel_protocol)

    def test_melgan_measures_with_its_conditioning_protocol(self) -> None:
        # MelGAN declares one protocol for both roles.
        self.assertEqual(self._melgan.mel_protocol, self._melgan.metric_mel_protocol)

    def test_synthesis_returns_a_finite_waveform_for_both_families(self) -> None:
        # Mel-to-waveform synthesis produces the upsampled waveform batch.
        vocoder: Vocoder
        for vocoder in (self._hifigan, self._melgan):
            with torch.no_grad():
                waveform: torch.Tensor = vocoder.synthesize(self._mel)
            self.assertEqual(waveform.shape[0], 1)
            self.assertEqual(waveform.shape[-1], 16 * 256)
            self.assertEqual(waveform.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(waveform).all()))

    def test_protocol_is_not_runtime_checkable(self) -> None:
        # The contract is structural documentation, not a runtime gate.
        with self.assertRaises(TypeError):
            isinstance(self._melgan, Vocoder)


if __name__ == "__main__":
    unittest.main()
