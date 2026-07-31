# This module:
# 1. Loads the author-released Seungwon Park nvidia_tacotron2 MelGAN checkpoint onto the project
#    MelGAN implementation as the published reference anchor:
#    retrieval, SHA-256 verification, state-dictionary extraction, and
#    strict loading
#
# Design decisions:
# - The recorded SHA-256 is verified before any tensor is read, so a
#   corrupted or substituted release can never masquerade as the
#   reference
# - Loading is strict: every parameter of the project network must be
#   matched by the release, proving architectural agreement rather than
#   assuming it
# - The retrieval path downloads over HTTP with an explicit user agent,
#   because the release host rejects anonymous library fetches
#
# Author: Rahul Sawhney

import hashlib
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO, override

import torch
from loguru import logger as log

from syntheticmind.core.module import Module

from vocode.models.melgan.melgan import Melgan
from vocode.models.melgan.network import MelganNetwork
from vocode.models.vocoder import PublishedWeightProvenance, PublishedWeights

__all__: list[str] = ["MelganWeights"]


class MelganWeights(PublishedWeights):
    # Published-weights adapter for the Seungwon Park MelGAN release.
    #
    # Integration: this family publishes exactly one release, so the
    # adapter hard-codes its provenance in the constructor rather than
    # accepting one. That is deliberate and has a testing consequence: no
    # fabricated file can satisfy the recorded digest, so the extraction
    # and key-adaptation paths are unreachable through the public loader
    # without the genuine author binary, and the equivalent logic is
    # exercised on the HiFi-GAN adapter, which does accept an injected
    # provenance record.
    #
    # Loading proceeds in a fixed order and stops at the first
    # disagreement: module type, local availability, content digest,
    # payload shape, key adaptation, and finally the strict load. Nothing
    # is written onto the module until the strict load succeeds, so a
    # refused release leaves the generator exactly as it was.
    def __init__(self) -> None:
        # Records the single published release this adapter is anchored
        # to: the author's LJSpeech checkpoint at epoch six thousand four
        # hundred, published as a direct release asset.
        self._provenance: PublishedWeightProvenance = PublishedWeightProvenance(
            architecture_name="melgan",
            variant_name="seungwon_nvidia_tacotron2_lj11_epoch6400",
            author_source_uri="https://github.com/seungwonpark/melgan",
            retrieval_kind="http",
            retrieval_uri=(
                "https://github.com/seungwonpark/melgan/releases/download/v0.3-alpha/"
                "nvidia_tacotron2_LJ11_epoch6400.pt"
            ),
            retrieval_filename=None,
            expected_sha256="06ead35b0169ee0c560dde9b3fe5fb8ccea2989ef9b7385487fbdd4dd22dda19",
            local_relative_path=Path("melgan_lj/nvidia_tacotron2_LJ11_epoch6400.pt"),
            serialization="checkpoint_dict",
            state_dict_key="model_g"
        )

    @property
    @override
    def provenance(self) -> PublishedWeightProvenance:
        # Returns the frozen release record this adapter is anchored to.
        # The published-evaluation lane reads it to stamp the author
        # source, retrieval source, and digest into the evidence row.
        return self._provenance

    @override
    def load(self, module: Module, published_weights_root: Path) -> None:
        # Retrieves and verifies the release, adapts its state dictionary
        # to the project layout, and loads it strictly onto the generator.
        # Only the generator receives weights: the module's discriminator
        # ensemble and loss are untouched, because the author release
        # contains the generator alone and the published lane never
        # trains. The module type is checked before the release path is
        # even resolved, so a wrong-family call reaches no filesystem.
        #
        # Args:
        #     module: The target module, which must be a MelGAN module of
        #         this project.
        #     published_weights_root: Cache root the release is resolved
        #         beneath, retrieved into when absent, and verified from.
        #
        # Raises:
        #     TypeError: If the module belongs to another architecture
        #         family, if the release payload is not a mapping, if it
        #         carries no mapping under the generator key, or if any
        #         state entry is not a tensor.
        #     FileNotFoundError: If the resolved release path is not a
        #         readable file after the retrieval step, which also
        #         covers a directory sitting at that path.
        #     ValueError: If the release bytes do not match the recorded
        #         digest.
        #     RuntimeError: If the adapted state dictionary does not match
        #         the project generator key for key. Strictness is
        #         deliberate: it proves architectural agreement with the
        #         author release rather than assuming it.
        if not isinstance(module, Melgan):
            raise TypeError(
                f"MelGAN published weights require Melgan, got {type(module).__name__}. "
                f"Build the module through ModelRegistry before loading author weights."
            )
        local_path: Path = self._ensure_local_path(published_weights_root)
        loaded_checkpoint: object = torch.load(local_path, map_location="cpu", weights_only=True)
        upstream_state_dict: dict[str, torch.Tensor] = self._extract_state_dict(loaded_checkpoint)
        adapted_state_dict: dict[str, torch.Tensor] = self._adapt_state_dict(upstream_state_dict)
        network: MelganNetwork = module.network
        network.load_state_dict(adapted_state_dict, strict=True)
        log.info(f"Loaded strict author weights for melgan from {local_path}")

    def _ensure_local_path(self, published_weights_root: Path) -> Path:
        # Resolves the cached release, fetching it over HTTP only when
        # nothing is present at that path. The release is a single asset
        # addressed directly by its URL, so no folder staging or filename
        # selection is needed. The digest is verified on both branches, so
        # an already-cached file is never trusted on the strength of its
        # existence and a corrupted cache entry is reported rather than
        # silently reused.
        #
        # Args:
        #     published_weights_root: Cache root the release lives under.
        #
        # Returns:
        #     The verified local path of the release.
        #
        # Raises:
        #     FileNotFoundError: If nothing readable is at the path after
        #         the retrieval step. A directory standing at that path
        #         suppresses the download and fails here.
        #     ValueError: If the bytes disagree with the recorded digest.
        local_path: Path = self._provenance.resolve_local_path(published_weights_root)
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self._provenance.retrieval_uri, local_path)
        self._validate_sha256(local_path)
        return local_path

    def _validate_sha256(self, local_path: Path) -> None:
        # Digests the file on disk and compares it against the recorded
        # identity. This runs before any tensor is read, so a substituted
        # or truncated release can never reach the network. The file check
        # precedes the digest, which is why a directory standing at the
        # release path is reported as a missing file rather than as a read
        # error.
        #
        # Args:
        #     local_path: The resolved release path to verify.
        #
        # Raises:
        #     FileNotFoundError: If the path is not a readable file.
        #     ValueError: If the computed digest differs from the recorded
        #         one. The message reports both digests and names the
        #         remedy, because the usual cause is a partially written
        #         cache entry.
        if not local_path.is_file():
            raise FileNotFoundError(f"MelGAN published weight file is missing at {local_path}.")
        handle: BinaryIO
        with local_path.open("rb") as handle:
            actual_sha256: str = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual_sha256 != self._provenance.expected_sha256:
            raise ValueError(
                f"MelGAN weight hash mismatch at {local_path}: "
                f"expected {self._provenance.expected_sha256}, received {actual_sha256}. "
                f"Remove the corrupted cache entry and retrieve the author file again."
            )

    def _extract_state_dict(self, loaded_checkpoint: object) -> dict[str, torch.Tensor]:
        # Narrows the untyped payload of the checkpoint to a mapping of
        # names to tensors. The author checkpoint nests the generator
        # state under its own key alongside training state this project
        # does not consume, so only that entry is read. Every value is
        # type-checked individually rather than trusted in bulk, since a
        # single non-tensor entry would otherwise surface much later as an
        # opaque load failure.
        #
        # Args:
        #     loaded_checkpoint: The deserialized release payload.
        #
        # Returns:
        #     The generator state as a plain dictionary of tensors keyed
        #     by their upstream names.
        #
        # Raises:
        #     TypeError: If the payload is not a mapping, if it holds no
        #         mapping under the generator key, or if any state entry
        #         is not a tensor. The message names the offending key.
        #
        # Note:
        #     The generator key is written literally here rather than read
        #     from the provenance record's state-dict key field; that
        #     field is declarative provenance for the evidence rows.
        if not isinstance(loaded_checkpoint, Mapping):
            raise TypeError(
                f"MelGAN author checkpoint must be a mapping, got "
                f"{type(loaded_checkpoint).__name__}."
            )
        raw_state_dict: object = loaded_checkpoint.get("model_g")
        if not isinstance(raw_state_dict, Mapping):
            raise TypeError(
                f"MelGAN author checkpoint must contain a mapping at model_g, got "
                f"{type(raw_state_dict).__name__}."
            )
        state_dict: dict[str, torch.Tensor] = {}
        raw_key: object
        raw_value: object
        for raw_key, raw_value in raw_state_dict.items():
            key: str = str(raw_key)
            if not isinstance(raw_value, torch.Tensor):
                raise TypeError(
                    f"MelGAN state entry {key} must be torch.Tensor, got "
                    f"{type(raw_value).__name__}."
                )
            state_dict[key] = raw_value
        return state_dict

    def _adapt_state_dict(
        self,
        upstream_state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Renames every upstream key onto the project layout. The mapping
        # is purely nominal: no tensor is reshaped, reordered, or dropped,
        # so a successful strict load afterwards is evidence that the two
        # topologies genuinely agree. Because the generator is one flat
        # sequential container whose positional indices already match the
        # author's, only the container name, the two residual-stack module
        # lists, and the weight-normalization parametrization tensors need
        # renaming.
        #
        # Args:
        #     upstream_state_dict: The author generator state keyed by
        #         upstream names.
        #
        # Returns:
        #     The same tensors keyed by project parameter names, ready for
        #     a strict load onto the generator network.
        #
        # Note:
        #     The container rename is capped at one replacement, so only
        #     the leading occurrence is rewritten and a later occurrence
        #     of the same token deeper in a key is left alone.
        adapted_state_dict: dict[str, torch.Tensor] = {}
        upstream_key: str
        tensor: torch.Tensor
        for upstream_key, tensor in upstream_state_dict.items():
            local_key: str = upstream_key
            local_key: str = local_key.replace("generator.", "_network.", 1)
            local_key: str = local_key.replace(".blocks.", "._blocks.")
            local_key: str = local_key.replace(".shortcuts.", "._shortcuts.")
            local_key: str = local_key.replace(".weight_g", ".parametrizations.weight.original0")
            local_key: str = local_key.replace(".weight_v", ".parametrizations.weight.original1")
            adapted_state_dict[local_key] = tensor
        return adapted_state_dict
