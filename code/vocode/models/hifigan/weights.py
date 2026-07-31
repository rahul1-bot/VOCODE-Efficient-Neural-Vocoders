# This module:
# 1. Loads the author-released jik876 HiFi-GAN checkpoints (V1, V2, V3)
#    onto the project implementation as the published reference anchors:
#    retrieval, SHA-256 verification, state-dictionary extraction, and
#    strict loading
#
# Design decisions:
# - Downloads verify the recorded SHA-256 before any tensor is read, so
#   a corrupted or substituted release can never masquerade as the
#   reference
# - Loading is strict: every parameter of the project network must be
#   matched by the release, proving architectural agreement rather than
#   assuming it
# - The provenance record (author source, retrieval source, identity)
#   travels with the weights into the evidence rows
#
# Author: Rahul Sawhney

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO, Literal, override

import torch
from loguru import logger as log

from syntheticmind.core.module import Module

from vocode.models.hifigan.hifigan import Hifigan
from vocode.models.hifigan.network import HifiganNetwork
from vocode.models.vocoder import PublishedWeightProvenance, PublishedWeights

__all__: list[str] = ["HifiganWeights"]


type HifiganArchitectureName = Literal["hifigan_v1", "hifigan_v2", "hifigan_v3"]


class HifiganWeights(PublishedWeights):
    # Published-weights adapter for the jik876 HiFi-GAN releases.
    #
    # Integration: the three released widths share one adapter class and
    # differ only in the provenance record handed to the constructor, so
    # the named factories are the intended entry points and the registry
    # calls them. The constructor is nonetheless public and takes an
    # arbitrary provenance record, which is what lets the test suite drive
    # every load path against a fabricated release whose recorded digest
    # matches its own bytes.
    #
    # Loading proceeds in a fixed order and stops at the first
    # disagreement: module type, local availability, content digest,
    # payload shape, key adaptation, and finally the strict load. Nothing
    # is written onto the module until the strict load succeeds, so a
    # refused release leaves the generator exactly as it was.
    def __init__(self, provenance: PublishedWeightProvenance) -> None:
        # Binds the release identity this adapter retrieves and loads.
        #
        # Args:
        #     provenance: The frozen release record. Its recorded digest
        #         is the only accepted identity for the bytes this adapter
        #         will load, and its relative path fixes where they are
        #         cached.
        self._provenance: PublishedWeightProvenance = provenance

    @classmethod
    def v1(cls) -> HifiganWeights:
        # Builds the adapter for the author's LJSpeech V1 generator
        # release.
        #
        # Returns:
        #     An adapter anchored to the V1 release; its digest and local
        #     layout are distinct from the V2 and V3 adapters, so the
        #     three cannot be confused on disk or after verification.
        provenance: PublishedWeightProvenance = cls._build_provenance(
            architecture_name="hifigan_v1",
            variant_name="jik876_lj_v1",
            source_filename="LJ_V1/generator_v1",
            local_relative_path=Path("hifigan_v1_lj/generator_v1"),
            expected_sha256="bb4b0cb7f9df59b8e57bb2e51a1bede57b43e9f0454863e3971c491f255505e4"
        )
        return cls(provenance)

    @classmethod
    def v2(cls) -> HifiganWeights:
        # Builds the adapter for the author's LJSpeech V2 generator
        # release, the narrow-width sibling of V1.
        #
        # Returns:
        #     An adapter anchored to the V2 release.
        provenance: PublishedWeightProvenance = cls._build_provenance(
            architecture_name="hifigan_v2",
            variant_name="jik876_lj_v2",
            source_filename="LJ_V2/generator_v2",
            local_relative_path=Path("hifigan_v2_lj/generator_v2"),
            expected_sha256="3fac378c5918fb2c102733f21eeaa8e9a4ca6cda24dbfddc55bbb947c78d562f"
        )
        return cls(provenance)

    @classmethod
    def v3(cls) -> HifiganWeights:
        # Builds the adapter for the author's LJSpeech V3 generator
        # release, which is the three-stage recipe using the lighter
        # residual block.
        #
        # Returns:
        #     An adapter anchored to the V3 release.
        provenance: PublishedWeightProvenance = cls._build_provenance(
            architecture_name="hifigan_v3",
            variant_name="jik876_lj_v3",
            source_filename="LJ_V3/generator_v3",
            local_relative_path=Path("hifigan_v3_lj/generator_v3"),
            expected_sha256="b5e89fc0c45924525b7bd0c974aaf8b55aa8e0f9115a83356632d5aa11b8a554"
        )
        return cls(provenance)

    @classmethod
    def _build_provenance(
        cls,
        architecture_name: HifiganArchitectureName,
        variant_name: str,
        source_filename: str,
        local_relative_path: Path,
        expected_sha256: str
    ) -> PublishedWeightProvenance:
        # Fills in the fields the three releases share and leaves the four
        # that distinguish them as arguments. All three are published in
        # one shared drive folder, so the retrieval URI is common and the
        # filename inside that folder is what selects a width.
        #
        # Args:
        #     architecture_name: The registered width this release
        #         anchors.
        #     variant_name: The author's own label for the release.
        #     source_filename: Path of the generator file inside the
        #         shared folder.
        #     local_relative_path: Cache layout beneath the weights root,
        #         distinct per width so the three never collide.
        #     expected_sha256: The digest the retrieved bytes must match.
        #
        # Returns:
        #     The frozen provenance record for one released width.
        return PublishedWeightProvenance(
            architecture_name=architecture_name,
            variant_name=variant_name,
            author_source_uri="https://github.com/jik876/hifi-gan",
            retrieval_kind="google_drive_folder",
            retrieval_uri="https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y",
            retrieval_filename=source_filename,
            expected_sha256=expected_sha256,
            local_relative_path=local_relative_path,
            serialization="checkpoint_dict",
            state_dict_key="generator"
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
        # Retrieves and verifies the release, adapts its state dictionary to
        # the project layout, and loads it strictly onto the generator.
        # Only the generator receives weights: the module's discriminators
        # and loss are untouched, because the author release contains the
        # generator alone and the published lane never trains.
        #
        # Args:
        #     module: The target module, which must be a HiFi-GAN module
        #         of this project. It is checked before anything is
        #         retrieved or read.
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
        #         digest, or if the retrieval filename is missing from a
        #         folder-published record.
        #     RuntimeError: If the adapted state dictionary does not match
        #         the project generator key for key. Strictness is
        #         deliberate: it proves architectural agreement with the
        #         author release rather than assuming it, so a partial
        #         match is a failure and not a warning.
        if not isinstance(module, Hifigan):
            raise TypeError(
                f"HiFi-GAN published weights require Hifigan, got {type(module).__name__}. "
                f"Build the module through ModelRegistry before loading author weights."
            )
        local_path: Path = self._ensure_local_path(published_weights_root)
        loaded_checkpoint: object = torch.load(local_path, map_location="cpu", weights_only=True)
        upstream_state_dict: dict[str, torch.Tensor] = self._extract_state_dict(loaded_checkpoint)
        adapted_state_dict: dict[str, torch.Tensor] = self._adapt_state_dict(upstream_state_dict)
        network: HifiganNetwork = module.network
        network.load_state_dict(adapted_state_dict, strict=True)
        log.info(
            f"Loaded strict author weights for {self._provenance.architecture_name} "
            f"from {local_path}"
        )

    def _ensure_local_path(self, published_weights_root: Path) -> Path:
        # Resolves the cached release, retrieving it only when nothing is
        # present at that path. The digest is verified on both branches,
        # so an already-cached file is never trusted on the strength of
        # its existence and a corrupted cache entry is reported rather
        # than silently reused.
        #
        # Args:
        #     published_weights_root: Cache root the release lives under.
        #
        # Returns:
        #     The verified local path of the release.
        #
        # Raises:
        #     FileNotFoundError: If nothing readable is at the path after
        #         the retrieval step.
        #     ValueError: If the bytes disagree with the recorded digest.
        local_path: Path = self._provenance.resolve_local_path(published_weights_root)
        if local_path.exists():
            self._validate_sha256(local_path)
            return local_path
        self._download(published_weights_root, local_path)
        self._validate_sha256(local_path)
        return local_path

    def _download(self, published_weights_root: Path, local_path: Path) -> None:
        # Retrieves the author release from the shared drive folder. The
        # folder is fetched whole into a staging directory beside the
        # cache, because the transport addresses a folder rather than a
        # single asset, and only the file named by the provenance record
        # is copied to the cache path. gdown is imported here rather than
        # at module scope so the retrieval dependency is required only on
        # the path that actually downloads.
        #
        # Args:
        #     published_weights_root: Cache root; the staging directory is
        #         created beneath it.
        #     local_path: Destination the selected file is copied to.
        #
        # Raises:
        #     ValueError: If the provenance record names no file inside
        #         the folder, which makes the selection undefined.
        #     FileNotFoundError: If the download completes without
        #         producing the named file, which indicates the published
        #         folder layout changed.
        import gdown

        source_filename: str | None = self._provenance.retrieval_filename
        if source_filename is None:
            raise ValueError(
                f"HiFi-GAN retrieval filename is missing for {self._provenance.architecture_name}."
            )
        download_root: Path = published_weights_root / ".hifigan_jik876"
        download_root.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(
            url=self._provenance.retrieval_uri,
            output=str(download_root),
            quiet=False,
            use_cookies=False
        )
        source_path: Path = download_root / source_filename
        if not source_path.is_file():
            raise FileNotFoundError(
                f"HiFi-GAN download completed without {source_path}. "
                f"Verify the jik876 Google Drive layout before retrying."
            )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.copy(local_path)

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
            raise FileNotFoundError(
                f"HiFi-GAN published weight file is missing at {local_path}."
            )
        handle: BinaryIO
        with local_path.open("rb") as handle:
            actual_sha256: str = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual_sha256 != self._provenance.expected_sha256:
            raise ValueError(
                f"HiFi-GAN weight hash mismatch at {local_path}: "
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
                f"HiFi-GAN author checkpoint must be a mapping, got "
                f"{type(loaded_checkpoint).__name__}."
            )
        raw_state_dict: object = loaded_checkpoint.get("generator")
        if not isinstance(raw_state_dict, Mapping):
            raise TypeError(
                f"HiFi-GAN author checkpoint must contain a mapping at generator, got "
                f"{type(raw_state_dict).__name__}."
            )
        state_dict: dict[str, torch.Tensor] = {}
        raw_key: object
        raw_value: object
        for raw_key, raw_value in raw_state_dict.items():
            key: str = str(raw_key)
            if not isinstance(raw_value, torch.Tensor):
                raise TypeError(
                    f"HiFi-GAN state entry {key} must be torch.Tensor, got "
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
        # topologies genuinely agree. The renames cover the generator's
        # own members, the two residual-block convolution roles, and the
        # weight-normalization parametrization, whose magnitude and
        # direction tensors the modern parametrization API stores under
        # different names than the release does.
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
        #     The substitutions are applied in sequence to a single key,
        #     so each one sees the result of the previous. The
        #     convolution-list patterns are written with their surrounding
        #     dots, which is what stops the pattern for the plain list
        #     from also matching the two numbered lists of the heavier
        #     residual block.
        adapted_state_dict: dict[str, torch.Tensor] = {}
        upstream_key: str
        tensor: torch.Tensor
        for upstream_key, tensor in upstream_state_dict.items():
            local_key: str = upstream_key
            local_key: str = local_key.replace("conv_pre.", "_pre_convolution.")
            local_key: str = local_key.replace("conv_post.", "_post_convolution.")
            local_key: str = local_key.replace("ups.", "_upsample_layers.")
            local_key: str = local_key.replace("resblocks.", "_residual_blocks.")
            local_key: str = local_key.replace(".convs1.", "._dilated_convolutions.")
            local_key: str = local_key.replace(".convs2.", "._refinement_convolutions.")
            local_key: str = local_key.replace(".convs.", "._convolutions.")
            local_key: str = local_key.replace(".weight_g", ".parametrizations.weight.original0")
            local_key: str = local_key.replace(".weight_v", ".parametrizations.weight.original1")
            adapted_state_dict[local_key] = tensor
        return adapted_state_dict
