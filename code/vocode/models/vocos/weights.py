# This module:
# 1. Loads the author-released charactr Vocos 24 kHz checkpoint onto the project
#    Vocos implementation as the published reference anchor:
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
# - The release ships weight-normalization parametrizations; extraction
#   maps them onto the project layout before the strict load
#
# Author: Rahul Sawhney

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO, override

import torch
from loguru import logger as log

from syntheticmind.core.module import Module

from vocode.models.vocoder import PublishedWeightProvenance, PublishedWeights
from vocode.models.vocos.network import VocosNetwork
from vocode.models.vocos.vocos import Vocos

__all__: list[str] = ["VocosWeights"]


class VocosWeights(PublishedWeights):
    # Published-weights adapter for the charactr Vocos 24 kHz release.
    #
    # Integration: this family publishes exactly one release, so the
    # adapter hard-codes its provenance in the constructor rather than
    # accepting one. Unlike the checkpoint-style releases of the
    # convolutional families, the payload here is a bare state dictionary
    # with no nesting key, and it carries entries the project network does
    # not own, so extraction drops a fixed set of external buffers before
    # the strict load.
    #
    # Loading proceeds in a fixed order and stops at the first
    # disagreement: module type, local availability, content digest,
    # payload shape, key adaptation, and finally the strict load. Nothing
    # is written onto the module until the strict load succeeds, so a
    # refused release leaves the generator exactly as it was.
    def __init__(self) -> None:
        # Records the single published release this adapter is anchored
        # to: the author's 24 kHz mel model, retrieved as one named file
        # from its model-hub repository.
        self._provenance: PublishedWeightProvenance = PublishedWeightProvenance(
            architecture_name="vocos",
            variant_name="charactr_mel_24khz",
            author_source_uri="https://github.com/charactr-platform/vocos",
            retrieval_kind="huggingface_file",
            retrieval_uri="charactr/vocos-mel-24khz",
            retrieval_filename="pytorch_model.bin",
            expected_sha256="97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a",
            local_relative_path=Path("vocos_mel_24khz/pytorch_model.bin"),
            serialization="pytorch_state_dict",
            state_dict_key=None
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
        # ensembles and loss are untouched, because the published lane
        # never trains.
        #
        # Args:
        #     module: The target module, which must be a Vocos module of
        #         this project. The subclassed adaptation also satisfies
        #         this check by inheritance, but its generator differs
        #         from the released one, so such a load would be refused
        #         by the strict key comparison rather than here.
        #     published_weights_root: Cache root the release is resolved
        #         beneath, retrieved into when absent, and verified from.
        #
        # Raises:
        #     TypeError: If the module belongs to another architecture
        #         family, if the release payload is not a mapping, or if
        #         any state entry is not a tensor.
        #     FileNotFoundError: If the resolved release path is not a
        #         readable file after the retrieval step, which also
        #         covers a directory sitting at that path.
        #     ValueError: If the release bytes do not match the recorded
        #         digest, or if the provenance record names no file to
        #         retrieve from the repository.
        #     RuntimeError: If the adapted state dictionary does not match
        #         the project generator key for key. Strictness is
        #         deliberate: it proves architectural agreement with the
        #         author release rather than assuming it.
        if not isinstance(module, Vocos):
            raise TypeError(
                f"Vocos published weights require Vocos, got {type(module).__name__}. "
                f"Build the module through ModelRegistry before loading author weights."
            )
        local_path: Path = self._ensure_local_path(published_weights_root)
        loaded_checkpoint: object = torch.load(local_path, map_location="cpu", weights_only=True)
        upstream_state_dict: dict[str, torch.Tensor] = self._extract_state_dict(loaded_checkpoint)
        adapted_state_dict: dict[str, torch.Tensor] = self._adapt_state_dict(upstream_state_dict)
        network: VocosNetwork = module.network
        network.load_state_dict(adapted_state_dict, strict=True)
        log.info(f"Loaded strict author weights for vocos from {local_path}")

    def _ensure_local_path(self, published_weights_root: Path) -> Path:
        # Resolves the cached release, retrieving it from the model hub
        # only when nothing is present at that path. The hub client places
        # the file under the requested directory but may return a path
        # that differs from the recorded layout, so the result is copied
        # into place when the two disagree. The digest is verified on both
        # branches, so an already-cached file is never trusted on the
        # strength of its existence. The hub client is imported here
        # rather than at module scope so the retrieval dependency is
        # required only on the path that actually downloads.
        #
        # Args:
        #     published_weights_root: Cache root the release lives under.
        #
        # Returns:
        #     The verified local path of the release.
        #
        # Raises:
        #     ValueError: If the provenance record names no file inside
        #         the repository, or if the bytes disagree with the
        #         recorded digest.
        #     FileNotFoundError: If nothing readable is at the path after
        #         the retrieval step. A directory standing at that path
        #         suppresses the download and fails here.
        from huggingface_hub import hf_hub_download

        local_path: Path = self._provenance.resolve_local_path(published_weights_root)
        if not local_path.exists():
            source_filename: str | None = self._provenance.retrieval_filename
            if source_filename is None:
                raise ValueError("Vocos retrieval filename is missing.")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path: Path = Path(
                hf_hub_download(
                    repo_id=self._provenance.retrieval_uri,
                    filename=source_filename,
                    local_dir=str(local_path.parent)
                )
            )
            if downloaded_path.resolve() != local_path.resolve():
                downloaded_path.copy(local_path)
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
            raise FileNotFoundError(f"Vocos published weight file is missing at {local_path}.")
        handle: BinaryIO
        with local_path.open("rb") as handle:
            actual_sha256: str = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual_sha256 != self._provenance.expected_sha256:
            raise ValueError(
                f"Vocos weight hash mismatch at {local_path}: "
                f"expected {self._provenance.expected_sha256}, received {actual_sha256}. "
                f"Remove the corrupted cache entry and retrieve the author file again."
            )

    def _extract_state_dict(self, loaded_checkpoint: object) -> dict[str, torch.Tensor]:
        # Narrows the untyped payload to a mapping of names to tensors.
        # The release is a bare state dictionary rather than a training
        # checkpoint, so the payload is read whole with no nesting key to
        # descend into. Every value is type-checked individually rather
        # than trusted in bulk, since a single non-tensor entry would
        # otherwise surface much later as an opaque load failure.
        #
        # Args:
        #     loaded_checkpoint: The deserialized release payload.
        #
        # Returns:
        #     The release state as a plain dictionary of tensors keyed by
        #     their upstream names, still including the entries the
        #     project network does not own.
        #
        # Raises:
        #     TypeError: If the payload is not a mapping, or if any entry
        #         is not a tensor. The message names the offending key.
        if not isinstance(loaded_checkpoint, Mapping):
            raise TypeError(
                f"Vocos author weights must be a mapping, got {type(loaded_checkpoint).__name__}."
            )
        state_dict: dict[str, torch.Tensor] = {}
        raw_key: object
        raw_value: object
        for raw_key, raw_value in loaded_checkpoint.items():
            key: str = str(raw_key)
            if not isinstance(raw_value, torch.Tensor):
                raise TypeError(
                    f"Vocos state entry {key} must be torch.Tensor, got "
                    f"{type(raw_value).__name__}."
                )
            state_dict[key] = raw_value
        return state_dict

    def _adapt_state_dict(
        self,
        upstream_state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Drops the entries the project network does not own, then renames
        # the rest onto the project layout. Apart from those drops the
        # mapping is purely nominal: no tensor is reshaped or reordered,
        # so a successful strict load afterwards is evidence that the two
        # topologies genuinely agree. The renames cover the backbone's
        # embedding and normalizations, the block stack and each block's
        # internals, and the head's projection.
        #
        # Args:
        #     upstream_state_dict: The release state keyed by upstream
        #         names.
        #
        # Returns:
        #     The retained tensors keyed by project parameter names, ready
        #     for a strict load onto the generator network.
        #
        # Note:
        #     The substitutions are applied in sequence to one key, so
        #     they must stay ordered from the more specific patterns to
        #     the more general: the backbone's own normalization keys are
        #     rewritten before the bare per-block normalization pattern,
        #     which would otherwise match them first.
        adapted_state_dict: dict[str, torch.Tensor] = {}
        upstream_key: str
        tensor: torch.Tensor
        for upstream_key, tensor in upstream_state_dict.items():
            if self._is_external_feature_state(upstream_key):
                continue
            local_key: str = upstream_key
            local_key: str = local_key.replace("backbone.embed.", "_backbone._embedding.")
            local_key: str = local_key.replace("backbone.norm.", "_backbone._normalization.")
            local_key: str = local_key.replace(
                "backbone.final_layer_norm.",
                "_backbone._final_normalization."
            )
            local_key: str = local_key.replace("backbone.convnext.", "_backbone._blocks.")
            local_key: str = local_key.replace(".dwconv.", "._depthwise_convolution.")
            local_key: str = local_key.replace(".norm.", "._normalization.")
            local_key: str = local_key.replace(".pwconv1.", "._pointwise_first.")
            local_key: str = local_key.replace(".pwconv2.", "._pointwise_second.")
            local_key: str = local_key.replace(".gamma", "._gamma")
            local_key: str = local_key.replace("head.out.", "_head._output_projection.")
            adapted_state_dict[local_key] = tensor
        return adapted_state_dict

    def _is_external_feature_state(self, upstream_key: str) -> bool:
        # Reports whether a release entry belongs to state this project
        # keeps outside the network. The release bundles the author's mel
        # feature extractor and the inverse transform's analysis window
        # into its state dictionary, whereas this project owns the mel
        # transform at module level and registers the window as a
        # non-persistent buffer. Those entries are therefore not
        # architectural disagreements to be reported but state the project
        # deliberately holds elsewhere, and dropping them is what allows
        # the remaining keys to load strictly.
        #
        # Args:
        #     upstream_key: One key of the release state dictionary.
        #
        # Returns:
        #     Whether the entry is dropped before adaptation. The set is
        #     enumerated exactly rather than matched by prefix, so an
        #     unrecognized extra key still reaches the strict load and
        #     fails there instead of being silently discarded.
        return upstream_key in (
            "feature_extractor.mel_spec.mel_scale.fb",
            "feature_extractor.mel_spec.spectrogram.window",
            "head.istft.window"
        )
