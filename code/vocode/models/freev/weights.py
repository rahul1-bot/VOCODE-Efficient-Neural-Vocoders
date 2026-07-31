# This module:
# 1. Loads the author-released official FreeV checkpoint onto the project
#    FreeV implementation as the published reference anchor:
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
# - The generator entry is loaded verbatim under the author's own key
#   names, because the project network reproduces those names exactly,
#   including the two normalization modules its forward pass never invokes;
#   a remapping layer here would be able to hide a genuine divergence from
#   the strict load, which is the one check this adapter exists to perform
# - The network's pseudo-inverse projection is a non-persistent buffer, so
#   the release neither carries it nor needs to, and the strict load is
#   unaffected by its absence
# - Verification runs on every load, not only after a download, so a cache
#   entry that was corrupted after retrieval is still caught
# - The Hugging Face client is imported inside the retrieval helper, so
#   constructing the adapter or reading its provenance never requires the
#   dependency to be installed
#
# Author: Rahul Sawhney

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO, override

import torch
from loguru import logger as log

from syntheticmind.core.module import Module

from vocode.models.freev.freev import Freev
from vocode.models.freev.network import FreevNetwork
from vocode.models.vocoder import PublishedWeightProvenance, PublishedWeights

__all__: list[str] = ["FreevWeights"]


class FreevWeights(PublishedWeights):
    # Published-weights adapter for FreeV. It binds one immutable release
    # identity and knows how to turn it into loaded parameters: resolve the
    # release locally, retrieve it if absent, verify its bytes against the
    # recorded digest, unwrap the training checkpoint down to the generator
    # state, and load that state strictly onto the project network.
    #
    # The adapter is the reproduction anchor of this architecture. A strict
    # load that succeeds is evidence that the project network agrees with the
    # published one parameter for parameter; every failure mode below is
    # therefore reported rather than repaired, because silently accommodating
    # a mismatch would destroy exactly the evidence the load produces.
    #
    # The release loaded here is the author anchor: it establishes
    # architectural agreement and supplies the published quality value the
    # registered checkpoint gate targets. It is distinct from the FreeV
    # Retained Project Checkpoint, which is fitted from random
    # initialization under the project budget. Every reported measurement of
    # this configuration is inherited from that project-trained state and
    # never from these weights.
    #
    # Integration: it satisfies the
    # vocode.models.vocoder.PublishedWeights protocol through provenance and
    # load, so the registry treats it identically to every other
    # architecture's adapter.
    def __init__(self) -> None:
        # Records the immutable release identity of the author's
        # one-million-step LJSpeech generator: where it was published, how it
        # is retrieved, the digest its bytes must produce, where it lives
        # locally, and that it is a training checkpoint whose weights sit
        # under the generator key rather than a bare state dictionary.
        self._provenance: PublishedWeightProvenance = PublishedWeightProvenance(
            architecture_name="freev",
            variant_name="bakerbunker_ljspeech_g_01000000",
            author_source_uri="https://github.com/BakerBunker/FreeV",
            retrieval_kind="huggingface_file",
            retrieval_uri="Bakerbunker/FreeV_Model_Logs",
            retrieval_filename="freev_g_01000000",
            expected_sha256="a2e92b9cb1278f49d41c43f8e05f09f61757e257fc1e63d2f97f4c91f4d45b9c",
            local_relative_path=Path("freev_lj/freev_g_01000000"),
            serialization="checkpoint_dict",
            state_dict_key="generator"
        )

    @property
    @override
    def provenance(self) -> PublishedWeightProvenance:
        # Returns the frozen release record. It is readable without any
        # retrieval, so provenance can be reported and audited offline.
        return self._provenance

    @override
    def load(self, module: Module, published_weights_root: Path) -> None:
        # Loads the author release onto the module's generator network.
        #
        # The gates run in a fixed order and each one refuses before the next
        # can do work: the module type is checked before any filesystem access,
        # the release is resolved and its digest verified before any tensor is
        # deserialized, the checkpoint structure is validated before any tensor
        # reaches the network, and only then does the strict load run. Only the
        # generator network is loaded; the discriminator ensembles are left at
        # their initialization, because the reference anchor is a synthesis
        # claim and the release carries no discriminator state under this key.
        #
        # Args:
        #     module: The harness module to load onto. It must be a Freev,
        #         since the release's parameter names describe that network
        #         alone.
        #     published_weights_root: Local root the release is cached under;
        #         the release's recorded relative path is resolved beneath it.
        #
        # Raises:
        #     TypeError: If the module is not a Freev, or the deserialized
        #         checkpoint is not a mapping carrying a mapping of tensors at
        #         the generator key.
        #     ValueError: If the recorded retrieval filename is absent, or the
        #         local file's SHA-256 does not equal the recorded digest.
        #     FileNotFoundError: If the resolved release path is not a regular
        #         file, which includes a directory left at that path.
        #     RuntimeError: Raised by the strict load when the release and the
        #         project network disagree on any parameter name or shape. This
        #         is the substantive reproduction failure the adapter exists to
        #         surface, and it is deliberately not softened to a relaxed
        #         load; for this architecture it is also what proves the
        #         unused reference normalizations were materialized.
        if not isinstance(module, Freev):
            raise TypeError(
                f"FreeV published weights require Freev, got {type(module).__name__}. "
                f"Build the module through ModelRegistry before loading author weights."
            )
        local_path: Path = self._ensure_local_path(published_weights_root)
        loaded_checkpoint: object = torch.load(local_path, map_location="cpu", weights_only=True)
        state_dict: dict[str, torch.Tensor] = self._extract_state_dict(loaded_checkpoint)
        network: FreevNetwork = module.network
        network.load_state_dict(state_dict, strict=True)
        log.info(f"Loaded strict author weights for freev from {local_path}")

    def _ensure_local_path(self, published_weights_root: Path) -> Path:
        # Resolves the release locally, retrieving it once if it is absent, and
        # returns the verified path.
        #
        # The download is directed at the release's own directory, which is
        # created first, so the file lands at the recorded relative layout; the
        # copy afterwards only runs when the client resolved the file somewhere
        # else, for instance through a shared cache. Verification runs
        # unconditionally at the end, so an existing cache entry is re-checked
        # on every load and never trusted merely for existing.
        #
        # Args:
        #     published_weights_root: Local root the release is cached under.
        #
        # Raises:
        #     ValueError: If the provenance carries no retrieval filename, or
        #         the resolved file fails digest verification.
        #     FileNotFoundError: If the resolved path is not a regular file.
        from huggingface_hub import hf_hub_download

        local_path: Path = self._provenance.resolve_local_path(published_weights_root)
        if not local_path.exists():
            source_filename: str | None = self._provenance.retrieval_filename
            if source_filename is None:
                raise ValueError("FreeV retrieval filename is missing.")
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
        # Verifies that the local file's bytes hash to the recorded digest.
        #
        # The file is digested by streaming rather than read into memory, and
        # this runs before torch.load is ever called, so a substituted or
        # truncated release is rejected before it can be deserialized.
        #
        # Args:
        #     local_path: Resolved path of the release under the weights root.
        #
        # Raises:
        #     FileNotFoundError: If the path is not a regular file; a directory
        #         left at the release path fails here rather than at the read.
        #     ValueError: If the computed digest differs from the recorded one.
        #         The message carries both digests and names the remedy, since
        #         the only correct response is to discard the cache entry and
        #         retrieve the author file again.
        if not local_path.is_file():
            raise FileNotFoundError(f"FreeV published weight file is missing at {local_path}.")
        handle: BinaryIO
        with local_path.open("rb") as handle:
            actual_sha256: str = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual_sha256 != self._provenance.expected_sha256:
            raise ValueError(
                f"FreeV weight hash mismatch at {local_path}: "
                f"expected {self._provenance.expected_sha256}, received {actual_sha256}. "
                f"Remove the corrupted cache entry and retrieve the author file again."
            )

    def _extract_state_dict(self, loaded_checkpoint: object) -> dict[str, torch.Tensor]:
        # Unwraps the author's training checkpoint down to the generator state.
        #
        # The release is a full training checkpoint rather than a bare state
        # dictionary, so the generator entry is selected and every one of its
        # values is type-checked before any of them can reach the network. Keys
        # are copied under their own names: no renaming happens here, because
        # the project network already reproduces the author's attribute names.
        #
        # Args:
        #     loaded_checkpoint: Whatever torch.load returned, typed as object
        #         because a deserialized payload carries no guarantee of shape.
        #
        # Raises:
        #     TypeError: If the payload is not a mapping, if it carries no
        #         mapping at the generator key, or if any entry of that mapping
        #         is not a tensor. The last message names the offending key, so
        #         a malformed release is identifiable from the log alone.
        #
        # Returns:
        #     The generator parameters as a flat key-to-tensor mapping, ready
        #     for the strict load.
        if not isinstance(loaded_checkpoint, Mapping):
            raise TypeError(
                f"FreeV author checkpoint must be a mapping, got "
                f"{type(loaded_checkpoint).__name__}."
            )
        raw_state_dict: object = loaded_checkpoint.get("generator")
        if not isinstance(raw_state_dict, Mapping):
            raise TypeError(
                f"FreeV author checkpoint must contain a mapping at generator, got "
                f"{type(raw_state_dict).__name__}."
            )
        state_dict: dict[str, torch.Tensor] = {}
        raw_key: object
        raw_value: object
        for raw_key, raw_value in raw_state_dict.items():
            key: str = str(raw_key)
            if not isinstance(raw_value, torch.Tensor):
                raise TypeError(
                    f"FreeV state entry {key} must be torch.Tensor, got "
                    f"{type(raw_value).__name__}."
                )
            state_dict[key] = raw_value
        return state_dict
