# This module:
# 1. Defines the closed architecture-name vocabulary of the study and the
#    structural protocols shared across the model layer: the Vocoder
#    synthesis contract and the PublishedWeights adapter contract
# 2. Defines the published-weight provenance record binding every author
#    release to its source, retrieval method, content hash, and
#    serialization layout
#
# Design decisions:
# - Vocoder is a structural protocol rather than a base class, so metric
#   and profiling components can accept any module exposing the synthesis
#   surface without inheritance coupling
# - Published-weight provenance is a frozen record with a mandatory SHA-256,
#   because reference anchors must be traceable to exact release bytes
# - The architecture vocabulary is a closed literal type; adding a name is
#   a deliberate registry-level change, never a runtime string
#
# Author: Rahul Sawhney

from pathlib import Path
from typing import ClassVar, Literal, Protocol

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from syntheticmind.core.module import Module

from vocode.transforms.mel import MelConfig

__all__: list[str] = [
    "ArchitectureName",
    "PublishedWeightProvenance",
    "PublishedWeights",
    "Vocoder"
]


# The closed architecture vocabulary of the study. Every registry record,
# provenance record, run configuration, and evidence row is keyed by one
# of these names, so adding an entry is a deliberate registry-level change
# rather than a runtime string.
type ArchitectureName = Literal[
    "hifigan_v1",
    "hifigan_v2",
    "hifigan_v3",
    "melgan",
    "vocos",
    "bigvgan",
    "apnet2",
    "freev",
    "hiftnet",
    "lpcnet",
    "rndvoc",
    "vocosformer",
    "rfwave"
]
# The audited transports an author release may be retrieved over. Each
# kind fixes how the provenance retrieval fields are read: ``"http"``
# treats the retrieval URI as the direct asset URL, ``"huggingface_file"``
# treats it as a hub repository identifier paired with the retrieval
# filename, and ``"google_drive_folder"`` treats it as a shared folder
# from which the retrieval filename selects one asset.
type PublishedWeightRetrievalKind = Literal[
    "http",
    "huggingface_file",
    "google_drive_folder"
]
# How the release file is laid out once loaded: ``"pytorch_state_dict"``
# for a payload that is the state dictionary itself, and
# ``"checkpoint_dict"`` for a training checkpoint whose state dictionary
# is nested under a named key.
type PublishedWeightSerialization = Literal[
    "pytorch_state_dict",
    "checkpoint_dict"
]


class PublishedWeightProvenance(BaseModel):
    # Frozen release identity of one author checkpoint: source and
    # retrieval URIs, expected SHA-256, local layout, and how the
    # serialized state dictionary is keyed. The record travels with the
    # weights into the evidence rows of the published-evaluation lane, so
    # every reported published number is traceable to exact release bytes.
    #
    # Fields:
    #     architecture_name: The architecture this release anchors, drawn
    #         from the closed study vocabulary. An adapter wired to
    #         another family's release is a validation failure here.
    #     variant_name: The author's own name for this release, for
    #         example ``"jik876_lj_v1"``; it distinguishes several
    #         releases of one architecture family.
    #     author_source_uri: The authoring project the release originates
    #         from, recorded as the citable source rather than a mirror.
    #     retrieval_kind: The transport the release is fetched over, which
    #         fixes how ``retrieval_uri`` and ``retrieval_filename`` are
    #         interpreted.
    #     retrieval_uri: The retrieval address under that transport: a
    #         direct URL, a hub repository identifier, or a shared folder
    #         URL.
    #     retrieval_filename: The asset to select once the retrieval
    #         target is reached, including any path inside a folder
    #         release. ``None`` when the retrieval URI already addresses
    #         the asset itself.
    #     expected_sha256: The mandatory hex digest of the release bytes.
    #         Adapters verify it before reading any tensor, so a corrupted
    #         or substituted file cannot masquerade as the reference.
    #     local_relative_path: Where the release is cached beneath the
    #         published-weights root. Each release owns a family
    #         subdirectory so cached files never collide.
    #     serialization: The payload layout of the release file.
    #     state_dict_key: The key nesting the state dictionary inside a
    #         checkpoint payload, for example ``"generator"``; ``None``
    #         for a flat state-dictionary release.
    #
    # Note:
    #     ``serialization`` and ``state_dict_key`` are declarative
    #     provenance rather than loader inputs: each adapter implements
    #     the matching extraction against its own release directly, and
    #     neither field is read at load time. They exist so the recorded
    #     layout is auditable alongside the reported numbers.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True
    )
    architecture_name: ArchitectureName
    variant_name: str
    author_source_uri: str
    retrieval_kind: PublishedWeightRetrievalKind
    retrieval_uri: str
    retrieval_filename: str | None
    expected_sha256: str
    local_relative_path: Path
    serialization: PublishedWeightSerialization
    state_dict_key: str | None

    def resolve_local_path(self, published_weights_root: Path) -> Path:
        # Resolves where this release lives under the local weights root.
        # This is a pure path computation: nothing is created, read, or
        # checked for existence, so it is safe to call before retrieval
        # and against a root that does not exist.
        #
        # Args:
        #     published_weights_root: The cache root every release is
        #         stored beneath. A relative root yields a relative
        #         release path.
        #
        # Returns:
        #     The root joined with this release's recorded relative
        #     layout, preserving its family subdirectory.
        return published_weights_root / self.local_relative_path


class PublishedWeights(Protocol):
    # Structural contract of a published-weights adapter: it exposes its
    # provenance and loads the release onto a harness Module. Adapters
    # inherit this protocol explicitly so the contract is discoverable
    # from the class, but it is deliberately not runtime-checkable, and
    # isinstance against it raises.
    #
    # Integration: an adapter carries its provenance from construction,
    # while load performs the whole retrieval sequence in this order:
    # retrieve if absent, verify the SHA-256, extract the state
    # dictionary, adapt upstream keys to the project layout, and load
    # strictly onto the module's network. Strictness is the point: it
    # proves architectural agreement with the author release instead of
    # assuming it.
    #
    # The frozen release identity this adapter retrieves and verifies.
    @property
    def provenance(self) -> PublishedWeightProvenance: ...

    # Retrieves, verifies, and strictly loads the release onto the given
    # harness module, which must belong to the adapter's own architecture
    # family; loading is refused for any other module type.
    def load(self, module: Module, published_weights_root: Path) -> None: ...


class Vocoder(Protocol):
    # Structural synthesis contract consumed by the metric and profiling
    # components: the synthesis network, the conditioning and metric mel
    # protocols, and mel-to-waveform synthesis. Declaring it as a protocol
    # rather than a base class is what lets the MACs profiler accept any
    # module exposing this surface without inheritance coupling; like the
    # weights contract it is not runtime-checkable, so conformance is
    # established structurally rather than by isinstance.
    #
    # The synthesis network alone, excluding discriminators and loss
    # modules, so complexity profiling measures inference cost only.
    @property
    def network(self) -> nn.Module: ...

    # The mel protocol the generator is conditioned on, which also fixes
    # the band count and hop length of any probe mel built for profiling.
    @property
    def mel_protocol(self) -> MelConfig: ...

    # The mel protocol the mel-error metric extracts with. Families that
    # condition and measure on one grid return the same record from both
    # properties; HiFi-GAN conditions band-limited and measures
    # full-band, so its two records differ.
    @property
    def metric_mel_protocol(self) -> MelConfig: ...

    # Synthesizes a waveform batch from a conditioning mel batch. This is
    # the entry point the measurement stack calls rather than forward, so
    # a family may route it through additional preparation if needed.
    def synthesize(self, mel: torch.Tensor) -> torch.Tensor: ...
