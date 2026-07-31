# This module:
# 1. Loads the author-released yl4579 HiFTNet checkpoint onto the project
#    HiFTNet implementation as the published reference anchor:
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
# - The release bundles the generator together with the F0-extractor
#   state; extraction separates the two before loading
#
# Author: Rahul Sawhney

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO, override

import torch
from loguru import logger as log

from syntheticmind.core.module import Module

from vocode.models.hiftnet.hiftnet import Hiftnet
from vocode.models.hiftnet.network import HiftnetNetwork
from vocode.models.vocoder import PublishedWeightProvenance, PublishedWeights

__all__: list[str] = ["HiftnetWeights"]


class HiftnetWeights(PublishedWeights):
    # Adapter binding the author-released HiFTNet checkpoint to the project
    # implementation. It satisfies the PublishedWeights contract: it exposes
    # an immutable provenance record and loads that release onto an
    # already-constructed module. It never builds a module, and the module
    # never reads a release, so local training and reference anchoring stay
    # separable and a reproduction run can state exactly which of the two
    # produced a given set of parameters.
    #
    # Integration: the load path is deliberately unforgiving. Retrieval,
    # hash verification, extraction, key adaptation, and strict loading each
    # raise rather than degrade, so a release that disagrees with this
    # implementation fails loudly instead of silently anchoring the study to
    # partially-initialized parameters. The one tolerated mismatch is the
    # release's voicing-detector state, which this implementation does not
    # reproduce and which is dropped under an exact count check.
    def __init__(self) -> None:
        # Records the release identity. The author source and the retrieval
        # source differ by design: the original release is a zip archive, so
        # the extracted checkpoint is mirrored to a project-controlled
        # repository that serves the single file directly. The recorded hash
        # is over that mirrored file and is what makes the substitution safe
        # to audit.
        self._provenance: PublishedWeightProvenance = PublishedWeightProvenance(
            architecture_name="hiftnet",
            variant_name="yl4579_ljspeech_g_00155000",
            author_source_uri="https://huggingface.co/yl4579/HiFTNet/blob/main/LJSpeech/cp_hifigan.zip",
            retrieval_kind="huggingface_file",
            retrieval_uri="r-sawhney/vocode-phase1-assets",
            retrieval_filename="hiftnet_lj/g_00155000",
            expected_sha256="639713061aa90a8665f0469b071f29c143a4250ed86c4dce6cb5d4b2a7f2b9f1",
            local_relative_path=Path("hiftnet_lj/g_00155000"),
            serialization="checkpoint_dict",
            state_dict_key="generator"
        )

    @property
    @override
    def provenance(self) -> PublishedWeightProvenance:
        # Returns the immutable release identity, consulted by the registry
        # and by reporting code that must cite the exact anchor used.
        return self._provenance

    @override
    def load(self, module: Module, published_weights_root: Path) -> None:
        # Loads the author release onto a constructed HiFTNet module in five
        # ordered stages: the module type is checked, the file is retrieved
        # and hash-verified, the state dictionary is extracted from the
        # checkpoint wrapper, its keys are adapted to local names, and the
        # result is applied strictly to the generator network.
        #
        # Deserialization restricts unpickling to tensor data, so a
        # substituted file cannot execute code during loading; the hash check
        # has already run by that point, making this a second independent
        # barrier rather than the only one.
        #
        # The parameters land on the module's network rather than on the
        # module, because the release describes only the generator. The
        # discriminators and the loss keep their fresh initialization, which
        # is correct for a synthesis-only reference anchor and means a module
        # loaded this way is ready to measure but not to resume training.
        #
        # Args:
        #     module: The target module, which must be a Hiftnet instance.
        #     published_weights_root: Local directory under which releases are
        #         cached; the file is downloaded into it on first use.
        #
        # Raises:
        #     TypeError: If the module is not a Hiftnet, if the checkpoint is
        #         not a mapping, or if any state entry is not a tensor.
        #     FileNotFoundError: If the file is absent after retrieval.
        #     ValueError: If the retrieval filename is unset or the content
        #         hash does not match the recorded value.
        #     RuntimeError: If the release's detector-state layout has
        #         changed, or if strict loading finds any key mismatch.
        if not isinstance(module, Hiftnet):
            raise TypeError(
                f"HiFTNet published weights require Hiftnet, got {type(module).__name__}. "
                f"Build the module through ModelRegistry before loading author weights."
            )
        local_path: Path = self._ensure_local_path(published_weights_root)
        loaded_checkpoint: object = torch.load(local_path, map_location="cpu", weights_only=True)
        upstream_state_dict: dict[str, torch.Tensor] = self._extract_state_dict(loaded_checkpoint)
        adapted_state_dict: dict[str, torch.Tensor] = self._adapt_state_dict(upstream_state_dict)
        network: HiftnetNetwork = module.network
        network.load_state_dict(adapted_state_dict, strict=True)
        log.info(f"Loaded strict author weights for hiftnet from {local_path}")

    def _ensure_local_path(self, published_weights_root: Path) -> Path:
        # Resolves the release to a verified local file, downloading it if the
        # cache is cold. The hash is validated on every call rather than only
        # after a download, so a cache entry corrupted or replaced between
        # runs is caught with the same force as a bad download.
        #
        # The retrieval library is imported inside the method so that
        # constructing this adapter, and importing the module at all, does not
        # require the download dependency; only an actual load does.
        from huggingface_hub import hf_hub_download

        local_path: Path = self._provenance.resolve_local_path(published_weights_root)
        if not local_path.exists():
            source_filename: str | None = self._provenance.retrieval_filename
            if source_filename is None:
                raise ValueError("HiFTNet retrieval filename is missing.")
            published_weights_root.mkdir(parents=True, exist_ok=True)
            downloaded_path: Path = Path(
                hf_hub_download(
                    repo_id=self._provenance.retrieval_uri,
                    filename=source_filename,
                    local_dir=str(published_weights_root)
                )
            )
            # The download library may place the file at a path of its own
            # choosing; when that differs from the provenance-declared
            # location, the file is copied so that later runs find it exactly
            # where the record says it lives.
            if downloaded_path.resolve() != local_path.resolve():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                downloaded_path.copy(local_path)
        self._validate_sha256(local_path)
        return local_path

    def _validate_sha256(self, local_path: Path) -> None:
        # Verifies the file's content hash against the recorded value before
        # any tensor is read, so a corrupted or substituted release cannot
        # masquerade as the reference anchor. The failure message names the
        # remedy explicitly, because the actionable response is to discard the
        # cache entry rather than to adjust the expectation.
        #
        # Raises:
        #     FileNotFoundError: If no regular file exists at the path.
        #     ValueError: If the computed digest differs from the recorded
        #         one.
        if not local_path.is_file():
            raise FileNotFoundError(f"HiFTNet published weight file is missing at {local_path}.")
        handle: BinaryIO
        with local_path.open("rb") as handle:
            actual_sha256: str = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual_sha256 != self._provenance.expected_sha256:
            raise ValueError(
                f"HiFTNet weight hash mismatch at {local_path}: "
                f"expected {self._provenance.expected_sha256}, received {actual_sha256}. "
                f"Remove the corrupted cache entry and retrieve the author file again."
            )

    def _extract_state_dict(self, loaded_checkpoint: object) -> dict[str, torch.Tensor]:
        # Unwraps the generator state from the author's checkpoint container,
        # which bundles the generator alongside other training state. Every
        # entry is required to be a tensor: unlike the pitch-checkpoint
        # adapter in the network module, which tolerates and skips
        # non-tensor bookkeeping, this path treats a non-tensor as proof that
        # the release layout is not the one the provenance describes.
        #
        # Raises:
        #     TypeError: If the checkpoint is not a mapping, if it carries no
        #         mapping under the generator key, or if any value is not a
        #         tensor.
        if not isinstance(loaded_checkpoint, Mapping):
            raise TypeError(
                f"HiFTNet author checkpoint must be a mapping, got "
                f"{type(loaded_checkpoint).__name__}."
            )
        raw_state_dict: object = loaded_checkpoint.get("generator")
        if not isinstance(raw_state_dict, Mapping):
            raise TypeError(
                f"HiFTNet author checkpoint must contain a mapping at generator, got "
                f"{type(raw_state_dict).__name__}."
            )
        state_dict: dict[str, torch.Tensor] = {}
        raw_key: object
        raw_value: object
        for raw_key, raw_value in raw_state_dict.items():
            key: str = str(raw_key)
            if not isinstance(raw_value, torch.Tensor):
                raise TypeError(
                    f"HiFTNet state entry {key} must be torch.Tensor, got "
                    f"{type(raw_value).__name__}."
                )
            state_dict[key] = raw_value
        return state_dict

    def _adapt_state_dict(
        self,
        upstream_state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Rewrites every upstream key onto this implementation's member names.
        # The rules divide into three groups: the pitch network's nested
        # names, the generator's own names, and the two weight-normalization
        # tensors, whose names changed when the parameterization moved into
        # the framework's parametrization registry. Order matters, because the
        # rules are substring replacements and a broader pattern applied first
        # would consume text a narrower one owns.
        #
        # The detector-state count is a deliberate tripwire rather than a
        # tolerance. Silently dropping unrecognized keys would let a future
        # release quietly withhold parameters this implementation needs, and
        # strict loading downstream would not catch it because the missing
        # entries were never presented. Requiring the exact known count means
        # any change to what the release omits stops the load and forces an
        # audit before the adapter is relaxed.
        #
        # Raises:
        #     RuntimeError: If the number of skipped detector entries differs
        #         from the recorded contract.
        adapted_state_dict: dict[str, torch.Tensor] = {}
        ignored_detector_state_count: int = 0
        upstream_key: str
        tensor: torch.Tensor
        for upstream_key, tensor in upstream_state_dict.items():
            if self._is_unused_detector_state(upstream_key):
                ignored_detector_state_count: int = ignored_detector_state_count + 1
                continue
            local_key: str = upstream_key
            local_key: str = local_key.replace("F0_model.", "_f0_model.")
            local_key: str = local_key.replace(".conv_block.", "._conv_block.")
            local_key: str = local_key.replace(".res_block1.", "._res_block_1.")
            local_key: str = local_key.replace(".res_block2.", "._res_block_2.")
            local_key: str = local_key.replace(".res_block3.", "._res_block_3.")
            local_key: str = local_key.replace(".pre_conv.", "._pre_convolution.")
            local_key: str = local_key.replace(".conv1by1.", "._projection.")
            local_key: str = local_key.replace(".conv.", "._convolution.")
            local_key: str = local_key.replace(".pool_block.0.", "._pool_batch_norm.")
            local_key: str = local_key.replace(".bilstm_classifier.", "._bilstm_classifier.")
            local_key: str = local_key.replace(".classifier.", "._classifier.")
            local_key: str = local_key.replace("conv_pre.", "_pre_convolution.")
            local_key: str = local_key.replace("m_source.l_linear.", "_source_module._linear.")
            local_key: str = local_key.replace("noise_convs.", "_noise_convolutions.")
            local_key: str = local_key.replace("noise_res.", "_noise_residuals.")
            local_key: str = local_key.replace("ups.", "_upsample_layers.")
            local_key: str = local_key.replace("resblocks.", "_residual_blocks.")
            local_key: str = local_key.replace("conv_post.", "_post_convolution.")
            local_key: str = local_key.replace(".convs1.", "._dilated_convolutions.")
            local_key: str = local_key.replace(".convs2.", "._refinement_convolutions.")
            local_key: str = local_key.replace(".alpha1.", "._alpha_1.")
            local_key: str = local_key.replace(".alpha2.", "._alpha_2.")
            local_key: str = local_key.replace(".weight_g", ".parametrizations.weight.original0")
            local_key: str = local_key.replace(".weight_v", ".parametrizations.weight.original1")
            adapted_state_dict[local_key] = tensor
        if ignored_detector_state_count != 16:
            raise RuntimeError(
                f"HiFTNet author checkpoint detector-state contract changed: expected 16 unused "
                f"entries, received {ignored_detector_state_count}. Audit the upstream checkpoint "
                f"before changing the adapter."
            )
        return adapted_state_dict

    def _is_unused_detector_state(self, upstream_key: str) -> bool:
        # Identifies parameters belonging to the reference pitch network's
        # voicing-detector head, which this implementation does not reproduce
        # because the generator consumes only the pitch regression output.
        return upstream_key.startswith(
            (
                "F0_model.detector_conv.",
                "F0_model.bilstm_detector.",
                "F0_model.detector."
            )
        )
