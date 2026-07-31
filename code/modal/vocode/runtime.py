# This module:
# 1. Prepares the shared Modal runtime: the version-constrained container, the
#    in-volume LJSpeech hydration, and the in-container import paths
#
# Design decisions:
# - The image bakes the syntheticmind harness and the vocode client from
#   the launching working tree, so every capsule executes exactly the
#   code state that spawned it; any source change invalidates the copy
#   layers on the next launch
# - Python dependency pins define the aligned measurement stack; the Modal
#   Debian base and host driver were not digest-pinned in the retained runs.
#   Changing a Python pin
#   changes the runtime identity of every subsequent capsule and is a
#   deliberate, committed act
# - Dataset hydration happens cloud-side into the volume, so local
#   storage never becomes a hidden dependency of cloud runs
#
# Author: Rahul Sawhney

import subprocess
import sys
from pathlib import Path

from loguru import logger as log

import modal

__all__: list[str] = [
    "LJSpeechCorpusPreparer",
    "ModalPythonPathConfigurator",
    "ModalVocodeImageBuilder"
]


class LJSpeechCorpusPreparer:
    # Runtime helper that hydrates the LJSpeech corpus inside the Modal volume.
    # Dataset preparation stays cloud-side so local storage does not become a hidden dependency.
    def ensure(self, local_root: Path) -> None:
        # Ensures the LJSpeech corpus is available inside the Modal volume.
        metadata_file: Path = local_root / "metadata.csv"
        wavs_directory: Path = local_root / "wavs"
        if metadata_file.exists() and wavs_directory.exists():
            log.info(f"LJSpeech corpus already present at {local_root}")
            return
        local_root.parent.mkdir(parents=True, exist_ok=True)
        archive_path: Path = local_root.parent / "LJSpeech-1.1.tar.bz2"
        if not archive_path.exists():
            self._download_archive(archive_path)
        self._extract_archive(archive_path=archive_path, extraction_root=local_root.parent)
        log.info(f"LJSpeech corpus ready at {local_root}")

    def _download_archive(self, archive_path: Path) -> None:
        # Downloads the LJSpeech archive into Modal storage when the corpus is absent.
        log.info(f"Downloading LJSpeech corpus to {archive_path}")
        download_command: list[str] = [
            "wget",
            "-q",
            "-O",
            str(archive_path),
            "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
        ]
        subprocess.run(download_command, check=True)

    def _extract_archive(self, archive_path: Path, extraction_root: Path) -> None:
        # Extracts the LJSpeech archive into the Modal dataset directory.
        log.info(f"Extracting LJSpeech corpus to {extraction_root}")
        extraction_command: list[str] = [
            "tar",
            "-xjf",
            str(archive_path),
            "-C",
            str(extraction_root)
        ]
        subprocess.run(extraction_command, check=True)


class ModalPythonPathConfigurator:
    # Runtime helper that configures import paths inside the Modal container.
    # It makes the project harness and VOCODE package importable from the copied workspace.
    def configure_workspace_paths(self) -> None:
        # Prepends copied workspace paths to sys.path inside the Modal container.
        self._prepend_path_once("/workspace")

    def _prepend_path_once(self, path_text: str) -> None:
        # Prepends a workspace path only if it is not already present in sys.path.
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


class ModalVocodeImageBuilder:
    # Modal image builder for VOCODE cloud execution.
    # The image definition pins dependencies required for B200 evaluation and training jobs.
    def build(self) -> modal.Image:
        # Builds the Modal container image used for VOCODE B200 execution.
        image: modal.Image = (
            modal.Image.debian_slim(python_version="3.14")
            .apt_install("git", "wget", "build-essential")
            .pip_install(
                "torch==2.13.0",
                "torchaudio==2.11.0",
                "numpy==2.3.3",
                "scipy==1.16.3",
                "pesq==0.0.4",
                "pystoi==0.4.1",
                "vocos==0.1.0",
                "huggingface-hub==1.14.0",
                "gdown==6.0.0",
                "pghipy==0.1.1",
                "pydantic==2.13.4",
                "pydantic-settings==2.12.0",
                "pyyaml==6.0.3",
                "loguru==0.7.3",
                "tqdm==4.67.1",
                "matplotlib==3.10.7",
                "annotated-types==0.7.0",
                "auraloss==0.4.0",
                "torchcrepe==0.0.24",
                "fvcore==0.1.5.post20221221",
                "torchao==0.17.0",
                "onnx==1.20.1",
                "onnxruntime==1.24.4"
            )
            .env({"TORCH_HOME": "/data/pretrained/torch"})
            .add_local_file("code/modal/vocode/runtime.py", remote_path="/root/runtime.py", copy=True)
            .add_local_file("code/modal/vocode/application.py", remote_path="/root/application.py", copy=True)
            .add_local_dir("code/vocode", remote_path="/workspace/vocode", copy=True)
            .add_local_dir("code/syntheticmind", remote_path="/workspace/syntheticmind", copy=True)
        )
        return image
