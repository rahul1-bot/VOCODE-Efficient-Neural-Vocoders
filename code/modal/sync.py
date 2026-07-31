# This module:
# 1. Synchronizes run artifacts from the Modal volume into the local
#    evidence mirrors, so completed cloud capsules can be downloaded for
#    admission review and packaging
#
# Design decisions:
# - Synchronization is pull-based and explicit; nothing downloads
#   automatically, keeping local evidence directories under deliberate
#   control
#
# Author: Rahul Sawhney

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import ClassVar, cast

from loguru import logger as log
from pydantic import BaseModel, ConfigDict, field_validator

__all__: list[str] = [
    "ModalArtifactSyncCli",
    "ModalArtifactSyncConfig",
    "ModalArtifactSynchronizer"
]


class ModalArtifactSyncConfig(BaseModel):
    # Configuration for synchronizing cloud artifacts into the local evidence tree.
    # The fields identify the Modal volume, remote path, and local destination explicitly.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    volume_name: str
    remote_relative_path: Path
    local_destination_root: Path
    overwrite_existing_local_files: bool = False

    @field_validator("remote_relative_path")
    @classmethod
    def validate_remote_relative_path(cls, value: Path) -> Path:
        normalized_parts: tuple[str, ...] = tuple(
            part for part in value.parts if part not in ("", "/")
        )
        if not normalized_parts or any(part in (".", "..") for part in normalized_parts):
            raise ValueError(f"remote_relative_path must identify a bounded volume path, got {value}.")
        return value


class ModalArtifactSynchronizer:
    # Synchronizer for pulling lightweight experiment artifacts from Modal storage.
    # It keeps cloud execution outputs aligned with the Git-tracked artifact mirror.
    def __init__(self, configuration: ModalArtifactSyncConfig) -> None:
        # Binds the collaborators this component uses.
        self._configuration: ModalArtifactSyncConfig = configuration

    def synchronize(self) -> Path:
        # Pulls a configured remote artifact path from Modal into the local evidence tree.
        modal_executable: str = self._resolve_modal_executable()
        local_destination: Path = self._resolve_local_destination()
        if local_destination.exists():
            if not self._configuration.overwrite_existing_local_files:
                log.info(f"Local destination {local_destination} already exists; skipping sync")
                return local_destination
        local_destination.parent.mkdir(parents=True, exist_ok=True)
        command_arguments: list[str] = [
            modal_executable,
            "volume",
            "get"
        ]
        if self._configuration.overwrite_existing_local_files:
            command_arguments.append("--force")
        command_arguments.extend([
            self._configuration.volume_name,
            str(self._configuration.remote_relative_path),
            str(local_destination)
        ])
        log.info(f"Invoking: {' '.join(command_arguments)}")
        completed_process: subprocess.CompletedProcess[bytes] = subprocess.run(
            command_arguments, check=False, capture_output=True
        )
        if completed_process.returncode != 0:
            stderr_text: str = completed_process.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"modal volume get failed with code {completed_process.returncode}: {stderr_text}"
            )
        log.info(f"Sync complete: {local_destination}")
        return local_destination

    @property
    def configuration(self) -> ModalArtifactSyncConfig:
        # Executes a public Modal operation used by the VOCODE cloud workflow.
        return self._configuration

    def _resolve_modal_executable(self) -> str:
        # Resolves the Modal executable used for subprocess-based storage commands.
        candidate: str | None = shutil.which("modal")
        if candidate is None:
            raise FileNotFoundError(
                "The `modal` CLI executable was not found on PATH. Install the modal package "
                "and ensure its console script is reachable."
            )
        return candidate

    def _resolve_local_destination(self) -> Path:
        # Resolves the local destination directory for synchronized artifacts.
        local_destination: Path = (
            self._configuration.local_destination_root
            / self._configuration.remote_relative_path.name
        )
        return local_destination

class ModalArtifactSyncCli:
    # Command-line interface for artifact synchronization from Modal to local storage.
    # The CLI keeps sync operations reproducible and visible in terminal history.
    def __init__(self, arguments: list[str]) -> None:
        # Binds the collaborators this component uses.
        self._arguments: list[str] = arguments

    def run(self) -> int:
        # Executes the requested Modal utility command from parsed CLI arguments.
        parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="modal-sync")
        parser.add_argument("--volume-name", type=str, required=True)
        parser.add_argument("--remote-relative-path", type=Path, required=True)
        parser.add_argument("--local-destination-root", type=Path, required=True)
        parser.add_argument("--overwrite-existing-local-files", action="store_true")
        parsed: argparse.Namespace = parser.parse_args(self._arguments)
        volume_name: str = cast(str, parsed.volume_name)
        remote_relative_path: Path = cast(Path, parsed.remote_relative_path)
        local_destination_root: Path = cast(Path, parsed.local_destination_root)
        overwrite_existing_local_files: bool = cast(
            bool, parsed.overwrite_existing_local_files
        )
        synchronizer: ModalArtifactSynchronizer = ModalArtifactSynchronizer(
            ModalArtifactSyncConfig(
                volume_name=volume_name,
                remote_relative_path=remote_relative_path,
                local_destination_root=local_destination_root,
                overwrite_existing_local_files=overwrite_existing_local_files
            )
        )
        synchronizer.synchronize()
        return 0


if __name__ == "__main__":
    cli: ModalArtifactSyncCli = ModalArtifactSyncCli(sys.argv[1:])
    raise SystemExit(cli.run())
