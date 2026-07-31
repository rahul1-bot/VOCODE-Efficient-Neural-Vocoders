# This module:
# 1. Provides the Modal volume inventory and upload operations: listing
#    volume contents, uploading published weights and pretrained assets,
#    and verifying what cloud runs will find
#
# Design decisions:
# - Volume state is inspected and mutated only through these explicit
#   commands, so cloud storage never drifts through ad-hoc side effects
#
# Author: Rahul Sawhney

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import ClassVar, cast

from loguru import logger as log
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

__all__: list[str] = [
    "ModalStorageCli",
    "ModalStorageClient",
    "ModalVolumeEntry",
    "ModalVolumeInventoryConfig",
    "ModalVolumeInventoryReport",
    "ModalVolumePathInventory",
    "ModalVolumeUploadConfig"
]


class ModalVolumeEntry(BaseModel):
    # Structured record describing one file or directory discovered inside a Modal volume.
    # Inventory records keep cloud storage reviewable without downloading large runtime assets.
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", populate_by_name=True)
    filename: str = Field(alias="Filename")
    entry_type: str = Field(alias="Type")
    created_or_modified: str = Field(alias="Created/Modified")
    size: str = Field(alias="Size")


class ModalVolumePathInventory(BaseModel):
    # Inventory section for one inspected Modal volume path.
    # It records entries and command errors so cloud-state assumptions remain explicit.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    remote_path: Path
    entries: list[ModalVolumeEntry]


class ModalVolumeInventoryReport(BaseModel):
    # Complete Modal volume inventory report written as lightweight project evidence.
    # The report documents datasets, pretrained caches, run outputs, and artifact mirrors.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    volume_name: str
    inventories: list[ModalVolumePathInventory]


class ModalVolumeInventoryConfig(BaseModel):
    # Configuration for Modal volume inventory collection.
    # Explicit fields define which volume paths are inspected and where the report is written.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    volume_name: str
    remote_paths: list[Path]
    environment_name: str | None = None


class ModalVolumeUploadConfig(BaseModel):
    # Configuration for uploading a local path into the Modal volume.
    # Upload operations are explicit so cloud state changes are auditable.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    volume_name: str
    local_path: Path
    remote_path: Path
    force: bool = False
    environment_name: str | None = None


class ModalStorageClient:
    # Client wrapper around Modal CLI storage operations.
    # The class centralizes subprocess command construction for inventory and upload workflows.
    _entry_list_adapter: ClassVar[TypeAdapter[list[ModalVolumeEntry]]] = TypeAdapter(
        list[ModalVolumeEntry]
    )

    def __init__(self) -> None:
        # Binds the collaborators this component uses.
        self._modal_executable: str = self._resolve_modal_executable()

    def create_inventory_report(
        self, configuration: ModalVolumeInventoryConfig
    ) -> ModalVolumeInventoryReport:
        # Collects a structured inventory report for configured Modal volume paths.
        inventories: list[ModalVolumePathInventory] = []
        remote_path_index: int = 0
        while remote_path_index < len(configuration.remote_paths):
            remote_path: Path = configuration.remote_paths[remote_path_index]
            entries: list[ModalVolumeEntry] = self.list_volume_path(
                volume_name=configuration.volume_name,
                remote_path=remote_path,
                environment_name=configuration.environment_name
            )
            inventories.append(
                ModalVolumePathInventory(remote_path=remote_path, entries=entries)
            )
            remote_path_index += 1
        report: ModalVolumeInventoryReport = ModalVolumeInventoryReport(
            volume_name=configuration.volume_name, inventories=inventories
        )
        return report

    def write_inventory_report(
        self, report: ModalVolumeInventoryReport, output_path: Path
    ) -> None:
        # Writes the Modal inventory report to a local JSON evidence file.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        log.info(f"Wrote Modal inventory report to {output_path}")

    def list_volume_path(
        self, volume_name: str, remote_path: Path, environment_name: str | None = None
    ) -> list[ModalVolumeEntry]:
        # Lists one Modal volume path through the Modal CLI and captures structured output.
        command_arguments: list[str] = [
            self._modal_executable,
            "volume",
            "ls",
            "--json"
        ]
        if environment_name is not None:
            command_arguments.extend(["--env", environment_name])
        command_arguments.extend([
            volume_name,
            self._format_remote_path(remote_path)
        ])
        log.info(f"Invoking: {' '.join(command_arguments)}")
        completed_process: subprocess.CompletedProcess[str] = subprocess.run(
            command_arguments, check=False, capture_output=True, text=True
        )
        if completed_process.returncode != 0:
            raise RuntimeError(
                "modal volume ls failed with code "
                f"{completed_process.returncode}: {completed_process.stderr}"
            )
        entries: list[ModalVolumeEntry] = self._entry_list_adapter.validate_json(
            completed_process.stdout
        )
        return entries

    def upload_path(self, configuration: ModalVolumeUploadConfig) -> None:
        # Uploads a local path into the configured Modal volume destination.
        if not configuration.local_path.exists():
            raise FileNotFoundError(f"Local path does not exist: {configuration.local_path}")
        command_arguments: list[str] = [
            self._modal_executable,
            "volume",
            "put"
        ]
        if configuration.force:
            command_arguments.append("--force")
        if configuration.environment_name is not None:
            command_arguments.extend(["--env", configuration.environment_name])
        command_arguments.extend([
            configuration.volume_name,
            str(configuration.local_path),
            self._format_remote_path(configuration.remote_path)
        ])
        log.info(f"Invoking: {' '.join(command_arguments)}")
        completed_process: subprocess.CompletedProcess[str] = subprocess.run(
            command_arguments, check=False, capture_output=True, text=True
        )
        if completed_process.returncode != 0:
            raise RuntimeError(
                "modal volume put failed with code "
                f"{completed_process.returncode}: {completed_process.stderr}"
            )
        log.info(completed_process.stdout.strip())

    def _resolve_modal_executable(self) -> str:
        # Resolves the Modal executable used for subprocess-based storage commands.
        candidate: str | None = shutil.which("modal")
        if candidate is None:
            raise FileNotFoundError(
                "The `modal` CLI executable was not found on PATH. Install the modal "
                "package and ensure its console script is reachable."
            )
        return candidate

    def _format_remote_path(self, remote_path: Path) -> str:
        # Formats a Modal volume path with a single leading slash for CLI calls.
        remote_path_text: str = str(remote_path)
        if not remote_path_text.startswith("/"):
            remote_path_text: str = f"/{remote_path_text}"
        return remote_path_text


class ModalStorageCli:
    # Command-line interface for Modal storage maintenance tasks.
    # It exposes inventory and upload commands without hiding cloud-side state changes.
    def __init__(self, arguments: list[str]) -> None:
        # Binds the collaborators this component uses.
        self._arguments: list[str] = arguments

    def run(self) -> int:
        # Executes the requested Modal utility command from parsed CLI arguments.
        parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="modal-storage")
        subparsers: argparse._SubParsersAction[argparse.ArgumentParser] = (
            parser.add_subparsers(dest="command", required=True)
        )

        inventory_parser: argparse.ArgumentParser = subparsers.add_parser("inventory")
        inventory_parser.add_argument("--volume-name", type=str, default="vocode-data")
        inventory_parser.add_argument("--remote-path", type=Path, action="append")
        inventory_parser.add_argument("--output-path", type=Path)
        inventory_parser.add_argument("--environment-name", type=str)

        upload_parser: argparse.ArgumentParser = subparsers.add_parser("upload")
        upload_parser.add_argument("--volume-name", type=str, default="vocode-data")
        upload_parser.add_argument("--local-path", type=Path, required=True)
        upload_parser.add_argument("--remote-path", type=Path, required=True)
        upload_parser.add_argument("--force", action="store_true")
        upload_parser.add_argument("--environment-name", type=str)

        parsed: argparse.Namespace = parser.parse_args(self._arguments)
        parsed_command: str = cast(str, parsed.command)
        client: ModalStorageClient = ModalStorageClient()
        if parsed_command == "inventory":
            remote_paths: list[Path] | None = cast(list[Path] | None, parsed.remote_path)
            if remote_paths is None:
                remote_paths: list[Path] | None = [
                    Path("/datasets"),
                    Path("/pretrained"),
                    Path("/experiment_artifacts")
                ]
            volume_name: str = cast(str, parsed.volume_name)
            output_path: Path | None = cast(Path | None, parsed.output_path)
            environment_name: str | None = cast(str | None, parsed.environment_name)
            report: ModalVolumeInventoryReport = client.create_inventory_report(
                ModalVolumeInventoryConfig(
                    volume_name=volume_name,
                    remote_paths=remote_paths,
                    environment_name=environment_name
                )
            )
            if output_path is not None:
                client.write_inventory_report(report=report, output_path=output_path)
            sys.stdout.write(f"{report.model_dump_json(indent=2)}\n")
            return 0
        if parsed_command == "upload":
            volume_name: str = cast(str, parsed.volume_name)
            local_path: Path = cast(Path, parsed.local_path)
            remote_path: Path = cast(Path, parsed.remote_path)
            force: bool = cast(bool, parsed.force)
            environment_name: str | None = cast(str | None, parsed.environment_name)
            client.upload_path(
                ModalVolumeUploadConfig(
                    volume_name=volume_name,
                    local_path=local_path,
                    remote_path=remote_path,
                    force=force,
                    environment_name=environment_name
                )
            )
            return 0
        raise ValueError(f"Unsupported command: {parsed_command}")


if __name__ == "__main__":
    cli: ModalStorageCli = ModalStorageCli(sys.argv[1:])
    raise SystemExit(cli.run())
