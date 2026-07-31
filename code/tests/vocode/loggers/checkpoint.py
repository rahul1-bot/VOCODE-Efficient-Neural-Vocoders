# This module:
# 1. Verifies that CheckpointLogger derives the per-run manifest from the live
#    ModelCheckpoint callback: the best path, the monitored metric, and the
#    selection direction are read from the callback rather than restated
# 2. Verifies the last-checkpoint rule: last.ckpt is recorded only when the file
#    exists on disk, so a manifest never points at an absent file
# 3. Verifies that the manifest JSON lands inside the run's checkpoints directory
#    and reparses into the record the writer returned
# 4. Verifies that the manifest record fails closed on unknown fields, unknown
#    label vocabularies, loose scalar types, and mutation after construction
#
# Design decisions:
# - The ModelCheckpoint callback is constructed directly and its post-fit state is
#   set on the instance, because the manifest contract is about reading a settled
#   callback; no training loop is run and no checkpoint tensor is ever written
# - Checkpoint files are fabricated as minimal text files inside a temporary
#   directory, since the writer only tests for their existence
# - Written artifacts are verified by reading the JSON back from disk and
#   asserting its exact key set and values, not by trusting the returned record
#
# Author: Rahul Sawhney

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from syntheticmind.callbacks.model_checkpoint import ModelCheckpoint

from vocode.loggers.checkpoint import CheckpointLogger, CheckpointManifest


class CheckpointCallbackFactory:
    # Builds settled ModelCheckpoint callbacks: a checkpoint directory, a
    # monitored metric with its direction, and an optional recorded best path.
    def __init__(self, checkpoint_directory: Path) -> None:
        # Binds the directory every built callback writes into.
        self._checkpoint_directory: Path = checkpoint_directory

    def with_selection(self, monitor: str, mode: Literal["min", "max"]) -> ModelCheckpoint:
        # Builds a callback that ranks by one metric in one direction and has
        # not yet selected a checkpoint.
        return ModelCheckpoint(
            dirpath=self._checkpoint_directory,
            monitor=monitor,
            mode=mode
        )

    def with_recorded_best(
        self,
        monitor: str,
        mode: Literal["min", "max"],
        best_file_name: str
    ) -> ModelCheckpoint:
        # Fabricates the kept checkpoint file and sets the post-fit best path
        # the writer reads, standing in for a completed fit.
        callback: ModelCheckpoint = self.with_selection(monitor, mode)
        best_path: Path = self._checkpoint_directory / best_file_name
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_text("checkpoint payload placeholder", encoding="utf-8")
        callback.best_path: Path | None = best_path
        return callback


class ManifestFileReader:
    # Reads a written manifest back from disk as its parsed JSON payload, so
    # assertions bind to file content rather than to writer state.
    def __init__(self, checkpoint_directory: Path) -> None:
        # Locates the manifest the writer is expected to produce.
        self._manifest_path: Path = checkpoint_directory / "project_checkpoint_manifest.json"

    @property
    def path(self) -> Path:
        # Returns where the manifest is expected on disk.
        return self._manifest_path

    @property
    def payload(self) -> dict[str, str | int | None]:
        # Returns the raw JSON mapping, so key sets and cell values are read
        # as a downloaded capsule would present them.
        return json.loads(self._manifest_path.read_text(encoding="utf-8"))

    @property
    def record(self) -> CheckpointManifest:
        # Reparses the written JSON through the record, proving the artifact
        # still satisfies its own schema.
        return CheckpointManifest.model_validate_json(
            self._manifest_path.read_text(encoding="utf-8")
        )


class CheckpointManifestWriteTest(unittest.TestCase):
    # Verifies what the writer records: manifest location, callback-derived
    # selection state, run identity, and the JSON that reaches disk.
    def setUp(self) -> None:
        # Points the run directory two levels below the temporary root and
        # deliberately creates neither it nor the checkpoints directory, so
        # the writer's own directory creation is exercised rather than
        # assumed. The reader is aimed at the manifest's fixed filename
        # before anything is written, which is what lets a test assert its
        # absence as well as its content.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._run_directory: Path = Path(self._temporary_directory.name) / "runs" / "run-0001"
        self._checkpoint_directory: Path = self._run_directory / "checkpoints"
        self._callbacks: CheckpointCallbackFactory = CheckpointCallbackFactory(
            self._checkpoint_directory
        )
        self._logger: CheckpointLogger = CheckpointLogger()
        self._reader: ManifestFileReader = ManifestFileReader(self._checkpoint_directory)

    def tearDown(self) -> None:
        # Removes the temporary root and every fabricated checkpoint file and
        # manifest written beneath it.
        self._temporary_directory.cleanup()

    def _write_manifest(self, model_checkpoint: ModelCheckpoint) -> CheckpointManifest:
        # Writes one manifest with fixed run identity, so each test varies only
        # the callback state under examination.
        #
        # Args:
        #     model_checkpoint: The settled callback whose best path,
        #         monitor, and mode the writer reads. This is the only
        #         input any case in this class varies.
        #
        # Returns:
        #     The record the writer returned, so a test can compare it
        #     against the JSON that reached disk.
        return self._logger.write(
            model_checkpoint=model_checkpoint,
            run_directory=self._run_directory,
            evidence_label="project_trained_reproduction",
            architecture_name="hifigan_v1",
            variant_name="baseline",
            seed=11,
            unique_id="hifigan_v1_seed11_20260101",
            experiment_name="reproduction_sweep",
            run_id="run-0001"
        )

    def test_manifest_is_written_inside_the_checkpoints_directory(self) -> None:
        # A downloaded capsule carries its own checkpoint index.
        self._write_manifest(self._callbacks.with_selection("val_loss", "min"))
        self.assertTrue(
            self._reader.path.exists(),
            msg="the manifest must be written next to the checkpoint files it indexes"
        )

    def test_missing_checkpoints_directory_is_created(self) -> None:
        # A run that never produced a checkpoint file still gets a manifest.
        self.assertFalse(self._checkpoint_directory.exists())
        manifest: CheckpointManifest = self._write_manifest(
            self._callbacks.with_selection("val_loss", "min")
        )
        self.assertTrue(self._checkpoint_directory.is_dir())
        self.assertEqual(manifest.checkpoint_directory, self._checkpoint_directory)

    def test_written_json_reparses_into_the_returned_record(self) -> None:
        # The returned record and the on-disk evidence are the same document.
        manifest: CheckpointManifest = self._write_manifest(
            self._callbacks.with_recorded_best("val_loss", "min", "epoch01.ckpt")
        )
        self.assertEqual(self._reader.record, manifest)

    def test_manifest_payload_carries_the_full_key_set(self) -> None:
        # The manifest documents identity, location, selection, and timestamp.
        self._write_manifest(self._callbacks.with_selection("val_loss", "min"))
        expected_keys: set[str] = {
            "evidence_label",
            "architecture_name",
            "variant_name",
            "seed",
            "unique_id",
            "experiment_name",
            "run_id",
            "checkpoint_directory",
            "best_checkpoint_path",
            "last_checkpoint_path",
            "monitor",
            "mode",
            "created_at_utc"
        }
        self.assertEqual(set(self._reader.payload.keys()), expected_keys)

    def test_run_identity_is_carried_into_the_payload(self) -> None:
        # Every manifest is attributable to the run that produced it.
        self._write_manifest(self._callbacks.with_selection("val_loss", "min"))
        payload: dict[str, str | int | None] = self._reader.payload
        self.assertEqual(payload["evidence_label"], "project_trained_reproduction")
        self.assertEqual(payload["architecture_name"], "hifigan_v1")
        self.assertEqual(payload["variant_name"], "baseline")
        self.assertEqual(payload["seed"], 11)
        self.assertEqual(payload["unique_id"], "hifigan_v1_seed11_20260101")
        self.assertEqual(payload["experiment_name"], "reproduction_sweep")
        self.assertEqual(payload["run_id"], "run-0001")

    def test_selection_criterion_is_read_from_the_live_callback(self) -> None:
        # The manifest reports what the harness actually ranked by, so a
        # configuration that never reached the callback cannot be claimed.
        manifest: CheckpointManifest = self._write_manifest(
            self._callbacks.with_selection("val_mel_error", "max")
        )
        self.assertEqual(manifest.monitor, "val_mel_error")
        self.assertEqual(manifest.mode, "max")
        payload: dict[str, str | int | None] = self._reader.payload
        self.assertEqual(payload["monitor"], "val_mel_error")
        self.assertEqual(payload["mode"], "max")

    def test_best_checkpoint_path_is_taken_from_the_callback(self) -> None:
        # The kept checkpoint is whatever the callback selected during fit.
        model_checkpoint: ModelCheckpoint = self._callbacks.with_recorded_best(
            "val_loss",
            "min",
            "epoch07.ckpt"
        )
        manifest: CheckpointManifest = self._write_manifest(model_checkpoint)
        self.assertEqual(manifest.best_checkpoint_path, model_checkpoint.best_path)
        self.assertEqual(
            self._reader.payload["best_checkpoint_path"],
            str(self._checkpoint_directory / "epoch07.ckpt")
        )

    def test_absent_best_checkpoint_is_recorded_as_null(self) -> None:
        # A run that never selected a checkpoint reports no best path.
        manifest: CheckpointManifest = self._write_manifest(
            self._callbacks.with_selection("val_loss", "min")
        )
        self.assertIsNone(manifest.best_checkpoint_path)
        self.assertIsNone(self._reader.payload["best_checkpoint_path"])

    def test_absent_last_checkpoint_is_recorded_as_null(self) -> None:
        # A bounded verification run can finish without last.ckpt; the
        # manifest must not point at a file that does not exist.
        manifest: CheckpointManifest = self._write_manifest(
            self._callbacks.with_selection("val_loss", "min")
        )
        self.assertIsNone(manifest.last_checkpoint_path)
        self.assertIsNone(self._reader.payload["last_checkpoint_path"])

    def test_present_last_checkpoint_is_recorded(self) -> None:
        # When the rolling checkpoint exists on disk it is indexed by path.
        self._checkpoint_directory.mkdir(parents=True, exist_ok=True)
        last_checkpoint_path: Path = self._checkpoint_directory / "last.ckpt"
        last_checkpoint_path.write_text("checkpoint payload placeholder", encoding="utf-8")
        manifest: CheckpointManifest = self._write_manifest(
            self._callbacks.with_selection("val_loss", "min")
        )
        self.assertEqual(manifest.last_checkpoint_path, last_checkpoint_path)
        self.assertEqual(
            self._reader.payload["last_checkpoint_path"],
            str(last_checkpoint_path)
        )

    def test_creation_timestamp_is_utc(self) -> None:
        # Manifests from different machines are comparable without local
        # timezone ambiguity.
        manifest: CheckpointManifest = self._write_manifest(
            self._callbacks.with_selection("val_loss", "min")
        )
        self.assertIsNotNone(manifest.created_at_utc.tzinfo)
        self.assertEqual(manifest.created_at_utc.utcoffset(), timedelta(0))

    def test_repeated_write_replaces_the_previous_manifest(self) -> None:
        # The manifest is a current index, not an accumulating log.
        self._write_manifest(self._callbacks.with_selection("val_loss", "min"))
        second_manifest: CheckpointManifest = self._write_manifest(
            self._callbacks.with_selection("val_mel_error", "max")
        )
        self.assertEqual(self._reader.record, second_manifest)
        self.assertEqual(self._reader.payload["monitor"], "val_mel_error")


class CheckpointManifestValidationTest(unittest.TestCase):
    # Verifies that the manifest record is frozen, closed to unknown fields,
    # strictly typed, and bound to the closed label vocabularies.
    def setUp(self) -> None:
        # Builds one fully valid manifest that every case in this class
        # mutates a single field of. Starting from a known-good record means
        # each rejection test isolates exactly one violation, so a failure
        # names the rule that broke rather than leaving several candidates.
        # No file is written here: these cases exercise the record's
        # validation, not the writer.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._checkpoint_directory: Path = Path(self._temporary_directory.name) / "checkpoints"
        self._manifest: CheckpointManifest = CheckpointManifest(
            evidence_label="project_hybrid_variants",
            architecture_name="vocos",
            variant_name="hybrid_snake",
            seed=3,
            unique_id="vocos_seed3_20260101",
            experiment_name="hybrid_sweep",
            run_id="run-0002",
            checkpoint_directory=self._checkpoint_directory,
            best_checkpoint_path=None,
            last_checkpoint_path=None,
            monitor="val_loss",
            mode="min",
            created_at_utc=datetime.fromisoformat("2026-01-01T00:00:00+00:00")
        )

    def tearDown(self) -> None:
        # Removes the temporary root and every fabricated checkpoint file and
        # manifest written beneath it.
        self._temporary_directory.cleanup()

    def test_manifest_is_immutable_after_construction(self) -> None:
        # A written manifest is evidence; it cannot be edited in place.
        with self.assertRaises(ValidationError):
            self._manifest.seed: int = 9

    def test_unknown_field_is_rejected(self) -> None:
        # The record is closed, so stray keys cannot accumulate across versions.
        manifest_fields: dict[str, str | int | None | Path | datetime] = self._manifest.model_dump()
        manifest_fields["reviewer_comment"] = "looks fine"
        with self.assertRaises(ValidationError):
            CheckpointManifest(**manifest_fields)

    def test_unknown_evidence_label_is_rejected(self) -> None:
        # Evidence labels are a closed vocabulary of the study's lanes.
        manifest_fields: dict[str, str | int | None | Path | datetime] = self._manifest.model_dump()
        manifest_fields["evidence_label"] = "published_checkpoint_evaluation"
        with self.assertRaises(ValidationError):
            CheckpointManifest(**manifest_fields)

    def test_unknown_selection_direction_is_rejected(self) -> None:
        # The monitored direction is either minimization or maximization.
        manifest_fields: dict[str, str | int | None | Path | datetime] = self._manifest.model_dump()
        manifest_fields["mode"] = "minimum"
        with self.assertRaises(ValidationError):
            CheckpointManifest(**manifest_fields)

    def test_string_seed_is_rejected_under_strict_validation(self) -> None:
        # Strict validation refuses coercion, so identity fields cannot drift
        # into stringified form.
        manifest_fields: dict[str, str | int | None | Path | datetime] = self._manifest.model_dump()
        manifest_fields["seed"] = "3"
        with self.assertRaises(ValidationError):
            CheckpointManifest(**manifest_fields)

    def test_missing_selection_criterion_is_rejected(self) -> None:
        # A manifest without its ranking criterion cannot justify its best path.
        manifest_fields: dict[str, str | int | None | Path | datetime] = self._manifest.model_dump()
        del manifest_fields["monitor"]
        with self.assertRaises(ValidationError):
            CheckpointManifest(**manifest_fields)
