# This module:
# 1. Verifies the locked-schema preflight of the summary CSV writer: missing and
#    empty surfaces pass, a matching header passes, any header that differs from
#    the result-row schema is rejected, and every accepting path leaves the
#    inspected surface exactly as it found it
# 2. Verifies that a rejected schema fails closed: the incompatible file is left
#    byte-for-byte unchanged instead of being rewritten or appended to
# 3. Verifies the append protocol: parent directories are created, the header is
#    written exactly once, historical rows survive, and absent measurements land
#    as empty cells
#
# Design decisions:
# - Incompatible historical surfaces are fabricated as small CSV files with
#   hand-written headers (truncated, permuted, extended), because the rejection
#   path is exactly what protects real first-schema evidence files
# - Written artifacts are verified by reading the file back and asserting the
#   exact header tuple and the parsed row mapping, not by trusting the return
#   value of the writer
# - The advisory-lock behavior under genuine concurrency is not exercised here:
#   it would require launching competing processes, which the suite forbids; the
#   single-writer header decision and row append inside one lock scope are what
#   these tests bound
#
# Author: Rahul Sawhney

import csv
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path

from vocode.loggers.result import ExperimentResultRow
from vocode.loggers.writer import ExperimentResultWriter


class ResultRowFactory:
    # Builds ExperimentResultRow records with fixed identity arguments so the
    # writer tests vary only the metric buffer and the target file.
    def __init__(self) -> None:
        # Fixes the architecture identity every built row carries.
        self._architecture_name: str = "hifigan_v1"

    def with_unique_id(self, unique_id: str) -> ExperimentResultRow:
        # Builds one row whose identifier distinguishes it in the written file,
        # leaving every other field fixed.
        # The metric buffer names only three measurements out of the panel's
        # many optional columns, which is deliberate: it makes every row
        # carry both populated and absent cells, so the same fixture proves
        # value round-tripping and empty-cell rendering at once.
        #
        # Args:
        #     unique_id: Value written into the identity column, and the
        #         only thing distinguishing two rows from this factory. Row
        #         ordering assertions read this column back.
        #
        # Returns:
        #     A validated row whose date field is stamped at construction,
        #     so two rows built in one test are not guaranteed to differ in
        #     their timestamp and must be told apart by identifier.
        return ExperimentResultRow.from_metric_buffer(
            unique_id=unique_id,
            architecture_name=self._architecture_name,
            dataset_name="ljspeech",
            variant_name="project_trained_reproduction_baseline",
            seed=0,
            hyperparameters_summary="learning_rate=0.0002",
            interpretation_notes="capsule note",
            metric_buffer={"pesq": 3.25, "stoi": 0.91, "parameter_count": 13500000.0}
        )


class SummaryCsvFabricator:
    # Writes small fabricated summary files with chosen headers, standing in for
    # historical and legacy surfaces the writer may encounter on disk.
    def __init__(self, root: Path) -> None:
        # Binds the temporary root every fabricated surface is written under.
        self._root: Path = root

    def with_header(self, file_name: str, columns: tuple[str, ...]) -> Path:
        # Writes a header-only surface, which is what the preflight examines.
        target: Path = self._root / file_name
        target.write_text(f"{','.join(columns)}\n", encoding="utf-8")
        return target

    def with_header_and_row(
        self,
        file_name: str,
        columns: tuple[str, ...],
        row_values: tuple[str, ...]
    ) -> Path:
        # Writes a surface that already carries history, so a rejected write
        # can be shown to leave existing evidence intact.
        target: Path = self._root / file_name
        target.write_text(
            f"{','.join(columns)}\n{','.join(row_values)}\n",
            encoding="utf-8"
        )
        return target

    def empty(self, file_name: str) -> Path:
        # Writes a zero-byte surface, which carries no schema claim.
        target: Path = self._root / file_name
        target.write_text("", encoding="utf-8")
        return target

    def historical_row_values(self) -> tuple[str, ...]:
        # Builds a schema-shaped row identifiable by its unique_id cell, with
        # every measurement cell left empty.
        # Deriving the tuple from csv_columns rather than hard-coding it
        # keeps this fixture valid when the schema grows, so the
        # preserved-history test cannot silently degrade into writing a
        # malformed historical row.
        #
        # Returns:
        #     One cell per schema column in schema order, carrying the
        #     marker "historical" in the identity column so the row remains
        #     recognizable after the writer appends beneath it.
        return tuple(
            "historical" if column_name == "unique_id" else ""
            for column_name in ExperimentResultRow.csv_columns
        )


class SummaryFileReader:
    # Reads a summary CSV back from disk as its raw header line and its parsed
    # row mappings, so assertions bind to file content rather than to writer state.
    def __init__(self, summary_path: Path) -> None:
        # Binds the summary file the writer is expected to append to.
        self._summary_path: Path = summary_path

    @property
    def header_columns(self) -> tuple[str, ...]:
        # Returns the first line as parsed columns; an empty file reads as no
        # columns rather than raising.
        with self._summary_path.open(newline="", encoding="utf-8") as summary_file:
            header_reader: Iterator[list[str]] = csv.reader(summary_file)
            return tuple(next(header_reader, ()))

    @property
    def rows(self) -> list[dict[str, str]]:
        # Returns every data row keyed by column name.
        with self._summary_path.open(newline="", encoding="utf-8") as summary_file:
            return list(csv.DictReader(summary_file))

    @property
    def line_count(self) -> int:
        # Counts physical lines, which is how a repeated header is detected.
        return len(self._summary_path.read_text(encoding="utf-8").splitlines())


class SummarySchemaPreflightTest(unittest.TestCase):
    # Verifies the cheap schema check that runs before expensive evaluation:
    # which existing surfaces are accepted and which abort the run.
    def setUp(self) -> None:
        # Gives each case its own temporary root and fabricator, so a
        # surface written by one test is invisible to the others and no case
        # depends on execution order.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._fabricator: SummaryCsvFabricator = SummaryCsvFabricator(self._root)

    def tearDown(self) -> None:
        # Removes the temporary root and everything under it, including any
        # partial file a failing write may have left behind.
        self._temporary_directory.cleanup()

    def test_missing_summary_passes_preflight(self) -> None:
        # The first write creates the header, so an absent surface is valid.
        # The check is read-only, so it must not materialize the file either.
        absent_path: Path = self._root / "absent" / "experiments.csv"
        writer: ExperimentResultWriter = ExperimentResultWriter(experiments_csv=absent_path)
        writer.preflight_schema()
        self.assertFalse(
            absent_path.exists(),
            msg="preflight inspects the surface and must not create it"
        )

    def test_empty_summary_passes_preflight(self) -> None:
        # A zero-byte file carries no schema claim and is adopted by the first write.
        empty_path: Path = self._fabricator.empty("experiments.csv")
        writer: ExperimentResultWriter = ExperimentResultWriter(experiments_csv=empty_path)
        writer.preflight_schema()
        self.assertEqual(
            empty_path.stat().st_size,
            0,
            msg="preflight must not write a header into the surface it checked"
        )

    def test_matching_header_passes_preflight(self) -> None:
        # A surface written under the current schema is safe to append to, and
        # the check leaves it exactly as it found it.
        matching_path: Path = self._fabricator.with_header(
            "experiments.csv",
            ExperimentResultRow.csv_columns
        )
        content_before: str = matching_path.read_text(encoding="utf-8")
        writer: ExperimentResultWriter = ExperimentResultWriter(experiments_csv=matching_path)
        writer.preflight_schema()
        self.assertEqual(matching_path.read_text(encoding="utf-8"), content_before)

    def test_legacy_header_is_rejected_with_remediation_guidance(self) -> None:
        # A first-schema summary aborts the run in seconds and names both the
        # observed and the expected schema.
        legacy_path: Path = self._fabricator.with_header(
            "experiments_legacy.csv",
            ("unique_id", "date", "architecture_name", "pesq", "stoi")
        )
        writer: ExperimentResultWriter = ExperimentResultWriter(experiments_csv=legacy_path)
        with self.assertRaisesRegex(ValueError, "schema-compatible summary surface"):
            writer.preflight_schema()

    def test_permuted_header_is_rejected(self) -> None:
        # Column order is part of the schema, because the writer serializes
        # positionally against it.
        permuted_columns: tuple[str, ...] = (
            ExperimentResultRow.csv_columns[1],
            ExperimentResultRow.csv_columns[0]
        ) + ExperimentResultRow.csv_columns[2:]
        permuted_path: Path = self._fabricator.with_header("experiments_permuted.csv", permuted_columns)
        writer: ExperimentResultWriter = ExperimentResultWriter(experiments_csv=permuted_path)
        with self.assertRaisesRegex(ValueError, "Existing summary schema differs"):
            writer.preflight_schema()

    def test_header_with_an_additional_column_is_rejected(self) -> None:
        # A superset schema is still a different schema and cannot be appended to.
        extended_columns: tuple[str, ...] = tuple(
            [*ExperimentResultRow.csv_columns, "reviewer_comment"]
        )
        extended_path: Path = self._fabricator.with_header("experiments_extended.csv", extended_columns)
        writer: ExperimentResultWriter = ExperimentResultWriter(experiments_csv=extended_path)
        with self.assertRaisesRegex(ValueError, "Existing summary schema differs"):
            writer.preflight_schema()

    def test_rejection_names_the_offending_file(self) -> None:
        # The error points the operator at the exact surface to replace.
        legacy_path: Path = self._fabricator.with_header(
            "experiments_named.csv",
            ("unique_id", "date", "pesq")
        )
        writer: ExperimentResultWriter = ExperimentResultWriter(experiments_csv=legacy_path)
        with self.assertRaises(ValueError) as raised:
            writer.preflight_schema()
        self.assertIn(
            str(legacy_path),
            str(raised.exception),
            msg="the schema error must identify the summary file it examined"
        )


class SummaryRowAppendTest(unittest.TestCase):
    # Verifies the append protocol against a fresh summary surface: directory
    # creation, single header emission, and round-tripped cell values.
    def setUp(self) -> None:
        # Targets a summary path one directory below the temporary root and
        # deliberately does not create that directory, so every case in this
        # class starts from a surface that does not yet exist and the first
        # write is exercised as a genuine creation.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._summary_path: Path = self._root / "summary" / "experiments.csv"
        self._writer: ExperimentResultWriter = ExperimentResultWriter(
            experiments_csv=self._summary_path
        )
        self._factory: ResultRowFactory = ResultRowFactory()
        self._reader: SummaryFileReader = SummaryFileReader(self._summary_path)

    def tearDown(self) -> None:
        # Removes the temporary root and everything under it, including any
        # partial file a failing write may have left behind.
        self._temporary_directory.cleanup()

    def test_first_write_creates_the_parent_directory(self) -> None:
        # A run never has to prepare the summary directory itself.
        self.assertFalse(self._summary_path.parent.exists())
        self._writer.write_row(self._factory.with_unique_id("run_a"))
        self.assertTrue(
            self._summary_path.exists(),
            msg="the writer must create missing parent directories before appending"
        )

    def test_written_header_is_the_locked_schema(self) -> None:
        # The header on disk is the schema authority, verified by reading it back.
        self._writer.write_row(self._factory.with_unique_id("run_a"))
        self.assertEqual(self._reader.header_columns, ExperimentResultRow.csv_columns)

    def test_second_write_appends_without_repeating_the_header(self) -> None:
        # The summary is an append-only index over immutable capsules.
        self._writer.write_row(self._factory.with_unique_id("run_a"))
        self._writer.write_row(self._factory.with_unique_id("run_b"))
        self.assertEqual(self._reader.line_count, 3, msg="expected one header line and two rows")
        recorded_identifiers: list[str] = [row["unique_id"] for row in self._reader.rows]
        self.assertEqual(recorded_identifiers, ["run_a", "run_b"])

    def test_a_separate_writer_instance_appends_to_the_same_surface(self) -> None:
        # Independent capsules share one index without re-emitting the header.
        self._writer.write_row(self._factory.with_unique_id("run_a"))
        second_writer: ExperimentResultWriter = ExperimentResultWriter(
            experiments_csv=self._summary_path
        )
        second_writer.write_row(self._factory.with_unique_id("run_b"))
        self.assertEqual(self._reader.header_columns, ExperimentResultRow.csv_columns)
        self.assertEqual(len(self._reader.rows), 2)

    def test_measured_values_round_trip_through_the_file(self) -> None:
        # Cell values are read back from disk, not taken from the record.
        self._writer.write_row(self._factory.with_unique_id("run_a"))
        recorded_row: dict[str, str] = self._reader.rows[0]
        self.assertEqual(recorded_row["unique_id"], "run_a")
        self.assertEqual(recorded_row["architecture_name"], "hifigan_v1")
        self.assertEqual(float(recorded_row["pesq"]), 3.25)
        self.assertEqual(int(recorded_row["parameter_count"]), 13500000)

    def test_absent_measurements_are_written_as_empty_cells(self) -> None:
        # An unproduced measurement reaches the file as an empty cell rather
        # than as a fabricated zero.
        self._writer.write_row(self._factory.with_unique_id("run_a"))
        recorded_row: dict[str, str] = self._reader.rows[0]
        self.assertEqual(recorded_row["mcd"], "")
        self.assertEqual(recorded_row["utmos_strong"], "")
        self.assertEqual(recorded_row["latency_p95_ms"], "")

    def test_preflight_after_a_write_accepts_the_written_surface(self) -> None:
        # The writer's own output satisfies its own schema check, and the check
        # neither appends a row nor rewrites the header.
        self._writer.write_row(self._factory.with_unique_id("run_a"))
        self._writer.preflight_schema()
        self.assertEqual(self._reader.header_columns, ExperimentResultRow.csv_columns)
        self.assertEqual(len(self._reader.rows), 1)


class SummaryIncompatibleSchemaWriteTest(unittest.TestCase):
    # Verifies that the write path refuses incompatible surfaces and leaves
    # historical evidence untouched, while compatible surfaces keep their rows.
    def setUp(self) -> None:
        # Pairs a fabricator for the pre-existing surface with a row factory
        # for the attempted append, which is the combination every case in
        # this class needs to drive a write against a file it did not write.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._fabricator: SummaryCsvFabricator = SummaryCsvFabricator(self._root)
        self._factory: ResultRowFactory = ResultRowFactory()

    def tearDown(self) -> None:
        # Removes the temporary root and everything under it, including any
        # partial file a failing write may have left behind.
        self._temporary_directory.cleanup()

    def test_write_into_a_legacy_summary_raises(self) -> None:
        # A mixed-schema append would corrupt the study index, so the write aborts.
        legacy_path: Path = self._fabricator.with_header_and_row(
            "experiments_legacy.csv",
            ("unique_id", "date", "pesq"),
            ("first_schema_run", "2025-01-01T00:00:00+00:00", "3.1")
        )
        writer: ExperimentResultWriter = ExperimentResultWriter(experiments_csv=legacy_path)
        with self.assertRaisesRegex(ValueError, "Existing summary schema differs"):
            writer.write_row(self._factory.with_unique_id("run_a"))

    def test_rejected_write_leaves_the_legacy_file_unchanged(self) -> None:
        # Failing closed means the historical surface is never rewritten.
        legacy_path: Path = self._fabricator.with_header_and_row(
            "experiments_legacy.csv",
            ("unique_id", "date", "pesq"),
            ("first_schema_run", "2025-01-01T00:00:00+00:00", "3.1")
        )
        content_before: str = legacy_path.read_text(encoding="utf-8")
        writer: ExperimentResultWriter = ExperimentResultWriter(experiments_csv=legacy_path)
        with self.assertRaises(ValueError):
            writer.write_row(self._factory.with_unique_id("run_a"))
        self.assertEqual(
            legacy_path.read_text(encoding="utf-8"),
            content_before,
            msg="an incompatible summary must survive a rejected write byte for byte"
        )

    def test_write_into_a_compatible_historical_summary_preserves_existing_rows(self) -> None:
        # A schema-compatible surface accepts the append and keeps its history.
        historical_path: Path = self._fabricator.with_header_and_row(
            "experiments.csv",
            ExperimentResultRow.csv_columns,
            self._fabricator.historical_row_values()
        )
        writer: ExperimentResultWriter = ExperimentResultWriter(experiments_csv=historical_path)
        writer.write_row(self._factory.with_unique_id("run_a"))
        reader: SummaryFileReader = SummaryFileReader(historical_path)
        recorded_identifiers: list[str] = [row["unique_id"] for row in reader.rows]
        self.assertEqual(recorded_identifiers, ["historical", "run_a"])
        self.assertEqual(reader.header_columns, ExperimentResultRow.csv_columns)
