# This module:
# 1. Appends experiment result rows to the shared summary CSV under an
#    exclusive advisory lock, writing the header exactly once
# 2. Validates the schema of an existing summary before any row is written,
#    and exposes that check as a preflight runnable before expensive
#    evaluation starts
#
# Report alignment:
# - The summary surface this class appends to is the study's result
#   register, the index the report publishes beside its admission and
#   exclusion registers. Refusing a schema mismatch is what keeps that
#   register a single coherent table rather than a concatenation of
#   incompatible measurement contracts
#
# Design decisions:
# - The lock is held across the header decision and the row write, so
#   concurrent capsules on a shared filesystem cannot interleave a duplicate
#   header or a torn row. Acquiring it only around the write would leave a
#   window in which two writers both observe an empty file and both emit a
#   header
# - The lock is fcntl advisory and therefore binds only processes that also
#   take it; a cooperating fleet is the assumed deployment, and an external
#   editor appending to the same file is outside the guarantee
# - A schema mismatch is a hard error directing the run to a compatible
#   summary surface, because appending mixed schemas would silently corrupt
#   the study's index. The error names both the observed and the expected
#   column tuples, so the incompatibility is diagnosable from the failure
#   alone without opening the file
# - The schema check is exposed twice, once as a cheap preflight and once
#   inside the locked write, because the two answer different questions:
#   the preflight fails a doomed run in seconds before any measurement is
#   spent, while the locked check is the authoritative guard that also
#   covers a file another writer created in the interim
# - The summary is only an append index; the authoritative evidence remains
#   the immutable per-run capsule. Nothing here rewrites or deduplicates
#   existing rows, so a repeated run appends a second row rather than
#   replacing the first
#
# Author: Rahul Sawhney

import csv
import fcntl
import io
import os
from collections.abc import Iterator
from pathlib import Path

from vocode.loggers.result import ExperimentResultRow

__all__: list[str] = ["ExperimentResultWriter"]


class ExperimentResultWriter:
    # Locked append access to one summary CSV surface. Rows follow the
    # ExperimentResultRow schema; the writer refuses to touch a file whose
    # header disagrees with it. The instance is cheap and stateless beyond
    # the bound path, so it may be constructed per run and discarded; it
    # holds no open handle and no lock between calls, which is what allows
    # many concurrent runs to share one summary file.
    #
    # Integration: ExperimentLogger constructs one writer per run and calls
    # write_row exactly once at flush. A runner that wants the schema
    # guarantee before spending measurement time calls preflight_schema
    # first, at whatever point it considers the run committed.
    def __init__(self, experiments_csv: Path) -> None:
        # Binds the summary CSV path; no file is touched until a preflight
        # or write.
        #
        # Args:
        #     experiments_csv: Path of the shared append-only summary. It
        #         need not exist: the first write creates the file, its
        #         parent directories, and the header.
        self._experiments_csv: Path = experiments_csv

    def preflight_schema(self) -> None:
        # Validates the existing summary header before any expensive
        # evaluation starts, so a schema mismatch aborts the run in seconds
        # rather than after the measurement. Missing or empty files pass,
        # because the first write creates the header.
        # Only the first line is read and no lock is taken, which keeps the
        # check cheap enough to run unconditionally at startup. The result
        # is therefore advisory rather than binding: another writer may
        # create or replace the file afterwards, so write_row repeats the
        # check under its own lock.
        #
        # Raises:
        #     ValueError: If the file exists, is non-empty, and its header
        #         line does not match the current result-row schema exactly,
        #         including column order.
        if not self._experiments_csv.exists() or self._experiments_csv.stat().st_size == 0:
            return
        with self._experiments_csv.open(newline="", encoding="utf-8") as csv_file:
            header_line: str = csv_file.readline()
        self._validate_header_line(header_line)

    def write_row(self, result_row: ExperimentResultRow) -> None:
        # Appends one result row under an exclusive advisory lock: the header
        # is read and validated (or scheduled for writing when absent) and
        # the row is appended at the end of file, all within one lock scope,
        # so concurrent writers cannot interleave.
        #
        # The sequence is: create the parent directories, open the file in
        # append-update mode so it exists without being truncated, take the
        # exclusive lock on the descriptor, rewind to read the first line,
        # decide from it whether a header is already present and validate it
        # if so, seek back to the end, then emit the header when absent and
        # finally the row. The lock is released when the enclosing with
        # block closes the descriptor, which is also what flushes the row, so
        # no other writer can observe a file that has been appended to but
        # not yet unlocked.
        #
        # Args:
        #     result_row: The validated record to serialize. Its
        #         to_csv_row mapping is written under the schema's column
        #         order, and its None values become empty cells, preserving
        #         the distinction between a measured zero and an unproduced
        #         measurement.
        #
        # Raises:
        #     ValueError: If the file already carries a header that
        #         disagrees with the current result-row schema. The row is
        #         not written and the file is left untouched, so a
        #         mismatched summary is never partially appended to.
        #
        # Note:
        #     Nothing here deduplicates: calling this twice for one run
        #     appends two rows. ExperimentLogger, not this class, enforces
        #     the one-row-per-run rule.
        self._experiments_csv.parent.mkdir(parents=True, exist_ok=True)
        with self._experiments_csv.open("a+", newline="", encoding="utf-8") as csv_file:
            fcntl.flock(csv_file.fileno(), fcntl.LOCK_EX)
            csv_file.seek(0)
            header_line: str = csv_file.readline()
            has_header: bool = bool(header_line.strip())
            if has_header:
                self._validate_header_line(header_line)
            csv_file.seek(0, os.SEEK_END)
            writer: csv.DictWriter = csv.DictWriter(
                csv_file,
                fieldnames=list(ExperimentResultRow.csv_columns),
                extrasaction="ignore"
            )
            if not has_header:
                writer.writeheader()
            writer.writerow(result_row.to_csv_row())

    def _validate_header_line(self, header_line: str) -> None:
        # Compares one raw CSV header line against the current result-row schema.
        # The line is parsed with the csv reader rather than split on commas,
        # so a quoted column name containing a comma is compared correctly.
        # Comparison is against the schema tuple as a tuple, which makes it
        # order-sensitive: a summary carrying the same column names in a
        # different order is rejected, because the writer appends positional
        # rows and a reordered header would silently misalign every value.
        #
        # Args:
        #     header_line: The file's first line, including its trailing
        #         newline. An empty or whitespace-only line parses to an
        #         empty tuple and therefore fails the comparison; callers
        #         screen that case out before reaching here.
        #
        # Raises:
        #     ValueError: If the parsed columns differ from
        #         ExperimentResultRow.csv_columns in content or order. The
        #         message reports the summary's path, both column tuples,
        #         and the required remedy of pointing the run at a
        #         schema-compatible surface.
        header_reader: Iterator[list[str]] = csv.reader(io.StringIO(header_line))
        observed_columns: tuple[str, ...] = tuple(next(header_reader, ()))
        expected_columns: tuple[str, ...] = ExperimentResultRow.csv_columns
        if observed_columns != expected_columns:
            raise ValueError(
                f"Existing summary schema differs at {self._experiments_csv}; "
                f"observed={observed_columns}, expected={expected_columns}. "
                f"Point this run at a schema-compatible summary surface instead of "
                f"appending mixed schemas."
            )
