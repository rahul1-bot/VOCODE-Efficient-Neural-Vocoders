# This module:
# 1. Verifies that ExperimentResultRow is the single schema authority behind one
#    summary CSV row: the column tuple, the declared field order, and the CSV
#    serialization mapping describe one identical schema
# 2. Verifies the metric-buffer conversion: identity fields are carried through,
#    absent measurements stay absent, measured zeros survive, and count columns
#    are narrowed to integers
# 3. Verifies that the record fails closed on unknown fields, loose scalar types,
#    and mutation after construction
#
# Design decisions:
# - The measurement column list is derived from the schema tuple instead of being
#   restated here, so a buffer key that stops matching its column surfaces in the
#   full-panel test rather than silently reading as an absent measurement
# - The construction timestamp is asserted through its parsed structure (UTC
#   offset, second resolution) rather than a fixed instant, because the record
#   stamps itself at the moment of construction
# - No file system is touched in this file; CSV file behavior belongs to the
#   writer and is verified there
#
# Author: Rahul Sawhney

import unittest
from datetime import datetime, timedelta

from pydantic import ValidationError

from vocode.loggers.result import ExperimentResultRow


class ResultRowSchema:
    # Partitions the locked column tuple into the identity columns supplied by
    # the caller and the measurement columns read from the metric buffer.
    def __init__(self) -> None:
        # Names the columns the caller supplies; everything else in the schema
        # must therefore come from the metric buffer.
        self._identity_columns: tuple[str, ...] = (
            "unique_id",
            "date",
            "architecture_name",
            "dataset_name",
            "variant_name",
            "seed",
            "hyperparameters_summary",
            "interpretation_notes"
        )

    @property
    def identity_columns(self) -> tuple[str, ...]:
        # Returns the caller-supplied columns.
        return self._identity_columns

    @property
    def measurement_columns(self) -> tuple[str, ...]:
        # Derives the measurement columns from the live schema rather than
        # restating them, so a renamed column surfaces in the full-panel test.
        return tuple(
            column_name
            for column_name in ExperimentResultRow.csv_columns
            if column_name not in self._identity_columns
        )


class MetricBufferBuilder:
    # Builds metric buffers for row construction: a full panel covering every
    # measurement column and a sparse panel covering a chosen subset.
    def __init__(self) -> None:
        # Binds the schema partition the panels are built against.
        self._schema: ResultRowSchema = ResultRowSchema()

    def full_panel(self) -> dict[str, float | int]:
        # Gives every measurement column a distinct value, so a value landing
        # in the wrong column is detectable rather than coincidentally equal.
        # Values are the one-based column positions, which guarantees
        # distinctness without hand-maintaining a table and keeps every value
        # positive, so no entry can be confused with an absent measurement.
        #
        # Returns:
        #     A buffer covering every measurement column the live schema
        #     declares, derived from that schema rather than restated, so a
        #     newly added column is exercised automatically.
        return {
            column_name: float(position + 1)
            for position, column_name in enumerate(self._schema.measurement_columns)
        }

    def sparse_panel(self) -> dict[str, float | int]:
        # Covers the interesting conversion cases: an ordinary float, a
        # measured zero, and two counts that must narrow to integers.
        # The zero is the load-bearing entry, because it is the one value
        # that a naive absence check would discard; the two counts differ in
        # how they arrive, one as a float and one already an integer, so both
        # narrowing paths are exercised. The columns left unnamed are what
        # the absence assertions read.
        #
        # Returns:
        #     A deliberately partial buffer, so a row built from it carries
        #     populated and absent columns at once.
        return {"pesq": 3.25, "stoi": 0.0, "parameter_count": 13500000.0, "pesq_failure_count": 2}


class ResultRowFactory:
    # Builds ExperimentResultRow records with fixed identity arguments so tests
    # vary only the metric buffer under examination.
    def __init__(self) -> None:
        # Fixes the identity arguments every built row carries.
        self._unique_id: str = "hifigan_v1_seed0_20260101"
        self._architecture_name: str = "hifigan_v1"
        self._dataset_name: str = "ljspeech"
        self._variant_name: str = "project_trained_reproduction_baseline"
        self._seed: int = 11
        self._hyperparameters_summary: str = "learning_rate=0.0002;batch_size=16"
        self._interpretation_notes: str = "smoke capsule"

    def from_buffer(self, metric_buffer: dict[str, float | int]) -> ExperimentResultRow:
        # Converts one metric buffer under the fixed identity arguments.
        return ExperimentResultRow.from_metric_buffer(
            unique_id=self._unique_id,
            architecture_name=self._architecture_name,
            dataset_name=self._dataset_name,
            variant_name=self._variant_name,
            seed=self._seed,
            hyperparameters_summary=self._hyperparameters_summary,
            interpretation_notes=self._interpretation_notes,
            metric_buffer=metric_buffer
        )

    @property
    def unique_id(self) -> str:
        # Returns the identity value the row is expected to carry, so the
        # assertion compares against the argument rather than a repeated literal.
        return self._unique_id

    @property
    def architecture_name(self) -> str:
        # Returns the architecture argument every built row receives.
        return self._architecture_name

    @property
    def dataset_name(self) -> str:
        # Returns the dataset argument every built row receives.
        return self._dataset_name

    @property
    def variant_name(self) -> str:
        # Returns the variant argument every built row receives.
        return self._variant_name

    @property
    def seed(self) -> int:
        # Returns the seed argument every built row receives.
        return self._seed

    @property
    def hyperparameters_summary(self) -> str:
        # Returns the compact hyperparameter summary every built row receives.
        return self._hyperparameters_summary

    @property
    def interpretation_notes(self) -> str:
        # Returns the analyst note every built row receives.
        return self._interpretation_notes


class ExperimentResultSchemaAuthorityTest(unittest.TestCase):
    # Verifies that the column tuple, the declared model fields, and the CSV
    # mapping agree on membership and on order.
    def setUp(self) -> None:
        # Builds the schema partition, a row factory, and a buffer builder.
        # None of these touch the file system or hold state between cases, so
        # the fixture is rebuilt per test purely for isolation rather than
        # because anything here is expensive.
        self._schema: ResultRowSchema = ResultRowSchema()
        self._factory: ResultRowFactory = ResultRowFactory()
        self._buffers: MetricBufferBuilder = MetricBufferBuilder()

    def test_csv_columns_follow_the_declared_field_order(self) -> None:
        # The schema tuple is exactly the model field order, so serialization
        # order cannot drift away from the record definition.
        declared_fields: tuple[str, ...] = tuple(ExperimentResultRow.model_fields)
        self.assertEqual(
            declared_fields,
            ExperimentResultRow.csv_columns,
            msg="csv_columns must mirror the declared field order of the record"
        )

    def test_csv_columns_contain_no_duplicates(self) -> None:
        # A duplicated column would make one field silently overwrite another
        # during serialization.
        unique_columns: set[str] = set(ExperimentResultRow.csv_columns)
        self.assertEqual(
            len(unique_columns),
            len(ExperimentResultRow.csv_columns),
            msg="every schema column must appear exactly once"
        )

    def test_csv_row_keys_match_the_schema_order(self) -> None:
        # The serialized mapping is keyed by the schema tuple in schema order.
        row: ExperimentResultRow = self._factory.from_buffer(self._buffers.sparse_panel())
        csv_row: dict[str, str | int | float | None] = row.to_csv_row()
        self.assertEqual(
            tuple(csv_row.keys()),
            ExperimentResultRow.csv_columns,
            msg="to_csv_row must emit exactly the schema columns in schema order"
        )

    def test_identity_and_measurement_columns_partition_the_schema(self) -> None:
        # Every schema column is either supplied identity or read measurement,
        # so no column can be left without a source.
        partitioned: tuple[str, ...] = self._schema.identity_columns + self._schema.measurement_columns
        self.assertEqual(
            sorted(partitioned),
            sorted(ExperimentResultRow.csv_columns),
            msg="identity and measurement columns must cover the schema exactly once each"
        )


class ExperimentResultBufferConversionTest(unittest.TestCase):
    # Verifies how a run's metric buffer becomes a typed row: key coverage,
    # absence handling, numeric narrowing, and the construction timestamp.
    def setUp(self) -> None:
        # Builds the schema partition alongside the factory and buffer
        # builder, because the cases here assert over the measurement columns
        # as a derived set rather than naming them, which is what keeps the
        # coverage assertions honest as the schema grows.
        self._schema: ResultRowSchema = ResultRowSchema()
        self._factory: ResultRowFactory = ResultRowFactory()
        self._buffers: MetricBufferBuilder = MetricBufferBuilder()

    def test_full_panel_buffer_populates_every_measurement_column(self) -> None:
        # Every measurement column is read under its own column name, so a
        # mistyped buffer key cannot masquerade as an unproduced measurement.
        row: ExperimentResultRow = self._factory.from_buffer(self._buffers.full_panel())
        csv_row: dict[str, str | int | float | None] = row.to_csv_row()
        unpopulated_columns: list[str] = [
            column_name
            for column_name in self._schema.measurement_columns
            if csv_row[column_name] is None
        ]
        self.assertEqual(
            unpopulated_columns,
            [],
            msg="a fully populated buffer must leave no measurement column empty"
        )

    def test_full_panel_values_are_carried_without_reordering(self) -> None:
        # Each column receives the value stored under its own buffer key.
        metric_buffer: dict[str, float | int] = self._buffers.full_panel()
        row: ExperimentResultRow = self._factory.from_buffer(metric_buffer)
        csv_row: dict[str, str | int | float | None] = row.to_csv_row()
        mismatched_columns: list[str] = [
            column_name
            for column_name in self._schema.measurement_columns
            if csv_row[column_name] != metric_buffer[column_name]
        ]
        self.assertEqual(
            mismatched_columns,
            [],
            msg="measurement values must land in the column named by their buffer key"
        )

    def test_absent_measurements_are_recorded_as_none(self) -> None:
        # An unproduced measurement stays absent instead of becoming a
        # fabricated zero.
        row: ExperimentResultRow = self._factory.from_buffer(self._buffers.sparse_panel())
        self.assertIsNone(row.mcd, msg="an absent metric must not be fabricated")
        self.assertIsNone(row.utmos_strong, msg="an absent metric must not be fabricated")
        self.assertIsNone(row.latency_p95_ms, msg="an absent metric must not be fabricated")

    def test_measured_zero_is_distinguished_from_absence(self) -> None:
        # A measured zero survives conversion as a real value.
        row: ExperimentResultRow = self._factory.from_buffer(self._buffers.sparse_panel())
        self.assertIsNotNone(row.stoi, msg="a measured zero must not collapse into absence")
        self.assertEqual(row.stoi, 0.0, msg="a measured zero must be preserved exactly")

    def test_count_columns_are_narrowed_to_integers(self) -> None:
        # Denominator and failure counts are integers even when the buffer
        # carried them as floats.
        row: ExperimentResultRow = self._factory.from_buffer(self._buffers.sparse_panel())
        self.assertIsInstance(row.parameter_count, int, msg="parameter_count must be an integer")
        self.assertEqual(row.parameter_count, 13500000)
        self.assertIsInstance(row.pesq_failure_count, int, msg="failure counts must be integers")
        self.assertEqual(row.pesq_failure_count, 2)

    def test_identity_arguments_are_carried_into_the_row(self) -> None:
        # Identity columns come from the caller, never from the metric buffer.
        row: ExperimentResultRow = self._factory.from_buffer(self._buffers.sparse_panel())
        self.assertEqual(row.unique_id, self._factory.unique_id)
        self.assertEqual(row.architecture_name, self._factory.architecture_name)
        self.assertEqual(row.dataset_name, self._factory.dataset_name)
        self.assertEqual(row.variant_name, self._factory.variant_name)
        self.assertEqual(row.seed, self._factory.seed)
        self.assertEqual(row.hyperparameters_summary, self._factory.hyperparameters_summary)
        self.assertEqual(row.interpretation_notes, self._factory.interpretation_notes)

    def test_timestamp_is_utc_at_second_resolution(self) -> None:
        # Rows are totally ordered across machines because the stamp is UTC
        # without local-timezone ambiguity and without sub-second noise.
        row: ExperimentResultRow = self._factory.from_buffer(self._buffers.sparse_panel())
        stamped_at: datetime = datetime.fromisoformat(row.date)
        self.assertIsNotNone(stamped_at.tzinfo, msg="the row timestamp must be timezone aware")
        self.assertEqual(
            stamped_at.utcoffset(),
            timedelta(0),
            msg="the row timestamp must be expressed in UTC"
        )
        self.assertEqual(stamped_at.microsecond, 0, msg="the row timestamp uses second resolution")

    def test_empty_buffer_produces_a_row_of_absent_measurements(self) -> None:
        # A run that produced nothing still yields a valid identity row.
        row: ExperimentResultRow = self._factory.from_buffer({})
        csv_row: dict[str, str | int | float | None] = row.to_csv_row()
        populated_columns: list[str] = [
            column_name
            for column_name in self._schema.measurement_columns
            if csv_row[column_name] is not None
        ]
        self.assertEqual(
            populated_columns,
            [],
            msg="an empty buffer must leave every measurement column absent"
        )
        self.assertEqual(row.unique_id, self._factory.unique_id)


class ExperimentResultRowValidationTest(unittest.TestCase):
    # Verifies that the frozen, closed, strict record rejects mutation,
    # unknown columns, and loosely typed scalars.
    def setUp(self) -> None:
        # Builds one valid row that every case in this class mutates a single
        # field of. Starting from a known-good record means each rejection
        # test isolates exactly one violation, so a failure names the rule
        # that broke rather than leaving several candidates.
        self._factory: ResultRowFactory = ResultRowFactory()
        self._buffers: MetricBufferBuilder = MetricBufferBuilder()
        self._row: ExperimentResultRow = self._factory.from_buffer(self._buffers.sparse_panel())

    def test_row_is_immutable_after_construction(self) -> None:
        # A written row is evidence; it cannot be edited in place.
        with self.assertRaises(ValidationError):
            self._row.seed: int = 99

    def test_unknown_column_is_rejected(self) -> None:
        # Extra keys are refused, so a stray column cannot accumulate silently.
        row_fields: dict[str, str | int | float | None] = self._row.model_dump()
        row_fields["unregistered_metric"] = 1.0
        with self.assertRaises(ValidationError):
            ExperimentResultRow(**row_fields)

    def test_string_seed_is_rejected_under_strict_validation(self) -> None:
        # Strict validation refuses coercion, so a stringified identity value
        # fails at construction rather than reaching the CSV.
        row_fields: dict[str, str | int | float | None] = self._row.model_dump()
        row_fields["seed"] = "11"
        with self.assertRaises(ValidationError):
            ExperimentResultRow(**row_fields)

    def test_missing_identity_field_is_rejected(self) -> None:
        # Identity columns are mandatory; a row without them is unattributable.
        row_fields: dict[str, str | int | float | None] = self._row.model_dump()
        del row_fields["unique_id"]
        with self.assertRaises(ValidationError):
            ExperimentResultRow(**row_fields)


class ExperimentResultCsvSerializationTest(unittest.TestCase):
    # Verifies that the serialized mapping reproduces the record values and
    # preserves absence as None rather than as a value.
    def setUp(self) -> None:
        # Needs no schema partition: these cases name the columns they assert
        # on directly, because the point is that a named field and its named
        # cell agree, not that the column set is complete.
        self._factory: ResultRowFactory = ResultRowFactory()
        self._buffers: MetricBufferBuilder = MetricBufferBuilder()

    def test_serialized_values_match_the_record_fields(self) -> None:
        # The mapping is a projection of the record, not a recomputation.
        row: ExperimentResultRow = self._factory.from_buffer(self._buffers.sparse_panel())
        csv_row: dict[str, str | int | float | None] = row.to_csv_row()
        self.assertEqual(csv_row["unique_id"], row.unique_id)
        self.assertEqual(csv_row["date"], row.date)
        self.assertEqual(csv_row["seed"], row.seed)
        self.assertEqual(csv_row["pesq"], row.pesq)
        self.assertEqual(csv_row["parameter_count"], row.parameter_count)
        self.assertEqual(csv_row["interpretation_notes"], row.interpretation_notes)

    def test_absent_measurements_serialize_as_none(self) -> None:
        # None reaches the CSV writer, which renders it as an empty cell.
        row: ExperimentResultRow = self._factory.from_buffer({})
        csv_row: dict[str, str | int | float | None] = row.to_csv_row()
        self.assertIsNone(csv_row["pesq"])
        self.assertIsNone(csv_row["real_time_factor"])
        self.assertIsNone(csv_row["test_utterance_count"])
