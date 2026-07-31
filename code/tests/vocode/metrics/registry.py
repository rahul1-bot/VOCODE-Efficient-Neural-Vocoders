# This module:
# 1. Verifies the closed metric vocabulary of MetricRegistry: the ordered
#    fourteen-name enumeration, its partition into the static and waveform
#    families, and the position of the callback-only rtf name in that order
# 2. Verifies selection validation for empty, unknown, and duplicated names
# 3. Verifies the metric-name to result-column mapping and by-name construction
#    of every buildable metric, including the rtf refusal
# 4. Verifies the MetricSelection record: its default panel, immutability,
#    strict field typing, and delegation of name validation to the registry
#
# Design decisions:
# - Construction is asserted by returned instance type and configuration rather
#   than by scoring behaviour, because the registry's responsibility ends at
#   building the object; scoring correctness belongs to each metric's own tests
# - The utmos entry is built and inspected through its configuration only. Its
#   predictor weights load lazily on the first scoring call, so registry-level
#   construction reaches no network path and none is exercised here
# - Output-column names are pinned exactly, because the result schema is a
#   published artifact contract rather than an internal implementation detail
#
# Author: Rahul Sawhney

import unittest

from pydantic import ValidationError

from vocode.metrics.f0 import F0Rmse
from vocode.metrics.las import LogAmplitudeSpectrumRmse
from vocode.metrics.macs import MacsProfiler
from vocode.metrics.mcd import MelCepstralDistortion
from vocode.metrics.mel import MelError
from vocode.metrics.parameters import ParameterCount
from vocode.metrics.periodicity import PeriodicityRmse
from vocode.metrics.pesq import Pesq
from vocode.metrics.registry import MetricName, MetricRegistry, MetricSelection
from vocode.metrics.size import ModelSize
from vocode.metrics.stft import MultiResolutionStftError
from vocode.metrics.stoi import Stoi
from vocode.metrics.utmos import UtmosPredictor
from vocode.metrics.voicing import VoicingF1


class MetricVocabularyTest(unittest.TestCase):
    # Verifies the closed name vocabulary, its declared order, and the
    # partition of that vocabulary into the static and waveform families.
    def setUp(self) -> None:
        # Prepares the registry shared by the vocabulary checks.
        self._registry: MetricRegistry = MetricRegistry()

    def test_vocabulary_enumerates_fourteen_atomic_metrics(self) -> None:
        # The registry publishes exactly fourteen atomic metric names.
        self.assertEqual(
            len(self._registry.names),
            14,
            msg=f"Expected fourteen registered metrics, got {self._registry.names}"
        )

    def test_vocabulary_is_the_exact_declared_name_sequence(self) -> None:
        # The vocabulary is an ordered tuple, not an unordered collection.
        expected_names: tuple[MetricName, ...] = (
            "pesq",
            "stoi",
            "mel",
            "stft",
            "mcd",
            "las",
            "f0",
            "periodicity",
            "voicing",
            "utmos",
            "rtf",
            "macs",
            "parameters",
            "size"
        )
        self.assertEqual(self._registry.names, expected_names)

    def test_vocabulary_names_are_unique(self) -> None:
        # A repeated name would make family membership and validation ambiguous.
        self.assertEqual(
            len(set(self._registry.names)),
            len(self._registry.names),
            msg=f"Duplicate metric name in vocabulary {self._registry.names}"
        )

    def test_families_partition_the_vocabulary_around_the_callback_metric(self) -> None:
        # Every name is static, waveform, or the callback-only rtf name.
        covered_names: set[MetricName] = set(self._registry.static_names) | set(
            self._registry.waveform_names
        ) | {"rtf"}
        self.assertEqual(covered_names, set(self._registry.names))

    def test_static_and_waveform_families_are_disjoint(self) -> None:
        # No metric may belong to two lifecycle families at once.
        self.assertTrue(
            set(self._registry.static_names).isdisjoint(self._registry.waveform_names),
            msg=(
                f"static={self._registry.static_names} overlaps "
                f"waveform={self._registry.waveform_names}"
            )
        )

    def test_rtf_belongs_to_neither_measurement_family(self) -> None:
        # rtf is a prediction-stage callback, not a static or waveform metric.
        self.assertNotIn("rtf", self._registry.static_names)
        self.assertNotIn("rtf", self._registry.waveform_names)

    def test_waveform_metrics_precede_rtf_which_precedes_the_static_metrics(self) -> None:
        # The declared order groups per-utterance work before complexity work.
        rtf_position: int = self._registry.names.index("rtf")
        waveform_positions: list[int] = [
            self._registry.names.index(name) for name in self._registry.waveform_names
        ]
        static_positions: list[int] = [
            self._registry.names.index(name) for name in self._registry.static_names
        ]
        self.assertLess(max(waveform_positions), rtf_position)
        self.assertLess(rtf_position, min(static_positions))

    def test_family_members_follow_the_vocabulary_order(self) -> None:
        # Each family lists its members in the order the vocabulary declares.
        waveform_in_vocabulary_order: tuple[MetricName, ...] = tuple(
            name for name in self._registry.names if name in self._registry.waveform_names
        )
        static_in_vocabulary_order: tuple[MetricName, ...] = tuple(
            name for name in self._registry.names if name in self._registry.static_names
        )
        self.assertEqual(waveform_in_vocabulary_order, self._registry.waveform_names)
        self.assertEqual(static_in_vocabulary_order, self._registry.static_names)


class MetricSelectionValidationTest(unittest.TestCase):
    # Verifies that selection validation accepts known unique names in caller
    # order and rejects empty, unknown, and duplicated selections.
    def setUp(self) -> None:
        # Prepares the registry shared by the validation checks.
        self._registry: MetricRegistry = MetricRegistry()

    def test_validation_returns_the_selection_in_caller_order(self) -> None:
        # Validation preserves the caller's ordering rather than re-sorting.
        selected_names: tuple[MetricName, ...] = ("size", "pesq", "macs")
        self.assertEqual(self._registry.validate(selected_names), selected_names)

    def test_validation_accepts_the_entire_vocabulary(self) -> None:
        # Selecting every registered name is a legal selection.
        self.assertEqual(
            self._registry.validate(self._registry.names),
            self._registry.names
        )

    def test_validation_accepts_a_single_name(self) -> None:
        # The minimum legal selection is one name.
        self.assertEqual(self._registry.validate(("mel",)), ("mel",))

    def test_empty_selection_is_rejected(self) -> None:
        # A run with no metrics measures nothing and must fail before starting.
        with self.assertRaisesRegex(ValueError, "At least one metric name is required"):
            self._registry.validate(())

    def test_unknown_name_is_rejected_and_named(self) -> None:
        # The error names the offender so a typo is immediately identifiable.
        with self.assertRaisesRegex(ValueError, "sisdr"):
            self._registry.validate(("pesq", "sisdr"))

    def test_unknown_name_rejection_lists_the_available_vocabulary(self) -> None:
        # The error also states what the caller could have selected instead.
        with self.assertRaisesRegex(ValueError, "available="):
            self._registry.validate(("sisdr",))

    def test_duplicated_name_is_rejected(self) -> None:
        # A repeated name would double-weight one metric in the panel.
        with self.assertRaisesRegex(ValueError, "must be unique"):
            self._registry.validate(("pesq", "mel", "pesq"))


class MetricOutputNameMappingTest(unittest.TestCase):
    # Verifies the mapping from metric names onto the result-schema column
    # names that carry the unit or definition of the reported quantity.
    def setUp(self) -> None:
        # Prepares the registry shared by the column-mapping checks.
        self._registry: MetricRegistry = MetricRegistry()

    def test_renamed_metrics_map_onto_their_result_columns(self) -> None:
        # Columns whose name must state a unit or definition are remapped.
        expected_columns: dict[MetricName, str] = {
            "mel": "mel_error",
            "stft": "multi_resolution_stft_error",
            "las": "las_rmse",
            "f0": "f0_rmse_cents",
            "periodicity": "periodicity_rmse",
            "voicing": "vuv_f1",
            "utmos": "utmos_strong",
            "rtf": "real_time_factor",
            "macs": "macs_per_second_audio",
            "parameters": "parameter_count",
            "size": "model_size_megabytes"
        }
        name: MetricName
        expected_column: str
        for name, expected_column in expected_columns.items():
            with self.subTest(metric=name):
                self.assertEqual(self._registry.output_name(name), expected_column)

    def test_self_describing_metrics_pass_through_unchanged(self) -> None:
        # Names that already read as their own column need no mapping.
        for name in ("pesq", "stoi", "mcd"):
            with self.subTest(metric=name):
                self.assertEqual(self._registry.output_name(name), name)

    def test_every_registered_name_maps_to_a_distinct_column(self) -> None:
        # Two metrics sharing a column would silently overwrite each other.
        columns: list[str] = [self._registry.output_name(name) for name in self._registry.names]
        self.assertEqual(len(set(columns)), len(columns), msg=f"Colliding columns in {columns}")

    def test_unregistered_name_passes_through_unchanged(self) -> None:
        # Mapping is a pure rename; membership is enforced by validation.
        self.assertEqual(self._registry.output_name("sisdr"), "sisdr")


class MetricConstructionTest(unittest.TestCase):
    # Verifies by-name construction: the returned type for every buildable
    # metric, the sample-rate and device parameters, and the rtf refusal.
    #
    # The two optional construction parameters are checked from both sides:
    # the metric that consumes each one must receive it, and a metric that
    # does not consume it must be unaffected, since a rate silently leaking
    # into a metric that defines its own operating rate would corrupt every
    # score it produced. Freshness is checked as well, because the
    # pitch-family metrics accumulate across a pass and a shared instance
    # would carry one run's frames into the next.
    def setUp(self) -> None:
        # Prepares the registry shared by the construction checks.
        self._registry: MetricRegistry = MetricRegistry()

    def test_every_buildable_name_returns_its_registered_type(self) -> None:
        # The registry is the single construction authority for the vocabulary.
        expected_types: dict[MetricName, type] = {
            "pesq": Pesq,
            "stoi": Stoi,
            "mel": MelError,
            "stft": MultiResolutionStftError,
            "mcd": MelCepstralDistortion,
            "las": LogAmplitudeSpectrumRmse,
            "f0": F0Rmse,
            "periodicity": PeriodicityRmse,
            "voicing": VoicingF1,
            "utmos": UtmosPredictor,
            "macs": MacsProfiler,
            "parameters": ParameterCount,
            "size": ModelSize
        }
        name: MetricName
        expected_type: type
        for name, expected_type in expected_types.items():
            with self.subTest(metric=name):
                self.assertIsInstance(self._registry.build(name), expected_type)

    def test_every_registered_name_is_buildable_except_rtf(self) -> None:
        # The vocabulary and the construction surface agree apart from rtf.
        buildable_names: tuple[MetricName, ...] = tuple(
            name for name in self._registry.names if name != "rtf"
        )
        self.assertEqual(len(buildable_names), len(self._registry.names) - 1)
        for name in buildable_names:
            with self.subTest(metric=name):
                self.assertIsNotNone(self._registry.build(name))

    def test_rtf_refuses_construction_as_a_prediction_stage_callback(self) -> None:
        # rtf needs the run's timing configuration, so it is not stateless.
        with self.assertRaisesRegex(ValueError, "prediction-stage callback"):
            self._registry.build("rtf")

    def test_unknown_name_refuses_construction(self) -> None:
        # Construction cannot invent a metric outside the vocabulary.
        with self.assertRaisesRegex(ValueError, "Unknown metric: sisdr"):
            self._registry.build("sisdr")

    def test_cepstral_distortion_defaults_to_the_ljspeech_sample_rate(self) -> None:
        # An unspecified sample rate falls back to the corpus rate.
        metric: object = self._registry.build("mcd")
        self.assertIsInstance(metric, MelCepstralDistortion)
        self.assertEqual(metric.configuration.sample_rate, 22050)

    def test_cepstral_distortion_honours_an_explicit_sample_rate(self) -> None:
        # MCD is the only metric that consumes the run's sample rate.
        metric: object = self._registry.build("mcd", sample_rate=16000)
        self.assertIsInstance(metric, MelCepstralDistortion)
        self.assertEqual(metric.configuration.sample_rate, 16000)

    def test_sample_rate_is_ignored_by_rate_independent_metrics(self) -> None:
        # PESQ operates at its own standard rate and resamples its inputs, so
        # the run's sample rate must not reach its configuration.
        metric: object = self._registry.build("pesq", sample_rate=8000)
        self.assertIsInstance(metric, Pesq)
        self.assertEqual(metric.configuration.target_sample_rate, 16000)

    def test_utmos_binds_the_requested_execution_device(self) -> None:
        # UTMOS is the only metric that consumes the trainer device.
        metric: object = self._registry.build("utmos", device="cpu")
        self.assertIsInstance(metric, UtmosPredictor)
        self.assertEqual(metric.configuration.device, "cpu")

    def test_construction_returns_a_fresh_instance_per_call(self) -> None:
        # Shared metric state across runs would leak accumulated values.
        first_instance: object = self._registry.build("f0")
        second_instance: object = self._registry.build("f0")
        self.assertIsNot(first_instance, second_instance)


class MetricSelectionRecordTest(unittest.TestCase):
    # Verifies the frozen selection record: its default panel, immutability,
    # strict typing, and registry-delegated name validation.
    def test_default_panel_covers_the_lightweight_quality_and_complexity_set(self) -> None:
        # The default selection is the panel every run computes without opt-in.
        self.assertEqual(
            MetricSelection().names,
            ("pesq", "stoi", "mel", "rtf", "parameters", "size")
        )

    def test_default_panel_is_a_valid_registry_selection(self) -> None:
        # The default must survive the same validation an explicit one faces.
        registry: MetricRegistry = MetricRegistry()
        self.assertEqual(registry.validate(MetricSelection().names), MetricSelection().names)

    def test_explicit_selection_is_retained_in_caller_order(self) -> None:
        # The record stores the selection exactly as the caller declared it.
        selection: MetricSelection = MetricSelection(names=("size", "mel", "parameters"))
        self.assertEqual(selection.names, ("size", "mel", "parameters"))

    def test_selection_record_is_frozen(self) -> None:
        # A run's metric panel cannot change once the record exists.
        selection: MetricSelection = MetricSelection()
        with self.assertRaises(ValidationError):
            selection.names: tuple[MetricName, ...] = ("mel",)

    def test_unknown_name_is_rejected_through_the_registry(self) -> None:
        # Selection validity and registry membership can never disagree.
        with self.assertRaises(ValidationError):
            MetricSelection(names=("sisdr",))

    def test_duplicated_name_is_rejected_through_the_registry(self) -> None:
        # The record inherits the registry's uniqueness requirement.
        with self.assertRaises(ValidationError):
            MetricSelection(names=("mel", "mel"))

    def test_empty_selection_is_rejected_through_the_registry(self) -> None:
        # The record inherits the registry's non-empty requirement.
        with self.assertRaises(ValidationError):
            MetricSelection(names=())

    def test_extra_fields_are_rejected(self) -> None:
        # A misspelled field must fail loudly rather than be ignored.
        with self.assertRaises(ValidationError):
            MetricSelection(names=("mel",), metrics=("pesq",))

    def test_mutable_name_sequence_is_rejected_under_strict_typing(self) -> None:
        # The selection is a frozen tuple, so a list is not silently coerced.
        with self.assertRaises(ValidationError):
            MetricSelection(names=["mel"])


if __name__ == "__main__":
    unittest.main()
