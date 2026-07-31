# This module:
# 1. Verifies that the VOCODE argument parser registers one subcommand per
#    evidence lane and stage, and that each subcommand binds its evidence
#    category, stage, and default dataset split
# 2. Verifies the argument converters: the boolean text domain and the
#    batch-limit domain, both directly and through parsed argument vectors
# 3. Verifies that malformed invocations fail at parse time: missing required
#    flags, unknown subcommands, and values outside a closed choice set
# 4. Verifies the layered configuration sources: the YAML file, the
#    dotted-path overrides applied onto it, and the command-line arguments
#    that take precedence over both
# 5. Verifies the argument-to-configuration mapping performed by the request
#    record and the configuration factory, including the architecture-specific
#    data defaults and every rejection path of resolution
#
# Design decisions:
# - No run is ever executed: VocodeCli.run dispatches a lane runner inside a
#   RunTracker, so the tests stop at the resolved ExperimentConfiguration and
#   reach the parser and its argument converters through the same instance
#   members that run itself uses
# - No dataset, checkpoint, or weights directory is read: every resolved path
#   stays a value, because resolution validates settings rather than loading
#   the artifacts they name
# - Parse failures are asserted as SystemExit with argparse usage text
#   captured, because the parser converts every misuse into a usage error
# - Configuration files are written into a temporary directory created per
#   test and removed in tearDown, so no fixture file is committed
# - The architecture-specific data defaults are asserted through a typed
#   expectation record covering one architecture per default arm, including
#   the unlisted architecture that falls through to the generic arm
#
# Author: Rahul Sawhney

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar, get_args

from pydantic import BaseModel, ConfigDict, ValidationError

from vocode.cli import (
    VocodeCli,
    VocodeCliArchitectureSet,
    VocodeCliCommandName,
    VocodeCliConfigurationFactory,
    VocodeCliFileConfiguration,
    VocodeCliOverrideParser,
    VocodeCliRequest,
    VocodeCliYamlLoader,
)
from vocode.configs.run import ExperimentConfiguration
from vocode.metrics.registry import MetricRegistry
from vocode.models.vocoder import ArchitectureName
from vocode.optimization.registry import OptimizationVariantName


class CommandDispatchExpectation(BaseModel):
    # One expected binding of a subcommand onto its evidence lane, executed
    # stage, and default dataset split.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    command_name: str
    evidence_category: str
    stage: str
    dataset_split_name: str


class ArchitectureDataDefaultExpectation(BaseModel):
    # One expected arm of the architecture-specific data-pipeline defaults.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    architecture_name: str
    validation_batch_size: int
    training_segment_size: int | None
    peak_normalization_enabled: bool
    resample_rate: int | None
    training_random_peak_gain_range_db: tuple[float, float] | None


class CommandVectorBuilder:
    # Builds argument vectors carrying the flags every atomic command
    # requires, so each case states only the arguments it exercises.
    def __init__(
        self,
        architecture_name: str = "hifigan_v1",
        seed: int = 0,
        run_id: str = "run-alpha"
    ) -> None:
        # Binds the cell identity every built vector declares.
        self._architecture_name: str = architecture_name
        self._seed: int = seed
        self._run_id: str = run_id

    def build(self, command_name: str, extra_arguments: list[str] | None = None) -> list[str]:
        # Produces the subcommand followed by the three required identity flags.
        resolved_extra_arguments: list[str] = list(extra_arguments) if extra_arguments is not None else []
        return [
            command_name,
            "--model",
            self._architecture_name,
            "--seed",
            str(self._seed),
            "--run-id",
            self._run_id,
            *resolved_extra_arguments
        ]

    def build_with_dataset_root(
        self,
        command_name: str,
        dataset_root: str = "corpus",
        extra_arguments: list[str] | None = None
    ) -> list[str]:
        # Adds the corpus location, which configuration resolution requires.
        resolved_extra_arguments: list[str] = list(extra_arguments) if extra_arguments is not None else []
        return self.build(command_name, ["--dataset-root", dataset_root, *resolved_extra_arguments])


class QuietArgumentParser:
    # Parses argument vectors with the usage text argparse writes on failure
    # captured, so rejection cases do not pollute the test report.
    def __init__(self, parser: argparse.ArgumentParser) -> None:
        # Binds the parser built by the command-line entry point under test.
        self._parser: argparse.ArgumentParser = parser

    def parse(self, command_line_arguments: list[str]) -> argparse.Namespace:
        # Parses one vector, discarding the usage text argparse writes on failure.
        captured_usage: io.StringIO = io.StringIO()
        with contextlib.redirect_stderr(captured_usage):
            return self._parser.parse_args(command_line_arguments)


class ConfigurationFileWriter:
    # Writes YAML configuration documents into one temporary directory.
    def __init__(self, directory: Path) -> None:
        # Binds the temporary directory every written document lands in.
        self._directory: Path = directory

    def write(self, file_name: str, document_text: str) -> Path:
        # Writes one document and returns the path a --config flag can name.
        document_path: Path = self._directory / file_name
        document_path.write_text(document_text)
        return document_path


class VocodeCliArchitectureVocabularyTest(unittest.TestCase):
    # Verifies that the command-line architecture choices mirror the closed
    # architecture vocabulary of the model registry.
    def test_command_line_choices_match_the_registry_vocabulary(self) -> None:
        # A drift between the two lists would silently hide an architecture from the CLI.
        registry_names: tuple[str, ...] = get_args(ArchitectureName.__value__)
        self.assertEqual(VocodeCliArchitectureSet.names, registry_names)

    def test_architecture_choices_are_unique(self) -> None:
        # A duplicated choice would make the usage text lie about the vocabulary.
        self.assertEqual(
            len(set(VocodeCliArchitectureSet.names)),
            len(VocodeCliArchitectureSet.names)
        )


class VocodeCliArgumentParserConstructionTest(unittest.TestCase):
    # Verifies subcommand registration, the lane bindings each subcommand
    # carries, required arguments, and closed choice sets.
    def setUp(self) -> None:
        # Binds the parser under test and the expected binding of every subcommand.
        self._cli: VocodeCli = VocodeCli()
        self._parser: QuietArgumentParser = QuietArgumentParser(self._cli._argument_parser)
        self._vectors: CommandVectorBuilder = CommandVectorBuilder()
        self._expectations: tuple[CommandDispatchExpectation, ...] = (
            CommandDispatchExpectation(
                command_name="train-reproduction",
                evidence_category="project_trained_reproduction",
                stage="train",
                dataset_split_name="train"
            ),
            CommandDispatchExpectation(
                command_name="validation-reproduction",
                evidence_category="project_trained_reproduction",
                stage="validation",
                dataset_split_name="validation"
            ),
            CommandDispatchExpectation(
                command_name="test-reproduction",
                evidence_category="project_trained_reproduction",
                stage="test",
                dataset_split_name="test"
            ),
            CommandDispatchExpectation(
                command_name="train-hybrid",
                evidence_category="project_hybrid_variants",
                stage="train",
                dataset_split_name="train"
            ),
            CommandDispatchExpectation(
                command_name="test-hybrid",
                evidence_category="project_hybrid_variants",
                stage="test",
                dataset_split_name="test"
            ),
            CommandDispatchExpectation(
                command_name="test-published",
                evidence_category="published_checkpoint_evaluation",
                stage="test",
                dataset_split_name="test"
            ),
            CommandDispatchExpectation(
                command_name="evaluate-optimized-variant",
                evidence_category="project_optimized_variants",
                stage="test",
                dataset_split_name="test"
            ),
            CommandDispatchExpectation(
                command_name="recover-optimized-variant",
                evidence_category="project_optimized_variants",
                stage="train",
                dataset_split_name="train"
            )
        )

    def test_the_expectations_cover_the_whole_command_vocabulary(self) -> None:
        # A subcommand registered without an expectation would go unverified below.
        self.assertEqual(
            tuple(expectation.command_name for expectation in self._expectations),
            get_args(VocodeCliCommandName.__value__)
        )

    def test_every_atomic_command_binds_its_lane_stage_and_default_split(self) -> None:
        # The subcommand alone decides the evidence lane and the stage it executes.
        expectation: CommandDispatchExpectation
        for expectation in self._expectations:
            parsed_arguments: argparse.Namespace = self._parser.parse(
                self._vectors.build(expectation.command_name)
            )
            self.assertEqual(parsed_arguments.command_name, expectation.command_name)
            self.assertEqual(
                parsed_arguments.evidence_category,
                expectation.evidence_category,
                msg=f"{expectation.command_name} bound the wrong evidence category"
            )
            self.assertEqual(
                parsed_arguments.stage,
                expectation.stage,
                msg=f"{expectation.command_name} bound the wrong stage"
            )
            self.assertEqual(
                parsed_arguments.dataset_split_name,
                expectation.dataset_split_name,
                msg=f"{expectation.command_name} bound the wrong default split"
            )

    def test_missing_subcommand_is_rejected(self) -> None:
        # The parser requires one atomic command; there is no default lane.
        with self.assertRaises(SystemExit):
            self._parser.parse([])

    def test_unknown_subcommand_is_rejected(self) -> None:
        # The command vocabulary is closed.
        with self.assertRaises(SystemExit):
            self._parser.parse(["train-everything"])

    def test_required_flags_are_enforced(self) -> None:
        # Architecture, seed, and run identifier are the identity of a cell.
        with self.assertRaises(SystemExit):
            self._parser.parse(["test-published", "--seed", "0", "--run-id", "run-alpha"])
        with self.assertRaises(SystemExit):
            self._parser.parse(["test-published", "--model", "vocos", "--run-id", "run-alpha"])
        with self.assertRaises(SystemExit):
            self._parser.parse(["test-published", "--model", "vocos", "--seed", "0"])

    def test_unknown_architecture_choice_is_rejected(self) -> None:
        # The parser refuses an architecture outside the registry vocabulary.
        with self.assertRaises(SystemExit):
            self._parser.parse(
                ["test-published", "--model", "hifigan_v9", "--seed", "0", "--run-id", "run-alpha"]
            )

    def test_unknown_metric_choice_is_rejected(self) -> None:
        # Metric names are validated against the registry before anything runs.
        with self.assertRaises(SystemExit):
            self._parser.parse(self._vectors.build("test-published", ["--metric", "snr"]))

    def test_registered_metric_choices_are_accepted_and_accumulate(self) -> None:
        # Repeated metric flags build the selection in the order they were given.
        parsed_arguments: argparse.Namespace = self._parser.parse(
            self._vectors.build("test-published", ["--metric", "pesq", "--metric", "stoi"])
        )
        self.assertEqual(parsed_arguments.metrics, ["pesq", "stoi"])
        self.assertIn("pesq", MetricRegistry.names)

    def test_unknown_optimization_variant_choice_is_rejected(self) -> None:
        # The variant vocabulary is the optimization registry literal.
        with self.assertRaises(SystemExit):
            self._parser.parse(
                self._vectors.build(
                    "evaluate-optimized-variant",
                    ["--optimization-variant", "int2_weight_only"]
                )
            )

    def test_registered_optimization_variant_choice_is_accepted(self) -> None:
        # Every registered variant name is a legal command-line choice.
        registered_names: tuple[str, ...] = get_args(OptimizationVariantName.__value__)
        parsed_arguments: argparse.Namespace = self._parser.parse(
            self._vectors.build(
                "evaluate-optimized-variant",
                ["--optimization-variant", "pruned_50_recovered"]
            )
        )
        self.assertEqual(parsed_arguments.optimization_variant_name, "pruned_50_recovered")
        self.assertIn("pruned_50_recovered", registered_names)

    def test_unknown_hardware_and_precision_choices_are_rejected(self) -> None:
        # Hardware and precision lanes are closed vocabularies at the boundary.
        with self.assertRaises(SystemExit):
            self._parser.parse(self._vectors.build("test-published", ["--hardware-name", "gh200"]))
        with self.assertRaises(SystemExit):
            self._parser.parse(self._vectors.build("test-published", ["--precision-name", "int8"]))

    def test_split_flag_overrides_the_subcommand_default(self) -> None:
        # The split is suppressed when absent, so the subcommand default survives.
        default_arguments: argparse.Namespace = self._parser.parse(self._vectors.build("test-published"))
        overridden_arguments: argparse.Namespace = self._parser.parse(
            self._vectors.build("test-published", ["--split", "validation"])
        )
        self.assertEqual(default_arguments.dataset_split_name, "test")
        self.assertEqual(overridden_arguments.dataset_split_name, "validation")

    def test_unknown_split_choice_is_rejected(self) -> None:
        # Only the three registered partitions may be evaluated.
        with self.assertRaises(SystemExit):
            self._parser.parse(self._vectors.build("test-published", ["--split", "holdout"]))

    def test_optional_arguments_default_to_absent(self) -> None:
        # Every optional flag stays unset so the resolution layer can decide.
        parsed_arguments: argparse.Namespace = self._parser.parse(self._vectors.build("test-published"))
        self.assertIsNone(parsed_arguments.config_path)
        self.assertIsNone(parsed_arguments.dataset_root)
        self.assertIsNone(parsed_arguments.hardware_name)
        self.assertIsNone(parsed_arguments.precision_name)
        self.assertIsNone(parsed_arguments.accelerator)
        self.assertIsNone(parsed_arguments.metrics)
        self.assertEqual(parsed_arguments.override_values, [])

    def test_path_arguments_are_parsed_into_path_objects(self) -> None:
        # Path-valued flags cross the boundary as paths, never as strings.
        parsed_arguments: argparse.Namespace = self._parser.parse(
            self._vectors.build(
                "test-published",
                [
                    "--dataset-root",
                    "corpus",
                    "--artifact-root",
                    "artifacts",
                    "--project-checkpoint-path",
                    "checkpoints/best.ckpt"
                ]
            )
        )
        self.assertEqual(parsed_arguments.dataset_root, Path("corpus"))
        self.assertEqual(parsed_arguments.artifact_root, Path("artifacts"))
        self.assertEqual(parsed_arguments.project_checkpoint_path, Path("checkpoints/best.ckpt"))


class VocodeCliArgumentConversionTest(unittest.TestCase):
    # Verifies the boolean and batch-limit argument converters, both as
    # direct calls and through parsed argument vectors.
    def setUp(self) -> None:
        # Binds the entry point, so the converters are reachable both ways.
        self._cli: VocodeCli = VocodeCli()
        self._parser: QuietArgumentParser = QuietArgumentParser(self._cli._argument_parser)
        self._vectors: CommandVectorBuilder = CommandVectorBuilder()

    def test_boolean_text_is_accepted_case_insensitively(self) -> None:
        # The two boolean spellings are accepted regardless of letter case.
        self.assertTrue(self._cli._parse_boolean_text("true"))
        self.assertTrue(self._cli._parse_boolean_text("TRUE"))
        self.assertFalse(self._cli._parse_boolean_text("false"))
        self.assertFalse(self._cli._parse_boolean_text("False"))

    def test_non_boolean_text_is_rejected_with_the_expected_domain(self) -> None:
        # Numeric and colloquial spellings are refused rather than guessed.
        declared_value: str
        for declared_value in ("1", "0", "yes", "no", ""):
            with self.assertRaisesRegex(argparse.ArgumentTypeError, "Expected true or false"):
                self._cli._parse_boolean_text(declared_value)

    def test_boolean_flags_parse_into_typed_values(self) -> None:
        # Boolean flags reach the namespace as bools, not as text.
        parsed_arguments: argparse.Namespace = self._parser.parse(
            self._vectors.build(
                "test-published",
                ["--pin-memory", "true", "--persistent-workers", "false"]
            )
        )
        self.assertIs(parsed_arguments.pin_memory, True)
        self.assertIs(parsed_arguments.persistent_workers, False)

    def test_invalid_boolean_flag_value_fails_at_parse_time(self) -> None:
        # A misspelled boolean fails before any configuration is resolved.
        with self.assertRaises(SystemExit):
            self._parser.parse(self._vectors.build("test-published", ["--pin-memory", "yes"]))

    def test_integer_batch_limit_is_parsed_as_a_count(self) -> None:
        # An integer argument stays an integer batch count.
        parsed_limit: int | float = self._cli._parse_batch_limit("7")
        self.assertIsInstance(parsed_limit, int)
        self.assertEqual(parsed_limit, 7)

    def test_fractional_batch_limit_is_parsed_as_a_fraction(self) -> None:
        # A float argument inside the unit interval stays a fraction.
        parsed_limit: int | float = self._cli._parse_batch_limit("0.25")
        self.assertIsInstance(parsed_limit, float)
        self.assertEqual(parsed_limit, 0.25)

    def test_out_of_domain_batch_limits_are_rejected(self) -> None:
        # Negative counts, fractions outside the unit interval, bools, and text are refused.
        declared_value: str
        for declared_value in ("-2", "0.0", "1.5", "true", "abc"):
            with self.assertRaises(argparse.ArgumentTypeError):
                self._cli._parse_batch_limit(declared_value)

    def test_batch_limit_flags_parse_into_typed_values(self) -> None:
        # The four limit flags each carry their own parsed value.
        parsed_arguments: argparse.Namespace = self._parser.parse(
            self._vectors.build(
                "test-published",
                [
                    "--limit-train-batches",
                    "2",
                    "--limit-val-batches",
                    "0.5",
                    "--limit-test-batches",
                    "0",
                    "--limit-predict-batches",
                    "1.0"
                ]
            )
        )
        self.assertEqual(parsed_arguments.limit_train_batches, 2)
        self.assertEqual(parsed_arguments.limit_val_batches, 0.5)
        self.assertEqual(parsed_arguments.limit_test_batches, 0)
        self.assertEqual(parsed_arguments.limit_predict_batches, 1.0)

    def test_invalid_batch_limit_flag_value_fails_at_parse_time(self) -> None:
        # A fraction above the whole loader fails before resolution.
        with self.assertRaises(SystemExit):
            self._parser.parse(
                self._vectors.build("test-published", ["--limit-test-batches", "2.5"])
            )


class VocodeCliRequestMappingTest(unittest.TestCase):
    # Verifies that the parsed namespace maps onto the frozen request record
    # without losing or reshaping any field.
    def setUp(self) -> None:
        # Binds the parser whose namespaces the request record is built from.
        self._cli: VocodeCli = VocodeCli()
        self._parser: QuietArgumentParser = QuietArgumentParser(self._cli._argument_parser)
        self._vectors: CommandVectorBuilder = CommandVectorBuilder()

    def test_request_carries_the_command_lane_and_identity_fields(self) -> None:
        # The request is the parsed invocation before any resolution happens.
        parsed_arguments: argparse.Namespace = self._parser.parse(
            self._vectors.build_with_dataset_root("test-reproduction")
        )
        request: VocodeCliRequest = VocodeCliRequest.from_namespace(parsed_arguments)
        self.assertEqual(request.command_name, "test-reproduction")
        self.assertEqual(request.evidence_category, "project_trained_reproduction")
        self.assertEqual(request.stage, "test")
        self.assertEqual(request.dataset_split_name, "test")
        self.assertEqual(request.architecture_name, "hifigan_v1")
        self.assertEqual(request.seed, 0)
        self.assertEqual(request.run_id, "run-alpha")
        self.assertEqual(request.dataset_root, Path("corpus"))

    def test_metric_flags_become_an_immutable_tuple(self) -> None:
        # The accumulated metric list is frozen into the record.
        parsed_arguments: argparse.Namespace = self._parser.parse(
            self._vectors.build("test-published", ["--metric", "pesq", "--metric", "mel"])
        )
        request: VocodeCliRequest = VocodeCliRequest.from_namespace(parsed_arguments)
        self.assertEqual(request.metrics, ("pesq", "mel"))

    def test_absent_metric_flags_stay_absent(self) -> None:
        # An unspecified panel leaves the decision to the configuration default.
        parsed_arguments: argparse.Namespace = self._parser.parse(self._vectors.build("test-published"))
        request: VocodeCliRequest = VocodeCliRequest.from_namespace(parsed_arguments)
        self.assertIsNone(request.metrics)

    def test_override_flags_are_copied_into_the_request(self) -> None:
        # The request owns its override list rather than aliasing the namespace list.
        parsed_arguments: argparse.Namespace = self._parser.parse(
            self._vectors.build("test-published", ["--override", "num_workers=2"])
        )
        request: VocodeCliRequest = VocodeCliRequest.from_namespace(parsed_arguments)
        self.assertEqual(request.override_values, ["num_workers=2"])
        parsed_arguments.override_values.append("pin_memory=true")
        self.assertEqual(request.override_values, ["num_workers=2"])

    def test_request_is_frozen_against_field_assignment(self) -> None:
        # The parsed invocation cannot be edited after it is recorded.
        parsed_arguments: argparse.Namespace = self._parser.parse(self._vectors.build("test-published"))
        request: VocodeCliRequest = VocodeCliRequest.from_namespace(parsed_arguments)
        with self.assertRaises(ValidationError):
            request.seed: int = 4


class VocodeCliYamlLoaderTest(unittest.TestCase):
    # Verifies that the optional configuration file is read into a string
    # keyed mapping and that malformed documents fail with a named path.
    def setUp(self) -> None:
        # Opens a temporary directory so no configuration fixture is committed.
        self._temporary_root: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._writer: ConfigurationFileWriter = ConfigurationFileWriter(Path(self._temporary_root.name))
        self._loader: VocodeCliYamlLoader = VocodeCliYamlLoader()

    def tearDown(self) -> None:
        # Removes the written documents so no test leaves state behind.
        self._temporary_root.cleanup()

    def test_absent_configuration_path_yields_the_empty_mapping(self) -> None:
        # Running without a configuration file is the normal case.
        self.assertEqual(self._loader.load(None), {})

    def test_mapping_document_is_loaded_with_its_parsed_value_types(self) -> None:
        # Scalars keep the types YAML gives them so validation sees real values.
        document_path: Path = self._writer.write(
            "settings.yaml",
            "experiment_name: from_file\ntrain_epoch_count: 3\npin_memory: true\n"
        )
        loaded_document: dict[str, object] = self._loader.load(document_path)
        self.assertEqual(loaded_document["experiment_name"], "from_file")
        self.assertEqual(loaded_document["train_epoch_count"], 3)
        self.assertIs(loaded_document["pin_memory"], True)

    def test_empty_document_yields_the_empty_mapping(self) -> None:
        # An empty file is an absent configuration, not a failure.
        document_path: Path = self._writer.write("empty.yaml", "")
        self.assertEqual(self._loader.load(document_path), {})

    def test_non_mapping_document_is_rejected(self) -> None:
        # A sequence document cannot describe named settings.
        document_path: Path = self._writer.write("sequence.yaml", "- first\n- second\n")
        with self.assertRaisesRegex(ValueError, "must contain a mapping"):
            self._loader.load(document_path)

    def test_non_string_key_is_rejected(self) -> None:
        # Setting names must be strings so they can address model fields.
        document_path: Path = self._writer.write("numeric_key.yaml", "7: value\n")
        with self.assertRaisesRegex(ValueError, "Configuration key must be a string"):
            self._loader.load(document_path)

    def test_missing_configuration_file_raises_a_file_error(self) -> None:
        # A named file that does not exist is a caller mistake, not an empty configuration.
        with self.assertRaises(FileNotFoundError):
            self._loader.load(Path(self._temporary_root.name) / "absent.yaml")


class VocodeCliOverrideParserTest(unittest.TestCase):
    # Verifies that dotted-path override flags are typed, applied onto the
    # file configuration, and rejected when malformed.
    def setUp(self) -> None:
        # Binds the override layer, which is exercised without a parser.
        self._override_parser: VocodeCliOverrideParser = VocodeCliOverrideParser()

    def test_override_values_are_parsed_into_their_yaml_types(self) -> None:
        # Override text is typed before validation, not left as strings.
        resolved_configuration: dict[str, object] = self._override_parser.apply(
            {},
            ["train_epoch_count=3", "pin_memory=true", "experiment_name=smoke"]
        )
        self.assertEqual(resolved_configuration["train_epoch_count"], 3)
        self.assertIs(resolved_configuration["pin_memory"], True)
        self.assertEqual(resolved_configuration["experiment_name"], "smoke")

    def test_override_replaces_the_file_value(self) -> None:
        # The override layer sits above the file layer.
        resolved_configuration: dict[str, object] = self._override_parser.apply(
            {"num_workers": 1},
            ["num_workers=4"]
        )
        self.assertEqual(resolved_configuration["num_workers"], 4)

    def test_apply_leaves_the_incoming_configuration_untouched(self) -> None:
        # Resolution copies rather than mutating the loaded document.
        file_configuration: dict[str, object] = {"num_workers": 1}
        self._override_parser.apply(file_configuration, ["num_workers=4"])
        self.assertEqual(file_configuration["num_workers"], 1)

    def test_value_containing_an_equals_sign_is_split_once(self) -> None:
        # Only the first separator delimits the key.
        resolved_configuration: dict[str, object] = self._override_parser.apply(
            {},
            ["hypothesis=quality=preserved"]
        )
        self.assertEqual(resolved_configuration["hypothesis"], "quality=preserved")

    def test_override_without_separator_is_rejected(self) -> None:
        # A bare token cannot address a setting.
        with self.assertRaisesRegex(ValueError, "must use key=value syntax"):
            self._override_parser.apply({}, ["num_workers"])

    def test_override_with_empty_key_is_rejected(self) -> None:
        # An empty key would silently discard the value.
        with self.assertRaisesRegex(ValueError, "Override key cannot be empty"):
            self._override_parser.apply({}, ["   =4"])

    def test_empty_override_list_returns_the_file_configuration(self) -> None:
        # No overrides means the file layer passes through unchanged.
        resolved_configuration: dict[str, object] = self._override_parser.apply({"seed": 1}, [])
        self.assertEqual(resolved_configuration, {"seed": 1})


class VocodeCliFileConfigurationTest(unittest.TestCase):
    # Verifies the shape of the optional configuration file record: optional
    # fields, boundary coercion, closed key set, and immutability.
    def test_every_field_is_optional(self) -> None:
        # The command line can supply any setting, so the file record has no requirements.
        file_configuration: VocodeCliFileConfiguration = VocodeCliFileConfiguration()
        self.assertIsNone(file_configuration.experiment_name)
        self.assertIsNone(file_configuration.dataset_root)
        self.assertIsNone(file_configuration.metrics)

    def test_path_values_are_coerced_at_the_file_boundary(self) -> None:
        # YAML carries strings, so path settings are converted when the file is validated.
        file_configuration: VocodeCliFileConfiguration = VocodeCliFileConfiguration(
            artifact_root="artifacts",
            dataset_root="corpus"
        )
        self.assertEqual(file_configuration.artifact_root, Path("artifacts"))
        self.assertEqual(file_configuration.dataset_root, Path("corpus"))

    def test_unknown_setting_is_rejected(self) -> None:
        # A misspelled setting fails loudly instead of being ignored.
        with self.assertRaises(ValidationError):
            VocodeCliFileConfiguration(dataset_rooot="corpus")

    def test_non_positive_batch_size_is_rejected(self) -> None:
        # Positive-integer settings are validated at the file boundary.
        with self.assertRaises(ValidationError):
            VocodeCliFileConfiguration(training_batch_size=0)

    def test_unknown_hardware_name_is_rejected(self) -> None:
        # The file cannot widen a closed vocabulary that the parser restricts.
        with self.assertRaises(ValidationError):
            VocodeCliFileConfiguration(hardware_name="gh200")

    def test_record_is_frozen_against_field_assignment(self) -> None:
        # File settings cannot be edited after they are validated.
        file_configuration: VocodeCliFileConfiguration = VocodeCliFileConfiguration()
        with self.assertRaises(ValidationError):
            file_configuration.experiment_name: str | None = "edited"


class VocodeCliConfigurationResolutionTest(unittest.TestCase):
    # Verifies that the configuration factory maps one parsed invocation onto
    # a validated ExperimentConfiguration, applying source precedence and the
    # architecture-specific data defaults.
    def setUp(self) -> None:
        # Binds the whole resolution chain: parser, request record, and factory.
        self._cli: VocodeCli = VocodeCli()
        self._parser: QuietArgumentParser = QuietArgumentParser(self._cli._argument_parser)
        self._vectors: CommandVectorBuilder = CommandVectorBuilder()
        self._factory: VocodeCliConfigurationFactory = VocodeCliConfigurationFactory()

    def _build_configuration(self, command_line_arguments: list[str]) -> ExperimentConfiguration:
        # Runs one vector through parsing and resolution, stopping before execution.
        parsed_arguments: argparse.Namespace = self._parser.parse(command_line_arguments)
        request: VocodeCliRequest = VocodeCliRequest.from_namespace(parsed_arguments)
        return self._factory.build(request)

    def test_minimal_invocation_resolves_the_declared_defaults(self) -> None:
        # An invocation with only the required flags still fully describes a run.
        configuration: ExperimentConfiguration = self._build_configuration(
            self._vectors.build_with_dataset_root("test-published")
        )
        self.assertEqual(configuration.experiment_name, "vocode_checkpoint01")
        self.assertEqual(configuration.accelerator, "auto")
        self.assertEqual(configuration.train_epoch_count, 1)
        self.assertEqual(configuration.published_weights_root, Path("published_weights"))
        self.assertEqual(configuration.artifact_root, Path("experiment_artifacts"))
        self.assertEqual(configuration.artifact_layout.hardware_name, "b200")
        self.assertEqual(configuration.artifact_layout.precision_name, "fp32")

    def test_identity_flags_reach_the_artifact_layout(self) -> None:
        # The capsule directory is addressed by the identity the command declared.
        configuration: ExperimentConfiguration = self._build_configuration(
            self._vectors.build_with_dataset_root(
                "test-published",
                extra_arguments=["--hardware-name", "m3_max", "--precision-name", "fp16"]
            )
        )
        self.assertEqual(configuration.artifact_layout.hardware_name, "m3_max")
        self.assertEqual(configuration.artifact_layout.precision_name, "fp16")
        self.assertEqual(configuration.artifact_layout.run_id, "run-alpha")
        self.assertEqual(configuration.artifact_layout.seed, 0)

    def test_real_time_factor_settings_reach_the_timing_protocol(self) -> None:
        # Warm-up and repetition counts are executed protocol inputs.
        configuration: ExperimentConfiguration = self._build_configuration(
            self._vectors.build_with_dataset_root(
                "test-published",
                extra_arguments=["--rtf-warmup-iterations", "0", "--rtf-repetition-count", "2"]
            )
        )
        self.assertEqual(configuration.real_time_factor_configuration.warmup_iterations, 0)
        self.assertEqual(configuration.real_time_factor_configuration.timed_repetitions, 2)

    def test_metric_flags_reach_the_metric_selection(self) -> None:
        # The selected panel replaces the default rather than extending it.
        configuration: ExperimentConfiguration = self._build_configuration(
            self._vectors.build_with_dataset_root(
                "test-published",
                extra_arguments=["--metric", "pesq", "--metric", "parameters"]
            )
        )
        self.assertEqual(configuration.metric_selection.names, ("pesq", "parameters"))

    def test_batch_limit_flags_reach_the_configuration(self) -> None:
        # Execution limits survive resolution with their parsed types.
        configuration: ExperimentConfiguration = self._build_configuration(
            self._vectors.build_with_dataset_root(
                "test-published",
                extra_arguments=["--limit-test-batches", "3", "--limit-predict-batches", "0.5"]
            )
        )
        self.assertEqual(configuration.limit_test_batches, 3)
        self.assertEqual(configuration.limit_predict_batches, 0.5)

    def test_data_flags_reach_the_data_configuration(self) -> None:
        # Loader policy is resolved into the frozen data record.
        configuration: ExperimentConfiguration = self._build_configuration(
            self._vectors.build_with_dataset_root(
                "test-published",
                extra_arguments=[
                    "--test-batch-size",
                    "4",
                    "--num-workers",
                    "2",
                    "--pin-memory",
                    "true",
                    "--max-test-utterances",
                    "8"
                ]
            )
        )
        self.assertEqual(configuration.data_configuration.dataset_root, Path("corpus"))
        self.assertEqual(configuration.data_configuration.test_batch_size, 4)
        self.assertEqual(configuration.data_configuration.num_workers, 2)
        self.assertTrue(configuration.data_configuration.pin_memory)
        self.assertEqual(configuration.data_configuration.max_test_utterances, 8)

    def test_architecture_specific_data_defaults_are_applied(self) -> None:
        # Each architecture carries the loader defaults its published recipe requires.
        expectation: ArchitectureDataDefaultExpectation
        for expectation in (
            ArchitectureDataDefaultExpectation(
                architecture_name="hifigan_v1",
                validation_batch_size=1,
                training_segment_size=8192,
                peak_normalization_enabled=True,
                resample_rate=None,
                training_random_peak_gain_range_db=None
            ),
            ArchitectureDataDefaultExpectation(
                architecture_name="melgan",
                validation_batch_size=1,
                training_segment_size=16000,
                peak_normalization_enabled=False,
                resample_rate=None,
                training_random_peak_gain_range_db=None
            ),
            ArchitectureDataDefaultExpectation(
                architecture_name="vocos",
                validation_batch_size=1,
                training_segment_size=16384,
                peak_normalization_enabled=False,
                resample_rate=24000,
                training_random_peak_gain_range_db=(-6.0, -1.0)
            ),
            ArchitectureDataDefaultExpectation(
                architecture_name="rfwave",
                validation_batch_size=16,
                training_segment_size=32512,
                peak_normalization_enabled=False,
                resample_rate=24000,
                training_random_peak_gain_range_db=None
            ),
            ArchitectureDataDefaultExpectation(
                architecture_name="lpcnet",
                validation_batch_size=1,
                training_segment_size=2400,
                peak_normalization_enabled=False,
                resample_rate=16000,
                training_random_peak_gain_range_db=None
            ),
            # HiFTNet is deliberately the last entry: it is the one architecture here that the
            # resolver's match statement does not name, so it exercises the generic fallback
            # arm and pins what an architecture receives before its recipe is transcribed.
            ArchitectureDataDefaultExpectation(
                architecture_name="hiftnet",
                validation_batch_size=16,
                training_segment_size=None,
                peak_normalization_enabled=False,
                resample_rate=None,
                training_random_peak_gain_range_db=None
            )
        ):
            vectors: CommandVectorBuilder = CommandVectorBuilder(
                architecture_name=expectation.architecture_name
            )
            configuration: ExperimentConfiguration = self._build_configuration(
                vectors.build_with_dataset_root("train-reproduction")
            )
            self.assertEqual(
                configuration.data_configuration.validation_batch_size,
                expectation.validation_batch_size,
                msg=f"{expectation.architecture_name} validation batch size drifted"
            )
            self.assertEqual(
                configuration.data_configuration.training_segment_size,
                expectation.training_segment_size,
                msg=f"{expectation.architecture_name} training segment size drifted"
            )
            self.assertEqual(
                configuration.data_configuration.peak_normalization_enabled,
                expectation.peak_normalization_enabled,
                msg=f"{expectation.architecture_name} peak normalization drifted"
            )
            self.assertEqual(
                configuration.data_configuration.resample_rate,
                expectation.resample_rate,
                msg=f"{expectation.architecture_name} resample rate drifted"
            )
            self.assertEqual(
                configuration.data_configuration.training_random_peak_gain_range_db,
                expectation.training_random_peak_gain_range_db,
                msg=f"{expectation.architecture_name} training gain range drifted"
            )

    def test_explicit_segment_size_overrides_the_architecture_default(self) -> None:
        # An explicit crop length wins over the recipe default.
        configuration: ExperimentConfiguration = self._build_configuration(
            self._vectors.build_with_dataset_root(
                "train-reproduction",
                extra_arguments=["--training-segment-size", "4096"]
            )
        )
        self.assertEqual(configuration.data_configuration.training_segment_size, 4096)

    def test_optimized_command_resolves_the_variant_scoped_capsule(self) -> None:
        # The optimized lane binds provenance, device, and the variant directory segment.
        configuration: ExperimentConfiguration = self._build_configuration(
            self._vectors.build_with_dataset_root(
                "evaluate-optimized-variant",
                extra_arguments=[
                    "--optimization-variant",
                    "int8_dynamic",
                    "--hypothesis-id",
                    "h_int8_dynamic",
                    "--code-commit-hash",
                    "0f1e2d3c4b5a",
                    "--project-checkpoint-path",
                    "checkpoints/best.ckpt",
                    "--hardware-name",
                    "cpu",
                    "--accelerator",
                    "cpu"
                ]
            )
        )
        self.assertEqual(configuration.evidence_category, "project_optimized_variants")
        self.assertEqual(configuration.artifact_layout.variant_name, "int8_dynamic")
        self.assertEqual(configuration.summary_csv_path.name, "experiments_v2.csv")
        self.assertIn("int8_dynamic", configuration.run_directory.parts)

    def test_variant_flag_outside_the_optimized_lane_leaves_the_layout_unsegmented(self) -> None:
        # Only the optimized lane addresses a variant directory segment.
        configuration: ExperimentConfiguration = self._build_configuration(
            self._vectors.build_with_dataset_root(
                "test-reproduction",
                extra_arguments=[
                    "--optimization-variant",
                    "int8_dynamic",
                    "--project-checkpoint-path",
                    "checkpoints/best.ckpt"
                ]
            )
        )
        self.assertIsNone(configuration.artifact_layout.variant_name)
        self.assertNotIn("int8_dynamic", configuration.run_directory.parts)

    def test_optimized_command_without_a_variant_flag_is_rejected(self) -> None:
        # The variant segment has no default, so the capsule has no address.
        with self.assertRaisesRegex(ValidationError, "variant_name is required"):
            self._build_configuration(
                self._vectors.build_with_dataset_root(
                    "evaluate-optimized-variant",
                    extra_arguments=["--hardware-name", "cpu", "--accelerator", "cpu"]
                )
            )

    def test_optimized_command_without_hypothesis_and_commit_is_rejected(self) -> None:
        # A variant capsule cannot be resolved without its scientific identity.
        with self.assertRaisesRegex(ValidationError, "require explicit provenance"):
            self._build_configuration(
                self._vectors.build_with_dataset_root(
                    "evaluate-optimized-variant",
                    extra_arguments=[
                        "--optimization-variant",
                        "int8_dynamic",
                        "--project-checkpoint-path",
                        "checkpoints/best.ckpt",
                        "--hardware-name",
                        "cpu",
                        "--accelerator",
                        "cpu"
                    ]
                )
            )

    def test_optimized_command_without_an_explicit_accelerator_is_rejected(self) -> None:
        # An automatically resolved device could contradict the declared hardware lane.
        with self.assertRaisesRegex(ValidationError, "must declare an explicit accelerator"):
            self._build_configuration(
                self._vectors.build_with_dataset_root(
                    "evaluate-optimized-variant",
                    extra_arguments=[
                        "--optimization-variant",
                        "int8_dynamic",
                        "--hypothesis-id",
                        "h_int8_dynamic",
                        "--code-commit-hash",
                        "0f1e2d3c4b5a",
                        "--project-checkpoint-path",
                        "checkpoints/best.ckpt",
                        "--hardware-name",
                        "cpu"
                    ]
                )
            )

    def test_missing_dataset_root_is_rejected(self) -> None:
        # The corpus location has no default because it is machine specific.
        with self.assertRaisesRegex(ValueError, "dataset_root is required"):
            self._build_configuration(self._vectors.build("test-published"))

    def test_reproduction_evaluation_requires_a_project_checkpoint(self) -> None:
        # Evaluating a trained reproduction without its checkpoint has nothing to load.
        declared_command: str
        for declared_command in ("validation-reproduction", "test-reproduction"):
            with self.assertRaisesRegex(ValueError, "project_checkpoint_path is required"):
                self._build_configuration(self._vectors.build_with_dataset_root(declared_command))

    def test_reproduction_training_does_not_require_a_project_checkpoint(self) -> None:
        # A training run produces the checkpoint rather than consuming one.
        configuration: ExperimentConfiguration = self._build_configuration(
            self._vectors.build_with_dataset_root("train-reproduction")
        )
        self.assertIsNone(configuration.project_checkpoint_path)

    def test_split_that_contradicts_the_stage_is_rejected(self) -> None:
        # The stage-to-split binding is enforced when the record is constructed.
        with self.assertRaisesRegex(ValidationError, "requires dataset_split_name"):
            self._build_configuration(
                self._vectors.build_with_dataset_root(
                    "test-published",
                    extra_arguments=["--split", "validation"]
                )
            )

    def test_negative_worker_count_is_rejected_during_resolution(self) -> None:
        # Worker counts are non-negative by contract.
        with self.assertRaisesRegex(ValueError, "Expected a non-negative integer"):
            self._build_configuration(
                self._vectors.build_with_dataset_root(
                    "test-published",
                    extra_arguments=["--num-workers", "-1"]
                )
            )

    def test_non_positive_batch_size_is_rejected_during_resolution(self) -> None:
        # Batch sizes are positive by contract.
        with self.assertRaisesRegex(ValueError, "Expected a positive integer"):
            self._build_configuration(
                self._vectors.build_with_dataset_root(
                    "test-published",
                    extra_arguments=["--test-batch-size", "0"]
                )
            )

    def test_non_positive_peak_normalization_value_is_rejected_during_resolution(self) -> None:
        # The normalization target is a positive amplitude.
        with self.assertRaisesRegex(ValueError, "Expected a positive float"):
            self._build_configuration(
                self._vectors.build_with_dataset_root(
                    "train-reproduction",
                    extra_arguments=["--peak-normalization-value", "0.0"]
                )
            )


class VocodeCliLayeredSourceTest(unittest.TestCase):
    # Verifies the precedence of the three configuration sources: the YAML
    # file, the dotted-path overrides applied onto it, and the command line.
    def setUp(self) -> None:
        # Writes the file layer once, so each case varies only the upper layers.
        self._temporary_root: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._writer: ConfigurationFileWriter = ConfigurationFileWriter(Path(self._temporary_root.name))
        self._cli: VocodeCli = VocodeCli()
        self._parser: QuietArgumentParser = QuietArgumentParser(self._cli._argument_parser)
        self._vectors: CommandVectorBuilder = CommandVectorBuilder()
        self._factory: VocodeCliConfigurationFactory = VocodeCliConfigurationFactory()
        self._configuration_path: Path = self._writer.write(
            "run.yaml",
            "experiment_name: from_file\ntrain_epoch_count: 4\ndataset_root: corpus\nnum_workers: 2\n"
        )

    def tearDown(self) -> None:
        # Removes the written documents so no test leaves state behind.
        self._temporary_root.cleanup()

    def _build_configuration(self, extra_arguments: list[str]) -> ExperimentConfiguration:
        # Resolves one invocation that always names the shared configuration file.
        configuration_arguments: list[str] = [
            "--config",
            str(self._configuration_path),
            *extra_arguments
        ]
        parsed_arguments: argparse.Namespace = self._parser.parse(
            self._vectors.build("train-reproduction", configuration_arguments)
        )
        request: VocodeCliRequest = VocodeCliRequest.from_namespace(parsed_arguments)
        return self._factory.build(request)

    def test_file_values_are_used_when_no_flag_supplies_them(self) -> None:
        # The file layer supplies every setting the command line left absent.
        configuration: ExperimentConfiguration = self._build_configuration([])
        self.assertEqual(configuration.experiment_name, "from_file")
        self.assertEqual(configuration.train_epoch_count, 4)
        self.assertEqual(configuration.data_configuration.dataset_root, Path("corpus"))
        self.assertEqual(configuration.data_configuration.num_workers, 2)

    def test_command_line_flags_win_over_file_values(self) -> None:
        # The command line is the most specific source and takes precedence.
        configuration: ExperimentConfiguration = self._build_configuration(
            ["--experiment-name", "from_command_line", "--train-epoch-count", "9"]
        )
        self.assertEqual(configuration.experiment_name, "from_command_line")
        self.assertEqual(configuration.train_epoch_count, 9)

    def test_overrides_win_over_file_values(self) -> None:
        # An override edits the file layer before it is validated.
        configuration: ExperimentConfiguration = self._build_configuration(
            ["--override", "experiment_name=from_override"]
        )
        self.assertEqual(configuration.experiment_name, "from_override")

    def test_command_line_flags_win_over_overrides(self) -> None:
        # Precedence runs command line, then override, then file.
        configuration: ExperimentConfiguration = self._build_configuration(
            [
                "--override",
                "experiment_name=from_override",
                "--experiment-name",
                "from_command_line"
            ]
        )
        self.assertEqual(configuration.experiment_name, "from_command_line")

    def test_override_naming_an_unknown_setting_is_rejected(self) -> None:
        # Overrides address file settings only, so a stray key fails validation.
        with self.assertRaises(ValidationError):
            self._build_configuration(["--override", "seed=9"])

    def test_override_with_an_invalid_value_is_rejected(self) -> None:
        # The override layer is validated exactly like the file it edits.
        with self.assertRaises(ValidationError):
            self._build_configuration(["--override", "train_epoch_count=0"])

    def test_file_only_settings_are_reachable_through_the_file_layer(self) -> None:
        # Settings without a command-line flag are still configurable from the file.
        configuration_path: Path = self._writer.write(
            "strategy.yaml",
            "dataset_root: corpus\npartition_strategy: ordered_identifier_holdout\n"
        )
        parsed_arguments: argparse.Namespace = self._parser.parse(
            self._vectors.build("train-reproduction", ["--config", str(configuration_path)])
        )
        request: VocodeCliRequest = VocodeCliRequest.from_namespace(parsed_arguments)
        configuration: ExperimentConfiguration = self._factory.build(request)
        self.assertEqual(
            configuration.data_configuration.partition_strategy,
            "ordered_identifier_holdout"
        )


if __name__ == "__main__":
    unittest.main()
