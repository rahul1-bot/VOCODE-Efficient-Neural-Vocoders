# This module:
# 1. Verifies that ExperimentConfiguration accepts a complete run description
#    and holds its declared defaults under a strict, frozen, extra-forbidding
#    configuration
# 2. Verifies the batch-limit domain: integer counts, fractional loader
#    limits, and the rejection of negative counts, out-of-range fractions,
#    and boolean values
# 3. Verifies the stage-to-split binding that prevents one split from being
#    reported under another split's label
# 4. Verifies the optimized-variant contract: mandatory provenance fields and
#    the accelerator that must match the declared hardware lane
# 5. Verifies that the artifact-path properties delegate to the artifact
#    layout rather than composing paths themselves
#
# Design decisions:
# - Configurations are built through a record builder that supplies the
#   valid collaborators once, so each case states only the field under test
# - No dataset is touched: LJSpeechDataConfig is a frozen settings record and
#   validates its own fields without reading the corpus root
# - The boolean batch limit is asserted twice, once through construction
#   where strict union validation rejects it and once against the field
#   validator directly, because only the direct call reaches the explicit
#   bool guard and its message
# - Path expectations are asserted against the artifact layout's own
#   properties, so the delegation contract is verified without restating the
#   directory grammar that the layout tests already pin
#
# Author: Rahul Sawhney

import unittest
from pathlib import Path

from pydantic import ValidationError

from vocode.configs.layout import DatasetSplitName, ExperimentArtifactLayout, ExperimentStage, HardwareName
from vocode.configs.run import BatchLimit, ExperimentConfiguration
from vocode.data.ljspeech_datamodule import LJSpeechDataConfig
from vocode.metrics.registry import MetricSelection
from vocode.metrics.rtf import RealTimeFactorConfig


class ExperimentConfigurationBuilder:
    # Builds valid experiment configurations and their artifact layouts so
    # the test cases state only the fields they vary.
    def __init__(self, artifact_root: Path) -> None:
        # Binds the artifact root and the two collaborators every record needs.
        self._artifact_root: Path = artifact_root
        self._data_configuration: LJSpeechDataConfig = LJSpeechDataConfig(dataset_root=Path("corpus"))
        self._real_time_factor_configuration: RealTimeFactorConfig = RealTimeFactorConfig()

    def build_reproduction_layout(self) -> ExperimentArtifactLayout:
        # Builds the layout of a reproduction capsule, which carries no variant.
        return ExperimentArtifactLayout(
            artifact_root=self._artifact_root,
            evidence_category="project_trained_reproduction",
            architecture_name="hifigan_v1",
            hardware_name="b200",
            precision_name="fp32",
            seed=0,
            run_id="run-alpha"
        )

    def build_optimized_layout(self, hardware_name: HardwareName = "cpu") -> ExperimentArtifactLayout:
        # Builds the layout of an optimized capsule on the requested hardware lane.
        return ExperimentArtifactLayout(
            artifact_root=self._artifact_root,
            evidence_category="project_optimized_variants",
            architecture_name="vocos",
            hardware_name=hardware_name,
            precision_name="fp32",
            seed=0,
            run_id="run-beta",
            variant_name="int8_dynamic"
        )

    def build_reproduction(
        self,
        stage: ExperimentStage = "test",
        dataset_split_name: DatasetSplitName = "test",
        train_epoch_count: int = 1,
        limit_test_batches: BatchLimit = None,
        metric_selection: MetricSelection | None = None
    ) -> ExperimentConfiguration:
        # Builds a complete reproduction run description around the varied fields.
        return ExperimentConfiguration(
            experiment_name="vocode_reproduction",
            run_id="run-alpha",
            hypothesis="Reproduction matches the published operating point.",
            interpretation_notes="",
            evidence_category="project_trained_reproduction",
            stage=stage,
            dataset_split_name=dataset_split_name,
            architecture_name="hifigan_v1",
            seed=0,
            artifact_layout=self.build_reproduction_layout(),
            published_weights_root=Path("published_weights"),
            data_configuration=self._data_configuration,
            real_time_factor_configuration=self._real_time_factor_configuration,
            metric_selection=metric_selection if metric_selection is not None else MetricSelection(),
            train_epoch_count=train_epoch_count,
            limit_test_batches=limit_test_batches
        )

    def build_optimized(
        self,
        accelerator: str = "cpu",
        hardware_name: HardwareName = "cpu",
        optimization_variant_name: str | None = "int8_dynamic",
        project_checkpoint_path: Path | None = Path("checkpoints/best.ckpt"),
        optimization_hypothesis_id: str | None = "h_int8_dynamic",
        code_commit_hash: str | None = "0f1e2d3c4b5a"
    ) -> ExperimentConfiguration:
        # Builds a fully provenanced optimized-variant run description.
        return ExperimentConfiguration(
            experiment_name="vocode_optimized",
            run_id="run-beta",
            hypothesis="Dynamic integer quantization preserves perceptual quality.",
            interpretation_notes="",
            evidence_category="project_optimized_variants",
            stage="test",
            dataset_split_name="test",
            architecture_name="vocos",
            seed=0,
            artifact_layout=self.build_optimized_layout(hardware_name),
            published_weights_root=Path("published_weights"),
            project_checkpoint_path=project_checkpoint_path,
            optimization_variant_name=optimization_variant_name,
            optimization_hypothesis_id=optimization_hypothesis_id,
            code_commit_hash=code_commit_hash,
            data_configuration=self._data_configuration,
            real_time_factor_configuration=self._real_time_factor_configuration,
            accelerator=accelerator
        )

    @property
    def data_configuration(self) -> LJSpeechDataConfig:
        # Returns the shared data record so rejection cases can restate it.
        return self._data_configuration


class ExperimentConfigurationConstructionTest(unittest.TestCase):
    # Verifies field acceptance, declared defaults, strict typing, required
    # fields, extra-field rejection, and immutability.
    def setUp(self) -> None:
        # Builds run descriptions under a relative root; no path is materialized.
        self._builder: ExperimentConfigurationBuilder = ExperimentConfigurationBuilder(
            Path("experiment_artifacts")
        )

    def test_valid_record_exposes_declared_identity_fields(self) -> None:
        # A well-formed record keeps the identity it was constructed with.
        configuration: ExperimentConfiguration = self._builder.build_reproduction()
        self.assertEqual(configuration.experiment_name, "vocode_reproduction")
        self.assertEqual(configuration.run_id, "run-alpha")
        self.assertEqual(configuration.evidence_category, "project_trained_reproduction")
        self.assertEqual(configuration.architecture_name, "hifigan_v1")
        self.assertEqual(configuration.seed, 0)
        self.assertEqual(configuration.published_weights_root, Path("published_weights"))

    def test_execution_fields_carry_declared_defaults(self) -> None:
        # An unspecified run executes one epoch, resolves its device, and profiles.
        configuration: ExperimentConfiguration = self._builder.build_reproduction()
        self.assertEqual(configuration.train_epoch_count, 1)
        self.assertEqual(configuration.accelerator, "auto")
        self.assertTrue(configuration.runtime_profiling_enabled)
        self.assertEqual(configuration.runtime_profile_interval_steps, 50)

    def test_optional_provenance_fields_default_to_absent(self) -> None:
        # A reproduction run declares no checkpoint, variant, or commit binding.
        configuration: ExperimentConfiguration = self._builder.build_reproduction()
        self.assertIsNone(configuration.project_checkpoint_path)
        self.assertIsNone(configuration.hiftnet_f0_checkpoint_path)
        self.assertIsNone(configuration.optimization_variant_name)
        self.assertIsNone(configuration.optimization_hypothesis_id)
        self.assertIsNone(configuration.code_commit_hash)

    def test_metric_selection_defaults_to_the_lightweight_panel(self) -> None:
        # The default selection is the registry's lightweight quality and complexity panel.
        configuration: ExperimentConfiguration = self._builder.build_reproduction()
        self.assertEqual(
            configuration.metric_selection.names,
            ("pesq", "stoi", "mel", "rtf", "parameters", "size")
        )

    def test_explicit_metric_selection_is_preserved(self) -> None:
        # A run may narrow its metric panel without touching the rest of the record.
        selection: MetricSelection = MetricSelection(names=("pesq", "parameters"))
        configuration: ExperimentConfiguration = self._builder.build_reproduction(
            metric_selection=selection
        )
        self.assertEqual(configuration.metric_selection.names, ("pesq", "parameters"))

    def test_string_published_weights_root_is_rejected_under_strict_validation(self) -> None:
        # Strict validation refuses to coerce a string into a path field.
        with self.assertRaises(ValidationError):
            ExperimentConfiguration(
                experiment_name="vocode_reproduction",
                run_id="run-alpha",
                hypothesis="h",
                interpretation_notes="",
                evidence_category="project_trained_reproduction",
                stage="test",
                dataset_split_name="test",
                architecture_name="hifigan_v1",
                seed=0,
                artifact_layout=self._builder.build_reproduction_layout(),
                published_weights_root="published_weights",
                data_configuration=self._builder.data_configuration,
                real_time_factor_configuration=RealTimeFactorConfig()
            )

    def test_missing_required_field_is_rejected(self) -> None:
        # The record has no implicit hypothesis; an incomplete run cannot be described.
        with self.assertRaises(ValidationError):
            ExperimentConfiguration(
                experiment_name="vocode_reproduction",
                run_id="run-alpha",
                interpretation_notes="",
                evidence_category="project_trained_reproduction",
                stage="test",
                dataset_split_name="test",
                architecture_name="hifigan_v1",
                seed=0,
                artifact_layout=self._builder.build_reproduction_layout(),
                published_weights_root=Path("published_weights"),
                data_configuration=self._builder.data_configuration,
                real_time_factor_configuration=RealTimeFactorConfig()
            )

    def test_unknown_field_is_rejected(self) -> None:
        # Experiment settings cannot accumulate unvalidated keys between runs.
        with self.assertRaises(ValidationError):
            ExperimentConfiguration(
                experiment_name="vocode_reproduction",
                run_id="run-alpha",
                hypothesis="h",
                interpretation_notes="",
                evidence_category="project_trained_reproduction",
                stage="test",
                dataset_split_name="test",
                architecture_name="hifigan_v1",
                seed=0,
                artifact_layout=self._builder.build_reproduction_layout(),
                published_weights_root=Path("published_weights"),
                data_configuration=self._builder.data_configuration,
                real_time_factor_configuration=RealTimeFactorConfig(),
                learning_rate=0.0002
            )

    def test_non_positive_epoch_count_is_rejected(self) -> None:
        # A training run must execute at least one epoch.
        with self.assertRaises(ValidationError):
            self._builder.build_reproduction(train_epoch_count=0)

    def test_record_is_frozen_against_field_assignment(self) -> None:
        # Settings cannot drift after the run has been described.
        configuration: ExperimentConfiguration = self._builder.build_reproduction()
        with self.assertRaises(ValidationError):
            configuration.seed: int = 3


class ExperimentConfigurationBatchLimitTest(unittest.TestCase):
    # Verifies the batch-limit domain across all four limit fields: absolute
    # integer counts, fractional loader limits, and the rejected values.
    def setUp(self) -> None:
        # Builds run descriptions whose test-batch limit is the varied field.
        self._builder: ExperimentConfigurationBuilder = ExperimentConfigurationBuilder(
            Path("experiment_artifacts")
        )

    def test_batch_limits_default_to_absent(self) -> None:
        # An unlimited run declares no limit on any loader.
        configuration: ExperimentConfiguration = self._builder.build_reproduction()
        self.assertIsNone(configuration.limit_train_batches)
        self.assertIsNone(configuration.limit_val_batches)
        self.assertIsNone(configuration.limit_test_batches)
        self.assertIsNone(configuration.limit_predict_batches)

    def test_integer_batch_limit_is_accepted_as_an_absolute_count(self) -> None:
        # An integer limit is a batch count and survives validation unchanged.
        configuration: ExperimentConfiguration = self._builder.build_reproduction(limit_test_batches=4)
        self.assertEqual(configuration.limit_test_batches, 4)

    def test_zero_integer_batch_limit_is_accepted(self) -> None:
        # Zero batches is a meaningful count that disables a loader.
        configuration: ExperimentConfiguration = self._builder.build_reproduction(limit_test_batches=0)
        self.assertEqual(configuration.limit_test_batches, 0)

    def test_fractional_batch_limit_inside_the_unit_interval_is_accepted(self) -> None:
        # A float limit is a fraction of the loader and its upper bound is inclusive.
        declared_limit: float
        for declared_limit in (0.1, 0.5, 1.0):
            configuration: ExperimentConfiguration = self._builder.build_reproduction(
                limit_test_batches=declared_limit
            )
            self.assertEqual(
                configuration.limit_test_batches,
                declared_limit,
                msg=f"fractional limit {declared_limit} should be accepted"
            )

    def test_negative_integer_batch_limit_is_rejected(self) -> None:
        # A negative batch count has no execution meaning.
        with self.assertRaisesRegex(ValidationError, "integer batch limits must be >= 0"):
            self._builder.build_reproduction(limit_test_batches=-1)

    def test_zero_fraction_batch_limit_is_rejected(self) -> None:
        # A zero fraction is excluded so an empty pass is expressed as the integer zero.
        with self.assertRaisesRegex(ValidationError, r"float batch limits must be in \(0.0, 1.0\]"):
            self._builder.build_reproduction(limit_test_batches=0.0)

    def test_fraction_batch_limit_above_one_is_rejected(self) -> None:
        # A fraction cannot exceed the whole loader.
        with self.assertRaisesRegex(ValidationError, r"float batch limits must be in \(0.0, 1.0\]"):
            self._builder.build_reproduction(limit_test_batches=1.5)

    def test_boolean_batch_limit_is_rejected_at_construction(self) -> None:
        # A bool must never pass as a count of zero or one.
        with self.assertRaises(ValidationError):
            self._builder.build_reproduction(limit_test_batches=True)

    def test_batch_limit_validator_names_the_boolean_rejection(self) -> None:
        # The field validator states why a bool is invalid rather than coercing it.
        with self.assertRaisesRegex(ValueError, "bool is invalid"):
            ExperimentConfiguration.validate_batch_limit(True)

    def test_batch_limit_validator_passes_absent_and_valid_values_through(self) -> None:
        # The validator is a domain guard, not a transformation.
        self.assertIsNone(ExperimentConfiguration.validate_batch_limit(None))
        self.assertEqual(ExperimentConfiguration.validate_batch_limit(6), 6)
        self.assertEqual(ExperimentConfiguration.validate_batch_limit(0.25), 0.25)


class ExperimentConfigurationStageBindingTest(unittest.TestCase):
    # Verifies that the executed stage and the evaluated dataset split must
    # name the same partition.
    def setUp(self) -> None:
        # Builds run descriptions whose stage and split are the varied fields.
        self._builder: ExperimentConfigurationBuilder = ExperimentConfigurationBuilder(
            Path("experiment_artifacts")
        )

    def test_matching_stage_and_split_are_accepted(self) -> None:
        # Every stage may run against the split of the same name.
        declared_stage: ExperimentStage
        for declared_stage in ("train", "validation", "test"):
            matching_split: DatasetSplitName = declared_stage
            configuration: ExperimentConfiguration = self._builder.build_reproduction(
                stage=declared_stage,
                dataset_split_name=matching_split
            )
            self.assertEqual(
                configuration.dataset_split_name,
                declared_stage,
                msg=f"stage {declared_stage} should accept its own split"
            )

    def test_mismatched_stage_and_split_are_rejected(self) -> None:
        # Evidence produced from one split can never carry another split's label.
        with self.assertRaisesRegex(ValidationError, "requires dataset_split_name"):
            self._builder.build_reproduction(stage="test", dataset_split_name="validation")

    def test_mismatch_message_names_both_the_stage_and_the_received_split(self) -> None:
        # The failure states the expected and the received split for debugging.
        with self.assertRaisesRegex(ValidationError, "stage='validation'"):
            self._builder.build_reproduction(stage="validation", dataset_split_name="train")


class ExperimentConfigurationOptimizedVariantTest(unittest.TestCase):
    # Verifies the provenance and accelerator requirements that bind an
    # optimized-variant capsule to its scientific and hardware identity.
    def setUp(self) -> None:
        # Builds optimized run descriptions whose provenance fields are varied.
        self._builder: ExperimentConfigurationBuilder = ExperimentConfigurationBuilder(
            Path("experiment_artifacts")
        )

    def test_complete_optimized_variant_record_is_accepted(self) -> None:
        # A fully declared optimized capsule passes every provenance requirement.
        configuration: ExperimentConfiguration = self._builder.build_optimized()
        self.assertEqual(configuration.optimization_variant_name, "int8_dynamic")
        self.assertEqual(configuration.optimization_hypothesis_id, "h_int8_dynamic")
        self.assertEqual(configuration.code_commit_hash, "0f1e2d3c4b5a")
        self.assertEqual(configuration.accelerator, "cpu")

    def test_missing_variant_name_is_rejected(self) -> None:
        # The capsule cannot start under a generic experiment label.
        with self.assertRaisesRegex(ValidationError, "optimization_variant_name"):
            self._builder.build_optimized(optimization_variant_name=None)

    def test_missing_project_checkpoint_path_is_rejected(self) -> None:
        # An optimized variant is measured against a declared base checkpoint.
        with self.assertRaisesRegex(ValidationError, "project_checkpoint_path"):
            self._builder.build_optimized(project_checkpoint_path=None)

    def test_missing_hypothesis_identifier_and_commit_are_both_reported(self) -> None:
        # The failure enumerates every missing provenance field at once.
        with self.assertRaisesRegex(ValidationError, "optimization_hypothesis_id.*code_commit_hash"):
            self._builder.build_optimized(
                optimization_hypothesis_id=None,
                code_commit_hash=None
            )

    def test_automatic_accelerator_is_rejected(self) -> None:
        # A resolved device must not be free to contradict the declared hardware lane.
        with self.assertRaisesRegex(ValidationError, "must declare an explicit accelerator"):
            self._builder.build_optimized(accelerator="auto")

    def test_cpu_lane_requires_the_cpu_accelerator(self) -> None:
        # A CPU capsule measured on another device would mislabel its lane.
        with self.assertRaisesRegex(ValidationError, "hardware_name='cpu' requires accelerator='cpu'"):
            self._builder.build_optimized(hardware_name="cpu", accelerator="cuda")

    def test_datacenter_lanes_require_a_cuda_accelerator(self) -> None:
        # NVIDIA lanes accept only the CUDA or GPU accelerator labels.
        declared_hardware: HardwareName
        for declared_hardware in ("b200", "h100", "a100_80gb", "l40s"):
            with self.assertRaisesRegex(ValidationError, "requires accelerator='cuda'"):
                self._builder.build_optimized(hardware_name=declared_hardware, accelerator="cpu")

    def test_datacenter_lanes_accept_cuda_and_gpu_labels(self) -> None:
        # Both accepted spellings of the accelerator resolve to the same lane.
        declared_accelerator: str
        for declared_accelerator in ("cuda", "gpu"):
            configuration: ExperimentConfiguration = self._builder.build_optimized(
                hardware_name="b200",
                accelerator=declared_accelerator
            )
            self.assertEqual(
                configuration.accelerator,
                declared_accelerator,
                msg=f"accelerator {declared_accelerator} should be accepted on the b200 lane"
            )

    def test_apple_lanes_accept_metal_and_cpu_accelerators(self) -> None:
        # Apple lanes may fall back to the CPU without changing the recorded lane.
        declared_hardware: HardwareName
        declared_accelerator: str
        for declared_hardware, declared_accelerator in (
            ("m3_max", "mps"),
            ("m3_max", "cpu"),
            ("mps", "mps"),
            ("mps", "cpu")
        ):
            configuration: ExperimentConfiguration = self._builder.build_optimized(
                hardware_name=declared_hardware,
                accelerator=declared_accelerator
            )
            self.assertEqual(configuration.artifact_layout.hardware_name, declared_hardware)
            self.assertEqual(
                configuration.accelerator,
                declared_accelerator,
                msg=f"accelerator {declared_accelerator} should be accepted on the {declared_hardware} lane"
            )

    def test_apple_lanes_reject_a_cuda_accelerator(self) -> None:
        # A Metal lane cannot be measured through CUDA.
        with self.assertRaisesRegex(ValidationError, "requires accelerator 'mps' or 'cpu'"):
            self._builder.build_optimized(hardware_name="m3_max", accelerator="cuda")

    def test_provenance_requirements_do_not_apply_to_other_categories(self) -> None:
        # Reproduction capsules stay valid without variant provenance or a pinned device.
        configuration: ExperimentConfiguration = self._builder.build_reproduction()
        self.assertEqual(configuration.accelerator, "auto")
        self.assertIsNone(configuration.optimization_variant_name)


class ExperimentConfigurationArtifactPathDelegationTest(unittest.TestCase):
    # Verifies that every artifact path exposed by the configuration is the
    # artifact layout's path, so runners never compose one themselves.
    def setUp(self) -> None:
        # Binds one configuration and its layout as the two sides of the comparison.
        self._builder: ExperimentConfigurationBuilder = ExperimentConfigurationBuilder(
            Path("experiment_artifacts")
        )
        self._configuration: ExperimentConfiguration = self._builder.build_reproduction()
        self._layout: ExperimentArtifactLayout = self._configuration.artifact_layout

    def test_artifact_root_and_run_directory_come_from_the_layout(self) -> None:
        # The capsule root and run directory are the layout's own values.
        self.assertEqual(self._configuration.artifact_root, self._layout.artifact_root)
        self.assertEqual(self._configuration.run_directory, self._layout.run_directory)

    def test_checkpoint_and_summary_paths_come_from_the_layout(self) -> None:
        # Checkpoint storage and the summary surface are layout decisions.
        self.assertEqual(self._configuration.checkpoints_directory, self._layout.checkpoints_directory)
        self.assertEqual(self._configuration.summary_csv_path, self._layout.summary_csv_path)

    def test_persisted_record_paths_come_from_the_layout(self) -> None:
        # Resolved configuration, hyperparameters, manifest, and metrics all delegate.
        self.assertEqual(
            self._configuration.resolved_configuration_path,
            self._layout.resolved_configuration_path
        )
        self.assertEqual(self._configuration.hyperparameters_path, self._layout.hyperparameters_path)
        self.assertEqual(self._configuration.run_manifest_path, self._layout.run_manifest_path)
        self.assertEqual(self._configuration.metrics_path, self._layout.metrics_path)

    def test_delegated_paths_nest_under_the_declared_artifact_root(self) -> None:
        # Every delegated path stays inside the capsule tree of this run.
        declared_path: Path
        for declared_path in (
            self._configuration.run_directory,
            self._configuration.checkpoints_directory,
            self._configuration.summary_csv_path,
            self._configuration.metrics_path
        ):
            self.assertTrue(
                declared_path.is_relative_to(self._configuration.artifact_root),
                msg=f"{declared_path} escaped the artifact root"
            )


if __name__ == "__main__":
    unittest.main()
