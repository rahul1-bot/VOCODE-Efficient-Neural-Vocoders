# This module:
# 1. Verifies that ExperimentArtifactLayout accepts only well-typed identity
#    fields under its strict, frozen, extra-forbidding configuration
# 2. Verifies the variant-binding invariant that ties the variant segment to
#    the optimized-variants evidence category in both directions
# 3. Verifies every derived path property: the hardware-precision label per
#    evidence lane, the capsule directory tree, and the file locations of the
#    summary, configuration, manifest, metric, and log artifacts
# 4. Verifies that directory creation materializes exactly the run
#    directories the record declares and stays idempotent
#
# Design decisions:
# - Path assertions compare against paths composed with pathlib rather than
#   literal strings, so the expectations stay platform independent
# - Every path property except directory creation is pure computation, so
#   those cases use a relative artifact root and never touch the filesystem
# - Only the directory-creation case allocates a temporary directory, which
#   is removed in tearDown, so no test leaves state behind
# - Rejection cases construct the record directly instead of going through
#   the record builder, because the invalid value is the subject of the test
#   and the builder parameters carry the closed label types
# - The label matrix is exercised across an NVIDIA lane, an Apple lane, and
#   the CPU lane, because the optimized-variants root drops the precision
#   segment and renames the CPU lane while every other category keeps both
#
# Author: Rahul Sawhney

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from vocode.configs.layout import EvidenceCategory, ExperimentArtifactLayout, HardwareName, PrecisionName
from vocode.models.vocoder import ArchitectureName


class LayoutRecordBuilder:
    # Builds valid artifact-layout records rooted at one artifact root so the
    # test cases state only the fields they vary.
    def __init__(self, artifact_root: Path) -> None:
        # Binds the artifact root every record built here is anchored to.
        self._artifact_root: Path = artifact_root

    def build_standard(
        self,
        evidence_category: EvidenceCategory = "project_trained_reproduction",
        hardware_name: HardwareName = "b200",
        precision_name: PrecisionName = "fp32",
        architecture_name: ArchitectureName = "hifigan_v1",
        seed: int = 0,
        run_id: str = "run-alpha"
    ) -> ExperimentArtifactLayout:
        # Builds a record in a category that carries no variant segment.
        return ExperimentArtifactLayout(
            artifact_root=self._artifact_root,
            evidence_category=evidence_category,
            architecture_name=architecture_name,
            hardware_name=hardware_name,
            precision_name=precision_name,
            seed=seed,
            run_id=run_id
        )

    def build_optimized(
        self,
        variant_name: str = "int8_dynamic",
        hardware_name: HardwareName = "cpu",
        precision_name: PrecisionName = "fp32",
        architecture_name: ArchitectureName = "vocos",
        seed: int = 0,
        run_id: str = "run-beta"
    ) -> ExperimentArtifactLayout:
        # Builds a record in the one category that owns a variant segment.
        return ExperimentArtifactLayout(
            artifact_root=self._artifact_root,
            evidence_category="project_optimized_variants",
            architecture_name=architecture_name,
            hardware_name=hardware_name,
            precision_name=precision_name,
            seed=seed,
            run_id=run_id,
            variant_name=variant_name
        )


class ExperimentArtifactLayoutConstructionTest(unittest.TestCase):
    # Verifies field acceptance, declared defaults, strict typing, closed
    # label domains, extra-field rejection, and immutability.
    def setUp(self) -> None:
        # Builds records under a relative root, so no case touches the filesystem.
        self._builder: LayoutRecordBuilder = LayoutRecordBuilder(Path("experiment_artifacts"))

    def test_valid_record_exposes_declared_identity_fields(self) -> None:
        # A well-formed record keeps every identity field it was constructed with.
        layout: ExperimentArtifactLayout = self._builder.build_standard(
            hardware_name="h100",
            precision_name="bf16",
            architecture_name="melgan",
            seed=1337,
            run_id="run-identity"
        )
        self.assertEqual(layout.artifact_root, Path("experiment_artifacts"))
        self.assertEqual(layout.evidence_category, "project_trained_reproduction")
        self.assertEqual(layout.architecture_name, "melgan")
        self.assertEqual(layout.hardware_name, "h100")
        self.assertEqual(layout.precision_name, "bf16")
        self.assertEqual(layout.seed, 1337)
        self.assertEqual(layout.run_id, "run-identity")

    def test_dataset_name_and_variant_name_carry_declared_defaults(self) -> None:
        # The dataset label defaults to the studied corpus and the variant segment is absent.
        layout: ExperimentArtifactLayout = self._builder.build_standard()
        self.assertEqual(layout.dataset_name, "ljspeech")
        self.assertIsNone(layout.variant_name)

    def test_string_artifact_root_is_rejected_under_strict_validation(self) -> None:
        # Strict validation refuses to coerce a string into the artifact root Path.
        with self.assertRaises(ValidationError):
            ExperimentArtifactLayout(
                artifact_root="experiment_artifacts",
                evidence_category="project_trained_reproduction",
                architecture_name="hifigan_v1",
                hardware_name="b200",
                precision_name="fp32",
                seed=0,
                run_id="run-alpha"
            )

    def test_boolean_seed_is_rejected_under_strict_validation(self) -> None:
        # Strict validation refuses a bool where the seed integer is declared.
        with self.assertRaises(ValidationError):
            ExperimentArtifactLayout(
                artifact_root=Path("experiment_artifacts"),
                evidence_category="project_trained_reproduction",
                architecture_name="hifigan_v1",
                hardware_name="b200",
                precision_name="fp32",
                seed=True,
                run_id="run-alpha"
            )

    def test_unknown_evidence_category_is_rejected(self) -> None:
        # The evidence-category vocabulary is closed at construction.
        with self.assertRaises(ValidationError):
            ExperimentArtifactLayout(
                artifact_root=Path("experiment_artifacts"),
                evidence_category="project_unlisted_variants",
                architecture_name="hifigan_v1",
                hardware_name="b200",
                precision_name="fp32",
                seed=0,
                run_id="run-alpha"
            )

    def test_unknown_hardware_name_is_rejected(self) -> None:
        # The hardware vocabulary is closed at construction.
        with self.assertRaises(ValidationError):
            ExperimentArtifactLayout(
                artifact_root=Path("experiment_artifacts"),
                evidence_category="project_trained_reproduction",
                architecture_name="hifigan_v1",
                hardware_name="gh200",
                precision_name="fp32",
                seed=0,
                run_id="run-alpha"
            )

    def test_unknown_precision_name_is_rejected(self) -> None:
        # The precision vocabulary is closed at construction.
        with self.assertRaises(ValidationError):
            ExperimentArtifactLayout(
                artifact_root=Path("experiment_artifacts"),
                evidence_category="project_trained_reproduction",
                architecture_name="hifigan_v1",
                hardware_name="b200",
                precision_name="int8",
                seed=0,
                run_id="run-alpha"
            )

    def test_unknown_architecture_name_is_rejected(self) -> None:
        # The architecture vocabulary is the registry literal, not a free string.
        with self.assertRaises(ValidationError):
            ExperimentArtifactLayout(
                artifact_root=Path("experiment_artifacts"),
                evidence_category="project_trained_reproduction",
                architecture_name="hifigan_v9",
                hardware_name="b200",
                precision_name="fp32",
                seed=0,
                run_id="run-alpha"
            )

    def test_unknown_field_is_rejected(self) -> None:
        # The record is closed, so an unrecognized key cannot accumulate silently.
        with self.assertRaises(ValidationError):
            ExperimentArtifactLayout(
                artifact_root=Path("experiment_artifacts"),
                evidence_category="project_trained_reproduction",
                architecture_name="hifigan_v1",
                hardware_name="b200",
                precision_name="fp32",
                seed=0,
                run_id="run-alpha",
                checkpoint_epoch=4
            )

    def test_dataset_name_cannot_leave_its_single_registered_value(self) -> None:
        # The dataset label is a closed literal pinned to the studied corpus.
        with self.assertRaises(ValidationError):
            ExperimentArtifactLayout(
                artifact_root=Path("experiment_artifacts"),
                evidence_category="project_trained_reproduction",
                architecture_name="hifigan_v1",
                hardware_name="b200",
                precision_name="fp32",
                seed=0,
                run_id="run-alpha",
                dataset_name="vctk"
            )

    def test_record_is_frozen_against_field_assignment(self) -> None:
        # Identity fields cannot drift after construction.
        layout: ExperimentArtifactLayout = self._builder.build_standard()
        with self.assertRaises(ValidationError):
            layout.seed: int = 11


class ExperimentArtifactLayoutVariantBindingTest(unittest.TestCase):
    # Verifies that the variant segment is required by the optimized-variants
    # category and rejected by every other category.
    def setUp(self) -> None:
        # Builds records under a relative root, so no case touches the filesystem.
        self._builder: LayoutRecordBuilder = LayoutRecordBuilder(Path("experiment_artifacts"))

    def test_optimized_category_without_variant_name_is_rejected(self) -> None:
        # An optimized capsule without its variant label has no addressable directory.
        with self.assertRaisesRegex(ValidationError, "variant_name is required"):
            ExperimentArtifactLayout(
                artifact_root=Path("experiment_artifacts"),
                evidence_category="project_optimized_variants",
                architecture_name="vocos",
                hardware_name="cpu",
                precision_name="fp32",
                seed=0,
                run_id="run-beta"
            )

    def test_variant_name_outside_optimized_category_is_rejected(self) -> None:
        # Only the optimized-variants lane owns a variant segment.
        declared_category: EvidenceCategory
        for declared_category in (
            "published_checkpoint_evaluation",
            "project_trained_reproduction",
            "project_hybrid_variants"
        ):
            with self.assertRaisesRegex(ValidationError, "variant_name is only valid"):
                ExperimentArtifactLayout(
                    artifact_root=Path("experiment_artifacts"),
                    evidence_category=declared_category,
                    architecture_name="vocos",
                    hardware_name="cpu",
                    precision_name="fp32",
                    seed=0,
                    run_id="run-beta",
                    variant_name="int8_dynamic"
                )

    def test_optimized_category_with_variant_name_is_accepted(self) -> None:
        # The bound pair is the only accepted optimized-variant form.
        layout: ExperimentArtifactLayout = self._builder.build_optimized(variant_name="pruned_50")
        self.assertEqual(layout.evidence_category, "project_optimized_variants")
        self.assertEqual(layout.variant_name, "pruned_50")


class ExperimentArtifactLayoutPathCompositionTest(unittest.TestCase):
    # Verifies the hardware-precision label per lane and every derived
    # directory and file path of the run capsule.
    def setUp(self) -> None:
        # Path composition is pure computation, so the root stays relative here.
        self._artifact_root: Path = Path("experiment_artifacts")
        self._builder: LayoutRecordBuilder = LayoutRecordBuilder(self._artifact_root)

    def test_nvidia_lane_label_prefixes_the_vendor_and_pins_precision(self) -> None:
        # Datacenter accelerators carry a vendor prefix and the precision segment.
        declared_hardware: HardwareName
        for declared_hardware in ("b200", "h100", "a100_80gb", "l40s"):
            layout: ExperimentArtifactLayout = self._builder.build_standard(
                hardware_name=declared_hardware,
                precision_name="fp16"
            )
            self.assertEqual(
                layout.hardware_precision_directory,
                f"nvidia_{declared_hardware}_fp16",
                msg=f"unexpected label for hardware lane {declared_hardware}"
            )

    def test_non_nvidia_lane_label_is_the_bare_hardware_and_precision(self) -> None:
        # Local and generic lanes carry no vendor prefix.
        declared_hardware: HardwareName
        for declared_hardware in ("cuda", "mps", "m3_max", "cpu"):
            layout: ExperimentArtifactLayout = self._builder.build_standard(
                hardware_name=declared_hardware,
                precision_name="bf16"
            )
            self.assertEqual(
                layout.hardware_precision_directory,
                f"{declared_hardware}_bf16",
                msg=f"unexpected label for hardware lane {declared_hardware}"
            )

    def test_optimized_lane_label_drops_precision_and_renames_the_cpu_lane(self) -> None:
        # Precision is the experimental variable there, so the label encodes hardware only.
        cpu_layout: ExperimentArtifactLayout = self._builder.build_optimized(hardware_name="cpu")
        nvidia_layout: ExperimentArtifactLayout = self._builder.build_optimized(hardware_name="b200")
        apple_layout: ExperimentArtifactLayout = self._builder.build_optimized(hardware_name="m3_max")
        self.assertEqual(cpu_layout.hardware_precision_directory, "modal_cpu8")
        self.assertEqual(nvidia_layout.hardware_precision_directory, "nvidia_b200")
        self.assertEqual(apple_layout.hardware_precision_directory, "m3_max")

    def test_optimized_lane_label_ignores_the_declared_precision(self) -> None:
        # Two precisions in the optimized lane resolve to the same context directory.
        first_layout: ExperimentArtifactLayout = self._builder.build_optimized(precision_name="fp32")
        second_layout: ExperimentArtifactLayout = self._builder.build_optimized(precision_name="fp16")
        self.assertEqual(
            first_layout.hardware_precision_directory,
            second_layout.hardware_precision_directory
        )

    def test_evaluation_context_joins_dataset_and_hardware_precision_label(self) -> None:
        # The context label is the dataset name and the lane label.
        layout: ExperimentArtifactLayout = self._builder.build_standard(
            hardware_name="m3_max",
            precision_name="fp32"
        )
        self.assertEqual(layout.evaluation_context, "ljspeech_m3_max_fp32")

    def test_context_directory_encodes_category_dataset_and_lane(self) -> None:
        # The context directory is the evidence root of one lane.
        layout: ExperimentArtifactLayout = self._builder.build_standard()
        expected_directory: Path = (
            self._artifact_root / "project_trained_reproduction" / "ljspeech" / "nvidia_b200_fp32"
        )
        self.assertEqual(layout.context_directory, expected_directory)

    def test_model_directory_appends_architecture_without_variant_segment(self) -> None:
        # Non-optimized capsules address the architecture directly.
        layout: ExperimentArtifactLayout = self._builder.build_standard(architecture_name="bigvgan")
        self.assertEqual(layout.model_directory, layout.context_directory / "bigvgan")

    def test_model_directory_appends_variant_segment_for_optimized_capsules(self) -> None:
        # Optimized capsules insert the variant between architecture and runs.
        layout: ExperimentArtifactLayout = self._builder.build_optimized(
            architecture_name="vocos",
            variant_name="torch_compile"
        )
        self.assertEqual(
            layout.model_directory,
            layout.context_directory / "vocos" / "torch_compile"
        )

    def test_run_directory_nests_the_run_identifier_under_runs(self) -> None:
        # The run identifier is the leaf of the capsule tree.
        layout: ExperimentArtifactLayout = self._builder.build_standard(run_id="run-gamma")
        self.assertEqual(layout.run_directory, layout.model_directory / "runs" / "run-gamma")

    def test_run_scoped_directories_hang_off_the_run_directory(self) -> None:
        # Checkpoints, logs, metrics, and seed records are siblings inside the run capsule.
        layout: ExperimentArtifactLayout = self._builder.build_standard()
        run_directory: Path = layout.run_directory
        self.assertEqual(layout.checkpoints_directory, run_directory / "checkpoints")
        self.assertEqual(layout.logs_directory, run_directory / "logs")
        self.assertEqual(layout.metrics_directory, run_directory / "metrics")
        self.assertEqual(layout.seed_directory, run_directory / "seed_records")

    def test_run_scoped_files_use_their_canonical_names(self) -> None:
        # Every persisted run artifact has one fixed location.
        layout: ExperimentArtifactLayout = self._builder.build_standard()
        run_directory: Path = layout.run_directory
        self.assertEqual(layout.resolved_configuration_path, run_directory / "resolved_config.yaml")
        self.assertEqual(layout.hyperparameters_path, run_directory / "hyperparameters.yaml")
        self.assertEqual(layout.run_manifest_path, run_directory / "run_manifest.yaml")
        self.assertEqual(layout.metrics_path, run_directory / "metrics" / "metrics.json")
        self.assertEqual(layout.execution_log_path, run_directory / "logs" / "execution.log")

    def test_summary_csv_path_uses_the_first_schema_outside_the_optimized_lane(self) -> None:
        # Historical summary surfaces keep their original file name.
        layout: ExperimentArtifactLayout = self._builder.build_standard(
            evidence_category="published_checkpoint_evaluation"
        )
        self.assertEqual(
            layout.summary_csv_path,
            layout.context_directory / "summary" / "experiments.csv"
        )

    def test_summary_csv_path_uses_the_versioned_schema_in_the_optimized_lane(self) -> None:
        # The optimized lane writes the second-schema surface so frozen evidence stays untouched.
        layout: ExperimentArtifactLayout = self._builder.build_optimized()
        self.assertEqual(
            layout.summary_csv_path,
            layout.context_directory / "summary" / "experiments_v2.csv"
        )

    def test_summary_surface_is_shared_across_architectures_of_one_lane(self) -> None:
        # The summary lives above the architecture, so one lane aggregates one file.
        first_layout: ExperimentArtifactLayout = self._builder.build_standard(architecture_name="vocos")
        second_layout: ExperimentArtifactLayout = self._builder.build_standard(architecture_name="melgan")
        self.assertEqual(first_layout.summary_csv_path, second_layout.summary_csv_path)

    def test_distinct_seeds_do_not_separate_capsules_by_themselves(self) -> None:
        # The run identifier, not the seed, distinguishes capsule directories. The seed is
        # recorded as provenance but takes no part in path composition, so two seeds sharing
        # one run identifier address the same capsule and the second would write over the
        # first. Separating seeds is therefore the caller's responsibility when it chooses the
        # run identifier, and this case pins that the layout does not do it for them.
        first_layout: ExperimentArtifactLayout = self._builder.build_standard(seed=0, run_id="run-shared")
        second_layout: ExperimentArtifactLayout = self._builder.build_standard(seed=7, run_id="run-shared")
        self.assertEqual(first_layout.run_directory, second_layout.run_directory)


class ExperimentArtifactLayoutDirectoryCreationTest(unittest.TestCase):
    # Verifies that run-directory creation materializes the declared capsule
    # directories, is repeatable, and creates nothing it did not declare.
    def setUp(self) -> None:
        # Only this case writes, so it roots its records in a temporary tree.
        self._temporary_root: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._artifact_root: Path = Path(self._temporary_root.name)
        self._builder: LayoutRecordBuilder = LayoutRecordBuilder(self._artifact_root)

    def tearDown(self) -> None:
        # Removes the created capsule tree so no test leaves state behind.
        self._temporary_root.cleanup()

    def test_create_run_directories_materializes_the_declared_directories(self) -> None:
        # Every directory the runner writes into exists after creation.
        layout: ExperimentArtifactLayout = self._builder.build_standard()
        layout.create_run_directories()
        self.assertTrue(layout.run_directory.is_dir())
        self.assertTrue(layout.checkpoints_directory.is_dir())
        self.assertTrue(layout.logs_directory.is_dir())
        self.assertTrue(layout.metrics_directory.is_dir())
        self.assertTrue(layout.summary_csv_path.parent.is_dir())

    def test_create_run_directories_is_idempotent(self) -> None:
        # Re-running a command against an existing capsule must not fail.
        layout: ExperimentArtifactLayout = self._builder.build_standard()
        layout.create_run_directories()
        layout.create_run_directories()
        self.assertTrue(layout.run_directory.is_dir())

    def test_create_run_directories_leaves_the_seed_directory_to_its_writer(self) -> None:
        # Seed records are created by the component that writes them, not by the layout.
        layout: ExperimentArtifactLayout = self._builder.build_standard()
        layout.create_run_directories()
        self.assertFalse(layout.seed_directory.exists())

    def test_create_run_directories_writes_no_artifact_files(self) -> None:
        # Creation prepares directories only; no summary or metric file is fabricated.
        layout: ExperimentArtifactLayout = self._builder.build_optimized()
        layout.create_run_directories()
        self.assertFalse(layout.summary_csv_path.exists())
        self.assertFalse(layout.metrics_path.exists())
        self.assertFalse(layout.execution_log_path.exists())

    def test_optimized_capsule_creates_the_variant_scoped_tree(self) -> None:
        # The created tree carries the variant segment of the optimized lane.
        layout: ExperimentArtifactLayout = self._builder.build_optimized(
            architecture_name="hifigan_v1",
            variant_name="pruned_30"
        )
        layout.create_run_directories()
        expected_directory: Path = (
            self._artifact_root
            / "project_optimized_variants"
            / "ljspeech"
            / "modal_cpu8"
            / "hifigan_v1"
            / "pruned_30"
            / "runs"
            / "run-beta"
        )
        self.assertEqual(layout.run_directory, expected_directory)
        self.assertTrue(expected_directory.is_dir())


if __name__ == "__main__":
    unittest.main()
