# This module:
# 1. Verifies the recovery-variant resolution performed at runner
#    construction: the three registered recovery arms are accepted and every
#    other variant name, including an absent one, fails closed
# 2. Verifies the architecture resolution performed at construction: each
#    architecture family resolves its own pruning arm, and an architecture
#    without a declared technique resolution fails closed
# 3. Verifies the stage guard of the run entry point, including that a
#    refused run leaves no artifact behind
#
# Design decisions:
# - No fit, validate, or checkpoint load is ever executed here; recovery
#   training builds a model registry entry, restores a SHA-verified baseline
#   checkpoint, and fine-tunes on the real corpus, all of which are outside
#   the scope of a unit suite
# - Configurations are constructed through the real validated
#   ExperimentConfiguration inside a temporary artifact root, so the runner
#   receives exactly the record the command-line surface would hand it
# - The stage guard is the first statement of run(), which makes it the one
#   entry-point behaviour that can be asserted without training; the absence
#   of any created artifact directory is asserted alongside it
#
# Author: Rahul Sawhney

import tempfile
import unittest
from pathlib import Path

from syntheticmind.utilities.exceptions import MisconfigurationError

from vocode.configs.layout import ExperimentArtifactLayout, ExperimentStage
from vocode.configs.run import ExperimentConfiguration
from vocode.data.ljspeech_datamodule import LJSpeechDataConfig
from vocode.metrics.rtf import RealTimeFactorConfig
from vocode.models.vocoder import ArchitectureName
from vocode.optimization.recovery import OptimizationRecoveryRunner


class RecoveryConfigurationBuilder:
    # Builds validated run configurations inside a temporary artifact root.
    # The records are the real validated ones the command-line surface produces,
    # so the runner's construction-time resolution is exercised against exactly
    # the object it receives in production; every path is anchored in the
    # temporary root, so no case reaches the corpus or the real artifact tree.
    def __init__(self, root: Path) -> None:
        # Binds the temporary root every built configuration is anchored in.
        #
        # Args:
        #     root: The per-case temporary directory all built paths descend
        #         from.
        self._root: Path = root

    def build_optimized(
        self,
        architecture_name: ArchitectureName,
        variant_name: str,
        stage: ExperimentStage
    ) -> ExperimentConfiguration:
        # Builds an optimized-variant run description for the requested cell and stage.
        # The variant name and the stage are both varied because the runner
        # resolves the variant at construction and gates the stage at run, so the
        # two guards need to be reachable independently.
        #
        # Args:
        #     architecture_name: The architecture the run declares, which
        #         selects the pruning arm the runner resolves.
        #     variant_name: The optimization variant the run declares,
        #         including ones the recovery runner must refuse.
        #     stage: The stage the run declares, including ones the run entry
        #         point must refuse.
        #
        # Returns:
        #     A validated optimized-variant run description anchored in the
        #     temporary root.
        layout: ExperimentArtifactLayout = ExperimentArtifactLayout(
            artifact_root=self._root / "artifacts",
            evidence_category="project_optimized_variants",
            architecture_name=architecture_name,
            hardware_name="cpu",
            precision_name="fp32",
            seed=0,
            run_id="run-0001",
            variant_name=variant_name
        )
        return ExperimentConfiguration(
            experiment_name="recovery-unit",
            run_id="run-0001",
            hypothesis="Recovery restores quality lost to masking.",
            interpretation_notes="Unit-suite configuration; no run is executed.",
            evidence_category="project_optimized_variants",
            stage=stage,
            dataset_split_name=stage,
            architecture_name=architecture_name,
            seed=0,
            artifact_layout=layout,
            published_weights_root=self._root / "published_weights",
            project_checkpoint_path=self._root / "baseline.ckpt",
            optimization_variant_name=variant_name,
            optimization_hypothesis_id="H-OPT-1",
            code_commit_hash="0123456789abcdef",
            data_configuration=LJSpeechDataConfig(dataset_root=self._root / "corpus"),
            real_time_factor_configuration=RealTimeFactorConfig(),
            accelerator="cpu"
        )

    def build_reproduction(self, architecture_name: ArchitectureName) -> ExperimentConfiguration:
        # Builds a reproduction run description, which declares no optimization variant.
        # This is the only way to reach the absent-variant refusal, because the
        # optimized evidence category requires a variant name to validate.
        #
        # Args:
        #     architecture_name: The architecture the run declares.
        #
        # Returns:
        #     A validated reproduction run description carrying no
        #     optimization variant.
        layout: ExperimentArtifactLayout = ExperimentArtifactLayout(
            artifact_root=self._root / "artifacts",
            evidence_category="project_trained_reproduction",
            architecture_name=architecture_name,
            hardware_name="cpu",
            precision_name="fp32",
            seed=0,
            run_id="run-0002"
        )
        return ExperimentConfiguration(
            experiment_name="reproduction-unit",
            run_id="run-0002",
            hypothesis="Baseline reproduction without any optimization variant.",
            interpretation_notes="Unit-suite configuration; no run is executed.",
            evidence_category="project_trained_reproduction",
            stage="train",
            dataset_split_name="train",
            architecture_name=architecture_name,
            seed=0,
            artifact_layout=layout,
            published_weights_root=self._root / "published_weights",
            data_configuration=LJSpeechDataConfig(dataset_root=self._root / "corpus"),
            real_time_factor_configuration=RealTimeFactorConfig(),
            accelerator="cpu"
        )


class RecoveryVariantResolutionTest(unittest.TestCase):
    # Verifies which variant names the recovery runner accepts at construction.
    def setUp(self) -> None:
        # Opens the temporary root and the builder every case draws a configuration from.
        self._temporary_directory: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
        self._builder: RecoveryConfigurationBuilder = RecoveryConfigurationBuilder(
            Path(self._temporary_directory.name)
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_registered_recovery_variants_are_accepted(self) -> None:
        # The runner trains exactly the three registered recovery arms, and each
        # arm is the one the configuration declared.
        variant_name: str
        for variant_name in ("pruned_50_recovered", "pruned_50_recovered_half", "dense_continued"):
            configuration: ExperimentConfiguration = self._builder.build_optimized(
                "vocos",
                variant_name,
                "train"
            )
            runner: OptimizationRecoveryRunner = OptimizationRecoveryRunner(configuration)
            self.assertEqual(
                runner._recovery_variant,
                variant_name,
                msg=f"{variant_name} must resolve to its own recovery arm."
            )

    def test_evaluation_only_variant_is_refused(self) -> None:
        # A variant that carries no recovery training cannot be trained by this runner.
        configuration: ExperimentConfiguration = self._builder.build_optimized(
            "vocos",
            "int8_dynamic",
            "train"
        )
        with self.assertRaisesRegex(MisconfigurationError, "Optimization recovery supports"):
            OptimizationRecoveryRunner(configuration)

    def test_masked_curve_variant_is_refused(self) -> None:
        # The masked curve is an evaluation-time technique, not a recovery arm.
        configuration: ExperimentConfiguration = self._builder.build_optimized(
            "vocos",
            "pruned_50",
            "train"
        )
        with self.assertRaisesRegex(MisconfigurationError, "pruned_50_recovered"):
            OptimizationRecoveryRunner(configuration)

    def test_absent_variant_name_is_refused(self) -> None:
        # A run without a declared optimization variant has no recovery arm to execute.
        configuration: ExperimentConfiguration = self._builder.build_reproduction("vocos")
        with self.assertRaisesRegex(MisconfigurationError, "None"):
            OptimizationRecoveryRunner(configuration)


class RecoveryArchitectureResolutionTest(unittest.TestCase):
    # Verifies the pruning-arm resolution performed at runner construction.
    def setUp(self) -> None:
        # Opens the temporary root and the builder every case draws a configuration from.
        self._temporary_directory: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
        self._builder: RecoveryConfigurationBuilder = RecoveryConfigurationBuilder(
            Path(self._temporary_directory.name)
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_each_architecture_family_resolves_its_own_pruning_arm(self) -> None:
        # A Linear-bearing backbone resolves the structured arm and a
        # convolution-dominant generator resolves the global unstructured arm.
        expected_arms: dict[ArchitectureName, str] = {
            "vocos": "linear_structured",
            "hifigan_v1": "global_unstructured"
        }
        architecture_name: ArchitectureName
        expected_arm: str
        for architecture_name, expected_arm in expected_arms.items():
            configuration: ExperimentConfiguration = self._builder.build_optimized(
                architecture_name,
                "pruned_50_recovered",
                "train"
            )
            runner: OptimizationRecoveryRunner = OptimizationRecoveryRunner(configuration)
            self.assertEqual(
                runner._pruning_arm,
                expected_arm,
                msg=f"{architecture_name} must resolve the {expected_arm} pruning arm."
            )

    def test_architecture_without_a_declared_resolution_is_refused(self) -> None:
        # The excluded architecture has no technique resolution and cannot be recovered.
        configuration: ExperimentConfiguration = self._builder.build_optimized(
            "hiftnet",
            "pruned_50_recovered",
            "train"
        )
        with self.assertRaisesRegex(MisconfigurationError, "No technique resolution declared"):
            OptimizationRecoveryRunner(configuration)


class RecoveryStageValidationTest(unittest.TestCase):
    # Verifies the stage guard protecting the recovery run entry point.
    def setUp(self) -> None:
        # Opens the temporary root and retains it, so the artifact assertion can read it.
        self._temporary_directory: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._builder: RecoveryConfigurationBuilder = RecoveryConfigurationBuilder(self._root)

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_evaluation_stages_are_refused_by_the_run_entry_point(self) -> None:
        # Recovery is a training path; evaluation stages must not enter it.
        stage: ExperimentStage
        for stage in ("test", "validation"):
            configuration: ExperimentConfiguration = self._builder.build_optimized(
                "vocos",
                "pruned_50_recovered",
                stage
            )
            runner: OptimizationRecoveryRunner = OptimizationRecoveryRunner(configuration)
            with self.assertRaisesRegex(MisconfigurationError, "only supports stage='train'"):
                runner.run()

    def test_refused_run_writes_no_artifact(self) -> None:
        # The guard runs before any capsule directory is created.
        configuration: ExperimentConfiguration = self._builder.build_optimized(
            "vocos",
            "dense_continued",
            "test"
        )
        runner: OptimizationRecoveryRunner = OptimizationRecoveryRunner(configuration)
        with self.assertRaises(MisconfigurationError):
            runner.run()
        self.assertFalse(
            (self._root / "artifacts").exists(),
            msg="A refused recovery run must leave no artifact tree behind."
        )


if __name__ == "__main__":
    unittest.main()
