# This module:
# 1. Verifies ReproductionTrainingRunner construction, the per-family training
#    policy (adversarial, autoregressive, and flow step cadence versus the
#    epoch cadence), the artifact-to-harness precision mapping, and the
#    callback policy that withholds EarlyStopping and EMA from the reference
#    recipes that never used them
# 2. Verifies the harness Trainer each dispatch method builds: epoch budget,
#    precision, checkpoint cadence and boundaries, validation cadence and
#    scope, logging cadence, batch limits, and callback ordering
# 3. Verifies the evidence identity the runner stamps onto its rows (unique
#    id, variant prefix, evidence label, hyperparameter dump, metrics
#    snapshot) and its validation errors (registry status gate, missing
#    project checkpoint, checkpoint without a model state dictionary)
#
# Design decisions:
# - A harness entry point is replaced by a recording stand-in inside a scoped
#   context manager, so the Trainer the runner actually builds is inspected
#   while no fit, validation, test, or prediction loop ever executes; the
#   original entry point is restored on both the success and the failure path
# - Module specs are assembled around a minimal harness Module stub rather
#   than registry-built reference vocoders, because registry construction is
#   the registry's own contract and building real architectures per test would
#   dominate the runtime budget
# - Every configuration is rooted in a per-test temporary artifact tree, so no
#   test reads the LJSpeech corpus, author weights, or the real artifact tree
# - Boundaries excluded: _build_module_spec and the post-gate body of run(),
#   because both construct reference architectures through the model registry
#   and then drive harness loops over the real corpus
#
# Author: Rahul Sawhney

import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.early_stopping import EarlyStopping
from syntheticmind.callbacks.ema import EMACallback
from syntheticmind.callbacks.model_checkpoint import ModelCheckpoint
from syntheticmind.callbacks.nan_inf_guard import NanInfGuard
from syntheticmind.callbacks.runtime_profiler import RuntimeProfiler
from syntheticmind.core.module import Module
from syntheticmind.core.trainer import Trainer
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.types import HyperparameterDict, Precision

from vocode.configs.layout import EvidenceCategory, ExperimentArtifactLayout, ExperimentStage, PrecisionName
from vocode.configs.run import ExperimentConfiguration
from vocode.data.ljspeech_datamodule import LJSpeechDataConfig, LJSpeechDataModule
from vocode.loggers.experiment import ExperimentLogger
from vocode.metrics.registry import MetricSelection
from vocode.metrics.rtf import RealTimeFactorConfig, RealTimeFactorMonitor
from vocode.metrics.sequence import MetricSequence
from vocode.models.registry import ArchitectureModuleSpec
from vocode.models.vocoder import ArchitectureName
from vocode.trainers.reproduction import ReproductionTrainingRunner


class StubArchitectureModule(Module):
    # Minimal harness Module standing in for a registry-built architecture, so
    # runner policy can be exercised without constructing a reference vocoder.
    def __init__(self) -> None:
        # Opens the harness Module without registering any submodule.
        super().__init__()


class TrainerEntryPointInterception:
    # Scoped replacement of one harness Trainer entry point with a recording
    # stand-in, capturing the Trainer instances a runner builds while keeping
    # every harness loop unexecuted.
    #
    # Integration: this is what makes the runner's Trainer assembly observable
    # without a corpus. The runner is driven for real, so it builds its callbacks
    # and its Trainer exactly as it would in production, and only the final loop
    # invocation is replaced. The captured Trainer is then inspected for the
    # settings the assembly chose, and the captured keyword arguments for what
    # the runner passed alongside the module, which is how the resume path is
    # told apart from a weight load. The replacement is made on the Trainer class
    # itself and is therefore process-wide while the block is open, so the
    # original is restored in the exit path on both the success and the failure
    # branch.
    def __init__(self, entry_point_name: str) -> None:
        # Retains the original entry point and opens the recording lists.
        # The original is captured at construction rather than at entry, so a
        # nested or repeated use restores what was actually replaced.
        #
        # Args:
        #     entry_point_name: The harness entry point to intercept, which is
        #         one of the fit, validate, test, and predict methods.
        self._entry_point_name: str = entry_point_name
        self._original_entry_point: Callable[..., object] = getattr(Trainer, entry_point_name)
        self._recorded_trainers: list[Trainer] = []
        self._recorded_keyword_arguments: list[dict[str, object]] = []

    @property
    def recorded_trainer(self) -> Trainer:
        # Returns the Trainer captured by the first recorded invocation.
        # A case that reads this without an invocation having happened fails on
        # the empty list, which is the correct outcome: it means the assembly
        # under test never reached its entry point.
        return self._recorded_trainers[0]

    @property
    def recorded_trainer_count(self) -> int:
        # Returns how many invocations were recorded. Asserting the count is
        # what proves a dispatch method built one Trainer rather than several.
        return len(self._recorded_trainers)

    @property
    def recorded_keyword_arguments(self) -> dict[str, object]:
        # Returns a copy of the keyword arguments of the first invocation.
        # The copy keeps a case from mutating the recording it has read.
        return dict(self._recorded_keyword_arguments[0])

    def __enter__(self) -> TrainerEntryPointInterception:
        # Installs the recording stand-in in place of the harness entry point.
        # The recording lists are bound into the closure rather than reached
        # through the instance, because the stand-in is installed on the class
        # and receives the calling Trainer as its first argument, not this
        # interception.
        recorded_trainers: list[Trainer] = self._recorded_trainers
        recorded_keyword_arguments: list[dict[str, object]] = self._recorded_keyword_arguments

        def record_invocation(
            trainer: Trainer,
            *invocation_arguments: object,
            **keyword_arguments: object
        ) -> None:
            # Records the calling Trainer together with its keyword arguments.
            del invocation_arguments
            recorded_trainers.append(trainer)
            recorded_keyword_arguments.append(dict(keyword_arguments))

        setattr(Trainer, self._entry_point_name, record_invocation)
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback_object: TracebackType | None
    ) -> None:
        # Restores the original entry point on both the success and failure path.
        del exception_type, exception_value, traceback_object
        setattr(Trainer, self._entry_point_name, self._original_entry_point)


class RunConfigurationFactory:
    # Builds valid ExperimentConfiguration records rooted in one temporary
    # artifact tree, with distinct batch limits per stage so forwarding can be
    # told apart from cross-wiring.
    #
    # Integration: the batch limits are deliberately all different from one
    # another, so an assertion that a stage's Trainer carries the right limit
    # proves the runner forwarded that stage's own value rather than one that
    # merely happens to match. Every configuration is the real validated record
    # and every path descends from the per-case temporary root, so no case reads
    # the corpus, the author weights, or the real artifact tree.
    def __init__(self, temporary_root: Path) -> None:
        # Binds the temporary root every built configuration is anchored in.
        #
        # Args:
        #     temporary_root: The per-case temporary directory all built paths
        #         descend from.
        self._temporary_root: Path = temporary_root

    def build_training_run(self, architecture_name: ArchitectureName) -> ExperimentConfiguration:
        # Builds a full-precision training run description.
        return self._compose(architecture_name, "project_trained_reproduction", "train", "fp32")

    def build_training_run_with_precision(
        self,
        architecture_name: ArchitectureName,
        precision_name: PrecisionName
    ) -> ExperimentConfiguration:
        # Builds a training run description on the requested precision label.
        return self._compose(
            architecture_name,
            "project_trained_reproduction",
            "train",
            precision_name
        )

    def build_training_run_without_runtime_profiling(
        self,
        architecture_name: ArchitectureName
    ) -> ExperimentConfiguration:
        # Builds a training run description with runtime telemetry switched off.
        return self._compose(
            architecture_name,
            "project_trained_reproduction",
            "train",
            "fp32",
            runtime_profiling_enabled=False
        )

    def build_smoke_training_run(
        self,
        architecture_name: ArchitectureName
    ) -> ExperimentConfiguration:
        # Builds a training run description whose epoch is truncated to two batches.
        # Two batches is far below any registered step interval, which is what
        # makes this the configuration that exposes the cadence fallback: under a
        # step cadence such a run would never validate or checkpoint at all.
        return self._compose(
            architecture_name,
            "project_trained_reproduction",
            "train",
            "fp32",
            limit_train_batches=2
        )

    def build_evaluation_run(
        self,
        architecture_name: ArchitectureName,
        stage: ExperimentStage
    ) -> ExperimentConfiguration:
        # Builds an evaluation run description on the requested stage.
        return self._compose(architecture_name, "project_trained_reproduction", stage, "fp32")

    def build_evaluation_run_without_real_time_factor(
        self,
        architecture_name: ArchitectureName
    ) -> ExperimentConfiguration:
        # Builds an evaluation run description whose panel omits the timing metric.
        return self._compose(
            architecture_name,
            "project_trained_reproduction",
            "test",
            "fp32",
            metric_selection=MetricSelection(names=("pesq", "mel"))
        )

    def build_evaluation_run_with_checkpoint(
        self,
        architecture_name: ArchitectureName,
        project_checkpoint_path: Path
    ) -> ExperimentConfiguration:
        # Builds an evaluation run description citing the given project checkpoint.
        return self._compose(
            architecture_name,
            "project_trained_reproduction",
            "validation",
            "fp32",
            project_checkpoint_path=project_checkpoint_path
        )

    def build_run_for_evidence_category(
        self,
        evidence_category: EvidenceCategory
    ) -> ExperimentConfiguration:
        # Builds a run description under the requested evidence category.
        return self._compose("hifigan_v1", evidence_category, "test", "fp32")

    def _compose(
        self,
        architecture_name: ArchitectureName,
        evidence_category: EvidenceCategory,
        stage: ExperimentStage,
        precision_name: PrecisionName,
        project_checkpoint_path: Path | None = None,
        limit_train_batches: int | None = None,
        runtime_profiling_enabled: bool = True,
        metric_selection: MetricSelection | None = None
    ) -> ExperimentConfiguration:
        # Composes the full run description from the layout and the varied fields.
        # Everything the cases do not vary is fixed here, so two configurations
        # differ only in what the case under test is about.
        #
        # Args:
        #     architecture_name: The architecture the run declares, which
        #         selects the family policy under test.
        #     evidence_category: The evidence family, which decides the
        #         checkpoint evidence label.
        #     stage: The stage the run declares.
        #     precision_name: The artifact precision label whose mapping onto
        #         the harness domain is under test.
        #     project_checkpoint_path: The checkpoint the run cites, absent by
        #         default so the missing-checkpoint refusal is reachable.
        #         Default: ``None``.
        #     limit_train_batches: Training batch limit, absent by default so
        #         the step cadence is reachable. Default: ``None``.
        #     runtime_profiling_enabled: Whether the telemetry callback is
        #         built. Default: ``True``.
        #     metric_selection: The metric panel, defaulting to the full one
        #         so the timing monitor is present unless a case removes it.
        #         Default: ``None``.
        #
        # Returns:
        #     A validated run description anchored in the temporary root.
        layout: ExperimentArtifactLayout = ExperimentArtifactLayout(
            artifact_root=self._temporary_root / "artifacts",
            evidence_category=evidence_category,
            architecture_name=architecture_name,
            hardware_name="cpu",
            precision_name=precision_name,
            seed=13,
            run_id="run_0001"
        )
        resolved_metric_selection: MetricSelection = (
            metric_selection if metric_selection is not None else MetricSelection()
        )
        return ExperimentConfiguration(
            experiment_name="vocode_reproduction",
            run_id="run_0001",
            hypothesis="Reference recipes reproduce within the reported spread.",
            interpretation_notes="Synthetic configuration used for wiring verification.",
            evidence_category=evidence_category,
            stage=stage,
            dataset_split_name=stage,
            architecture_name=architecture_name,
            seed=13,
            artifact_layout=layout,
            published_weights_root=self._temporary_root / "published_weights",
            project_checkpoint_path=project_checkpoint_path,
            data_configuration=LJSpeechDataConfig(dataset_root=self._temporary_root / "corpus"),
            metric_selection=resolved_metric_selection,
            real_time_factor_configuration=RealTimeFactorConfig(),
            train_epoch_count=3,
            limit_train_batches=limit_train_batches,
            limit_val_batches=2,
            limit_test_batches=3,
            limit_predict_batches=4,
            accelerator="cpu",
            runtime_profiling_enabled=runtime_profiling_enabled,
            runtime_profile_interval_steps=25
        )


class ReproductionRunnerConstructionTest(unittest.TestCase):
    # Verifies that construction binds the configuration and opens per-runner
    # state without producing any evidence row.
    def setUp(self) -> None:
        # Binds one runner over a full-precision training configuration.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._factory: RunConfigurationFactory = RunConfigurationFactory(
            Path(self._temporary_directory.name)
        )
        self._configuration: ExperimentConfiguration = self._factory.build_training_run("hifigan_v1")
        self._runner: ReproductionTrainingRunner = ReproductionTrainingRunner(self._configuration)

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_construction_writes_no_rows(self) -> None:
        # A freshly constructed runner has produced no experiment rows.
        self.assertEqual(self._runner.row_count, 0, msg="Construction must not write evidence rows")

    def test_construction_creates_no_artifact_directories(self) -> None:
        # Construction alone leaves the artifact tree untouched.
        self.assertFalse(
            self._configuration.run_directory.exists(),
            msg="Construction must not create the run capsule directory"
        )

    def test_each_runner_owns_its_row_counter(self) -> None:
        # Two runners over the same configuration keep independent counters.
        other_runner: ReproductionTrainingRunner = ReproductionTrainingRunner(self._configuration)
        self.assertEqual(other_runner.row_count, self._runner.row_count)
        self.assertIsNot(other_runner, self._runner)


class ReproductionFamilyPolicyTest(unittest.TestCase):
    # Verifies the family partition that selects the step-cadence training
    # policy and the per-family validation interval.
    def setUp(self) -> None:
        # Binds one runner whose family predicates are queried per architecture.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._factory: RunConfigurationFactory = RunConfigurationFactory(
            Path(self._temporary_directory.name)
        )
        self._runner: ReproductionTrainingRunner = ReproductionTrainingRunner(
            self._factory.build_training_run("hifigan_v1")
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_adversarial_architectures_form_the_gan_family(self) -> None:
        # Exactly the ten adversarial vocoders are recognized as the GAN family.
        adversarial_names: tuple[ArchitectureName, ...] = (
            "hifigan_v1",
            "hifigan_v2",
            "hifigan_v3",
            "melgan",
            "vocos",
            "bigvgan",
            "apnet2",
            "freev",
            "rndvoc",
            "vocosformer"
        )
        architecture_name: ArchitectureName
        for architecture_name in adversarial_names:
            self.assertTrue(
                self._runner._is_gan_vocoder(architecture_name),
                msg=f"{architecture_name} belongs to the adversarial family"
            )

    def test_non_adversarial_architectures_are_outside_the_gan_family(self) -> None:
        # The autoregressive, flow, and excluded architectures are not adversarial.
        non_adversarial_names: tuple[ArchitectureName, ...] = ("lpcnet", "rfwave", "hiftnet")
        architecture_name: ArchitectureName
        for architecture_name in non_adversarial_names:
            self.assertFalse(
                self._runner._is_gan_vocoder(architecture_name),
                msg=f"{architecture_name} is not an adversarial vocoder"
            )

    def test_step_cadence_covers_the_gan_family_plus_lpcnet_and_rfwave(self) -> None:
        # LPCNet and RFWave join the adversarial family on the lean step cadence.
        step_cadence_names: tuple[ArchitectureName, ...] = (
            "hifigan_v1",
            "melgan",
            "vocos",
            "bigvgan",
            "apnet2",
            "freev",
            "rndvoc",
            "vocosformer",
            "lpcnet",
            "rfwave"
        )
        architecture_name: ArchitectureName
        for architecture_name in step_cadence_names:
            self.assertTrue(
                self._runner._uses_step_cadence_policy(architecture_name),
                msg=f"{architecture_name} trains on the step-cadence policy"
            )

    def test_hiftnet_keeps_the_epoch_cadence_policy(self) -> None:
        # HiFTNet is the single architecture left on the epoch-cadence policy.
        self.assertFalse(self._runner._uses_step_cadence_policy("hiftnet"))

    def test_validation_interval_is_five_thousand_steps_for_lpcnet_and_rfwave(self) -> None:
        # The costly validation passes validate on the checkpoint cadence.
        self.assertEqual(self._runner._resolve_validation_step_interval("lpcnet"), 5000)
        self.assertEqual(self._runner._resolve_validation_step_interval("rfwave"), 5000)

    def test_validation_interval_is_one_thousand_steps_for_the_gan_family(self) -> None:
        # Adversarial vocoders keep the fleet-wide one-thousand-step cadence.
        self.assertEqual(self._runner._resolve_validation_step_interval("hifigan_v1"), 1000)
        self.assertEqual(self._runner._resolve_validation_step_interval("bigvgan"), 1000)


class ReproductionPrecisionMappingTest(unittest.TestCase):
    # Verifies that artifact precision labels map onto the harness precision
    # domain and that the mapped values are accepted by a harness Trainer.
    def setUp(self) -> None:
        # Opens the factory each precision label draws its configuration from.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._factory: RunConfigurationFactory = RunConfigurationFactory(
            Path(self._temporary_directory.name)
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_float32_label_maps_to_full_precision(self) -> None:
        # The fp32 artifact label becomes the harness 32-true mode.
        runner: ReproductionTrainingRunner = ReproductionTrainingRunner(
            self._factory.build_training_run_with_precision("hifigan_v1", "fp32")
        )
        self.assertEqual(runner._resolve_trainer_precision(), "32-true")

    def test_float16_label_maps_to_mixed_precision(self) -> None:
        # The fp16 artifact label becomes the harness 16-mixed mode.
        runner: ReproductionTrainingRunner = ReproductionTrainingRunner(
            self._factory.build_training_run_with_precision("hifigan_v1", "fp16")
        )
        self.assertEqual(runner._resolve_trainer_precision(), "16-mixed")

    def test_bfloat16_label_maps_to_brain_float_mixed_precision(self) -> None:
        # The bf16 artifact label becomes the harness bf16-mixed mode.
        runner: ReproductionTrainingRunner = ReproductionTrainingRunner(
            self._factory.build_training_run_with_precision("hifigan_v1", "bf16")
        )
        self.assertEqual(runner._resolve_trainer_precision(), "bf16-mixed")

    def test_mapped_precision_is_accepted_by_the_harness_trainer(self) -> None:
        # Every mapped label constructs a Trainer carrying that precision.
        # The three preceding cases prove the mapping produces the intended
        # literal; this one proves the harness accepts it, which is what catches
        # a literal that is correct in this file but absent from the harness
        # precision domain.
        precision_labels: tuple[PrecisionName, ...] = ("fp32", "fp16", "bf16")
        precision_label: PrecisionName
        for precision_label in precision_labels:
            runner: ReproductionTrainingRunner = ReproductionTrainingRunner(
                self._factory.build_training_run_with_precision("hifigan_v1", precision_label)
            )
            resolved_precision: Precision = runner._resolve_trainer_precision()
            trainer: Trainer = Trainer(
                max_epochs=1,
                accelerator="cpu",
                precision=resolved_precision,
                enable_progress_bar=False
            )
            self.assertEqual(
                trainer.precision,
                resolved_precision,
                msg=f"Trainer must carry the precision mapped from {precision_label}"
            )


class ReproductionCallbackPolicyTest(unittest.TestCase):
    # Verifies the training callback policy per family and the optional
    # runtime-telemetry callback.
    def setUp(self) -> None:
        # Binds a runner and the monitored checkpoint handed into every callback list.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._factory: RunConfigurationFactory = RunConfigurationFactory(
            Path(self._temporary_directory.name)
        )
        self._configuration: ExperimentConfiguration = self._factory.build_training_run("hifigan_v1")
        self._runner: ReproductionTrainingRunner = ReproductionTrainingRunner(self._configuration)
        self._model_checkpoint: ModelCheckpoint = ModelCheckpoint(
            dirpath=self._configuration.checkpoints_directory,
            monitor="val_loss",
            mode="min"
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_step_cadence_family_receives_no_early_stopping_and_no_weight_averaging(self) -> None:
        # Adversarial recipes keep only the checkpoint, profiler, and guard callbacks.
        module_spec: ArchitectureModuleSpec = ArchitectureModuleSpec(
            architecture_name="hifigan_v1",
            variant_name="v1",
            module=StubArchitectureModule(),
            configuration_dump={}
        )
        callbacks: list[Callback] = self._runner._build_training_callbacks(
            module_spec,
            self._model_checkpoint
        )
        callback_types: list[type[Callback]] = [type(callback) for callback in callbacks]
        self.assertEqual(callback_types, [ModelCheckpoint, RuntimeProfiler, NanInfGuard])
        self.assertNotIn(EarlyStopping, callback_types, msg="EarlyStopping is foreign to GAN recipes")
        self.assertNotIn(EMACallback, callback_types, msg="EMA is foreign to GAN recipes")

    def test_epoch_cadence_family_receives_early_stopping_and_weight_averaging(self) -> None:
        # Epoch-cadence recipes add EarlyStopping on val_loss and EMA weight averaging.
        # Their settings are asserted as well as their presence, because a
        # patience or a decay that drifts changes the recipe as surely as
        # removing the callback would.
        module_spec: ArchitectureModuleSpec = ArchitectureModuleSpec(
            architecture_name="hiftnet",
            variant_name="yl4579_ljspeech",
            module=StubArchitectureModule(),
            configuration_dump={}
        )
        callbacks: list[Callback] = self._runner._build_training_callbacks(
            module_spec,
            self._model_checkpoint
        )
        self.assertEqual(
            [type(callback) for callback in callbacks],
            [ModelCheckpoint, EarlyStopping, EMACallback, RuntimeProfiler, NanInfGuard]
        )
        self.assertIsInstance(callbacks[1], EarlyStopping)
        self.assertIsInstance(callbacks[2], EMACallback)
        early_stopping: EarlyStopping = callbacks[1]
        exponential_moving_average: EMACallback = callbacks[2]
        self.assertEqual(early_stopping.monitor, "val_loss")
        self.assertEqual(early_stopping.mode, "min")
        self.assertEqual(early_stopping.patience, 10)
        self.assertAlmostEqual(exponential_moving_average.decay, 0.999)

    def test_checkpoint_callback_is_first_in_every_family(self) -> None:
        # The monitored checkpoint leads the callback list for both policies.
        family_names: tuple[ArchitectureName, ...] = ("hifigan_v1", "hiftnet")
        architecture_name: ArchitectureName
        for architecture_name in family_names:
            module_spec: ArchitectureModuleSpec = ArchitectureModuleSpec(
                architecture_name=architecture_name,
                variant_name="reference",
                module=StubArchitectureModule(),
                configuration_dump={}
            )
            callbacks: list[Callback] = self._runner._build_training_callbacks(
                module_spec,
                self._model_checkpoint
            )
            self.assertIs(
                callbacks[0],
                self._model_checkpoint,
                msg=f"{architecture_name} must place the checkpoint callback first"
            )

    def test_runtime_profiler_carries_the_configured_interval(self) -> None:
        # The profiler is built once with the run's profiling interval.
        profiler_callbacks: list[Callback] = self._runner._build_runtime_profiler_callbacks()
        self.assertEqual(len(profiler_callbacks), 1)
        self.assertIsInstance(profiler_callbacks[0], RuntimeProfiler)
        profiler: RuntimeProfiler = profiler_callbacks[0]
        self.assertEqual(
            profiler._profile_every_n_steps,
            self._configuration.runtime_profile_interval_steps
        )

    def test_disabled_profiling_yields_no_telemetry_callback(self) -> None:
        # Runs with profiling disabled build an empty telemetry list.
        runner: ReproductionTrainingRunner = ReproductionTrainingRunner(
            self._factory.build_training_run_without_runtime_profiling("hifigan_v1")
        )
        self.assertEqual(runner._build_runtime_profiler_callbacks(), [])

    def test_disabled_profiling_removes_the_profiler_from_training_callbacks(self) -> None:
        # The training callback list drops the profiler when telemetry is off.
        runner: ReproductionTrainingRunner = ReproductionTrainingRunner(
            self._factory.build_training_run_without_runtime_profiling("hifigan_v1")
        )
        module_spec: ArchitectureModuleSpec = ArchitectureModuleSpec(
            architecture_name="hifigan_v1",
            variant_name="v1",
            module=StubArchitectureModule(),
            configuration_dump={}
        )
        callbacks: list[Callback] = runner._build_training_callbacks(
            module_spec,
            self._model_checkpoint
        )
        self.assertEqual([type(callback) for callback in callbacks], [ModelCheckpoint, NanInfGuard])


class ReproductionTrainingWiringTest(unittest.TestCase):
    # Verifies the Trainer, checkpoint boundaries, and cadence the training
    # assembly produces, with the harness fit entry point intercepted.
    def setUp(self) -> None:
        # Opens the factory each wiring case draws its configuration from.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._factory: RunConfigurationFactory = RunConfigurationFactory(
            Path(self._temporary_directory.name)
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_step_cadence_training_builds_a_global_step_validation_trainer(self) -> None:
        # Adversarial training validates on the global-step interval with no epoch-end pass.
        configuration: ExperimentConfiguration = self._factory.build_training_run("hifigan_v1")
        trainer: Trainer = self._assemble_training_trainer(configuration, "hifigan_v1", "v1")
        self.assertEqual(trainer.max_epochs, configuration.train_epoch_count)
        self.assertEqual(trainer.precision, "32-true")
        self.assertEqual(trainer.fit_loop.epoch_loop.val_check_interval, 1000)
        self.assertEqual(trainer.fit_loop.epoch_loop.val_check_interval_scope, "global_step")
        self.assertFalse(trainer.fit_loop.epoch_loop.validate_at_epoch_end)

    def test_step_cadence_training_checkpoints_every_five_thousand_steps(self) -> None:
        # The checkpoint boundary for step-cadence families is the five-thousand-step cadence.
        configuration: ExperimentConfiguration = self._factory.build_training_run("hifigan_v1")
        trainer: Trainer = self._assemble_training_trainer(configuration, "hifigan_v1", "v1")
        self.assertIsInstance(trainer.callbacks[0], ModelCheckpoint)
        model_checkpoint: ModelCheckpoint = trainer.callbacks[0]
        self.assertEqual(model_checkpoint.every_n_train_steps, 5000)
        self.assertEqual(model_checkpoint.dirpath, configuration.checkpoints_directory)
        self.assertEqual(model_checkpoint.monitor, "val_loss")
        self.assertEqual(model_checkpoint.mode, "min")
        self.assertEqual(model_checkpoint.save_top_k, 1)
        self.assertTrue(model_checkpoint.save_last)
        self.assertTrue(model_checkpoint.save_on_exception)

    def test_epoch_cadence_training_keeps_the_harness_validation_defaults(self) -> None:
        # Epoch-cadence training validates once per epoch with no step interval.
        configuration: ExperimentConfiguration = self._factory.build_training_run("hiftnet")
        trainer: Trainer = self._assemble_training_trainer(
            configuration,
            "hiftnet",
            "yl4579_ljspeech"
        )
        self.assertAlmostEqual(trainer.fit_loop.epoch_loop.val_check_interval, 1.0)
        self.assertEqual(trainer.fit_loop.epoch_loop.val_check_interval_scope, "epoch")
        self.assertTrue(trainer.fit_loop.epoch_loop.validate_at_epoch_end)
        self.assertIsInstance(trainer.callbacks[0], ModelCheckpoint)
        model_checkpoint: ModelCheckpoint = trainer.callbacks[0]
        self.assertIsNone(model_checkpoint.every_n_train_steps)

    def test_adversarial_training_logs_on_the_dense_five_step_cadence(self) -> None:
        # Adversarial loss components are logged every five optimizer steps.
        configuration: ExperimentConfiguration = self._factory.build_training_run("hifigan_v1")
        trainer: Trainer = self._assemble_training_trainer(configuration, "hifigan_v1", "v1")
        self.assertEqual(trainer.log_every_n_steps, 5)

    def test_non_adversarial_training_logs_on_the_ten_step_cadence(self) -> None:
        # Non-adversarial families keep the ten-step logging cadence.
        configuration: ExperimentConfiguration = self._factory.build_training_run("rfwave")
        trainer: Trainer = self._assemble_training_trainer(
            configuration,
            "rfwave",
            "bfs18_24khz_ljspeech"
        )
        self.assertEqual(trainer.log_every_n_steps, 10)

    def test_flow_architecture_validates_on_the_checkpoint_cadence(self) -> None:
        # RFWave validates every five thousand global steps.
        configuration: ExperimentConfiguration = self._factory.build_training_run("rfwave")
        trainer: Trainer = self._assemble_training_trainer(
            configuration,
            "rfwave",
            "bfs18_24khz_ljspeech"
        )
        self.assertEqual(trainer.fit_loop.epoch_loop.val_check_interval, 5000)
        self.assertEqual(trainer.fit_loop.epoch_loop.val_check_interval_scope, "global_step")

    def test_train_batch_limit_falls_back_to_epoch_cadence(self) -> None:
        # A truncated epoch would never reach a step interval, so cadence reverts.
        configuration: ExperimentConfiguration = self._factory.build_smoke_training_run("hifigan_v1")
        trainer: Trainer = self._assemble_training_trainer(configuration, "hifigan_v1", "v1")
        self.assertAlmostEqual(trainer.fit_loop.epoch_loop.val_check_interval, 1.0)
        self.assertEqual(trainer.fit_loop.epoch_loop.val_check_interval_scope, "epoch")
        self.assertTrue(trainer.fit_loop.epoch_loop.validate_at_epoch_end)
        self.assertIsInstance(trainer.callbacks[0], ModelCheckpoint)
        model_checkpoint: ModelCheckpoint = trainer.callbacks[0]
        self.assertIsNone(model_checkpoint.every_n_train_steps)
        self.assertEqual(trainer.fit_loop.epoch_loop.limit_train_batches, 2)

    def test_training_forwards_the_resume_checkpoint_path(self) -> None:
        # The configured project checkpoint enters fit as the resume path.
        checkpoint_path: Path = Path(self._temporary_directory.name) / "resume.ckpt"
        torch.save({"model_state_dict": {}}, checkpoint_path)
        configuration: ExperimentConfiguration = self._factory.build_evaluation_run_with_checkpoint(
            "hifigan_v1",
            checkpoint_path
        )
        module_spec: ArchitectureModuleSpec = self._build_module_spec("hifigan_v1", "v1")
        runner: ReproductionTrainingRunner = ReproductionTrainingRunner(configuration)
        experiment_logger: ExperimentLogger = runner._build_experiment_logger(module_spec)
        datamodule: LJSpeechDataModule = LJSpeechDataModule(configuration.data_configuration)
        with TrainerEntryPointInterception("fit") as interception:
            runner._run_training(module_spec, datamodule, experiment_logger)
        self.assertEqual(interception.recorded_trainer_count, 1)
        self.assertEqual(interception.recorded_keyword_arguments["ckpt_path"], checkpoint_path)

    def test_training_writes_the_checkpoint_manifest(self) -> None:
        # The manifest records the monitored selection criterion of the run.
        configuration: ExperimentConfiguration = self._factory.build_training_run("hifigan_v1")
        self._assemble_training_trainer(configuration, "hifigan_v1", "v1")
        manifest_path: Path = (
            configuration.run_directory / "checkpoints" / "project_checkpoint_manifest.json"
        )
        self.assertTrue(manifest_path.exists(), msg="Training must write the checkpoint manifest")
        manifest_payload: dict[str, object] = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest_payload["evidence_label"], "project_trained_reproduction")
        self.assertEqual(manifest_payload["monitor"], "val_loss")
        self.assertEqual(manifest_payload["mode"], "min")
        self.assertEqual(manifest_payload["variant_name"], "v1")
        self.assertEqual(manifest_payload["seed"], configuration.seed)

    def _build_module_spec(
        self,
        architecture_name: ArchitectureName,
        variant_name: str
    ) -> ArchitectureModuleSpec:
        # Builds the module spec the training assembly is driven with.
        return ArchitectureModuleSpec(
            architecture_name=architecture_name,
            variant_name=variant_name,
            module=StubArchitectureModule(),
            configuration_dump={"learning_rate": 0.0002}
        )

    def _assemble_training_trainer(
        self,
        configuration: ExperimentConfiguration,
        architecture_name: ArchitectureName,
        variant_name: str
    ) -> Trainer:
        # Runs the training assembly with fit intercepted and returns the built Trainer.
        # The whole assembly runs for real, including the checkpoint manifest
        # write that follows fit, so a case may assert either the Trainer this
        # returns or the artifacts the assembly left behind.
        #
        # Args:
        #     configuration: The run description the assembly is driven with.
        #     architecture_name: The architecture the module spec declares.
        #     variant_name: The variant the module spec declares.
        #
        # Returns:
        #     The Trainer the runner built and would have called fit on.
        runner: ReproductionTrainingRunner = ReproductionTrainingRunner(configuration)
        module_spec: ArchitectureModuleSpec = self._build_module_spec(architecture_name, variant_name)
        experiment_logger: ExperimentLogger = runner._build_experiment_logger(module_spec)
        datamodule: LJSpeechDataModule = LJSpeechDataModule(configuration.data_configuration)
        with TrainerEntryPointInterception("fit") as interception:
            runner._run_training(module_spec, datamodule, experiment_logger)
        return interception.recorded_trainer


class ReproductionEvaluationDispatchWiringTest(unittest.TestCase):
    # Verifies the single-pass Trainers and callback ordering of the
    # validation, prediction, and test dispatch methods.
    def setUp(self) -> None:
        # Binds a runner with the module, logger, and datamodule the dispatch
        # methods are called with.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._factory: RunConfigurationFactory = RunConfigurationFactory(
            Path(self._temporary_directory.name)
        )
        self._configuration: ExperimentConfiguration = self._factory.build_evaluation_run(
            "hifigan_v1",
            "test"
        )
        self._runner: ReproductionTrainingRunner = ReproductionTrainingRunner(self._configuration)
        self._module: StubArchitectureModule = StubArchitectureModule()
        self._module_spec: ArchitectureModuleSpec = ArchitectureModuleSpec(
            architecture_name="hifigan_v1",
            variant_name="v1",
            module=self._module,
            configuration_dump={}
        )
        self._experiment_logger: ExperimentLogger = self._runner._build_experiment_logger(
            self._module_spec
        )
        self._datamodule: LJSpeechDataModule = LJSpeechDataModule(
            self._configuration.data_configuration
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_validation_dispatch_builds_a_single_pass_trainer(self) -> None:
        # Validation runs one pass with the run's validation batch limit.
        with TrainerEntryPointInterception("validate") as interception:
            self._runner._dispatch_validation(
                self._module,
                self._datamodule,
                self._experiment_logger
            )
        trainer: Trainer = interception.recorded_trainer
        self.assertEqual(trainer.max_epochs, 1)
        self.assertIs(trainer.logger, self._experiment_logger)
        self.assertEqual(trainer.validate_loop.limit_batches, self._configuration.limit_val_batches)

    def test_prediction_dispatch_places_the_timing_monitor_ahead_of_the_profiler(self) -> None:
        # The real-time-factor monitor brackets the synthesis calls it times.
        with TrainerEntryPointInterception("predict") as interception:
            self._runner._dispatch_prediction(
                self._module,
                self._datamodule,
                self._experiment_logger
            )
        trainer: Trainer = interception.recorded_trainer
        self.assertIsInstance(trainer.callbacks[0], RealTimeFactorMonitor)
        self.assertIsInstance(trainer.callbacks[1], RuntimeProfiler)
        self.assertEqual(
            trainer.predict_loop.limit_batches,
            self._configuration.limit_predict_batches
        )

    def test_prediction_dispatch_omits_the_monitor_when_rtf_is_unselected(self) -> None:
        # A metric panel without rtf carries no timing monitor into prediction.
        configuration: ExperimentConfiguration = (
            self._factory.build_evaluation_run_without_real_time_factor("hifigan_v1")
        )
        runner: ReproductionTrainingRunner = ReproductionTrainingRunner(configuration)
        experiment_logger: ExperimentLogger = runner._build_experiment_logger(self._module_spec)
        with TrainerEntryPointInterception("predict") as interception:
            runner._dispatch_prediction(
                self._module,
                LJSpeechDataModule(configuration.data_configuration),
                experiment_logger
            )
        trainer: Trainer = interception.recorded_trainer
        monitors: list[Callback] = [
            callback for callback in trainer.callbacks
            if isinstance(callback, RealTimeFactorMonitor)
        ]
        self.assertEqual(monitors, [], msg="rtf must be absent from the prediction callbacks")

    def test_test_dispatch_leads_with_the_metric_sequence(self) -> None:
        # The objective metric panel is the first callback of the test pass.
        with TrainerEntryPointInterception("test") as interception:
            self._runner._dispatch_test(self._module, self._datamodule, self._experiment_logger)
        trainer: Trainer = interception.recorded_trainer
        self.assertIsInstance(trainer.callbacks[0], MetricSequence)
        self.assertIsInstance(trainer.callbacks[1], RuntimeProfiler)
        self.assertEqual(trainer.max_epochs, 1)
        self.assertEqual(trainer.test_loop.limit_batches, self._configuration.limit_test_batches)

    def test_timing_monitor_carries_the_run_timing_protocol(self) -> None:
        # The monitor is constructed from the run's real-time-factor configuration.
        with TrainerEntryPointInterception("predict") as interception:
            self._runner._dispatch_prediction(
                self._module,
                self._datamodule,
                self._experiment_logger
            )
        self.assertIsInstance(interception.recorded_trainer.callbacks[0], RealTimeFactorMonitor)
        monitor: RealTimeFactorMonitor = interception.recorded_trainer.callbacks[0]
        self.assertEqual(
            monitor.configuration,
            self._configuration.real_time_factor_configuration
        )


class ReproductionEvidenceIdentityTest(unittest.TestCase):
    # Verifies the identity fields and provenance record the runner stamps
    # onto every row it writes.
    def setUp(self) -> None:
        # Binds a runner and the module spec whose identity fields are read.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._factory: RunConfigurationFactory = RunConfigurationFactory(
            Path(self._temporary_directory.name)
        )
        self._configuration: ExperimentConfiguration = self._factory.build_training_run("hifigan_v1")
        self._runner: ReproductionTrainingRunner = ReproductionTrainingRunner(self._configuration)
        self._module_spec: ArchitectureModuleSpec = ArchitectureModuleSpec(
            architecture_name="hifigan_v1",
            variant_name="v1",
            module=StubArchitectureModule(),
            configuration_dump={"learning_rate": 0.0002}
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_unique_id_composes_architecture_category_stage_seed_and_run(self) -> None:
        # The row identifier is unique across the study by construction.
        self.assertEqual(
            self._runner._build_unique_id(self._module_spec),
            "hifigan_v1_project_trained_reproduction_train_seed13_run_0001"
        )

    def test_variant_name_carries_the_project_trained_prefix(self) -> None:
        # Project-trained rows stay distinguishable from published rows in the shared CSV.
        self.assertEqual(self._runner._build_variant_name(self._module_spec), "project_trained_v1")

    def test_hybrid_category_keeps_its_own_checkpoint_evidence_label(self) -> None:
        # Hybrid variant runs label their checkpoints as hybrid evidence.
        runner: ReproductionTrainingRunner = ReproductionTrainingRunner(
            self._factory.build_run_for_evidence_category("project_hybrid_variants")
        )
        self.assertEqual(runner._resolve_evidence_label(), "project_hybrid_variants")

    def test_other_categories_collapse_onto_the_reproduction_evidence_label(self) -> None:
        # Every remaining training-capable category writes reproduction checkpoints.
        categories: tuple[EvidenceCategory, ...] = (
            "project_trained_reproduction",
            "published_checkpoint_evaluation"
        )
        evidence_category: EvidenceCategory
        for evidence_category in categories:
            runner: ReproductionTrainingRunner = ReproductionTrainingRunner(
                self._factory.build_run_for_evidence_category(evidence_category)
            )
            self.assertEqual(
                runner._resolve_evidence_label(),
                "project_trained_reproduction",
                msg=f"{evidence_category} maps onto the reproduction checkpoint label"
            )

    def test_hyperparameter_summary_states_the_recipe_identity(self) -> None:
        # The compact summary column names architecture, variant, seed, and stage.
        summary: str = self._runner._summarize_hyperparameters(self._module_spec)
        self.assertIn("architecture=hifigan_v1", summary)
        self.assertIn("variant=v1", summary)
        self.assertIn("seed=13", summary)
        self.assertIn("stage=train", summary)
        self.assertIn("epochs=3", summary)

    def test_hyperparameter_dump_records_the_full_run_provenance(self) -> None:
        # The dump reconstructs the invocation from identity, limits, and dumps.
        dump: HyperparameterDict = self._runner._build_hyperparameter_dump(self._module_spec)
        self.assertEqual(dump["architecture_name"], "hifigan_v1")
        self.assertEqual(dump["variant_name"], "project_trained_v1")
        self.assertEqual(dump["evidence_category"], "project_trained_reproduction")
        self.assertEqual(dump["stage"], "train")
        self.assertEqual(dump["seed"], 13)
        self.assertEqual(dump["run_directory"], str(self._configuration.run_directory))
        self.assertEqual(dump["model_configuration"], self._module_spec.configuration_dump)
        self.assertEqual(dump["metrics"], list(self._configuration.metric_selection.names))
        self.assertEqual(dump["limit_val_batches"], 2)
        self.assertEqual(dump["limit_test_batches"], 3)
        self.assertEqual(dump["limit_predict_batches"], 4)

    def test_hyperparameter_dump_records_an_absent_checkpoint_as_none(self) -> None:
        # A run without a project checkpoint records the absence explicitly.
        dump: HyperparameterDict = self._runner._build_hyperparameter_dump(self._module_spec)
        self.assertIsNone(dump["project_checkpoint_path"])

    def test_hyperparameter_dump_records_a_present_checkpoint_as_text(self) -> None:
        # A configured checkpoint enters the dump as its path text.
        checkpoint_path: Path = Path(self._temporary_directory.name) / "project.ckpt"
        runner: ReproductionTrainingRunner = ReproductionTrainingRunner(
            self._factory.build_evaluation_run_with_checkpoint("hifigan_v1", checkpoint_path)
        )
        dump: HyperparameterDict = runner._build_hyperparameter_dump(self._module_spec)
        self.assertEqual(dump["project_checkpoint_path"], str(checkpoint_path))

    def test_experiment_logger_binds_the_row_identity(self) -> None:
        # The logger flushes rows under the runner's own unique identifier.
        experiment_logger: ExperimentLogger = self._runner._build_experiment_logger(
            self._module_spec
        )
        self.assertEqual(
            experiment_logger.unique_id,
            self._runner._build_unique_id(self._module_spec)
        )
        self.assertTrue((self._configuration.run_directory / "metrics").exists())


class ReproductionMetricsSnapshotTest(unittest.TestCase):
    # Verifies the per-run metrics snapshot written from the logger buffer.
    def setUp(self) -> None:
        # Binds a runner and the logger whose buffer the snapshot is written from.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._factory: RunConfigurationFactory = RunConfigurationFactory(
            Path(self._temporary_directory.name)
        )
        self._configuration: ExperimentConfiguration = self._factory.build_training_run("hifigan_v1")
        self._runner: ReproductionTrainingRunner = ReproductionTrainingRunner(self._configuration)
        self._module_spec: ArchitectureModuleSpec = ArchitectureModuleSpec(
            architecture_name="hifigan_v1",
            variant_name="v1",
            module=StubArchitectureModule(),
            configuration_dump={}
        )
        self._experiment_logger: ExperimentLogger = self._runner._build_experiment_logger(
            self._module_spec
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_snapshot_matches_the_reduced_metric_buffer(self) -> None:
        # The snapshot is exactly the logger's reduced buffer.
        self._experiment_logger.log_metrics({"val_loss": 0.25, "pesq": 3.5}, step=4)
        self._runner._write_metrics_snapshot(self._experiment_logger)
        snapshot: dict[str, float] = json.loads(
            self._configuration.metrics_path.read_text(encoding="utf-8")
        )
        self.assertEqual(snapshot, self._experiment_logger.metric_buffer)

    def test_snapshot_keys_are_sorted_for_diff_stability(self) -> None:
        # Sorted keys keep two runs of the same cell textually comparable.
        self._experiment_logger.log_metrics({"stoi": 0.9, "mel": 0.2, "pesq": 3.5}, step=1)
        self._runner._write_metrics_snapshot(self._experiment_logger)
        snapshot_text: str = self._configuration.metrics_path.read_text(encoding="utf-8")
        snapshot: dict[str, float] = json.loads(snapshot_text)
        self.assertEqual(list(snapshot.keys()), sorted(snapshot.keys()))

    def test_snapshot_is_written_inside_the_run_capsule(self) -> None:
        # The snapshot lands on the layout-owned metrics path of this run.
        self._experiment_logger.log_metrics({"pesq": 3.5}, step=0)
        self._runner._write_metrics_snapshot(self._experiment_logger)
        self.assertTrue(self._configuration.metrics_path.exists())
        self.assertEqual(self._configuration.metrics_path.name, "metrics.json")


class ReproductionValidationErrorTest(unittest.TestCase):
    # Verifies the gates that refuse a cell before it can produce evidence.
    def setUp(self) -> None:
        # Opens the factory each refused cell draws its configuration from.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._factory: RunConfigurationFactory = RunConfigurationFactory(
            Path(self._temporary_directory.name)
        )

    def tearDown(self) -> None:
        # Removes the temporary artifact root.
        self._temporary_directory.cleanup()

    def test_run_refuses_an_architecture_that_is_not_training_level_ready(self) -> None:
        # The registry status gate names the status and the recorded blocker.
        runner: ReproductionTrainingRunner = ReproductionTrainingRunner(
            self._factory.build_training_run("hiftnet")
        )
        with self.assertRaisesRegex(MisconfigurationError, "not training-level ready"):
            runner.run()

    def test_status_gate_fires_before_any_evidence_is_written(self) -> None:
        # A refused cell leaves no run capsule and no row behind.
        configuration: ExperimentConfiguration = self._factory.build_training_run("hiftnet")
        runner: ReproductionTrainingRunner = ReproductionTrainingRunner(configuration)
        with self.assertRaises(MisconfigurationError):
            runner.run()
        self.assertEqual(runner.row_count, 0)
        self.assertFalse(configuration.run_directory.exists())

    def test_evaluation_without_a_project_checkpoint_is_refused(self) -> None:
        # Evaluating a project-trained model without its weights is meaningless evidence.
        runner: ReproductionTrainingRunner = ReproductionTrainingRunner(
            self._factory.build_evaluation_run("hifigan_v1", "validation")
        )
        with self.assertRaisesRegex(MisconfigurationError, "project_checkpoint_path is required"):
            runner._load_project_checkpoint(StubArchitectureModule())

    def test_checkpoint_without_a_model_state_dictionary_is_refused(self) -> None:
        # A checkpoint missing the model_state_dict entry cannot restore weights.
        checkpoint_path: Path = Path(self._temporary_directory.name) / "incomplete.ckpt"
        torch.save({"optimizer_state_dict": {}}, checkpoint_path)
        runner: ReproductionTrainingRunner = ReproductionTrainingRunner(
            self._factory.build_evaluation_run_with_checkpoint("hifigan_v1", checkpoint_path)
        )
        with self.assertRaisesRegex(MisconfigurationError, "does not contain a model_state_dict"):
            runner._load_project_checkpoint(StubArchitectureModule())

    def test_checkpoint_with_a_model_state_dictionary_restores_the_module(self) -> None:
        # A well-formed harness checkpoint restores without raising.
        checkpoint_path: Path = Path(self._temporary_directory.name) / "complete.ckpt"
        torch.save({"model_state_dict": {}}, checkpoint_path)
        runner: ReproductionTrainingRunner = ReproductionTrainingRunner(
            self._factory.build_evaluation_run_with_checkpoint("hifigan_v1", checkpoint_path)
        )
        runner._load_project_checkpoint(StubArchitectureModule())
        self.assertEqual(runner.row_count, 0)


if __name__ == "__main__":
    unittest.main()
