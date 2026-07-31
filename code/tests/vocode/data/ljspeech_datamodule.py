# This module:
# 1. Verifies LJSpeechDataConfig: the shipped defaults of every data-pipeline
#    setting and the frozen, strict, extra-forbidding record semantics
# 2. Verifies the executed ordered-identifier holdout partition: the final
#    records by identifier order form the test split, the preceding block forms
#    validation, the remainder trains, and the partition is exhaustive,
#    disjoint, and independent of any random draw
# 3. Verifies the harness lifecycle (prepare_data, per-stage setup and teardown,
#    dataloader hooks) and the dataloader policy: per-stage batch sizes, shuffle
#    selection, seeded shuffle reproducibility, the worker-option branch, the
#    collated batch, and training-only augmentation scoping
#
# Design decisions:
# - Every test builds a ten-utterance synthetic corpus in a temporary directory,
#   so the real LJSpeech tree is never read and split arithmetic stays legible
# - Split membership is asserted through the public dataset record sequence
#   behind each stage dataloader, which is what a runner actually consumes
# - Shuffle reproducibility is asserted on the sampler index order rather than
#   by loading audio, because the seeded generator is the contract under test
# - The worker branch is constructed and inspected but never iterated: spawning
#   worker processes would dominate the runtime budget and adds no coverage of
#   this module, which only maps configuration onto DataLoader arguments
# - Training-only augmentation is asserted by comparing the training and
#   validation batches produced by one datamodule, since that is the observable
#   form of the documented scoping rule
#
# Author: Rahul Sawhney

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from pydantic import ValidationError
from scipy.io import wavfile

from syntheticmind.core.datamodule import DataModule
from syntheticmind.utilities.seed import SeedManager
from syntheticmind.utilities.types import TrainerStage

from vocode.data.ljspeech_datamodule import LJSpeechDataConfig, LJSpeechDataModule
from vocode.data.ljspeech_dataset import LJSpeechBatch, LJSpeechDataset


class SyntheticCorpusBuilder:
    # Writes a minimal corpus in the LJSpeech layout so no test reads the real tree.
    def __init__(self, root: Path, sample_rate: int) -> None:
        # Binds the corpus layout and the amplitude every generated tone peaks at.
        self._root: Path = root
        self._sample_rate: int = sample_rate
        self._wavs_directory: Path = root / "wavs"
        self._metadata_path: Path = root / "metadata.csv"
        self._amplitude: float = 0.3

    def build(self, utterance_count: int) -> tuple[str, ...]:
        # Writes numbered utterances of growing length and their metadata file.
        self._wavs_directory.mkdir(parents=True, exist_ok=True)
        identifiers: tuple[str, ...] = tuple(
            f"LJ001-{index:04d}" for index in range(1, utterance_count + 1)
        )
        index: int
        identifier: str
        for index, identifier in enumerate(identifiers):
            self._write_waveform(identifier, 1500 + index * 50)
        self._metadata_path.write_text(
            "\n".join(
                f"{identifier}|Raw text {index}|Normalized text {index}"
                for index, identifier in enumerate(identifiers)
            )
            + "\n",
            encoding="utf-8"
        )
        return identifiers

    @property
    def peak_amplitude(self) -> float:
        # Returns the peak every generated tone reaches before quantization.
        return self._amplitude

    def _write_waveform(self, identifier: str, sample_count: int) -> None:
        # Writes one int16 tone under the corpus wavs directory.
        positions: np.ndarray = np.arange(sample_count, dtype=np.float32)
        tone: np.ndarray = self._amplitude * np.sin(
            2.0 * np.pi * 220.0 * positions / self._sample_rate
        )
        wavfile.write(
            str(self._wavs_directory / f"{identifier}.wav"),
            self._sample_rate,
            (tone * 32767.0).astype(np.int16)
        )


class LJSpeechDataConfigurationTest(unittest.TestCase):
    # Verifies the frozen data-pipeline record and its validation surface.
    def setUp(self) -> None:
        # Builds the default configuration over a placeholder corpus root.
        self._dataset_root: Path = Path("corpus")
        self._configuration: LJSpeechDataConfig = LJSpeechDataConfig(dataset_root=self._dataset_root)
        self._fields: dict[str, object] = self._configuration.model_dump()

    def test_defaults_match_the_documented_pipeline_settings(self) -> None:
        # The shipped defaults define the reproduction pipeline of the study.
        self.assertEqual(self._configuration.seed, 0)
        self.assertEqual(self._configuration.training_batch_size, 16)
        self.assertEqual(self._configuration.validation_batch_size, 16)
        self.assertEqual(self._configuration.test_batch_size, 16)
        self.assertEqual(self._configuration.num_workers, 0)
        self.assertFalse(self._configuration.persistent_workers)
        self.assertIsNone(self._configuration.prefetch_factor)
        self.assertFalse(self._configuration.pin_memory)
        self.assertEqual(self._configuration.test_split_size, 525)
        self.assertEqual(self._configuration.validation_split_size, 100)
        self.assertIsNone(self._configuration.max_test_utterances)
        self.assertIsNone(self._configuration.training_segment_size)
        self.assertFalse(self._configuration.peak_normalization_enabled)
        self.assertEqual(self._configuration.peak_normalization_value, 0.95)
        self.assertIsNone(self._configuration.resample_rate)
        self.assertIsNone(self._configuration.training_random_peak_gain_range_db)

    def test_prediction_batch_size_defaults_to_one_utterance(self) -> None:
        # Synthesis timing is only meaningful on unbatched utterances.
        self.assertEqual(self._configuration.prediction_batch_size, 1)

    def test_ordered_identifier_holdout_is_the_only_registered_strategy(self) -> None:
        # The partition label is a closed vocabulary of one registered policy.
        self.assertEqual(self._configuration.partition_strategy, "ordered_identifier_holdout")
        with self.assertRaises(ValidationError):
            LJSpeechDataConfig.model_validate({**self._fields, "partition_strategy": "random_split"})

    def test_configuration_is_frozen(self) -> None:
        # A bound pipeline configuration cannot drift during a run.
        with self.assertRaises(ValidationError):
            self._configuration.seed: int = 7

    def test_configuration_rejects_unknown_field(self) -> None:
        # A forbidden extra turns a mistyped setting into a construction failure.
        with self.assertRaises(ValidationError):
            LJSpeechDataConfig.model_validate({**self._fields, "batch_size": 16})

    def test_configuration_rejects_string_dataset_root(self) -> None:
        # The corpus root crosses the boundary as a Path, never as a string.
        with self.assertRaises(ValidationError):
            LJSpeechDataConfig.model_validate({**self._fields, "dataset_root": "corpus"})

    def test_configuration_rejects_non_positive_batch_sizes(self) -> None:
        # A loader cannot be built for an empty or negative batch.
        with self.assertRaises(ValidationError):
            LJSpeechDataConfig.model_validate({**self._fields, "training_batch_size": 0})
        with self.assertRaises(ValidationError):
            LJSpeechDataConfig.model_validate({**self._fields, "prediction_batch_size": -1})

    def test_configuration_rejects_negative_worker_count(self) -> None:
        # Worker counts are non-negative; zero selects in-process loading.
        with self.assertRaises(ValidationError):
            LJSpeechDataConfig.model_validate({**self._fields, "num_workers": -1})

    def test_configuration_accepts_the_zero_worker_policy(self) -> None:
        # Zero workers is the documented default and must remain valid.
        configuration: LJSpeechDataConfig = LJSpeechDataConfig.model_validate(
            {**self._fields, "num_workers": 0}
        )
        self.assertEqual(configuration.num_workers, 0)


class LJSpeechOrderedHoldoutPartitionTest(unittest.TestCase):
    # Verifies the executed ordered-identifier holdout split contract.
    # The fixture below sizes the corpus at ten utterances and holds out two for test and
    # three for validation, leaving five to train on. Those numbers are chosen so that every
    # partition boundary can be written as a short slice of the identifier tuple and read
    # against the contract by eye, which is what makes the expectations in this class
    # independent restatements of the policy rather than echoes of the implementation.
    def setUp(self) -> None:
        # Builds a ten-utterance corpus and a datamodule with a legible split.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._builder: SyntheticCorpusBuilder = SyntheticCorpusBuilder(self._root, 22050)
        self._identifiers: tuple[str, ...] = self._builder.build(10)
        self._configuration: LJSpeechDataConfig = LJSpeechDataConfig(
            dataset_root=self._root,
            test_split_size=2,
            validation_split_size=3,
            training_batch_size=2,
            validation_batch_size=2,
            test_batch_size=2
        )
        self._datamodule: LJSpeechDataModule = LJSpeechDataModule(self._configuration)

    def tearDown(self) -> None:
        # Removes the synthetic corpus created for this test.
        self._temporary_directory.cleanup()

    def test_test_partition_holds_out_the_final_identifiers(self) -> None:
        # The last records in identifier order form the test partition.
        self._datamodule.setup("test")
        self.assertEqual(
            self._partition_identifiers(self._datamodule.test_dataloader()),
            self._identifiers[-2:]
        )

    def test_validation_partition_precedes_the_test_block(self) -> None:
        # Validation takes the block immediately before the held-out test records.
        self._datamodule.setup("validate")
        self.assertEqual(
            self._partition_identifiers(self._datamodule.val_dataloader()),
            self._identifiers[-5:-2]
        )

    def test_training_partition_takes_the_remaining_identifiers(self) -> None:
        # Training receives everything the two holdout blocks did not claim.
        self._datamodule.setup("fit")
        self.assertEqual(
            self._partition_identifiers(self._datamodule.train_dataloader()),
            self._identifiers[:-5]
        )

    def test_partitions_are_disjoint_and_exhaustive(self) -> None:
        # Every corpus record belongs to exactly one partition.
        self._datamodule.setup("fit")
        self._datamodule.setup("test")
        train_identifiers: tuple[str, ...] = self._partition_identifiers(
            self._datamodule.train_dataloader()
        )
        validation_identifiers: tuple[str, ...] = self._partition_identifiers(
            self._datamodule.val_dataloader()
        )
        test_identifiers: tuple[str, ...] = self._partition_identifiers(
            self._datamodule.test_dataloader()
        )
        combined: list[str] = list(train_identifiers + validation_identifiers + test_identifiers)
        self.assertEqual(len(combined), len(set(combined)), msg="Partitions must not overlap")
        self.assertEqual(sorted(combined), sorted(self._identifiers))

    def test_partition_membership_is_stable_across_datamodules(self) -> None:
        # Membership never depends on a random draw, so every machine agrees.
        self._datamodule.setup("test")
        other: LJSpeechDataModule = LJSpeechDataModule(
            self._configuration.model_copy(update={"seed": 999})
        )
        other.setup("test")
        self.assertEqual(
            self._partition_identifiers(self._datamodule.test_dataloader()),
            self._partition_identifiers(other.test_dataloader())
        )

    def test_prediction_reuses_the_test_partition(self) -> None:
        # Generated-audio export measures exactly the held-out test material.
        self._datamodule.setup("predict")
        self.assertEqual(
            self._partition_identifiers(self._datamodule.predict_dataloader()),
            self._identifiers[-2:]
        )

    def test_test_partition_cap_truncates_from_the_front(self) -> None:
        # Bounded verification runs cap the evaluation partition without changing its
        # ordering. Widening the holdout to four records and capping it at one isolates the
        # direction of the truncation: the surviving utterance is the first of the
        # four-record block, that is the fourth from the end of the corpus, so the cap keeps
        # the front of the partition and does not take the last utterances of the corpus.
        capped: LJSpeechDataModule = LJSpeechDataModule(
            self._configuration.model_copy(update={"test_split_size": 4, "max_test_utterances": 1})
        )
        capped.setup("test")
        self.assertEqual(
            self._partition_identifiers(capped.test_dataloader()),
            self._identifiers[-4:-3]
        )

    def test_cap_above_the_partition_size_changes_nothing(self) -> None:
        # A cap wider than the split leaves the held-out block intact.
        capped: LJSpeechDataModule = LJSpeechDataModule(
            self._configuration.model_copy(update={"max_test_utterances": 50})
        )
        capped.setup("test")
        self.assertEqual(
            self._partition_identifiers(capped.test_dataloader()),
            self._identifiers[-2:]
        )

    def test_holdout_larger_than_the_corpus_is_rejected(self) -> None:
        # A split leaving no training material is a configuration error.
        oversized: LJSpeechDataModule = LJSpeechDataModule(
            self._configuration.model_copy(update={"test_split_size": 8, "validation_split_size": 2})
        )
        with self.assertRaises(ValueError):
            oversized.setup("fit")

    def _partition_identifiers(self, loader: torch.utils.data.DataLoader) -> tuple[str, ...]:
        # Reads the identifier membership of the dataset behind a stage loader.
        dataset: LJSpeechDataset = loader.dataset
        return tuple(record.identifier for record in dataset.records)


class LJSpeechDataModuleLifecycleTest(unittest.TestCase):
    # Verifies the harness lifecycle hooks and their stage-scoped guards.
    def setUp(self) -> None:
        # Builds a ten-utterance corpus and the datamodule under test.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._builder: SyntheticCorpusBuilder = SyntheticCorpusBuilder(self._root, 22050)
        self._builder.build(10)
        self._configuration: LJSpeechDataConfig = LJSpeechDataConfig(
            dataset_root=self._root,
            test_split_size=2,
            validation_split_size=3,
            training_batch_size=2,
            validation_batch_size=2,
            test_batch_size=2
        )
        self._datamodule: LJSpeechDataModule = LJSpeechDataModule(self._configuration)

    def tearDown(self) -> None:
        # Removes the synthetic corpus created for this test.
        self._temporary_directory.cleanup()

    def test_datamodule_implements_the_harness_contract(self) -> None:
        # The trainer consumes this component through the harness base contract.
        self.assertIsInstance(self._datamodule, DataModule)
        self.assertEqual(
            repr(self._datamodule),
            "LJSpeechDataModule(train=True, val=True, test=True, predict=True)"
        )

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The datamodule exposes exactly the configuration it was built with.
        self.assertIs(self._datamodule.configuration, self._configuration)

    def test_prepare_data_validates_the_corpus_root(self) -> None:
        # Corpus validation happens once and builds no dataset of its own.
        self._datamodule.prepare_data()
        with self.assertRaises(RuntimeError):
            self._datamodule.train_dataloader()
        with self.assertRaises(RuntimeError):
            self._datamodule.test_dataloader()

    def test_prepare_data_rejects_an_absent_corpus_root(self) -> None:
        # Incomplete corpus hydration fails before a run starts.
        absent: LJSpeechDataModule = LJSpeechDataModule(
            self._configuration.model_copy(update={"dataset_root": self._root / "absent"})
        )
        with self.assertRaises(FileNotFoundError):
            absent.prepare_data()

    def test_fit_setup_builds_the_training_and_validation_datasets(self) -> None:
        # The fit stage needs both loaders and no evaluation dataset.
        self._datamodule.setup("fit")
        self.assertEqual(self._partition_size(self._datamodule.train_dataloader()), 5)
        self.assertEqual(self._partition_size(self._datamodule.val_dataloader()), 3)
        with self.assertRaises(RuntimeError):
            self._datamodule.test_dataloader()

    def test_validate_setup_builds_only_the_validation_dataset(self) -> None:
        # A standalone validation run needs no training material.
        self._datamodule.setup("validate")
        self.assertEqual(self._partition_size(self._datamodule.val_dataloader()), 3)
        with self.assertRaises(RuntimeError):
            self._datamodule.train_dataloader()

    def test_dataloaders_requested_before_setup_are_rejected(self) -> None:
        # Requesting a stage that was never set up fails with corrective guidance.
        with self.assertRaisesRegex(RuntimeError, "train_dataloader"):
            self._datamodule.train_dataloader()
        with self.assertRaisesRegex(RuntimeError, "val_dataloader"):
            self._datamodule.val_dataloader()
        with self.assertRaisesRegex(RuntimeError, "test_dataloader"):
            self._datamodule.test_dataloader()
        with self.assertRaisesRegex(RuntimeError, "predict_dataloader"):
            self._datamodule.predict_dataloader()

    def test_fit_teardown_releases_both_fit_datasets(self) -> None:
        # Stage-local datasets are released once the stage completes.
        self._datamodule.setup("fit")
        self._datamodule.teardown("fit")
        with self.assertRaises(RuntimeError):
            self._datamodule.train_dataloader()
        with self.assertRaises(RuntimeError):
            self._datamodule.val_dataloader()

    def test_test_teardown_releases_the_evaluation_dataset(self) -> None:
        # The test and prediction stages share one released dataset slot.
        self._datamodule.setup("test")
        self._datamodule.teardown("test")
        with self.assertRaises(RuntimeError):
            self._datamodule.test_dataloader()

    def test_predict_teardown_releases_the_evaluation_dataset(self) -> None:
        # Prediction releases the same slot it borrowed from the test partition.
        self._datamodule.setup("predict")
        self._datamodule.teardown("predict")
        with self.assertRaises(RuntimeError):
            self._datamodule.predict_dataloader()

    def test_unsupported_setup_stage_is_rejected(self) -> None:
        # The stage vocabulary is closed; an unknown stage cannot build datasets.
        unregistered_stage: TrainerStage = "sanity_checking"
        with self.assertRaises(ValueError):
            self._datamodule.setup(unregistered_stage)

    def test_unsupported_teardown_stage_is_rejected(self) -> None:
        # The same closed vocabulary governs resource release.
        unregistered_stage: TrainerStage = "sanity_checking"
        with self.assertRaises(ValueError):
            self._datamodule.teardown(unregistered_stage)

    def _partition_size(self, loader: torch.utils.data.DataLoader) -> int:
        # Counts the records of the dataset behind a stage dataloader.
        dataset: LJSpeechDataset = loader.dataset
        return len(dataset)


class LJSpeechDataLoaderPolicyTest(unittest.TestCase):
    # Verifies per-stage loader construction, shuffling, and seeded reproducibility.
    def setUp(self) -> None:
        # Builds a ten-utterance corpus and a datamodule with distinct batch sizes.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._builder: SyntheticCorpusBuilder = SyntheticCorpusBuilder(self._root, 22050)
        self._builder.build(10)
        self._configuration: LJSpeechDataConfig = LJSpeechDataConfig(
            dataset_root=self._root,
            seed=1234,
            test_split_size=2,
            validation_split_size=3,
            training_batch_size=2,
            validation_batch_size=3,
            test_batch_size=1
        )
        self._datamodule: LJSpeechDataModule = LJSpeechDataModule(self._configuration)

    def tearDown(self) -> None:
        # Removes the synthetic corpus created for this test.
        self._temporary_directory.cleanup()

    def test_each_stage_loader_uses_its_configured_batch_size(self) -> None:
        # Batch sizes are per stage, because evaluation and training differ.
        self._datamodule.setup("fit")
        self._datamodule.setup("test")
        self.assertEqual(self._datamodule.train_dataloader().batch_size, 2)
        self.assertEqual(self._datamodule.val_dataloader().batch_size, 3)
        self.assertEqual(self._datamodule.test_dataloader().batch_size, 1)

    def test_prediction_loader_uses_the_prediction_batch_size(self) -> None:
        # Synthesis timing requires the unbatched prediction loader.
        self._datamodule.setup("predict")
        self.assertEqual(self._datamodule.predict_dataloader().batch_size, 1)

    def test_only_the_training_loader_shuffles(self) -> None:
        # Evaluation order stays stable so measurements remain comparable.
        self._datamodule.setup("fit")
        self._datamodule.setup("test")
        self.assertIsInstance(
            self._datamodule.train_dataloader().sampler,
            torch.utils.data.RandomSampler
        )
        self.assertIsInstance(
            self._datamodule.val_dataloader().sampler,
            torch.utils.data.SequentialSampler
        )
        self.assertIsInstance(
            self._datamodule.test_dataloader().sampler,
            torch.utils.data.SequentialSampler
        )

    def test_shuffle_generator_is_seeded_from_the_configuration(self) -> None:
        # Shuffle order is a function of the configured seed alone.
        self._datamodule.setup("fit")
        loader: torch.utils.data.DataLoader = self._datamodule.train_dataloader()
        generator: torch.Generator | None = loader.generator
        self.assertIsNotNone(generator, msg="The training loader must carry a seeded generator")
        self.assertEqual(generator.initial_seed(), 1234)

    def test_repeated_training_loaders_shuffle_identically(self) -> None:
        # Two loaders from one configuration must traverse the same order.
        self._datamodule.setup("fit")
        first_order: list[int] = list(iter(self._datamodule.train_dataloader().sampler))
        second_order: list[int] = list(iter(self._datamodule.train_dataloader().sampler))
        self.assertEqual(first_order, second_order)
        self.assertEqual(sorted(first_order), list(range(5)))

    def test_differing_seeds_produce_differing_shuffle_orders(self) -> None:
        # The seed is the only control over shuffle order, so it must bite.
        self._datamodule.setup("fit")
        reseeded: LJSpeechDataModule = LJSpeechDataModule(
            self._configuration.model_copy(update={"seed": 4321})
        )
        reseeded.setup("fit")
        self.assertNotEqual(
            list(iter(self._datamodule.train_dataloader().sampler)),
            list(iter(reseeded.train_dataloader().sampler)),
            msg="Two seeds produced the same shuffle order, so the seed is not wired through"
        )

    def test_zero_worker_policy_omits_the_worker_only_options(self) -> None:
        # Torch rejects persistent workers and prefetching without workers.
        self._datamodule.setup("fit")
        loader: torch.utils.data.DataLoader = self._datamodule.train_dataloader()
        self.assertEqual(loader.num_workers, 0)
        self.assertFalse(loader.persistent_workers)

    def test_worker_policy_wires_prefetching_and_the_harness_seed_hook(self) -> None:
        # With workers the loader carries the harness worker-init seeding hook.
        worker_datamodule: LJSpeechDataModule = LJSpeechDataModule(
            self._configuration.model_copy(
                update={"num_workers": 2, "persistent_workers": True, "prefetch_factor": 2}
            )
        )
        worker_datamodule.setup("fit")
        loader: torch.utils.data.DataLoader = worker_datamodule.train_dataloader()
        self.assertEqual(loader.num_workers, 2)
        self.assertEqual(loader.prefetch_factor, 2)
        self.assertTrue(loader.persistent_workers)
        self.assertEqual(
            loader.worker_init_fn,
            SeedManager.worker_init_fn,
            msg="Worker seeding must route through the harness SeedManager hook"
        )

    def test_validation_loader_produces_the_collated_batch_dictionary(self) -> None:
        # The configured collator turns examples into the runner batch contract.
        self._datamodule.setup("validate")
        batch: LJSpeechBatch = next(iter(self._datamodule.val_dataloader()))
        self.assertEqual(
            set(batch.keys()),
            {
                "identifier",
                "audio_path",
                "raw_text",
                "normalized_text",
                "duration_seconds",
                "waveform_length",
                "sample_rate",
                "waveform"
            }
        )
        waveform: torch.Tensor = batch["waveform"]
        self.assertEqual(waveform.shape[0], 3)
        self.assertEqual(batch["sample_rate"], 22050)


class LJSpeechAugmentationScopeTest(unittest.TestCase):
    # Verifies that training-only augmentations never reach evaluation splits.
    def setUp(self) -> None:
        # Builds a corpus and a datamodule requesting cropping and random gain.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._builder: SyntheticCorpusBuilder = SyntheticCorpusBuilder(self._root, 22050)
        self._builder.build(10)
        self._segment_size: int = 512
        self._datamodule: LJSpeechDataModule = LJSpeechDataModule(
            LJSpeechDataConfig(
                dataset_root=self._root,
                test_split_size=2,
                validation_split_size=3,
                training_batch_size=2,
                validation_batch_size=2,
                training_segment_size=self._segment_size,
                training_random_peak_gain_range_db=(-6.0, -6.0)
            )
        )
        self._datamodule.setup("fit")

    def tearDown(self) -> None:
        # Removes the synthetic corpus created for this test.
        self._temporary_directory.cleanup()

    def test_training_batches_are_cropped_to_the_configured_segment(self) -> None:
        # Fixed-length crops keep training batches rectangular by construction.
        batch: LJSpeechBatch = next(iter(self._datamodule.train_dataloader()))
        waveform: torch.Tensor = batch["waveform"]
        self.assertEqual(waveform.shape, (2, self._segment_size))

    def test_training_batches_carry_the_drawn_peak_gain(self) -> None:
        # The degenerate decibel range fixes the training peak exactly. The comparison is
        # nonetheless looser than the whole-utterance one in the dataset suite, because the
        # gain normalizes the utterance and the crop is then drawn at random: the retained
        # segment need not contain the sample that reached the peak, so the batch maximum sits
        # at or below the drawn target rather than exactly on it.
        batch: LJSpeechBatch = next(iter(self._datamodule.train_dataloader()))
        waveform: torch.Tensor = batch["waveform"]
        expected_peak: float = 10.0 ** (-6.0 / 20.0)
        self.assertAlmostEqual(float(waveform.abs().max()), expected_peak, places=2)

    def test_validation_batches_keep_whole_utterances_at_source_amplitude(self) -> None:
        # Evaluation must measure the corpus, not an augmented view of it.
        batch: LJSpeechBatch = next(iter(self._datamodule.val_dataloader()))
        waveform: torch.Tensor = batch["waveform"]
        self.assertGreater(
            waveform.shape[-1],
            self._segment_size,
            msg="Validation utterances must not be cropped to the training segment"
        )
        self.assertAlmostEqual(
            float(waveform.abs().max()),
            self._builder.peak_amplitude,
            places=3,
            msg="Validation waveforms must keep the corpus amplitude"
        )


if __name__ == "__main__":
    unittest.main()
