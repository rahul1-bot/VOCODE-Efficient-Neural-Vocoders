# This module:
# 1. Defines LJSpeechDataConfig, the frozen record of every data-pipeline
#    setting: per-stage batch sizes, worker policy, deterministic split
#    sizes, and the optional preprocessing controls
# 2. Implements the LJSpeech datamodule that builds stage-appropriate
#    datasets and dataloaders from that record
#
# Harness contract (syntheticmind):
# - Subclasses the harness DataModule: prepare_data validates the corpus
#   (invoked from rank zero by the trainer), setup builds the datasets for
#   the requested stage, the four dataloader hooks each return a single
#   DataLoader, and teardown releases stage-local datasets
# - Worker seeding routes through the harness SeedManager worker-init hook
#   and a torch Generator seeded from the configuration, so shuffle order is
#   reproducible per seed
#
# Design decisions:
# - The split is an ordered-identifier holdout: the final records by
#   identifier order form the test partition, the preceding block forms
#   validation, and the remainder trains; membership never depends on a
#   random draw, so every machine derives identical partitions
# - Training-only augmentations (segment cropping, random peak gain) are
#   bound at dataset construction and never applied to evaluation splits
# - The prediction batch size defaults to one because synthesis timing
#   requires unbatched utterances
#
# Author: Rahul Sawhney

from pathlib import Path
from typing import ClassVar, Literal, override

import torch
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveFloat, PositiveInt

from syntheticmind.core.datamodule import DataModule
from syntheticmind.utilities.seed import SeedManager
from syntheticmind.utilities.types import TrainerStage

from vocode.data.ljspeech_dataset import LJSpeechBatchCollator, LJSpeechDataset, LJSpeechRecord, LJSpeechSource

__all__: list[str] = ["LJSpeechDataConfig", "LJSpeechDataModule"]


class LJSpeechDataConfig(BaseModel):
    # Frozen data-pipeline configuration.
    #
    # Fields:
    #     dataset_root: Directory containing the extracted LJSpeech corpus.
    #     seed: Seed for the dataloader shuffle generator. Default: ``0``.
    #     training_batch_size: Batch size of the training loader.
    #         Default: ``16``.
    #     validation_batch_size: Batch size of the validation loader.
    #         Default: ``16``.
    #     test_batch_size: Batch size of the test loader. Default: ``16``.
    #     prediction_batch_size: Batch size of the prediction loader; one
    #         by default because the warm real-time factor protocol times
    #         synthesis at batch one, where a batch mean is a genuine
    #         per-utterance latency. Default: ``1``.
    #     num_workers: Dataloader worker processes; zero keeps loading in
    #         the main process and drops the worker-only options.
    #         Default: ``0``.
    #     persistent_workers: Whether workers survive between epochs;
    #         applies only with workers. Default: ``False``.
    #     prefetch_factor: Batches prefetched per worker; applies only with
    #         workers. Default: ``None``.
    #     pin_memory: Whether host tensors are allocated in pinned memory
    #         for faster device transfer. Default: ``False``.
    #     test_split_size: Number of identifier-ordered records held out as
    #         the adaptive-evaluation partition. The default is the
    #         registered study size. Default: ``525``.
    #     validation_split_size: Number of records preceding the evaluation
    #         block held out for validation. The default is the registered
    #         study size. Default: ``100``.
    #     partition_strategy: Split policy label; the ordered-identifier
    #         holdout is the only registered strategy.
    #     max_test_utterances: Optional cap applied to the evaluation
    #         partition after the split, used by bounded verification
    #         runs. Default: ``None``.
    #     training_segment_size: Sample length of random training crops;
    #         ``None`` trains on whole utterances. Default: ``None``.
    #     peak_normalization_enabled: Whether waveforms are peak-normalized
    #         at load. Default: ``False``.
    #     peak_normalization_value: Target peak amplitude when
    #         normalization is enabled. Default: ``0.95``.
    #     resample_rate: Optional target sample rate applied at load;
    #         ``None`` keeps the corpus rate. Default: ``None``.
    #     training_random_peak_gain_range_db: Optional random gain range in
    #         decibels applied to training crops only. Default: ``None``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    dataset_root: Path
    seed: int = 0
    training_batch_size: PositiveInt = 16
    validation_batch_size: PositiveInt = 16
    test_batch_size: PositiveInt = 16
    prediction_batch_size: PositiveInt = 1
    num_workers: NonNegativeInt = 0
    persistent_workers: bool = False
    prefetch_factor: PositiveInt | None = None
    pin_memory: bool = False
    test_split_size: PositiveInt = 525
    validation_split_size: PositiveInt = 100
    partition_strategy: Literal["ordered_identifier_holdout"] = "ordered_identifier_holdout"
    max_test_utterances: PositiveInt | None = None
    training_segment_size: PositiveInt | None = None
    peak_normalization_enabled: bool = False
    peak_normalization_value: PositiveFloat = 0.95
    resample_rate: PositiveInt | None = None
    training_random_peak_gain_range_db: tuple[float, float] | None = None


class LJSpeechDataModule(DataModule):
    # Harness datamodule for LJSpeech. It owns deterministic split
    # construction, stage-scoped dataset lifecycles, and the dataloader
    # policy derived from the frozen configuration.
    #
    # Integration: this class is the sole implementation of the
    # ordered_identifier_holdout partition contract, and every runner
    # obtains its data through it rather than partitioning the corpus
    # itself. The contract is: take the corpus sorted by identifier, hold
    # out the final test_split_size records as the evaluation partition,
    # take the validation_split_size records immediately preceding that
    # block as validation, and train on everything before both. The
    # partition is consequently exhaustive, disjoint, and a pure function
    # of the corpus and two integers. It uses no split seed and is
    # identical on every machine; the seed governs shuffle order alone.
    # Under the registered defaults over LJSpeech 1.1 this realizes the
    # study membership of 12,475 training, 100 validation, and 525
    # adaptive-evaluation utterances, and identifier order preserves the
    # chapter structure of the corpus.
    #
    # The evaluation partition is adaptive rather than untouched: its
    # scores informed continuation decisions and one inference correction,
    # so the stage name test is a stable schema label and not a claim of
    # an untouched final test set. Training loaders shuffle and crop
    # fixed-length segments, whereas validation and evaluation consume
    # complete utterances. The prediction stage reuses the evaluation
    # partition, so exported audio and real-time-factor measurements
    # describe exactly the material the evaluation metrics were computed
    # on. A split leaving no training records is rejected rather than
    # yielding an empty training set, and max_test_utterances truncates
    # the evaluation partition from its front for bounded verification
    # runs without disturbing the ordering.
    def __init__(self, configuration: LJSpeechDataConfig) -> None:
        # Binds the configuration, opens the corpus source and collator, and
        # prepares the empty per-stage dataset slots that setup fills.
        super().__init__()
        self._configuration: LJSpeechDataConfig = configuration
        self._source: LJSpeechSource = LJSpeechSource(configuration.dataset_root)
        self._collator: LJSpeechBatchCollator = LJSpeechBatchCollator()
        self._train_dataset: LJSpeechDataset | None = None
        self._validation_dataset: LJSpeechDataset | None = None
        self._test_dataset: LJSpeechDataset | None = None

    @override
    def prepare_data(self) -> None:
        # Validates the corpus root once before any dataset is built; the
        # trainer invokes this from rank zero only.
        self._source.validate_dataset_root()

    @override
    def setup(self, stage: TrainerStage) -> None:
        # Splits the identifier-ordered records and builds the datasets the
        # requested stage needs: fit builds training (with augmentations)
        # and validation, the evaluation stages build their single split,
        # and prediction reuses the test partition.
        sorted_records: tuple[LJSpeechRecord, ...] = self._source.load_records()
        train_records, validation_records, test_records = self._split_records(sorted_records)
        match stage:
            case "fit":
                self._train_dataset: LJSpeechDataset | None = self._build_dataset(
                    train_records,
                    segment_size=self._configuration.training_segment_size,
                    random_peak_gain_range_db=self._configuration.training_random_peak_gain_range_db
                )
                self._validation_dataset: LJSpeechDataset | None = self._build_dataset(
                    validation_records,
                    segment_size=None
                )
            case "validate":
                self._validation_dataset: LJSpeechDataset | None = self._build_dataset(validation_records, segment_size=None)
            case "test":
                self._test_dataset: LJSpeechDataset | None = self._build_dataset(test_records, segment_size=None)
            case "predict":
                self._test_dataset: LJSpeechDataset | None = self._build_dataset(test_records, segment_size=None)
            case _:
                raise ValueError(f"Unsupported TrainerStage: {stage}")

    @override
    def teardown(self, stage: TrainerStage) -> None:
        # Releases stage-local data resources after trainer execution.
        match stage:
            case "fit":
                self._train_dataset: LJSpeechDataset | None = None
                self._validation_dataset: LJSpeechDataset | None = None
            case "validate":
                self._validation_dataset: LJSpeechDataset | None = None
            case "test" | "predict":
                self._test_dataset: LJSpeechDataset | None = None
            case _:
                raise ValueError(f"Unsupported TrainerStage: {stage}")

    @override
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        # Returns the dataloader used for project-trained reproduction. This is the only
        # stage loader that shuffles, and it does so through the seeded generator, so the
        # traversal order is reproducible from the configured seed alone. It raises rather
        # than building an empty loader when setup('fit') has not run.
        if self._train_dataset is None:
            raise RuntimeError(
                "train_dataloader() requested before setup('fit') initialized the training dataset."
            )
        return self._build_dataloader(
            dataset=self._train_dataset,
            batch_size=self._configuration.training_batch_size,
            shuffle=True
        )

    @override
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        # Returns the dataloader used for validation. Either setup('fit') or
        # setup('validate') provides its dataset. Evaluation never shuffles, so measurements
        # from separate runs are taken in the same utterance order and stay comparable.
        if self._validation_dataset is None:
            raise RuntimeError(
                "val_dataloader() requested before setup('fit') or setup('validate') initialized "
                "the validation dataset."
            )
        return self._build_dataloader(
            dataset=self._validation_dataset,
            batch_size=self._configuration.validation_batch_size,
            shuffle=False
        )

    @override
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        # Returns the dataloader used for evaluation over the adaptive-evaluation partition,
        # unshuffled and provided by setup('test'), consuming complete utterances.
        if self._test_dataset is None:
            raise RuntimeError(
                "test_dataloader() requested before setup('test') initialized the test dataset."
            )
        return self._build_dataloader(
            dataset=self._test_dataset,
            batch_size=self._configuration.test_batch_size,
            shuffle=False
        )

    @override
    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        # Returns the dataloader used for prediction or generated-audio export. It reads the
        # same evaluation partition under the separate prediction batch size, which defaults
        # to one because the warm real-time factor protocol times synthesis at batch one.
        if self._test_dataset is None:
            raise RuntimeError(
                "predict_dataloader() requested before setup('predict') initialized the test dataset."
            )
        return self._build_dataloader(
            dataset=self._test_dataset,
            batch_size=self._configuration.prediction_batch_size,
            shuffle=False
        )

    @property
    def configuration(self) -> LJSpeechDataConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    def _split_records(
        self,
        sorted_records: tuple[LJSpeechRecord, ...]
    ) -> tuple[tuple[LJSpeechRecord, ...], tuple[LJSpeechRecord, ...], tuple[LJSpeechRecord, ...]]:
        # Holds out the final identifier-ordered records without randomized membership.
        # The three slices are taken from one sorted sequence and are adjacent by construction,
        # which is what makes the partition exhaustive and disjoint without any membership
        # bookkeeping. The optional test cap is applied last, after the boundaries are fixed,
        # so capping shrinks the evaluated set without moving any record between partitions.
        #
        # Args:
        #     sorted_records: The corpus in identifier order, as the source
        #         guarantees it.
        #
        # Returns:
        #     The training, validation, and test partitions in that order.
        #
        # Raises:
        #     ValueError: If the configured strategy is not the registered
        #         one, or if the two holdout blocks together claim the
        #         whole corpus and leave nothing to train on.
        if self._configuration.partition_strategy != "ordered_identifier_holdout":
            raise ValueError(
                f"Unsupported LJSpeech partition strategy: "
                f"{self._configuration.partition_strategy}"
            )
        test_size: int = self._configuration.test_split_size
        validation_size: int = self._configuration.validation_split_size
        total_records: int = len(sorted_records)
        reserved_size: int = test_size + validation_size
        if reserved_size >= total_records:
            raise ValueError(
                f"test_split_size ({test_size}) + validation_split_size ({validation_size}) "
                f"must be less than total dataset size ({total_records})"
            )
        test_records: tuple[LJSpeechRecord, ...] = sorted_records[-test_size:]
        validation_records: tuple[LJSpeechRecord, ...] = sorted_records[-reserved_size:-test_size]
        train_records: tuple[LJSpeechRecord, ...] = sorted_records[:-reserved_size]
        cap: int | None = self._configuration.max_test_utterances
        if cap is not None and cap < len(test_records):
            test_records: tuple[LJSpeechRecord, ...] = test_records[:cap]
        return train_records, validation_records, test_records

    def _build_dataloader(
        self,
        dataset: LJSpeechDataset,
        batch_size: int,
        shuffle: bool
    ) -> torch.utils.data.DataLoader:
        # Builds one loader under the configured worker policy. The
        # zero-worker branch omits the worker-only options because torch
        # rejects them without workers; the worker branch adds persistent
        # workers, prefetching, and the harness worker seeding hook. Both
        # branches share the collator and the seeded shuffle generator.
        num_workers: int = self._configuration.num_workers
        generator: torch.Generator = self._build_generator()
        if num_workers < 1:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,
                collate_fn=self._collator,
                pin_memory=self._configuration.pin_memory,
                generator=generator
            )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collator,
            persistent_workers=self._configuration.persistent_workers,
            prefetch_factor=self._configuration.prefetch_factor,
            pin_memory=self._configuration.pin_memory,
            worker_init_fn=SeedManager.worker_init_fn,
            generator=generator
        )

    def _build_generator(self) -> torch.Generator:
        # Builds the shuffle generator seeded from the configuration, so the
        # training shuffle order is a function of the seed alone.
        generator: torch.Generator = torch.Generator()
        generator.manual_seed(self._configuration.seed)
        return generator

    def _build_dataset(
        self,
        records: tuple[LJSpeechRecord, ...],
        segment_size: int | None,
        random_peak_gain_range_db: tuple[float, float] | None = None
    ) -> LJSpeechDataset:
        # Builds stage-specific LJSpeech access while keeping preprocessing artifact-visible.
        return LJSpeechDataset(
            records=records,
            segment_size=segment_size,
            peak_normalization_enabled=self._configuration.peak_normalization_enabled,
            peak_normalization_value=self._configuration.peak_normalization_value,
            resample_rate=self._configuration.resample_rate,
            random_peak_gain_range_db=random_peak_gain_range_db
        )
