# This module:
# 1. Resolves and validates the LJSpeech corpus: metadata parsing into
#    frozen records, deterministic identifier ordering, and existence checks
#    at the dataset-root boundary
# 2. Loads waveforms lazily as float32 tensors in [-1, 1] with the optional
#    preprocessing chain (random peak gain, peak normalization, resampling,
#    segment cropping)
# 3. Collates variable-length examples into the padded batch dictionary
#    consumed by every runner
#
# Design decisions:
# - The dataset returns raw waveforms rather than mel features because each
#   vocoder family owns a different mel protocol and must not receive a
#   forced shared representation
# - Records are validated Pydantic models sorted by identifier, so split
#   construction and artifact rows are stable across machines
# - Waveform loading goes through scipy's WAV reader with explicit int16
#   scaling, keeping the amplitude convention identical to the published
#   reference recipes
# - Collation right-pads waveforms into a rectangular tensor while carrying
#   the true lengths and full metadata tuples alongside, so padding never
#   silently enters a measurement
#
# Author: Rahul Sawhney

import random
from pathlib import Path
from typing import ClassVar, override

import numpy as np
import torch
import torchaudio
from pydantic import BaseModel, ConfigDict, field_validator
from scipy.io import wavfile
from torch.nn.utils.rnn import pad_sequence

__all__: list[str] = [
    "LJSpeechBatch",
    "LJSpeechBatchCollator",
    "LJSpeechBatchValue",
    "LJSpeechDataset",
    "LJSpeechExample",
    "LJSpeechRecord",
    "LJSpeechSource"
]


type LJSpeechBatchValue = int | tuple[str, ...] | tuple[float, ...] | tuple[int, ...] | torch.Tensor

type LJSpeechBatch = dict[str, LJSpeechBatchValue]


class LJSpeechRecord(BaseModel):
    # Immutable metadata record for one LJSpeech utterance.
    # The record validates the corpus identifier, linked waveform path, and transcript fields
    # before any waveform tensor is created, which prevents malformed metadata from entering
    # training or evaluation batches.
    #
    # Fields:
    #     identifier: LJSpeech utterance identifier. It is the sort key
    #         that fixes split membership and the name artifact rows are
    #         attributed to, so it must be non-empty.
    #     audio_path: Location of the utterance waveform. Validation
    #         requires an existing file with a WAV suffix, so metadata can
    #         never promise audio the corpus does not hold.
    #     raw_text: Transcript as published in the corpus metadata.
    #     normalized_text: Normalized transcript. Metadata parsing splits
    #         on at most two separators, so this field may itself contain
    #         the separator character.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    identifier: str
    audio_path: Path
    raw_text: str
    normalized_text: str

    @field_validator("identifier")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        # LJSpeech identifiers are used for deterministic sorting and artifact naming.
        if not value:
            raise ValueError("identifier must not be empty")
        return value

    @field_validator("audio_path")
    @classmethod
    def validate_audio_path(cls, value: Path) -> Path:
        # The dataset layer accepts only materialized WAV files from the corpus wavs directory.
        if not value.exists():
            raise FileNotFoundError(f"Audio file does not exist: {value}")
        if value.suffix != ".wav":
            raise ValueError(f"Expected a .wav audio file, got {value}")
        return value

    @field_validator("raw_text", "normalized_text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        # Empty transcript fields indicate corrupt metadata and would weaken artifact auditability.
        if not value:
            raise ValueError("text fields must not be empty")
        return value


class LJSpeechExample(BaseModel):
    # Loaded utterance object returned by the PyTorch dataset.
    # The example carries both the validated metadata and the waveform tensor so downstream
    # runners can preserve provenance while applying architecture-specific preprocessing later.
    #
    # Fields:
    #     record: Validated metadata of this utterance, carried alongside
    #         the tensor so a measurement never loses its attribution.
    #     waveform: Loaded audio as float32 in [-1, 1] by the int16 scaling
    #         convention of the reference recipes.
    #     sample_rate: Rate the waveform is actually at, which is the
    #         resample target when resampling was applied and the corpus
    #         rate otherwise.
    #     waveform_length: True sample count of this waveform, measured
    #         after the preprocessing chain and before any batch padding,
    #         so padding can never be mistaken for signal downstream.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True, arbitrary_types_allowed=True)
    record: LJSpeechRecord
    waveform: torch.Tensor
    sample_rate: int
    waveform_length: int

    @field_validator("sample_rate", "waveform_length")
    @classmethod
    def validate_positive_integers(cls, value: int) -> int:
        # Zero-length audio or invalid sample-rate metadata is rejected before collation.
        if value < 1:
            raise ValueError(f"Expected a positive integer, got {value}")
        return value


class LJSpeechSource:
    # Corpus source responsible for resolving LJSpeech metadata and waveform locations.
    # The source caches parsed records after the first load because the metadata file is static
    # during a run, while validation remains explicit at the dataset-root boundary.
    def __init__(self, dataset_root: Path) -> None:
        # Expected LJSpeech layout: dataset root with metadata.csv and a wavs subdirectory.
        self._dataset_root: Path = dataset_root
        self._metadata_path: Path = dataset_root / "metadata.csv"
        self._wavs_path: Path = dataset_root / "wavs"
        self._records: tuple[LJSpeechRecord, ...] | None = None

    def validate_dataset_root(self) -> None:
        # Fail fast on incomplete corpus hydration before any training or evaluation job starts.
        if not self._dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self._dataset_root}")
        if not self._metadata_path.exists():
            raise FileNotFoundError(f"metadata.csv does not exist: {self._metadata_path}")
        if not self._wavs_path.exists():
            raise FileNotFoundError(f"wavs directory does not exist: {self._wavs_path}")

    def load_records(self) -> tuple[LJSpeechRecord, ...]:
        # Record ordering is deterministic so split construction and artifact rows remain stable.
        # The corpus root is validated, every non-blank metadata line becomes a validated
        # record, and the result is sorted by identifier before it is cached. Sorting here is
        # what makes the downstream ordered holdout reproducible: the partition is a function
        # of identifier order alone, never of the order lines happen to appear in the file.
        #
        # Returns:
        #     The identifier-sorted records of the corpus. The same tuple
        #     object is returned on every later call, because the metadata
        #     file is static for the lifetime of a run.
        #
        # Raises:
        #     FileNotFoundError: If the corpus root, its metadata file, its
        #         waveform directory, or any referenced waveform is absent.
        #     ValueError: If a non-blank metadata line does not carry the
        #         three expected fields, or if the file yields no record at
        #         all, since an empty corpus cannot support a run.
        if self._records is not None:
            return self._records
        self.validate_dataset_root()
        records: list[LJSpeechRecord] = []
        metadata_lines: tuple[str, ...] = self._read_metadata_lines()
        for metadata_line in metadata_lines:
            record: LJSpeechRecord = self._create_record(metadata_line)
            records.append(record)
        if not records:
            raise ValueError(f"No records were loaded from {self._metadata_path}")
        sorted_records: tuple[LJSpeechRecord, ...] = tuple(sorted(records, key=lambda candidate: candidate.identifier))
        self._records: tuple[LJSpeechRecord, ...] | None = sorted_records
        return self._records

    @property
    def dataset_root(self) -> Path:
        # Returns the validated LJSpeech dataset root used by this source.
        return self._dataset_root

    def _read_metadata_lines(self) -> tuple[str, ...]:
        # Blank metadata lines are ignored; malformed nonblank lines are rejected downstream.
        with self._metadata_path.open("r", encoding="utf-8") as metadata_file:
            return tuple(line.rstrip("\n") for line in metadata_file if line.strip())

    def _create_record(self, metadata_line: str) -> LJSpeechRecord:
        # LJSpeech metadata uses identifier, raw text, and normalized text separated by pipes.
        line_parts: list[str] = metadata_line.split("|", 2)
        if len(line_parts) != 3:
            raise ValueError(
                f"Expected three metadata fields after split('|', 2), got {len(line_parts)} "
                f"for line: {metadata_line}"
            )
        identifier: str = line_parts[0]
        raw_text: str = line_parts[1]
        normalized_text: str = line_parts[2]
        audio_path: Path = self._wavs_path / f"{identifier}.wav"
        # Duration is derived from the loaded waveform during collation because metadata.csv
        # does not provide a canonical duration field.
        return LJSpeechRecord(
            identifier=identifier,
            audio_path=audio_path,
            raw_text=raw_text,
            normalized_text=normalized_text
        )


class LJSpeechDataset(torch.utils.data.Dataset[LJSpeechExample]):
    # PyTorch dataset that loads waveform tensors lazily from validated LJSpeech records.
    # The dataset deliberately returns raw waveform rather than mel features because each vocoder
    # family owns a different mel protocol and must not receive a forced shared representation.
    #
    # Integration: this class performs no partitioning of its own. It
    # receives one already-partitioned record sequence from
    # LJSpeechDataModule, which applies the ordered_identifier_holdout
    # policy: with the corpus sorted by identifier, the final
    # test_split_size records form the adaptive-evaluation partition, the
    # block immediately before them forms validation, and everything
    # earlier trains. Membership therefore uses no split seed and is
    # identical on every machine, and index order inside this dataset is
    # the identifier order of the partition it was given. The preprocessing
    # arguments are bound here at construction, which is the mechanism by
    # which the datamodule scopes segment cropping and random peak gain to
    # the training split alone; validation and evaluation waveforms
    # receive no training-only transformation and are consumed complete.
    def __init__(
        self,
        records: tuple[LJSpeechRecord, ...],
        segment_size: int | None = None,
        peak_normalization_enabled: bool = False,
        peak_normalization_value: float = 0.95,
        resample_rate: int | None = None,
        random_peak_gain_range_db: tuple[float, float] | None = None
    ) -> None:
        # Binds the record sequence and the preprocessing policy.
        #
        # Args:
        #     records: Validated utterance records backing the dataset.
        #     segment_size: Sample length of random crops; ``None`` returns
        #         whole utterances. Default: ``None``.
        #     peak_normalization_enabled: Whether waveforms are scaled to
        #         the target peak at load. Default: ``False``.
        #     peak_normalization_value: Target peak amplitude for
        #         normalization. Default: ``0.95``.
        #     resample_rate: Optional target sample rate; resampling
        #         precedes cropping so segment sizes are expressed at the
        #         target rate. Default: ``None``.
        #     random_peak_gain_range_db: Optional decibel range for the
        #         per-utterance random peak draw applied before
        #         normalization. Default: ``None``.
        super().__init__()
        # Records are immutable Pydantic models, so the dataset can expose them safely.
        self._records: tuple[LJSpeechRecord, ...] = records
        self._segment_size: int | None = int(segment_size) if segment_size is not None else None
        self._peak_normalization_enabled: bool = peak_normalization_enabled
        self._peak_normalization_value: float = float(peak_normalization_value)
        self._resample_rate: int | None = int(resample_rate) if resample_rate is not None else None
        self._random_peak_gain_range_db: tuple[float, float] | None = random_peak_gain_range_db

    @property
    def records(self) -> tuple[LJSpeechRecord, ...]:
        # Returns the immutable record sequence backing the dataset.
        return self._records

    def __len__(self) -> int:
        # Returns the number of records exposed by this dataset contract.
        return len(self._records)

    @override
    def __getitem__(self, index: int) -> LJSpeechExample:
        # Waveforms are normalized to float32 in [-1, 1] to match the model-facing tensor contract.
        # Loading is lazy and the preprocessing chain runs in a fixed order: read the WAV file,
        # reduce any multi-channel material to mono, scale the int16 samples into [-1, 1], apply
        # the optional random peak gain, apply the optional peak normalization, resample, and
        # finally crop or pad to the segment length. The order is load-bearing in two places.
        # Peak normalization runs after the random gain, so a recipe that enables both ends at
        # the normalization target and the drawn gain does not survive. Resampling runs before
        # cropping, so a segment size is always expressed in samples at the target rate.
        #
        # Args:
        #     index: Position in this partition's identifier-ordered record
        #         sequence.
        #
        # Returns:
        #     The loaded example, whose waveform_length is measured after
        #     the whole chain, so it describes the tensor actually returned
        #     rather than the file on disk.
        record: LJSpeechRecord = self._records[index]
        loaded_sample_rate, loaded_int16 = wavfile.read(str(record.audio_path))
        if loaded_int16.ndim > 1:
            # Multi-channel files are reduced to mono so every vocoder receives one waveform track.
            loaded_int16: np.ndarray = loaded_int16.mean(axis=1)
        loaded_float32: np.ndarray = loaded_int16.astype(np.float32) / 32768.0
        if self._random_peak_gain_range_db is not None:
            loaded_float32: np.ndarray = self._apply_random_peak_gain(loaded_float32)
        if self._peak_normalization_enabled:
            loaded_float32: np.ndarray = self._normalize_peak(loaded_float32)
        waveform: torch.Tensor = torch.from_numpy(loaded_float32)
        effective_sample_rate: int = int(loaded_sample_rate)
        if self._resample_rate is not None and self._resample_rate != effective_sample_rate:
            # Resampling precedes segment extraction so segment sizes are expressed at the target rate.
            waveform: torch.Tensor = torchaudio.functional.resample(
                waveform,
                orig_freq=effective_sample_rate,
                new_freq=self._resample_rate
            )
            effective_sample_rate: int = self._resample_rate
        waveform: torch.Tensor = self._crop_or_pad_segment(waveform)
        return LJSpeechExample(
            record=record,
            waveform=waveform,
            sample_rate=effective_sample_rate,
            waveform_length=int(waveform.shape[-1])
        )

    def _normalize_peak(self, waveform: np.ndarray) -> np.ndarray:
        # Peak-normalizes audio before segment extraction when the training recipe requires it.
        maximum_absolute: float = float(np.max(np.abs(waveform))) if waveform.size > 0 else 0.0
        if maximum_absolute <= 0.0:
            return waveform
        return (waveform / maximum_absolute * self._peak_normalization_value).astype(np.float32)

    def _apply_random_peak_gain(self, waveform: np.ndarray) -> np.ndarray:
        # Draws a per-utterance random peak level so training sees amplitude-diverse material.
        if self._random_peak_gain_range_db is None:
            return waveform
        maximum_absolute: float = float(np.max(np.abs(waveform))) if waveform.size > 0 else 0.0
        if maximum_absolute <= 0.0:
            return waveform
        minimum_db, maximum_db = self._random_peak_gain_range_db
        target_peak: float = float(10.0 ** (random.uniform(minimum_db, maximum_db) / 20.0))
        return (waveform / maximum_absolute * target_peak).astype(np.float32)

    def _crop_or_pad_segment(self, waveform: torch.Tensor) -> torch.Tensor:
        # Training draws fixed-length segments while evaluation keeps full-length utterances.
        # The start offset is drawn on every access, so repeated reads of one index return
        # different crops across epochs; that resampling is the point of the augmentation.
        # An utterance shorter than the segment is right-padded with silence instead of being
        # dropped, which keeps every training batch rectangular without discarding material.
        if self._segment_size is None:
            return waveform
        if waveform.shape[-1] >= self._segment_size:
            maximum_start_index: int = waveform.shape[-1] - self._segment_size
            start_index: int = random.randint(0, maximum_start_index) if maximum_start_index > 0 else 0
            return waveform[start_index:start_index + self._segment_size]
        padding_amount: int = self._segment_size - waveform.shape[-1]
        return torch.nn.functional.pad(waveform, (0, padding_amount))


class LJSpeechBatchCollator:
    # Collates variable-length LJSpeech examples into the batch dictionary consumed by runners.
    # Metadata fields remain tuple-based for provenance, while waveform tensors are padded into
    # a rectangular batch tensor suitable for PyTorch modules and metric computation.
    def __call__(self, examples: list[LJSpeechExample]) -> LJSpeechBatch:
        # Empty batches indicate an upstream sampler or split-construction failure.
        #
        # Args:
        #     examples: Loaded utterances in sampler order, which the
        #         collation preserves so tensor rows and provenance rows
        #         describe the same utterance.
        #
        # Returns:
        #     The batch mapping every runner consumes, whose key set is a
        #     contract: ``identifier``, ``audio_path``, ``raw_text``, and
        #     ``normalized_text`` carry provenance as string tuples;
        #     ``duration_seconds`` and ``waveform_length`` carry the true
        #     per-utterance measurements taken before padding;
        #     ``sample_rate`` is a single integer because LJSpeech is a
        #     single-rate corpus; and ``waveform`` is the right-padded
        #     rectangular tensor. Durations are derived from the loaded
        #     waveform rather than read from metadata, which has no
        #     canonical duration field.
        #
        # Raises:
        #     ValueError: If the batch is empty.
        #
        # Note:
        #     Retaining waveform_length beside the padded tensor is what
        #     makes true-length evaluation possible downstream: reference
        #     and synthesized waveforms are truncated to their common
        #     unpadded length before a quality proxy is computed, so
        #     trailing padding never enters a measurement. The distinction
        #     is material rather than formal, because padded-batch and
        #     true-length quality diverge for architectures whose
        #     receptive field spans the padded region.
        if not examples:
            raise ValueError("examples must not be empty")
        identifiers: tuple[str, ...] = tuple(example.record.identifier for example in examples)
        audio_paths: tuple[str, ...] = tuple(str(example.record.audio_path) for example in examples)
        raw_texts: tuple[str, ...] = tuple(example.record.raw_text for example in examples)
        normalized_texts: tuple[str, ...] = tuple(example.record.normalized_text for example in examples)
        durations_seconds: tuple[float, ...] = tuple(
            example.waveform_length / example.sample_rate for example in examples
        )
        waveform_lengths: tuple[int, ...] = tuple(example.waveform_length for example in examples)
        # LJSpeech is a single-sample-rate corpus; the first example defines the batch rate.
        sample_rate: int = examples[0].sample_rate
        padded_waveforms: torch.Tensor = self._pad_waveforms(examples)
        return {
            "identifier": identifiers,
            "audio_path": audio_paths,
            "raw_text": raw_texts,
            "normalized_text": normalized_texts,
            "duration_seconds": durations_seconds,
            "waveform_length": waveform_lengths,
            "sample_rate": sample_rate,
            "waveform": padded_waveforms
        }

    def _pad_waveforms(self, examples: list[LJSpeechExample]) -> torch.Tensor:
        # Right-padding preserves original waveform order and keeps true lengths available separately.
        waveforms: list[torch.Tensor] = [example.waveform for example in examples]
        return pad_sequence(waveforms, batch_first=True)
