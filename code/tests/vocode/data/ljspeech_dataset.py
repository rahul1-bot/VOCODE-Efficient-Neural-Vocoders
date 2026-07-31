# This module:
# 1. Verifies LJSpeechRecord validation: identifier, transcript, and audio-path
#    rules that reject malformed metadata before any tensor is created
# 2. Verifies LJSpeechSource corpus resolution: dataset-root checks, pipe-
#    separated metadata parsing, deterministic identifier ordering, record
#    caching, and the failures raised on malformed or empty metadata
# 3. Verifies LJSpeechDataset loading and its preprocessing chain (random peak
#    gain, peak normalization, resampling, segment cropping and padding), and
#    LJSpeechBatchCollator padding with provenance carried alongside
#
# Design decisions:
# - Every test builds a minimal synthetic corpus in a temporary directory: a few
#   short generated tones plus a metadata file in the LJSpeech layout. The real
#   LJSpeech tree is never read, so the suite runs on any machine
# - Waveforms are short tones written as int16, which exercises the scipy WAV
#   reader and the 32768 scaling convention the loader depends on
# - Segment cropping draws a random start offset, so crop assertions bound the
#   length and use the degenerate full-length case, where the offset can only be
#   zero, to assert exact content
# - The random peak gain is exercised with a degenerate decibel range whose
#   bounds coincide, which makes the drawn target peak deterministic
#
# Author: Rahul Sawhney

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from pydantic import ValidationError
from scipy.io import wavfile

from vocode.data.ljspeech_dataset import (
    LJSpeechBatch,
    LJSpeechBatchCollator,
    LJSpeechDataset,
    LJSpeechExample,
    LJSpeechRecord,
    LJSpeechSource,
)


class SyntheticCorpusBuilder:
    # Writes a minimal corpus in the LJSpeech layout so no test reads the real tree.
    def __init__(self, root: Path, sample_rate: int) -> None:
        # Binds the corpus layout and the amplitude every generated tone peaks at.
        self._root: Path = root
        self._sample_rate: int = sample_rate
        self._wavs_directory: Path = root / "wavs"
        self._metadata_path: Path = root / "metadata.csv"
        self._amplitude: float = 0.3

    def build(self, identifiers: tuple[str, ...], sample_counts: tuple[int, ...]) -> None:
        # Writes one tone per identifier and lists them in the given metadata order.
        identifier: str
        sample_count: int
        for identifier, sample_count in zip(identifiers, sample_counts):
            self.write_waveform(identifier, sample_count)
        self.write_metadata(
            tuple(
                f"{identifier}|Raw text {index}|Normalized text {index}"
                for index, identifier in enumerate(identifiers)
            )
        )

    def write_waveform(self, identifier: str, sample_count: int) -> Path:
        # Writes one int16 tone under the corpus wavs directory.
        self._wavs_directory.mkdir(parents=True, exist_ok=True)
        positions: np.ndarray = np.arange(sample_count, dtype=np.float32)
        tone: np.ndarray = self._amplitude * np.sin(
            2.0 * np.pi * 220.0 * positions / self._sample_rate
        )
        audio_path: Path = self._wavs_directory / f"{identifier}.wav"
        wavfile.write(str(audio_path), self._sample_rate, (tone * 32767.0).astype(np.int16))
        return audio_path

    def write_metadata(self, lines: tuple[str, ...]) -> None:
        # Writes the pipe-separated metadata file the source parses.
        self._metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @property
    def wavs_directory(self) -> Path:
        # Returns the directory holding the generated waveforms.
        return self._wavs_directory

    @property
    def metadata_path(self) -> Path:
        # Returns the generated metadata file path.
        return self._metadata_path

    @property
    def peak_amplitude(self) -> float:
        # Returns the peak every generated tone reaches before quantization.
        return self._amplitude


class LJSpeechRecordValidationTest(unittest.TestCase):
    # Verifies the metadata record rules that guard the dataset boundary.
    def setUp(self) -> None:
        # Builds a one-utterance corpus supplying a materialized audio path.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._builder: SyntheticCorpusBuilder = SyntheticCorpusBuilder(self._root, 22050)
        self._builder.build(("LJ001-0001",), (2000,))
        self._audio_path: Path = self._builder.wavs_directory / "LJ001-0001.wav"

    def tearDown(self) -> None:
        # Removes the synthetic corpus created for this test.
        self._temporary_directory.cleanup()

    def test_valid_record_is_accepted(self) -> None:
        # A complete record over a materialized WAV file validates.
        record: LJSpeechRecord = LJSpeechRecord(
            identifier="LJ001-0001",
            audio_path=self._audio_path,
            raw_text="Raw text",
            normalized_text="Normalized text"
        )
        self.assertEqual(record.identifier, "LJ001-0001")
        self.assertEqual(record.audio_path, self._audio_path)

    def test_empty_identifier_is_rejected(self) -> None:
        # Identifiers drive deterministic sorting and artifact naming.
        with self.assertRaises(ValidationError):
            LJSpeechRecord(
                identifier="",
                audio_path=self._audio_path,
                raw_text="Raw text",
                normalized_text="Normalized text"
            )

    def test_missing_audio_file_is_rejected(self) -> None:
        # The dataset layer accepts only materialized corpus audio.
        with self.assertRaises(FileNotFoundError):
            LJSpeechRecord(
                identifier="LJ001-9999",
                audio_path=self._builder.wavs_directory / "LJ001-9999.wav",
                raw_text="Raw text",
                normalized_text="Normalized text"
            )

    def test_non_wav_audio_file_is_rejected(self) -> None:
        # The corpus contract is WAV, so another container cannot be admitted.
        transcript_path: Path = self._root / "LJ001-0001.txt"
        transcript_path.write_text("not audio", encoding="utf-8")
        with self.assertRaises(ValidationError):
            LJSpeechRecord(
                identifier="LJ001-0001",
                audio_path=transcript_path,
                raw_text="Raw text",
                normalized_text="Normalized text"
            )

    def test_empty_transcript_fields_are_rejected(self) -> None:
        # Empty transcripts indicate corrupt metadata and weaken auditability.
        with self.assertRaises(ValidationError):
            LJSpeechRecord(
                identifier="LJ001-0001",
                audio_path=self._audio_path,
                raw_text="",
                normalized_text="Normalized text"
            )
        with self.assertRaises(ValidationError):
            LJSpeechRecord(
                identifier="LJ001-0001",
                audio_path=self._audio_path,
                raw_text="Raw text",
                normalized_text=""
            )

    def test_string_audio_path_is_rejected_under_strict_validation(self) -> None:
        # Paths cross the boundary as Path objects, never as strings.
        with self.assertRaises(ValidationError):
            LJSpeechRecord(
                identifier="LJ001-0001",
                audio_path=str(self._audio_path),
                raw_text="Raw text",
                normalized_text="Normalized text"
            )

    def test_record_is_frozen(self) -> None:
        # A validated record cannot drift once split construction has used it.
        record: LJSpeechRecord = LJSpeechRecord(
            identifier="LJ001-0001",
            audio_path=self._audio_path,
            raw_text="Raw text",
            normalized_text="Normalized text"
        )
        with self.assertRaises(ValidationError):
            record.identifier: str = "LJ001-0002"

    def test_example_rejects_non_positive_measurements(self) -> None:
        # Zero-length audio or an invalid rate is rejected before collation.
        record: LJSpeechRecord = LJSpeechRecord(
            identifier="LJ001-0001",
            audio_path=self._audio_path,
            raw_text="Raw text",
            normalized_text="Normalized text"
        )
        with self.assertRaises(ValidationError):
            LJSpeechExample(
                record=record,
                waveform=torch.zeros(0),
                sample_rate=22050,
                waveform_length=0
            )
        with self.assertRaises(ValidationError):
            LJSpeechExample(
                record=record,
                waveform=torch.zeros(10),
                sample_rate=0,
                waveform_length=10
            )


class LJSpeechSourceResolutionTest(unittest.TestCase):
    # Verifies corpus validation, metadata parsing, ordering, and caching.
    def setUp(self) -> None:
        # Builds a three-utterance corpus whose metadata order is not sorted.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._builder: SyntheticCorpusBuilder = SyntheticCorpusBuilder(self._root, 22050)
        self._builder.build(("LJ001-0003", "LJ001-0001", "LJ001-0002"), (2000, 3000, 2500))
        self._source: LJSpeechSource = LJSpeechSource(self._root)

    def tearDown(self) -> None:
        # Removes the synthetic corpus created for this test.
        self._temporary_directory.cleanup()

    def test_complete_corpus_validates(self) -> None:
        # A root holding metadata and a wavs directory passes the boundary check.
        self._source.validate_dataset_root()
        self.assertEqual(self._source.dataset_root, self._root)

    def test_missing_dataset_root_is_rejected(self) -> None:
        # Incomplete corpus hydration must fail before a job starts.
        with self.assertRaises(FileNotFoundError):
            LJSpeechSource(self._root / "absent").validate_dataset_root()

    def test_missing_metadata_file_is_rejected(self) -> None:
        # Without metadata there is no record set to build splits from.
        self._builder.metadata_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self._source.validate_dataset_root()

    def test_missing_wavs_directory_is_rejected(self) -> None:
        # Without the audio directory no record can resolve its waveform.
        audio_path: Path
        for audio_path in self._builder.wavs_directory.iterdir():
            audio_path.unlink()
        self._builder.wavs_directory.rmdir()
        with self.assertRaises(FileNotFoundError):
            self._source.validate_dataset_root()

    def test_records_are_sorted_by_identifier(self) -> None:
        # Split construction and artifact rows depend on a machine-independent order.
        records: tuple[LJSpeechRecord, ...] = self._source.load_records()
        self.assertEqual(
            [record.identifier for record in records],
            ["LJ001-0001", "LJ001-0002", "LJ001-0003"],
            msg="Records must be ordered by identifier regardless of metadata order"
        )

    def test_records_are_cached_after_the_first_load(self) -> None:
        # The metadata file is static during a run, so parsing happens once.
        first_load: tuple[LJSpeechRecord, ...] = self._source.load_records()
        self.assertIs(self._source.load_records(), first_load)

    def test_record_audio_paths_resolve_under_the_wavs_directory(self) -> None:
        # Waveform locations are derived from the identifier, not from metadata.
        record: LJSpeechRecord
        for record in self._source.load_records():
            self.assertEqual(record.audio_path.parent, self._builder.wavs_directory)
            self.assertEqual(record.audio_path.name, f"{record.identifier}.wav")

    def test_transcript_fields_are_parsed_from_the_pipe_separated_line(self) -> None:
        # The record carries the raw and normalized transcripts in metadata order.
        records: tuple[LJSpeechRecord, ...] = self._source.load_records()
        self.assertEqual(records[0].raw_text, "Raw text 1")
        self.assertEqual(records[0].normalized_text, "Normalized text 1")

    def test_normalized_text_retains_embedded_separators(self) -> None:
        # Splitting stops after two separators, so transcripts may contain pipes.
        self._builder.write_metadata(("LJ001-0001|Raw text|Normalized | text | tail",))
        records: tuple[LJSpeechRecord, ...] = LJSpeechSource(self._root).load_records()
        self.assertEqual(records[0].normalized_text, "Normalized | text | tail")

    def test_blank_metadata_lines_are_ignored(self) -> None:
        # Trailing or interleaved blank lines are formatting, not records.
        self._builder.write_metadata(("LJ001-0001|Raw text|Normalized text", "", "   "))
        records: tuple[LJSpeechRecord, ...] = LJSpeechSource(self._root).load_records()
        self.assertEqual(len(records), 1)

    def test_metadata_without_records_is_rejected(self) -> None:
        # An empty corpus cannot support training or evaluation.
        self._builder.write_metadata(("", "   "))
        with self.assertRaises(ValueError):
            LJSpeechSource(self._root).load_records()

    def test_malformed_metadata_line_is_rejected(self) -> None:
        # A line missing a transcript field indicates a corrupt metadata file.
        self._builder.write_metadata(("LJ001-0001|only two fields",))
        with self.assertRaises(ValueError):
            LJSpeechSource(self._root).load_records()

    def test_metadata_referencing_absent_audio_is_rejected(self) -> None:
        # Metadata may not promise waveforms the corpus does not hold.
        self._builder.write_metadata(("LJ999-9999|Raw text|Normalized text",))
        with self.assertRaises(FileNotFoundError):
            LJSpeechSource(self._root).load_records()


class LJSpeechDatasetLoadingTest(unittest.TestCase):
    # Verifies lazy waveform loading and the returned example contract.
    def setUp(self) -> None:
        # Builds a three-utterance corpus and the whole-utterance dataset over it.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._builder: SyntheticCorpusBuilder = SyntheticCorpusBuilder(self._root, 22050)
        self._sample_counts: tuple[int, ...] = (2000, 3000, 2500)
        self._builder.build(("LJ001-0001", "LJ001-0002", "LJ001-0003"), self._sample_counts)
        self._records: tuple[LJSpeechRecord, ...] = LJSpeechSource(self._root).load_records()
        self._dataset: LJSpeechDataset = LJSpeechDataset(records=self._records)

    def tearDown(self) -> None:
        # Removes the synthetic corpus created for this test.
        self._temporary_directory.cleanup()

    def test_length_matches_the_record_count(self) -> None:
        # The dataset exposes exactly the records it was constructed with.
        self.assertEqual(len(self._dataset), len(self._records))

    def test_records_property_returns_the_injected_sequence(self) -> None:
        # Runners read the record sequence to build artifact rows.
        self.assertEqual(self._dataset.records, self._records)

    def test_item_carries_its_record_and_measured_length(self) -> None:
        # Provenance travels with the tensor so measurements stay attributable.
        example: LJSpeechExample = self._dataset[0]
        self.assertEqual(example.record, self._records[0])
        self.assertEqual(example.waveform_length, self._sample_counts[0])
        self.assertEqual(example.waveform.shape[-1], self._sample_counts[0])

    def test_waveform_is_float32_inside_the_unit_range(self) -> None:
        # Waveforms reach the models as float32 in [-1, 1] by the int16 convention.
        example: LJSpeechExample = self._dataset[0]
        self.assertEqual(example.waveform.dtype, torch.float32)
        self.assertLessEqual(float(example.waveform.abs().max()), 1.0)
        self.assertAlmostEqual(
            float(example.waveform.abs().max()),
            self._builder.peak_amplitude,
            places=3,
            msg="The loaded peak must match the written tone amplitude"
        )

    def test_sample_rate_matches_the_corpus_rate(self) -> None:
        # Without resampling the example reports the rate stored in the file.
        self.assertEqual(self._dataset[0].sample_rate, 22050)

    def test_items_follow_the_sorted_record_order(self) -> None:
        # Index order is identifier order, which keeps split membership stable.
        index: int
        for index in range(len(self._dataset)):
            self.assertEqual(self._dataset[index].record.identifier, self._records[index].identifier)


class LJSpeechDatasetPreprocessingTest(unittest.TestCase):
    # Verifies the optional gain, normalization, resampling, and cropping chain.
    def setUp(self) -> None:
        # Builds a two-utterance corpus with known lengths for the crop assertions.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._builder: SyntheticCorpusBuilder = SyntheticCorpusBuilder(self._root, 22050)
        self._source_length: int = 2000
        self._builder.build(("LJ001-0001", "LJ001-0002"), (self._source_length, 3000))
        self._records: tuple[LJSpeechRecord, ...] = LJSpeechSource(self._root).load_records()

    def tearDown(self) -> None:
        # Removes the synthetic corpus created for this test.
        self._temporary_directory.cleanup()

    def test_segment_cropping_returns_the_requested_length(self) -> None:
        # Training draws fixed-length segments regardless of utterance length.
        dataset: LJSpeechDataset = LJSpeechDataset(records=self._records, segment_size=512)
        example: LJSpeechExample = dataset[0]
        self.assertEqual(example.waveform.shape[-1], 512)
        self.assertEqual(example.waveform_length, 512)

    def test_full_length_segment_reproduces_the_whole_utterance(self) -> None:
        # With no room to offset, the crop must be the utterance itself.
        whole: LJSpeechDataset = LJSpeechDataset(records=self._records)
        cropped: LJSpeechDataset = LJSpeechDataset(
            records=self._records,
            segment_size=self._source_length
        )
        self.assertTrue(
            torch.equal(cropped[0].waveform, whole[0].waveform),
            msg="A segment as long as the utterance must not alter the waveform"
        )

    def test_short_utterances_are_right_padded_to_the_segment_size(self) -> None:
        # Padding preserves the leading samples and appends silence.
        whole: LJSpeechDataset = LJSpeechDataset(records=self._records)
        padded: LJSpeechDataset = LJSpeechDataset(records=self._records, segment_size=5000)
        example: LJSpeechExample = padded[0]
        self.assertEqual(example.waveform.shape[-1], 5000)
        self.assertTrue(torch.equal(example.waveform[: self._source_length], whole[0].waveform))
        self.assertTrue(
            bool((example.waveform[self._source_length:] == 0.0).all()),
            msg="Padding must append silence rather than repeat samples"
        )

    def test_peak_normalization_scales_to_the_configured_peak(self) -> None:
        # Recipes that require a fixed peak get exactly that peak at load.
        dataset: LJSpeechDataset = LJSpeechDataset(
            records=self._records,
            peak_normalization_enabled=True,
            peak_normalization_value=0.75
        )
        self.assertAlmostEqual(float(dataset[0].waveform.abs().max()), 0.75, places=5)

    def test_random_peak_gain_draws_the_requested_decibel_level(self) -> None:
        # A degenerate decibel range fixes the drawn target peak exactly.
        dataset: LJSpeechDataset = LJSpeechDataset(
            records=self._records,
            random_peak_gain_range_db=(-6.0, -6.0)
        )
        expected_peak: float = 10.0 ** (-6.0 / 20.0)
        self.assertAlmostEqual(float(dataset[0].waveform.abs().max()), expected_peak, places=4)

    def test_resampling_changes_the_reported_rate_and_length(self) -> None:
        # The example reports the target rate once resampling has been applied. The length is
        # bounded rather than pinned: a polyphase resampler determines its output length from
        # the rational rate ratio and its own filter, so the result may land a sample either
        # side of the ideal ratio. A tolerance of two samples is wide enough to absorb that
        # while remaining far too tight to pass if resampling had not run at all.
        dataset: LJSpeechDataset = LJSpeechDataset(records=self._records, resample_rate=16000)
        example: LJSpeechExample = dataset[0]
        self.assertEqual(example.sample_rate, 16000)
        expected_length: int = round(self._source_length * 16000 / 22050)
        self.assertAlmostEqual(
            example.waveform_length,
            expected_length,
            delta=2,
            msg=f"Resampled length {example.waveform_length} is far from {expected_length}"
        )

    def test_resampling_precedes_segment_extraction(self) -> None:
        # Segment sizes are expressed at the target rate, so the crop is exact.
        dataset: LJSpeechDataset = LJSpeechDataset(
            records=self._records,
            resample_rate=16000,
            segment_size=1024
        )
        example: LJSpeechExample = dataset[0]
        self.assertEqual(example.sample_rate, 16000)
        self.assertEqual(example.waveform.shape[-1], 1024)


class LJSpeechBatchCollationTest(unittest.TestCase):
    # Verifies padded batching and the provenance carried beside the tensor.
    def setUp(self) -> None:
        # Builds a corpus of two differing lengths and the examples to collate.
        self._temporary_directory: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._root: Path = Path(self._temporary_directory.name)
        self._builder: SyntheticCorpusBuilder = SyntheticCorpusBuilder(self._root, 22050)
        self._sample_counts: tuple[int, ...] = (2000, 3000)
        self._builder.build(("LJ001-0001", "LJ001-0002"), self._sample_counts)
        self._records: tuple[LJSpeechRecord, ...] = LJSpeechSource(self._root).load_records()
        self._dataset: LJSpeechDataset = LJSpeechDataset(records=self._records)
        self._collator: LJSpeechBatchCollator = LJSpeechBatchCollator()
        self._examples: list[LJSpeechExample] = [self._dataset[0], self._dataset[1]]

    def tearDown(self) -> None:
        # Removes the synthetic corpus created for this test.
        self._temporary_directory.cleanup()

    def test_batch_exposes_the_documented_keys(self) -> None:
        # Every runner reads this dictionary, so its key set is a contract.
        batch: LJSpeechBatch = self._collator(self._examples)
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

    def test_waveforms_are_right_padded_to_the_longest_utterance(self) -> None:
        # Padding builds a rectangular tensor without reordering the batch.
        batch: LJSpeechBatch = self._collator(self._examples)
        waveform: torch.Tensor = batch["waveform"]
        self.assertIsInstance(
            waveform,
            torch.Tensor,
            msg="The waveform entry of the batch mapping must be the padded tensor"
        )
        self.assertEqual(waveform.shape, (2, max(self._sample_counts)))
        self.assertTrue(
            torch.equal(waveform[0, : self._sample_counts[0]], self._examples[0].waveform)
        )
        self.assertTrue(
            bool((waveform[0, self._sample_counts[0]:] == 0.0).all()),
            msg="The shorter utterance must be right-padded with silence"
        )

    def test_true_lengths_survive_padding(self) -> None:
        # Padding never enters a measurement, because true lengths travel along.
        batch: LJSpeechBatch = self._collator(self._examples)
        self.assertEqual(batch["waveform_length"], self._sample_counts)

    def test_durations_are_derived_from_length_and_rate(self) -> None:
        # Duration is computed from the loaded waveform, not from metadata.
        batch: LJSpeechBatch = self._collator(self._examples)
        expected_durations: tuple[float, ...] = tuple(
            count / 22050 for count in self._sample_counts
        )
        self.assertEqual(batch["duration_seconds"], expected_durations)

    def test_metadata_tuples_follow_the_example_order(self) -> None:
        # Provenance rows must align with the tensor rows they describe.
        batch: LJSpeechBatch = self._collator(self._examples)
        self.assertEqual(batch["identifier"], ("LJ001-0001", "LJ001-0002"))
        self.assertEqual(batch["raw_text"], ("Raw text 0", "Raw text 1"))
        self.assertEqual(batch["normalized_text"], ("Normalized text 0", "Normalized text 1"))
        self.assertEqual(
            batch["audio_path"],
            tuple(str(record.audio_path) for record in self._records)
        )

    def test_batch_sample_rate_is_a_scalar(self) -> None:
        # LJSpeech is a single-rate corpus, so one rate describes the batch.
        batch: LJSpeechBatch = self._collator(self._examples)
        self.assertEqual(batch["sample_rate"], 22050)

    def test_empty_batch_is_rejected(self) -> None:
        # An empty batch indicates a sampler or split-construction failure.
        with self.assertRaises(ValueError):
            self._collator([])


if __name__ == "__main__":
    unittest.main()
