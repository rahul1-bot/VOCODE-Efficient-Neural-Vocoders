# vocode.data: The LJSpeech Data Pipeline

`data/` owns corpus access and the deterministic dataset partition. All twelve configurations of the study consume the same partition through this package, and no other module reads the corpus.

| Module | Contents |
|---|---|
| `ljspeech_dataset.py` | `LJSpeechSource`, `LJSpeechRecord`, `LJSpeechExample`, `LJSpeechDataset`, and `LJSpeechBatchCollator`. |
| `ljspeech_datamodule.py` | `LJSpeechDataConfig` and `LJSpeechDataModule`. |

`LJSpeechSource` enumerates the LJSpeech 1.1 corpus and produces one `LJSpeechRecord` per utterance. `LJSpeechDataset` materializes `LJSpeechExample` items, pairing each waveform with the conditioning features of the selected architecture, and `LJSpeechBatchCollator` assembles padded batches. The partition is identifier-sorted and disjoint: the ordered utterance identifiers are split into contiguous training, validation, and evaluation blocks without a random seed, so split membership is a fixed property of the corpus rather than a resampled choice. The study partition fixes 12,475 training, 100 validation, and 525 evaluation utterances of the 13,100-utterance corpus.

`LJSpeechDataConfig` carries the data settings of one run: the corpus root, the split sizes, the segment length, the per-stage batch sizes, and the worker policy. `LJSpeechDataModule` binds those settings to the framework `DataModule` contract and constructs the stage-specific dataloaders. Training loaders shuffle and crop fixed-length segments under the architecture-native crop length, while validation and evaluation loaders consume complete utterances in deterministic order.

## Related Components

Architecture-specific data defaults (sampling rate, mel configuration, crop length, and batch size) are selected by the architecture choice in `vocode/cli.py` and validated in `../configs/`. The conditioning features themselves are computed by `../transforms/`. The mirrored tests live in `../../tests/vocode/data/`.
