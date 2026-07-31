# tests.vocode.data: The Data Pipeline Tests

This directory holds 83 tests mirroring `vocode/data/`.

The module `ljspeech_dataset.py` tests the corpus access layer: record enumeration from a constructed corpus fixture, the identifier-sorted deterministic partition and its disjointness, example materialization with conditioning features, and batch collation with padding.

The module `ljspeech_datamodule.py` tests `LJSpeechDataConfig` and `LJSpeechDataModule`: per-stage dataloader construction, training-loader shuffling and fixed-length cropping, complete-utterance evaluation loading, split-size handling, and the worker policy.

Corpus fixtures are written to temporary directories with synthesized audio, so the tests exercise the real file-reading path without the real corpus.
