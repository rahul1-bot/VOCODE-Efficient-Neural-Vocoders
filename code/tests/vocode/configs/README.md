# tests.vocode.configs: The Configuration Tests

This directory holds 72 tests mirroring `vocode/configs/`.

The module `run.py` tests `ExperimentConfiguration`: construction from valid settings, validation failures for invalid values, agreement between evidence category and stage, the closed lane-label vocabulary, and the immutability of validated configurations.

The module `layout.py` tests `ExperimentArtifactLayout`: capsule path derivation across evidence categories, architectures, lanes, seeds, and run identifiers; the capsule-internal locations for checkpoints, logs, and metrics; and the summary-table location of each context directory.

All paths are exercised against temporary directories, and nothing touches a real artifact tree.
