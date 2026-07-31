# tests.vocode.loggers: The Evidence Persistence Tests

This directory holds 105 tests mirroring `vocode/loggers/`.

The module `experiment.py` tests `ExperimentLogger`: capsule creation, manifest and resolved-configuration persistence, hyperparameter recording, and execution-log appending. The module `tracker.py` tests `RunTracker`: lifecycle progression, failure recording, and terminal-state semantics. The module `result.py` tests `ExperimentResultRow`: the measured-row schema, its identity fields, and value serialization. The module `writer.py` tests `ExperimentResultWriter`: append-only summary-table writing into `experiments.csv` and `experiments_v2.csv`, header discipline, and row integrity across repeated appends. The module `checkpoint.py` tests `CheckpointManifest` and `CheckpointLogger`: the checkpoint evidence layout, the manifest content, and retained-state identification.

Every test writes into temporary directories, and no test touches a real capsule.
