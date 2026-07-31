# vocode.loggers: Evidence Persistence

`loggers/` writes the durable evidence of every execution. Each atomic run produces one capsule directory, every file inside a capsule is written by a module of this package, and nothing else in the tree persists run evidence.

| Module | Contents |
|---|---|
| `experiment.py` | `ExperimentLogger`, which creates the capsule and persists the run-level records. |
| `tracker.py` | `RunTracker`, which follows the execution lifecycle of a run. |
| `result.py` | `ExperimentResultRow`, the schema of one measured row. |
| `writer.py` | `ExperimentResultWriter`, which appends measured rows to the summary tables. |
| `checkpoint.py` | `CheckpointManifest` and `CheckpointLogger`, which persist checkpoint evidence. |

`ExperimentLogger` creates the capsule and writes `run_manifest.yaml` with the identity of the executed cell, `resolved_config.yaml` with the exact validated configuration, `hyperparameters.yaml`, and the append-only `logs/execution.log`. `RunTracker` records stage progress, failures, and completion, so the capsule states whether an execution finished and under which terminal condition.

`ExperimentResultRow` defines one measured row: the identity fields (evidence category, architecture, variant, lane, seed, and run identifier), the measured quality, timing, and footprint values, and the provenance fields consumed by the study registers. `ExperimentResultWriter` appends that row to the summary table of the capsule's context directory, which is `summary/experiments.csv` or, in the optimized-variant lane, `summary/experiments_v2.csv`. Rows are appended and never rewritten, so summary tables grow monotonically with executions.

`CheckpointManifest` and `CheckpointLogger` persist checkpoint evidence under `checkpoints/`: the durable training states with their update counts, and the manifest that identifies which state was selected and retained.

## Related Components

The capsule paths this package writes to are derived exclusively by `../configs/layout.py`. The curated study packages under `../../../artifacts/` are assembled from these capsules. The mirrored tests live in `../../tests/vocode/loggers/`.
