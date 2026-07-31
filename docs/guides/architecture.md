# Architecture

The software is organized as a study-agnostic training framework and a study client that layers on it. The framework, `code/syntheticmind/`, owns trainer orchestration, the `Module` and `DataModule` contracts, accelerators, strategies, loops, callbacks, loggers, and resumable checkpoint state; nothing in it references vocoders. The client, `code/vocode/`, owns the architectures, objectives, data pipeline, measurement stack, deployment transformations, and evidence writing. This boundary keeps reusable machinery separate from experiment logic, and it is enforced by imports: the framework never imports the client.

## The Atomic Execution Model

The single entry point is `code/vocode/cli.py`. One invocation resolves one configuration, executes one architecture, seed, and stage, and writes one evidence capsule. Orchestration lives outside the process, so a failed run cannot affect a sibling run, and parallel execution is performed by launching multiple commands.

| Subcommand | Evidence category | Stage |
|---|---|---|
| `train-reproduction` | `project_trained_reproduction` | train |
| `validation-reproduction` | `project_trained_reproduction` | validation |
| `test-reproduction` | `project_trained_reproduction` | test |
| `train-hybrid` | `project_hybrid_variants` | train |
| `test-hybrid` | `project_hybrid_variants` | test |
| `test-published` | `published_checkpoint_evaluation` | test |
| `evaluate-optimized-variant` | `project_optimized_variants` | test |
| `recover-optimized-variant` | `project_optimized_variants` | train |

Configuration resolves in four layers: declared defaults, then an optional `--config` YAML file, then repeatable `--override key=value` pairs, then explicit flags. The resolved result is validated before execution (`code/vocode/configs/run.py`), and an unknown override key fails rather than being ignored. The flags `--model`, `--seed`, and `--run-id` are always required; reproduction evaluation additionally requires `--project-checkpoint-path`, and the optimized-variants lane requires `--hypothesis-id` and `--code-commit-hash`.

## Capsule Anatomy

Every execution writes one capsule directory, derived by `code/vocode/configs/layout.py`. The capsule tree is category, then dataset, then the hardware-precision label, then architecture, then (in the optimized-variants lane only) the variant, then the run identifier:

```
<artifact-root>/<evidence-category>/ljspeech/<hardware-precision>/<architecture>[/<variant>]/runs/<run-id>/
├── run_manifest.yaml       The identity of the executed cell.
├── resolved_config.yaml    The exact validated configuration.
├── hyperparameters.yaml
├── checkpoints/            Durable training states.
├── logs/execution.log
├── metrics/metrics.json
└── seed_records/           Written by the seed-record writer.
```

Datacenter NVIDIA lanes carry a vendor prefix in the hardware-precision label (for example `nvidia_b200_fp32`, or `cpu_fp32` without one). The optimized-variants lane encodes hardware only (`nvidia_b200`, `modal_cpu8`), because numeric precision is the experimental variable there and lives in the variant name. The seed deliberately takes no part in path composition; it is recorded as provenance inside the capsule, so two seeds of one run identifier address the same capsule directory. Each capsule appends one measured row to the summary table of its context directory (`summary/experiments.csv`, or `summary/experiments_v2.csv` in the optimized-variants lane, whose versioned schema keeps historical summary files frozen). Rows are appended and never rewritten.

## Data Flow

1. A command is launched locally or scheduled onto Modal (`code/modal/`); the container path invokes the same command, so cloud execution never changes behavior.
2. The command resolves and validates its configuration, constructs the architecture module and DataModule, and executes the stage through the framework trainer.
3. The measurement stack (`code/vocode/metrics/`) computes the selected metrics; the loggers (`code/vocode/loggers/`) persist the capsule and append the summary row.
4. Completed capsules are synchronized from the cloud volume (`code/modal/sync.py`) and curated into the three study packages under `artifacts/`, whose registers the report cites.

## Hardware Lanes

A deployment artifact executes on one requested hardware lane, `b200` or `cpu`. B200 jobs request 16 CPUs and 64 GiB, CPU jobs request 8 CPUs and 16 GiB, and paired contrasts always match the requested resource profile rather than the physical host. The identity transformation on the same lane defines the paired control of every transformed group.

## Failure Discipline

Unsupported architecture and transformation combinations fail explicitly before execution expansion and are recorded in the exclusion registers; author-weight loads are strict and never fall back to project-trained state; and non-finite training states halt the run rather than training through corruption.
