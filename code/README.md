# VOCODE Code Tree

`code/` holds the complete source of the VOCODE study and none of its results. The measured evidence lives in `../artifacts/`, and the repository root `../README.md` states the study design, the decision rule, and the principal findings. Four packages sit side by side, and only `vocode` contains study-specific logic.

| Package | Role | Depends on |
|---|---|---|
| `syntheticmind/` | A PyTorch training framework providing the trainer, loops, accelerators, strategies, callbacks, loggers, and checkpoint state. | Nothing in this tree. |
| `vocode/` | The study client: models, objectives, data, metrics, deployment interventions, evidence writing, and the command-line interface. | `syntheticmind` |
| `tests/` | The permanent pytest suite, mirroring `vocode/` directory for directory. | `vocode`, `syntheticmind` |
| `modal/` | Cloud scheduling: the container image, the volume contract, the launch entrypoints, and the storage and synchronization utilities. | `vocode`, through a container copy. |

## Package Summaries

`syntheticmind/` is an independent PyTorch research framework that is unaware of this study. It owns trainer orchestration for the fit, validate, test, and predict stages; the `Module` and `DataModule` contracts; the CPU, CUDA, and MPS accelerators; the single-device and distributed strategy boundaries; the training, evaluation, prediction, and fit loops; the callback and logger interfaces; and native resumable checkpoints that carry model, optimizer, scheduler, callback, epoch, step, and random-number-generator state.

`vocode/` is the study client layered on that framework. Its `models/` package holds the architecture modules, each with its generator network, its discriminators where the training recipe is adversarial, and a `weights.py` adapter where a published release exists. The `losses/` package holds the per-architecture training objectives, `transforms/` holds the mel and LPC signal paths, `data/` holds the LJSpeech dataset and DataModule, and `metrics/` holds the atomic measurements with shared pitch extraction and ordered selection. The `optimization/` package implements the deployment interventions (compilation, quantization, pruning, export, sampling, and recovery) together with the registry that decides which interventions apply to which architecture. The `configs/` package validates every run setting and derives the artifact capsule path, `loggers/` persists run, result, and checkpoint evidence, and `trainers/` carries the reproduction, author-weight, and optimized-variant execution roles. The module `vocode/cli.py` is the single execution entry point.

`tests/` is a permanent pytest suite that mirrors `vocode/` directory for directory, so the tests for any module sit at the same relative path under `tests/`. It collects 2,202 tests plus 42 subtests and runs offline: inputs are constructed, and author releases are fabricated with self-consistent digests, so no test downloads a checkpoint or requires the corpus.

`modal/` is cloud scheduling, independent of the experiment logic. Its `vocode/` subdirectory defines the application, the volume, and the version-pinned image, and exposes the launch entrypoints for training, recovery, and author-weight evaluation. A container invokes the same atomic command-line interface a local shell would, so cloud execution adds transport and resources and never changes behavior. The modules `modal/storage.py` and `modal/sync.py` manage volume contents and pull finished run records back for review. Modal commands launch from the repository root, because the image builder copies `code/vocode/` and `code/syntheticmind/` by those paths.

## Configuration and Commands

`pytest.ini` sets `testpaths = tests`, declares `python_files = *.py` so that every module under `tests/` is a test module, selects `--import-mode=importlib`, and sets `pythonpath = .`, which makes `vocode` and `syntheticmind` resolve without environment variables. `ruff.toml` sets `target-version = "py314"`, `line-length = 120`, and the rule selection E4, E7, E9, F, and I; its import-sorting configuration declares `vocode` as first party and gives the `syntheticmind` framework its own import section between the third-party and first-party blocks.

Both tools run from this directory:

```bash
python -m pytest
ruff check .
```

Either tool accepts a narrower target, for example `python -m pytest tests/vocode/optimization` or `ruff check vocode/models`.

## The Command-Line Boundary

`vocode/cli.py` is atomic: one invocation resolves one configuration from the layered sources (declared defaults, then a `--config` YAML file, then repeatable `--override key=value` pairs, then explicit flags), executes exactly one architecture, seed, and stage, and writes exactly one artifact capsule. The flags `--model`, `--seed`, and `--run-id` identify the executed cell and are always required; the reproduction evaluation commands additionally require `--project-checkpoint-path`, and the optimized-variants lane requires `--hypothesis-id` and `--code-commit-hash`. Orchestration lives outside the process, so a failed cell cannot affect sibling cells, and parallel execution is performed by launching multiple atomic commands. The eight subcommands and their evidence categories are listed in the repository root `../README.md`.

A capsule is a directory at `<artifact-root>/<evidence-category>/ljspeech/<hardware-precision>/<architecture>[/<variant>]/runs/<run-id>/`; the variant segment exists exactly for the optimized-variants lane, and the seed is recorded inside the capsule rather than in the path. It holds `run_manifest.yaml`, `resolved_config.yaml`, and `hyperparameters.yaml` beside `checkpoints/`, `logs/execution.log`, `metrics/metrics.json`, and `seed_records/`, and it appends its measured row to the summary table of its context directory (`summary/experiments.csv`, or `summary/experiments_v2.csv` in the optimized-variant lane). These capsules are the source records from which `../artifacts/` was assembled.
