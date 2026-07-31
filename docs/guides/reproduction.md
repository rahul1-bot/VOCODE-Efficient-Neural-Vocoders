# Reproduction

This walkthrough goes from a clean machine to a measured evidence capsule. The repository stores no pretrained weights and no corpus: LJSpeech, the author-released checkpoints (SHA-256-verified before a strict load), and the UTMOS predictor are fetched from their sources at run time.

## 1. Install

Python 3.14 with the exact pins of the study's container image (declared in `code/modal/vocode/runtime.py`):

```bash
git clone https://github.com/rahul1-bot/VOCODE-Efficient-Neural-Vocoders.git
cd VOCODE-Efficient-Neural-Vocoders
pip install -r requirements.txt
```

## 2. Verify Offline

The suite and linter require no checkpoint, corpus, or GPU:

```bash
cd code
python -m pytest    # 2,202 tests plus 42 subtests
ruff check .
```

## 3. Train a Configuration

Local training uses the same atomic command that the study scheduled onto Modal; `--dataset-root` must point at an LJSpeech 1.1 directory:

```bash
python -m vocode.cli train-reproduction \
    --model vocos --seed 1234 --run-id vocos_reproduction_training \
    --dataset-root /path/to/LJSpeech-1.1 \
    --hardware-name cpu --precision-name fp32 --accelerator cpu \
    --artifact-root ./experiment_artifacts
```

Cloud training replaces the local invocation with the Modal launch described in `cloud.md`. Durable checkpoints appear every 5,000 updates under the capsule's `checkpoints/` directory, and an interrupted run resumes from its last durable state.

## 4. Evaluate a Retained Checkpoint

```bash
python -m vocode.cli test-reproduction \
    --model vocos --seed 42 --run-id vocos_seed42_evaluation \
    --project-checkpoint-path /path/to/retained_checkpoint.pt \
    --dataset-root /path/to/LJSpeech-1.1 \
    --hardware-name cpu --precision-name fp32 --accelerator cpu \
    --artifact-root ./experiment_artifacts
```

The study evaluated seeds 42, 43, and 44 per configuration; each invocation is one seed.

## 5. Run a Deployment Intervention

The optimized-variants lane requires the hypothesis identifier and the executing commit as provenance; variant names follow the vocabulary in `deployment.md`:

```bash
python -m vocode.cli evaluate-optimized-variant \
    --model vocos --optimization-variant fp16_weights \
    --seed 42 --run-id vocos_fp16_seed42 \
    --hypothesis-id H2 --code-commit-hash $(git rev-parse HEAD) \
    --project-checkpoint-path /path/to/retained_checkpoint.pt \
    --dataset-root /path/to/LJSpeech-1.1 \
    --hardware-name cpu --precision-name fp32 --accelerator cpu \
    --artifact-root ./experiment_artifacts
```

## 6. Locate the Output

Every invocation writes one capsule at `<artifact-root>/<evidence-category>/ljspeech/<hardware-precision>/<architecture>[/<variant>]/runs/<run-id>/`, containing `run_manifest.yaml`, `resolved_config.yaml`, `hyperparameters.yaml`, `checkpoints/`, `logs/execution.log`, `metrics/metrics.json`, and `seed_records/`, and appends one measured row to the context directory's summary table (`summary/experiments.csv`, or `summary/experiments_v2.csv` in the optimized-variants lane). The seed is recorded inside the capsule rather than in the path, so repeated seeds of one run identifier address one capsule.

## Scope of Reproduction

Re-running training reproduces the procedure, not the bitwise states: the study's accepted runs predate deterministic construction seeding, so 1234 is a recorded rather than reconstructible seed, and the Retained Project Checkpoints are immutable evidence rather than regeneration targets. The measured evidence behind every reported number ships in `artifacts/` and is verifiable without any execution through the procedure in `evidence.md`.
