# Cloud

All study executions ran on Modal as isolated single-use container jobs. The cloud layer, `code/modal/`, holds no experiment logic: every job invokes the same atomic command a local shell would, so cloud execution adds transport and resources and never changes behavior.

## Prerequisites

A Modal account and the Modal client: `pip install modal`, then `modal setup` to authenticate. Nothing else is required. The volume `vocode-data` is created automatically on first use, the first job that needs the corpus downloads LJSpeech into the volume (`https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2`, extracted cloud-side), author-weight retrieval happens at run time under `/data/pretrained` (the root can also be pre-seeded with the storage utility), and no Modal secrets are used by any job.

## Infrastructure

The application `vocode-checkpoint01`, the volume, and the container image are defined once in `code/modal/vocode/application.py`, and the launch files import them. The image (`code/modal/vocode/runtime.py`) is Debian-slim Python 3.14 with `git`, `wget`, and `build-essential` (the compiler serves the `pesq` source build), the exact dependency pins of `requirements.txt`, and `TORCH_HOME` at `/data/pretrained/torch`. It copies `code/vocode` and `code/syntheticmind` into `/workspace`, so every capsule executes exactly the code state that spawned it, and any source change invalidates the copy layers on the next launch. The Python pins define the aligned measurement stack; the Debian base and host driver were not digest-pinned in the retained runs. Containers mount the volume at `/data` with three maintained roots:

| Root | Contents |
|---|---|
| `datasets` | The LJSpeech corpus, hydrated automatically when absent |
| `pretrained` | Author-released weights and framework caches |
| `experiment_artifacts` | Run output capsules |

## Resource Lanes

| `--hardware-name` | GPU | CPU units | Host memory |
|---|---|---:|---:|
| `b200` | B200 | 16 | 64 GiB |
| `h100` | H100 | 16 | 64 GiB |
| `a100_80gb` | A100-80GB | 16 | 64 GiB |
| `l40s` | L40S | 8 | 32 GiB |
| `cpu` | none | 8 | 16 GiB |

One shared single-use worker class serves every lane; the base declaration is CPU-only, and the lane profile is applied at dispatch, so a GPU lane is always an explicit invocation decision. The study trained eleven configurations on `l40s` and RFWave on `a100_80gb`, and evaluated on `b200` and `cpu`. Paired contrasts match the requested profile, never the physical host.

## Launch Workflow

`code/modal/vocode/train.py` exposes `launch_training`, which maps stage, evidence category, and optimization variant onto the atomic commands. Its parser accepts roughly thirty-five flags with validated choices; `--run-id` defaults to a UTC timestamp, and the launching repository commit is recorded automatically (twelve hexadecimal characters, suffixed `-dirty` when the code tree has uncommitted changes) so every capsule carries code provenance. Launch defaults are conveniences, not study values; the executed settings of every study run are recorded in its capsule's `resolved_config.yaml`.

Dispatch is attached by default; `--spawn-remote` spawns the call and logs the function-call identifier and the dashboard URL instead of blocking, and `modal run --detach` keeps the app alive after the local terminal closes. One Modal attempt is capped at 24 hours, so long full-scale training runs resume automatically: the resolver looks for `last.ckpt`, then the newest durable checkpoint, under the run's own `checkpoints/` directory and continues from it. Runs with any batch limit are treated as bounded verification and never resume automatically.

`code/modal/vocode/published.py` exposes `launch_published_evaluation` and `verify` for the nine author-release architectures on a worker pinned to a B200 at FP32 with a 7,200-second timeout; `verify` is the bounded preflight capped at 50 test utterances.

```bash
modal run --detach code/modal/vocode/train.py::launch_training \
    --architecture-name vocos --hardware-name l40s --precision-name bf16 \
    --run-id vocos_full_bf16bs32 --spawn-remote

modal run code/modal/vocode/published.py::verify --architecture-name hifigan_v1
```

## Storage and Synchronization

`code/modal/storage.py` provides volume inventory (defaulting to the three maintained roots) and explicit uploads, including pre-seeding `/pretrained`; `code/modal/sync.py` pulls completed capsules from the volume into the local evidence tree, refuses unbounded remote paths, and leaves existing destinations untouched unless `--overwrite-existing-local-files` is passed. Nothing downloads automatically.

```bash
python code/modal/storage.py upload --volume-name vocode-data \
    --local-path pretrained --remote-path /pretrained --force

python code/modal/sync.py --volume-name vocode-data \
    --remote-relative-path experiment_artifacts/project_trained_reproduction \
    --local-destination-root experiment_artifacts
```
