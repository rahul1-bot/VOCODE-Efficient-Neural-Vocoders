# The VOCODE Modal Launch Surfaces

`modal/vocode/` is the launch layer that schedules VOCODE work onto Modal. Each launch file owns a local path that parses arguments into a validated frozen configuration and dispatches a worker, and a container path that hydrates the corpus, invokes the atomic VOCODE command-line interface, and commits the volume. Cloud execution adds transport and resources and never changes behavior.

| Module | Contents |
|---|---|
| `application.py` | The cloud infrastructure defined once: the Modal application `vocode-checkpoint01`, the volume `vocode-data` (created automatically when missing), and the container image. The launch files import these objects rather than constructing their own. |
| `runtime.py` | `ModalVocodeImageBuilder`, `LJSpeechCorpusPreparer`, and `ModalPythonPathConfigurator`. |
| `train.py` | The local entrypoint `launch_training` for training, recovery, and evaluation dispatch across all lanes. |
| `published.py` | The local entrypoints `launch_published_evaluation` and `verify` for author-weight evaluation on the fixed B200 lane. |

## The Image and the Volume

`ModalVocodeImageBuilder` builds a Debian-slim Python 3.14 image: `git`, `wget`, and `build-essential` through apt (the compiler serves the `pesq` source build), the pinned dependency set (`torch==2.13.0`, `torchaudio==2.11.0`, `torchao==0.17.0`, `onnx==1.20.1`, `onnxruntime==1.24.4`, and the metric stack, identical to the repository `requirements.txt`), and `TORCH_HOME` pointed at `/data/pretrained/torch`. The image copies `code/vocode` and `code/syntheticmind` into `/workspace` and copies `runtime.py` and `application.py` beside the launch files, so every capsule executes exactly the code state that spawned it; any source change invalidates the copy layers on the next launch. The Python pins define the aligned measurement stack, while the Debian base and host driver were not digest-pinned in the retained runs.

`LJSpeechCorpusPreparer` hydrates the corpus cloud-side on first use: when `/data/datasets/LJSpeech-1.1` is absent, the container downloads the archive from `https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2` and extracts it into the volume, so local storage never becomes a hidden dependency. Author-released weights resolve under `/data/pretrained`: the weight adapters retrieve their recorded releases at run time, and the root can also be pre-seeded explicitly with `modal/storage.py upload`. No Modal secrets are required by any job.

## Training and Recovery Dispatch

`launch_training` parses roughly thirty-five flags into a frozen configuration validated twice (argparse choices, then the strict Pydantic model). The essential surface:

| Flag | Default | Meaning |
|---|---|---|
| `--architecture-name` | `hifigan_v1` | One of the thirteen registered architectures. |
| `--stage`, `--dataset-split-name` | `train`, `train` | Stage and split must agree; validation and test stages require `--project-checkpoint-path`. |
| `--hardware-name`, `--precision-name` | `b200`, `fp32` | The resource lane and the numeric precision. |
| `--seed` | `42` | The run seed. |
| `--run-id` | UTC timestamp | The capsule leaf directory; generated as `%Y%m%dT%H%M%S` when omitted. |
| `--optimization-variant` | none | Selects the optimized-variants lane; with `--stage train` it maps to `recover-optimized-variant`, otherwise to `evaluate-optimized-variant`. |
| `--evidence-category` | `project_trained_reproduction` | `project_hybrid_variants` exposes the train and test stages only. |
| `--train-epoch-count` | `200` | The epoch ceiling of training stages. |
| `--metric` | `pesq, stoi, mel, rtf, parameters, size` | Repeatable; supplying any metric replaces the default panel. Fourteen names are available. |
| `--limit-train-batches` and the other three limits | none | Bounded verification controls; a bounded run is excluded from automatic resume. |
| `--spawn-remote` | attached | Spawns the call and logs the function-call identifier and dashboard URL instead of blocking. |

The stage selector maps onto the atomic commands `train-reproduction`, `validation-reproduction`, `test-reproduction`, `train-hybrid`, `test-hybrid`, `recover-optimized-variant`, and `evaluate-optimized-variant`. The parser records the launching repository commit automatically (`git rev-parse --short=12 HEAD`, suffixed `-dirty` when the code tree has uncommitted changes) so container-side records carry code provenance. Launch defaults are launch conveniences, not study values; the study's executed settings are recorded in each capsule's `resolved_config.yaml`.

One shared single-use worker class serves every lane: the base declaration is CPU-only at 8 CPUs, 16 GiB, and an 86,400-second timeout, and the lane profile is applied at dispatch through `Cls.with_options`. Because one Modal attempt is capped at 24 hours, long full-scale training runs resume automatically: the resume resolver looks for `last.ckpt`, then the newest `checkpoint-*.ckpt`, under the run's own `checkpoints/` directory and re-launches from it; runs with any batch limit are treated as bounded verification and never resume automatically.

## Published-Weight Dispatch

`launch_published_evaluation` and `verify` serve the nine architectures with author releases (`hifigan_v1`, `hifigan_v2`, `hifigan_v3`, `melgan`, `vocos`, `bigvgan`, `apnet2`, `freev`, `hiftnet`). Both build a `test-published` invocation pinned to `--hardware-name b200 --precision-name fp32 --accelerator cuda` on a single-use worker fixed to a B200, 16 CPUs, 64 GiB, and a 7,200-second timeout. The evaluation path defaults to one timed repetition per utterance (`--rtf-repetition-count 1`), and `verify` prepends a bounded preflight profile capped at 50 test utterances. The flag surface mirrors the training parser's evaluation subset, including the replaceable `--metric` panel and `--spawn-remote`.

## Requested Resource Profiles

| `--hardware-name` | GPU | CPU units | Host memory |
|---|---|---:|---:|
| `b200` | B200 | 16 | 64 GiB |
| `h100` | H100 | 16 | 64 GiB |
| `a100_80gb` | A100-80GB | 16 | 64 GiB |
| `l40s` | L40S | 8 | 32 GiB |
| `cpu` | none | 8 | 16 GiB |

## Invocation

Both launch files pass `--dataset-root /data/datasets/LJSpeech-1.1` and `--published-weights-root /data/pretrained` into the atomic command; `--artifact-root` defaults to `/data/experiment_artifacts` and is validated to lie under `/data`. Image copy layers resolve against the launching working directory, so `modal run` is issued from the repository root, the directory containing `code/`.

```bash
modal run --detach code/modal/vocode/train.py::launch_training --architecture-name vocos \
    --hardware-name l40s --precision-name bf16 --run-id vocos_full_bf16bs32 --spawn-remote

modal run code/modal/vocode/published.py::verify --architecture-name hifigan_v1
```
