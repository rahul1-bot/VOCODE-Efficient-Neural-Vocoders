# Modal Cloud Scheduling

`modal/` is the cloud-scheduling and storage-integration boundary of the project. It holds no model, training, metric, or artifact-schema logic. Every cloud job it starts is one invocation of the atomic VOCODE command-line interface executed inside a Modal container, so a scheduled job and a local shell command carry the same evidence semantics. Each dispatch is one isolated job on a single-use container, so configurations, variants, lanes, and executions are scheduled independently of one another.

## Prerequisites

Cloud execution requires a Modal account and the Modal client: `pip install modal`, then `modal setup` to authenticate. Nothing else is needed: the volume `vocode-data` is created automatically on first use, the LJSpeech corpus is downloaded into it by the first job that needs it, author-weight retrieval happens at run time, and no Modal secrets are used by any job.

## Contents

| Entry | Contents |
|---|---|
| `vocode/` | The Modal launch surfaces: the shared application, volume, and image definition, the training and recovery dispatcher, and the author-weight evaluation dispatcher. Documented in `vocode/README.md`. |
| `storage.py` | Volume inventory and upload operations. |
| `sync.py` | Pull-based synchronization of completed run capsules. |

`storage.py` defines `ModalStorageClient`, which wraps `modal volume ls --json` and `modal volume put` as subprocess calls, validates each listing through explicit models, and can persist a structured inventory report as JSON evidence. `ModalStorageCli` exposes the subcommands `inventory` and `upload`; `inventory` inspects `/datasets`, `/pretrained`, and `/experiment_artifacts` by default, `upload` seeds volume content (for example, pre-staging author weights under `/pretrained`), and both accept `--environment-name` for non-default Modal environments. The `modal` executable is resolved from `PATH`, and its absence is an explicit error rather than a silent skip.

`sync.py` defines `ModalArtifactSynchronizer`, which wraps `modal volume get`, resolves the destination as the remote basename under the given root, and leaves an existing destination untouched unless `--overwrite-existing-local-files` is passed. Remote paths are validated as bounded, so empty, current-directory, and parent-directory components are rejected. Synchronization is deliberately pull-based and explicit: nothing downloads automatically.

## The Volume Contract

Both utilities and every cloud job address the Modal volume `vocode-data`, which containers mount at `/data`. The three maintained roots are `datasets` for the LJSpeech corpus, `pretrained` for author-released weights and framework caches, and `experiment_artifacts` for run output capsules. The volume commands address these roots as `/datasets`, `/pretrained`, and `/experiment_artifacts`, and container code sees them under `/data`.

## Command Shapes

```bash
python code/modal/storage.py inventory --volume-name vocode-data \
    --remote-path /datasets --remote-path /pretrained --remote-path /experiment_artifacts \
    --output-path modal_vocode_data_inventory.json

python code/modal/storage.py upload --volume-name vocode-data \
    --local-path pretrained --remote-path /pretrained --force

python code/modal/sync.py --volume-name vocode-data \
    --remote-relative-path experiment_artifacts/project_trained_reproduction \
    --local-destination-root experiment_artifacts
```
