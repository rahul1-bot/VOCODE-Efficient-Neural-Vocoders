# vocode.configs: Experiment Configuration and Artifact Layout

`configs/` defines the two configuration surfaces every execution passes through: the validated run configuration and the derived artifact capsule layout. No other module constructs capsule paths or interprets raw run settings.

| Module | Contents |
|---|---|
| `run.py` | `ExperimentConfiguration`, the validated model of one atomic run. |
| `layout.py` | `ExperimentArtifactLayout`, the derivation of every on-disk capsule path. |

`ExperimentConfiguration` carries the complete identity and settings of one run: the experiment name, run identifier, seed, hypothesis fields, and code commit hash; the architecture under measurement; the evidence category and stage; the dataset roots and partition sizes; the per-stage batch sizes and dataloader worker policy; the trainer controls; the optimization variant where one applies; and the closed hardware and precision lane labels. Values are validated at construction, so an invalid setting fails before any execution begins, and unknown override keys are rejected rather than ignored.

`ExperimentArtifactLayout` derives the on-disk capsule location from the validated configuration: the artifact root, then the evidence category, the dataset label, the hardware-precision directory (`nvidia_<hardware>_<precision>` on datacenter lanes, hardware-only labels such as `nvidia_b200` or `modal_cpu8` in the optimized-variants lane, where precision is the experimental variable), the architecture, the variant segment exactly for the optimized-variants lane, and finally `runs/<run-id>`. The seed deliberately takes no part in path composition and is recorded inside the capsule as provenance. The layout also derives the capsule-internal locations for checkpoints, logs, metrics, and seed records, together with the summary table of the surrounding context directory (`experiments.csv`, or the versioned `experiments_v2.csv` for optimized variants). Every path the loggers write is produced here, so no runner or logger ever composes a path by hand.

## Related Components

The configuration reaches this package through `vocode/cli.py`, which layers declared defaults, the optional `--config` YAML file, repeatable `--override key=value` pairs, and explicit flags before validation. The resolved configuration is persisted verbatim into each capsule as `resolved_config.yaml`, so every run record under `../../../artifacts/` can be read against the exact settings that produced it. The mirrored tests live in `../../tests/vocode/configs/`.
