# vocode: The Study Client

`vocode/` implements the VOCODE study on top of the `syntheticmind` training framework. It contains the vocoder architectures, their training objectives, the LJSpeech data pipeline, the measurement stack, the deployment transformations, and the evidence-writing execution roles. The package contains no measured results; it produces the run records from which `../../artifacts/` was assembled.

| Subpackage | Contents |
|---|---|
| `configs/` | The validated experiment configuration and the artifact capsule layout. |
| `data/` | The LJSpeech dataset, the deterministic identifier-sorted partition, and the DataModule. |
| `loggers/` | Persistence of run, result, and checkpoint evidence into capsules and summary tables. |
| `losses/` | The architecture-native training objectives and their shared adversarial components. |
| `metrics/` | The atomic quality, timing, and footprint measurements with ordered selection. |
| `models/` | Eleven architecture directories covering the thirteen registered configuration names. |
| `optimization/` | The deployment transformations and the architecture-to-technique registry. |
| `trainers/` | The three execution roles: reproduction training, published-weight evaluation, and optimized-variant evaluation. |
| `transforms/` | The mel-spectrogram and LPCNet feature-extraction signal paths. |

## The Command-Line Interface

`cli.py` is the single execution entry point of the package. It defines the closed architecture vocabulary in `VocodeCliArchitectureSet` and the layered configuration resolution: declared defaults are read first, an optional YAML file is loaded through `VocodeCliYamlLoader`, repeatable `--override` pairs are applied through `VocodeCliOverrideParser`, explicit flags are applied last, and `VocodeCliConfigurationFactory` combines the layers into one validated configuration.

The interface exposes eight atomic subcommands, each bound to its evidence category. The subcommands `train-reproduction`, `validation-reproduction`, and `test-reproduction` serve project-trained reproduction; `train-hybrid` and `test-hybrid` serve the project hybrid variants; `test-published` serves author-released checkpoint evaluation; and `evaluate-optimized-variant` and `recover-optimized-variant` serve the optimized-variants lane. One invocation resolves one configuration, executes one architecture, seed, and stage, and writes one artifact capsule.

## Related Components

Every module in this package is mirrored by a test module at the same relative path under `../tests/vocode/`. The evidence written by this package is curated into the three study packages under `../../artifacts/`, whose registers the report cites.
