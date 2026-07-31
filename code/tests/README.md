# tests: The Permanent Test Suite

`tests/` is the permanent pytest suite of the project. It collects 2,202 tests plus 42 subtests, and every test executes offline. The suite mirrors `../vocode/` directory for directory and file for file, so the tests for any module sit at the same relative path under `tests/vocode/`. The configuration file `../pytest.ini` declares every Python file under this tree a test module, resolves imports through `--import-mode=importlib`, and sets `pythonpath = .` so that `vocode` and `syntheticmind` resolve without environment variables.

Three disciplines keep the suite runnable anywhere. First, the suite is offline: no test downloads a checkpoint, a corpus, or a predictor, author-release adapters are tested against fabricated archives with self-consistent SHA-256 digests, and dataset tests construct their corpus fixtures on disk. Second, inputs are constructed: waveforms, spectrograms, and conditioning features are synthesized tensors with controlled shapes and values, so behavior is asserted against known inputs rather than recorded data. Third, every test executes on the CPU, so no GPU is required.

| Directory | Tests | Coverage |
|---|---:|---|
| `vocode/cli.py` | 77 | The atomic command-line interface. |
| `vocode/configs/` | 72 | The experiment configuration and the artifact layout. |
| `vocode/data/` | 83 | The LJSpeech dataset, partition, and DataModule. |
| `vocode/loggers/` | 105 | Evidence persistence. |
| `vocode/losses/` | 234 | All thirteen objective modules. |
| `vocode/metrics/` | 360 | The complete measurement stack. |
| `vocode/models/` | 73 | The registry and the shared vocoder contracts. |
| `vocode/models/<architecture>/` | 782 | The eleven architecture directories. |
| `vocode/optimization/` | 208 | The deployment transformations and the variant registry. |
| `vocode/trainers/` | 123 | The three execution roles. |
| `vocode/transforms/` | 85 | The mel and LPCNet feature paths. |

The suite executes from `code/`:

```bash
python -m pytest                              # the full suite
python -m pytest tests/vocode/optimization    # one directory
python -m pytest tests/vocode/models/vocos    # one architecture
```
