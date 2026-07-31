# tests.vocode: The Study Client Tests

`tests/vocode/` mirrors the `vocode` package. Its subdirectories follow the source layout exactly (`configs/`, `data/`, `loggers/`, `losses/`, `metrics/`, `models/`, `optimization/`, `trainers/`, and `transforms/`), and each carries its own `README.md`.

One test module sits at this level. `cli.py` holds 77 tests for `vocode/cli.py`, the atomic command boundary. The tests cover the eight subcommands with their bound evidence categories, stages, and default splits; the layered configuration resolution across declared defaults, the YAML file, repeatable overrides, and explicit flags; the required-flag contract of each lane; override validation, in which an unknown key fails rather than being ignored; and the closed architecture vocabulary.

The command-line tests execute no training and download nothing. They drive argument parsing and configuration resolution to the validated-configuration boundary and assert on the resolved objects and on explicit failures.
