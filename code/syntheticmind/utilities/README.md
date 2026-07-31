# syntheticmind.utilities: The Framework Utilities

`utilities/` collects the framework's shared infrastructure helpers. Each module is small and single-purpose, and nothing here depends on the trainer, so the utilities are usable from any layer.

| Module | Contents |
|---|---|
| `seed.py` | `SeedManager`, deterministic seeding across Python, NumPy, and PyTorch, including worker seeding for dataloaders. The study's recorded training seed and its evaluation seeds pass through this module. |
| `checkpoint.py` | `save_checkpoint` and `load_checkpoint`, the persistence of the `CheckpointState` contract from `../state/`. |
| `data_transfer.py` | `move_data_to_device`, recursive device transfer for tensors and arbitrarily nested containers. |
| `distributed.py` | `DistributedUtils` and the rank helpers (`get_rank`, `get_local_rank`, `get_world_size`, `is_distributed`, `is_rank_zero`, `barrier`, `all_reduce`, and `rank_zero_only`), which degrade gracefully to single-process semantics. |
| `metric_accumulator.py` | `MetricAccumulator`, the step-level and epoch-level metric aggregation behind the `Module.log` contract. |
| `optimizers.py` | `build_optimizer`, the construction of concrete optimizers from the declarative configuration in `../core/optimizer.py`. |
| `schedulers.py` | `build_scheduler`, the corresponding construction of learning-rate schedulers. |
| `compile.py` | `apply_compile` and `apply_channels_last`, model compilation and memory-format helpers. |
| `exceptions.py` | `FrameworkError` and its family: `MisconfigurationError`, `TrainingInterruptedError`, `CheckpointError`, and `DistributedError`, the framework's explicit error vocabulary. |
| `types.py` | `DeviceTransferable`, the structural type contract used by device transfer. |

## Related Components

The trainer in `../core/`, the loops in `../loops/`, and the callbacks in `../callbacks/` consume these helpers; none of the helpers imports from those packages in return.
