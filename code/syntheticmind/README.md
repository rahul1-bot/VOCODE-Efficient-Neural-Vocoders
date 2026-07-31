# syntheticmind: The PyTorch Training Framework

`syntheticmind/` is an independent PyTorch research framework and the training substrate of the VOCODE study. It is study-agnostic by design: nothing in this package references vocoders, audio, or the `vocode` package, so the framework boundary separates reusable training machinery from experiment-specific logic.

| Subpackage | Contents |
|---|---|
| `core/` | The `Trainer`, the `Module` and `DataModule` contracts, the hook interfaces, the optimization configuration, and the checkpoint hooks. |
| `loops/` | The fit, training-epoch, evaluation, and prediction loops the trainer dispatches to. |
| `accelerators/` | The device abstraction covering CPU, CUDA, and Apple MPS. |
| `strategies/` | The execution-strategy boundary covering single-device and distributed data-parallel execution. |
| `callbacks/` | The lifecycle callbacks: checkpointing, early stopping, exponential moving averages, monitoring, guarding, and profiling. |
| `loggers/` | The metric-logger interface with CSV and TensorBoard implementations. |
| `state/` | The serializable trainer and checkpoint state containers. |
| `utilities/` | Seeding, checkpoint persistence, device transfer, distributed helpers, optimizer and scheduler construction, and the exception vocabulary. |

## Contracts

The public surface follows established PyTorch research vocabulary. Models subclass `Module`, which is itself a `torch.nn.Module`, and express behavior through `forward`, `training_step`, `validation_step`, `test_step`, `predict_step`, and `configure_optimizers`. Metrics are emitted through `Module.log` and `Module.log_dict` with explicit step and epoch reduction, logger visibility, and distributed-synchronization semantics. Data behavior is expressed through the `DataModule` methods `prepare_data`, `setup`, `teardown`, and the per-stage dataloader constructors.

`Trainer` orchestrates the fit, validate, test, and predict stages under parameters covering epoch ceilings, accelerator and strategy selection, precision, gradient accumulation and clipping, validation cadence, sanity validation, per-stage batch limits, deterministic seeding, and checkpoint resume. Manual optimization through `manual_backward`, `optimizer_step`, `toggle_optimizer`, and `untoggle_optimizer` supports multi-optimizer research models; the adversarial vocoders of this study use this path for their alternating generator and discriminator updates.

## Checkpoints

Checkpoints are native and resumable. A durable checkpoint carries model, optimizer, scheduler, callback, datamodule, epoch, step, and random-number-generator state, so an interrupted run continues from its last durable state rather than restarting. Every training trajectory, evaluation pass, and recovery arm of the VOCODE study executed through this framework.

## Related Components

Each subpackage carries its own `README.md`. The study client in `../vocode/` layers on this framework, and the test suite under `../tests/` exercises the framework through that client.
