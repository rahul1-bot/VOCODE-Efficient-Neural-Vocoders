# syntheticmind.core: The Trainer and the Contracts

`core/` defines the orchestration center of the framework and the two contracts user code implements.

| Module | Contents |
|---|---|
| `trainer.py` | `Trainer`, the orchestration entry point. |
| `module.py` | `Module`, the model contract, with `MetricLoggingConfiguration` and `LoggedMetric`. |
| `datamodule.py` | `DataModule`, the data contract, and `DatasetDataModule`. |
| `hooks.py` | `DataHooks`, `ModelHooks`, `MetricLoggingRule`, and `MetricLoggingContract`. |
| `optimizer.py` | `OptimizerConfig`, `SchedulerConfig`, and `OptimizationConfiguration`. |
| `saving.py` | `CheckpointHooks`, the hook surface around checkpoint save and load. |

`Trainer` binds the module, the datamodule, the accelerator, the strategy, the callbacks, and the loggers; dispatches the fit, validate, test, and predict stages to the loops in `../loops/`; owns checkpoint save and restore through the state objects in `../state/`; and manages evaluation sampling, where `_UnrepeatedDistributedSampler` prevents sample duplication under distributed evaluation.

`Module` defines the model contract: forward computation, the per-stage step methods, optimizer declaration through `configure_optimizers`, and structured metric logging. `DataModule` defines the data contract through per-stage dataloader construction, and `DatasetDataModule` is the direct dataset-backed implementation. The hook modules define the lifecycle surfaces available to models and datamodules and the rules for where logged metrics are reduced and consumed. The optimizer module defines the declarative optimization setup a module returns, which `../utilities/` resolves into concrete optimizers and schedulers.

## Related Components

The trainer executes any `Module` and `DataModule` pair without knowledge of the domain; every VOCODE architecture and dataset reaches it through these contracts via the execution roles in `../../vocode/trainers/`.
