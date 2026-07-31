# syntheticmind.strategies: The Execution Strategies

`strategies/` owns process and model placement: how many processes participate in a run and how the model is wrapped for them. The strategy composes with the accelerator from `../accelerators/`, which owns the device itself.

| Module | Contents |
|---|---|
| `strategy.py` | `Strategy`, the abstract contract covering model setup, process coordination, and teardown. |
| `single_device.py` | `SingleDeviceStrategy`, one process on one device. Every run of the VOCODE study executed under this strategy, which keeps each atomic job isolated. |
| `ddp.py` | `DDPStrategy`, PyTorch DistributedDataParallel execution for multi-process runs launched through a configured distributed environment. |

## Related Components

The distributed helper functions for rank queries, barriers, and reductions live in `../utilities/distributed.py`.
