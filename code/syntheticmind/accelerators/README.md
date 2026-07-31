# syntheticmind.accelerators: The Device Abstraction

`accelerators/` isolates device-specific behavior behind one interface, so the trainer and the loops never branch on device type.

| Module | Contents |
|---|---|
| `accelerator.py` | `Accelerator`, the abstract device contract covering availability detection, device resolution, and device-scoped setup and teardown. |
| `cpu.py` | `CPUAccelerator`, the CPU implementation and the default when no hardware accelerator is requested. |
| `cuda.py` | `CUDAAccelerator`, the NVIDIA CUDA implementation used by the study's cloud training and B200 evaluation lanes. |
| `mps.py` | `MPSAccelerator`, the Apple Silicon Metal Performance Shaders implementation for local macOS execution. |

## Related Components

The accelerator is selected through the trainer's `accelerator` parameter, which the VOCODE command-line interface forwards as `--accelerator`. It composes with the strategy in `../strategies/`, which owns process placement.
