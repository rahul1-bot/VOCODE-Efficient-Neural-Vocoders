# syntheticmind.callbacks: The Lifecycle Callbacks

`callbacks/` extends the trainer through a uniform hook interface. Callbacks observe and act on a run without living inside model or loop code, and callback state participates in checkpoints, so resumable runs restore callback progress alongside model state.

| Module | Contents |
|---|---|
| `callback.py` | `Callback`, the abstract hook surface every callback implements. |
| `model_checkpoint.py` | `ModelCheckpoint`, which writes durable checkpoints on a configured cadence with monitored-metric selection; this is the mechanism behind the study's registered checkpoint gate. |
| `early_stopping.py` | `EarlyStopping`, which stops a run on a monitored metric with patience. |
| `timer.py` | `TimerCallback`, which enforces wall-time budgets. |
| `watchdog.py` | `WatchdogCallback`, which detects stalled runs; both timing callbacks serve unattended cloud execution. |
| `nan_inf_guard.py` | `NanInfGuard`, which halts on non-finite losses or gradients instead of training through numerical corruption. |
| `data_leakage_guard.py` | `DataLeakageGuard`, which verifies split integrity between stages. |
| `ema.py` | `EMACallback`, which maintains an exponential moving average of model weights. |
| `lr_monitor.py` | `LearningRateMonitor`, which records per-group learning rates. |
| `progress_bar.py` | `ProgressBarCallback`, which reports stage progress. |
| `throughput_monitor.py` | `ThroughputMonitor`, which records samples-per-second telemetry. |
| `device_stats_monitor.py` | `DeviceStatsMonitor`, which records device utilization. |
| `runtime_profiler.py` | `RuntimeProfiler` with `RuntimeMetricAccumulator`, which produces runtime profiling artifacts for post-run inspection. |

## Related Components

Callbacks are attached through the trainer in `../core/trainer.py`, and their persisted state travels inside the checkpoint contract defined in `../state/`.
