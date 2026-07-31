# syntheticmind.loggers: The Metric Loggers

`loggers/` receives the metrics a `Module` emits through `log` and `log_dict` and persists them outside the process. Loggers are passive consumers: reduction and cadence are decided by the logging contract in `../core/hooks.py`, so every attached logger records the same values.

| Module | Contents |
|---|---|
| `logger.py` | `Logger`, the abstract contract covering hyperparameter recording and step-keyed metric recording. |
| `csv_logger.py` | `CSVLogger`, append-only CSV persistence; this is the format consumed by the study's training-history records. |
| `tensorboard.py` | `TensorBoardLogger`, which writes TensorBoard event files through the internal `_TensorBoardWriter`. |

Multiple loggers may be attached to one trainer, and each receives every logged value.

## Related Components

The training histories preserved under `../../../artifacts/study_1/` originate from the CSV logger output of the study's training runs.
