# syntheticmind.state: The Trainer and Checkpoint State

`state/` defines the serializable state containers that make runs resumable. Everything a checkpoint restores is declared here, so resumability is a property of explicit state objects rather than of scattered attributes.

| Module | Contents |
|---|---|
| `trainer_state.py` | `TrainerState`, the live progress of a run: the current stage, the epoch and global-step counters, and the stopping status. The loops advance it, and checkpoint save serializes it. |
| `checkpoint_state.py` | `CheckpointState`, the complete durable state of a run: model parameters, optimizer and scheduler states, callback states, datamodule state, trainer progress counters, and the random-number-generator state for Python, NumPy, and PyTorch. |

Restoring a `CheckpointState` continues a run from its last durable point with the same optimization trajectory and the same randomness stream.

## Related Components

The persistence functions live in `../utilities/checkpoint.py`. The durable training checkpoints of the study and its Retained Project Checkpoints are instances of this state contract.
