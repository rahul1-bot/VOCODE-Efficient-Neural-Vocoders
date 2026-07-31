# This module:
# 1. Defines FrameworkError, the root exception type for every failure the harness raises
# 2. Defines the concrete failure categories: MisconfigurationError for invalid caller
#    configuration, TrainingInterruptedError for user-initiated interruption,
#    CheckpointError for save and restore failures, and DistributedError for
#    process-group failures
# 3. Normalizes error text by stamping the category prefix onto the message at
#    construction time
#
# Design decisions:
# - One root type lets callers isolate harness failures with a single except clause
#   while still allowing category-precise handling where it matters
# - Category prefixes are applied in the constructors so every raise site produces
#   consistently labeled text without repeating the prefix by hand
# - TrainingInterruptedError carries a default message because interruption sites
#   usually have no extra detail beyond the fact of the interruption itself
#
# Author: Rahul Sawhney

__all__: list[str] = [
    "FrameworkError",
    "MisconfigurationError",
    "TrainingInterruptedError",
    "CheckpointError",
    "DistributedError",
]


class FrameworkError(Exception):
    # Root of the harness exception hierarchy. Catching this type isolates harness
    # failures from arbitrary Python errors raised by model or data code.
    pass


class MisconfigurationError(FrameworkError):
    # Raised when caller-supplied configuration is invalid or contradictory, for
    # example an unknown monitor key, a missing required hook, or an option
    # combination the harness refuses to run.
    def __init__(self, message: str) -> None:
        # Stamps the misconfiguration prefix so the failure category is visible in logs.
        super().__init__(f"Misconfiguration: {message}")


class TrainingInterruptedError(FrameworkError):
    # Raised when a run is cut short deliberately, typically translated from
    # KeyboardInterrupt so teardown can distinguish interruption from failure.
    def __init__(self, message: str = "Training was interrupted") -> None:
        # Accepts an optional detail message; the default covers the common case
        # where interruption carries no additional context.
        super().__init__(message)


class CheckpointError(FrameworkError):
    # Raised when writing or restoring a checkpoint fails, including payload
    # schema problems discovered while rehydrating component state.
    def __init__(self, message: str) -> None:
        # Stamps the checkpoint prefix so the failure category is visible in logs.
        super().__init__(f"Checkpoint error: {message}")


class DistributedError(FrameworkError):
    # Raised for distributed-execution failures such as process-group
    # initialization problems or collective-communication misuse.
    def __init__(self, message: str) -> None:
        # Stamps the distributed prefix so the failure category is visible in logs.
        super().__init__(f"Distributed error: {message}")
