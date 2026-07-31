# This module:
# 1. Defines the checkpoint hook pair through which modules contribute state to
#    checkpoint payloads and recover it on restore
#
# Design decisions:
# - The hooks receive the complete mutable payload dictionary rather than a
#   private sub-mapping, so a module can both add its own entries and inspect
#   what the harness has already recorded
# - Both defaults are no-ops, keeping the hooks strictly opt-in for modules
#   that carry state beyond their parameters and buffers
# - The pair is defined in its own class so Module composes it through
#   inheritance alongside the model and data hook families without coupling
#   checkpoint concerns to either
#
# Author: Rahul Sawhney

from syntheticmind.utilities.types import CheckpointDict

__all__: list[str] = ["CheckpointHooks"]


class CheckpointHooks:
    # Hook pair invoked by the trainer around checkpoint serialization. The
    # save hook runs after the harness has assembled the standard payload, and
    # the load hook runs after the standard payload has been restored.
    def on_save_checkpoint(self, checkpoint: CheckpointDict) -> None:
        # Invoked while a checkpoint payload is being assembled, immediately
        # before it is written. Implementations may insert additional entries
        # into the payload dictionary in place.
        pass

    def on_load_checkpoint(self, checkpoint: CheckpointDict) -> None:
        # Invoked after a checkpoint payload has been read and the standard
        # state has been restored. Implementations may read back the entries
        # they contributed during saving.
        pass
