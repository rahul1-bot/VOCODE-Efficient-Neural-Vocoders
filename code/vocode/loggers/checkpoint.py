# This module:
# 1. Writes the per-run checkpoint manifest after training: which checkpoint
#    files exist, which one is best, under what monitor and direction they
#    were selected, and the full run identity that produced them
#
# Harness contract (syntheticmind):
# - The manifest is derived from the live ModelCheckpoint callback after fit
#   returns: its best-path attribute and its monitor and mode settings are
#   read directly, so the manifest reports what the harness actually kept
#   rather than what the configuration requested
#
# Report alignment:
# - The best path recorded here is the callback's monitor-ranked selection.
#   It is not by construction the Retained Project Checkpoint, which the
#   report defines as the terminal durable state admitted by the registered
#   gate and explicitly not a retrospectively substituted validation
#   minimum. Recording the monitor, the direction, the best path, and the
#   last path together is what lets a reviewer tell the validation-best
#   state from the terminal state that evaluation actually inherits
#
# Design decisions:
# - The last-checkpoint entry is recorded only when last.ckpt exists on
#   disk, because a bounded verification run can finish without one and a
#   manifest must never point at an absent file
# - The manifest is JSON inside the checkpoints directory itself, so a
#   downloaded capsule carries its own checkpoint index
#
# Author: Rahul Sawhney

from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from syntheticmind.callbacks.model_checkpoint import ModelCheckpoint

__all__: list[str] = ["CheckpointEvidence", "CheckpointLogger", "CheckpointManifest"]

# The three evidence lanes that produce checkpoints. The label is recorded
# in the manifest so a downloaded checkpoint directory states which lane
# produced it, rather than leaving that to be inferred from its path. The
# trained-reproduction label covers the Study 1 budget-controlled training
# lane; the optimization-checkpoint label covers the Study 2 deployment
# transformation lane.
type CheckpointEvidence = Literal[
    "project_trained_reproduction",
    "project_optimization_checkpoint",
    "project_hybrid_variants"
]


class CheckpointManifest(BaseModel):
    # Frozen record of one run's checkpoint outcome: identity, directory,
    # best and last paths, and the selection criterion that ranked them.
    #
    # Fields:
    #     evidence_label: Which of the three evidence lanes produced these
    #         checkpoints.
    #     architecture_name: Vocoder family that was trained.
    #     variant_name: Variant label of the run.
    #     seed: Seed governing the run.
    #     unique_id: Stable per-run identifier, shared with the summary row
    #         and the run capsule's other artifacts, and therefore the join
    #         key from a checkpoint back to its evidence.
    #     experiment_name: Experiment the run belonged to.
    #     run_id: Identifier of the individual run.
    #     checkpoint_directory: Directory the checkpoints live in, which is
    #         also where this manifest is written.
    #     best_checkpoint_path: Path the callback ranked best under its own
    #         monitor and direction, or ``None`` when it never recorded one,
    #         which is the case for a run that completed no monitored
    #         validation pass. This is the validation-ranked state, not by
    #         construction the report's Retained Project Checkpoint.
    #     last_checkpoint_path: Path of the terminal durable checkpoint,
    #         recorded only when that file exists on disk and ``None``
    #         otherwise. Evaluation in this study inherits the terminal
    #         admitted state rather than the validation minimum, so this
    #         field and the best path answer different questions.
    #     monitor: Metric key the callback ranked checkpoints by.
    #     mode: Improvement direction for that metric, either ``"min"`` or
    #         ``"max"``.
    #     created_at_utc: UTC instant this manifest was assembled.
    #
    # The monitor and mode fields are what make the best path interpretable:
    # a path alone does not say what "best" meant, and the same run scored
    # under a different criterion would have kept a different file. Recording
    # the criterion alongside both paths is what keeps the validation-ranked
    # state and the terminal admitted state separable in the evidence. The
    # model is frozen, strict, and extra-forbidding, and permits arbitrary
    # types so Path and datetime can be fields; it serializes through
    # pydantic's JSON encoder, which renders both as strings.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True
    )
    evidence_label: CheckpointEvidence
    architecture_name: str
    variant_name: str
    seed: int
    unique_id: str
    experiment_name: str
    run_id: str
    checkpoint_directory: Path
    best_checkpoint_path: Path | None
    last_checkpoint_path: Path | None
    monitor: str
    mode: Literal["min", "max"]
    created_at_utc: datetime


class CheckpointLogger:
    # Writer producing the checkpoint manifest from the post-fit state of
    # the harness ModelCheckpoint callback. The class is stateless and holds
    # no configuration, so one instance serves any number of runs and the
    # entire contract lives in the single write call.
    #
    # Integration: a training runner calls write once, after
    # trainer.fit has returned, passing the same callback instance it gave
    # the trainer. Calling it before fit completes would record a
    # provisional best path as final.
    def write(
        self,
        model_checkpoint: ModelCheckpoint,
        run_directory: Path,
        evidence_label: CheckpointEvidence,
        architecture_name: str,
        variant_name: str,
        seed: int,
        unique_id: str,
        experiment_name: str,
        run_id: str
    ) -> CheckpointManifest:
        # Assembles the manifest from the callback's observed outcome (best
        # path, monitor, direction) and the run identity, records last.ckpt
        # only if it exists, and writes the manifest JSON into the
        # checkpoints directory.
        #
        # The selection fields are read from the live callback rather than
        # from the configuration that created it, so the manifest reports the
        # criterion the harness actually applied. The last-checkpoint entry
        # is probed on disk instead of being assumed, because a short run can
        # finish without one and a manifest that pointed at an absent file
        # would be worse than one that admits the file is missing.
        #
        # Args:
        #     model_checkpoint: The callback the trainer used, read after
        #         fit has returned for its best path, monitor, and mode.
        #     run_directory: Root of the run capsule; the checkpoints
        #         subdirectory beneath it holds the files and receives this
        #         manifest.
        #     evidence_label: Which evidence lane produced the run.
        #     architecture_name: Vocoder family that was trained.
        #     variant_name: Variant label of the run.
        #     seed: Seed governing the run.
        #     unique_id: Stable per-run identifier linking these
        #         checkpoints to the run's other evidence.
        #     experiment_name: Experiment the run belonged to.
        #     run_id: Identifier of the individual run.
        #
        # Returns:
        #     The manifest that was written, so a caller can assert on the
        #     recorded outcome without reparsing the file. The file itself
        #     is written as indented JSON at
        #     ``checkpoints/project_checkpoint_manifest.json``, a fixed
        #     location so a downloaded capsule carries its own checkpoint
        #     index.
        checkpoint_directory: Path = run_directory / "checkpoints"
        last_checkpoint_path: Path = checkpoint_directory / "last.ckpt"
        manifest: CheckpointManifest = CheckpointManifest(
            evidence_label=evidence_label,
            architecture_name=architecture_name,
            variant_name=variant_name,
            seed=seed,
            unique_id=unique_id,
            experiment_name=experiment_name,
            run_id=run_id,
            checkpoint_directory=checkpoint_directory,
            best_checkpoint_path=model_checkpoint.best_path,
            last_checkpoint_path=last_checkpoint_path if last_checkpoint_path.exists() else None,
            monitor=model_checkpoint.monitor,
            mode=model_checkpoint.mode,
            created_at_utc=datetime.now(tz=timezone.utc)
        )
        manifest_path: Path = checkpoint_directory / "project_checkpoint_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return manifest
