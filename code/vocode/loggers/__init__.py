# Package initializer for the experiment evidence writers; re-exports the
# ExperimentLogger entry point.
#
# The package divides into the per-run capsule writers and the shared index.
# RunTracker (tracker) owns a run's lifecycle record and writes the resolved
# configuration, manifest, and status; ExperimentLogger (experiment) buffers
# metrics and writes the metric artifacts; CheckpointLogger (checkpoint)
# records which checkpoints a run kept and why. ExperimentResultRow (result)
# is the schema authority for the shared summary, and
# ExperimentResultWriter (writer) appends to it under an exclusive lock.
#
# Only the logger entry point is re-exported here; the tracker, checkpoint
# logger, row record, and writer are imported from their own modules by the
# runners and by ExperimentLogger itself.
#
# Author: Rahul Sawhney

from vocode.loggers.experiment import ExperimentLogger

__all__: list[str] = ["ExperimentLogger"]
