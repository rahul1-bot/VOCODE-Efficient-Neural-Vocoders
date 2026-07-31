# Package initializer for the experiment configuration records; modules
# are imported by full path.
#
# This package holds the two frozen records that describe a run before
# anything executes:
# 1. layout: the closed label domains of the experiment tree and
#    ExperimentArtifactLayout, which computes every artifact path of one
#    run capsule from its identity fields
# 2. run: ExperimentConfiguration, the single record a runner receives,
#    which validates the batch-limit domain, the stage-to-split binding,
#    and the provenance and device requirements of optimized-variant runs
#
# The export list is deliberately empty, so importing the package pulls in
# no module and the two records stay addressable only by their full paths.
#
# Author: Rahul Sawhney

__all__: list[str] = []
