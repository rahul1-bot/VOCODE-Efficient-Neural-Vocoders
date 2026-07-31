# This package:
# 1. Holds one subpackage per vocoder architecture of the study, each
#    pairing the harness module with its network, its discriminators, and
#    its published-weights adapter where an author release exists
# 2. Re-exports the construction authority (ModelRegistry and its
#    implementation record) and the shared architecture vocabulary: the
#    closed architecture-name literal, the two structural protocols, and
#    the published-weight provenance record
#
# Design decisions:
# - The re-export set declares the package's public vocabulary, but the
#   codebase imports by full module path throughout, so this initializer
#   is documentation of the surface rather than the route callers take;
#   the build-option and built-module records deliberately stay behind
#   the registry module because they are construction inputs and outputs
#   rather than vocabulary
# - Every architecture is reached through the registry rather than
#   constructed directly, so the evidence lanes cannot diverge on how a
#   module was built
#
# Author: Rahul Sawhney

from vocode.models.registry import ArchitectureImplementationRecord, ModelRegistry
from vocode.models.vocoder import ArchitectureName, PublishedWeightProvenance, PublishedWeights, Vocoder

__all__: list[str] = [
    "ArchitectureName",
    "ArchitectureImplementationRecord",
    "ModelRegistry",
    "PublishedWeightProvenance",
    "PublishedWeights",
    "Vocoder"
]
