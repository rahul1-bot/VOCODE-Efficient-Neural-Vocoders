# Package initializer for the RFWave multi-band rectified-flow vocoder, a
# faithful reimplementation of the ICLR 2025 reference in which no author
# code is executed or linked; modules are imported by full path.
#
# This package:
# 1. Groups the four modules of the family: the backbone network and its
#    spectral and equalization collaborators, the rectified flow that owns
#    the subband geometry and the path construction, the Euler sampler that
#    integrates the learned field, and the harness module tying them together
#
# Design decisions:
# - The export list is deliberately empty and no submodule is imported here.
#   Importing the package therefore costs nothing and pulls in no dependency,
#   which matters because the registry imports model modules by path and must
#   not pay for families a given run never constructs
# - Keeping this file inert also means the import graph is visible at every
#   call site: a reader sees which module a symbol comes from rather than a
#   flattened package namespace that hides it
#
# Author: Rahul Sawhney

__all__: list[str] = []
