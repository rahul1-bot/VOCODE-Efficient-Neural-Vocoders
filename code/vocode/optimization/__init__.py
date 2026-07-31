# Package initializer for the Study 2 optimization techniques. The package
# transforms trained Study 1 models (quantization, compilation, pruning,
# export, sampling reduction) and never modifies the reproduction code
# path; modules are imported by full path.
#
# The package divides into three layers:
# 1. registry declares the closed variant vocabulary, decides applicability for
#    every (architecture, variant) cell with a recorded reason, and constructs
#    the technique instance for supported cells
# 2. compilation, quantization, pruning, sampling, and deployment implement the
#    techniques themselves, each mapping a harness Module onto a harness Module,
#    with export supplying the deployment lane its ONNX artifact production and
#    identity records
# 3. recovery runs the pruned-and-recovered and dense-continuation training
#    arms, which are the only paths in this package that train anything
#
# Design decisions:
# - Every technique is a transformation of an already-trained checkpoint, so
#   nothing here can alter what Study 1 measured
# - Optional backends (the weight-only quantization library, the ONNX exporter
#   and runtime) are imported inside the functions that need them, so deciding
#   applicability or constructing a technique never requires them to be present
# - The package exports nothing; every module is imported by its full path
#
# Author: Rahul Sawhney

__all__: list[str] = []
