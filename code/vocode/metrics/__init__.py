# Package initializer for the objective quality, speed, and complexity
# metrics; modules are imported by full path.
#
# This package:
# 1. Holds the closed metric vocabulary of the study in registry, which is
#    the sole authority on which metrics exist, how their names map onto
#    result columns, and how instances are constructed by name
# 2. Holds the per-utterance quality metrics (pesq, stoi, mel, stft, mcd,
#    las, utmos), the pitch-family metrics (f0, periodicity, voicing) with
#    their shared extractor in pitch, and the complexity metrics
#    (parameters, size, macs)
# 3. Holds the two callbacks that ride a harness loop rather than scoring a
#    signal pair: sequence, which drives the whole panel over a test pass,
#    and rtf, which measures the real-time factor over a prediction pass
#
# Design decisions:
# - The initializer re-exports nothing, so every consumer imports the
#   module it actually depends on and the package never becomes an implicit
#   dependency on the entire metric surface; several metrics pull in heavy
#   third-party analysis stacks that an unrelated import must not trigger
# - Metrics divide into three lifecycles rather than one interface: static
#   metrics measure a network once per run, per-utterance metrics score a
#   signal pair, and pitch-family metrics accumulate across a whole pass;
#   sequence is the component that reconciles the three into one panel
#
# Author: Rahul Sawhney

__all__: list[str] = []
