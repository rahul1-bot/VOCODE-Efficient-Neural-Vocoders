# tests.vocode.metrics: The Measurement Stack Tests

This directory holds 360 tests mirroring `vocode/metrics/`, with one test module per metric module. It is the largest directory of the suite, because every reported number of the study passes through these implementations.

The quality proxies are tested in `pesq.py`, `stoi.py`, and `utmos.py` for resampling and truncation discipline, operating-rate handling, and failure behavior; the UTMOS tests substitute a constructed predictor, so no torch.hub download occurs. The spectral and prosodic diagnostics are tested in `mel.py`, `stft.py`, `mcd.py`, `las.py`, `pitch.py`, `f0.py`, `periodicity.py`, and `voicing.py` for arithmetic on synthesized signal pairs, including the shared pitch front end consumed by the three pitch-domain metrics.

The timing and footprint measurements are tested in `rtf.py`, `macs.py`, `parameters.py`, and `size.py` for measurement semantics: synchronized timing aggregation, traced multiply-accumulate counting through the synthesis probe, deployable-parameter counting, and serialized-size computation. The composition layer is tested in `registry.py` and `sequence.py` for the closed metric vocabulary, the ordered validated selection, and sequence execution that assembles measured values in declaration order.

The true-length discipline is asserted explicitly: reference and synthesized waveforms are truncated to their common unpadded length before scoring, so padding can never inflate a quality value.
