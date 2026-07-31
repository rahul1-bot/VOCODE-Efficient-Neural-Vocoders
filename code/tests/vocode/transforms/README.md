# tests.vocode.transforms: The Conditioning Signal Path Tests

This directory holds 85 tests mirroring `vocode/transforms/`.

The module `mel.py` tests `MelConfig` and `MelSpectrogram`: the architecture-native settings of the cohort (80 Slaney-scale bands at 22.05 kHz, 100 normalized bands, and 100 unnormalized HTK-scale bands at 24 kHz), window and hop arithmetic, shape contracts, and numerical behavior on synthesized waveforms.

The module `lpc.py` tests the LPCNet front end: the 18 Bark-frequency cepstra with pitch period and correlation features at 16 kHz, mu-law companding and expansion round trips, and feature-frame alignment.

Both modules are tested purely on constructed signals, and conditioning differences between architectures are asserted to be configuration values rather than divergent code paths.
