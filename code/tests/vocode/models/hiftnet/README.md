# tests.vocode.models.hiftnet: The HiFTNet Tests

This directory holds 79 tests mirroring `vocode/models/hiftnet/`. HiFTNet is a documented non-executed exclusion of the study, and its implementation is nevertheless tested to the same standard as the executed architectures.

The module `hiftnet.py` tests the vocoder module: configuration validation, generator and discriminator construction, and the adversarial step recipe. The module `network.py` tests the generator: the JDC pitch-prediction path, the sine-excitation source module, the residual processing blocks, and the inverse-STFT reconstruction. The module `discriminator.py` tests the period and spectrogram ensembles: their output structure and exposed feature maps.

The module `weights.py` tests the author-release adapter, including the separate fundamental-frequency predictor checkpoint path, against fabricated archives with self-consistent digests, covering retrieval, SHA-256 validation, parameter-key adaptation, and strict loading without any download.
