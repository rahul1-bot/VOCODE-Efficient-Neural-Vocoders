# tests.vocode.models.apnet2: The APNet2 Tests

This directory holds 52 tests mirroring `vocode/models/apnet2/`.

The module `apnet2.py` tests the vocoder module: configuration validation, generator and discriminator construction, the adversarial step recipe, and the synthesis shape contracts. The module `network.py` tests the generator: the behavior of the parallel amplitude and phase branches, the ConvNeXt block arithmetic with global response normalization, and the spectra-to-waveform output contract. The module `discriminator.py` tests the period and resolution ensembles: their output structure and the feature maps they expose for feature matching.

The module `weights.py` tests the author-release adapter against fabricated archives with self-consistent SHA-256 digests, covering retrieval, validation, upstream state extraction, parameter-key adaptation, and strict loading without any download.
