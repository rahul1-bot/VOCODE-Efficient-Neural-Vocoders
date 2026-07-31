# tests.vocode.models.vocos: The Vocos Tests

This directory holds 59 tests mirroring `vocode/models/vocos/`.

The module `vocos.py` tests the vocoder module: configuration validation, generator and discriminator construction, the adversarial step recipe, and the synthesis shape contracts. The module `network.py` tests the generator: the ConvNeXt backbone behavior at constant temporal resolution and the inverse-STFT reconstruction head. The module `discriminator.py` tests the period and resolution ensembles: their output structure and exposed feature maps.

The module `weights.py` tests the author-release adapter against fabricated archives with self-consistent digests, covering retrieval, SHA-256 validation, parameter-key adaptation, and strict loading without any download.
