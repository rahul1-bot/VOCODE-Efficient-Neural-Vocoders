# tests.vocode.models.melgan: The MelGAN Tests

This directory holds 78 tests mirroring `vocode/models/melgan/`.

The module `melgan.py` tests the vocoder module: configuration validation, generator and discriminator construction, the adversarial step recipe, and the synthesis shape contracts. The module `network.py` tests the generator: the transposed-convolution upsampling and the residual-stack behavior. The module `discriminator.py` tests the multi-scale ensemble: the sub-discriminator structure and the feature maps it exposes for feature matching.

The module `weights.py` tests the author-release adapter against fabricated archives with self-consistent digests, covering retrieval, SHA-256 validation, parameter-key adaptation, and strict loading without any download.
