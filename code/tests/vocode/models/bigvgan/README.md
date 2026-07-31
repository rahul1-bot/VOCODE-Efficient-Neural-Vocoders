# tests.vocode.models.bigvgan: The BigVGAN-base Tests

This directory holds 83 tests mirroring `vocode/models/bigvgan/`.

The module `bigvgan.py` tests the vocoder module: configuration validation, generator and discriminator construction, the adversarial step recipe, and the synthesis shape contracts. The module `network.py` tests the generator: the Snake and SnakeBeta periodic activations, the Kaiser-windowed sinc filtering with its paired anti-aliased upsampling and downsampling, the activation-block behavior, and the upsampling arithmetic. The module `discriminator.py` tests the period and resolution ensembles: their output structure and exposed feature maps.

The module `weights.py` tests the author-release adapter against fabricated archives with self-consistent digests, covering retrieval, SHA-256 validation, parameter-key adaptation, and strict loading without any download.
