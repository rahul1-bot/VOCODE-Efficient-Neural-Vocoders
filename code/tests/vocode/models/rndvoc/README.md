# tests.vocode.models.rndvoc: The RNDVoC Tests

This directory holds 69 tests mirroring `vocode/models/rndvoc/`.

The module `rndvoc.py` tests the vocoder module: configuration validation, generator and discriminator construction, the adversarial step recipe, and the synthesis shape contracts. The module `network.py` tests the generator: the shared band split and merge round trips, the band-wise temporal processing, the grouped linear and band-shuffling behavior, the band-aware normalization family, and the range-null output composition. The module `discriminator.py` tests the period and resolution ensembles: their output structure and exposed feature maps.
