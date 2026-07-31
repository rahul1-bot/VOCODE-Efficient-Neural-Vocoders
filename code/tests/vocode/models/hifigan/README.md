# tests.vocode.models.hifigan: The HiFi-GAN Tests

This directory holds 121 tests mirroring `vocode/models/hifigan/`. It is the largest architecture directory of the suite because it serves the V1, V2, and V3 configurations.

The module `hifigan.py` tests the vocoder module across all three variants: the variant-specific configuration covering the upsampling schedule, the channel widths, and the residual-block selection; construction; the adversarial step recipe; and the synthesis shape contracts. The module `network.py` tests the generator: the upsampling arithmetic and the multi-receptive-field residual composition of each variant. The module `resblock.py` tests both residual-block designs: their dilation patterns, padding arithmetic, and forward behavior. The module `discriminator.py` tests the multi-period and multi-scale ensembles: the sub-discriminator structure and the exposed feature maps.

The module `weights.py` tests the author-release adapter for the V1, V2, and V3 releases against fabricated archives with self-consistent digests, covering retrieval, SHA-256 validation, parameter-key adaptation, and strict loading without any download.
