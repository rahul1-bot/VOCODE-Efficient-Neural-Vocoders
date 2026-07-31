# tests.vocode.models.freev: The FreeV Tests

This directory holds 48 tests mirroring `vocode/models/freev/`.

The module `freev.py` tests the vocoder module: configuration validation, generator construction, the composition of the shared APNet2 discriminator ensembles, the alternating adversarial update, and the synthesis shape contracts. The module `network.py` tests the generator: the handling of the pseudo-inverse mel prior, the ConvNeXt block arithmetic with global response normalization, and the spectral output contract.

The module `weights.py` tests the author-release adapter against fabricated archives with self-consistent digests, covering retrieval, SHA-256 validation, parameter-key adaptation, and strict loading without any download.
