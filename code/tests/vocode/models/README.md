# tests.vocode.models: The Architecture Tests

This directory holds 73 tests at its own level for the shared model infrastructure, plus 782 tests across the eleven architecture subdirectories, each of which carries its own `README.md`.

The module `registry.py` tests `ModelRegistry`: the closed vocabulary of the thirteen configuration names, resolution into architecture modules with their native data defaults, and explicit failure for unknown names. The module `vocoder.py` tests the shared contracts: the `Vocoder` module surface every architecture satisfies, and the `PublishedWeights` and `PublishedWeightProvenance` author-release contract with its recorded source, expected digest, and strict-load semantics.

The architecture subdirectories mirror the source layout: `apnet2/`, `bigvgan/`, `freev/`, `hifigan/`, `hiftnet/`, `lpcnet/`, `melgan/`, `rfwave/`, `rndvoc/`, `vocos/`, and `vocosformer/`. Weight-adapter tests throughout fabricate release archives with self-consistent SHA-256 digests, so retrieval, validation, extraction, key adaptation, and strict loading are exercised without any download.
