# tests.vocode.models.vocosformer: The VocosFormer Tests

This directory holds 30 tests mirroring `vocode/models/vocosformer/`. The directory is small by construction, because VocosFormer subclasses the reproduced Vocos module, so the discriminators, the losses, and the training recipe are covered by the tests in `../vocos/` and `../../losses/`.

The module `vocosformer.py` tests the vocoder module: configuration validation, the inheritance of the Vocos recipe, and the construction of the VocosFormer generator in place of the Vocos network. The module `network.py` tests the generator: the inserted residual-convolution and frame-attention blocks with their position handling inside the Vocos chassis, the capacity accounting relative to Vocos, and the synthesis shape contracts.
