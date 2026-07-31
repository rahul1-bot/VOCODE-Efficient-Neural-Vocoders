# Vocos

Vocos is an iSTFT-GAN vocoder that predicts complex spectral coefficients with a constant-resolution ConvNeXt stack, moving all learned computation to the acoustic-frame rate. In the cohort it is the 13.53-million-parameter configuration conditioned on 100 unnormalized HTK-scale mel bands at 24 kHz, trained on 16,384-sample crops for 124.8 thousand updates with a retained state at 120 thousand. It held the lowest real-time factor on both hardware lanes, and it was the selected architecture at every constraint vector of the study's deployment selector.

| Module | Contents |
|---|---|
| `vocos.py` | `VocosConfig` and `Vocos`, the vocoder module. |
| `network.py` | `VocosNetwork`, the generator. |
| `discriminator.py` | `VocosMultiPeriodDiscriminator` and `VocosMultiResolutionDiscriminator`. |
| `weights.py` | `VocosWeights`, the author-release adapter. |

`Vocos` validates its configuration, constructs the generator and both discriminator ensembles, executes the adversarial training recipe, and synthesizes waveforms. `VocosNetwork` runs a ConvNeXt backbone at constant temporal resolution, configured through `VocosNetworkConfig`, and reconstructs the waveform through an inverse-STFT head.

`VocosWeights` records the charactr-platform distribution (https://github.com/charactr-platform/vocos) as source provenance and follows the shared retrieval, SHA-256 validation, parameter-key adaptation, and strict-load contract.

## Related Components

The training objective lives in `../../losses/vocos.py` and applies the hinge form of the adversarial composition; VocosFormer shares this objective. The `../vocosformer/` directory subclasses this module for the project-defined attention-augmented comparison. The mel front end lives in `../../transforms/mel.py`. The mirrored tests live in `../../../tests/vocode/models/vocos/`.
