# HiFi-GAN V1, V2, and V3

HiFi-GAN is the canonical time-domain adversarial upsampler, and this directory serves three cohort configurations at distinct quality and footprint operating points, all conditioned on 80 mel bands at 22.05 kHz. V1 carries 13.94 million parameters and was trained for 254.8 thousand updates with a retained state at 245 thousand; V2 carries 0.93 million parameters and was trained for 124.8 thousand updates with a retained state at 120 thousand; V3 carries 1.46 million parameters and was trained for 254.8 thousand updates with a retained state at 245 thousand.

| Module | Contents |
|---|---|
| `hifigan.py` | `HifiganConfig` and `Hifigan`, the vocoder module for all three variants. |
| `network.py` | `HifiganNetwork`, the generator. |
| `resblock.py` | `ResBlock1` and `ResBlock2`, the two residual-block designs. |
| `discriminator.py` | `MultiPeriodDiscriminator` and `MultiScaleDiscriminator` with their sub-discriminators `DiscriminatorP` and `DiscriminatorS`. |
| `weights.py` | `HifiganWeights`, the author-release adapter. |

`Hifigan` carries the variant-specific configuration (the upsampling schedule, the channel widths, and the residual-block selection), constructs the generator and both discriminator ensembles, executes the adversarial training recipe, and synthesizes waveforms. `HifiganNetwork` performs transposed-convolution upsampling interleaved with multi-receptive-field residual stacks. V1 and V2 use the first residual-block design and V3 uses the second, following the published variant definitions.

`HifiganWeights` records the jik876 distribution (https://github.com/jik876/hifi-gan) as source provenance, covers the V1, V2, and V3 releases, and follows the shared retrieval, SHA-256 validation, parameter-key adaptation, and strict-load contract.

## Related Components

The training objective lives in `../../losses/hifigan.py` and combines least-squares adversarial, feature-matching, and weighted mel-reconstruction terms over both discriminator ensembles. The mel front end lives in `../../transforms/mel.py`. The mirrored tests live in `../../../tests/vocode/models/hifigan/`.
