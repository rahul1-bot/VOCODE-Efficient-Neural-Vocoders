# BigVGAN-base

BigVGAN-base is a time-domain adversarial upsampler with periodic activations and anti-aliased internal resampling. In the cohort it is the 14.03-million-parameter configuration conditioned on 100 normalized mel bands at 24 kHz, trained for 124.8 thousand updates with a retained state at 120 thousand.

| Module | Contents |
|---|---|
| `bigvgan.py` | `BigvganConfig` and `Bigvgan`, the vocoder module. |
| `network.py` | `BigvganNetwork`, the generator. |
| `discriminator.py` | `BigvganMultiPeriodDiscriminator` and `BigvganMultiResolutionDiscriminator`. |
| `weights.py` | `BigvganWeights`, the author-release adapter. |

`Bigvgan` validates its configuration, constructs the generator and both discriminator ensembles, executes the adversarial training recipe, and synthesizes waveforms from mel conditioning. `BigvganNetwork` performs transposed-convolution upsampling through anti-aliased activation blocks: the Snake and SnakeBeta periodic activations are wrapped in an anti-aliasing stage built from Kaiser-windowed sinc filters with paired upsampling and downsampling, which is the anti-aliased multi-periodicity design of the published architecture.

`BigvganWeights` records the NVIDIA distribution (https://github.com/NVIDIA/BigVGAN) as source provenance and follows the shared retrieval, SHA-256 validation, parameter-key adaptation, and strict-load contract.

## Related Components

The training objective lives in `../../losses/bigvgan.py` and combines least-squares adversarial, feature-matching, and weighted mel-reconstruction terms. The mel front end lives in `../../transforms/mel.py`. The mirrored tests live in `../../../tests/vocode/models/bigvgan/`.
