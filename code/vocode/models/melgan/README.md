# MelGAN

MelGAN is the classic lightweight time-domain adversarial upsampler of the cohort. It is the 4.27-million-parameter configuration conditioned on 80 mel bands at 22.05 kHz, trained on 16,000-sample crops for 124.8 thousand updates with a retained state at 120 thousand.

| Module | Contents |
|---|---|
| `melgan.py` | `MelganConfig` and `Melgan`, the vocoder module. |
| `network.py` | `MelganNetwork` and `ResStack`, the generator. |
| `discriminator.py` | `MelganMultiScaleDiscriminator` and `MelganDiscriminator`. |
| `weights.py` | `MelganWeights`, the author-release adapter. |

`Melgan` validates its configuration, constructs the generator and the multi-scale discriminator ensemble, executes the adversarial training recipe, and synthesizes waveforms. `MelganNetwork` performs transposed-convolution upsampling through residual stacks, following the published design.

`MelganWeights` records the seungwonpark distribution (https://github.com/seungwonpark/melgan, release v0.3-alpha) as source provenance and follows the shared retrieval, SHA-256 validation, parameter-key adaptation, and strict-load contract.

The training objective in `../../losses/melgan.py` consists of least-squares adversarial and feature-matching terms alone, with no mel-reconstruction term. This is the weakest reconstruction supervision in the cohort, and the report discusses it as consistent with the configuration's floor-level baseline result under the project budget.

## Related Components

The mel front end lives in `../../transforms/mel.py`. The mirrored tests live in `../../../tests/vocode/models/melgan/`.
