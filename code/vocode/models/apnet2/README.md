# APNet2

APNet2 is an iSTFT-GAN vocoder that predicts amplitude and phase spectra directly and reconstructs the waveform through the inverse short-time Fourier transform. In the cohort it is the 31.43-million-parameter configuration conditioned on 80 mel bands at 22.05 kHz, trained for 124.8 thousand updates with a retained state at 120 thousand.

| Module | Contents |
|---|---|
| `apnet2.py` | `Apnet2Config` and `Apnet2`, the vocoder module. |
| `network.py` | `Apnet2Network`, the generator. |
| `discriminator.py` | `Apnet2MultiPeriodDiscriminator` and `Apnet2MultiResolutionDiscriminator`. |
| `weights.py` | `Apnet2Weights`, the author-release adapter. |

`Apnet2` validates its configuration, constructs the generator and both discriminator ensembles, executes the adversarial training recipe, and synthesizes waveforms from mel conditioning. `Apnet2Network` computes parallel amplitude and phase branches built from ConvNeXt blocks with global response normalization and returns an `Apnet2GeneratorOutput` of spectra that the inverse transform converts to a waveform. The period and resolution discriminator ensembles supply the adversarial and feature-matching signals of the training recipe.

`Apnet2Weights` records the authors' distribution page (http://home.ustc.edu.cn/~redmist/APNet2/) as source provenance and follows the shared `PublishedWeights` contract: retrieval, SHA-256 validation, upstream state extraction, parameter-key adaptation, and strict loading.

## Related Components

The training objective lives in `../../losses/apnet2.py` and combines log-amplitude, anti-wrapping phase, and STFT-consistency terms with hinge adversarial and feature-matching supervision. The mel front end lives in `../../transforms/mel.py`. The mirrored tests live in `../../../tests/vocode/models/apnet2/`.
