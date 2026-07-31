# FreeV

FreeV is an iSTFT-GAN vocoder that initializes its amplitude prediction from a pseudo-inverse mel prior, which reduces learned computation relative to a free-form spectral predictor. In the cohort it is the 18.22-million-parameter configuration conditioned on 80 mel bands at 22.05 kHz, trained for 124.8 thousand updates with a retained state at 120 thousand.

| Module | Contents |
|---|---|
| `freev.py` | `FreevConfig` and `Freev`, the vocoder module. |
| `network.py` | `FreevNetwork`, the generator. |
| `weights.py` | `FreevWeights`, the author-release adapter. |

`Freev` validates its configuration and constructs the generator, and its adversarial machinery is deliberately not its own: the module builds the shared APNet2 discriminator ensembles (`Apnet2MultiPeriodDiscriminator` and `Apnet2MultiResolutionDiscriminator`) and drives the same alternating update pattern, so the FreeV-specific contribution is confined to the generator. There is therefore no discriminator module in this directory. `FreevNetwork` processes the pseudo-inverse-initialized amplitude path and a phase path through ConvNeXt blocks with global response normalization and returns a `FreevGeneratorOutput` for inverse-STFT reconstruction.

`FreevWeights` records the BakerBunker distribution (https://github.com/BakerBunker/FreeV) as source provenance and follows the shared retrieval, SHA-256 validation, parameter-key adaptation, and strict-load contract.

## Related Components

The training objective lives in `../../losses/freev.py` and applies the APNet2-style spectral composition over the FreeV spectrum. The mel front end lives in `../../transforms/mel.py`. The mirrored tests live in `../../../tests/vocode/models/freev/`.
