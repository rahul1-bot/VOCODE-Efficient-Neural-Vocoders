# HiFTNet

HiFTNet is an iSTFT-GAN vocoder with a harmonic-plus-noise source module driven by a learned fundamental-frequency predictor. It is implemented and adapter-equipped in this tree but is a documented non-executed exclusion of the study: no HiFTNet configuration was trained or evaluated, and it is not counted as a thirteenth result.

| Module | Contents |
|---|---|
| `hiftnet.py` | `HiftnetConfig` and `Hiftnet`, the vocoder module. |
| `network.py` | `HiftnetNetwork`, the generator. |
| `discriminator.py` | `HiftnetMultiPeriodDiscriminator` and `HiftnetMultiResolutionSpectrogramDiscriminator`. |
| `weights.py` | `HiftnetWeights`, the author-release adapter. |

`Hiftnet` validates its configuration, constructs the generator and both discriminator ensembles, and defines the adversarial training recipe. `HiftnetNetwork` combines the JDC pitch-prediction network, a sine-excitation source path built from a sine generator and a source module, residual processing blocks, and inverse-STFT reconstruction, configured through `HiftnetNetworkConfig` and returning a `HiftnetGeneratorOutput`.

`HiftnetWeights` records the LJSpeech release (https://huggingface.co/yl4579/HiFTNet) as source provenance and follows the shared retrieval, SHA-256 validation, parameter-key adaptation, and strict-load contract. The separate fundamental-frequency predictor checkpoint is addressed through the command-line flag `--hiftnet-f0-checkpoint-path`.

## Related Components

The training objective lives in `../../losses/hiftnet.py` and includes the top-k relative least-squares term. The mirrored tests live in `../../../tests/vocode/models/hiftnet/` and exercise this implementation to the same standard as the executed architectures.
