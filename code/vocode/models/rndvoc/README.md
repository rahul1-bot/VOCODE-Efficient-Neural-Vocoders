# RNDVoC

RNDVoC is an iSTFT-GAN vocoder built on range-null space decomposition: the analytically recoverable range-space component of the spectrum is computed directly, and the network learns only the null-space residual. In the cohort it is the 3.57-million-parameter configuration conditioned on 80 mel bands at 22.05 kHz, trained on 16,384-sample crops for 124.8 thousand updates with a retained state at 120 thousand. It produced the strongest baseline PESQ and STOI of the study and sits on both the speed-quality and the footprint-quality frontiers.

| Module | Contents |
|---|---|
| `rndvoc.py` | `RndvocConfig` and `Rndvoc`, the vocoder module. |
| `network.py` | `RndvocNetwork`, the generator. |
| `discriminator.py` | `RndvocMultiPeriodDiscriminator` and `RndvocMultiResolutionDiscriminator`. |

`Rndvoc` validates its configuration, constructs the generator and both discriminator ensembles, executes the adversarial training recipe, and synthesizes waveforms. `RndvocNetwork` splits the spectrum into bands through a shared band split, processes them with band-wise temporal modules, grouped linear projections, and a band shuffler, applies a family of band-aware normalizations (channel, band-wise layer, complex band-wise layer, and global response normalization), and merges the bands through a shared band merge into an `RndvocGeneratorOutput` for inverse-STFT reconstruction.

There is no author-weight adapter, because no commensurate author release is loaded for this architecture.

## Related Components

The training objective lives in `../../losses/rndvoc.py` and combines the spectral composition with the omni-directional phase term and hinge adversarial supervision. The mel front end lives in `../../transforms/mel.py`. The mirrored tests live in `../../../tests/vocode/models/rndvoc/`.
