# vocode.losses: Architecture-Native Training Objectives

`losses/` holds one objective module per trained architecture together with the shared components those objectives compose. The objectives are deliberately architecture-native and were never standardized across the cohort, so raw training and validation losses are not comparable across families and never rank configurations. The shared cross-model evaluation lives in `../metrics/`, not here.

Three shared components are composed by the architecture objectives. `adversarial.py` defines `LeastSquaresGanLoss`, the least-squares generator and discriminator terms used by the time-domain GAN objectives. `feature_matching.py` defines `FeatureMatchingLoss`, the discriminator feature-space L1 term. `mel_reconstruction.py` defines `MelReconstructionLoss`, the mel-spectrogram reconstruction term.

| Module | Objective |
|---|---|
| `melgan.py` | `MelganLoss` optimizes least-squares adversarial and feature-matching terms alone, with no reconstruction term. |
| `hifigan.py` | `HifiganLoss` combines least-squares adversarial, feature-matching, and weighted mel-reconstruction terms over the HiFi-GAN discriminator ensembles, serving the V1, V2, and V3 configurations. |
| `bigvgan.py` | `BigvganLoss` applies the same composition over the BigVGAN period and resolution discriminators. |
| `vocos.py` | `VocosLoss` applies the hinge form of the adversarial composition through `_VocosHingeGanLoss`; Vocos and VocosFormer share this objective. |
| `apnet2.py` | `Apnet2Loss` combines log-amplitude, anti-wrapping phase, and STFT-consistency terms computed by `Apnet2SpectrumAnalyzer` with hinge adversarial and feature-matching supervision. |
| `freev.py` | `FreevLoss` applies the APNet2-style spectral composition over the FreeV spectrum defined in `FreevSpectrum`. |
| `rndvoc.py` | `RndvocLoss` combines the spectral composition with the omni-directional phase term `RndvocOmniPhaseLoss` and hinge adversarial supervision. |
| `rfwave.py` | `RfwaveLoss` regresses conditional flow velocities in the waveform domain with auxiliary magnitude and band-overlap terms. |
| `lpcnet.py` | `LpcnetLoss` minimizes teacher-forced cross-entropy over the quantized mu-law excitation. |
| `hiftnet.py` | `HiftnetLoss` implements the HiFTNet composition, including the top-k relative least-squares term `_TopKRelativeLeastSquaresLoss`; it serves the non-executed HiFTNet architecture. |

Each objective module carries a frozen configuration class named `*LossConfig` that holds its term weights, so the executed coefficients are explicit and recoverable from the resolved run records under `../../../artifacts/`.

## Related Components

The discriminators these objectives score live beside their generators in `../models/`. The mirrored tests live in `../../tests/vocode/losses/`.
