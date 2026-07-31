# vocode.transforms: Conditioning Signal Paths

`transforms/` implements the two conditioning front ends of the cohort. Every architecture consumes features produced here, so conditioning differences between configurations are explicit constructor settings rather than divergent implementations.

| Module | Contents |
|---|---|
| `mel.py` | `MelConfig` and `MelSpectrogram`, the mel-spectrogram path shared by the eleven mel-conditioned configurations. |
| `lpc.py` | `LpcnetFeatureConfig`, `LpcnetFeatures`, `LpcnetFeatureExtractor`, and `LpcnetMuLaw`, the LPCNet front end. |

The mel configuration carries the architecture-native settings of the study: 80 Slaney-scale bands at 22.05 kHz for most configurations, 100 normalized bands for BigVGAN-base, and 100 unnormalized HTK-scale bands at 24 kHz for Vocos, VocosFormer, and RFWave, all over a 1,024-point Hann window with a 256-sample hop. The LPCNet front end operates at 16 kHz and produces 18 Bark-frequency cepstra with pitch period and correlation features, and `LpcnetMuLaw` provides the mu-law companding of the excitation domain.

Architecture-native conditioning is retained by design: forcing one global protocol would invalidate the published training recipes, so each configuration declares its native transform settings and the study compares executed configurations rather than re-normalized ones.

## Related Components

The transform settings of each configuration are validated in `../configs/` and recorded in the resolved run records under `../../../artifacts/`. The mirrored tests live in `../../tests/vocode/transforms/`.
