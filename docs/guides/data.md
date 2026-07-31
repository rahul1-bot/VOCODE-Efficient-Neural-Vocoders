# Data

All experiments use LJSpeech 1.1: 13,100 English utterances from one speaker, public under its released license. The corpus is fetched at run time and never stored in the repository. Corpus access is implemented in `code/vocode/data/`.

## Partition

Identifier-sorted disjoint blocks fix 12,475 training, 100 validation, and 525 evaluation utterances without a random seed. Identifier order preserves the chapter structure of the corpus, so split membership is a fixed property of the corpus rather than a random utterance sample. All twelve configurations consume exactly this partition.

The 525-utterance evaluation suffix is adaptive, not untouched: its PESQ values informed checkpoint continuation and extension decisions and one deterministic repair of the RFWave inference sampler. The report discloses this and scopes every conclusion accordingly.

## Loaders

Training loaders shuffle and crop fixed-length segments under the architecture-native crop length (2,400 to 32,512 samples). Validation and evaluation loaders consume complete utterances in deterministic order. Batches are padded by the collator, and quality metrics are computed at true length, so padding never enters a reported value.

## Conditioning

Every configuration retains its native conditioning interface, implemented in `code/vocode/transforms/`:

| Front end | Configurations | Specification |
|---|---|---|
| Mel, 80 bands | APNet2, FreeV, HiFi-GAN V1, V2, V3, MelGAN, RNDVoC | Slaney scale at 22.05 kHz |
| Mel, 100 bands normalized | BigVGAN-base | 24 kHz |
| Mel, 100 bands unnormalized | Vocos, VocosFormer, RFWave | HTK scale at 24 kHz |
| LPCNet features | LPCNet | 18 Bark-frequency cepstra with pitch period and correlation at 16 kHz |

All mel pipelines share a 1,024-point Hann window with a 256-sample hop. Forcing one global protocol would invalidate the published training recipes, so the study compares executed configurations rather than re-normalized ones.

## Training-Only Transformations

Three configurations apply augmentations during training only; evaluation waveforms receive none. HiFi-GAN V1, V2, and V3 apply peak normalization to 0.95; Vocos and VocosFormer apply uniform random peak gain in the range of negative 6 to negative 1 dB; LPCNet adds excitation noise with standard deviation 0.3.
