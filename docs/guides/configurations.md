# Configurations

The cohort comprises twelve Project-Trained Configurations. A Project-Trained Configuration is one architecture recipe fitted from random initialization under the project budget; its selected state is the Retained Project Checkpoint, which every downstream measurement inherits. Architecture implementations live in `code/vocode/models/`, one directory per architecture.

## The Executed Configuration Surface

| Configuration | Family | Conditioning | Params (M) | Segment | Batch | Precision | Updates (k) | Retained state |
|---|---|---|---:|---:|---:|---|---:|---:|
| APNet2 | iSTFT GAN | 80 mel / 22.05 kHz | 31.43 | 8,192 | 32 | BF16 | 124.8 | 120k |
| BigVGAN-base | Waveform GAN | 100 mel / 24 kHz | 14.03 | 8,192 | 32 | BF16 | 124.8 | 120k |
| FreeV | iSTFT GAN | 80 mel / 22.05 kHz | 18.22 | 8,192 | 32 | BF16 | 124.8 | 120k |
| HiFi-GAN V1 | Waveform GAN | 80 mel / 22.05 kHz | 13.94 | 8,192 | 32 | BF16 | 254.8 | 245k |
| HiFi-GAN V2 | Waveform GAN | 80 mel / 22.05 kHz | 0.93 | 8,192 | 32 | BF16 | 124.8 | 120k |
| HiFi-GAN V3 | Waveform GAN | 80 mel / 22.05 kHz | 1.46 | 8,192 | 32 | BF16 | 254.8 | 245k |
| LPCNet | Autoregressive | 20 feat / 16 kHz | 1.23 | 2,400 | 128 | FP32 | 31.3 | 30k |
| MelGAN | Waveform GAN | 80 mel / 22.05 kHz | 4.27 | 16,000 | 32 | BF16 | 124.8 | 120k |
| RNDVoC | iSTFT GAN | 80 mel / 22.05 kHz | 3.57 | 16,384 | 32 | BF16 | 124.8 | 120k |
| RFWave | Rectified flow | 100 mel / 24 kHz | 18.14 | 32,512 | 64 | BF16 | 125.0 | 120k |
| Vocos | iSTFT GAN | 100 mel / 24 kHz | 13.53 | 16,384 | 32 | BF16 | 124.8 | 120k |
| VocosFormer | iSTFT GAN + attention | 100 mel / 24 kHz | 13.78 | 16,384 | 32 | BF16 | 124.8 | 120k |

Params counts deployable parameters in millions; Segment is the training crop in samples; Updates is the executed budget in thousands; Retained state is the update count of the Retained Project Checkpoint selected by the registered gate.

## Objectives and Optimization

Objectives are architecture-native and live in `code/vocode/losses/`; they were deliberately not standardized, so raw training losses are never compared across families.

| Configuration | Objective | Optimizer | Initial learning rate |
|---|---|---|---|
| MelGAN | Least-squares adversarial and feature matching, no reconstruction term | Adam | 1e-4 |
| HiFi-GAN V1, V2, V3 | Least-squares adversarial, feature matching, weighted mel reconstruction | AdamW | 2e-4 |
| BigVGAN-base | The same composition over its discriminator ensembles | AdamW | 1e-4 |
| Vocos, VocosFormer | The hinge form of the adversarial composition | AdamW | 5e-4 |
| APNet2, FreeV, RNDVoC | Log-amplitude, anti-wrapping phase, and STFT-consistency terms with hinge adversarial and feature-matching supervision | AdamW | 2e-4 |
| RFWave | Conditional flow-velocity regression with auxiliary magnitude and band-overlap terms | AdamW | 2e-4 |
| LPCNet | Teacher-forced cross-entropy over the quantized mu-law excitation | Adam | 1e-3 |

LPCNet additionally trains under scheduled recurrent-weight sparsification between 2 and 20 thousand updates. The HiFi-GAN V1 and V3 budgets include 4,800 replayed updates after a durable resume. Exact coefficients and schedules are recoverable from the resolved run records and the code.

## Checkpoint Selection

Selection followed a registered gate: validation loss informed continuation, durable checkpoints were written every 5,000 updates, and a gain below 0.02 PESQ across gates indicated a plateau. Where a commensurate published or released-checkpoint PESQ anchor existed, the registered quality target was 93 percent of that anchor; RFWave used a registered PESQ floor of 3.55; LPCNet and VocosFormer had no commensurate anchor. Evaluation used the terminal durable checkpoint admitted by this gate, not a retrospectively substituted validation minimum.

## Special Cases

VocosFormer is the project-defined exception: a literature-derived, capacity-matched Vocos adaptation that inserts residual-convolution and self-attention components from the WavTokenizer decoder lineage into a shallower Vocos chassis, at 247,040 parameters (1.83 percent) above Vocos, under matched losses, data, and training quantum. HiFTNet is implemented and adapter-equipped in `code/vocode/models/hiftnet/` but is a documented non-executed exclusion, not a thirteenth result.
