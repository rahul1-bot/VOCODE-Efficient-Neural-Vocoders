# VocosFormer

This directory contains the Study 1 evidence package for VocosFormer, the project label for a literature-derived, parameter-matched attention-augmented Vocos adaptation. The configuration combines the Vocos mel-conditioned ConvNeXt/ISTFT chassis with the published WavTokenizer-style residual-convolution and self-attention position network. Its weights were trained from random initialization; the architecture was not invented by this project.

## Experimental Design

| Component | Contract |
|---|---|
| Dataset | LJSpeech 1.1; source audio resampled from 22.05 kHz to 24 kHz |
| Partition | Identifier-ordered holdout: 12,475 train, 100 validation, 525 test utterances |
| Training seed | 1234 |
| Evaluation seeds | 42, 43, and 44 |
| Training hardware | NVIDIA L40S, BF16 mixed precision |
| Test hardware | NVIDIA B200, FP32 |
| Selected state | Epoch 307, step 120,000 |
| Generator parameters | 13,778,690 |

The partition is deterministic and does not use a split seed. The three evaluation seeds identify repeated executions of the same selected state on the same test utterances; they are not independent training replicas.

## Results

| Seed | Recorded PESQ | True-length PESQ | Recorded STOI | True-length STOI | Mel error | Real-time factor |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.479803 | 3.512452 | 0.958694 | 0.975776 | 2.910696 | 0.011641 |
| 43 | 2.479801 | 3.512459 | 0.958694 | 0.975776 | 2.910696 | 0.013302 |
| 44 | 2.479751 | 3.512460 | 0.958694 | 0.975776 | 2.910696 | 0.011567 |
| Mean | 2.479785 | 3.512457 | 0.958694 | 0.975776 | 2.910696 | 0.012170 |

The original recorded lane used padded batch-16 evaluation and is retained as robustness evidence. A batch-1 re-evaluation of the same selected state at true utterance lengths raises PESQ by 1.0327, isolating padding contamination through unmasked global attention. Therefore 3.5125 is the defensible synthesis-quality result, while 2.4798 measures padding sensitivity. VocosFormer is a controlled architectural adaptation with Vocos as its matched control, not a faithful reproduction of either Vocos or WavTokenizer and not a project-original network.

## Contents

| Path | Evidence |
|---|---|
| `experiments.csv` | Index of the accepted training execution and three adaptive-benchmark executions. |
| `train/` | Training configuration, complete execution log, train/validation history, and selected-state description. |
| `test/` | Seed-specific B200 configurations, complete execution logs, and Study 1 measurements. |
| `figures/` | Training-convergence figure in PNG and vector PDF formats. |
