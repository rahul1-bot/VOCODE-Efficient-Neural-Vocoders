# MelGAN

This directory contains the Study 1 evidence package for MelGAN. One Seungwon-compatible generator was trained from random initialization and its selected durable state was evaluated in three complete adaptive-benchmark executions.

## Experimental Design

| Component | Contract |
|---|---|
| Dataset | LJSpeech 1.1 |
| Partition | Identifier-ordered holdout: 12,475 train, 100 validation, 525 test utterances |
| Training seed | 1234 |
| Evaluation seeds | 42, 43, and 44 |
| Training hardware | NVIDIA L40S, BF16 mixed precision |
| Test hardware | NVIDIA B200, FP32 |
| Selected state | Epoch 307, step 120,000 |
| Generator parameters | 4,266,050 |

The partition is deterministic and does not use a split seed. Training seed 1234 controls initialization and optimization, whereas seeds 42, 43, and 44 identify repeated executions of the same selected model on the same test utterances.

## Results

| Seed | PESQ | STOI | Mel error | Real-time factor |
|---:|---:|---:|---:|---:|
| 42 | 1.059746 | 0.418275 | 1.791590 | 0.095189 |
| 43 | 1.059759 | 0.418274 | 1.791471 | 0.066263 |
| 44 | 1.059746 | 0.418275 | 1.791590 | 0.061193 |
| Mean | 1.059750 | 0.418274 | 1.791550 | 0.074215 |

The selected Seungwon Park community checkpoint scores PESQ 2.396842 in the current evaluator. The project-trained mean reaches 44.2% of that anchor and therefore does not satisfy the registered gate-one quality threshold. The project loss equations match that community implementation's least-squares, sum-reduction recipe; the original MelGAN paper uses a hinge objective. The checkpoint probe excludes a broken MelGAN evaluator, while the limited training exposure and descending validation make budget-limited non-convergence the leading explanation rather than proof that compute alone caused the gap.

## Contents

| Path | Evidence |
|---|---|
| `experiments.csv` | Index of the accepted project-trained execution and its three adaptive-benchmark executions. |
| `train/` | Training configuration, complete execution log, train/validation history, and selected-state description. |
| `test/` | Seed-specific B200 test configurations, complete execution logs, and final Study 1 measurements. |
| `figures/` | Training-convergence figure in PNG and vector PDF formats. |
