# RNDVoC

This directory contains the Study 1 evidence package for RNDVoC. One shared-encoder 22 kHz model was trained from random initialization, and its selected durable state was evaluated in three complete adaptive-benchmark executions.

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
| Generator parameters | 3,572,361 |

The partition is deterministic and does not use a split seed. Training seed 1234 controls initialization and optimization, whereas seeds 42, 43, and 44 identify repeated executions of the same selected model on the same test utterances.

## Results

| Seed | PESQ | STOI | Mel error | Real-time factor |
|---:|---:|---:|---:|---:|
| 42 | 3.856987 | 0.987090 | 0.138808 | 0.053526 |
| 43 | 3.856412 | 0.987090 | 0.138808 | 0.059722 |
| 44 | 3.856987 | 0.987090 | 0.138808 | 0.054452 |
| Mean | 3.856795 | 0.987090 | 0.138808 | 0.055900 |

Mean PESQ reaches 96.7% of the paper-reported LJSpeech anchor of 3.987 and exceeds the registered gate-one threshold of 3.71. The paper row is a weaker comparison class than a same-harness released-checkpoint evaluation, so the percentage is reported without claiming exact protocol parity. RNDVoC nevertheless passes its pre-registered gate and produces the strongest project-trained quality row in Study 1.

## Contents

| Path | Evidence |
|---|---|
| `experiments.csv` | Index of the accepted project-trained execution and its three adaptive-benchmark executions. |
| `train/` | Training configuration, complete execution log, train/validation history, and selected-state description. |
| `test/` | Seed-specific B200 test configurations, complete execution logs, and final Study 1 measurements. |
| `figures/` | Training-convergence figure in PNG and vector PDF formats. |
