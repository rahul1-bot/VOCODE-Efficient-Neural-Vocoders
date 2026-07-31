# Vocos

This directory contains the Study 1 evidence package for Vocos. One 24 kHz ConvNeXt-ISTFT model was trained from random initialization, and its selected durable state was evaluated in three complete adaptive-benchmark executions.

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
| Generator parameters | 13,531,650 |

The partition is deterministic and does not use a split seed. Training seed 1234 controls initialization and optimization, whereas seeds 42, 43, and 44 identify repeated executions of the same selected model on the same test utterances.

## Results

| Seed | PESQ | STOI | Mel error | Real-time factor |
|---:|---:|---:|---:|---:|
| 42 | 3.565251 | 0.979314 | 3.046101 | 0.005486 |
| 43 | 3.565251 | 0.979314 | 3.046101 | 0.004344 |
| 44 | 3.565251 | 0.979314 | 3.046101 | 0.004344 |
| Mean | 3.565251 | 0.979314 | 3.046101 | 0.004725 |

Mean PESQ reaches 94.8% of the registered author-checkpoint anchor of 3.7612 and exceeds the gate-one threshold of 3.50. The anchor originates from the validated legacy evaluator and is PESQ-comparable only; the current mel-error value must not be compared with its legacy counterpart. Vocos passes its registered gate at the first budget and is the fastest Study 1 model under B200 FP32 timing.

## Contents

| Path | Evidence |
|---|---|
| `experiments.csv` | Index of the accepted project-trained execution and its three adaptive-benchmark executions. |
| `train/` | Training configuration, complete execution log, train/validation history, and selected-state description. |
| `test/` | Seed-specific B200 test configurations, complete execution logs, and final Study 1 measurements. |
| `figures/` | Training-convergence figure in PNG and vector PDF formats. |
