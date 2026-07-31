# LPCNet

This directory contains the Study 1 evidence package for LPCNet. One sparse autoregressive model was trained from random initialization, and its selected durable state was evaluated in three complete test-set quality executions.

## Experimental Design

| Component | Contract |
|---|---|
| Dataset | LJSpeech 1.1, resampled from 22.05 kHz to 16 kHz |
| Partition | Identifier-ordered holdout: 12,475 train, 100 validation, 525 test utterances |
| Training seed | 1234 |
| Evaluation seeds | 42, 43, and 44 |
| Training hardware | NVIDIA L40S, FP32 |
| Test hardware | NVIDIA B200, FP32 |
| Selected state | Epoch 306, step 30,000 |
| Generator parameters | 1,232,992 |

The partition is deterministic and does not use a split seed. Training seed 1234 controls initialization and optimization, whereas seeds 42, 43, and 44 identify separate executions of the same selected model on the same test utterances.

## Results

| Seed | PESQ | STOI | Mel error | Real-time factor |
|---:|---:|---:|---:|---:|
| 42 | 1.075959 | 0.526932 | 2.179116 | 11.363275 |
| 43 | 1.078827 | 0.528553 | 2.196004 | Not measured |
| 44 | 1.076264 | 0.527143 | 2.192043 | Not measured |
| Mean | 1.077017 | 0.527543 | 2.189054 | 11.363275 (n=1) |

All three seeds cover the complete 525-utterance adaptive evaluation partition for quality. Because framework-level autoregressive synthesis is exceptionally slow, deployment timing was retained from seed 42 only: 27 utterances after five warm-up utterances. Seeds 43 and 44 deliberately disabled the prediction lane and therefore make no real-time-factor claim.

The pre-registered floor rule triggered at 5.8% of the reference training recipe. Teacher-forced validation cross-entropy decreased from 5.6289 to 3.2031, but free-running synthesis remained unstable; this row is therefore retained as a valid compute-fraction result rather than a successful quality reproduction. Published optimized-C speed claims are not directly comparable with this PyTorch framework timing lane.

## Contents

| Path | Evidence |
|---|---|
| `experiments.csv` | Index of the accepted training execution and three project-trained quality evaluations. |
| `train/` | Training configuration, complete execution log, train/validation history, and selected-state description. |
| `test/` | Seed-specific B200 test configurations, complete execution logs, and final Study 1 measurements. |
| `figures/` | Training-convergence figure in PNG and vector PDF formats. |
