# RFWave

This directory contains the Study 1 evidence package for the project-trained reimplementation of the published RFWave architecture. The model weights were trained from random initialization under the project budget; this is a reproduction claim, not an architecture-invention claim.

## Experimental Design

| Component | Contract |
|---|---|
| Dataset | LJSpeech 1.1; source audio resampled from 22.05 kHz to 24 kHz |
| Partition | Identifier-ordered holdout: 12,475 train, 100 validation, 525 test utterances |
| Training seed | 1234 |
| Evaluation seeds | 42, 43, and 44 |
| Training hardware | NVIDIA A100 80 GB, BF16 mixed precision |
| Test hardware | NVIDIA B200, FP32 |
| Selected state | Epoch 615, step 120,000 |
| Generator parameters | 18,139,296 |

The partition is deterministic and does not use a split seed. RFWave samples Gaussian noise during its ten-step ODE solve, so the small three-seed quality spread is genuine inference variation rather than three independently trained models.

## Results

| Seed | Recorded PESQ | True-length PESQ | Recorded STOI | True-length STOI | Mel error | Real-time factor |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 3.666856 | 3.784768 | 0.980808 | 0.980942 | 3.837712 | 0.014322 |
| 43 | 3.670062 | 3.786457 | 0.980919 | 0.980903 | 3.837652 | 0.015042 |
| 44 | 3.670715 | 3.782428 | 0.980702 | 0.980909 | 3.837424 | 0.015066 |
| Mean | 3.669211 | 3.784551 | 0.980810 | 0.980918 | 3.837596 | 0.014810 |

RFWave reaches the registered strong band over an approximately 125,000-step project run, equal to 12.5% of the cited one-million-step author schedule; the reported metrics use its last durable checkpoint at step 120,000. Its recorded PESQ is second only to RNDVoC in the twelve-model Study 1 cohort. The cross-paper author value is context rather than a strict reproduction target because compute and evaluation protocols differ. Frame-aligned MCD and LAS-RMSE are not used for the headline comparison because the flow sampler has suspected group delay.

## Contents

| Path | Evidence |
|---|---|
| `experiments.csv` | Index of the accepted training execution and three adaptive-benchmark executions. |
| `train/` | Training configuration, complete execution log, train/validation history, and selected-state description. |
| `test/` | Seed-specific B200 configurations, complete execution logs, and Study 1 measurements. |
| `figures/` | Training-convergence figure in PNG and vector PDF formats. |
