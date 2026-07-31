# BigVGAN

This directory contains the Study 1 evidence package for BigVGAN-base. One 24 kHz, 100-band model was trained from random initialization, and its selected durable state was evaluated in three complete adaptive-benchmark executions.

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
| Generator parameters | 14,025,154 |

The partition is deterministic and does not use a split seed. Training seed 1234 controls initialization and optimization, whereas seeds 42, 43, and 44 identify separate executions of the same selected model on the same test utterances.

## Results

| Seed | PESQ | STOI | Mel error | Real-time factor |
|---:|---:|---:|---:|---:|
| 42 | 3.256716 | 0.973688 | 0.638447 | 0.154058 |
| 43 | 3.257049 | 0.973688 | 0.638400 | 0.185560 |
| 44 | 3.257181 | 0.973688 | 0.638468 | 0.132183 |
| Mean | 3.256982 | 0.973688 | 0.638439 | 0.157267 |

## Released-Checkpoint Reference

The author-released NVIDIA `bigvgan_base_24khz_100band` generator was evaluated once with seed 42 through the same project implementation and evaluator on the identical 525-utterance adaptive evaluation partition using NVIDIA B200 FP32 execution. The checkpoint loaded with zero missing and zero unexpected state-dict keys and produced PESQ 3.901279, STOI 0.988868, mel error 0.095962, and real-time factor 0.136930. This measurement validates the evaluation path and supplies the reference denominator for the project-trained result; it is not a project-trained execution and is excluded from both the three-seed mean and `experiments.csv`.

The three-seed result is a partial reproduction under the registered compute budget: mean PESQ is 83.5% of the current-harness released-checkpoint anchor. Training completed the 124,800-step budget and stopped after the first gate because the quality threshold and continuation condition were not met.

## Contents

| Path | Evidence |
|---|---|
| `experiments.csv` | Index of the accepted project-trained execution and its three project-trained adaptive-benchmark executions. |
| `train/` | Training configuration, complete execution log, train/validation history, and selected-state description. |
| `test/` | Seed-specific B200 test configurations, complete execution logs, and final Study 1 measurements. |
| `figures/` | Training-convergence figure in PNG and vector PDF formats. |
