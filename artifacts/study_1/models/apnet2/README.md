# APNet2

This directory contains the Study 1 evidence package for APNet2. One model was trained from random initialization and the selected training state was evaluated in three complete adaptive-benchmark executions.

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

The partition is deterministic and does not use a split seed. Training seed 1234 controls initialization and optimization, whereas seeds 42, 43, and 44 identify separate executions of the same selected model on the same test utterances.

## Results

| Seed | PESQ | STOI | Mel error | Real-time factor |
|---:|---:|---:|---:|---:|
| 42 | 2.541039 | 0.930045 | 0.543382 | 0.008689 |
| 43 | 2.541039 | 0.930045 | 0.543382 | 0.009360 |
| 44 | 2.541039 | 0.930045 | 0.543382 | 0.010303 |
| Mean | 2.541039 | 0.930045 | 0.543382 | 0.009451 |

## Released-Checkpoint Reference

The author-released APNet2 `g_01000000` checkpoint was evaluated once with seed 42 through the same project implementation and evaluator on the identical 525-utterance adaptive evaluation partition using NVIDIA B200 FP32 execution. The checkpoint loaded with zero missing and zero unexpected state-dict keys and produced PESQ 3.422746, STOI 0.973047, mel error 0.357415, and real-time factor 0.006552. This measurement validates the evaluation path and supplies the reference denominator for the project-trained result; it is not a project-trained execution and is excluded from both the three-seed mean and `experiments.csv`.

The three-seed result remains a partial reproduction under the registered compute budget: mean PESQ is 74.2% of the current-harness released-checkpoint anchor of 3.4227, and the training trajectory satisfied the flat-tail stopping rule.

## Contents

| Path | Evidence |
|---|---|
| `experiments.csv` | Index of the accepted project-trained execution and its three project-trained adaptive-benchmark executions. |
| `train/` | Training configuration, complete execution log, train/validation history, and selected-state description. |
| `test/` | Seed-specific B200 test configurations, complete execution logs, and final Study 1 measurements. |
| `figures/` | Training-convergence figure in PNG and vector PDF formats. |
