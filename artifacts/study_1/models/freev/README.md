# FreeV

This directory contains the Study 1 evidence package for FreeV. One model was trained from random initialization, and its selected training state was evaluated in three complete adaptive-benchmark executions.

## Experimental Design

| Component | Contract |
|---|---|
| Dataset | LJSpeech 1.1 at its native 22.05 kHz sample rate |
| Partition | Identifier-ordered holdout: 12,475 train, 100 validation, 525 test utterances |
| Training seed | 1234 |
| Evaluation seeds | 42, 43, and 44 |
| Training hardware | NVIDIA L40S, BF16 mixed precision |
| Test hardware | NVIDIA B200, FP32 |
| Selected state | Epoch 307, step 120,000 |
| Generator parameters | 18,220,557 |

The partition is deterministic and does not use a split seed. Training seed 1234 controls initialization and optimization, whereas seeds 42, 43, and 44 identify separate executions of the same selected model on the same test utterances.

## Results

| Seed | PESQ | STOI | Mel error | Real-time factor |
|---:|---:|---:|---:|---:|
| 42 | 3.339525 | 0.969317 | 0.174622 | 0.006353 |
| 43 | 3.339515 | 0.969317 | 0.174621 | 0.009221 |
| 44 | 3.339253 | 0.969317 | 0.174622 | 0.007458 |
| Mean | 3.339431 | 0.969317 | 0.174622 | 0.007677 |

## Released-Checkpoint Reference

The author-released FreeV `freev_g_01000000` checkpoint was evaluated once with seed 42 through the same project implementation and evaluator on the identical 525-utterance adaptive evaluation partition using NVIDIA B200 FP32 execution. The checkpoint loaded with zero missing and zero unexpected state-dict keys and produced PESQ 3.554188, STOI 0.977936, mel error 0.160992, and real-time factor 0.006850. This measurement validates the evaluation path and supplies the reference denominator for the project-trained result; it is not a project-trained execution and is excluded from both the three-seed mean and `experiments.csv`.

The comparison anchor uses the checkpoint released through the `Bakerbunker/FreeV_Model_Logs` repository. It is reference context only and is not a project-trained result row.

The three-seed result satisfies the registered gate-one criterion: mean PESQ is 94.0% of the current-harness released-checkpoint anchor and exceeds the locked threshold of 3.30. No second training gate was required.

## Contents

| Path | Evidence |
|---|---|
| `experiments.csv` | Index of the accepted project-trained execution and its three project-trained adaptive-benchmark executions. |
| `train/` | Training configuration, complete execution log, train/validation history, and selected-state description. |
| `test/` | Seed-specific B200 test configurations, complete execution logs, and final Study 1 measurements. |
| `figures/` | Training-convergence figure in PNG and vector PDF formats. |
