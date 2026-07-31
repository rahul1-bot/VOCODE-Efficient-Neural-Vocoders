# HiFi-GAN V3

This directory contains the Study 1 evidence package for HiFi-GAN V3. One model was trained from random initialization through two registered compute gates, and its selected durable state was evaluated in three complete adaptive-benchmark executions.

## Experimental Design

| Component | Contract |
|---|---|
| Dataset | LJSpeech 1.1 at its native 22.05 kHz sample rate |
| Partition | Identifier-ordered holdout: 12,475 train, 100 validation, 525 test utterances |
| Training seed | 1234 |
| Evaluation seeds | 42, 43, and 44 |
| Training hardware | NVIDIA L40S, BF16 mixed precision |
| Test hardware | NVIDIA B200, FP32 |
| Training plan | Two gates; final global step 249,990 |
| Evaluated state | Epoch 628, step 245,000 |
| Generator parameters | 1,464,322 |

The partition is deterministic and does not use a split seed. Training seed 1234 controls initialization and optimization, whereas seeds 42, 43, and 44 identify separate executions of the same selected model on the same test utterances.

## Results

| Seed | PESQ | STOI | Mel error | Real-time factor |
|---:|---:|---:|---:|---:|
| 42 | 2.576494 | 0.951972 | 0.360738 | 0.051098 |
| 43 | 2.576486 | 0.951972 | 0.360738 | 0.049458 |
| 44 | 2.576482 | 0.951972 | 0.360738 | 0.073402 |
| Mean | 2.576487 | 0.951972 | 0.360738 | 0.057986 |

Gate 1 evaluated the step-120,000 state and produced seed-42 PESQ 2.5024. Gate 2 improved seed-42 PESQ by 0.0741, but the final value remained 0.0235 below the locked 2.60 target when the registered 500,000-step batch-16-equivalent ceiling was reached. The accepted classification is therefore a partial reproduction at the registered compute ceiling.

## Released-Checkpoint Reference

The legacy author-checkpoint summary reports three-seed PESQ 2.8029 on the same 525-utterance B200 FP32 evaluation surface. The project-trained mean is 91.9% of that anchor. Only PESQ is used for this cross-harness comparison; the legacy STOI and mel-error values were produced by an earlier metric protocol. The author checkpoint is not a project-trained execution and is excluded from experiments.csv.

## Contents

| Path | Evidence |
|---|---|
| experiments.csv | Index of the accepted training execution and three project-trained adaptive-benchmark executions. |
| train/ | Two-gate training configuration, complete execution log, scientific metric history, and selected-state description. |
| test/ | Seed-specific B200 test configurations, complete execution logs, and final Study 1 measurements. |
| `figures/` | Training-convergence figure in PNG and vector PDF formats. |
