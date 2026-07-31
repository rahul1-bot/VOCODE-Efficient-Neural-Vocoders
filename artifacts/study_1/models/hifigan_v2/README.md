# HiFi-GAN V2

This directory contains the Study 1 evidence package for HiFi-GAN V2. One model was trained from random initialization to the registered first compute gate, and its selected durable state was evaluated in three complete adaptive-benchmark executions.

## Experimental Design

| Component | Contract |
|---|---|
| Dataset | LJSpeech 1.1 at its native 22.05 kHz sample rate |
| Partition | Identifier-ordered holdout: 12,475 train, 100 validation, 525 test utterances |
| Training seed | 1234 |
| Evaluation seeds | 42, 43, and 44 |
| Training hardware | NVIDIA L40S, BF16 mixed precision |
| Test hardware | NVIDIA B200, FP32 |
| Training plan | Gate 1; final global step 124,800 |
| Evaluated state | Epoch 307, step 120,000 |
| Generator parameters | 928,514 |

The partition is deterministic and does not use a split seed. Training seed 1234 controls initialization and optimization, whereas seeds 42, 43, and 44 identify separate executions of the same selected model on the same test utterances.

## Results

| Seed | PESQ | STOI | Mel error | Real-time factor |
|---:|---:|---:|---:|---:|
| 42 | 2.456189 | 0.945346 | 0.393531 | 0.130293 |
| 43 | 2.455956 | 0.945346 | 0.393528 | 0.099264 |
| 44 | 2.456042 | 0.945346 | 0.393529 | 0.112527 |
| Mean | 2.456062 | 0.945346 | 0.393529 | 0.114028 |

The gate-one mean reaches 82.3% of the released-checkpoint anchor but remains below the registered PESQ target of 2.78. The validation proxy reached its global minimum at step 121,000 and closed within the same flat tail, so the close-and-climbing criterion did not authorize a second gate. The accepted classification is a partial reproduction stopped by the registered gate-one rule.

## Released-Checkpoint Reference

The legacy author-checkpoint summary reports three-seed PESQ 2.9860 on the same 525-utterance B200 FP32 evaluation surface. Only PESQ is used for this cross-harness comparison; the legacy STOI and mel-error values were produced by an earlier metric protocol. The author checkpoint is not a project-trained execution and is excluded from experiments.csv.

## Contents

| Path | Evidence |
|---|---|
| experiments.csv | Index of the accepted training execution and three project-trained adaptive-benchmark executions. |
| train/ | Gate-one training configuration, complete execution log, scientific metric history, and selected-state description. |
| test/ | Seed-specific B200 test configurations, complete execution logs, and final Study 1 measurements. |
| `figures/` | Training-convergence figure in PNG and vector PDF formats. |
