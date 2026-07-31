# Vocos Test

This directory contains the accepted Study 1 adaptive-benchmark executions for evaluation seeds 42, 43, and 44. Every execution evaluates the same epoch-307, step-120,000 Vocos model on all 525 adaptive-evaluation LJSpeech utterances.

## Protocol

| Field | Value |
|---|---|
| Model sample rate | 24 kHz |
| Hardware | NVIDIA B200 |
| Precision | FP32 |
| Quality evaluation batch size | 16 |
| RTF batch size | 1 |
| Test shuffle | Disabled |
| Evaluation seeds | 42, 43, 44 |
| RTF warm-up utterances | 5 |
| RTF timed utterances | 520 |
| Selected state | The state identified in the corresponding training record. |

The evaluation seed changes the execution-level random state only; it does not alter dataset membership. Quality values are identical across the three executions, while real-time factor records ordinary host timing variation.

## Results

| Seed | Run identifier | PESQ | STOI | Mel error | RTF |
|---:|---|---:|---:|---:|---:|
| 42 | `vocos_gate1_test_20260704_1120` | 3.565251 | 0.979314 | 3.046101 | 0.005486 |
| 43 | `vocos_final_seed43_test_20260706_2207` | 3.565251 | 0.979314 | 3.046101 | 0.004344 |
| 44 | `vocos_final_seed44_test_20260706_2207` | 3.565251 | 0.979314 | 3.046101 | 0.004344 |

Each seed directory contains its resolved experiment configuration, complete execution log, and one-row scientific metric table. Runtime telemetry remains outside the compact metric table.
