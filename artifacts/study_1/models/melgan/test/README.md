# MelGAN Test

This directory contains the accepted Study 1 adaptive-benchmark executions for evaluation seeds 42, 43, and 44. Every execution evaluates the same epoch-307, step-120,000 MelGAN model on all 525 adaptive-evaluation LJSpeech utterances.

## Protocol

| Field | Value |
|---|---|
| Model sample rate | 22.05 kHz |
| Hardware | NVIDIA B200 |
| Precision | FP32 |
| Quality evaluation batch size | 16 |
| RTF batch size | 1 |
| Test shuffle | Disabled |
| Evaluation seeds | 42, 43, 44 |
| RTF warm-up utterances | 5 |
| RTF timed utterances | 520 |
| Selected state | The state identified in the corresponding training record. |

The evaluation seed changes the execution-level random state only; it does not alter dataset membership. Study 1 quality values are effectively invariant across the executions, while real-time factor records ordinary host timing variation.

## Results

| Seed | Run identifier | PESQ | STOI | Mel error | RTF |
|---:|---|---:|---:|---:|---:|
| 42 | `melgan_gate1_test_fix_20260704_1225` | 1.059746 | 0.418275 | 1.791590 | 0.095189 |
| 43 | `melgan_final_seed43_test_20260706_2207` | 1.059759 | 0.418274 | 1.791471 | 0.066263 |
| 44 | `melgan_final_seed44_test_20260706_2207` | 1.059746 | 0.418275 | 1.791590 | 0.061193 |

Each seed directory contains its resolved experiment configuration, complete execution log, and one-row scientific metric table. Runtime telemetry remains outside the compact metric table.
