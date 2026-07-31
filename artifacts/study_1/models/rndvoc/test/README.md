# RNDVoC Test

This directory contains the accepted Study 1 adaptive-benchmark executions for evaluation seeds 42, 43, and 44. Every execution evaluates the same epoch-307, step-120,000 RNDVoC model on all 525 adaptive-evaluation LJSpeech utterances.

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

The evaluation seed changes the execution-level random state only; it does not alter dataset membership. Study 1 quality measurements are stable across the three executions, while real-time factor records ordinary host timing variation.

## Results

| Seed | Run identifier | PESQ | STOI | Mel error | RTF |
|---:|---|---:|---:|---:|---:|
| 42 | `rndvoc_gate1_test_20260707_0805` | 3.856987 | 0.987090 | 0.138808 | 0.053526 |
| 43 | `rndvoc_final_seed43_test_20260707_0845` | 3.856412 | 0.987090 | 0.138808 | 0.059722 |
| 44 | `rndvoc_final_seed44_test_20260707_0845` | 3.856987 | 0.987090 | 0.138808 | 0.054452 |

Each seed directory contains its resolved experiment configuration, complete execution log, and one-row scientific metric table. Runtime telemetry remains outside the compact metric table.
