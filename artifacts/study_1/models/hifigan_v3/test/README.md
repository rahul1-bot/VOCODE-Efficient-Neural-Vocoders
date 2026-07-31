# HiFi-GAN V3 Test

This directory contains the accepted Study 1 adaptive-benchmark executions for evaluation seeds 42, 43, and 44. Every execution evaluates the same epoch-628, step-245,000 HiFi-GAN V3 model state on the same 525 identifier-ordered LJSpeech test utterances.

## Protocol

| Field | Value |
|---|---|
| Model sample rate | 22.05 kHz |
| Hardware | NVIDIA B200 |
| Precision | FP32 |
| Quality-evaluation batch size | 16 |
| RTF batch size | 1 |
| Test shuffle | Disabled |
| Evaluation seeds | 42, 43, 44 |
| RTF warm-up utterances | 5 |
| RTF timed utterances | 520 |
| Selected state | The state identified in the corresponding training record. |

The evaluation seed changes the execution-level random state only; it does not alter dataset membership. The same 1,464,322-parameter project-trained checkpoint is used throughout. Quality measurements are effectively deterministic across the three executions, while real-time factor captures host-level timing variation.

## Results

| Seed | Run identifier | PESQ | STOI | Mel error | RTF |
|---:|---|---:|---:|---:|---:|
| 42 | hifigan_v3_gate2_test_20260704_1550 | 2.576494 | 0.951972 | 0.360738 | 0.051098 |
| 43 | hifigan_v3_final_seed43_test_20260706_2207 | 2.576486 | 0.951972 | 0.360738 | 0.049458 |
| 44 | hifigan_v3_final_seed44_test_20260706_2207 | 2.576482 | 0.951972 | 0.360738 | 0.073402 |

All three rows use the project-trained selected checkpoint. The legacy author-checkpoint PESQ anchor is model-level context and is not part of this three-seed result.

Each seed directory contains its resolved experiment configuration, complete execution log, and one-row Study 1 measurement table. Runtime telemetry remains outside the compact metric table.
