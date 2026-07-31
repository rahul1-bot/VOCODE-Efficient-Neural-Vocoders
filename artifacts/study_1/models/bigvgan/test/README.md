# BigVGAN Test

This directory contains the accepted Study 1 adaptive-benchmark executions for evaluation seeds 42, 43, and 44. Every execution evaluates the same epoch-307, step-120,000 BigVGAN model state on the same 525 identifier-ordered LJSpeech test utterances.

## Protocol

| Field | Value |
|---|---|
| Model sample rate | 24 kHz |
| Hardware | NVIDIA B200 |
| Precision | FP32 |
| Test batch size | 16 |
| Test shuffle | Disabled |
| Evaluation seeds | 42, 43, 44 |
| RTF warm-up utterances | 5 |
| RTF timed utterances | 520 |
| Selected state | The state identified in the corresponding training record. |

The evaluation seed changes the execution-level random state only; it does not alter dataset membership. The same 14,025,154-parameter project-trained checkpoint is used throughout. Study 1 quality measurements are stable across the three executions, while real-time factor captures ordinary host timing variation.

## Results

| Seed | Run identifier | PESQ | STOI | Mel error | RTF |
|---:|---|---:|---:|---:|---:|
| 42 | `bigvgan_gate1_test_20260705_0450` | 3.256716 | 0.973688 | 0.638447 | 0.154058 |
| 43 | `bigvgan_final_seed43_test_20260706_2207` | 3.257049 | 0.973688 | 0.638400 | 0.185560 |
| 44 | `bigvgan_final_seed44_test_20260706_2207` | 3.257181 | 0.973688 | 0.638468 | 0.132183 |

All three rows use the project-trained checkpoint. The separately measured author-released checkpoint is a model-level reference anchor and is not part of this three-seed result.

Each seed directory contains its resolved experiment configuration, complete execution log, and one-row scientific metric table. Runtime telemetry remains outside the compact metric table.
