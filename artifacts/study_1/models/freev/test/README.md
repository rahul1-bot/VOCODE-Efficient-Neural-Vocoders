# FreeV Test

This directory contains the accepted Study 1 adaptive-benchmark executions for evaluation seeds 42, 43, and 44. Every execution evaluates the same epoch-307, step-120,000 FreeV model state on the same 525 identifier-ordered LJSpeech test utterances.

## Protocol

| Field | Value |
|---|---|
| Model sample rate | 22.05 kHz |
| Hardware | NVIDIA B200 |
| Precision | FP32 |
| Test batch size | 16 |
| Test shuffle | Disabled |
| Evaluation seeds | 42, 43, 44 |
| RTF warm-up utterances | 5 |
| RTF timed utterances | 520 |
| Selected state | The state identified in the corresponding training record. |

The evaluation seed changes the execution-level random state only; it does not alter dataset membership. The same 18,220,557-parameter project-trained checkpoint is used throughout. Study 1 quality measurements are stable across the three executions, while real-time factor captures ordinary host timing variation.

## Results

| Seed | Run identifier | PESQ | STOI | Mel error | RTF |
|---:|---|---:|---:|---:|---:|
| 42 | `freev_gate1_test_20260704_1445` | 3.339525 | 0.969317 | 0.174622 | 0.006353 |
| 43 | `freev_final_seed43_test_20260706_2207` | 3.339515 | 0.969317 | 0.174621 | 0.009221 |
| 44 | `freev_final_seed44_test_20260706_2207` | 3.339253 | 0.969317 | 0.174622 | 0.007458 |

All three rows use the project-trained checkpoint. The separately measured author-released checkpoint is a model-level reference anchor and is not part of this three-seed result.

Each seed directory contains its resolved experiment configuration, complete execution log, and one-row scientific metric table. Runtime telemetry remains outside the compact metric table.
