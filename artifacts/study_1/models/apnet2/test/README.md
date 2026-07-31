# APNet2 Test

This directory contains the accepted Study 1 adaptive-benchmark executions for evaluation seeds 42, 43, and 44. Every execution evaluates the same epoch-307, step-120,000 APNet2 model state on the same 525 identifier-ordered LJSpeech test utterances.

## Protocol

| Field | Value |
|---|---|
| Hardware | NVIDIA B200 |
| Precision | FP32 |
| Test batch size | 16 |
| Test shuffle | Disabled |
| Evaluation seeds | 42, 43, 44 |
| RTF warm-up utterances | 5 |
| RTF timed utterances | 520 |
| Selected state | The state identified in the corresponding training record. |

The evaluation seed changes the execution-level random state only; it does not alter dataset membership. Study 1 quality measurements are effectively invariant across the three executions, while real-time factor captures ordinary host timing variation.

## Results

| Seed | Run identifier | PESQ | STOI | Mel error | RTF |
|---:|---|---:|---:|---:|---:|
| 42 | `apnet2_gate1_test_20260704_1305` | 2.541039 | 0.930045 | 0.543382 | 0.008689 |
| 43 | `apnet2_final_seed43_test_20260706_2207` | 2.541039 | 0.930045 | 0.543382 | 0.009360 |
| 44 | `apnet2_final_seed44_test_20260706_2207` | 2.541039 | 0.930045 | 0.543382 | 0.010303 |

All three rows use the project-trained checkpoint. The separately measured author-released checkpoint is a model-level reference anchor and is not part of this three-seed result.

Each seed directory contains its resolved experiment configuration, complete execution log, and one-row scientific metric table. Runtime telemetry remains outside the compact metric table.
