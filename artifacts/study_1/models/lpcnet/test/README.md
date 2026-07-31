# LPCNet Test

This directory contains the accepted Study 1 adaptive-benchmark executions for evaluation seeds 42, 43, and 44. Every execution evaluates the same epoch-306, step-30,000 LPCNet state on all 525 adaptive-evaluation LJSpeech utterances.

## Protocol

| Field | Value |
|---|---|
| Model sample rate | 16 kHz |
| Hardware | NVIDIA B200 |
| Precision | FP32 |
| Quality batch size | 16 for seed 42; 64 for seeds 43 and 44 |
| Test shuffle | Disabled |
| Evaluation seeds | 42, 43, 44 |
| RTF evidence | Seed 42 only; 5 warm-up and 27 timed utterances |
| Selected state | The state identified in the corresponding training record. |

The evaluation seed changes the execution-level random state only; it does not alter dataset membership. Quality metrics cover all 525 utterances in every seed. Prediction timing was deliberately disabled for seeds 43 and 44 after the complete seed-42 timing lane established that the framework implementation is slower than real time.

## Results

| Seed | Run identifier | PESQ | STOI | Mel error | RTF |
|---:|---|---:|---:|---:|---:|
| 42 | `lpcnet_gate1_test_20260707_0905` | 1.075959 | 0.526932 | 2.179116 | 11.363275 |
| 43 | `lpcnet_final_seed43_test_20260707_1215` | 1.078827 | 0.528553 | 2.196004 | Not measured |
| 44 | `lpcnet_final_seed44_test_20260707_1215` | 1.076264 | 0.527143 | 2.192043 | Not measured |

Each seed directory contains its resolved experiment configuration, complete execution log, and one-row scientific metric table. An absent real-time factor is represented as absent evidence, not zero throughput.
