# HiFi-GAN V1 — pruned_30

## Experimental Design

This result evaluates `pruned_30` against `baseline_b200` on the B200 lane. Seeds 42, 43, and 44 repeat inference from the same model state on the fixed 525-utterance LJSpeech adaptive evaluation partition; they are not separate training replicas.

## Results

| PESQ | Δ PESQ | STOI | RTF | p50 ms | p95 ms | Host RSS MiB | Speedup | Size ratio | Classification |
|---|---|---|---|---|---|---|---|---|---|
| 3.127395 | -0.190860 | 0.965021 | 0.001311 | 7.569 | 14.258 | 8812.9 | 0.971× | 1.000 | harmful |

Dense masking caused more than 0.10 PESQ loss and provides no physical storage or sparse-kernel deployment gain.

Latency percentiles summarize the 520 post-warm-up batch-one synthesis times, where each observation is the mean of three synchronized calls. Host RSS is the process-level resident-memory high-water mark across the complete execution, not accelerator allocation or model-only memory.

## Contents

`config.yaml` defines the intervention and same-lane denominator. `samples.csv` contains the baseline and intervention means for each adaptive-benchmark utterance and the paired differences used for the reported confidence intervals. Each seed directory contains its executed configuration, execution log, and complete measurement row.
