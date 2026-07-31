# HiFi-GAN V1 — baseline_cpu

## Experimental Design

This result evaluates `baseline_cpu` against `baseline_cpu` on the CPU lane. Seeds 42, 43, and 44 repeat inference from the same model state on the fixed 525-utterance LJSpeech adaptive evaluation partition; they are not separate training replicas.

## Results

| PESQ | Δ PESQ | STOI | RTF | p50 ms | p95 ms | Host RSS MiB | Speedup | Size ratio | Classification |
|---|---|---|---|---|---|---|---|---|---|
| 3.317273 | +0.000000 | 0.974801 | 0.086864 | 592.690 | 841.973 | 12909.5 | 1.000× | 1.000 | denominator |

Same-lane identity denominator.

Latency percentiles summarize the 520 post-warm-up batch-one synthesis times, where each observation is the mean of three synchronized calls. Host RSS is the process-level resident-memory high-water mark across the complete execution, not accelerator allocation or model-only memory.

## Contents

`config.yaml` defines the intervention and same-lane denominator. Each seed directory contains its executed configuration, execution log, and complete measurement row.
