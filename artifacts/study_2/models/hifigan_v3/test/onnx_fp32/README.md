# HiFi-GAN V3 — onnx_fp32

## Experimental Design

This result evaluates `onnx_fp32` against `baseline_cpu` on the CPU lane. Seeds 42, 43, and 44 repeat inference from the same model state on the fixed 525-utterance LJSpeech adaptive evaluation partition; they are not separate training replicas.

## Results

| PESQ | Δ PESQ | STOI | RTF | p50 ms | p95 ms | Host RSS MiB | Speedup | Size ratio | Classification |
|---|---|---|---|---|---|---|---|---|---|
| 2.577028 | -0.000007 | 0.951975 | 0.018941 | 129.373 | 180.350 | 16698.9 | 0.912× | 0.996 | inconclusive |

The measured quality–speed–size trade-off does not cross the descriptive benefit or harm thresholds.

Latency percentiles summarize the 520 post-warm-up batch-one synthesis times, where each observation is the mean of three synchronized calls. Host RSS is the process-level resident-memory high-water mark across the complete execution, not accelerator allocation or model-only memory. ONNX Runtime session construction took 0.0129 s on average; first-inference cold-start time was not retained.

## Contents

`config.yaml` defines the intervention and same-lane denominator. `samples.csv` contains the baseline and intervention means for each adaptive-benchmark utterance and the paired differences used for the reported confidence intervals. Each seed directory contains its executed configuration, execution log, and complete measurement row.
