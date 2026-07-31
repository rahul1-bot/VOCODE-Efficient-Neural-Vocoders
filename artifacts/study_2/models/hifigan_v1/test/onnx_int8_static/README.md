# HiFi-GAN V1 — onnx_int8_static

## Experimental Design

This result evaluates `onnx_int8_static` against `baseline_cpu` on the CPU lane. Seeds 42, 43, and 44 repeat inference from the same model state on the fixed 525-utterance LJSpeech adaptive evaluation partition; they are not separate training replicas.

## Results

| PESQ | Δ PESQ | STOI | RTF | p50 ms | p95 ms | Host RSS MiB | Speedup | Size ratio | Classification |
|---|---|---|---|---|---|---|---|---|---|
| 1.988645 | -1.328627 | 0.961685 | 0.090568 | 616.182 | 872.311 | 16704.1 | 0.959× | 0.257 | harmful |

Mean PESQ decreased by more than 0.10 against the same-lane denominator.

Latency percentiles summarize the 520 post-warm-up batch-one synthesis times, where each observation is the mean of three synchronized calls. Host RSS is the process-level resident-memory high-water mark across the complete execution, not accelerator allocation or model-only memory. ONNX Runtime session construction took 0.1115 s on average; first-inference cold-start time was not retained.

## Contents

`config.yaml` defines the intervention and same-lane denominator. `samples.csv` contains the baseline and intervention means for each adaptive-benchmark utterance and the paired differences used for the reported confidence intervals. Each seed directory contains its executed configuration, execution log, and complete measurement row.
