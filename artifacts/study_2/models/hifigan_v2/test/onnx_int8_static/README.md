# HiFi-GAN V2 — onnx_int8_static

## Experimental Design

This result evaluates `onnx_int8_static` against `baseline_cpu` on the CPU lane. Seeds 42, 43, and 44 repeat inference from the same model state on the fixed 525-utterance LJSpeech adaptive evaluation partition; they are not separate training replicas.

## Results

| PESQ | Δ PESQ | STOI | RTF | p50 ms | p95 ms | Host RSS MiB | Speedup | Size ratio | Classification |
|---|---|---|---|---|---|---|---|---|---|
| 1.668232 | -0.786855 | 0.929172 | 0.025417 | 170.870 | 247.683 | 11529.2 | 1.161× | 0.318 | harmful |

Mean PESQ decreased by more than 0.10 against the same-lane denominator.

Latency percentiles summarize the 520 post-warm-up batch-one synthesis times, where each observation is the mean of three synchronized calls. Host RSS is the process-level resident-memory high-water mark across the complete execution, not accelerator allocation or model-only memory. ONNX Runtime session construction took 0.0768 s on average; first-inference cold-start time was not retained.

## Contents

`config.yaml` defines the intervention and same-lane denominator. `samples.csv` contains the baseline and intervention means for each adaptive-benchmark utterance and the paired differences used for the reported confidence intervals. Each seed directory contains its executed configuration, execution log, and complete measurement row.
