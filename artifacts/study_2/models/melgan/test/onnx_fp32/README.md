# MelGAN — onnx_fp32

## Experimental Design

This result evaluates `onnx_fp32` against `baseline_cpu` on the CPU lane. Seeds 42, 43, and 44 repeat inference from the same model state on the fixed 525-utterance LJSpeech adaptive evaluation partition; they are not separate training replicas.

## Results

| PESQ | Δ PESQ | STOI | RTF | p50 ms | p95 ms | Host RSS MiB | Speedup | Size ratio | Classification |
|---|---|---|---|---|---|---|---|---|---|
| 1.043041 | -0.000006 | 0.418146 | 0.045805 | 312.318 | 437.004 | 13395.0 | 0.555× | 1.000 | inconclusive |

MelGAN PESQ is floor-censored near the instrument lower bound; objective quality preservation cannot be inferred safely.

Latency percentiles summarize the 520 post-warm-up batch-one synthesis times, where each observation is the mean of three synchronized calls. Host RSS is the process-level resident-memory high-water mark across the complete execution, not accelerator allocation or model-only memory. ONNX Runtime session construction took 0.0336 s on average; first-inference cold-start time was not retained.

## Contents

`config.yaml` defines the intervention and same-lane denominator. `samples.csv` contains the baseline and intervention means for each adaptive-benchmark utterance and the paired differences used for the reported confidence intervals. Each seed directory contains its executed configuration, execution log, and complete measurement row.
