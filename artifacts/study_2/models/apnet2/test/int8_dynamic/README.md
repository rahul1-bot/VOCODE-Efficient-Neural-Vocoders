# APNet2 — int8_dynamic

## Experimental Design

This result evaluates `int8_dynamic` against `baseline_cpu` on the CPU lane. Seeds 42, 43, and 44 repeat inference from the same model state on the fixed 525-utterance LJSpeech adaptive evaluation partition; they are not separate training replicas.

## Results

| PESQ | Δ PESQ | STOI | RTF | p50 ms | p95 ms | Host RSS MiB | Speedup | Size ratio | Classification |
|---|---|---|---|---|---|---|---|---|---|
| 2.584164 | +0.004003 | 0.922269 | 0.011291 | 76.398 | 95.055 | 9791.9 | 1.416× | 0.400 | beneficial |

Objective PESQ stayed within 0.05 while same-lane speed improved by at least 10% or deployable size fell by at least 45%.

Latency percentiles summarize the 520 post-warm-up batch-one synthesis times, where each observation is the mean of three synchronized calls. Host RSS is the process-level resident-memory high-water mark across the complete execution, not accelerator allocation or model-only memory.

## Contents

`config.yaml` defines the intervention and same-lane denominator. `samples.csv` contains the baseline and intervention means for each adaptive-benchmark utterance and the paired differences used for the reported confidence intervals. Each seed directory contains its executed configuration, execution log, and complete measurement row.
