# RFWave — torch_compile_overhead

## Experimental Design

This result evaluates `torch_compile_overhead` against `baseline_b200` on the B200 lane. Seeds 42, 43, and 44 repeat inference from the same model state on the fixed 525-utterance LJSpeech adaptive evaluation partition; they are not separate training replicas.

## Results

| PESQ | Δ PESQ | STOI | RTF | p50 ms | p95 ms | Host RSS MiB | Speedup | Size ratio | Classification |
|---|---|---|---|---|---|---|---|---|---|
| 3.780225 | -0.000094 | 0.980793 | 0.006682 | 46.372 | 59.006 | 7399.4 | 1.361× | 1.000 | beneficial |

Objective PESQ stayed within 0.05 while same-lane speed improved by at least 10% or deployable size fell by at least 45%.

Latency percentiles summarize the 520 post-warm-up batch-one synthesis times, where each observation is the mean of three synchronized calls. Host RSS is the process-level resident-memory high-water mark across the complete execution, not accelerator allocation or model-only memory.

## Contents

`config.yaml` defines the intervention and same-lane denominator. `samples.csv` contains the baseline and intervention means for each adaptive-benchmark utterance and the paired differences used for the reported confidence intervals. Each seed directory contains its executed configuration, execution log, and complete measurement row.
