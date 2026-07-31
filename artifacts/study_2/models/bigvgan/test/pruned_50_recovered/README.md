# BigVGAN — pruned_50_recovered

## Experimental Design

This result evaluates `pruned_50_recovered` against `baseline_b200` on the B200 lane. Seeds 42, 43, and 44 repeat inference from the same model state on the fixed 525-utterance LJSpeech adaptive evaluation partition; they are not separate training replicas.

## Results

| PESQ | Δ PESQ | STOI | RTF | p50 ms | p95 ms | Host RSS MiB | Speedup | Size ratio | Classification |
|---|---|---|---|---|---|---|---|---|---|
| 3.336666 | +0.016560 | 0.974793 | 0.005472 | 36.195 | 50.076 | 8703.7 | 0.990× | 1.000 | inconclusive |

Dense masked tensors do not establish deployable acceleration or storage reduction even when objective quality is retained.

Latency percentiles summarize the 520 post-warm-up batch-one synthesis times, where each observation is the mean of three synchronized calls. Host RSS is the process-level resident-memory high-water mark across the complete execution, not accelerator allocation or model-only memory.

## Contents

`config.yaml` defines the intervention and same-lane denominator. `samples.csv` contains the baseline and intervention means for each adaptive-benchmark utterance and the paired differences used for the reported confidence intervals. Each seed directory contains its executed configuration, execution log, and complete measurement row.
