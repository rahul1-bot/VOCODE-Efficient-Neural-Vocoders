# HiFi-GAN V3 — fp16_weights — Seed 42

This execution evaluates HiFi-GAN V3 with the `fp16_weights` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 42. It measured PESQ 2.576375 and real-time factor 0.000328 on the registered B200 lane. The 520 batch-one timing observations had p50 2.012 ms and p95 2.754 ms. Peak host-process resident memory was 8776.8 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
