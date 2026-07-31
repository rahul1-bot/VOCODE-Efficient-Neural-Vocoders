# HiFi-GAN V2 — fp16_weights — Seed 42

This execution evaluates HiFi-GAN V2 with the `fp16_weights` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 42. It measured PESQ 2.456073 and real-time factor 0.001098 on the registered B200 lane. The 520 batch-one timing observations had p50 6.736 ms and p95 8.696 ms. Peak host-process resident memory was 8919.2 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
