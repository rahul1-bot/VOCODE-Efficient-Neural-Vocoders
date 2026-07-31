# HiFi-GAN V3 — torch_compile — Seed 43

This execution evaluates HiFi-GAN V3 with the `torch_compile` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 43. It measured PESQ 2.576861 and real-time factor 0.000475 on the registered B200 lane. The 520 batch-one timing observations had p50 2.790 ms and p95 4.400 ms. Peak host-process resident memory was 7468.7 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
