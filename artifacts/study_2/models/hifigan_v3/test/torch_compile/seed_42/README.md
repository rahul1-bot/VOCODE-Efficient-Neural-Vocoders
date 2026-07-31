# HiFi-GAN V3 — torch_compile — Seed 42

This execution evaluates HiFi-GAN V3 with the `torch_compile` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 42. It measured PESQ 2.576884 and real-time factor 0.000397 on the registered B200 lane. The 520 batch-one timing observations had p50 2.429 ms and p95 3.591 ms. Peak host-process resident memory was 7467.3 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
