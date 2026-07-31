# HiFi-GAN V1 — torch_compile — Seed 42

This execution evaluates HiFi-GAN V1 with the `torch_compile` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 42. It measured PESQ 3.318010 and real-time factor 0.013777 on the registered B200 lane. The 520 batch-one timing observations had p50 92.013 ms and p95 129.465 ms. Peak host-process resident memory was 7578.0 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
