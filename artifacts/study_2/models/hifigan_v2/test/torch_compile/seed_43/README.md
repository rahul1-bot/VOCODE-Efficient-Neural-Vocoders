# HiFi-GAN V2 — torch_compile — Seed 43

This execution evaluates HiFi-GAN V2 with the `torch_compile` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 43. It measured PESQ 2.455506 and real-time factor 0.002191 on the registered B200 lane. The 520 batch-one timing observations had p50 13.637 ms and p95 20.954 ms. Peak host-process resident memory was 7469.1 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
