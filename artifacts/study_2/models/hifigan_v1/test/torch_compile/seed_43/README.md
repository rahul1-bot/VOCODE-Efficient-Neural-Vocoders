# HiFi-GAN V1 — torch_compile — Seed 43

This execution evaluates HiFi-GAN V1 with the `torch_compile` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 43. It measured PESQ 3.318140 and real-time factor 0.001450 on the registered B200 lane. The 520 batch-one timing observations had p50 8.890 ms and p95 13.444 ms. Peak host-process resident memory was 7566.3 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
