# RNDVoC — torch_compile — Seed 43

This execution evaluates RNDVoC with the `torch_compile` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 43. It measured PESQ 3.893425 and real-time factor 0.001377 on the registered B200 lane. The 520 batch-one timing observations had p50 6.162 ms and p95 24.432 ms. Peak host-process resident memory was 8560.3 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
