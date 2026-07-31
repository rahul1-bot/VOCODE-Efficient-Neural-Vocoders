# VocosFormer — torch_compile — Seed 44

This execution evaluates VocosFormer with the `torch_compile` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 44. It measured PESQ 2.476713 and real-time factor 0.000327 on the registered B200 lane. The 520 batch-one timing observations had p50 1.948 ms and p95 2.928 ms. Peak host-process resident memory was 7678.2 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
