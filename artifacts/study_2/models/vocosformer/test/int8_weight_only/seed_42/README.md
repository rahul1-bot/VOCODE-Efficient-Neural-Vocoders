# VocosFormer — int8_weight_only — Seed 42

This execution evaluates VocosFormer with the `int8_weight_only` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 42. It measured PESQ 2.476358 and real-time factor 0.000580 on the registered B200 lane. The 520 batch-one timing observations had p50 3.668 ms and p95 4.508 ms. Peak host-process resident memory was 6813.2 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
