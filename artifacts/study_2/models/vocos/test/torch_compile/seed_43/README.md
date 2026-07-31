# Vocos — torch_compile — Seed 43

This execution evaluates Vocos with the `torch_compile` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 43. It measured PESQ 3.609623 and real-time factor 0.000316 on the registered B200 lane. The 520 batch-one timing observations had p50 1.753 ms and p95 2.605 ms. Peak host-process resident memory was 7497.4 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
