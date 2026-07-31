# Vocos — torch_compile_overhead — Seed 43

This execution evaluates Vocos with the `torch_compile_overhead` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 43. It measured PESQ 3.609623 and real-time factor 0.000670 on the registered B200 lane. The 520 batch-one timing observations had p50 2.758 ms and p95 16.037 ms. Peak host-process resident memory was 7750.5 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
