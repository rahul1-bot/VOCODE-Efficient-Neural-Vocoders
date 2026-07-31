# VocosFormer — torch_compile — Seed 43

This execution evaluates VocosFormer with the `torch_compile` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 43. It measured PESQ 2.476714 and real-time factor 0.000337 on the registered B200 lane. The 520 batch-one timing observations had p50 2.004 ms and p95 2.846 ms. Peak host-process resident memory was 7695.5 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
