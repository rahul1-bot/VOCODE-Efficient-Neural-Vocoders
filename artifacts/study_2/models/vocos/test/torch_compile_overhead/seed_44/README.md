# Vocos — torch_compile_overhead — Seed 44

This execution evaluates Vocos with the `torch_compile_overhead` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 44. It measured PESQ 3.609623 and real-time factor 0.000796 on the registered B200 lane. The 520 batch-one timing observations had p50 4.044 ms and p95 17.039 ms. Peak host-process resident memory was 7724.7 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
