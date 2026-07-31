# VocosFormer — torch_compile_overhead — Seed 44

This execution evaluates VocosFormer with the `torch_compile_overhead` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 44. It measured PESQ 2.476637 and real-time factor 0.001191 on the registered B200 lane. The 520 batch-one timing observations had p50 3.961 ms and p95 16.243 ms. Peak host-process resident memory was 8024.0 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
