# FreeV — torch_compile_overhead — Seed 44

This execution evaluates FreeV with the `torch_compile_overhead` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 44. It measured PESQ 3.379004 and real-time factor 0.000734 on the registered B200 lane. The 520 batch-one timing observations had p50 3.548 ms and p95 11.595 ms. Peak host-process resident memory was 8354.2 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
