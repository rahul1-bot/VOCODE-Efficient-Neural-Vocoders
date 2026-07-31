# FreeV — torch_compile_overhead — Seed 43

This execution evaluates FreeV with the `torch_compile_overhead` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 43. It measured PESQ 3.379005 and real-time factor 0.001031 on the registered B200 lane. The 520 batch-one timing observations had p50 5.859 ms and p95 16.940 ms. Peak host-process resident memory was 8342.9 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
