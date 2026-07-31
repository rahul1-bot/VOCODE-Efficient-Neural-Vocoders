# FreeV — torch_compile_overhead — Seed 42

This execution evaluates FreeV with the `torch_compile_overhead` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 42. It measured PESQ 3.379004 and real-time factor 0.001281 on the registered B200 lane. The 520 batch-one timing observations had p50 5.408 ms and p95 26.985 ms. Peak host-process resident memory was 8262.0 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
