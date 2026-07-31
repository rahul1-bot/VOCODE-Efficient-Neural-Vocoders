# RFWave — torch_compile_overhead — Seed 42

This execution evaluates RFWave with the `torch_compile_overhead` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 42. It measured PESQ 3.781549 and real-time factor 0.006678 on the registered B200 lane. The 520 batch-one timing observations had p50 46.275 ms and p95 59.355 ms. Peak host-process resident memory was 7404.5 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
