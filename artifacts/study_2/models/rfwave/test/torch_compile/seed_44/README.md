# RFWave — torch_compile — Seed 44

This execution evaluates RFWave with the `torch_compile` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 44. It measured PESQ 3.776916 and real-time factor 0.006942 on the registered B200 lane. The 520 batch-one timing observations had p50 48.142 ms and p95 60.109 ms. Peak host-process resident memory was 7177.2 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
