# RFWave — torch_compile — Seed 42

This execution evaluates RFWave with the `torch_compile` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 42. It measured PESQ 3.781549 and real-time factor 0.006360 on the registered B200 lane. The 520 batch-one timing observations had p50 43.927 ms and p95 56.019 ms. Peak host-process resident memory was 7205.3 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
