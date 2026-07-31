# RFWave — int8_weight_only — Seed 42

This execution evaluates RFWave with the `int8_weight_only` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 42. It measured PESQ 3.780232 and real-time factor 0.006717 on the registered B200 lane. The 520 batch-one timing observations had p50 42.965 ms and p95 51.743 ms. Peak host-process resident memory was 6995.0 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
