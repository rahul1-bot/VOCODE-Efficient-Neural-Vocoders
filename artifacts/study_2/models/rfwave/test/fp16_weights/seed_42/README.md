# RFWave — fp16_weights — Seed 42

This execution evaluates RFWave with the `fp16_weights` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 42. It measured PESQ 3.782009 and real-time factor 0.004392 on the registered B200 lane. The 520 batch-one timing observations had p50 28.504 ms and p95 35.758 ms. Peak host-process resident memory was 11618.2 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
