# RFWave — torch_compile_overhead — Seed 44

This execution evaluates RFWave with the `torch_compile_overhead` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 44. It measured PESQ 3.776916 and real-time factor 0.006707 on the registered B200 lane. The 520 batch-one timing observations had p50 46.748 ms and p95 58.999 ms. Peak host-process resident memory was 7397.1 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
