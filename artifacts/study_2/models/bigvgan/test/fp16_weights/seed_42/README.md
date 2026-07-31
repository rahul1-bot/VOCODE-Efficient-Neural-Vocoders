# BigVGAN — fp16_weights — Seed 42

This execution evaluates BigVGAN with the `fp16_weights` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 42. It measured PESQ 3.318409 and real-time factor 0.005055 on the registered B200 lane. The 520 batch-one timing observations had p50 33.928 ms and p95 45.021 ms. Peak host-process resident memory was 8764.8 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
