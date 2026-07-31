# MelGAN — torch_compile — Seed 43

This execution evaluates MelGAN with the `torch_compile` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 43. It measured PESQ 1.043391 and real-time factor 0.000514 on the registered B200 lane. The 520 batch-one timing observations had p50 3.038 ms and p95 5.051 ms. Peak host-process resident memory was 7338.2 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
