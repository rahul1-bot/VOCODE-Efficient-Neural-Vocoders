# MelGAN — torch_compile — Seed 42

This execution evaluates MelGAN with the `torch_compile` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 42. It measured PESQ 1.043383 and real-time factor 0.000540 on the registered B200 lane. The 520 batch-one timing observations had p50 3.116 ms and p95 6.028 ms. Peak host-process resident memory was 7429.8 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
