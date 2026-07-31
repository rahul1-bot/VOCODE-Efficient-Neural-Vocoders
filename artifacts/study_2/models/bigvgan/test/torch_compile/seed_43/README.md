# BigVGAN — torch_compile — Seed 43

This execution evaluates BigVGAN with the `torch_compile` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 43. It measured PESQ 3.319802 and real-time factor 0.004169 on the registered B200 lane. The 520 batch-one timing observations had p50 27.918 ms and p95 37.771 ms. Peak host-process resident memory was 7682.1 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
