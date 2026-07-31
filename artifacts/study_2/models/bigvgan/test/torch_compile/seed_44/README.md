# BigVGAN — torch_compile — Seed 44

This execution evaluates BigVGAN with the `torch_compile` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 44. It measured PESQ 3.319942 and real-time factor 0.004387 on the registered B200 lane. The 520 batch-one timing observations had p50 29.342 ms and p95 40.515 ms. Peak host-process resident memory was 7661.1 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
