# MelGAN — onnx_fp32 — Seed 43

This execution evaluates MelGAN with the `onnx_fp32` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 43. It measured PESQ 1.043041 and real-time factor 0.043470 on the registered CPU lane. The 520 batch-one timing observations had p50 296.947 ms and p95 414.899 ms. Peak host-process resident memory was 13526.9 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
