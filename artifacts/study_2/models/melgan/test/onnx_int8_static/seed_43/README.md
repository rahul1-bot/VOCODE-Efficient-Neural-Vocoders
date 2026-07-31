# MelGAN — onnx_int8_static — Seed 43

This execution evaluates MelGAN with the `onnx_int8_static` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 43. It measured PESQ 1.039164 and real-time factor 0.039320 on the registered CPU lane. The 520 batch-one timing observations had p50 268.644 ms and p95 373.124 ms. Peak host-process resident memory was 13626.6 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
