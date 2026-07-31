# MelGAN — onnx_int8_static — Seed 42

This execution evaluates MelGAN with the `onnx_int8_static` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 42. It measured PESQ 1.038501 and real-time factor 0.036915 on the registered CPU lane. The 520 batch-one timing observations had p50 252.173 ms and p95 348.649 ms. Peak host-process resident memory was 13710.5 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
