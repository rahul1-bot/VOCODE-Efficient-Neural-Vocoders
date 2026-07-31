# HiFi-GAN V3 — onnx_fp32 — Seed 42

This execution evaluates HiFi-GAN V3 with the `onnx_fp32` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 42. It measured PESQ 2.577028 and real-time factor 0.021355 on the registered CPU lane. The 520 batch-one timing observations had p50 145.997 ms and p95 203.367 ms. Peak host-process resident memory was 16364.3 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
