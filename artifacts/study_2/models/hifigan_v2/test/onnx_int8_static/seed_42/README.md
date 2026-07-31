# HiFi-GAN V2 — onnx_int8_static — Seed 42

This execution evaluates HiFi-GAN V2 with the `onnx_int8_static` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 42. It measured PESQ 1.709979 and real-time factor 0.025184 on the registered CPU lane. The 520 batch-one timing observations had p50 168.329 ms and p95 240.591 ms. Peak host-process resident memory was 11295.5 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
