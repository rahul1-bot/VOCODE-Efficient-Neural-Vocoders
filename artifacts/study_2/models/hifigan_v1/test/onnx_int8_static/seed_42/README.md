# HiFi-GAN V1 — onnx_int8_static — Seed 42

This execution evaluates HiFi-GAN V1 with the `onnx_int8_static` configuration on all 525 adaptive-benchmark LJSpeech utterances using evaluation seed 42. It measured PESQ 2.002533 and real-time factor 0.080870 on the registered CPU lane. The 520 batch-one timing observations had p50 550.210 ms and p95 803.067 ms. Peak host-process resident memory was 16945.7 MiB.

`config.yaml` defines the executed data, intervention, and timing conditions. `execution.log` records the run events, and `metrics.csv` contains the complete Study 2 measurement row.
