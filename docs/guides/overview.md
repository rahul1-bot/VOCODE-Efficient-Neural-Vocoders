# Overview

VOCODE benchmarks twelve neural vocoders under one data, evaluation, and measurement protocol, then measures how executed deployment transformations change each trained model. Every configuration is trained from scratch on LJSpeech with its architecture-native objective and identical data splits; every transformed artifact is judged against a baseline on identical hardware under an explicit decision rule. The full study is presented in the fifteen-page report at `../report/report.pdf`.

## Research Questions

| ID | Question |
|---|---|
| RQ1 | Under identical data splits and one evaluation implementation, but architecture-native objectives and budgets, how do the twelve configurations trade objective quality, inference latency, parameter count, and deployable storage? |
| RQ2 | Relative to same-hardware baselines, how do executed deployment transformations shift objective quality, warm real-time factor, and serialized artifact size, and how strongly do those effects depend on architecture and execution backend? |
| RQ3 | Which measured deployment artifact is selected under explicit hardware-specific constraints on quality, parameter count, and deployable storage? |

## Principal Findings

1. No configuration was Pareto-optimal across quality, latency, and footprint. True-length PESQ ranked RNDVoC (3.895), RFWave (3.785), and Vocos (3.614) strongest; UTMOS disagreed and ranked HiFi-GAN V1 highest (4.203). The B200 speed-quality frontier comprises Vocos, RFWave, and RNDVoC.
2. The decision rule labeled the 69 transformed groups as 19 beneficial, 18 inconclusive, and 32 harmful over 273 executions.
3. FP16 weight casting was the most consistently favorable intervention: eight of ten groups beneficial at half serialized size.
4. The largest quality-preserving speedups were 2.071 times (ONNX FP32 export of HiFi-GAN V2) and 1.403 times (default-mode compilation of HiFi-GAN V3); default-mode compilation preserved PESQ in all eleven groups.
5. Dense magnitude pruning was harmful in 18 of 19 groups, and static ONNX INT8 compressed models to a median size ratio of 0.265 at a median PESQ loss of 0.770.
6. RFWave solver-step reduction was the only smooth, monotone speed-quality control: speedups of 1.201, 2.200, and 4.509 at PESQ losses of 0.046, 0.369, and 1.636 for 8, 4, and 2 Euler steps.
7. Constraint-aware selection chose FP16-cast Vocos on the B200 lane and dynamic-INT8 Vocos on the CPU lane at the moderate constraint vector; RNDVoC remained the highest-PESQ feasible state at every vector yet was never the latency-minimizing choice.

## Scope

The study is deliberately conditional: one speaker (LJSpeech), one training trajectory per configuration under architecture-native budgets, an adaptive evaluation suffix, objective proxies without a listening test, and warm timing without host pairing. The report states these limits precisely; conclusions are auditable comparisons, not causal architecture rankings.

## Reading Order

`architecture.md` explains how the software produces evidence; `data.md`, `configurations.md`, and `training.md` specify what was trained; `metrics.md`, `deployment.md`, and `experiments.md` specify what was measured; `evidence.md` traces the results to files in `../../artifacts/`; `cloud.md` and `reproduction.md` cover execution.
