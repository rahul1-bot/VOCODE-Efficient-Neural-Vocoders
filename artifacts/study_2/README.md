# Study 2 — Efficient Neural Vocoder Optimization

## Research Question

Which post-training, training-aware, structural, compilation, and exported-runtime interventions improve deployment efficiency without unacceptable objective degradation when applied to the controlled Study 1 neural-vocoder cohort?

## Experimental Design

| Component | Design |
|---|---|
| Baseline cohort | Twelve Study 1 neural-vocoder configurations |
| Dataset | LJSpeech 1.1; 12,475 train, 100 validation, 525 adaptive-evaluation utterances |
| Evaluation repetitions | Seeds 42, 43, and 44 on one retained model state |
| Execution lanes | Modal B200 (16 CPU units, 64 GiB host memory) and CPU (8 CPU units, 16 GiB host memory), compared only within lane |
| Timing | Batch one; five warm-ups; 520 timed utterances; three synchronized calls averaged per utterance |
| Software environment | Python 3.14, PyTorch 2.13.0, torchaudio 2.11.0 |
| Intervention families | Compilation, reduced precision, quantization, ONNX Runtime, masked pruning with recovery, and ODE step reduction |

The result set contains 91 complete model–variant groups and 273 seed-level executions. Every result covers all 525 adaptive-evaluation utterances, records the full measurement panel, completes the timing protocol, and has zero mandatory metric failures. Evaluation seeds repeat inference from the same model state and therefore measure execution stability rather than training variability. The retained `test/` directory name is a stable schema label. Study 1 scores on these utterances informed some training gates and an inference correction, so the partition is not an untouched final test set.

Containers were scheduled independently rather than co-located or interleaved. CPU SKU, physical host identity, region, affinity, background contention, image digest, driver, and cuDNN build were not retained. Accordingly, same-lane means the same requested resource and software profile, not the same physical host. Absolute latency and speedup estimates are conditional on those scheduled Modal lanes.

## Result Coverage

| Model | Result groups |
|---|---|
| APNet2 | 6 |
| BigVGAN | 8 |
| FreeV | 7 |
| HiFi-GAN V1 | 9 |
| HiFi-GAN V2 | 7 |
| HiFi-GAN V3 | 8 |
| LPCNet | 0 |
| MelGAN | 8 |
| RFWave | 13 |
| RNDVoC | 6 |
| Vocos | 11 |
| VocosFormer | 8 |

## Descriptive Efficiency Highlights

| Model | Variant | Δ PESQ | Speedup | Size ratio |
|---|---|---|---|---|
| HiFi-GAN V2 | onnx_fp32 | -0.0001 | 2.071× | 0.982 |
| RFWave | fp16_weights | +0.0000 | 1.955× | 0.500 |
| HiFi-GAN V3 | fp16_weights | -0.0004 | 1.874× | 0.503 |
| APNet2 | int8_dynamic | +0.0040 | 1.416× | 0.400 |
| HiFi-GAN V3 | torch_compile | +0.0001 | 1.403× | 1.000 |
| RFWave | torch_compile | -0.0001 | 1.379× | 1.000 |
| RFWave | torch_compile_overhead | -0.0001 | 1.361× | 1.000 |
| BigVGAN | torch_compile | -0.0002 | 1.290× | 1.000 |
| RFWave | int8_weight_only | -0.0025 | 1.278× | 0.393 |
| RNDVoC | fp16_weights | +0.0006 | 1.272× | 0.504 |
| APNet2 | torch_compile | +0.0000 | 1.245× | 1.000 |
| Vocos | int8_dynamic | -0.0469 | 1.206× | 0.274 |

Classification is a post hoc descriptive taxonomy selected after execution, not a registered rule or population-level significance claim. A result is labelled beneficial only when mean PESQ stays within 0.05 of the same-lane denominator and execution improves by at least 10% or deployable size falls by at least 45%, without a speed regression greater than 10%. A PESQ loss greater than 0.10 is harmful. Dense masking cannot be beneficial because it changes neither tensor shape nor runtime kernel. MelGAN interventions remain inconclusive because PESQ is floor-censored near the instrument lower bound.

## Excluded Evidence

The 29 exclusion records summarize 88 attempts: nine invalid complete groups, fifteen superseded compilation attempts, four incomplete LPCNet groups, and one duplicate execution. They are exclusion records rather than registered experimental cells and do not contribute to result means or confidence intervals.

## Statistical Analysis

For each intervention, the three repeated evaluations are averaged within utterance and paired with the same-lane baseline over the 525 adaptive benchmark utterances. Two-sided Student-t 95% confidence intervals are reported for PESQ, STOI, mel error, multi-resolution STFT error, MCD, LAS-RMSE, and UTMOS. The utterance is the statistical unit; the inference seeds are not treated as independent model replicas.

The package also reports RTF, p50 and p95 synthesis-call latency, and peak host-process RSS. Latency percentiles are computed over the 520 post-warm-up utterance-level mean call times. RSS is a process high-water mark over the full execution and must not be interpreted as model-only or accelerator memory. ONNX groups retain session-construction time; first-inference cold-start time was not retained.

## Measurement Boundary

UTMOS uses the SpeechMOS UTMOS22 strong predictor release v1.2.0 after resampling candidate audio to 16 kHz. MAC measurements distinguish partial traces from failed profiles, and failed profiles have no numeric MAC value. No blinded listening study or multi-speaker/out-of-distribution evaluation was executed. Therefore “quality” means only the named objective and learned-proxy metrics on single-speaker LJSpeech. Recovery conclusions describe one retained recovered checkpoint, not independent recovery-seed variability.

LJSpeech 1.1 is released in the public domain. The SpeechMOS evaluator implementation is MIT-licensed; its third-party source and checkpoint are not redistributed in this package. Project-trained vocoder states are original experiment outputs, and the package contains no duplicated third-party executable source.

## Contents

`results.csv` is the 91-row model–variant result table. `exclusions.csv` summarizes non-result executions and their scientific disposition. `models/` contains model summaries, seed-level experiment tables, paired statistics, intervention configurations, execution logs, measurements, and recovery-training records where applicable. `figures/` directories contain publication visualizations derived from the finalized numerical results. Raw execution capsules are retained on the project execution volume, while this package carries the complete curated measurement evidence.
