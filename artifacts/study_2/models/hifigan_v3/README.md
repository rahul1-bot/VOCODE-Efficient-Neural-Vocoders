# HiFi-GAN V3

This directory contains the Study 2 intervention results for HiFi-GAN V3 under the common LJSpeech evaluation design.

## Results

| Variant | Lane | PESQ | Δ PESQ | Speedup | Size ratio | State |
|---|---|---|---|---|---|---|
| baseline_b200 | b200 | 2.576807 | +0.000000 | 1.000× | 1.000 | denominator |
| baseline_cpu | cpu | 2.577035 | +0.000000 | 1.000× | 1.000 | denominator |
| fp16_weights | b200 | 2.576375 | -0.000432 | 1.874× | 0.503 | beneficial |
| onnx_fp32 | cpu | 2.577028 | -0.000007 | 0.912× | 0.996 | inconclusive |
| onnx_int8_static | cpu | 1.823604 | -0.753431 | 0.581× | 0.264 | harmful |
| pruned_50 | b200 | 1.988729 | -0.588077 | 1.481× | 1.000 | harmful |
| pruned_50_recovered | b200 | 2.591055 | +0.014248 | 1.614× | 1.000 | inconclusive |
| torch_compile | b200 | 2.576888 | +0.000081 | 1.403× | 1.000 | beneficial |

## Excluded Executions

| Variant | Category | Attempts | Reason |
|---|---|---|---|
| torch_compile | superseded_compilation_attempt | 3 | Preliminary compilation attempts used an incompatible runtime and were superseded by completed aligned-runtime evaluations. |

## Interpretation

All quality statements refer only to the named objective and learned-proxy metrics on single-speaker LJSpeech. Evaluation seeds repeat the same retained model state. Dense masked pruning is not interpreted as physical size or sparse-runtime acceleration.

## Contents

`experiments.csv` contains the completed seed-level measurements, and `statistics.csv` contains paired utterance-level differences and confidence intervals. `test/` contains variant and seed records, `train/` is present only when recovery training contributed to a reported result, and `figures/` contains model-specific publication figures.
