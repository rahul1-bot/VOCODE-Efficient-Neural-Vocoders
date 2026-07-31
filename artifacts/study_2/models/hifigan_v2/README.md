# HiFi-GAN V2

This directory contains the Study 2 intervention results for HiFi-GAN V2 under the common LJSpeech evaluation design.

## Results

| Variant | Lane | PESQ | Δ PESQ | Speedup | Size ratio | State |
|---|---|---|---|---|---|---|
| baseline_b200 | b200 | 2.455618 | +0.000000 | 1.000× | 1.000 | denominator |
| baseline_cpu | cpu | 2.455088 | +0.000000 | 1.000× | 1.000 | denominator |
| fp16_weights | b200 | 2.456110 | +0.000493 | 1.014× | 0.516 | beneficial |
| onnx_fp32 | cpu | 2.454959 | -0.000129 | 2.071× | 0.982 | beneficial |
| onnx_int8_static | cpu | 1.668232 | -0.786855 | 1.161× | 0.318 | harmful |
| pruned_50 | b200 | 1.881731 | -0.573887 | 0.712× | 1.000 | harmful |
| torch_compile | b200 | 2.455473 | -0.000145 | 0.448× | 1.002 | harmful |

## Excluded Executions

| Variant | Category | Attempts | Reason |
|---|---|---|---|
| pruned_50_recovered | invalid_complete_group | 3 | The evaluation used the zero-step unrecovered control rather than the completed 13-epoch recovery state. |
| torch_compile | superseded_compilation_attempt | 3 | Preliminary compilation attempts used an incompatible runtime and were superseded by completed aligned-runtime evaluations. |

## Interpretation

All quality statements refer only to the named objective and learned-proxy metrics on single-speaker LJSpeech. Evaluation seeds repeat the same retained model state. Dense masked pruning is not interpreted as physical size or sparse-runtime acceleration.

## Contents

`experiments.csv` contains the completed seed-level measurements, and `statistics.csv` contains paired utterance-level differences and confidence intervals. `test/` contains variant and seed records, `train/` is present only when recovery training contributed to a reported result, and `figures/` contains model-specific publication figures.
