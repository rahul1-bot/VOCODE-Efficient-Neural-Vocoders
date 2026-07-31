# HiFi-GAN V1

This directory contains the Study 2 intervention results for HiFi-GAN V1 under the common LJSpeech evaluation design.

## Results

| Variant | Lane | PESQ | Δ PESQ | Speedup | Size ratio | State |
|---|---|---|---|---|---|---|
| baseline_b200 | b200 | 3.318255 | +0.000000 | 1.000× | 1.000 | denominator |
| baseline_cpu | cpu | 3.317273 | +0.000000 | 1.000× | 1.000 | denominator |
| fp16_weights | b200 | 3.316679 | -0.001576 | 1.081× | 0.501 | beneficial |
| onnx_fp32 | cpu | 3.317273 | +0.000000 | 0.893× | 0.998 | harmful |
| onnx_int8_static | cpu | 1.988645 | -1.328627 | 0.959× | 0.257 | harmful |
| pruned_30 | b200 | 3.127395 | -0.190860 | 0.971× | 1.000 | harmful |
| pruned_50 | b200 | 2.255041 | -1.063214 | 1.363× | 1.000 | harmful |
| pruned_70 | b200 | 1.304445 | -2.013810 | 1.272× | 1.000 | harmful |
| torch_compile | b200 | 3.318022 | -0.000233 | 0.136× | 1.000 | harmful |

## Excluded Executions

| Variant | Category | Attempts | Reason |
|---|---|---|---|
| pruned_50_recovered | invalid_complete_group | 3 | The evaluation used a ten-batch stress state rather than the completed 13-epoch recovery state. |
| torch_compile | superseded_compilation_attempt | 3 | Preliminary compilation attempts used an incompatible runtime and were superseded by completed aligned-runtime evaluations. |

## Interpretation

All quality statements refer only to the named objective and learned-proxy metrics on single-speaker LJSpeech. Evaluation seeds repeat the same retained model state. Dense masked pruning is not interpreted as physical size or sparse-runtime acceleration.

## Contents

`experiments.csv` contains the completed seed-level measurements, and `statistics.csv` contains paired utterance-level differences and confidence intervals. `test/` contains variant and seed records, `train/` is present only when recovery training contributed to a reported result, and `figures/` contains model-specific publication figures.
