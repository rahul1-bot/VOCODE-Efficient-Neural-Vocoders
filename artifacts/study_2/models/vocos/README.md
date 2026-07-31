# Vocos

This directory contains the Study 2 intervention results for Vocos under the common LJSpeech evaluation design.

## Results

| Variant | Lane | PESQ | Δ PESQ | Speedup | Size ratio | State |
|---|---|---|---|---|---|---|
| baseline_b200 | b200 | 3.609623 | +0.000000 | 1.000× | 1.000 | denominator |
| baseline_cpu | cpu | 3.610011 | +0.000000 | 1.000× | 1.000 | denominator |
| fp16_weights | b200 | 3.610732 | +0.001109 | 1.107× | 0.500 | beneficial |
| int8_dynamic | cpu | 3.563127 | -0.046884 | 1.206× | 0.274 | beneficial |
| int8_weight_only | b200 | 3.611810 | +0.002187 | 0.982× | 0.278 | beneficial |
| pruned_30 | b200 | 1.748683 | -1.860939 | 0.750× | 1.000 | harmful |
| pruned_50 | b200 | 1.465455 | -2.144167 | 0.960× | 1.000 | harmful |
| pruned_50_recovered | b200 | 3.134346 | -0.475277 | 0.760× | 1.000 | harmful |
| pruned_70 | b200 | 1.307128 | -2.302495 | 0.758× | 1.000 | harmful |
| torch_compile | b200 | 3.609623 | +0.000000 | 1.074× | 1.000 | inconclusive |
| torch_compile_overhead | b200 | 3.609623 | +0.000000 | 0.390× | 1.000 | harmful |

## Excluded Executions

| Variant | Category | Attempts | Reason |
|---|---|---|---|
| dense_continued | invalid_complete_group | 3 | The checkpoint declared by the evaluation recipe does not match the retained dense-continuation state, so the evaluated state cannot be reconstructed. |
| pruned_50_recovered_half | invalid_complete_group | 3 | The checkpoint declared by the evaluation recipe does not match the retained half-budget recovery state, so the evaluated state cannot be reconstructed. |
| torch_compile | superseded_compilation_attempt | 3 | Preliminary compilation attempts used an incompatible runtime and were superseded by completed aligned-runtime evaluations. |
| torch_compile_overhead | superseded_compilation_attempt | 3 | Preliminary compilation attempts used an incompatible runtime and were superseded by completed aligned-runtime evaluations. |
| pruned_30 | duplicate_complete_execution | 1 | An earlier completed execution already represents this model, variant, and seed; the later duplicate is not aggregated. |

## Interpretation

All quality statements refer only to the named objective and learned-proxy metrics on single-speaker LJSpeech. Evaluation seeds repeat the same retained model state. Dense masked pruning is not interpreted as physical size or sparse-runtime acceleration.

## Contents

`experiments.csv` contains the completed seed-level measurements, and `statistics.csv` contains paired utterance-level differences and confidence intervals. `test/` contains variant and seed records, `train/` is present only when recovery training contributed to a reported result, and `figures/` contains model-specific publication figures.
