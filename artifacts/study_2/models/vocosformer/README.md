# VocosFormer

This directory contains the Study 2 intervention results for VocosFormer under the common LJSpeech evaluation design.

## Results

| Variant | Lane | PESQ | Δ PESQ | Speedup | Size ratio | State |
|---|---|---|---|---|---|---|
| baseline_b200 | b200 | 2.476623 | +0.000000 | 1.000× | 1.000 | denominator |
| baseline_cpu | cpu | 2.477956 | +0.000000 | 1.000× | 1.000 | denominator |
| fp16_weights | b200 | 2.476222 | -0.000400 | 0.554× | 0.500 | inconclusive |
| int8_dynamic | cpu | 2.475423 | -0.002533 | 1.097× | 0.672 | inconclusive |
| int8_weight_only | b200 | 2.476176 | -0.000446 | 0.617× | 0.674 | inconclusive |
| pruned_50 | b200 | 1.591730 | -0.884892 | 1.126× | 1.000 | harmful |
| torch_compile | b200 | 2.476714 | +0.000091 | 0.921× | 1.000 | inconclusive |
| torch_compile_overhead | b200 | 2.476634 | +0.000011 | 0.345× | 1.000 | harmful |

## Excluded Executions

| Variant | Category | Attempts | Reason |
|---|---|---|---|
| pruned_50_recovered | invalid_complete_group | 3 | The evaluation used the zero-step unrecovered control rather than the completed 13-epoch recovery state. |
| torch_compile | superseded_compilation_attempt | 3 | Preliminary compilation attempts used an incompatible runtime and were superseded by completed aligned-runtime evaluations. |
| torch_compile_overhead | superseded_compilation_attempt | 3 | Preliminary compilation attempts used an incompatible runtime and were superseded by completed aligned-runtime evaluations. |

## Interpretation

All quality statements refer only to the named objective and learned-proxy metrics on single-speaker LJSpeech. Evaluation seeds repeat the same retained model state. Dense masked pruning is not interpreted as physical size or sparse-runtime acceleration.

## Contents

`experiments.csv` contains the completed seed-level measurements, and `statistics.csv` contains paired utterance-level differences and confidence intervals. `test/` contains variant and seed records, `train/` is present only when recovery training contributed to a reported result, and `figures/` contains model-specific publication figures.
