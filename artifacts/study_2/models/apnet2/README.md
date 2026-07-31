# APNet2

This directory contains the Study 2 intervention results for APNet2 under the common LJSpeech evaluation design.

## Results

| Variant | Lane | PESQ | Δ PESQ | Speedup | Size ratio | State |
|---|---|---|---|---|---|---|
| baseline_b200 | b200 | 2.581322 | +0.000000 | 1.000× | 1.000 | denominator |
| baseline_cpu | cpu | 2.580160 | +0.000000 | 1.000× | 1.000 | denominator |
| fp16_weights | b200 | 2.579939 | -0.001383 | 1.085× | 0.500 | beneficial |
| int8_dynamic | cpu | 2.584164 | +0.004003 | 1.416× | 0.400 | beneficial |
| pruned_50 | b200 | 1.196365 | -1.384957 | 0.835× | 1.000 | harmful |
| torch_compile | b200 | 2.581322 | +0.000000 | 1.245× | 1.000 | beneficial |

## Excluded Executions

| Variant | Category | Attempts | Reason |
|---|---|---|---|
| pruned_50_recovered | invalid_complete_group | 3 | The evaluation used the zero-step unrecovered control rather than the completed 13-epoch recovery state. |
| torch_compile | superseded_compilation_attempt | 3 | Preliminary compilation attempts used an incompatible runtime and were superseded by completed aligned-runtime evaluations. |

## Interpretation

All quality statements refer only to the named objective and learned-proxy metrics on single-speaker LJSpeech. Evaluation seeds repeat the same retained model state. Dense masked pruning is not interpreted as physical size or sparse-runtime acceleration.

## Contents

`experiments.csv` contains the completed seed-level measurements, and `statistics.csv` contains paired utterance-level differences and confidence intervals. `test/` contains variant and seed records, `train/` is present only when recovery training contributed to a reported result, and `figures/` contains model-specific publication figures.
