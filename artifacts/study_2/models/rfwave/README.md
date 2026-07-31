# RFWave

This directory contains the Study 2 intervention results for RFWave under the common LJSpeech evaluation design.

## Results

| Variant | Lane | PESQ | Δ PESQ | Speedup | Size ratio | State |
|---|---|---|---|---|---|---|
| baseline_b200 | b200 | 3.780319 | +0.000000 | 1.000× | 1.000 | denominator |
| baseline_cpu | cpu | 3.784927 | +0.000000 | 1.000× | 1.000 | denominator |
| fp16_weights | b200 | 3.780354 | +0.000035 | 1.955× | 0.500 | beneficial |
| int8_dynamic | cpu | 3.137174 | -0.647753 | 1.407× | 0.390 | harmful |
| int8_weight_only | b200 | 3.777815 | -0.002504 | 1.278× | 0.393 | beneficial |
| ode_steps_2 | b200 | 2.144021 | -1.636298 | 4.509× | 1.000 | harmful |
| ode_steps_4 | b200 | 3.411451 | -0.368868 | 2.200× | 1.000 | harmful |
| ode_steps_8 | b200 | 3.734706 | -0.045614 | 1.201× | 1.000 | beneficial |
| pruned_30 | b200 | 1.040654 | -2.739665 | 1.033× | 1.000 | harmful |
| pruned_50 | b200 | 1.061558 | -2.718761 | 0.918× | 1.000 | harmful |
| pruned_70 | b200 | 1.067908 | -2.712411 | 1.042× | 1.000 | harmful |
| torch_compile | b200 | 3.780225 | -0.000094 | 1.379× | 1.000 | beneficial |
| torch_compile_overhead | b200 | 3.780225 | -0.000094 | 1.361× | 1.000 | beneficial |

## Excluded Executions

| Variant | Category | Attempts | Reason |
|---|---|---|---|
| dense_continued | invalid_complete_group | 3 | The dense continuation source ran one epoch rather than the registered 13-epoch recovery-matched budget. |
| pruned_50_recovered | invalid_complete_group | 3 | The evaluation used a ten-batch stress state rather than the completed 13-epoch recovery state. |
| torch_compile | superseded_compilation_attempt | 3 | Preliminary compilation attempts used an incompatible runtime and were superseded by completed aligned-runtime evaluations. |
| torch_compile_overhead | superseded_compilation_attempt | 3 | Preliminary compilation attempts used an incompatible runtime and were superseded by completed aligned-runtime evaluations. |

## Interpretation

All quality statements refer only to the named objective and learned-proxy metrics on single-speaker LJSpeech. Evaluation seeds repeat the same retained model state. Dense masked pruning is not interpreted as physical size or sparse-runtime acceleration.

## Contents

`experiments.csv` contains the completed seed-level measurements, and `statistics.csv` contains paired utterance-level differences and confidence intervals. `test/` contains variant and seed records, `train/` is present only when recovery training contributed to a reported result, and `figures/` contains model-specific publication figures.
