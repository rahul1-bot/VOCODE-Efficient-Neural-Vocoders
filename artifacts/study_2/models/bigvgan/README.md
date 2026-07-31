# BigVGAN

This directory contains the Study 2 intervention results for BigVGAN under the common LJSpeech evaluation design.

## Results

| Variant | Lane | PESQ | Δ PESQ | Speedup | Size ratio | State |
|---|---|---|---|---|---|---|
| baseline_b200 | b200 | 3.320106 | +0.000000 | 1.000× | 1.000 | denominator |
| baseline_cpu | cpu | 3.320796 | +0.000000 | 1.000× | 1.000 | denominator |
| fp16_weights | b200 | 3.318408 | -0.001698 | 1.010× | 0.502 | beneficial |
| pruned_30 | b200 | 3.172114 | -0.147992 | 0.925× | 1.000 | harmful |
| pruned_50 | b200 | 2.171290 | -1.148816 | 1.000× | 1.000 | harmful |
| pruned_50_recovered | b200 | 3.336666 | +0.016560 | 0.990× | 1.000 | inconclusive |
| pruned_70 | b200 | 1.078243 | -2.241863 | 1.012× | 1.000 | harmful |
| torch_compile | b200 | 3.319909 | -0.000197 | 1.290× | 1.000 | beneficial |

## Excluded Executions

| Variant | Category | Attempts | Reason |
|---|---|---|---|
| torch_compile | superseded_compilation_attempt | 3 | Preliminary compilation attempts used an incompatible runtime and were superseded by completed aligned-runtime evaluations. |

## Interpretation

All quality statements refer only to the named objective and learned-proxy metrics on single-speaker LJSpeech. Evaluation seeds repeat the same retained model state. Dense masked pruning is not interpreted as physical size or sparse-runtime acceleration.

## Contents

`experiments.csv` contains the completed seed-level measurements, and `statistics.csv` contains paired utterance-level differences and confidence intervals. `test/` contains variant and seed records, `train/` is present only when recovery training contributed to a reported result, and `figures/` contains model-specific publication figures.
