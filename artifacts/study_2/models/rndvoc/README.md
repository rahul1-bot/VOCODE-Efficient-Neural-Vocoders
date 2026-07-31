# RNDVoC

This directory contains the Study 2 intervention results for RNDVoC under the common LJSpeech evaluation design.

## Results

| Variant | Lane | PESQ | Δ PESQ | Speedup | Size ratio | State |
|---|---|---|---|---|---|---|
| baseline_b200 | b200 | 3.893826 | +0.000000 | 1.000× | 1.000 | denominator |
| baseline_cpu | cpu | 3.892666 | +0.000000 | 1.000× | 1.000 | denominator |
| fp16_weights | b200 | 3.894432 | +0.000606 | 1.272× | 0.504 | beneficial |
| pruned_50 | b200 | 3.687412 | -0.206414 | 1.093× | 1.000 | harmful |
| pruned_50_recovered | b200 | 3.872111 | -0.021715 | 1.114× | 1.000 | inconclusive |
| torch_compile | b200 | 3.893425 | -0.000401 | 1.056× | 1.000 | inconclusive |

## Excluded Executions

| Variant | Category | Attempts | Reason |
|---|---|---|---|
| torch_compile | superseded_compilation_attempt | 3 | Preliminary compilation attempts used an incompatible runtime and were superseded by completed aligned-runtime evaluations. |

## Interpretation

All quality statements refer only to the named objective and learned-proxy metrics on single-speaker LJSpeech. Evaluation seeds repeat the same retained model state. Dense masked pruning is not interpreted as physical size or sparse-runtime acceleration.

## Contents

`experiments.csv` contains the completed seed-level measurements, and `statistics.csv` contains paired utterance-level differences and confidence intervals. `test/` contains variant and seed records, `train/` is present only when recovery training contributed to a reported result, and `figures/` contains model-specific publication figures.
