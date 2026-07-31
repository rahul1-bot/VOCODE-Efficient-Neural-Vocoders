# MelGAN

This directory contains the Study 2 intervention results for MelGAN under the common LJSpeech evaluation design.

## Results

| Variant | Lane | PESQ | Δ PESQ | Speedup | Size ratio | State |
|---|---|---|---|---|---|---|
| baseline_b200 | b200 | 1.043223 | +0.000000 | 1.000× | 1.000 | denominator |
| baseline_cpu | cpu | 1.043047 | +0.000000 | 1.000× | 1.000 | denominator |
| fp16_weights | b200 | 1.043830 | +0.000607 | 0.910× | 0.502 | inconclusive |
| onnx_fp32 | cpu | 1.043041 | -0.000006 | 0.555× | 1.000 | inconclusive |
| onnx_int8_static | cpu | 1.038671 | -0.004377 | 0.679× | 0.265 | inconclusive |
| pruned_50 | b200 | 1.043532 | +0.000309 | 1.159× | 1.000 | inconclusive |
| pruned_50_recovered | b200 | 1.036193 | -0.007030 | 1.106× | 1.000 | inconclusive |
| torch_compile | b200 | 1.043391 | +0.000168 | 1.304× | 1.000 | inconclusive |

## Excluded Executions

| Variant | Category | Attempts | Reason |
|---|---|---|---|
| torch_compile | superseded_compilation_attempt | 3 | Preliminary compilation attempts used an incompatible runtime and were superseded by completed aligned-runtime evaluations. |

## Interpretation

All quality statements refer only to the named objective and learned-proxy metrics on single-speaker LJSpeech. Evaluation seeds repeat the same retained model state. Dense masked pruning is not interpreted as physical size or sparse-runtime acceleration.

## Contents

`experiments.csv` contains the completed seed-level measurements, and `statistics.csv` contains paired utterance-level differences and confidence intervals. `test/` contains variant and seed records, `train/` is present only when recovery training contributed to a reported result, and `figures/` contains model-specific publication figures.
