# VocosFormer — Test

This directory contains the complete Study 2 evaluation groups for VocosFormer. Each group uses seeds 42, 43, and 44 on the same retained model state and the same 525 adaptive-benchmark LJSpeech utterances. The directory name is a stable package-schema label, not a claim of untouched final-test independence.

## Groups

| Variant | Lane | Denominator | Classification |
|---|---|---|---|
| baseline_b200 | b200 | baseline_b200 | denominator |
| baseline_cpu | cpu | baseline_cpu | denominator |
| fp16_weights | b200 | baseline_b200 | inconclusive |
| int8_dynamic | cpu | baseline_cpu | inconclusive |
| int8_weight_only | b200 | baseline_b200 | inconclusive |
| pruned_50 | b200 | baseline_b200 | harmful |
| torch_compile | b200 | baseline_b200 | inconclusive |
| torch_compile_overhead | b200 | baseline_b200 | harmful |
