# Deployment

Each Retained Project Checkpoint was transformed with the deployment variants below and measured against its same-lane baseline. The transformations are implemented in `code/vocode/optimization/`, and the registry in `registry.py` decides which variant applies to which architecture. Nothing retrains a model except the explicitly named recovery arm.

## Variant Vocabulary

The register names below are the exact `variant_name` values of the evidence CSVs.

| Variant | Technique |
|---|---|
| `baseline_b200`, `baseline_cpu` | The identity transformation, defining the paired same-lane control |
| `torch_compile` | TorchInductor compilation, default mode |
| `torch_compile_overhead` | TorchInductor compilation, reduce-overhead mode |
| `fp16_weights` | FP16 weight casting at half serialized size |
| `int8_dynamic` | PyTorch dynamic INT8 quantization |
| `int8_weight_only` | torchao weight-only integer quantization |
| `onnx_fp32` | ONNX FP32 export executed through ONNX Runtime |
| `onnx_int8_static` | Static QDQ INT8 quantization of the exported graph |
| `pruned_30`, `pruned_50`, `pruned_70` | Magnitude pruning at 30, 50, and 70 percent sparsity with dense masks |
| `pruned_50_recovered` | The 50 percent state after 13 recovery epochs |
| `ode_steps_8`, `ode_steps_4`, `ode_steps_2` | RFWave Euler solver reduction against the ten-step baseline |

## Technique Detail

Static ONNX INT8 uses ONNX Runtime 1.24.4 on its CPU provider with eight intra-op threads, QDQ format, per-channel QInt8 weights, QUInt8 activations, and coverage of Conv, ConvTranspose, MatMul, and Gemm. Each execution reconstructs its quantized graph, calibrating on 48 mel batches from the execution-seeded shuffled training loader, so these executions combine calibration-set and runtime variation. Dynamic and weight-only INT8 retain their distinct PyTorch and torchao semantics; an unsupported INT4 probe contributes no result.

Pruning is magnitude-based with dense masks, so it measures tolerance to weight removal, not sparse storage or sparse kernels. Vocos, FreeV, VocosFormer, and RFWave use L2-ranked structured output-channel pruning over backbone Linear projections with spectral heads excluded; the remaining architectures use global L1 unstructured pruning over convolutional, Linear, and recurrent weights. Selected 50 percent states received 13 recovery epochs at one tenth of the training learning rate, masks held during fine-tuning and zeros baked into the evaluated checkpoint; absent an admitted dense-continuation control, recovered rows support descriptive rather than causal claims.

FP16 casting halves serialized storage; architecture-specific boundary adapters keep numerics correct, including the RFWave backbone adapters that preserve the time-embedding dtype under reduced precision.

## Admission

Admission requires an executable transformed path, nonzero operator or parameter coverage, all 525 utterances under three executions, and a same-profile pre-transformation measurement; unsupported cells fail before execution expansion. The executed surface contains 22 pre-transformation controls and 69 transformed groups (91 groups, 273 executions). LPCNet completed no admissible transformed execution on either lane, leaving eleven transformed configurations, and 29 exclusion records document 88 rejected attempts in `artifacts/study_2/exclusions.csv`.

The launch vocabulary is wider than the admitted surface: `int4_weight_only` (the unsupported INT4 probe), `dense_continued` (the dense-continuation control), and `pruned_50_recovered_half` exist as launchable variant names, and none of them contributes an admitted group. The attempted `dense_continued` and `pruned_50_recovered_half` executions are recorded in the exclusion register, and the INT4 probe is unsupported and contributes no result, as the report states.
