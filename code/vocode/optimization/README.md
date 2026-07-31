# vocode.optimization: Deployment Transformations

`optimization/` implements the deployment interventions of the study and the registry that decides which interventions apply to which architecture. Every transformation operates on a fixed Retained Project Checkpoint. Nothing in this package retrains a model, with the single exception of the explicitly named recovery arm.

| Module | Contents |
|---|---|
| `compilation.py` | `TorchCompileAcceleration` applies TorchInductor compilation to the synthesis path in default and reduce-overhead modes. |
| `quantization.py` | `DynamicInt8Quantization` (PyTorch dynamic INT8), `WeightOnlyIntQuantization` (torchao weight-only integer quantization), and `WeightOnlyFp16Storage` (FP16 weight casting), together with the numerical boundary adapters `HalfPrecisionNetworkAdapter`, `Bfloat16InputBoundary`, and the RFWave-specific `HalfPrecisionRfwaveBackboneAdapter` and `Bfloat16RfwaveBackboneBoundary`. |
| `pruning.py` | `StructuredMagnitudePruning` applies L2-ranked structured output-channel pruning over backbone Linear projections with spectral heads excluded, used by Vocos, FreeV, VocosFormer, and RFWave; `UnstructuredMagnitudePruning` applies global L1 unstructured pruning over convolutional, Linear, and recurrent weights for the remaining architectures; `MaskedMagnitudePruning` and `DenseContinuedIdentity` name the mask-holding and dense-control semantics; and `PrunedRecoveredVerification` validates recovered states. |
| `export.py` | `OnnxExporter` produces the ONNX FP32 export as an `OnnxArtifactRecord`, and `OnnxStaticQuantizer` performs static QDQ INT8 quantization with per-channel QInt8 weights, QUInt8 activations, and calibration on mel batches drawn from the execution-seeded shuffled training loader. |
| `deployment.py` | `OnnxRuntimeDeployment` executes exported artifacts through ONNX Runtime on its CPU provider and wraps the session in `OnnxNetworkModule`, so the measurement stack sees one synthesis interface. |
| `sampling.py` | `OdeStepReduction` reduces the RFWave ten-step Euler baseline to eight, four, or two solver steps. |
| `recovery.py` | `OptimizationRecoveryRunner` executes the recovery arm: 13 fine-tuning epochs at one tenth of the training learning rate for selected 50-percent-pruned states, with masks held during fine-tuning and zeros baked into the evaluated checkpoint. |
| `registry.py` | `OptimizationVariantRegistry` defines the closed vocabulary of variant names, resolves each variant to a technique per architecture through `TechniqueResolutionTable` and `ArchitectureTechniqueResolution`, defines the baseline identity variant, and records the support decisions that separate admissible executions from documented exclusions. |

Unsupported combinations fail explicitly before execution expansion and are recorded as exclusions; they never degrade into silently skipped work. This failure discipline is the code path behind the 29-row exclusion register in `../../../artifacts/study_2/exclusions.csv`.

## Related Components

The transformations are applied to checkpoints by `../trainers/optimized.py`, and the transformed artifacts are measured by `../metrics/`. The mirrored tests live in `../../tests/vocode/optimization/`.
