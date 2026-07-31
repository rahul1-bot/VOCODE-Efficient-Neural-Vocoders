# LPCNet

LPCNet is the sample-level autoregressive configuration of the cohort. Linear prediction carries the signal envelope, while a recurrent network models the quantized mu-law excitation. It is the 1.23-million-parameter configuration conditioned on 20 features at 16 kHz (18 Bark-frequency cepstra with pitch period and correlation), trained in FP32 for 31.3 thousand updates with a retained state at 30 thousand.

| Module | Contents |
|---|---|
| `lpcnet.py` | `LpcnetConfig` and `Lpcnet`, the vocoder module. |
| `network.py` | `LpcnetNetwork`, the excitation model. |
| `sampling.py` | `LpcnetSampler`, the free-running synthesis path. |
| `sparsification.py` | `LpcnetSparsifier` and `LpcnetWeightClipper`, the training-time sparsification of the published recipe. |

`Lpcnet` validates its configuration, constructs the network, executes the teacher-forced training recipe, and exposes free-running synthesis through the sampler. `LpcnetNetwork` combines a fractional input embedding, a recurrent core with a dual fully connected output, hierarchical output probabilities through a tree-probability module, and the linear-prediction path, returning an `LpcnetNetworkOutput`. `LpcnetSampler` performs sample-by-sample synthesis, drawing each excitation value and combining it with the linear prediction to reconstruct the waveform. `LpcnetSparsifier` applies the scheduled recurrent-weight sparsification of the published recipe between 2 and 20 thousand updates, and `LpcnetWeightClipper` bounds the recurrent weights during training.

There is no discriminator and no author-weight adapter: the objective is teacher-forced cross-entropy, and no commensurate author release is loaded. The study reports the PyTorch-path timing of this implementation, which omits the authors' optimized sparse C implementation, and its baseline evaluation documents free-running instability at the executed fraction of the reference training recipe; both caveats are stated in the report.

## Related Components

The training objective lives in `../../losses/lpcnet.py`, and the feature front end lives in `../../transforms/lpc.py`. The mirrored tests live in `../../../tests/vocode/models/lpcnet/`.
