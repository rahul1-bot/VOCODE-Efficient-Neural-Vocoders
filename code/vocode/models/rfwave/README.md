# RFWave

RFWave is the rectified-flow configuration of the cohort: a multi-band spectral representation is reconstructed by integrating a learned velocity field, so the solver step count is an explicit latency and quality control. It is the 18.14-million-parameter configuration conditioned on 100 unnormalized HTK-scale mel bands at 24 kHz, trained on 32,512-sample crops at batch size 64 for 125.0 thousand updates with a retained state at 120 thousand. It is the one cohort member trained on an 80 GB NVIDIA A100.

| Module | Contents |
|---|---|
| `rfwave.py` | `RfwaveConfig` and `Rfwave`, the vocoder module. |
| `flow.py` | `RfwaveRectifiedFlow` and `RfwaveTrainTuple`, the rectified-flow formulation. |
| `network.py` | `RfwaveBackbone`, `RfwaveSpectralTransform`, and `RfwavePqmfEqualizer`, the generator components. |
| `sampling.py` | `RfwaveOdeSampler`, the deterministic solver. |

`Rfwave` validates its configuration, constructs the backbone and the flow, executes the flow-matching training recipe, and synthesizes waveforms through the sampler. `RfwaveRectifiedFlow` builds training tuples that pair noised states with target velocities and defines the conditional velocity-field objective interface. `RfwaveBackbone` consists of ConvNeXt-V2 adaptive blocks modulated by adaptive layer normalization under sinusoidal time embeddings and Fourier input features; `RfwaveSpectralTransform` and `RfwavePqmfEqualizer` perform the multi-band spectral analysis, synthesis, and pseudo-quadrature-mirror-filter equalization. `RfwaveOdeSampler` integrates the learned velocity field with deterministic Euler steps; the study baseline uses ten steps, and the deployment intervention in `../../optimization/sampling.py` reduces the count to eight, four, or two.

There is no discriminator and no author-weight adapter: the configuration is a project-trained reimplementation of the published architecture, and no author release is loaded.

## Related Components

The training objective lives in `../../losses/rfwave.py` and regresses conditional flow velocities with auxiliary magnitude and band-overlap terms. The mel front end lives in `../../transforms/mel.py`. The mirrored tests live in `../../../tests/vocode/models/rfwave/`.
