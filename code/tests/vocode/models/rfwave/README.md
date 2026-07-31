# tests.vocode.models.rfwave: The RFWave Tests

This directory holds 80 tests mirroring `vocode/models/rfwave/`.

The module `rfwave.py` tests the vocoder module: configuration validation, backbone and flow construction, the flow-matching training recipe, and the sampled-synthesis shape contracts. The module `flow.py` tests the rectified-flow formulation: training-tuple construction pairing noised states with target velocities, and the conditional velocity-field interface. The module `network.py` tests the backbone: the time-embedding and Fourier-feature behavior, the adaptive layer normalization, the ConvNeXt-V2 adaptive blocks, the spectral transform, and the pseudo-quadrature-mirror-filter equalization, including preservation of the time-embedding dtype under reduced-precision execution. The module `sampling.py` tests the ODE sampler: deterministic Euler integration, step-count handling against the ten-step baseline, and the output contracts under reduced step counts.
