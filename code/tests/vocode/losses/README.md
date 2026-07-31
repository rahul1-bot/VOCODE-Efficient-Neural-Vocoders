# tests.vocode.losses: The Objective Tests

This directory holds 234 tests mirroring `vocode/losses/`, with one test module per objective module.

The shared components are tested for term arithmetic on constructed discriminator outputs and spectrograms: `adversarial.py` covers the least-squares generator and discriminator targets, `feature_matching.py` covers the feature-space L1 aggregation, and `mel_reconstruction.py` covers the mel-reconstruction term.

The architecture objectives (`apnet2.py`, `bigvgan.py`, `freev.py`, `hifigan.py`, `hiftnet.py`, `lpcnet.py`, `melgan.py`, `rfwave.py`, `rndvoc.py`, and `vocos.py`) are tested for composition and numerics: configuration weighting, the hinge and least-squares adversarial forms, the spectral terms (log-amplitude, anti-wrapping phase, STFT consistency, and the omni-directional phase term), the flow-velocity regression with its auxiliary terms, and the teacher-forced cross-entropy over the mu-law excitation.

Assertions run on synthesized tensors with known values, so the arithmetic of each term is verified directly: results are finite, signs and targets are correct, gradients flow to the intended parameters, and configuration coefficients are applied where declared.
