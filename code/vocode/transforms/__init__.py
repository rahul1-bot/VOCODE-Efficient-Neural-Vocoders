# Package initializer for the architecture-specific audio and
# mel-spectrogram transforms; modules are imported by full path.
#
# This package holds the signal analyses the architectures are defined
# against:
# 1. mel: MelConfig, the frozen mel-extraction protocol with one factory
#    per reference recipe, and MelSpectrogram, which executes a protocol
#    either through torchaudio or through the manual reflect-padded STFT
#    the published HiFi-GAN-family extraction requires
# 2. lpc: the LPCNet frame analysis (Bark-frequency band energies, DCT
#    cepstra, autocorrelation pitch search, Levinson-Durbin coefficients)
#    and the reference 8-bit mu-law companding protocol
#
# The transforms are per-architecture on purpose: reference recipes
# disagree on sample rate, band count, frequency ceiling, mel scale, STFT
# centering, and basis normalization, so no shared representation is
# imposed across families.
#
# The export list is deliberately empty, so importing the package pulls in
# no module and no transform is reachable without naming its full path.
#
# Author: Rahul Sawhney

__all__: list[str] = []
