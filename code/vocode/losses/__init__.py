# Package initializer for the vocoder training objectives; re-exports the
# loss classes and their configuration records.
#
# The package holds two kinds of module. The shared primitives
# (adversarial, feature_matching, mel_reconstruction) implement one
# reduction each and are composed by the architecture objectives. The
# architecture modules (apnet2, bigvgan, freev, hifigan, hiftnet, lpcnet,
# melgan, rfwave, rndvoc, vocos) each publish one composite objective and
# its frozen weight record, and each module header states that family's
# published composition and update semantics.
#
# Every consumer in this repository imports from the concrete module rather
# than through this initializer, so the re-export list below is a
# convenience surface and is deliberately not exhaustive: the HiFTNet,
# LPCNet, RFWave, RNDVoC, and Vocos objectives are reached only through
# their own modules.
#
# Report alignment:
# - Objectives here are architecture-native and were deliberately not
#   standardized across families. Raw training and validation losses are
#   therefore not comparable between these modules and never rank the
#   Project-Trained Configurations; the report ranks configurations only
#   through the shared evaluation proxies, never through these values. A
#   validation scalar produced by one family is comparable against its own
#   history and against nothing else.
#
# Author: Rahul Sawhney

from vocode.losses.adversarial import LeastSquaresGanLoss
from vocode.losses.apnet2 import Apnet2Loss, Apnet2LossConfig, Apnet2Spectrum, Apnet2SpectrumAnalyzer
from vocode.losses.bigvgan import BigvganLoss, BigvganLossConfig
from vocode.losses.feature_matching import FeatureMatchingLoss
from vocode.losses.freev import FreevLoss, FreevLossConfig, FreevSpectrum
from vocode.losses.hifigan import HifiganLoss, HifiganLossConfig
from vocode.losses.mel_reconstruction import MelReconstructionLoss
from vocode.losses.melgan import MelganLoss, MelganLossConfig

__all__: list[str] = [
    "Apnet2Loss",
    "Apnet2LossConfig",
    "Apnet2Spectrum",
    "Apnet2SpectrumAnalyzer",
    "BigvganLoss",
    "BigvganLossConfig",
    "FeatureMatchingLoss",
    "FreevLoss",
    "FreevLossConfig",
    "FreevSpectrum",
    "HifiganLoss",
    "HifiganLossConfig",
    "LeastSquaresGanLoss",
    "MelganLoss",
    "MelganLossConfig",
    "MelReconstructionLoss"
]
