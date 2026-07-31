# This module:
# 1. Implements VocosFormer for the Study 1 reproduction cohort: the
#    literature-derived, parameter-matched Vocos adaptation whose ConvNeXt
#    backbone is augmented with the WavTokenizer residual-convolution and
#    full-self-attention position network
#
# Harness contract (syntheticmind):
# - Subclasses the reproduced Vocos module, so discriminators, losses,
#   optimizers, schedules, metrics, and test instrumentation are inherited
#   byte-identical; only the generator network is replaced, which is the
#   controlled-variable design
#
# Design decisions:
# - The trimmed defaults hold the parameter count at the Vocos baseline
#   within two percent, so backbone capacity cannot masquerade as the
#   attention contribution
# - The row carries no novelty claim: Vocos supplies the chassis and
#   training objective, WavTokenizer supplies the published position-network
#   concept
#
# Author: Rahul Sawhney

from pydantic import PositiveInt

from vocode.models.vocos.vocos import Vocos, VocosConfig
from vocode.models.vocosformer.network import VocosformerNetwork, VocosformerNetworkConfig

__all__: list[str] = ["Vocosformer", "VocosformerConfig"]


class VocosformerConfig(VocosConfig):
    # Frozen VocosFormer configuration. Inherits every Vocos training field
    # unchanged (losses, discriminators, optimizer, schedule, mel protocol)
    # so the backbone remains the single experimental variable, and adds the
    # position-network fields with parameter-matched trimmed defaults.
    #
    # Fields:
    #     intermediate_dimension: Expanded width inside each ConvNeXt
    #         block, trimmed below the Vocos baseline to absorb the
    #         position network's parameters. Default: ``1344``.
    #     layer_count: Number of ConvNeXt blocks, halved against the
    #         baseline for the same reason. Default: ``4``.
    #     position_group_count: Group count of every normalization inside
    #         the position network. Default: ``32``.
    #     position_dropout: Dropout rate inside the position network's
    #         residual convolution blocks. Default: ``0.1``.
    #
    # Every other field is inherited from the Vocos record at its
    # published value and must stay there: the learning rate, both Adam
    # moments, the gradient-clipping norm, the cosine horizon, the loss
    # weights, and the mel protocol are what the two rows hold in common,
    # and a difference in any of them would make the comparison
    # uncontrolled. The record stays frozen and extra-forbidding by
    # inheritance, so a misspelled position-network field is rejected
    # rather than silently ignored.
    intermediate_dimension: PositiveInt = 1344
    layer_count: PositiveInt = 4
    position_group_count: PositiveInt = 32
    position_dropout: float = 0.1

    @classmethod
    def matched_vocos_24khz(cls) -> VocosformerConfig:
        # Builds the parameter-matched VocosFormer configuration against the Vocos baseline.
        # Every field already defaults to its matched value, so the
        # factory constructs the record unmodified; it exists as the named
        # entry point the registry routes through, so the matched recipe
        # is reached by name rather than by relying on defaults at the
        # call site.
        #
        # Returns:
        #     The frozen configuration whose network holds its parameter
        #     count within two percent of the reproduced Vocos baseline.
        return cls()


class Vocosformer(Vocos):
    # VocosFormer module: the reproduced Vocos training and evaluation
    # machinery with only the generator network replaced by the
    # attention-augmented variant. The row is a literature-derived,
    # parameter-matched adaptation of Vocos and carries no novelty claim:
    # Vocos supplies the chassis and the training objective, and the
    # published WavTokenizer position-network concept supplies the
    # attention insertion.
    #
    # Integration: subclassing rather than duplicating is what makes the
    # comparison controlled. Every training step, evaluation step,
    # optimizer and schedule declaration, discriminator ensemble, loss
    # module, mel protocol, and metric surface is inherited from the
    # reproduced Vocos module unchanged, so the only difference between
    # the two rows is the generator network. Anything this class adds
    # beyond the network replacement would weaken that claim.
    #
    # The registry treats the row as training-level ready but carries no
    # author-weight adapter for it, since no published release exists to
    # anchor. It is therefore evaluated exclusively through
    # project-trained checkpoints.
    def __init__(self, configuration: VocosformerConfig) -> None:
        # Runs the full Vocos construction, then replaces the generator
        # network with the attention-augmented VocosFormer backbone.
        # The baseline network is genuinely built first and then
        # discarded, which costs one construction but guarantees that
        # every other member the parent creates is wired exactly as it is
        # in the baseline row.
        #
        # Args:
        #     configuration: The frozen matched recipe. Its topology
        #         fields are forwarded into a network configuration
        #         record, and the inverse-STFT padding mode is fixed to
        #         the centered grid here, matching the parent.
        super().__init__(configuration)
        self._configuration: VocosformerConfig = configuration
        self.network: VocosformerNetwork = VocosformerNetwork(
            VocosformerNetworkConfig(
                input_channels=configuration.input_channels,
                hidden_dimension=configuration.hidden_dimension,
                intermediate_dimension=configuration.intermediate_dimension,
                layer_count=configuration.layer_count,
                n_fft=configuration.n_fft,
                hop_length=configuration.hop_length,
                padding="center",
                position_group_count=configuration.position_group_count,
                position_dropout=configuration.position_dropout
            )
        )

    @property
    def configuration(self) -> VocosformerConfig:
        # Returns the immutable configuration attached to this component.
        # The property is narrowed to the adaptation's record type, which
        # is safe because the record extends the baseline's, so callers
        # typed against the parent continue to work while callers needing
        # the position-network fields reach them without a cast.
        return self._configuration
