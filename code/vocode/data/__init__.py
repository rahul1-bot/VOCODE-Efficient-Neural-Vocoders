# Package initializer for the LJSpeech dataset and datamodule; modules
# are imported by full path.
#
# This package holds the corpus access path every runner shares:
# 1. ljspeech_dataset: metadata resolution into frozen identifier-sorted
#    records, lazy waveform loading with the optional preprocessing chain,
#    and the collator producing the padded batch mapping
# 2. ljspeech_datamodule: LJSpeechDataConfig and the harness datamodule
#    that owns the ordered-identifier holdout partition, the stage-scoped
#    dataset lifecycle, and the dataloader policy
#
# The partition contract lives in the datamodule alone, so no runner
# derives split membership for itself and every machine evaluates the same
# held-out utterances.
#
# The export list is deliberately empty, so importing the package pulls in
# no module and both components stay addressable only by full path.
#
# Author: Rahul Sawhney

__all__: list[str] = []
